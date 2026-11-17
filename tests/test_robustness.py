"""Tests for the robustness evaluator on synthetic tones, with the audio decoder faked.

The toy extractor reads band energies of a pure tone whose frequency is the
item id, so neighbours are items with close tones and white noise visibly
disturbs them. ``random`` (the real negative control) ignores the audio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import musicsim.evaluation
import musicsim.relevance  # noqa: F401  (registers "label_match")
from musicsim import audio, paths
from musicsim.audio_perturbations import PERTURBATIONS
from musicsim.config import Config, ConfigError
from musicsim.embeddings import build_embeddings, features_cache_dir
from musicsim.evaluation.base import Evaluator
from musicsim.evaluation.robustness import (
    NULL_CONDITION,
    RobustnessEvaluator,
    _embed,
    jaccard_at_k,
    original_rank,
    parse_conditions,
    rank_gallery,
)
from musicsim.experiments import extract_stage, run_experiment
from musicsim.extractors.base import Extractor
from musicsim.registry import register_extractor

N_ITEMS = 60
N_BANDS = 16
SAMPLE_RATE = 8000
N_SAMPLES = 4000


def _item_of(path: str) -> int:
    return int(path.split("/")[-1].split(".")[0])


def _tone(item_id: int) -> np.ndarray:
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * (200.0 + 55.0 * item_id) * t)).astype(np.float32)


def _band_features(y: np.ndarray) -> np.ndarray:
    power = np.abs(np.fft.rfft(y.astype(np.float64))) ** 2
    energy = np.array([band.sum() for band in np.array_split(power, N_BANDS)])
    return np.log(energy / energy.sum() + 1e-6).astype(np.float32)


@register_extractor("toy_robust")
class _ToyRobustExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        return self.extract_from_waveform(item_id, audio.load_clip(path, self.config))

    def extract_from_waveform(self, item_id: int, y: np.ndarray) -> np.ndarray:
        return _band_features(y)


@pytest.fixture(autouse=True)
def _fake_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio, "load_clip", lambda path, config: _tone(_item_of(str(path))))


def _write_index(split_of=lambda i: "test") -> pd.DataFrame:
    ids = list(range(N_ITEMS))
    paths.dataset_dir("toy_robust").mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame(
        {
            "item_id": ids,
            "path": [f"synthetic/{i}.wav" for i in ids],
            "split": [split_of(i) for i in ids],
            "artist_id": [100 + i for i in ids],
            "group_id": ids,
            "genre_top": [f"g{i // 20}" for i in ids],
            "genres_all": ["[]"] * N_ITEMS,
        }
    )
    index.to_csv(paths.dataset_index("toy_robust"), index=False)
    return index


def _config(
    conditions: dict | None = None,
    representations: list | None = None,
    robustness: bool = True,
    embedding: dict | None = None,
) -> Config:
    _write_index()
    default_conditions = {"default": [{"noise_snr_db": 10.0}, {"noise_snr_db": 30.0}]}
    data: dict = {
        "experiment": "e00_toy_robust",
        "seed": 0,
        "dataset": {"directory": "toy_robust"},
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "clip_seconds": N_SAMPLES / SAMPLE_RATE,
            "clip_samples": N_SAMPLES,
            "min_seconds": 0.1,
            "mono": True,
        },
        "extractor": {"name": "toy_robust", "dim": N_BANDS},
        "embedding": {
            "fit_on": "test",
            "standardize": True,
            "l2_normalize": True,
            "pca": {"enabled": False, "n_components": None},
            **(embedding or {}),
        },
        "graph": {"builder": "knn", "k": 10, "near_duplicate_threshold": 0.9999},
        "retrieval": {"gallery": "fma_all", "relevance": "label_match"},
        "bootstrap": {"n_resamples": 100, "confidence": 0.95},
        "evaluation": {"retrieval": {"enabled": False}},
        "runtime": {"n_jobs": 1, "progress": False, "cache": True},
        "stages": ["evaluate"],
    }
    if robustness:
        data["evaluation"]["robustness"] = {
            "enabled": True,
            "galleries": ["fma_all"],
            "conditions": conditions or default_conditions,
        }
    if representations is not None:
        data["representations"] = representations
    return Config(data=data)


def _read(run, name: str = "robustness.csv") -> pd.DataFrame:
    return pd.read_csv(run.metrics_dir / name)


# --- pure functions -----------------------------------------------------------
def test_jaccard_at_k_of_known_lists() -> None:
    a = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    b = np.array([[1, 2, 0, 3], [0, 1, 3, 2], [2, 3, 0, 1]])
    # top-2 sets: {0,1} vs {1,2} -> 1/3; {0,1} vs {0,1} -> 1; {0,1} vs {2,3} -> 0.
    np.testing.assert_array_equal(jaccard_at_k(a, b, 2), [1 / 3, 1.0, 0.0])


def test_rank_gallery_excludes_the_group_and_original_rank_keeps_the_original() -> None:
    G = np.array([[1.0, 0.0], [0.9, 0.4359], [0.0, 1.0], [-1.0, 0.0]])
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    group = np.array([0, 0, 1, 2])  # rows 0 and 1 are near-duplicates
    Q = G.copy()
    Q[0] = [0.5, 0.866]  # a perturbed version of row 0

    order = rank_gallery(Q, G, group)

    # Candidates of query 0: rows 2 (cos .87) and 3 (cos -.5); its group goes last.
    assert order[0].tolist() == [2, 3, 0, 1]
    # The original (cos .5) is beaten only by row 2; its duplicate (row 1) is removed.
    assert original_rank(Q, G, group)[0] == 2
    # Unperturbed queries find their original first.
    assert original_rank(Q, G, group)[2:].tolist() == [1, 1]


def test_parse_conditions() -> None:
    assert len(parse_conditions("all", "x")) == sum(len(v) for v in PERTURBATIONS.values()) == 13
    assert parse_conditions([{"noise_snr_db": 20}, {"tempo_ratio": 1.1}], "x") == [
        ("noise_snr_db", 20.0),
        ("tempo_ratio", 1.1),
    ]
    with pytest.raises(ConfigError, match="not in the closed list"):
        parse_conditions([{"noise_snr_db": 25}], "x")
    with pytest.raises(ConfigError, match="repeat"):
        parse_conditions([{"gain_db": 6}, {"gain_db": 6}], "x")


# --- evaluator ----------------------------------------------------------------
def test_null_condition_is_exact_and_random_ignores_the_audio(
    isolated_roots: dict[str, Path],
) -> None:
    out = _read(run_experiment(_config(representations=["toy_robust", "random"])))
    null = out[out["perturbation"] == NULL_CONDITION[0]]
    assert set(null["representation"]) == {"toy_robust", "random"}
    by_metric = null.set_index(["representation", "metric"])
    for label in ("toy_robust", "random"):
        assert by_metric.loc[(label, "jaccard"), "value"] == 1.0
        assert by_metric.loc[(label, "original_rank"), "value"] == 1.0
        assert by_metric.loc[(label, "original_top10"), "value"] == 1.0
        for metric in ("dR", "dMAP", "dnDCG"):
            assert by_metric.loc[(label, metric), "value"] == 0.0
            assert by_metric.loc[(label, metric), "ci_low"] == 0.0
            assert by_metric.loc[(label, metric), "ci_high"] == 0.0

    random = out[out["representation"] == "random"]
    assert (random[random["metric"] == "jaccard"]["value"] == 1.0).all()
    assert (random[random["metric"].isin(["dR", "dMAP", "dnDCG"])]["value"] == 0.0).all()
    assert (random[random["metric"] == "original_rank"]["value"] == 1.0).all()


def test_noise_disturbs_a_content_sensitive_representation(
    isolated_roots: dict[str, Path],
) -> None:
    out = _read(run_experiment(_config()))
    jaccard = out[out["metric"] == "jaccard"].set_index("level")["value"]
    assert jaccard.loc[0.0] == 1.0  # null
    assert jaccard.loc[10.0] < 1.0
    assert jaccard.loc[10.0] <= jaccard.loc[30.0] <= 1.0


def test_output_schema_and_row_counts(isolated_roots: dict[str, Path]) -> None:
    run = run_experiment(_config())
    summary, per_query = _read(run), _read(run, "robustness_per_query.csv")
    assert list(summary.columns) == [
        "representation", "seed", "gallery", "perturbation", "level", "metric", "k",
        "value", "ci_low", "ci_high", "n_queries",
    ]  # fmt: skip
    assert list(per_query.columns) == [
        "item_id", "representation", "seed", "gallery", "perturbation", "level", "metric",
        "k", "value",
    ]  # fmt: skip
    n_conditions = 1 + 2  # null + two noise levels
    assert len(summary) == n_conditions * 6  # five per-query metrics + original_top10
    assert len(per_query) == n_conditions * 5 * N_ITEMS
    assert (summary["n_queries"] == N_ITEMS).all()
    assert not (per_query["metric"] == "original_top10").any()


def test_two_runs_write_identical_csv(isolated_roots: dict[str, Path]) -> None:
    config = _config(representations=["toy_robust", "random"])
    first, second = run_experiment(config), run_experiment(config)
    for name in ("robustness.csv", "robustness_per_query.csv"):
        assert (first.metrics_dir / name).read_bytes() == (second.metrics_dir / name).read_bytes()


def test_perturbed_vectors_are_cached(
    isolated_roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    first = run_experiment(config)

    def boom(self, item_id, y):
        raise AssertionError("extraction ran although every condition was cached")

    monkeypatch.setattr(_ToyRobustExtractor, "extract_from_waveform", boom)
    second = run_experiment(config)
    assert (first.metrics_dir / "robustness.csv").read_bytes() == (
        second.metrics_dir / "robustness.csv"
    ).read_bytes()


def test_conditions_can_differ_per_representation(isolated_roots: dict[str, Path]) -> None:
    config = _config(
        conditions={"default": [{"noise_snr_db": 10.0}], "random": [{"gain_db": 6.0}]},
        representations=["toy_robust", "random"],
    )
    out = _read(run_experiment(config))
    seen = out.groupby("representation")["perturbation"].unique().map(sorted)
    assert seen.loc["toy_robust"] == ["noise_snr_db", "none"]
    assert seen.loc["random"] == ["gain_db", "none"]


def test_the_evaluator_is_off_unless_enabled(isolated_roots: dict[str, Path]) -> None:
    assert issubclass(RobustnessEvaluator, Evaluator)
    assert RobustnessEvaluator.default_enabled is False
    run = run_experiment(_config(robustness=False))
    assert not (run.metrics_dir / "robustness.csv").exists()


def test_perturbed_rows_do_not_change_the_fitted_transform(
    isolated_roots: dict[str, Path],
) -> None:
    """Scaler and PCA are fitted on the retrieval rows only, so the stacked
    embedding of an item equals its embedding in the plain pipeline."""
    config = _config(embedding={"fit_on": "training", "pca": {"enabled": True, "n_components": 6}})
    index = _write_index(lambda i: "training" if i < N_ITEMS // 2 else "test")
    extract_stage(config)
    features_dir = features_cache_dir(config)
    raw, ids = np.load(features_dir / "features.npy"), np.load(features_dir / "ids.npy")
    plain, _ = build_embeddings(raw, ids, index, config)

    noisy_condition = ("noise_snr_db", 10.0)
    stacked = _embed(
        config,
        index,
        {NULL_CONDITION: raw, noisy_condition: raw + np.float32(0.5)},
        [NULL_CONDITION, noisy_condition],
    )

    np.testing.assert_allclose(stacked[NULL_CONDITION], plain, atol=1e-6)
    assert not np.allclose(stacked[noisy_condition], plain)


def test_a_failing_item_is_reported_not_dropped(
    isolated_roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    extract_stage(config)  # retrieval features first, with the healthy extractor
    original = _ToyRobustExtractor.extract_from_waveform

    def flaky(self, item_id, y):
        if item_id == 7:
            raise RuntimeError("bad clip")
        return original(self, item_id, y)

    monkeypatch.setattr(_ToyRobustExtractor, "extract_from_waveform", flaky)
    with pytest.raises(ConfigError, match="could not be extracted"):
        run_experiment(config)
