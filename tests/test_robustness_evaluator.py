"""Tests for the robustness evaluator on synthetic tones written as real WAV files.

The toy extractor reads band energies of a pure tone whose frequency is the item
id, so neighbours are items with close tones and white noise visibly disturbs
them. Perturbed clips go through the real write-WAV-and-extract path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import musicsim.evaluation
import musicsim.relevance  # noqa: F401  (registers "label_match")
from musicsim import audio, paths
from musicsim import perturbations as pt
from musicsim.config import Config, ConfigError
from musicsim.evaluation.base import Evaluator
from musicsim.evaluation.robustness_evaluator import RobustnessEvaluator, parse_conditions
from musicsim.experiments import run_experiment
from musicsim.extractors.base import Extractor
from musicsim.registry import register_extractor

N_ITEMS = 60
N_BANDS = 16
SAMPLE_RATE = 8000
N_SAMPLES = 4000
NOISE_10 = {"condition": "noise", "level": 10}
NOISE_30 = {"condition": "noise", "level": 30}
IDENTITY = {"condition": "identity"}


def _tone(item_id: int) -> np.ndarray:
    t = np.arange(N_SAMPLES) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * (200.0 + 55.0 * item_id) * t)).astype(np.float32)


@register_extractor("toy_robust_eval")
class _ToyExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        y = audio.load_clip(path, self.config)
        power = np.abs(np.fft.rfft(y.astype(np.float64))) ** 2
        energy = np.array([band.sum() for band in np.array_split(power, N_BANDS)])
        return np.log(energy / energy.sum() + 1e-6).astype(np.float32)


@pytest.fixture(autouse=True)
def _admit_toy_extractor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "CACHE_SAFE_EXTRACTORS", frozenset({"mfcc", "toy_robust_eval"}))


def _config(
    tmp_path: Path,
    conditions: list[dict] | None = None,
    representations: list | None = None,
    robustness: bool = True,
) -> Config:
    folder = tmp_path / "clips"
    folder.mkdir(exist_ok=True)
    ids = list(range(N_ITEMS))
    for item in ids:
        sf.write(str(folder / f"{item}.wav"), _tone(item), SAMPLE_RATE, subtype="FLOAT")
    paths.dataset_dir("toy_robust_eval").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "item_id": ids,
            "path": [str(folder / f"{i}.wav") for i in ids],
            "split": ["test"] * N_ITEMS,
            "artist_id": [100 + i for i in ids],
            "group_id": ids,
            "genre_top": [f"g{i // 20}" for i in ids],
            "genres_all": ["[]"] * N_ITEMS,
        }
    ).to_csv(paths.dataset_index("toy_robust_eval"), index=False)
    data: dict = {
        "experiment": "e00_toy_robust_eval",
        "seed": 0,
        "dataset": {"directory": "toy_robust_eval"},
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "clip_seconds": N_SAMPLES / SAMPLE_RATE,
            "clip_samples": N_SAMPLES,
            "min_seconds": 0.1,
            "mono": True,
        },
        "extractor": {"name": "toy_robust_eval", "dim": N_BANDS},
        "embedding": {
            "fit_on": "test",
            "standardize": True,
            "l2_normalize": True,
            "pca": {"enabled": False, "n_components": None},
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
            "gallery": "fma_all",
            "conditions": conditions or [IDENTITY, NOISE_30, NOISE_10],
        }
    if representations is not None:
        data["representations"] = representations
    return Config(data=data)


def _read(run, name: str = "robustness.csv") -> pd.DataFrame:
    return pd.read_csv(run.metrics_dir / name)


def test_identity_is_exact(isolated_roots: dict[str, Path], tmp_path: Path) -> None:
    out = _read(run_experiment(_config(tmp_path)))
    identity = out[out["condition"] == "identity"].set_index("metric")
    assert identity.loc["jaccard@10", "value"] == 1.0
    assert identity.loc["self_rank_median", "value"] == 1.0
    assert identity.loc["self_rank_top1_fraction", "value"] == 1.0
    # With a deterministic extractor the cached originals equal the identity features.
    assert identity.loc["jaccard@10_vs_cached_original", "value"] == 1.0
    for metric in ("delta_R@10", "delta_MAP", "delta_nDCG@10"):
        assert identity.loc[metric, ["value", "ci_low", "ci_high"]].tolist() == [0.0, 0.0, 0.0]


def test_noise_disturbs_the_neighbourhood_monotonically(
    isolated_roots: dict[str, Path], tmp_path: Path
) -> None:
    out = _read(run_experiment(_config(tmp_path)))
    jaccard = out[out["metric"] == "jaccard@10"].set_index("level")["value"]
    assert jaccard.loc[0.0] == 1.0  # identity
    assert jaccard.loc[10.0] < 1.0
    assert jaccard.loc[10.0] <= jaccard.loc[30.0] <= 1.0
    rank = out[(out["metric"] == "self_rank_median") & (out["level"] == 10.0)]["value"].iloc[0]
    assert rank >= 1.0


def test_output_schema_and_row_counts(isolated_roots: dict[str, Path], tmp_path: Path) -> None:
    run = run_experiment(_config(tmp_path))
    summary, per_query = _read(run), _read(run, "robustness_per_query.csv")
    assert list(summary.columns) == [
        "representation", "seed", "gallery", "condition", "level", "metric", "value",
        "ci_low", "ci_high",
    ]  # fmt: skip
    assert list(per_query.columns) == [
        "item_id", "representation", "seed", "gallery", "condition", "level", "metric", "value",
    ]  # fmt: skip
    # Per condition: jaccard, two self-rank summaries and three deltas; plus one control row.
    assert len(summary) == 3 * 6 + 1
    assert len(per_query) == 3 * 5 * N_ITEMS
    assert set(per_query["metric"]) == {"jaccard@10", "R@10", "MAP", "nDCG@10", "self_rank"}


def test_two_runs_write_identical_csv_and_the_second_reuses_the_cache(
    isolated_roots: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    first = run_experiment(config)

    def boom(self, item_id, path):
        raise AssertionError("extracted although every condition was cached")

    monkeypatch.setattr(_ToyExtractor, "extract_one", boom)
    second = run_experiment(config)
    for name in ("robustness.csv", "robustness_per_query.csv"):
        assert (first.metrics_dir / name).read_bytes() == (second.metrics_dir / name).read_bytes()


def test_the_evaluator_is_off_unless_enabled(
    isolated_roots: dict[str, Path], tmp_path: Path
) -> None:
    assert issubclass(RobustnessEvaluator, Evaluator)
    assert RobustnessEvaluator.default_enabled is False
    run = run_experiment(_config(tmp_path, robustness=False))
    assert not (run.metrics_dir / "robustness.csv").exists()


def test_an_extractor_without_audio_to_perturb_is_refused(
    isolated_roots: dict[str, Path], tmp_path: Path
) -> None:
    config = _config(tmp_path, representations=["toy_robust_eval", "random"])
    with pytest.raises(ConfigError, match=r"not verified to respect runtime\.cache"):
        run_experiment(config)


def test_parse_conditions() -> None:
    parsed = parse_conditions([NOISE_10, IDENTITY, {"condition": "gain", "level": -6}])
    assert parsed == [("identity", 0.0), ("noise", 10.0), ("gain", -6.0)]
    with pytest.raises(ConfigError, match="must include 'identity'"):
        parse_conditions([NOISE_10])
    with pytest.raises(ConfigError, match="unknown condition"):
        parse_conditions([IDENTITY, {"condition": "reverb", "level": 1}])
    with pytest.raises(ConfigError, match="needs a 'level'"):
        parse_conditions([IDENTITY, {"condition": "noise"}])
    with pytest.raises(ConfigError, match="repeat"):
        parse_conditions([IDENTITY, NOISE_10, NOISE_10])
    with pytest.raises(ConfigError, match="non-empty list"):
        parse_conditions(None)
