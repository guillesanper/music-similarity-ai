"""Tests for :mod:`musicsim.experiments`: stage wiring, representation
swapping, and the runner end to end on a tiny synthetic dataset.

The synthetic extractor below places every item at ``(base, 0.3 * cos(theta),
0.3 * sin(theta))``, where ``base`` is ``(1, 0)`` for genre "A" and ``(0, 1)``
for genre "B" and ``theta`` is a distinct multiple of 30 degrees per item —
*except* items 1 and 2, which share ``theta = 0`` and are therefore an exact
near-duplicate pair (cosine similarity 1). Every other pair is at least 30
degrees apart, which keeps their cosine similarity under ~0.989 — comfortably
below :data:`musicsim.graphs.duplicates.NEAR_DUP_SIM` (0.9999, ~0.8 degrees).
A fixed-magnitude, evenly-spaced perturbation like this — rather than a
perturbation growing with the item id — is what keeps every non-planted pair
away from the threshold regardless of how many items there are: two vectors
``(1, 0, n)`` and ``(1, 0, n+1)`` converge in angle as ``n`` grows, which a
first version of this fixture ran into.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import musicsim.evaluation
import musicsim.graphs
import musicsim.relevance  # noqa: F401  (registers the "label_match" relevance function)
from musicsim import paths
from musicsim.config import Config, ConfigError
from musicsim.experiments import (
    embed_stage,
    extract_stage,
    representation_config,
    run_experiment,
)
from musicsim.extractors.base import Extractor
from musicsim.registry import register_extractor

_N_ITEMS = 12


@register_extractor("toy_genre_ret")
class _ToyGenreExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        is_a = item_id <= 6
        base = (1.0, 0.0) if is_a else (0.0, 1.0)
        angle_index = 0 if item_id in (1, 2) else item_id - 2
        theta = np.radians(30.0 * angle_index)
        return np.array(
            [base[0], base[1], 0.3 * np.cos(theta), 0.3 * np.sin(theta)], dtype=np.float32
        )


@pytest.fixture
def toy_retrieval_config(isolated_roots: dict[str, Path]) -> Config:
    paths.dataset_dir("toy_ret").mkdir(parents=True, exist_ok=True)
    item_ids = list(range(1, _N_ITEMS + 1))
    index = pd.DataFrame(
        {
            "item_id": item_ids,
            "path": [f"synthetic/{i:02d}.wav" for i in item_ids],
            "split": ["training"] * _N_ITEMS,
            "artist_id": item_ids,
            "group_id": item_ids,
            "genre_top": ["A" if i <= 6 else "B" for i in item_ids],
            "genres_all": ["[]"] * _N_ITEMS,
        }
    )
    index.to_csv(paths.dataset_index("toy_ret"), index=False)

    return Config(
        data={
            "experiment": "e00_toy_retrieval",
            "seed": 0,
            "dataset": {"directory": "toy_ret"},
            "extractor": {"name": "toy_genre_ret", "dim": 4},
            "embedding": {
                "fit_on": "training",
                "standardize": False,
                "l2_normalize": True,
                "pca": {"enabled": False, "n_components": None},
            },
            "graph": {
                "builder": "knn",
                "k": 2,
                "exclude_self": True,
                "near_duplicate_threshold": 0.9999,
            },
            "retrieval": {
                "ks": [1, 3, 10],
                "queries": "all",
                "variants": ["dedup", "raw", "artist_filter"],
                "relevance": "label_match",
            },
            "bootstrap": {"n_resamples": 200, "confidence": 0.95},
            "evaluation": {
                "retrieval": {"enabled": True, "queries": ["all"], "baselines": ["random"]},
            },
            "runtime": {"n_jobs": 1, "progress": False, "cache": True},
            "stages": ["extract", "embed", "graph", "evaluate"],
        }
    )


# --- representation_config ---------------------------------------------------
def test_representation_config_returns_the_same_object_for_the_main_extractor() -> None:
    config = Config(data={"extractor": {"name": "mfcc"}})
    assert representation_config(config, "mfcc") is config


def test_representation_config_swaps_only_the_extractor_section() -> None:
    config = Config(
        data={
            "extractor": {"name": "mfcc", "n_mfcc": 20},
            "seed": 1,
            "dataset": {"directory": "fma"},
        }
    )
    swapped = representation_config(config, "random")
    assert swapped.require("extractor.name") == "random"
    assert swapped.require("extractor.dim") == 80  # from configs/extractors/random.yaml
    assert swapped.require("seed") == 1
    assert swapped.require("dataset.directory") == "fma"


def test_representation_config_rejects_an_unknown_extractor() -> None:
    config = Config(data={"extractor": {"name": "mfcc"}})
    with pytest.raises(ConfigError, match="nonexistent_extractor"):
        representation_config(config, "nonexistent_extractor")


# --- extract_stage / embed_stage reused by the runner -------------------------
def test_extract_and_embed_stage_are_the_functions_the_cli_calls(
    toy_retrieval_config: Config,
) -> None:
    extract_result = extract_stage(toy_retrieval_config)
    assert extract_result.n_items == _N_ITEMS
    assert extract_result.n_failed == 0
    assert not extract_result.cached

    embed_result = embed_stage(toy_retrieval_config)
    assert embed_result.n_items == _N_ITEMS
    assert embed_result.n_dims == 4

    cached_again = extract_stage(toy_retrieval_config)
    assert cached_again.cached


# --- run_experiment end to end -------------------------------------------------
def test_run_experiment_writes_graph_and_retrieval_outputs(toy_retrieval_config: Config) -> None:
    run = run_experiment(toy_retrieval_config)

    graph_dir = run.directory / "graphs" / "toy_genre_ret"
    assert (graph_dir / "neighbors.npz").is_file()
    assert (graph_dir / "edges.csv").is_file()

    duplicates = pd.read_csv(graph_dir / "duplicates.csv")
    assert len(duplicates) == 1
    assert {duplicates.iloc[0]["id_a"], duplicates.iloc[0]["id_b"]} == {1, 2}

    per_query = pd.read_csv(run.metrics_dir / "per_query.csv")
    assert set(per_query["representation"]) == {"toy_genre_ret", "random"}
    assert set(per_query["variant"]) == {"dedup", "raw", "artist_filter"}
    assert per_query["value"].between(0, 1).all()

    retrieval = pd.read_csv(run.metrics_dir / "retrieval.csv")
    expected_columns = {
        "representation", "variant", "queries", "genre", "metric", "k",
        "value", "ci_low", "ci_high", "ci_low_artist", "ci_high_artist",
    }
    assert expected_columns.issubset(retrieval.columns)
    assert (retrieval["ci_low"] <= retrieval["value"] + 1e-9).all()
    assert (retrieval["value"] <= retrieval["ci_high"] + 1e-9).all()

    per_genre = pd.read_csv(run.metrics_dir / "retrieval_per_genre.csv")
    assert set(per_genre["genre"]) == {"A", "B"}

    record_status = (run.directory / "run.json").read_text(encoding="utf-8")
    assert '"status": "ok"' in record_status


def test_run_experiment_the_main_representation_beats_the_random_baseline(
    toy_retrieval_config: Config,
) -> None:
    run = run_experiment(toy_retrieval_config)
    retrieval = pd.read_csv(run.metrics_dir / "retrieval.csv")

    headline = retrieval[
        (retrieval["variant"] == "dedup") & (retrieval["metric"] == "MAP")
    ].set_index("representation")["value"]
    assert headline["toy_genre_ret"] > headline["random"]


def test_run_experiment_records_a_failed_stage(toy_retrieval_config: Config) -> None:
    # "graph" alone, with nothing extracted or embedded yet, must fail loudly
    # rather than silently produce an empty graph.
    bad = Config(data={**toy_retrieval_config.to_dict(), "stages": ["graph"]})
    with pytest.raises(ConfigError, match="no cached embeddings"):
        run_experiment(bad)

    record = (bad_run_directory_of(bad) / "run.json").read_text(encoding="utf-8")
    assert '"status": "failed"' in record
    assert '"error": "ConfigError' in record


def bad_run_directory_of(config: Config) -> Path:
    """The run directory of the single (failed) run for ``config``'s hash."""
    experiment_dir = paths.runs_root() / config.name
    (only,) = list(experiment_dir.iterdir())
    return only
