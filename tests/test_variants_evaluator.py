"""Tests for the graph-variants evaluator on a small clustered gallery with known structure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import musicsim.evaluation
import musicsim.graphs
import musicsim.relevance  # noqa: F401  (registers "label_match")
from musicsim import paths
from musicsim.config import Config, ConfigError
from musicsim.embeddings import load_embeddings
from musicsim.evaluation.retrieval import compute_per_query_metrics
from musicsim.evaluation.variants import GRAPH_VARIANTS, HEADLINE_METRICS, sign_consistency
from musicsim.experiments import run_experiment
from musicsim.extractors.base import Extractor
from musicsim.graphs.base import build_graph
from musicsim.graphs.duplicates import duplicate_groups, near_duplicate_pairs
from musicsim.registry import register_extractor
from musicsim.seeding import generator

N_ITEMS = 60  # k = 50 needs at least 51 items
N_GENRES = 3
DUPLICATE = 59  # an exact copy of item 0
_BASE = {i: (0 if i == DUPLICATE else i) for i in range(N_ITEMS)}
_CENTRES = generator(0, "toy_variants", "centres").normal(size=(N_GENRES, 8))


@register_extractor("toy_variants")
class _ToyVariantsExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        base = _BASE[item_id]
        noise = generator(0, "toy_variants", "noise", base).normal(scale=0.4, size=8)
        return (_CENTRES[base // (N_ITEMS // N_GENRES)] + noise).astype(np.float32)


def _config(representations: list | None = None, **variants: object) -> Config:
    ids = list(range(N_ITEMS))
    paths.dataset_dir("toy_variants").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "item_id": ids,
            "path": [f"synthetic/{i}.wav" for i in ids],
            "split": ["test"] * N_ITEMS,
            "artist_id": [100 + i for i in ids],
            "group_id": ids,
            "genre_top": [f"g{_BASE[i] // (N_ITEMS // N_GENRES)}" for i in ids],
            "genres_all": ["[]"] * N_ITEMS,
        }
    ).to_csv(paths.dataset_index("toy_variants"), index=False)
    data: dict = {
        "experiment": "e00_toy_variants",
        "seed": 0,
        "dataset": {"directory": "toy_variants"},
        "extractor": {"name": "toy_variants", "dim": 8},
        "embedding": {
            "fit_on": "test",
            "standardize": False,
            "l2_normalize": True,
            "pca": {"enabled": False, "n_components": None},
        },
        "graph": {
            "builder": "knn",
            "k": 10,
            "exclude_self": True,
            "near_duplicate_threshold": 0.9999,
        },
        "retrieval": {"gallery": "fma_all", "relevance": "label_match", "ks": [1, 5, 10, 20]},
        "bootstrap": {"n_resamples": 200, "confidence": 0.95},
        "evaluation": {
            "retrieval": {"enabled": False},
            "variants": {"enabled": True, "galleries": ["fma_all"], **variants},
        },
        "runtime": {"n_jobs": 1, "progress": False, "cache": True},
        "stages": ["extract", "embed", "graph", "evaluate"],
    }
    if representations is not None:
        data["representations"] = representations
    return Config(data=data)


@pytest.fixture
def config(isolated_roots: dict[str, Path]) -> Config:
    return _config()


def _read(run, name: str) -> pd.DataFrame:
    return pd.read_csv(run.metrics_dir / name)


def test_knn10_reproduces_plain_retrieval_and_is_not_completed_in_raw(config: Config) -> None:
    run = run_experiment(config)
    per_query = _read(run, "variants_per_query.csv")
    X, ids = load_embeddings(config)
    pairs = near_duplicate_pairs(X, ids, threshold=0.9999)
    assert len(pairs) == 1  # the planted duplicate
    genre = np.arange(N_ITEMS) // (N_ITEMS // N_GENRES)
    genre[DUPLICATE] = genre[0]
    groups = {"dedup": duplicate_groups(ids, pairs), "raw": np.arange(N_ITEMS)}
    for variant, group in groups.items():
        rows = np.arange(N_ITEMS)
        expected = compute_per_query_metrics(X, ids, rows, group, genre, [1, 5, 10, 20])
        got = per_query[(per_query["graph"] == "knn10") & (per_query["variant"] == variant)]
        merged = expected.merge(got, on=["item_id", "metric", "k"], suffixes=("_e", "_g"))
        assert len(merged) == len(expected) == len(got)
        np.testing.assert_allclose(merged["value_g"], merged["value_e"], atol=1e-12)
    raw = per_query[(per_query["graph"] == "knn10") & (per_query["variant"] == "raw")]
    assert not raw["completed"].any()


def test_completed_and_isolated_fractions(config: Config) -> None:
    run = run_experiment(config)
    summary = _read(run, "variants_summary.csv")
    headline = summary[(summary["metric"] == "R") & (summary["k"] == 10)]
    table = headline.pivot_table(
        index="graph", columns="variant", values=["completed_fraction", "isolated_fraction"]
    )
    print(table.to_string())
    assert (table.loc["knn5", "completed_fraction"] == 1.0).all()  # 5 < 10 neighbours
    assert (table.loc["knn20", "completed_fraction"] == 0.0).all()
    assert set(summary["graph"]) == set(GRAPH_VARIANTS)
    # Isolated nodes are a property of the graph: the same under every exclusion variant.
    assert (summary.groupby(["representation", "graph"])["isolated_fraction"].nunique() == 1).all()

    X, ids = load_embeddings(config)
    for graph_name, overrides in GRAPH_VARIANTS.items():
        graph = build_graph(X, ids, {**config.section("graph"), **overrides})
        stored = summary[summary["graph"] == graph_name]["isolated_fraction"].unique()
        assert stored == pytest.approx([(graph.degree == 0).mean()])
    # The check is not vacuous: the epsilon graph of this gallery does leave nodes isolated.
    assert summary[summary["graph"] == "epsilon"]["isolated_fraction"].iloc[0] > 0


def test_delta_of_a_representation_with_itself_is_exactly_zero(
    isolated_roots: dict[str, Path],
) -> None:
    config = _config(["toy_variants", {"label": "copy", "extractor": "toy_variants"}])
    deltas = _read(run_experiment(config), "variants_deltas.csv")
    assert len(deltas) == len(GRAPH_VARIANTS) * len(HEADLINE_METRICS)
    assert (deltas["delta"] == 0).all()
    assert (deltas["ci_low"] == 0).all() and (deltas["ci_high"] == 0).all()
    assert (deltas["sign"] == 0).all()
    assert (deltas["n_resamples"] == 200).all()
    assert sign_consistency(deltas)["same_sign"].all()


def test_random_is_below_a_clustered_representation_in_every_variant(
    isolated_roots: dict[str, Path],
) -> None:
    config = _config(
        [
            "toy_variants",
            {"label": "random_s1", "extractor": "random", "overrides": {"seed": 1}},
        ]
    )
    deltas = _read(run_experiment(config), "variants_deltas.csv")
    assert set(deltas["representation"]) == {"random_s1"} and set(deltas["reference"]) == {
        "toy_variants"
    }
    assert (deltas["delta"] < 0).all() and (deltas["sign"] == -1).all()
    consistency = sign_consistency(deltas)
    assert len(consistency) == len(HEADLINE_METRICS)
    assert consistency["same_sign"].all() and (consistency["sign"] == -1).all()
    assert (consistency["n_graphs"] == len(GRAPH_VARIANTS)).all()


def test_sign_consistency_detects_a_sign_change() -> None:
    base = {
        "gallery": "g", "representation": "r", "seed": 0, "reference": "ref",
        "metric": "R", "k": 10,
    }  # fmt: skip
    frame = pd.DataFrame([{**base, "graph": "a", "sign": 1}, {**base, "graph": "b", "sign": -1}])
    row = sign_consistency(frame).iloc[0]
    assert not row["same_sign"] and np.isnan(row["sign"]) and row["n_graphs"] == 2


def test_two_runs_give_identical_csvs(config: Config) -> None:
    first = run_experiment(config)
    first_files = {
        name: _read(first, name)
        for name in ("variants_per_query.csv", "variants_summary.csv", "variants_deltas.csv")
    }
    second = run_experiment(config)
    for name, frame in first_files.items():
        pd.testing.assert_frame_equal(frame, _read(second, name))


def test_evaluator_is_off_by_default(config: Config) -> None:
    data = config.to_dict()
    del data["evaluation"]["variants"]
    run = run_experiment(Config(data=data))
    assert not list(run.metrics_dir.glob("variants_*.csv"))


def test_evaluator_writes_three_files_with_their_schema(config: Config) -> None:
    run = run_experiment(config)
    assert list(_read(run, "variants_per_query.csv").columns) == [
        "item_id", "representation", "seed", "gallery", "graph", "variant",
        "metric", "k", "value", "completed",
    ]  # fmt: skip
    assert list(_read(run, "variants_summary.csv").columns) == [
        "representation", "seed", "gallery", "graph", "variant", "metric", "k",
        "value", "completed_fraction", "isolated_fraction",
    ]  # fmt: skip
    # Single representation: nothing to compare with, so the matrix is empty but well formed.
    assert list(_read(run, "variants_deltas.csv").columns) == [
        "representation", "seed", "reference", "gallery", "graph", "metric", "k",
        "delta", "ci_low", "ci_high", "sign", "n_resamples",
    ]  # fmt: skip


def test_mutual_proximity_refuses_a_large_gallery(isolated_roots: dict[str, Path]) -> None:
    config = _config(mutual_proximity_max_nodes=10)
    with pytest.raises(ConfigError, match=rf"{N_ITEMS} items.*10"):
        run_experiment(config)


def test_unknown_reference_is_rejected(isolated_roots: dict[str, Path]) -> None:
    with pytest.raises(ConfigError, match="reference"):
        run_experiment(_config(reference="nope"))
