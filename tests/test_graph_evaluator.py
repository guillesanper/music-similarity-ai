"""Tests for ``graph_metrics`` and the structure evaluator, on graphs small enough
to have hand-computable answers."""

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
from musicsim.evaluation.graph_evaluator import METRIC_VIEWS, graph_metrics
from musicsim.experiments import run_experiment
from musicsim.extractors.base import Extractor
from musicsim.graphs.base import Graph
from musicsim.registry import register_extractor


def _graph(neighbors: list[list[int]]) -> Graph:
    nb = np.array(neighbors, dtype=np.int32)
    return Graph(
        neighbors=nb,
        weights=np.full(nb.shape, 0.5, dtype=np.float32),
        ids=np.arange(len(nb)),
    )


# 0 -> 1, 1 -> 0 (mutual); 2 -> 0, 3 -> 0 (point at node 0); node 2 and 3 unreferenced.
STAR = _graph([[1], [0], [0], [0]])


def test_graph_metrics_on_star() -> None:
    # N_k = [3, 1, 0, 0]: mean 1, std sqrt(1.5), skewness 1 / sqrt(1.5); threshold 2 -> one hub.
    # Union edges are 0-1, 0-2, 0-3: a star without triangles.
    metrics = graph_metrics(STAR, np.array([0, 0, 1, 0]))
    assert metrics == pytest.approx(
        {
            "k_occurrence_skewness": 1 / np.sqrt(1.5),
            "label_homophily": 0.75,
            "antihub_fraction": 0.5,
            "hub_fraction": 0.25,
            "max_k_occurrence": 3.0,
            "reciprocity": 0.5,
            "mean_clustering": 0.0,
            "giant_fraction": 1.0,
            "n_components": 1.0,
        }
    )
    assert set(metrics) == set(METRIC_VIEWS)


def test_graph_metrics_rejects_edgeless_graph() -> None:
    empty = Graph(
        neighbors=np.full((3, 2), -1), weights=np.zeros((3, 2), dtype=np.float32), ids=np.arange(3)
    )
    with pytest.raises(ValueError, match="no edges"):
        graph_metrics(empty, np.zeros(3, dtype=int))


# --- end to end ----------------------------------------------------------------
_ANGLES = {1: 0.0, 2: 10.0, 3: 20.0, 4: 170.0, 5: 180.0, 6: 190.0}
_GENRE = {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B", 7: "A"}
# Item 7 is an exact copy of item 1: a near-duplicate, collapsed when it is present.
_ALL_ANGLES = {**_ANGLES, 7: 0.0}


@register_extractor("toy_structure")
class _ToyStructureExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        angle = np.radians(_ALL_ANGLES[item_id])
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)


def _config(ids: list[int], **structure: object) -> Config:
    paths.dataset_dir("toy_structure").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "item_id": ids,
            "path": [f"synthetic/{i}.wav" for i in ids],
            "split": ["test"] * len(ids),
            "artist_id": [100 + i for i in ids],
            "group_id": ids,
            "genre_top": [_GENRE[i] for i in ids],
            "genres_all": ["[]"] * len(ids),
        }
    ).to_csv(paths.dataset_index("toy_structure"), index=False)
    return Config(
        data={
            "experiment": "e00_toy_structure",
            "seed": 0,
            "dataset": {"directory": "toy_structure"},
            "extractor": {"name": "toy_structure", "dim": 2},
            "embedding": {
                "fit_on": "test",
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
            "retrieval": {"gallery": "fma_all", "relevance": "label_match"},
            "evaluation": {
                "retrieval": {"enabled": False},
                "structure": {
                    "enabled": True,
                    "galleries": ["fma_all"],
                    "baselines": [],
                    **structure,
                },
            },
            "runtime": {"n_jobs": 1, "progress": False, "cache": True},
            "stages": ["extract", "embed", "graph", "evaluate"],
        }
    )


@pytest.fixture
def structure_config(isolated_roots: dict[str, Path]) -> Config:
    return _config(sorted(_ANGLES))


def _with_representations(config: Config, representations: list) -> Config:
    data = config.to_dict()
    data["representations"] = representations
    del data["evaluation"]["structure"]["baselines"]
    return Config(data=data, source=config.source)


def test_evaluator_writes_structure_and_hubness(structure_config: Config) -> None:
    run = run_experiment(structure_config)
    structure = pd.read_csv(run.metrics_dir / "structure.csv")
    values = structure.set_index("metric")["value"]
    # Two well-separated 3-item clusters with k=2: every edge stays in its genre,
    # and the graph splits into the two clusters.
    assert values["label_homophily"] == pytest.approx(1.0)
    assert values["n_components"] == 2.0
    assert values["giant_fraction"] == 0.5

    hubness = pd.read_csv(run.metrics_dir / "hubness.csv")
    assert len(hubness) == 6
    assert hubness["k_occurrence"].sum() == 6 * 2  # every node lists k=2 neighbours


def test_evaluator_is_off_by_default(structure_config: Config) -> None:
    data = structure_config.to_dict()
    del data["evaluation"]["structure"]
    run = run_experiment(Config(data=data, source=structure_config.source))
    assert not (run.metrics_dir / "structure.csv").exists()


def test_output_schema_and_views(structure_config: Config) -> None:
    run = run_experiment(structure_config)
    structure = pd.read_csv(run.metrics_dir / "structure.csv")
    assert list(structure.columns) == [
        "representation",
        "seed",
        "gallery",
        "graph",
        "view",
        "metric",
        "value",
        "ci_low",
        "ci_high",
        "n_replicates",
    ]
    assert sorted(structure["metric"]) == sorted(METRIC_VIEWS)  # one row per metric
    assert (structure["view"] == structure["metric"].map(METRIC_VIEWS)).all()
    assert set(structure["view"]) == {"directed", "union"}
    assert structure["ci_low"].isna().all() and structure["ci_high"].isna().all()
    assert (structure["n_replicates"] == 0).all()
    assert set(structure["graph"]) == {"knn2"} and set(structure["gallery"]) == {"fma_all"}

    hubness = pd.read_csv(run.metrics_dir / "hubness.csv")
    assert list(hubness.columns) == [
        "item_id",
        "representation",
        "seed",
        "gallery",
        "graph",
        "k_occurrence",
    ]


def test_graphs_are_written_and_can_be_read_back(structure_config: Config) -> None:
    run = run_experiment(structure_config)
    folder = run.directory / "graphs" / "toy_structure" / "fma_all" / "knn2"
    assert (folder / "neighbors.npz").is_file() and (folder / "edges.csv").is_file()
    graph = Graph.read(folder)
    hubness = pd.read_csv(run.metrics_dir / "hubness.csv")
    np.testing.assert_array_equal(graph.ids, hubness["item_id"])
    assert len(pd.read_csv(folder / "edges.csv")) == 6 * 2


def test_planted_duplicate_is_collapsed(
    isolated_roots: dict[str, Path], structure_config: Config
) -> None:
    clean = pd.read_csv(run_experiment(structure_config).metrics_dir / "hubness.csv")
    run = run_experiment(_config([*sorted(_ANGLES), 7]))
    hubness = pd.read_csv(run.metrics_dir / "hubness.csv")
    assert len(hubness) == len(clean) == 6  # 7 items, one collapsed
    assert 7 not in set(hubness["item_id"])  # the copy with the larger id is the one dropped


def test_same_representation_under_two_labels_gives_identical_values(
    structure_config: Config,
) -> None:
    config = _with_representations(
        structure_config,
        ["toy_structure", {"label": "copy", "extractor": "toy_structure"}],
    )
    structure = pd.read_csv(run_experiment(config).metrics_dir / "structure.csv")
    by_label = {
        label: frame.set_index("metric")["value"]
        for label, frame in structure.groupby("representation")
    }
    pd.testing.assert_series_equal(by_label["toy_structure"], by_label["copy"], check_names=False)


def test_seed_column_is_the_seed_of_each_representation(structure_config: Config) -> None:
    config = _with_representations(
        structure_config,
        [
            "toy_structure",
            {"label": "random_s1", "extractor": "random", "overrides": {"seed": 1}},
            {"label": "random_s2", "extractor": "random", "overrides": {"seed": 2}},
        ],
    )
    run = run_experiment(config)
    expected = {"toy_structure": [0], "random_s1": [1], "random_s2": [2]}
    for name in ("structure.csv", "hubness.csv"):
        frame = pd.read_csv(run.metrics_dir / name)
        assert frame.groupby("representation")["seed"].unique().map(list).to_dict() == expected


def test_unknown_reference_is_a_config_error(isolated_roots: dict[str, Path]) -> None:
    with pytest.raises(ConfigError, match="reference is 'mert'"):
        run_experiment(_config(sorted(_ANGLES), reference="mert"))
