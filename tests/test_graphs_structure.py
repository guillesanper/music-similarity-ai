"""Tests for the graph-structure statistics and their evaluator, on graphs
small enough to have hand-computable answers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import musicsim.evaluation
import musicsim.graphs
import musicsim.relevance  # noqa: F401  (registers "label_match")
from musicsim import paths
from musicsim.config import Config
from musicsim.experiments import run_experiment
from musicsim.extractors.base import Extractor
from musicsim.graphs.base import Graph
from musicsim.graphs.structure import (
    adjacency,
    connected_components,
    gini,
    hubness_summary,
    in_degree,
    label_homophily,
    reciprocity,
    structure_summary,
)
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


def test_in_degree_counts_k_occurrences() -> None:
    np.testing.assert_array_equal(in_degree(STAR), [3, 1, 0, 0])


def test_adjacency_is_the_binary_directed_matrix() -> None:
    A = adjacency(STAR).toarray()
    assert A[2, 0] == 1 and A[0, 2] == 0
    assert A.sum() == 4


def test_reciprocity_counts_mutual_edges() -> None:
    # edges 0->1 and 1->0 are mutual; 2->0 and 3->0 are not.
    assert reciprocity(STAR) == pytest.approx(0.5)


def test_connected_components_are_weak() -> None:
    n, share = connected_components(_graph([[1], [0], [3], [2]]))
    assert (n, share) == (2, 0.5)
    n, share = connected_components(STAR)
    assert (n, share) == (1, 1.0)


def test_label_homophily_is_the_share_of_same_label_edges() -> None:
    labels = np.array([0, 0, 1, 0])  # edges: 0-1 same, 1-0 same, 2-0 diff, 3-0 same
    assert label_homophily(STAR, labels) == pytest.approx(0.75)


def test_label_homophily_rejects_misaligned_labels() -> None:
    with pytest.raises(ValueError, match="one entry per node"):
        label_homophily(STAR, np.array([0, 1]))


def test_gini_extremes() -> None:
    assert gini(np.array([2.0, 2.0, 2.0])) == pytest.approx(0.0)
    assert gini(np.array([0.0, 0.0, 0.0, 4.0])) == pytest.approx(0.75)
    assert gini(np.zeros(3)) == 0.0


def test_hubness_summary_flags_antihubs_and_skew() -> None:
    summary = hubness_summary(STAR)
    assert summary["antihub_fraction"] == pytest.approx(0.5)
    assert summary["in_degree_max"] == 3.0
    assert summary["hubness_skewness"] > 0


def test_uniform_in_degree_has_no_skew() -> None:
    ring = _graph([[1], [2], [3], [0]])
    summary = hubness_summary(ring)
    assert summary["hubness_skewness"] == 0.0
    assert summary["in_degree_gini"] == pytest.approx(0.0)
    assert summary["antihub_fraction"] == 0.0


def test_structure_summary_includes_homophily_only_with_labels() -> None:
    assert "label_homophily" not in structure_summary(STAR)
    assert "label_homophily" in structure_summary(STAR, np.zeros(4, dtype=int))


# --- end to end ----------------------------------------------------------------
_ANGLES = {1: 0.0, 2: 10.0, 3: 20.0, 4: 170.0, 5: 180.0, 6: 190.0}
_GENRE = {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B"}


@register_extractor("toy_structure")
class _ToyStructureExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        angle = np.radians(_ANGLES[item_id])
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)


@pytest.fixture
def structure_config(isolated_roots: dict[str, Path]) -> Config:
    paths.dataset_dir("toy_structure").mkdir(parents=True, exist_ok=True)
    ids = sorted(_ANGLES)
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
            "graph": {"builder": "knn", "k": 2, "exclude_self": True,
                      "near_duplicate_threshold": 0.9999},
            "retrieval": {"gallery": "fma_all", "relevance": "label_match"},
            "evaluation": {
                "retrieval": {"enabled": False},
                "graph_structure": {"enabled": True, "galleries": ["fma_all"], "baselines": []},
            },
            "runtime": {"n_jobs": 1, "progress": False, "cache": True},
            "stages": ["extract", "embed", "graph", "evaluate"],
        }
    )


def test_evaluator_writes_structure_and_in_degree(structure_config: Config) -> None:
    run = run_experiment(structure_config)
    structure = pd.read_csv(run.metrics_dir / "graph_structure.csv")
    values = structure.set_index("metric")["value"]
    # Two well-separated 3-item clusters with k=2: every edge stays in its genre,
    # and the graph splits into the two clusters.
    assert values["label_homophily"] == pytest.approx(1.0)
    assert values["n_components"] == 2.0
    assert values["n_nodes"] == 6.0

    degrees = pd.read_csv(run.metrics_dir / "in_degree.csv")
    assert len(degrees) == 6
    assert degrees["in_degree"].sum() == 6 * 2  # every node lists k=2 neighbours


def test_evaluator_is_off_by_default(structure_config: Config) -> None:
    data = structure_config.to_dict()
    del data["evaluation"]["graph_structure"]
    run = run_experiment(Config(data=data, source=structure_config.source))
    assert not (run.metrics_dir / "graph_structure.csv").exists()
