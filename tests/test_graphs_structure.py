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
from musicsim.graphs.base import PADDING, Graph
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


# --- graphs with padding ---------------------------------------------------------
# Node 2 has no neighbour at all; nodes 1 and 4 have one; nodes 0 and 3 have two.
#   edges: 0->1 (.9), 0->4 (.5), 1->0 (.8), 3->4 (.7), 3->0 (.3), 4->3 (.6)
_P = PADDING
PADDED = Graph(
    neighbors=np.array([[1, 4, _P], [0, _P, _P], [_P, _P, _P], [4, 0, _P], [3, _P, _P]]),
    weights=np.array(
        [[0.9, 0.5, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.0], [0.7, 0.3, 0.0], [0.6, 0.0, 0.0]],
        dtype=np.float32,
    ),
    ids=np.arange(5),
)
# No edge at all: every slot is padding.
EMPTY = Graph(
    neighbors=np.full((3, 2), _P), weights=np.zeros((3, 2), dtype=np.float32), ids=np.arange(3)
)


def test_in_degree_ignores_padding() -> None:
    np.testing.assert_array_equal(in_degree(PADDED), [2, 1, 0, 1, 2])
    np.testing.assert_array_equal(in_degree(EMPTY), [0, 0, 0])


def test_adjacency_has_no_edge_for_padding() -> None:
    A = adjacency(PADDED).toarray()
    assert A.sum() == 6
    # Padding read as an index would add 0/1/2/3/4 -> N-1 edges; only 0 and 3 point at node 4.
    np.testing.assert_array_equal(A[:, 4], [1, 0, 0, 1, 0])
    assert A[2].sum() == 0
    assert adjacency(EMPTY).nnz == 0


def test_label_homophily_counts_real_edges_only() -> None:
    labels = np.array([0, 0, 1, 1, 1])  # same: 0->1, 1->0, 3->4, 4->3; different: 0->4, 3->0
    assert label_homophily(PADDED, labels) == pytest.approx(4 / 6)
    assert label_homophily(EMPTY, np.zeros(3, dtype=int)) == 0.0


def test_reciprocity_on_padded_graph() -> None:
    # 0<->1 and 3<->4 are mutual (4 of 6 edges); 0->4 and 3->0 are not.
    assert reciprocity(PADDED) == pytest.approx(4 / 6)
    assert reciprocity(EMPTY) == 0.0


def test_isolated_node_is_its_own_component() -> None:
    assert connected_components(PADDED) == (2, 0.8)  # {0, 1, 3, 4} and {2}
    assert connected_components(EMPTY) == (3, pytest.approx(1 / 3))


def test_hub_threshold_uses_the_actual_mean_in_degree() -> None:
    # N_k = [2, 1, 0, 1, 2]: mean 1.2 (not k = 3), std sqrt(0.56). With hub_factor 0.5 the
    # threshold is about 1.57, so the two nodes with N_k = 2 are hubs; against k = 3 none would be.
    assert hubness_summary(PADDED, hub_factor=0.5)["hub_fraction"] == pytest.approx(0.4)


def test_structure_summary_on_padded_graph() -> None:
    summary = structure_summary(PADDED, np.array([0, 0, 1, 1, 1]))
    assert summary["k"] == 3.0
    assert summary["mean_degree"] == pytest.approx(6 / 5)
    assert summary["mean_edge_weight"] == pytest.approx(3.8 / 6)  # padding zeros do not dilute it
    assert summary["n_components"] == 2.0
    assert summary["largest_component_share"] == pytest.approx(0.8)
    assert summary["label_homophily"] == pytest.approx(4 / 6)


def test_structure_summary_of_edgeless_graph() -> None:
    summary = structure_summary(EMPTY, np.zeros(3, dtype=int))
    assert summary["mean_degree"] == 0.0
    assert summary["mean_edge_weight"] == 0.0
    assert summary["label_homophily"] == 0.0


def test_structure_summary_of_unpadded_graph_is_unchanged() -> None:
    # STAR by hand: N_k = [3, 1, 0, 0] (mean 1, std sqrt(1.5)), skewness 1 / sqrt(1.5),
    # Gini (2 * 15 - 5 * 4) / (4 * 4) = 0.625, no hub (threshold 1 + 2 * sqrt(1.5) > 3).
    summary = structure_summary(STAR, np.array([0, 0, 1, 0]))
    assert summary == pytest.approx(
        {
            "n_nodes": 4.0,
            "k": 1.0,
            "mean_degree": 1.0,
            "mean_edge_weight": 0.5,
            "hubness_skewness": 1 / np.sqrt(1.5),
            "in_degree_gini": 0.625,
            "in_degree_max": 3.0,
            "in_degree_std": np.sqrt(1.5),
            "antihub_fraction": 0.5,
            "hub_fraction": 0.0,
            "reciprocity": 0.5,
            "n_components": 1.0,
            "largest_component_share": 1.0,
            "label_homophily": 0.75,
        }
    )
    # Widening the rows with a padding column changes ``k`` and nothing else.
    widened = Graph(
        neighbors=np.hstack([STAR.neighbors, np.full((4, 1), _P)]),
        weights=np.hstack([STAR.weights, np.zeros((4, 1), dtype=np.float32)]),
        ids=STAR.ids,
    )
    expected = structure_summary(STAR, np.array([0, 0, 1, 0]))
    expected["k"] = 2.0
    assert structure_summary(widened, np.array([0, 0, 1, 0])) == pytest.approx(expected)


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
