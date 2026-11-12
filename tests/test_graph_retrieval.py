"""Tests for :mod:`musicsim.evaluation.graph_retrieval`: ranking a graph's neighbours
first and completing by cosine, on synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from musicsim.evaluation.graph_retrieval import graph_per_query_metrics, graph_ranking
from musicsim.evaluation.retrieval import compute_per_query_metrics
from musicsim.graphs.base import PADDING, Graph
from musicsim.graphs.knn import KnnGraphBuilder
from musicsim.graphs.variants import EpsilonGraphBuilder, MutualKnnGraphBuilder
from musicsim.relevance.base import label_codes

KS = [1, 5, 10, 20]


def _unit_row(angle_degrees: float) -> tuple[float, float]:
    angle = np.radians(angle_degrees)
    return (float(np.cos(angle)), float(np.sin(angle)))


@pytest.fixture
def gallery(synthetic_index, synthetic_embeddings) -> dict:
    X, ids = synthetic_embeddings
    return {
        "X": X,
        "ids": ids,
        "rows": np.arange(len(ids)),
        "genre": label_codes(synthetic_index["label"]),
        "groups": {
            "raw": np.arange(len(ids)),
            "dedup": synthetic_index["group_id"].to_numpy(),
            "artist_filter": label_codes(synthetic_index["artist_id"]),
        },
    }


def _reference(gallery: dict, variant: str) -> pd.DataFrame:
    return compute_per_query_metrics(
        gallery["X"], gallery["ids"], gallery["rows"], gallery["groups"][variant],
        gallery["genre"], KS,
    )


@pytest.mark.parametrize("variant", ["raw", "dedup", "artist_filter"])
def test_canonical_knn_reproduces_the_cosine_metrics_exactly(gallery: dict, variant: str) -> None:
    graph = KnnGraphBuilder({"k": 10}).build(gallery["X"], gallery["ids"])
    frame, _ = graph_per_query_metrics(
        graph, gallery["X"], gallery["ids"], gallery["rows"], gallery["groups"][variant],
        gallery["genre"], KS,
    )
    pd.testing.assert_frame_equal(frame, _reference(gallery, variant), check_exact=True)


def test_knn_is_never_completed_under_raw(gallery: dict) -> None:
    graph = KnnGraphBuilder({"k": 10}).build(gallery["X"], gallery["ids"])
    _, completed = graph_per_query_metrics(
        graph, gallery["X"], gallery["ids"], gallery["rows"], gallery["groups"]["raw"],
        gallery["genre"], KS,
    )
    assert completed.shape == (len(gallery["ids"]),)
    assert not completed.any()


def test_mutual_knn_completed_fraction_is_the_fraction_of_rows_below_ten(gallery: dict) -> None:
    graph = MutualKnnGraphBuilder({"k": 10}).build(gallery["X"], gallery["ids"])
    _, completed = graph_per_query_metrics(
        graph, gallery["X"], gallery["ids"], gallery["rows"], gallery["groups"]["raw"],
        gallery["genre"], KS,
    )
    np.testing.assert_array_equal(completed, graph.degree < 10)
    assert 0.0 < completed.mean() < 1.0  # the fixture must actually exercise both cases


def test_complete_graph_ranks_by_pure_cosine(gallery: dict) -> None:
    graph = EpsilonGraphBuilder({"k": 10, "epsilon": float("-inf")}).build(
        gallery["X"], gallery["ids"]
    )
    assert (graph.degree == len(gallery["ids"]) - 1).all()
    frame, completed = graph_per_query_metrics(
        graph, gallery["X"], gallery["ids"], gallery["rows"], gallery["groups"]["raw"],
        gallery["genre"], KS,
    )
    pd.testing.assert_frame_equal(frame, _reference(gallery, "raw"), check_exact=True)
    assert not completed.any()


def test_short_row_is_completed_by_cosine_after_the_graph_neighbours() -> None:
    # Cosine similarity to item 0 falls with the index: 1 > 2 > 3 > 4 > 5.
    X = np.asarray([_unit_row(a) for a in (0, 10, 20, 30, 40, 50)], dtype=np.float32)
    ids = np.arange(6)
    neighbors = np.full((6, 3), PADDING, dtype=np.int32)
    weights = np.zeros((6, 3), dtype=np.float32)
    neighbors[0, :2] = [3, 1]  # deliberately not the cosine order, and one slot short
    weights[0, :2] = [0.9, 0.5]
    graph = Graph(neighbors=neighbors, weights=weights, ids=ids)

    order, completed = graph_ranking(graph, X, np.array([0]), np.arange(6), completion_depth=3)
    np.testing.assert_array_equal(order[0], [3, 1, 2, 4, 5, 0])
    assert completed.tolist() == [True]
    _, full = graph_ranking(graph, X, np.array([0]), np.arange(6), completion_depth=2)
    assert full.tolist() == [False]


def test_graph_neighbour_in_the_exclusion_group_is_dropped_not_ranked() -> None:
    X = np.asarray([_unit_row(a) for a in (0, 10, 20, 30)], dtype=np.float32)
    ids = np.arange(4)
    group = np.array([0, 0, 1, 2])  # item 1 is item 0's duplicate
    neighbors = np.array([[1, 3], [0, 2], [1, 3], [2, 0]], dtype=np.int32)
    graph = Graph(neighbors=neighbors, weights=np.full((4, 2), 0.5, np.float32), ids=ids)

    order, completed = graph_ranking(graph, X, np.array([0]), group, completion_depth=2)
    np.testing.assert_array_equal(order[0], [3, 2, 0, 1])  # graph, cosine, group
    assert completed.tolist() == [True]  # one usable graph neighbour left


def test_ranking_is_deterministic(gallery: dict) -> None:
    graph = MutualKnnGraphBuilder({"k": 10}).build(gallery["X"], gallery["ids"])
    args = (
        graph, gallery["X"], gallery["ids"], gallery["rows"], gallery["groups"]["dedup"],
        gallery["genre"], KS,
    )
    first, flags_first = graph_per_query_metrics(*args)
    second, flags_second = graph_per_query_metrics(*args)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    np.testing.assert_array_equal(flags_first, flags_second)


def test_graph_over_other_ids_is_rejected(gallery: dict) -> None:
    graph = KnnGraphBuilder({"k": 10}).build(gallery["X"][:-1], gallery["ids"][:-1])
    with pytest.raises(ValueError, match="exactly the ids"):
        graph_per_query_metrics(
            graph, gallery["X"], gallery["ids"], gallery["rows"], gallery["groups"]["raw"],
            gallery["genre"], KS,
        )
