"""Tests for :mod:`musicsim.graphs.knn` and the ``Graph`` container."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from musicsim.graphs.base import Graph, build_graph
from musicsim.graphs.knn import KnnGraphBuilder, top_k_by_cosine


def _unit(vectors: list[tuple[float, float]]) -> np.ndarray:
    X = np.asarray(vectors, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_top_k_recovers_known_neighbors_by_angle() -> None:
    # Five unit vectors at increasing angles from item 0: similarity to item 0
    # strictly decreases with the angle, so the neighbour order is known.
    angles = np.radians([0, 10, 20, 30, 40])
    X = _unit(list(zip(np.cos(angles), np.sin(angles), strict=True)))

    neighbors, weights = top_k_by_cosine(X, k=2)

    assert neighbors[0].tolist() == [1, 2]  # item 0's closest, then next closest
    assert weights[0, 0] > weights[0, 1] > 0
    np.testing.assert_allclose(weights[0, 0], np.cos(angles[1]), atol=1e-5)


def test_self_is_excluded_by_index_even_with_a_duplicate_vector() -> None:
    # Item 0 and item 1 are identical (similarity 1 both ways); a value-based
    # exclusion rule could drop item 1 instead of item 0 for row 0.
    X = _unit([(1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.7, 0.3)])
    neighbors, weights = top_k_by_cosine(X, k=1)

    assert neighbors[0, 0] == 1  # row 0 never lists itself
    assert neighbors[1, 0] == 0  # row 1 never lists itself
    np.testing.assert_allclose(weights[0, 0], 1.0, atol=1e-6)


def test_without_exclude_self_row_i_may_list_itself() -> None:
    X = _unit([(1.0, 0.0), (0.9, 0.1), (0.0, 1.0)])
    neighbors, _ = top_k_by_cosine(X, k=1, exclude_self=False)
    assert neighbors[0, 0] == 0  # item 0 is its own best match


def test_k_out_of_range_is_rejected() -> None:
    X = _unit([(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    with pytest.raises(ValueError, match=r"\[1, 2\]"):
        top_k_by_cosine(X, k=3)  # only 2 other rows to exclude self


def test_block_rows_does_not_change_the_result() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(37, 5)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    whole = top_k_by_cosine(X, k=4, block_rows=1000)
    blocked = top_k_by_cosine(X, k=4, block_rows=3)
    np.testing.assert_array_equal(whole[0], blocked[0])
    np.testing.assert_allclose(whole[1], blocked[1], atol=1e-6)


def test_knn_graph_builder_matches_top_k_by_cosine() -> None:
    X = _unit([(1.0, 0.0), (0.9, 0.1), (0.0, 1.0), (0.1, 0.9)])
    ids = np.array([10, 20, 30, 40])
    graph = KnnGraphBuilder({"k": 2, "exclude_self": True}).build(X, ids)

    expected_neighbors, expected_weights = top_k_by_cosine(X, k=2)
    np.testing.assert_array_equal(graph.neighbors, expected_neighbors)
    np.testing.assert_allclose(graph.weights, expected_weights)
    np.testing.assert_array_equal(graph.ids, ids)


def test_build_graph_dispatches_through_the_registry() -> None:
    X = _unit([(1.0, 0.0), (0.9, 0.1), (0.0, 1.0)])
    ids = np.array([1, 2, 3])
    graph = build_graph(X, ids, {"builder": "knn", "k": 1})
    assert graph.k == 1
    assert graph.n_nodes == 3


def test_graph_to_edges_frame_has_the_b1_8_columns() -> None:
    graph = Graph(
        neighbors=np.array([[1], [0]], dtype=np.int32),
        weights=np.array([[0.9], [0.8]], dtype=np.float32),
        ids=np.array([100, 200]),
    )
    edges = graph.to_edges_frame()
    assert list(edges.columns) == ["src_id", "dst_id", "weight", "rank"]
    assert edges["src_id"].tolist() == [100, 200]
    assert edges["dst_id"].tolist() == [200, 100]
    assert edges["rank"].tolist() == [1, 1]


def test_graph_write_and_read_round_trip(tmp_path: Path) -> None:
    graph = Graph(
        neighbors=np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
        weights=np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], dtype=np.float32),
        ids=np.array([10, 20, 30]),
    )
    directory = tmp_path / "graph"
    graph.write(directory)

    assert (directory / "neighbors.npz").is_file()
    assert (directory / "edges.csv").is_file()

    loaded = Graph.read(directory)
    np.testing.assert_array_equal(loaded.neighbors, graph.neighbors)
    np.testing.assert_allclose(loaded.weights, graph.weights)
    np.testing.assert_array_equal(loaded.ids, graph.ids)


def test_graph_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="disagree"):
        Graph(
            neighbors=np.zeros((3, 2), dtype=np.int32),
            weights=np.zeros((3, 1), dtype=np.float32),
            ids=np.arange(3),
        )
