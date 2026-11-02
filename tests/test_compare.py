"""Tests for graph comparison, on 4-node graphs whose overlaps are counted by hand."""

from __future__ import annotations

import numpy as np
import pytest

from musicsim.graphs.base import PADDING, Graph
from musicsim.graphs.compare import (
    edge_overlap,
    k_occurrence_spearman,
    neighbor_jaccard,
    partition_ari,
)


def _graph(neighbors: list[list[int]], ids: np.ndarray | None = None) -> Graph:
    nb = np.array(neighbors, dtype=np.int32)
    return Graph(
        neighbors=nb,
        weights=np.where(nb == PADDING, 0.0, 1.0).astype(np.float32),
        ids=np.arange(len(nb)) if ids is None else ids,
    )


# k-occurrences [2, 2, 3, 1]: not constant.
G = _graph([[1, 2], [0, 2], [3, 0], [2, 1]])
# Shares 1/3, 1, 1/3, 1/3 of its neighbour sets with G; 5 of 11 directed edges.
H = _graph([[1, 3], [0, 2], [3, 1], [0, 1]])
# k-occurrences [2, 2, 1, 3].
REVERSED = _graph([[3, 1], [3, 0], [3, 1], [0, 2]])
# No directed edge in common with `RING_A`.
RING_A = _graph([[1], [0], [3], [2]])
RING_B = _graph([[2], [3], [0], [1]])


def test_identical_graph_scores_one_everywhere() -> None:
    assert neighbor_jaccard(G, G) == 1.0
    assert edge_overlap(G, G) == 1.0
    assert k_occurrence_spearman(G, G) == pytest.approx(1.0, abs=1e-15)
    assert partition_ari(np.array([0, 0, 1, 2]), np.array([5, 5, 7, 9])) == 1.0


def test_neighbor_jaccard_intermediate_case() -> None:
    # Per node: 1/3, 1, 1/3, 1/3.
    assert neighbor_jaccard(G, H) == pytest.approx(0.5)
    # First neighbour only: G -> 1, 0, 3, 2 and H -> 1, 0, 3, 0.
    assert neighbor_jaccard(G, H, k=1) == pytest.approx(0.75)


def test_edge_overlap_intermediate_case() -> None:
    assert edge_overlap(G, H) == pytest.approx(5 / 11)


def test_disjoint_graphs_have_zero_overlap() -> None:
    assert neighbor_jaccard(RING_A, RING_B) == 0.0
    assert edge_overlap(RING_A, RING_B) == 0.0


def test_nodes_without_neighbours_in_both_graphs_are_excluded() -> None:
    a = _graph([[1], [PADDING], [0]])
    b = _graph([[1], [PADDING], [1]])
    # Node 1 has no neighbour in either graph; node 0 agrees, node 2 does not.
    assert neighbor_jaccard(a, b) == pytest.approx(0.5)
    assert edge_overlap(a, b) == pytest.approx(1 / 3)


def test_empty_graphs_give_nan() -> None:
    empty = _graph([[PADDING], [PADDING], [PADDING]])
    assert np.isnan(neighbor_jaccard(empty, empty))
    assert np.isnan(edge_overlap(empty, empty))


def test_k_occurrence_spearman_intermediate_case() -> None:
    # Ranks [2.5, 2.5, 4, 1] against [2.5, 2.5, 1, 4]: centred (0, 0, 1.5, -1.5) and
    # (0, 0, -1.5, 1.5) are exact opposites.
    assert k_occurrence_spearman(G, REVERSED) == pytest.approx(-1.0)


def test_constant_k_occurrence_gives_nan() -> None:
    assert np.isnan(k_occurrence_spearman(RING_A, G))


def test_different_ids_raise() -> None:
    other = _graph([[1, 2], [0, 2], [3, 0], [2, 1]], ids=np.arange(10, 14))
    for func in (neighbor_jaccard, edge_overlap, k_occurrence_spearman):
        with pytest.raises(ValueError, match="same ids"):
            func(G, other)


def test_partition_ari_known_value_and_shape_check() -> None:
    assert partition_ari(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])) == pytest.approx(-0.5)
    with pytest.raises(ValueError, match="equal length"):
        partition_ari(np.array([0, 1]), np.array([0, 1, 2]))
