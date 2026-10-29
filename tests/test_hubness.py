"""Tests for the k-occurrence and its summary, on graphs small enough to count by hand."""

from __future__ import annotations

import numpy as np
import pytest

from musicsim.evaluation.hubness import k_occurrence, k_occurrence_summary
from musicsim.graphs.base import PADDING, Graph
from musicsim.graphs.structure import in_degree


def _graph(neighbors: list[list[int]]) -> Graph:
    nb = np.array(neighbors, dtype=np.int32)
    return Graph(
        neighbors=nb,
        weights=np.where(nb == PADDING, 0.0, 1.0).astype(np.float32),
        ids=np.arange(len(nb)),
    )


def test_directed_ring_is_regular() -> None:
    graph = _graph([[1], [2], [3], [4], [0]])
    n_k = k_occurrence(graph)
    assert n_k.dtype == np.int64
    np.testing.assert_array_equal(n_k, [1, 1, 1, 1, 1])
    assert k_occurrence_summary(n_k, graph.k) == {
        "skewness": 0.0,
        "antihub_fraction": 0.0,
        "hub_fraction": 0.0,
        "max_k_occurrence": 1.0,
    }


def test_star_k1() -> None:
    # Centre 0; leaves 1-4 point to it, and it points to leaf 1.
    graph = _graph([[1], [0], [0], [0], [0]])
    n_k = k_occurrence(graph)
    np.testing.assert_array_equal(n_k, [4, 1, 0, 0, 0])
    summary = k_occurrence_summary(n_k, graph.k)
    # Deviations from the mean 1 are [3, 0, -1, -1, -1]: m2 = 12/5, m3 = 24/5.
    assert summary["skewness"] == pytest.approx((24 / 5) / (12 / 5) ** 1.5)
    assert summary["antihub_fraction"] == 3 / 5
    assert summary["hub_fraction"] == 1 / 5  # only the centre has N_k = 4 > 2
    assert summary["max_k_occurrence"] == 4.0


def test_padding_is_not_counted() -> None:
    # Edges 0->1, 1->2, 3->0, 3->1; node 2 has no neighbour, and nobody points to node 3.
    graph = _graph([[1, PADDING], [2, PADDING], [PADDING, PADDING], [0, 1]])
    np.testing.assert_array_equal(k_occurrence(graph), [1, 2, 1, 0])


def test_all_padding_graph_has_zero_skewness() -> None:
    graph = _graph([[PADDING, PADDING]] * 3)
    n_k = k_occurrence(graph)
    np.testing.assert_array_equal(n_k, [0, 0, 0])
    summary = k_occurrence_summary(n_k, graph.k)
    assert summary["skewness"] == 0.0
    assert summary["antihub_fraction"] == 1.0
    assert summary["max_k_occurrence"] == 0.0


def test_hub_threshold_is_strictly_above_twice_k() -> None:
    summary = k_occurrence_summary(np.array([0, 3, 4, 5]), k=2)
    # With 2k = 4, only N_k = 5 is a hub; N_k = 4 sits exactly on the threshold.
    assert summary["hub_fraction"] == 1 / 4


def test_matches_in_degree_on_unpadded_knn_graph(rng: np.random.Generator) -> None:
    n, k = 30, 5
    neighbors = np.stack([rng.permutation(np.delete(np.arange(n), i))[:k] for i in range(n)])
    graph = _graph(neighbors.tolist())
    np.testing.assert_array_equal(k_occurrence(graph), in_degree(graph))
    assert k_occurrence(graph).sum() == n * k


@pytest.mark.parametrize("k", [0, -1])
def test_rejects_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="expected an integer >= 1"):
        k_occurrence_summary(np.array([1, 2]), k)


def test_rejects_invalid_n_k() -> None:
    with pytest.raises(ValueError, match="non-empty 1-D"):
        k_occurrence_summary(np.zeros((2, 2)), 1)
    with pytest.raises(ValueError, match="counts >= 0"):
        k_occurrence_summary(np.array([1, -1]), 1)
