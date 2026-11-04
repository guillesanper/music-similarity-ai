"""Tests for the k-occurrence and its summary, on graphs small enough to count by hand."""

from __future__ import annotations

import numpy as np
import pytest

from musicsim.evaluation.hubness import hubness_summary, k_occurrence
from musicsim.graphs.base import PADDING, Graph


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
    assert hubness_summary(n_k) == {
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
    summary = hubness_summary(n_k)
    # Deviations from the mean 1 are [3, 0, -1, -1, -1]: m2 = 12/5, m3 = 24/5.
    assert summary["skewness"] == pytest.approx((24 / 5) / (12 / 5) ** 1.5)
    assert summary["antihub_fraction"] == 3 / 5
    assert summary["hub_fraction"] == 1 / 5  # only the centre has N_k = 4 > 2 * 1
    assert summary["max_k_occurrence"] == 4.0


def test_padding_is_not_counted() -> None:
    # Edges 0->1, 1->2, 3->0, 3->1; node 2 has no neighbour, and nobody points to node 3.
    graph = _graph([[1, PADDING], [2, PADDING], [PADDING, PADDING], [0, 1]])
    np.testing.assert_array_equal(k_occurrence(graph), [1, 2, 1, 0])


def test_self_loops_and_repeated_neighbours_are_not_edges() -> None:
    # Row 0 lists itself and node 1 twice; to_csr keeps a single edge 0 -> 1.
    graph = _graph([[0, 1, 1], [2, PADDING, PADDING], [PADDING, PADDING, PADDING]])
    np.testing.assert_array_equal(k_occurrence(graph), [0, 1, 1])


def test_all_padding_graph_has_zero_skewness() -> None:
    n_k = k_occurrence(_graph([[PADDING, PADDING]] * 3))
    np.testing.assert_array_equal(n_k, [0, 0, 0])
    summary = hubness_summary(n_k)
    assert summary["skewness"] == 0.0
    assert summary["antihub_fraction"] == 1.0
    assert summary["hub_fraction"] == 0.0
    assert summary["max_k_occurrence"] == 0.0


def test_symmetric_star_with_padding_has_one_hub() -> None:
    # Centre 0 lists six leaves; each leaf lists only the centre, padded to width 6.
    rows = [[1, 2, 3, 4, 5, 6]] + [[0] + [PADDING] * 5 for _ in range(6)]
    n_k = k_occurrence(_graph(rows))
    np.testing.assert_array_equal(n_k, [6, 1, 1, 1, 1, 1, 1])
    # 2 * mean = 24/7 < 6: the centre is a hub. The nominal 2k = 12 would flag nothing.
    assert hubness_summary(n_k)["hub_fraction"] == 1 / 7


def test_hub_rule_matches_twice_k_on_fixed_degree_graph(rng: np.random.Generator) -> None:
    n, k = 40, 5
    neighbors = np.stack([rng.permutation(np.delete(np.arange(n), i))[:k] for i in range(n)])
    graph = _graph(neighbors.tolist())
    n_k = k_occurrence(graph)
    assert n_k.sum() == n * k
    assert hubness_summary(n_k)["hub_fraction"] == (n_k > 2 * k).mean()


def test_hub_threshold_is_strictly_above_twice_the_mean() -> None:
    # Mean 3, threshold 6: N_k = 6 sits exactly on it and is not a hub.
    assert hubness_summary(np.array([0, 0, 6, 6]))["hub_fraction"] == 0.0
    assert hubness_summary(np.array([0, 0, 5, 7]))["hub_fraction"] == 1 / 4


def test_rejects_invalid_n_k() -> None:
    with pytest.raises(ValueError, match="non-empty 1-D"):
        hubness_summary(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="non-empty 1-D"):
        hubness_summary(np.array([]))
    with pytest.raises(ValueError, match="counts >= 0"):
        hubness_summary(np.array([1, -1]))
