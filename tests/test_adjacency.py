"""Tests for the sparse adjacency view, on graphs small enough to write the
expected matrix by hand."""

from __future__ import annotations

import numpy as np
import pytest

from musicsim.graphs import to_csr
from musicsim.graphs.base import PADDING, Graph


def _graph(neighbors: list[list[int]], weights: list[list[float]]) -> Graph:
    nb = np.array(neighbors, dtype=np.int32)
    return Graph(
        neighbors=nb,
        weights=np.array(weights, dtype=np.float32),
        ids=np.arange(len(nb)),
    )


@pytest.fixture
def four_nodes() -> Graph:
    # 0 -> 1 (.9), 2 (.5) | 1 -> 0 (.8), 3 (.4) | 2 -> 3 (.7), 1 (.6) | 3 -> 2 (.3), 0 (.2)
    return _graph(
        [[1, 2], [0, 3], [3, 1], [2, 0]],
        [[0.9, 0.5], [0.8, 0.4], [0.7, 0.6], [0.3, 0.2]],
    )


def test_directed_binary(four_nodes: Graph) -> None:
    expected = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.float64,
    )
    A = to_csr(four_nodes)
    assert A.dtype == np.float64
    np.testing.assert_array_equal(A.toarray(), expected)


def test_directed_weighted(four_nodes: Graph) -> None:
    A = to_csr(four_nodes, weights="weight").toarray()
    assert A[0, 1] == np.float32(0.9)
    assert A[3, 0] == np.float32(0.2)
    assert A[1, 2] == 0.0


def test_union_takes_the_maximum(four_nodes: Graph) -> None:
    # pairs: (0,1) .9/.8 -> .9 | (0,2) .5/0 -> .5 | (0,3) 0/.2 -> .2
    #        (1,2) 0/.6 -> .6 | (1,3) .4/0 -> .4 | (2,3) .7/.3 -> .7
    f = np.float32
    expected = np.array(
        [
            [0, f(0.9), f(0.5), f(0.2)],
            [f(0.9), 0, f(0.6), f(0.4)],
            [f(0.5), f(0.6), 0, f(0.7)],
            [f(0.2), f(0.4), f(0.7), 0],
        ],
        dtype=np.float64,
    )
    A = to_csr(four_nodes, symmetrize="union", weights="weight")
    np.testing.assert_array_equal(A.toarray(), expected)
    assert (A != A.T).nnz == 0


def test_mutual_takes_the_minimum(four_nodes: Graph) -> None:
    # only the pairs that exist both ways survive: (0,1) .9/.8 and (2,3) .7/.3
    f = np.float32
    expected = np.zeros((4, 4))
    expected[0, 1] = expected[1, 0] = f(0.8)
    expected[2, 3] = expected[3, 2] = f(0.3)
    A = to_csr(four_nodes, symmetrize="mutual", weights="weight")
    np.testing.assert_array_equal(A.toarray(), expected)
    assert A.nnz == 4


def test_binary_union_and_mutual(four_nodes: Graph) -> None:
    union = to_csr(four_nodes, symmetrize="union").toarray()
    mutual = to_csr(four_nodes, symmetrize="mutual").toarray()
    assert union.sum() == 12  # six unordered pairs, both orientations
    assert mutual.sum() == 4
    assert set(np.unique(union)) == {0.0, 1.0}


def test_callable_maximum_equals_union(four_nodes: Graph) -> None:
    by_name = to_csr(four_nodes, symmetrize="union", weights="weight")
    by_func = to_csr(four_nodes, symmetrize=np.maximum, weights="weight")
    np.testing.assert_array_equal(by_func.toarray(), by_name.toarray())


def test_callable_receives_zero_where_a_direction_is_missing(four_nodes: Graph) -> None:
    # probabilistic sum a + b - a*b over the union of supports: (0,2) has only .5
    A = to_csr(four_nodes, symmetrize=lambda a, b: a + b - a * b, weights="weight")
    assert A[0, 2] == A[2, 0] == np.float64(np.float32(0.5))
    w01, w10 = np.float64(np.float32(0.9)), np.float64(np.float32(0.8))
    assert A[0, 1] == pytest.approx(w01 + w10 - w01 * w10)


def test_callable_result_with_wrong_shape_is_rejected(four_nodes: Graph) -> None:
    with pytest.raises(ValueError, match=r"shape \(\)"):
        to_csr(four_nodes, symmetrize=lambda a, b: 1.0)


def test_callable_negative_result_is_rejected(four_nodes: Graph) -> None:
    with pytest.raises(ValueError, match=r"-1.0"):
        to_csr(four_nodes, symmetrize=lambda a, b: -np.ones_like(a))


def test_callable_zero_result_stores_no_edge(four_nodes: Graph) -> None:
    A = to_csr(four_nodes, symmetrize=lambda a, b: np.zeros_like(a))
    assert A.nnz == 0
    assert A.shape == (4, 4)


@pytest.mark.parametrize("bad", [0.0, -0.1, np.nan])
def test_non_positive_weight_is_rejected(bad: float) -> None:
    g = _graph([[1], [0]], [[0.5], [bad]])
    with pytest.raises(ValueError, match="positive finite"):
        to_csr(g, weights="weight")


def test_non_positive_weight_is_fine_when_binary() -> None:
    g = _graph([[1], [0]], [[0.5], [0.0]])
    assert to_csr(g, weights="binary").nnz == 2


def test_unknown_options_are_rejected(four_nodes: Graph) -> None:
    with pytest.raises(ValueError, match="'sum'"):
        to_csr(four_nodes, symmetrize="sum")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'count'"):
        to_csr(four_nodes, weights="count")  # type: ignore[arg-type]


def test_padding_is_not_an_edge() -> None:
    # node 3 has no neighbours and node 2 has one; nothing may point at N-1 = 3
    g = _graph(
        [[1, 2], [0, PADDING], [0, PADDING], [PADDING, PADDING]],
        [[0.9, 0.5], [0.8, 0.0], [0.7, 0.0], [0.0, 0.0]],
    )
    for sym in (None, "union", "mutual"):
        A = to_csr(g, symmetrize=sym, weights="weight").toarray()
        assert A[:, 3].sum() == 0.0
        assert A[3, :].sum() == 0.0
    directed = to_csr(g).toarray()
    assert directed[1].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert to_csr(g, symmetrize="mutual").nnz == 4  # 0 <-> 1 and 0 <-> 2


def test_self_loops_are_dropped() -> None:
    g = _graph([[0, 1], [1, 0]], [[0.9, 0.5], [0.8, 0.7]])
    for sym in (None, "union"):
        A = to_csr(g, symmetrize=sym, weights="weight")
        assert A.diagonal().sum() == 0.0
        assert A.nnz == 2  # 0 -> 1 and 1 -> 0 survive
    assert to_csr(g, weights="weight")[0, 1] == np.float32(0.5)


def test_repeated_neighbour_keeps_the_larger_weight() -> None:
    g = _graph([[1, 1], [0, PADDING]], [[0.3, 0.6], [0.2, 0.0]])
    A = to_csr(g, weights="weight")
    assert A[0, 1] == np.float32(0.6)
    assert A.nnz == 2


def test_graph_without_edges() -> None:
    g = _graph([[PADDING], [PADDING]], [[0.0], [0.0]])
    for sym in (None, "union", "mutual", np.maximum):
        A = to_csr(g, symmetrize=sym)
        assert A.shape == (2, 2)
        assert A.nnz == 0
