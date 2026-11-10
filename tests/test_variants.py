"""Tests for :mod:`musicsim.graphs.variants` (mutual kNN, threshold, mutual proximity)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import skew

from musicsim.config import ConfigError
from musicsim.evaluation.hubness import k_occurrence
from musicsim.graphs.adjacency import to_csr
from musicsim.graphs.base import PADDING, build_graph
from musicsim.graphs.knn import top_k_by_cosine
from musicsim.graphs.variants import _mutual_proximity_row


def _random_unit(n: int = 150, d: int = 8, seed: int = 7) -> np.ndarray:
    X = np.random.default_rng(seed).normal(size=(n, d))
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def _cfg(builder: str, **extra: object) -> dict:
    return {"builder": builder, "k": 10, "exclude_self": True, **extra}


@pytest.mark.parametrize("k", [5, 10, 20, 50])
def test_knn_is_the_canonical_graph_for_every_k(
    k: int, synthetic_embeddings: tuple[np.ndarray, np.ndarray]
) -> None:
    X, ids = synthetic_embeddings
    graph = build_graph(X, ids, _cfg("knn", k=k))
    neighbors, weights = top_k_by_cosine(X, k)
    np.testing.assert_array_equal(graph.neighbors, neighbors)
    np.testing.assert_array_equal(graph.weights, weights)


def test_variants_are_registered_by_name() -> None:
    from musicsim.registry import GRAPH_BUILDERS

    for name in ("mutual_knn", "epsilon", "mutual_proximity"):
        assert name in GRAPH_BUILDERS


def test_mutual_knn_of_an_already_symmetric_knn_is_the_knn() -> None:
    # Three orthogonal groups of three near-identical vectors: with k = 2 every
    # node's neighbours are the other two of its group, so the kNN is symmetric.
    axes = np.eye(3)
    X = np.vstack(
        [axes[g] + 0.05 * np.array([0, 1, 0.5])[None] * s for g in range(3) for s in (0, 1, 2)]
    )
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    ids = np.arange(9)
    knn = build_graph(X, ids, _cfg("knn", k=2))
    mutual = build_graph(X, ids, _cfg("mutual_knn", k=2))
    np.testing.assert_array_equal(mutual.neighbors, knn.neighbors)
    np.testing.assert_array_equal(mutual.weights, knn.weights)


def test_mutual_knn_is_the_mutual_part_of_knn_and_pads_the_rest() -> None:
    X = _random_unit()
    ids = np.arange(X.shape[0])
    knn = build_graph(X, ids, _cfg("knn"))
    mutual = build_graph(X, ids, _cfg("mutual_knn"))

    # Same edge set as symmetrising the kNN with the minimum, restricted to kNN edges.
    expected = to_csr(knn, symmetrize="mutual")
    assert (to_csr(mutual) != expected).nnz == 0
    assert (to_csr(mutual, symmetrize="union") != to_csr(mutual)).nnz == 0  # already symmetric
    # Subset of the kNN, row by row, and degree consistent with the padding.
    for i in range(X.shape[0]):
        assert set(mutual.neighbors[i][mutual.neighbors[i] != PADDING]) <= set(knn.neighbors[i])
    np.testing.assert_array_equal(mutual.degree, (mutual.neighbors != PADDING).sum(axis=1))
    assert len(mutual.to_edges_frame()) == int(mutual.degree.sum())
    assert (mutual.weights[mutual.neighbors == PADDING] == 0.0).all()
    assert (mutual.degree < 10).any()  # the random cloud does leave rows short


def test_epsilon_with_minus_infinity_is_the_complete_graph() -> None:
    X = _random_unit(n=30)
    graph = build_graph(X, np.arange(30), _cfg("epsilon", epsilon=float("-inf")))
    assert graph.k == 29
    assert (graph.degree == 29).all()
    adjacency = to_csr(graph).toarray()
    np.testing.assert_array_equal(adjacency, 1.0 - np.eye(30))


def test_epsilon_matched_has_exactly_n_times_k_directed_edges() -> None:
    X = _random_unit()
    n = X.shape[0]
    graph = build_graph(X, np.arange(n), _cfg("epsilon"))
    assert int(graph.degree.sum()) == n * 10  # continuous data: no ties at the threshold

    # It is the N*k most similar off-diagonal pairs, hence symmetric.
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    cut = np.sort(S.ravel())[::-1][n * 10 - 1]
    expected = (cut <= S).astype(float)
    np.testing.assert_array_equal(to_csr(graph).toarray(), expected)
    assert (to_csr(graph, symmetrize="union") != to_csr(graph)).nnz == 0
    # Rows are best-first and the row width is the largest degree.
    assert graph.k == int(graph.degree.max())
    assert (np.diff(graph.weights, axis=1) <= 0).all()


def test_epsilon_above_every_similarity_isolates_every_node() -> None:
    X = _random_unit(n=20)
    graph = build_graph(X, np.arange(20), _cfg("epsilon", epsilon=2.0))
    assert (graph.degree == 0).all()
    assert (graph.neighbors == PADDING).all()


def test_mutual_proximity_matches_a_hand_computed_matrix() -> None:
    S = np.array(
        [
            [0.0, 0.9, 0.5, 0.1],
            [0.9, 0.0, 0.6, 0.3],
            [0.5, 0.6, 0.0, 0.2],
            [0.1, 0.3, 0.2, 0.0],
        ]
    )
    np.fill_diagonal(S, np.inf)
    np.testing.assert_allclose(_mutual_proximity_row(S, 0)[1:], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(_mutual_proximity_row(S, 1)[[0, 2, 3]], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(_mutual_proximity_row(S, 2)[[0, 1, 3]], [0.5, 0.5, 0.0])


def test_mutual_proximity_is_symmetric_and_invariant_to_monotone_transforms() -> None:
    X = _random_unit(n=40).astype(np.float64)
    S = X @ X.T
    np.fill_diagonal(S, np.inf)
    transformed = np.exp(3.0 * S)  # strictly increasing; keeps +inf on the diagonal
    n = S.shape[0]
    mp = np.array([_mutual_proximity_row(S, i) for i in range(n)])
    mp_t = np.array([_mutual_proximity_row(transformed, i) for i in range(n)])
    off = ~np.eye(n, dtype=bool)
    np.testing.assert_array_equal(mp[off], mp_t[off])
    np.testing.assert_array_equal(mp[off], mp.T[off])


def test_mutual_proximity_graph_is_a_knn_over_mp_weights() -> None:
    X = _random_unit(n=40)
    graph = build_graph(X, np.arange(40), _cfg("mutual_proximity", k=5))
    assert graph.neighbors.shape == (40, 5)
    assert (np.diff(graph.weights, axis=1) <= 0).all()
    assert ((graph.weights >= 0) & (graph.weights <= 1)).all()
    assert not (graph.neighbors == np.arange(40)[:, None]).any()
    # Weight of each edge is the MP of that pair.
    S = X.astype(np.float64) @ X.astype(np.float64).T
    np.fill_diagonal(S, np.inf)
    row = _mutual_proximity_row(S, 3)
    np.testing.assert_allclose(graph.weights[3], row[graph.neighbors[3]], atol=1e-6)


def test_mutual_proximity_reduces_k_occurrence_skewness_with_planted_hubness() -> None:
    # High-dimensional points scattered around a common mean: the points closest
    # to the mean become hubs of the plain kNN, the textbook source of hubness.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 100)) + 1.5 * rng.normal(size=100)[None]
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = X.astype(np.float32)
    ids = np.arange(300)
    knn = build_graph(X, ids, _cfg("knn"))
    mp = build_graph(X, ids, _cfg("mutual_proximity"))
    assert skew(k_occurrence(mp)) < skew(k_occurrence(knn))


@pytest.mark.parametrize("builder", ["mutual_knn", "epsilon", "mutual_proximity"])
def test_builders_are_deterministic(builder: str) -> None:
    X = _random_unit(n=60)
    a = build_graph(X, np.arange(60), _cfg(builder))
    b = build_graph(X, np.arange(60), _cfg(builder))
    np.testing.assert_array_equal(a.neighbors, b.neighbors)
    np.testing.assert_array_equal(a.weights, b.weights)


@pytest.mark.parametrize("builder", ["mutual_knn", "epsilon", "mutual_proximity"])
def test_exclude_self_false_is_rejected(builder: str) -> None:
    with pytest.raises(ValueError, match="exclude_self is False"):
        build_graph(_random_unit(n=20), np.arange(20), _cfg(builder, exclude_self=False))


def test_bad_epsilon_and_missing_k_are_config_errors() -> None:
    X = _random_unit(n=20)
    with pytest.raises(ConfigError, match="'matched' or a number"):
        build_graph(X, np.arange(20), _cfg("epsilon", epsilon="auto"))
    with pytest.raises(ConfigError, match=r"graph.k is missing"):
        build_graph(X, np.arange(20), {"builder": "mutual_knn"})
