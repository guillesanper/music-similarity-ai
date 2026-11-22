"""Tests for the robustness measures: neighbours of perturbed queries, Jaccard, self rank."""

from __future__ import annotations

import numpy as np
import pytest

from musicsim.evaluation.robustness import neighbor_set_jaccard, query_neighbors, self_rank


def _unit(rows: list[list[float]]) -> np.ndarray:
    X = np.asarray(rows, dtype=np.float64)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _cloud(n: int = 30, d: int = 6, seed: int = 0) -> np.ndarray:
    X = np.random.default_rng(seed).normal(size=(n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_unperturbed_queries_have_jaccard_one_and_rank_one() -> None:
    Z = _cloud()
    groups = np.arange(len(Z))
    nb = query_neighbors(Z, Z, groups, 5)
    assert np.array_equal(neighbor_set_jaccard(nb, nb), np.ones(len(Z)))
    assert np.array_equal(self_rank(Z, Z), np.ones(len(Z), dtype=np.int64))


def test_the_original_is_never_among_its_own_query_neighbours() -> None:
    Z = _cloud()
    nb = query_neighbors(Z, Z, np.arange(len(Z)), 10)
    assert not (nb == np.arange(len(Z))[:, None]).any()


def test_near_duplicates_of_the_original_are_excluded_too() -> None:
    Z = _cloud(12)
    Z[1] = Z[0]  # row 1 duplicates row 0 and shares its group
    groups = np.arange(12)
    groups[1] = 0
    nb = query_neighbors(Z, Z, groups, 5)
    assert 1 not in nb[0] and 0 not in nb[1]
    # Without the group, the duplicate is the best neighbour.
    assert query_neighbors(Z, Z, np.arange(12), 1)[0, 0] == 1


def test_neighbours_are_the_known_nearest_gallery_rows() -> None:
    # Unit vectors at 0, 10, 60 and 200 degrees; no two similarities tie.
    angles = np.deg2rad([0.0, 10.0, 60.0, 200.0])
    Z = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    nb = query_neighbors(Z, Z, np.arange(4), 2)
    assert nb.tolist() == [[1, 2], [0, 2], [1, 0], [2, 0]]


def test_blocks_do_not_change_the_result() -> None:
    Z = _cloud(25)
    Q = _cloud(25, seed=1)
    groups = np.arange(25) // 2
    assert np.array_equal(
        query_neighbors(Q, Z, groups, 4, block_rows=3), query_neighbors(Q, Z, groups, 4)
    )
    assert np.array_equal(self_rank(Q, Z, block_rows=4), self_rank(Q, Z))


def test_a_perturbed_query_ranks_its_original_where_the_geometry_puts_it() -> None:
    # Gallery at 0, 60 and 150 degrees. Queries 0 and 1 are each other's rows (swapped);
    # query 2 is unchanged.
    angles = np.deg2rad([0.0, 60.0, 150.0])
    Z = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    Q = Z[[1, 0, 2]]
    # Query 0 = Z[1]: Z[1] (cos 1) beats the original Z[0] (cos .5); Z[2] (cos 0) does not.
    # Query 1 = Z[0]: Z[0] (cos 1) beats the original Z[1] (cos .5); Z[2] (cos -.87) does not.
    assert self_rank(Q, Z).tolist() == [2, 2, 1]


def test_jaccard_of_known_rows_ignores_order() -> None:
    a = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    b = np.array([[3, 2, 1, 0], [0, 1, 4, 5], [4, 5, 6, 7]])
    assert neighbor_set_jaccard(a, b).tolist() == [1.0, 2 / 6, 0.0]


def test_misuse_raises() -> None:
    Z = _cloud(8)
    with pytest.raises(ValueError, match="same shape"):
        query_neighbors(Z[:4], Z, np.arange(8), 2)
    with pytest.raises(ValueError, match="groups must have shape"):
        query_neighbors(Z, Z, np.arange(5), 2)
    with pytest.raises(ValueError, match="candidate"):
        query_neighbors(Z, Z, np.zeros(8, dtype=int), 1)  # one group: nothing left
    with pytest.raises(ValueError, match="k must be"):
        query_neighbors(Z, Z, np.arange(8), 0)
    with pytest.raises(ValueError, match="same shape"):
        neighbor_set_jaccard(np.zeros((2, 3), int), np.zeros((2, 4), int))
