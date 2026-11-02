"""Tests for Louvain partitions, NMI, modularity and stability, on graphs with known answers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from musicsim.evaluation.communities import (
    louvain_partitions,
    partition_modularity,
    partition_nmi,
    partition_stability,
)

CLIQUE_LABELS = np.array([0] * 4 + [1] * 4)


def _bridged_cliques(n_isolated: int = 0) -> sparse.csr_matrix:
    """Two 4-cliques (0-3, 4-7) joined by the edge 3 - 4, plus optional isolated nodes."""
    n = 8 + n_isolated
    dense = np.zeros((n, n))
    for start in (0, 4):
        dense[start : start + 4, start : start + 4] = 1.0
    dense[np.diag_indices(n)] = 0.0
    dense[3, 4] = dense[4, 3] = 1.0
    return sparse.csr_matrix(dense)


def test_cliques_recover_the_labels_and_are_stable() -> None:
    adjacency = _bridged_cliques()
    partitions = louvain_partitions(adjacency, n_seeds=5, seed=0)
    assert partitions.shape == (5, 8)
    assert partition_nmi(partitions, CLIQUE_LABELS) == (1.0, 0.0)
    assert partition_stability(partitions) == 1.0
    # 12 edges inside the cliques plus the bridge; communities are the cliques.
    expected = 2 * (6 / 13 - (13 / 26) ** 2)
    mean, std = partition_modularity(adjacency, partitions)
    assert mean == pytest.approx(expected)
    assert std == pytest.approx(0.0)


def test_ids_are_relabelled_by_first_node() -> None:
    partitions = louvain_partitions(_bridged_cliques(), n_seeds=3, seed=1)
    np.testing.assert_array_equal(partitions, np.tile([0, 0, 0, 0, 1, 1, 1, 1], (3, 1)))


def test_same_seed_same_partitions_and_seed_changes_stream() -> None:
    adjacency = _bridged_cliques()
    a = louvain_partitions(adjacency, n_seeds=4, seed=7)
    b = louvain_partitions(adjacency.copy(), n_seeds=4, seed=7)
    np.testing.assert_array_equal(a, b)


def test_asymmetric_adjacency_is_rejected() -> None:
    adjacency = sparse.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="not symmetric"):
        louvain_partitions(adjacency, n_seeds=1, seed=0)


def test_isolated_node_is_a_singleton_community() -> None:
    adjacency = _bridged_cliques(n_isolated=1)
    partitions = louvain_partitions(adjacency, n_seeds=3, seed=0)
    for row in partitions:
        assert (row == row[8]).sum() == 1
        assert len(np.unique(row)) == 3


def test_weighted_uses_the_weights() -> None:
    # A path 0-1-2-3 whose middle edge is much weaker: weighted Louvain cuts it.
    dense = np.zeros((4, 4))
    for i, j, w in [(0, 1, 10.0), (1, 2, 0.1), (2, 3, 10.0)]:
        dense[i, j] = dense[j, i] = w
    adjacency = sparse.csr_matrix(dense)
    partitions = louvain_partitions(adjacency, n_seeds=3, seed=0, weighted=True)
    np.testing.assert_array_equal(partitions, np.tile([0, 0, 1, 1], (3, 1)))
    mean, _ = partition_modularity(adjacency, partitions, weighted=True)
    # m = 20.1; each community holds internal weight 10 and has total strength 20.1.
    expected = 2 * (10 / 20.1 - (20.1 / (2 * 20.1)) ** 2)
    assert mean == pytest.approx(expected)


def test_single_seed_stability_is_nan() -> None:
    assert np.isnan(partition_stability(np.array([[0, 0, 1, 1]])))


def test_stability_of_different_partitions_is_below_one() -> None:
    partitions = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    assert partition_stability(partitions) < 1.0
