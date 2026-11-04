"""Agreement between two graphs (or partitions) over the same nodes.

Four views of "how much do these two graphs differ": the overlap of every
node's neighbour set (:func:`neighbor_jaccard`, the headline), the overlap of
the whole directed edge set (:func:`edge_overlap`), the rank agreement of the
k-occurrences (:func:`k_occurrence_spearman`, i.e. whether the same nodes are
hubs) and the agreement of two community partitions (:func:`partition_ari`).

Decisions worth knowing:

* Both graphs must list exactly the same ``ids`` in the same order; otherwise
  row ``i`` of one graph is not row ``i`` of the other and every number would be
  meaningless, so it is an error rather than an alignment attempted here.
* Neighbour and edge sets are taken from :func:`musicsim.graphs.adjacency.to_csr`
  (binary, directed), the same definition of "edge" as the structure metrics:
  padding, self-loops and repeated neighbours are not edges.
* Jaccard is ``nan`` when it is undefined (no node has a neighbour in either
  graph, no edge in either graph) and so is the Spearman correlation of a
  constant k-occurrence, instead of an arbitrary ``0`` or ``1``.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from musicsim.evaluation.hubness import k_occurrence
from musicsim.graphs.adjacency import to_csr
from musicsim.graphs.base import Graph

__all__ = ["edge_overlap", "k_occurrence_spearman", "neighbor_jaccard", "partition_ari"]


def _check_aligned(a: Graph, b: Graph) -> None:
    if a.ids.shape != b.ids.shape or not np.array_equal(a.ids, b.ids):
        raise ValueError(
            f"graphs have different ids ({a.ids.shape[0]} vs {b.ids.shape[0]} nodes); "
            "expected the same ids in the same order"
        )


def _jaccard_rows(a: sparse.csr_matrix, b: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``(|A & B|, |A | B|)`` of two binary matrices."""
    inter = np.asarray(a.multiply(b).sum(axis=1), dtype=np.float64).ravel()
    size_a = np.asarray(a.sum(axis=1), dtype=np.float64).ravel()
    size_b = np.asarray(b.sum(axis=1), dtype=np.float64).ravel()
    return inter, size_a + size_b - inter


def neighbor_jaccard(a: Graph, b: Graph, *, k: int = 10) -> float:
    """Mean over nodes of ``|A & B| / |A | B|`` for the first ``k`` neighbours of each graph.

    Nodes with no neighbour in either graph are left out of the mean; if that
    leaves no node, the result is ``nan``.
    """
    _check_aligned(a, b)
    if k < 1:
        raise ValueError(f"k is {k}; expected a positive integer")

    def first_k(graph: Graph) -> sparse.csr_matrix:
        # Padding is a suffix, so a column slice keeps the first k real neighbours.
        cut = Graph(graph.neighbors[:, :k], graph.weights[:, :k], graph.ids)
        return to_csr(cut)

    inter, union = _jaccard_rows(first_k(a), first_k(b))
    defined = union > 0
    if not defined.any():
        return float("nan")
    return float((inter[defined] / union[defined]).mean())


def edge_overlap(a: Graph, b: Graph) -> float:
    """``|E_a & E_b| / |E_a | E_b|`` over directed edges; ``nan`` if neither has an edge."""
    _check_aligned(a, b)
    inter, union = _jaccard_rows(to_csr(a), to_csr(b))
    if union.sum() == 0:
        return float("nan")
    return float(inter.sum() / union.sum())


def k_occurrence_spearman(a: Graph, b: Graph) -> float:
    """Spearman correlation of the two k-occurrences; ``nan`` if either is constant."""
    _check_aligned(a, b)
    n_a, n_b = k_occurrence(a), k_occurrence(b)
    if n_a.min() == n_a.max() or n_b.min() == n_b.max():
        return float("nan")
    return float(spearmanr(n_a, n_b).statistic)


def partition_ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index between two partitions given as one community id per node."""
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError(
            f"partitions have shapes {a.shape} and {b.shape}; "
            "expected two 1-D arrays of equal length"
        )
    return float(adjusted_rand_score(a, b))
