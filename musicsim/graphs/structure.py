"""Structural statistics of a kNN graph: hubness, reciprocity, connectivity,
label homophily.

Everything here takes a :class:`~musicsim.graphs.base.Graph` (plus, for
homophily, one integer label code per node) and returns plain numbers or
arrays, so the evaluator in :mod:`musicsim.evaluation.structure` only has to
choose galleries and write CSVs.

Hubness is measured on the *k-occurrence* ``N_k(i)``: how many nodes list
``i`` among their ``k`` neighbours, i.e. its in-degree. In a high-dimensional
space ``N_k`` becomes right-skewed (Radovanovic et al. 2010): a few *hubs*
appear in many neighbour lists and many *antihubs* appear in none, which
distorts retrieval and any graph built on top of it. Its skewness is the
standard summary; the Gini coefficient and the antihub fraction complement it.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.stats import skew

from musicsim.graphs.base import Graph

__all__ = [
    "adjacency",
    "connected_components",
    "gini",
    "hubness_summary",
    "in_degree",
    "label_homophily",
    "reciprocity",
    "structure_summary",
]


def adjacency(graph: Graph) -> sparse.csr_matrix:
    """Binary directed adjacency matrix: ``A[i, j] = 1`` iff ``j`` is a neighbour of ``i``."""
    n, k = graph.neighbors.shape
    rows = np.repeat(np.arange(n), k)
    data = np.ones(n * k, dtype=np.int8)
    return sparse.csr_matrix((data, (rows, graph.neighbors.ravel())), shape=(n, n))


def in_degree(graph: Graph) -> np.ndarray:
    """The k-occurrence ``N_k`` of every node, ``(N,)``."""
    return np.bincount(graph.neighbors.ravel(), minlength=graph.n_nodes)


def gini(values: np.ndarray) -> float:
    """Gini coefficient of non-negative ``values`` (0 = equal, towards 1 = concentrated)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    total = v.sum()
    if v.size == 0 or total == 0:
        return 0.0
    ranks = np.arange(1, v.size + 1)
    return float((2 * (ranks * v).sum() - (v.size + 1) * total) / (v.size * total))


def hubness_summary(graph: Graph, *, hub_factor: float = 2.0) -> dict[str, float]:
    """Skewness, Gini, antihub and hub fractions of the in-degree distribution.

    A *hub* has ``N_k > mean + hub_factor * std``; since the mean of ``N_k`` is
    exactly ``k``, that is ``k + hub_factor * std``. An *antihub* has ``N_k = 0``.
    """
    nk = in_degree(graph).astype(np.float64)
    std = nk.std()
    return {
        "hubness_skewness": float(skew(nk)) if std > 0 else 0.0,
        "in_degree_gini": gini(nk),
        "in_degree_max": float(nk.max()),
        "in_degree_std": float(std),
        "antihub_fraction": float((nk == 0).mean()),
        "hub_fraction": float((nk > nk.mean() + hub_factor * std).mean()) if std > 0 else 0.0,
    }


def reciprocity(graph: Graph) -> float:
    """Fraction of directed edges ``i -> j`` whose reverse ``j -> i`` also exists."""
    A = adjacency(graph)
    A.data[:] = 1  # duplicates summed by the constructor must not count twice
    mutual = A.multiply(A.T)
    return float(mutual.nnz / A.nnz) if A.nnz else 0.0


def connected_components(graph: Graph) -> tuple[int, float]:
    """``(number of weakly connected components, share of nodes in the largest)``."""
    n_components, labels = csgraph.connected_components(
        adjacency(graph), directed=True, connection="weak"
    )
    sizes = np.bincount(labels)
    return int(n_components), float(sizes.max() / graph.n_nodes)


def label_homophily(graph: Graph, labels: np.ndarray) -> float:
    """Fraction of edges whose two endpoints share a label (``labels``: ``(N,)`` codes)."""
    labels = np.asarray(labels)
    if labels.shape != (graph.n_nodes,):
        raise ValueError(
            f"labels {labels.shape} does not have one entry per node ({graph.n_nodes})"
        )
    return float((labels[graph.neighbors] == labels[:, None]).mean())


def structure_summary(graph: Graph, labels: np.ndarray | None = None) -> dict[str, float]:
    """Every structural statistic of ``graph`` as one flat ``{metric: value}`` mapping."""
    n_components, largest = connected_components(graph)
    out: dict[str, float] = {
        "n_nodes": float(graph.n_nodes),
        "k": float(graph.k),
        "mean_edge_weight": float(graph.weights.mean()),
        **hubness_summary(graph),
        "reciprocity": reciprocity(graph),
        "n_components": float(n_components),
        "largest_component_share": largest,
    }
    if labels is not None:
        out["label_homophily"] = label_homophily(graph, labels)
    return out
