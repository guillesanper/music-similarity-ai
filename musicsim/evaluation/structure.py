"""Structural statistics of a kNN graph: label homophily, reciprocity, clustering
and connectivity.

These are pure functions; choosing galleries and writing CSVs is the job of
:mod:`musicsim.evaluation.graph_evaluator`, and hubness lives in
:mod:`musicsim.evaluation.hubness`.

Every function derives its edges from :func:`musicsim.graphs.adjacency.to_csr`,
the single place that decides what an edge is: padding slots, self-loops and
repeated neighbours are not edges. Homophily and reciprocity use the directed
view; clustering and components use the undirected union (``i - j`` if either
direction exists), which is what :func:`to_csr` returns with
``symmetrize="union"``.

Label homophily is the share of directed edges whose endpoints share a label.
On a kNN graph built without removing near-duplicates, and with the same
relevance labels, it is exactly the mean precision at ``k`` of the retrieval
evaluator. It is reported here as a structure metric, not as a second retrieval
number, and the report must say so.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from musicsim.graphs.adjacency import to_csr
from musicsim.graphs.base import Graph

__all__ = ["component_summary", "label_homophily", "mean_clustering", "reciprocity"]


def label_homophily(graph: Graph, labels: np.ndarray) -> float:
    """Fraction of directed edges whose two endpoints share a label.

    ``labels`` holds one integer code per node. Raises ``ValueError`` if the
    graph has no edge: ``0.0`` there would be indistinguishable from a graph
    with no homophily at all.
    """
    labels = np.asarray(labels)
    if labels.shape != (graph.n_nodes,):
        raise ValueError(
            f"labels {labels.shape} does not have one entry per node ({graph.n_nodes})"
        )
    edges = to_csr(graph).tocoo()
    if edges.nnz == 0:
        raise ValueError(f"graph with {graph.n_nodes} nodes has no edges; homophily is undefined")
    return float((labels[edges.row] == labels[edges.col]).mean())


def reciprocity(graph: Graph) -> float:
    """Fraction of directed edges ``i -> j`` whose reverse ``j -> i`` also exists.

    Raises ``ValueError`` if the graph has no edge.
    """
    A = to_csr(graph)
    if A.nnz == 0:
        raise ValueError(f"graph with {graph.n_nodes} nodes has no edges; reciprocity is undefined")
    return float(A.multiply(A.T).nnz / A.nnz)


def mean_clustering(adjacency: sparse.csr_matrix) -> float:
    """Mean local clustering coefficient of a symmetric binary adjacency matrix.

    The local coefficient of node ``i`` is ``2 * t_i / (d_i * (d_i - 1))``, with
    ``d_i`` its degree and ``t_i`` the triangles through it; it is ``0`` when
    ``d_i < 2``. The mean runs over *all* nodes, the convention of
    ``networkx.average_clustering``. Triangles come from sparse products, never
    from a dense matrix.
    """
    n = adjacency.shape[0]
    if adjacency.shape != (n, n):
        raise ValueError(f"adjacency has shape {adjacency.shape}; expected a square matrix")
    A = adjacency.copy()
    A.data[:] = 1.0
    degree = np.diff(A.indptr).astype(np.float64)
    # (A @ A)[i, j] counts the paths i - m - j; masked by A it keeps those closed by an
    # edge i - j, so the row sum is twice the number of triangles through i.
    closed_paths = np.asarray(A.multiply(A @ A).sum(axis=1)).ravel()
    pairs = degree * (degree - 1)
    local = np.divide(closed_paths, pairs, out=np.zeros(n), where=pairs > 0)
    return float(local.mean())


def component_summary(adjacency: sparse.csr_matrix) -> dict[str, float]:
    """``n_components`` and ``giant_fraction`` (share of nodes in the largest component).

    Components are weak: edge direction is ignored. A node without edges is its
    own component.
    """
    n_components, labels = csgraph.connected_components(
        adjacency, directed=True, connection="weak"
    )
    sizes = np.bincount(labels)
    return {
        "n_components": float(n_components),
        "giant_fraction": float(sizes.max() / adjacency.shape[0]),
    }
