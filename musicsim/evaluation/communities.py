"""Louvain communities of a symmetric graph and how they relate to the genre labels.

Louvain is stochastic (node visiting order), so a single partition is one draw.
:func:`louvain_partitions` runs several seeds and the other functions summarise
across them: NMI against the genre labels (a headline structure metric),
modularity, and the stability of the partition itself (mean pairwise ARI).

Decisions worth knowing:

* The seed of run ``i`` is ``derive_seed(seed, "louvain", i)`` and nothing else
  enters it, in particular not the label of the representation. Two identical
  graphs therefore receive identical partitions, and a difference between two
  representations is never an artefact of different random draws.
* Crisp graphs use ``weighted=False``: every edge counts 1. The adjacency of a
  kNN graph is a union of directed neighbour lists whose cosine weights can be
  non-positive, and Louvain requires positive weights; ignoring them also makes
  the crisp/fuzzy comparison one of unweighted against weighted. Fuzzy graphs
  pass ``weighted=True`` and their memberships are the weights.
* The adjacency must be symmetric: Louvain is defined on undirected graphs and
  silently symmetrising here would hide which rule (union, mutual, t-conorm) was
  meant. Build it with :func:`musicsim.graphs.adjacency.to_csr`.
* Isolated nodes are their own communities. Community ids are relabelled
  ``0..c-1`` in order of the first node that carries them, so partitions are
  comparable and reproducible whatever order networkx returns.
* NMI uses the arithmetic mean normalisation (the scikit-learn default).
  Spreads over seeds are population standard deviations. With one seed there is
  no pair of partitions to compare, so the stability is ``nan``.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
from networkx.algorithms import community as nx_community
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from musicsim.seeding import derive_seed

__all__ = [
    "louvain_partitions",
    "partition_modularity",
    "partition_nmi",
    "partition_stability",
]


def louvain_partitions(
    adjacency: sparse.csr_matrix,
    *,
    n_seeds: int = 10,
    seed: int,
    resolution: float = 1.0,
    weighted: bool = False,
) -> np.ndarray:
    """``int64`` ``(n_seeds, N)``: one community id per node and seed.

    Ids are relabelled ``0..c-1`` by first node. ``weighted=False`` treats every
    stored entry as an edge of weight 1; ``True`` uses the stored values, which
    must be positive.
    """
    if n_seeds < 1:
        raise ValueError(f"n_seeds is {n_seeds}; expected at least 1")
    if resolution <= 0:
        raise ValueError(f"resolution is {resolution}; expected a positive value")
    graph = _to_networkx(adjacency, weighted=weighted)
    n = adjacency.shape[0]

    partitions = np.empty((n_seeds, n), dtype=np.int64)
    for i in range(n_seeds):
        communities = nx_community.louvain_communities(
            graph,
            weight="weight" if weighted else None,
            resolution=resolution,
            seed=derive_seed(seed, "louvain", i),
        )
        raw = np.empty(n, dtype=np.int64)
        for community_id, members in enumerate(communities):
            raw[list(members)] = community_id
        partitions[i] = _relabel(raw)
    return partitions


def partition_nmi(partitions: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation over seeds of the NMI between each partition and ``labels``."""
    partitions = _check_partitions(partitions)
    labels = np.asarray(labels)
    if labels.shape != (partitions.shape[1],):
        raise ValueError(f"labels has shape {labels.shape}; expected ({partitions.shape[1]},)")
    values = np.array([normalized_mutual_info_score(labels, row) for row in partitions])
    return float(values.mean()), float(values.std())


def partition_modularity(
    adjacency: sparse.csr_matrix, partitions: np.ndarray, *, weighted: bool = False
) -> tuple[float, float]:
    """Mean and standard deviation over seeds of the modularity of each partition."""
    partitions = _check_partitions(partitions)
    if partitions.shape[1] != adjacency.shape[0]:
        raise ValueError(
            f"partitions cover {partitions.shape[1]} nodes; the adjacency has {adjacency.shape[0]}"
        )
    graph = _to_networkx(adjacency, weighted=weighted)
    if graph.number_of_edges() == 0:
        raise ValueError("the graph has no edges; modularity is undefined")
    values = []
    for row in partitions:
        groups = [set(np.flatnonzero(row == c).tolist()) for c in np.unique(row)]
        values.append(
            nx_community.modularity(graph, groups, weight="weight" if weighted else None)
        )
    return float(np.mean(values)), float(np.std(values))


def partition_stability(partitions: np.ndarray) -> float:
    """Mean ARI over all pairs of seeds; ``nan`` with a single seed."""
    partitions = _check_partitions(partitions)
    if partitions.shape[0] < 2:
        return float("nan")
    pairs = itertools.combinations(range(partitions.shape[0]), 2)
    return float(np.mean([adjusted_rand_score(partitions[a], partitions[b]) for a, b in pairs]))


def _to_networkx(adjacency: sparse.csr_matrix, *, weighted: bool) -> nx.Graph:
    """Undirected networkx graph with nodes ``0..N-1``; checks squareness and symmetry."""
    n_rows, n_cols = adjacency.shape
    if n_rows != n_cols:
        raise ValueError(f"adjacency has shape {adjacency.shape}; expected a square matrix")
    if (adjacency != adjacency.T).nnz:
        raise ValueError(
            "adjacency is not symmetric; symmetrise it first "
            "(e.g. to_csr(..., symmetrize='union'))"
        )
    if weighted:
        if adjacency.nnz and not (adjacency.data > 0).all():
            raise ValueError(
                f"adjacency has a weight {adjacency.data.min()}; "
                "weighted=True expects positive weights"
            )
        source = adjacency
    else:
        source = adjacency.copy()
        source.data = np.ones_like(source.data, dtype=np.float64)
    return nx.from_scipy_sparse_array(source)


def _relabel(raw: np.ndarray) -> np.ndarray:
    """Renumber community ids ``0..c-1`` in order of first appearance along the nodes."""
    _, first, inverse = np.unique(raw, return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first))
    return order[inverse.ravel()].astype(np.int64)


def _check_partitions(partitions: np.ndarray) -> np.ndarray:
    partitions = np.asarray(partitions)
    if partitions.ndim != 2 or partitions.size == 0:
        raise ValueError(
            f"partitions has shape {partitions.shape}; expected a non-empty (n_seeds, N) array"
        )
    return partitions
