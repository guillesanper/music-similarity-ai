"""Hubness of a kNN graph, measured on its k-occurrence.

The *k-occurrence* ``N_k(i)`` is how many nodes list ``i`` among their ``k``
neighbours, i.e. the in-degree of ``i``. In a high-dimensional space ``N_k``
becomes right-skewed (Radovanovic et al. 2010): a few *hubs* appear in many
neighbour lists and many *antihubs* appear in none. The skewness of ``N_k`` is
the standard summary and one of the two headline structure metrics.

``N_k`` is the column sum of :func:`musicsim.graphs.adjacency.to_csr`, so it
agrees with every other structure metric on what an edge is: padding slots,
self-loops and repeated neighbours are not edges and are not counted.

A hub is a node with ``N_k > 2 * mean(N_k)``. The mean is the mean degree, so
the rule equals the usual ``N_k > 2k`` whenever every row has ``k`` neighbours
(the canonical kNN graph) and stays meaningful when degrees vary (mutual kNN,
thresholds), where ``2k`` would flag nothing: no node of a mutual kNN graph can
exceed ``k``. A threshold based on the spread of ``N_k`` (``mean + 2 * std``) is
avoided on purpose: it rises with the skew it is meant to detect, hiding the
hubs of the most biased representation and so distorting the comparison
between representations.

Skewness is the population third standardised moment (``bias=True``), the
convention of the hubness literature. It is a ratio to the standard deviation,
which is zero for a constant ``N_k`` (a directed ring, an empty graph); the
skewness is then defined as ``0.0`` rather than ``nan`` so that perfectly
regular graphs yield a plain number in a results table.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew

from musicsim.graphs.adjacency import to_csr
from musicsim.graphs.base import Graph

__all__ = ["hubness_summary", "k_occurrence"]


def k_occurrence(graph: Graph) -> np.ndarray:
    """``int64`` ``(N,)``: how many nodes list each node as a neighbour.

    The column sums of ``to_csr(graph)``: padding, self-loops and repeated
    neighbours are not edges and do not count.
    """
    return np.asarray(to_csr(graph).sum(axis=0), dtype=np.int64).ravel()


def hubness_summary(n_k: np.ndarray) -> dict[str, float]:
    """Skewness, antihub and hub fractions and maximum of the k-occurrence ``n_k``.

    ``skewness`` is the population third standardised moment of ``n_k``, and
    ``0.0`` (not ``nan``) when ``n_k`` is constant. ``antihub_fraction`` is the
    share of nodes with ``N_k == 0``, ``hub_fraction`` the share with
    ``N_k > 2 * n_k.mean()`` and ``max_k_occurrence`` the largest ``N_k``.
    """
    n_k = np.asarray(n_k)
    if n_k.ndim != 1 or n_k.size == 0:
        raise ValueError(f"n_k has shape {n_k.shape}; expected a non-empty 1-D array")
    if (n_k < 0).any():
        raise ValueError(f"n_k has minimum {n_k.min()}; expected counts >= 0")
    values = n_k.astype(np.float64)
    return {
        "skewness": float(skew(values, bias=True)) if values.std() > 0 else 0.0,
        "antihub_fraction": float((n_k == 0).mean()),
        "hub_fraction": float((values > 2 * values.mean()).mean()),
        "max_k_occurrence": float(n_k.max()),
    }
