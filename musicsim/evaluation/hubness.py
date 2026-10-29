"""Hubness of a kNN graph, measured on its k-occurrence.

The *k-occurrence* ``N_k(i)`` is how many nodes list ``i`` among their ``k``
neighbours, i.e. the in-degree of ``i``. In a high-dimensional space ``N_k``
becomes right-skewed (Radovanovic et al. 2010): a few *hubs* appear in many
neighbour lists and many *antihubs* appear in none. The skewness of ``N_k`` is
the standard summary and one of the three headline structure metrics.

Two decisions differ from :func:`musicsim.graphs.structure.hubness_summary`:

* A hub is a node with ``N_k > 2k``, a threshold that depends only on ``k``.
  The older summary uses ``mean + 2 * std``, which moves with the distribution
  and so is not comparable between graphs with different skew. Here a hub means
  the same thing in every graph built with the same ``k``.
* The summary takes the ``N_k`` vector, not the graph, so it can be applied to
  any occurrence counts without rebuilding a :class:`~musicsim.graphs.base.Graph`.

Skewness is the population third standardised moment (``bias=True``), the
convention of the hubness literature. It is a ratio to the standard deviation,
which is zero for a constant ``N_k`` (a directed ring, an empty graph); the
skewness is then defined as ``0.0`` rather than ``nan`` so that perfectly
regular graphs yield a plain number in a results table.

Rows of a graph may be padded (see :mod:`musicsim.graphs.base`): a padding slot
is the absence of an edge and is never counted, so ``-1`` cannot be credited to
the last node.

The function is named ``k_occurrence_summary`` and not ``hubness_summary`` to
avoid two functions with the same name and different meanings.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew

from musicsim.graphs.base import PADDING, Graph

__all__ = ["k_occurrence", "k_occurrence_summary"]


def k_occurrence(graph: Graph) -> np.ndarray:
    """``int64`` ``(N,)``: how many times each node appears among the neighbours of the others.

    Padding slots are not counted.
    """
    neighbors = graph.neighbors
    return np.bincount(neighbors[neighbors != PADDING], minlength=graph.n_nodes).astype(np.int64)


def k_occurrence_summary(n_k: np.ndarray, k: int) -> dict[str, float]:
    """Skewness, antihub and hub fractions and maximum of the k-occurrence ``n_k``.

    ``skewness`` is the population third standardised moment of ``n_k``, and
    ``0.0`` (not ``nan``) when ``n_k`` is constant. ``antihub_fraction`` is the
    share of nodes with ``N_k == 0``, ``hub_fraction`` the share with
    ``N_k > 2 * k`` and ``max_k_occurrence`` the largest ``N_k``.
    """
    if k < 1:
        raise ValueError(f"k is {k}; expected an integer >= 1")
    n_k = np.asarray(n_k)
    if n_k.ndim != 1 or n_k.size == 0:
        raise ValueError(f"n_k has shape {n_k.shape}; expected a non-empty 1-D array")
    if (n_k < 0).any():
        raise ValueError(f"n_k has minimum {n_k.min()}; expected counts >= 0")
    values = n_k.astype(np.float64)
    return {
        "skewness": float(skew(values, bias=True)) if values.std() > 0 else 0.0,
        "antihub_fraction": float((n_k == 0).mean()),
        "hub_fraction": float((n_k > 2 * k).mean()),
        "max_k_occurrence": float(n_k.max()),
    }
