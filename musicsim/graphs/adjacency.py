"""Sparse adjacency view of a :class:`~musicsim.graphs.base.Graph`.

Structure metrics and community detection want an ``(N, N)`` matrix, while a
graph stores ``(N, k)`` neighbour lists. :func:`to_csr` is the single place
where that conversion happens, directed or symmetrised, so every consumer
agrees on what an edge is.

Decisions worth knowing:

* A sparse matrix cannot tell "an edge of weight 0" from "no edge", and
  community detection needs positive weights anyway. With ``weights="weight"``
  a non-positive (or non-finite) weight on a real edge is therefore an error,
  not something to drop quietly.
* Padding slots are the absence of an edge (see :mod:`musicsim.graphs.base`)
  and are ignored. Self-loops ``i -> i`` are dropped silently: the matrix has
  no diagonal, a self-loop carries no structural information, and a graph built
  with ``exclude_self=False`` is legitimate. :func:`musicsim.graphs.structure.adjacency`
  keeps them, so the two views differ on such graphs.
* If a row lists the same neighbour twice, the larger weight wins. Summing
  would double a similarity, which has no meaning.
* A symmetrising rule is applied to ``(w_ij, w_ji)`` over the union of both
  supports, with ``0`` where a direction is missing. ``"union"`` is the
  maximum and ``"mutual"`` the minimum (which is ``0``, hence no edge, unless
  both directions exist); a callable generalises this to the t-conorms of the
  fuzzy block. The callable is called once, on 1-D arrays, so it must be
  elementwise and should be commutative: the result is stored at ``(i, j)`` and
  ``(j, i)`` alike. Results that are ``0`` are not stored; negative or
  non-finite ones are rejected for the same reason as non-positive weights.
* Memory is proportional to the number of edges; no dense ``N x N`` array is
  ever built.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy import sparse

from musicsim.graphs.base import PADDING, Graph

__all__ = ["to_csr"]

_WEIGHTS = ("binary", "weight")
_RULES = ("union", "mutual")


def to_csr(
    graph: Graph,
    *,
    symmetrize: Literal["union", "mutual"]
    | Callable[[np.ndarray, np.ndarray], np.ndarray]
    | None = None,
    weights: Literal["binary", "weight"] = "binary",
) -> sparse.csr_matrix:
    """``(N, N)`` ``float64`` adjacency matrix of ``graph``, without diagonal.

    ``weights="binary"`` stores ``1.0`` per edge, ``"weight"`` the graph weight
    (which must be positive and finite on every real edge). ``symmetrize=None``
    keeps the directed ``i -> j`` orientation; ``"union"`` and ``"mutual"`` give
    a symmetric matrix with the maximum, respectively minimum, of the two
    directions; a callable ``f(w_ij, w_ji)`` does the same with its own rule.
    """
    if weights not in _WEIGHTS:
        raise ValueError(f"weights is {weights!r}; expected one of {_WEIGHTS}")
    if not (symmetrize is None or callable(symmetrize) or symmetrize in _RULES):
        raise ValueError(
            f"symmetrize is {symmetrize!r}; expected None, one of {_RULES} or a callable"
        )

    n, k = graph.neighbors.shape
    real = (graph.neighbors != PADDING).ravel()
    rows = np.repeat(np.arange(n, dtype=np.int64), k)[real]
    cols = graph.neighbors.ravel()[real].astype(np.int64)
    if weights == "binary":
        data = np.ones(rows.size, dtype=np.float64)
    else:
        data = graph.weights.ravel()[real].astype(np.float64)
        if not (np.isfinite(data) & (data > 0)).all():
            bad = float(data[~(np.isfinite(data) & (data > 0))][0])
            raise ValueError(
                f"graph has an edge weight {bad}; weights='weight' expects positive finite weights"
            )

    off_diagonal = rows != cols
    rows, cols, data = rows[off_diagonal], cols[off_diagonal], data[off_diagonal]
    rows, cols, data = _dedup_max(rows, cols, data, n)

    if symmetrize is not None:
        rows, cols, data = _symmetrize(rows, cols, data, n, symmetrize)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def _dedup_max(
    rows: np.ndarray, cols: np.ndarray, data: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One entry per ``(row, col)``, keeping the largest weight."""
    key = rows * n + cols
    order = np.lexsort((data, key))
    key, data = key[order], data[order]
    keep = np.ones(key.size, dtype=bool)
    keep[:-1] = key[1:] != key[:-1]  # sorted by weight within a key: the last is the maximum
    key, data = key[keep], data[keep]
    return key // n, key % n, data


def _symmetrize(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    n: int,
    rule: str | Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply ``rule`` to ``(w_ij, w_ji)`` for every unordered pair with an edge."""
    func = {"union": np.maximum, "mutual": np.minimum}[rule] if isinstance(rule, str) else rule
    lo, hi = np.minimum(rows, cols), np.maximum(rows, cols)
    pairs, inverse = np.unique(lo * n + hi, return_inverse=True)
    forward = rows < cols
    w_ij = np.zeros(pairs.size, dtype=np.float64)
    w_ji = np.zeros(pairs.size, dtype=np.float64)
    w_ij[inverse[forward]] = data[forward]
    w_ji[inverse[~forward]] = data[~forward]

    merged = np.asarray(func(w_ij, w_ji), dtype=np.float64)
    if merged.shape != w_ij.shape:
        raise ValueError(
            f"symmetrize returned an array of shape {merged.shape}; expected {w_ij.shape}"
        )
    if not (np.isfinite(merged) & (merged >= 0)).all():
        bad = float(merged[~(np.isfinite(merged) & (merged >= 0))][0])
        raise ValueError(f"symmetrize returned {bad}; expected finite values >= 0")

    present = merged > 0
    lo_p, hi_p, merged = pairs[present] // n, pairs[present] % n, merged[present]
    return (
        np.concatenate([lo_p, hi_p]),
        np.concatenate([hi_p, lo_p]),
        np.concatenate([merged, merged]),
    )
