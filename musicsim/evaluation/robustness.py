"""The two measures specific to robustness; the rest reuses retrieval.

Protocol: the **gallery** is the matrix of original embeddings ``Z``; the
**queries** are perturbed versions of the same clips, ``Q``, aligned row by row
with ``Z`` (``Q[i]`` is a perturbed ``Z[i]``). All matrices have rows of norm
1, so a similarity is a dot product, and everything is processed in row blocks
so memory is ``block_rows * N`` rather than ``N ** 2``.

A query's own original, and every near-duplicate of it, is excluded from its
candidates (:func:`query_neighbors`): the question is whether the *neighbourhood*
changes when the audio is degraded, not whether a clip finds itself. The
position of the original (:func:`self_rank`) is measured separately, over the
whole gallery, because there the original must be able to appear.
"""

from __future__ import annotations

import numpy as np

__all__ = ["neighbor_set_jaccard", "query_neighbors", "self_rank"]

_BLOCK_ROWS = 512


def query_neighbors(
    Q: np.ndarray, Z: np.ndarray, groups: np.ndarray, k: int, *, block_rows: int = _BLOCK_ROWS
) -> np.ndarray:
    """``int (N, k)``: each query's ``k`` nearest gallery rows, best first.

    ``groups`` is the exclusion group of each gallery row (the same code for
    near-duplicates); gallery rows in the group of ``Q[i]``'s original are not
    candidates for query ``i``, which includes the original itself. Ties keep
    the lower row first. If a query has fewer than ``k`` candidates left,
    ``ValueError``.
    """
    _check_aligned(Q, Z)
    groups = np.asarray(groups)
    if groups.shape != (len(Z),):
        raise ValueError(f"groups must have shape ({len(Z)},), got {groups.shape}")
    if not k >= 1:
        raise ValueError(f"k must be at least 1, got {k}")
    candidates = (
        len(Z)
        - np.bincount(np.unique(groups, return_inverse=True)[1])[
            np.unique(groups, return_inverse=True)[1]
        ]
    )
    if candidates.min() < k:
        raise ValueError(
            f"a query has only {int(candidates.min())} candidate(s) outside its group; "
            f"expected at least k={k}"
        )

    out = np.empty((len(Q), k), dtype=np.int64)
    for start in range(0, len(Q), block_rows):
        block = np.arange(start, min(start + block_rows, len(Q)))
        S = (Q[block] @ Z.T).astype(np.float64)
        S[groups[block][:, None] == groups[None, :]] = -np.inf
        out[block] = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return out


def neighbor_set_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``float (N,)``: per row, ``|A and B| / |A or B|`` of two ``(N, k)`` neighbour tables.

    Each row of ``a`` and ``b`` is read as a set of gallery rows; the order
    inside a row does not matter.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"expected two (N, k) tables of the same shape, got {a.shape}, {b.shape}")
    k = a.shape[1]
    shared = (a[:, :, None] == b[:, None, :]).any(axis=2).sum(axis=1)
    return shared / (2 * k - shared)


def self_rank(Q: np.ndarray, Z: np.ndarray, *, block_rows: int = _BLOCK_ROWS) -> np.ndarray:
    """``int (N,)``, 1-based: where gallery row ``i`` falls in the ranking by ``Q[i]``.

    Nothing is excluded, so the original competes with its near-duplicates. Ties
    do not count against it: the rank is one plus the number of rows *strictly*
    more similar to the query.
    """
    _check_aligned(Q, Z)
    out = np.empty(len(Q), dtype=np.int64)
    for start in range(0, len(Q), block_rows):
        block = np.arange(start, min(start + block_rows, len(Q)))
        S = (Q[block] @ Z.T).astype(np.float64)
        own = S[np.arange(len(block)), block]
        out[block] = 1 + (own[:, None] < S).sum(axis=1)
    return out


def _check_aligned(Q: np.ndarray, Z: np.ndarray) -> None:
    if Q.shape != Z.shape or Z.ndim != 2:
        raise ValueError(
            "queries and gallery must be (N, d) arrays of the same shape, "
            f"got {Q.shape} and {Z.shape}"
        )
