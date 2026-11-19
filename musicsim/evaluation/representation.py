"""Geometry of a representation: effective rank, anisotropy and linear CKA.

Three summaries of an ``(N, d)`` matrix of item vectors, independent of any
relevance labels:

- :func:`effective_rank` is how many directions the rows really use: the
  exponential of the Shannon entropy of the singular values normalised to sum
  to one (``Roy2007``). It is ``d`` for the ``d x d`` identity, ``1`` for a
  rank-one matrix, and it does not depend on the overall scale of ``X``.
- :func:`mean_cosine` is the anisotropy of the representation: the mean cosine
  similarity between two distinct items (``Ethayarajh2019``). ``0`` for
  orthogonal vectors, ``1`` when every row points the same way.
- :func:`linear_cka` is the linear centred kernel alignment of two
  representations of the same items (``Kornblith2019``): ``1`` for the same
  representation up to an orthogonal transform and an isotropic scale, which is
  what makes it fit for comparing spaces of different dimension.

The first two detect the collapse of a learned embedding, where every item is
mapped to (nearly) the same point or to a low-dimensional subspace; recall
alone does not reveal it, since a few directions can still separate a handful
of genres. They are complementary: a cloud squeezed onto a line through the
origin has effective rank 1 but a mean cosine close to 0 (half the points on
each side), and a cloud with a large common offset has mean cosine near 1 whatever
its rank.

Decisions (own, after ``Roy2007``):

- The effective rank is computed on ``X`` as given, **not centred**. That is the
  original definition, it gives exactly ``d`` for the identity and ``1`` for a
  collapse to a single point. Centring would measure the spread around the
  mean instead, and a collapsed cloud with a little noise could then score a
  high rank; the common offset that centring removes is what
  :func:`mean_cosine` measures.
- Zero singular values add nothing to the entropy (``0 log 0 = 0``).
- The mean cosine averages over the ``N (N - 1)`` ordered pairs ``i != j``.
  Including the diagonal would add a bias of order ``1 / N`` (``d`` orthonormal
  rows would give ``1 / d`` instead of ``0``). It is computed exactly without
  the ``N x N`` similarity matrix, from the unit rows ``u_i``:
  ``sum_{i != j} u_i . u_j = ||sum_i u_i||^2 - N``, so time is ``O(N d)`` and
  memory ``O(d)`` beyond the copy of ``X``.

Linear CKA uses the feature-space form of ``Kornblith2019``, with the columns
of both matrices centred: ``||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)``. It needs
only ``d x d`` products, never the ``N x N`` Gram matrices.

Every function works in ``float64`` whatever the input dtype, and rejects
empty, non-finite or degenerate input with a ``ValueError``: a ``nan`` score
in a results table would hide the problem that produced it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["effective_rank", "linear_cka", "mean_cosine"]

#: Fewest rows a pairwise or centred statistic needs: with one row there is no
#: pair of distinct items and every centred column is zero.
MIN_ROWS_PAIRWISE = 2


def effective_rank(X: np.ndarray) -> float:
    """Effective rank of ``X``: ``exp(H(p))`` with ``p = s / sum(s)`` (``Roy2007``).

    ``s`` are the singular values of ``X`` as given, without centring, and ``H``
    is the Shannon entropy in nats with ``0 log 0 = 0``. The result lies in
    ``[1, min(N, d)]``; it is ``d`` for the ``d x d`` identity and invariant to
    scaling ``X`` by a non-zero constant.
    """
    X = _as_matrix(X, "X", min_rows=1)
    singular = np.linalg.svd(X, compute_uv=False)
    total = singular.sum()
    if total == 0:
        raise ValueError("X is all zeros; the effective rank needs at least one non-zero entry")
    p = singular[singular > 0] / total
    return float(np.exp(-np.sum(p * np.log(p))))


def mean_cosine(X: np.ndarray) -> float:
    """Mean cosine similarity over the pairs ``i != j`` of rows of ``X`` (``Ethayarajh2019``).

    Computed as ``(||sum_i u_i||^2 - N) / (N (N - 1))`` from the unit rows
    ``u_i``, which is exact and avoids the ``N x N`` similarity matrix. Every
    row must be non-zero, since a zero vector has no direction.
    """
    X = _as_matrix(X, "X", min_rows=MIN_ROWS_PAIRWISE)
    norms = np.linalg.norm(X, axis=1)
    zero = np.flatnonzero(norms == 0)
    if zero.size:
        raise ValueError(
            f"X has {zero.size} zero row(s) (first at row {zero[0]}); "
            "expected non-zero rows, which have a direction"
        )
    total = (X / norms[:, None]).sum(axis=0)
    n = X.shape[0]
    return float((total @ total - n) / (n * (n - 1)))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representations of the same ``N`` items (``Kornblith2019``).

    ``X`` is ``(N, d_x)`` and ``Y`` is ``(N, d_y)``; row ``i`` of both is the
    same item. Columns are centred, then the score is
    ``||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)``, in ``[0, 1]``. It is invariant to
    orthogonal transforms and isotropic scaling of either matrix, and to adding
    a constant row to it. A representation that is constant over the items has
    no centred variance and is rejected.
    """
    X = _as_matrix(X, "X", min_rows=MIN_ROWS_PAIRWISE)
    Y = _as_matrix(Y, "Y", min_rows=MIN_ROWS_PAIRWISE)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows and Y has {Y.shape[0]}; "
            "expected the same items, one per row, in both"
        )
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    norm_x = np.linalg.norm(X.T @ X)
    norm_y = np.linalg.norm(Y.T @ Y)
    for name, norm in (("X", norm_x), ("Y", norm_y)):
        if norm == 0:
            raise ValueError(
                f"{name} is the same vector for every item; "
                "expected a representation that varies across items"
            )
    return float(np.linalg.norm(Y.T @ X) ** 2 / (norm_x * norm_y))


def _as_matrix(X: np.ndarray, name: str, *, min_rows: int) -> np.ndarray:
    """``X`` as a finite ``float64`` 2-D array with at least ``min_rows`` rows and one column."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < min_rows or X.shape[1] == 0:
        raise ValueError(
            f"{name} has shape {X.shape}; expected a 2-D array with at least "
            f"{min_rows} row(s) and one column"
        )
    if not np.isfinite(X).all():
        raise ValueError(f"{name} has non-finite values; expected finite numbers only")
    return X
