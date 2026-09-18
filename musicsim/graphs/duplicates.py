"""Near-duplicate detection from embeddings, recomputed for real (see
``INDEX_COLUMNS`` in :mod:`musicsim.datasets.base`): a dataset's ``group_id``
column is only a placeholder (each item its own group) until this module runs
on its embeddings. The archived FMA-small pipeline found 16 such pairs at
``NEAR_DUP_SIM`` on MFCC features, checked in ``tests/test_graphs_duplicates.py``
and reproduced by a real run of ``configs/experiments/e01_mfcc_baseline.yaml``.

Two items are near-duplicates when their cosine similarity is at or above
``NEAR_DUP_SIM`` — the same MP3 uploaded twice under two ids gives similarity 1
up to floating-point rounding, well above any similarity two different
recordings reach by chance. ``duplicate_groups`` then unions transitively
(A~B~C become one group of three), so :mod:`musicsim.evaluation.retrieval`'s
``dedup`` variant can exclude a query's whole near-duplicate group at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

__all__ = ["NEAR_DUP_SIM", "duplicate_groups", "near_duplicate_pairs"]

#: Cosine similarity at or above which two items count as near-duplicates.
NEAR_DUP_SIM = 0.9999

#: Rows per block in the exhaustive pairwise search: memory is
#: O(block_rows * N), not O(N**2).
_DEFAULT_BLOCK_ROWS = 2048


def near_duplicate_pairs(
    X: np.ndarray,
    ids: np.ndarray,
    *,
    threshold: float = NEAR_DUP_SIM,
    block_rows: int = _DEFAULT_BLOCK_ROWS,
) -> pd.DataFrame:
    """Every pair with cosine similarity ``>= threshold``, ``id_a < id_b``.

    Exhaustive (not limited to the kNN graph) but never materialises the full
    ``(N, N)`` similarity matrix: for the row block ``[i0, i1)`` only columns
    ``j >= i0`` are compared (pairs with ``j < i0`` already came out of an
    earlier block), and within the block only ``j > row`` is kept.
    """
    if block_rows < 1:
        raise ValueError(f"block_rows must be >= 1, got {block_rows}")
    n = X.shape[0]
    ids = np.asarray(ids)
    rows_a: list[np.ndarray] = []
    rows_b: list[np.ndarray] = []
    sims: list[np.ndarray] = []

    for i0 in range(0, n, block_rows):
        S = X[i0 : i0 + block_rows] @ X[i0:].T  # (block, n - i0)
        r, c = np.nonzero(threshold <= S)
        keep = c > r  # global column i0 + c > global row i0 + r
        rows_a.append(i0 + r[keep])
        rows_b.append(i0 + c[keep])
        sims.append(S[r[keep], c[keep]])

    a = np.concatenate(rows_a) if rows_a else np.empty(0, dtype=np.intp)
    b = np.concatenate(rows_b) if rows_b else np.empty(0, dtype=np.intp)
    s = np.concatenate(sims) if sims else np.empty(0, dtype=np.float32)

    pairs = pd.DataFrame({"id_a": ids[a], "id_b": ids[b], "similarity": s})
    swap = pairs["id_a"] > pairs["id_b"]
    pairs.loc[swap, ["id_a", "id_b"]] = pairs.loc[swap, ["id_b", "id_a"]].to_numpy()
    return pairs.sort_values(["id_a", "id_b"]).reset_index(drop=True)


def duplicate_groups(ids: np.ndarray, pairs: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Group id of each row of ``ids``: connected components of ``pairs``.

    ``pairs`` is a DataFrame with ``id_a``/``id_b`` columns (as returned by
    :func:`near_duplicate_pairs`) or a plain ``(M, 2)`` array of identifiers.
    An item with no duplicate is its own singleton group. Groups are numbered
    ``0..n_groups-1``; the numbering carries no meaning beyond identity.
    """
    ids = np.asarray(ids)
    n = len(ids)
    if isinstance(pairs, pd.DataFrame):
        array = pairs[["id_a", "id_b"]].to_numpy()
    else:
        array = np.asarray(pairs).reshape(-1, 2)

    if len(array) == 0:
        return np.arange(n, dtype=np.int64)

    position = pd.Series(np.arange(n), index=ids)
    missing = pd.Index(array.ravel()).difference(position.index)
    if len(missing):
        raise ValueError(f"{len(missing)} id(s) in pairs are not in ids, e.g. {list(missing[:5])}")

    a = position.loc[array[:, 0]].to_numpy()
    b = position.loc[array[:, 1]].to_numpy()
    adjacency = sparse.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(adjacency, directed=False)[1].astype(np.int64)
