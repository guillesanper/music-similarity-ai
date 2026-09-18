"""Cosine k-nearest-neighbour graph, the project's canonical crisp graph.

:func:`top_k_by_cosine` is the shared low-level primitive: given L2-normalised
rows (so cosine similarity is a plain dot product), it returns each row's top
``k`` neighbours, processed in row blocks so the full ``(N, N)`` similarity
matrix is never materialised at once (peak memory is ``block_rows`` times
``N``, not ``N`` squared). :class:`KnnGraphBuilder` wraps it for the B.1.8
``build_graph`` contract; :mod:`musicsim.evaluation.retrieval` calls it
directly with whatever depth a metric needs, which can differ from the
configured ``graph.k``.

Self-exclusion is done by row index, never by identifier or by thresholding
the similarity value: two duplicated tracks have similarity 1 too, and a
value-based rule would drop the wrong one.
"""

from __future__ import annotations

import numpy as np

from musicsim.graphs.base import Graph, GraphBuilder
from musicsim.registry import register_graph_builder

__all__ = ["KnnGraphBuilder", "top_k_by_cosine"]

#: Rows processed per block. Bounds peak memory to ``block_rows * N`` floats
#: instead of ``N**2``; unrelated to correctness, only to how much RAM one
#: call needs at a time.
_DEFAULT_BLOCK_ROWS = 512


def top_k_by_cosine(
    X: np.ndarray,
    k: int,
    *,
    exclude_self: bool = True,
    block_rows: int = _DEFAULT_BLOCK_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Each row's ``k`` nearest neighbours by cosine similarity, row-blocked.

    ``X`` is assumed L2-normalised (as every embedding in this project is), so
    cosine similarity is ``X @ X.T``. Returns ``(neighbors, weights)``, both
    ``(N, k)``, ``neighbors`` holding row positions into ``X`` (not
    identifiers) sorted by decreasing similarity. With ``exclude_self=True``
    row ``i`` never appears among its own neighbours, checked by index.
    """
    n = X.shape[0]
    if not 0 < k <= n - (1 if exclude_self else 0):
        limit = n - 1 if exclude_self else n
        raise ValueError(f"k must be in [1, {limit}] for {n} items, got {k}")
    if block_rows < 1:
        raise ValueError(f"block_rows must be >= 1, got {block_rows}")

    neighbors = np.empty((n, k), dtype=np.int64)
    weights = np.empty((n, k), dtype=np.float32)

    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        rows = np.arange(start, stop)
        S = X[rows] @ X.T  # (block, n)

        if exclude_self:
            S[np.arange(stop - start), rows] = -np.inf

        # argpartition narrows to the top k unordered (O(n) per row), then a
        # full sort of just those k columns orders them; far cheaper than
        # sorting all n columns when k << n.
        top = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(S, top, axis=1)
        order = np.argsort(-top_scores, axis=1, kind="stable")
        neighbors[start:stop] = np.take_along_axis(top, order, axis=1)
        weights[start:stop] = np.take_along_axis(top_scores, order, axis=1)

    return neighbors.astype(np.int32), np.clip(weights, -1.0, 1.0)


@register_graph_builder("knn")
class KnnGraphBuilder(GraphBuilder):
    """The canonical crisp graph: cosine kNN, self-exclusion by row index."""

    def build(self, X: np.ndarray, ids: np.ndarray) -> Graph:
        k = int(self.graph_cfg["k"])
        exclude_self = bool(self.graph_cfg.get("exclude_self", True))
        neighbors, weights = top_k_by_cosine(X, k, exclude_self=exclude_self)
        return Graph(neighbors=neighbors, weights=weights, ids=np.asarray(ids))
