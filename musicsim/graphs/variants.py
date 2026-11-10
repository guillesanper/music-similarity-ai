"""Crisp graph variants, to measure whether a conclusion survives a change of graph.

The canonical graph is the cosine kNN of :mod:`musicsim.graphs.knn`
(``knn``, with ``k`` in {5, 10, 20, 50} it is the same builder). Three more
builders define "who is connected to whom" differently:

* ``mutual_knn``: ``i -> j`` only if ``j`` is among the ``k`` neighbours of
  ``i`` *and* ``i`` is among those of ``j``. Rows keep at most ``k`` slots and
  are padded; weights are the cosine similarity.
* ``epsilon``: every pair whose cosine similarity reaches a global threshold.
  By default the threshold is chosen so the graph has as many directed edges as
  the canonical kNN (``N * k``): it is the ``N * k``-th largest off-diagonal
  similarity of the gallery itself, computed without labels. ``graph.epsilon``
  may instead be a number. The graph is symmetric, nodes may end up isolated
  (a fraction worth reporting: it is what separates a threshold from a kNN),
  and the row width is the largest degree, not ``k``. With ties at the
  threshold every tied pair is kept, so the edge count can exceed ``N * k``.
* ``mutual_proximity`` (Schnitzer et al. 2012), empirical version: the
  similarity is replaced by ``MP(i, j)``, the fraction of the other ``N - 2``
  items ``l`` with ``s(i, l) < s(i, j)`` and ``s(j, l) < s(j, i)``, and the
  graph is the kNN over it. ``MP`` only uses comparisons, so it is invariant to
  any strictly increasing transformation of the similarity, and it is symmetric.
  It is an integer count over ``N - 2``, so ties are frequent; they are broken
  by cosine similarity and then by row position, which keeps the graph
  deterministic. Edge weights are the ``MP`` values (the ``Graph`` contract
  orders a row by decreasing weight, and the ranking is by ``MP``). ``MP`` can
  be ``0`` on a real edge, which ``to_csr(weights="weight")`` rejects; use
  ``weights="binary"`` for these graphs. Cost: the dense ``(N, N)`` similarity
  stays in memory (the comparison needs every row) and the loop is ``O(N^3)``
  time with ``O(N^2)`` working memory per node. That is immediate for the
  800-item test gallery and impractical for several thousand items, where a
  Gaussian approximation of ``MP`` would be needed; it is NOT implemented here.

All of them read ``graph.k`` and use cosine similarity on L2-normalised rows
(``graph.metric`` is not read by any builder yet), exclude self by row index,
and refuse ``exclude_self: false``, since a self-neighbour is neither a mutual
neighbour nor a meaningful threshold edge.
"""

from __future__ import annotations

import numpy as np

from musicsim.config import ConfigError
from musicsim.graphs.base import PADDING, Graph, GraphBuilder
from musicsim.graphs.knn import top_k_by_cosine
from musicsim.registry import register_graph_builder

__all__ = ["EpsilonGraphBuilder", "MutualKnnGraphBuilder", "MutualProximityGraphBuilder"]

#: Rows of the similarity matrix computed at a time (bounds memory, not results).
_BLOCK_ROWS = 512


def _read_k(graph_cfg: dict, builder: str, n: int) -> int:
    if "k" not in graph_cfg:
        raise ConfigError(f"graph.k is missing; the {builder!r} builder expects it")
    k = int(graph_cfg["k"])
    if not 0 < k <= n - 1:
        raise ValueError(f"k must be in [1, {n - 1}] for {n} items, got {k}")
    return k


def _require_self_exclusion(graph_cfg: dict, builder: str) -> None:
    if not bool(graph_cfg.get("exclude_self", True)):
        raise ValueError(
            f"graph.exclude_self is False; the {builder!r} builder expects True "
            "(a self-neighbour is neither mutual nor a threshold edge)"
        )


def _similarity_block(X: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Cosine similarity of rows ``start:stop`` against all rows, diagonal at ``-inf``."""
    S = X[start:stop] @ X.T
    S[np.arange(stop - start), np.arange(start, stop)] = -np.inf
    return S


def _pad_rows(
    neighbors: list[np.ndarray], weights: list[np.ndarray], width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stack ragged rows into ``(N, width)`` arrays with a padding suffix."""
    n = len(neighbors)
    out_n = np.full((n, width), PADDING, dtype=np.int32)
    out_w = np.zeros((n, width), dtype=np.float32)
    for i, (row_n, row_w) in enumerate(zip(neighbors, weights, strict=True)):
        out_n[i, : row_n.size] = row_n
        out_w[i, : row_w.size] = row_w
    return out_n, out_w


@register_graph_builder("mutual_knn")
class MutualKnnGraphBuilder(GraphBuilder):
    """Edges of the cosine kNN whose reverse edge is also in the kNN."""

    def build(self, X: np.ndarray, ids: np.ndarray) -> Graph:
        _require_self_exclusion(self.graph_cfg, "mutual_knn")
        n = X.shape[0]
        k = _read_k(self.graph_cfg, "mutual_knn", n)
        neighbors, weights = top_k_by_cosine(X, k)

        src = np.repeat(np.arange(n, dtype=np.int64), k)
        forward = src * n + neighbors.ravel().astype(np.int64)
        backward = neighbors.ravel().astype(np.int64) * n + src
        keep = np.isin(backward, forward).reshape(n, k)

        # A stable sort on "dropped" moves kept entries to the front without
        # reordering them, so rows stay best-first and padding is a suffix.
        order = np.argsort(~keep, axis=1, kind="stable")
        neighbors = np.take_along_axis(neighbors, order, axis=1)
        weights = np.take_along_axis(weights, order, axis=1)
        keep = np.take_along_axis(keep, order, axis=1)
        neighbors[~keep] = PADDING
        weights[~keep] = 0.0
        return Graph(neighbors=neighbors, weights=weights, ids=np.asarray(ids))


def _matched_threshold(X: np.ndarray, target: int) -> float:
    """The ``target``-th largest off-diagonal cosine similarity of ``X``."""
    n = X.shape[0]
    candidates = []
    for start in range(0, n, _BLOCK_ROWS):
        flat = _similarity_block(X, start, min(start + _BLOCK_ROWS, n)).ravel()
        m = min(target, flat.size)
        candidates.append(np.partition(flat, flat.size - m)[flat.size - m :])
    pooled = np.concatenate(candidates)
    return float(np.partition(pooled, pooled.size - target)[pooled.size - target])


@register_graph_builder("epsilon")
class EpsilonGraphBuilder(GraphBuilder):
    """Every pair with cosine similarity at or above a global threshold."""

    def _threshold(self, X: np.ndarray) -> float:
        n = X.shape[0]
        setting = self.graph_cfg.get("epsilon", "matched")
        if isinstance(setting, str):
            if setting != "matched":
                raise ConfigError(f"graph.epsilon is {setting!r}; expected 'matched' or a number")
            k = _read_k(self.graph_cfg, "epsilon", n)
            return _matched_threshold(X, n * k)
        if isinstance(setting, bool):
            raise ConfigError(f"graph.epsilon is {setting!r}; expected 'matched' or a number")
        return float(setting)

    def build(self, X: np.ndarray, ids: np.ndarray) -> Graph:
        _require_self_exclusion(self.graph_cfg, "epsilon")
        n = X.shape[0]
        threshold = self._threshold(X)

        rows_n: list[np.ndarray] = []
        rows_w: list[np.ndarray] = []
        for start in range(0, n, _BLOCK_ROWS):
            stop = min(start + _BLOCK_ROWS, n)
            S = _similarity_block(X, start, stop)
            # The diagonal is excluded by index: with a threshold of -inf its
            # -inf marker would otherwise pass ``>=``.
            above = threshold <= S
            above[np.arange(stop - start), np.arange(start, stop)] = False
            for r in range(stop - start):
                cols = np.flatnonzero(above[r])
                order = np.argsort(-S[r, cols], kind="stable")
                rows_n.append(cols[order])
                rows_w.append(np.clip(S[r, cols[order]], -1.0, 1.0))

        width = max(1, max((row.size for row in rows_n), default=0))
        neighbors, weights = _pad_rows(rows_n, rows_w, width)
        return Graph(neighbors=neighbors, weights=weights, ids=np.asarray(ids))


def _mutual_proximity_row(S: np.ndarray, i: int) -> np.ndarray:
    """``MP(i, j)`` for every ``j``, ``(N,)``; entry ``i`` is meaningless.

    ``S`` is the full similarity matrix with ``+inf`` on the diagonal, which
    makes ``l = i`` and ``l = j`` fail the strict comparisons without any
    special case.
    """
    n = S.shape[0]
    closer_to_i = S[i][None, :] < S[i][:, None]  # [j, l]: s(i, l) < s(i, j)
    closer_to_j = S[:, i][:, None] > S  # [j, l]: s(j, l) < s(j, i)
    np.logical_and(closer_to_i, closer_to_j, out=closer_to_i)
    return closer_to_i.sum(axis=1) / (n - 2)


@register_graph_builder("mutual_proximity")
class MutualProximityGraphBuilder(GraphBuilder):
    """kNN over the empirical mutual proximity of the cosine similarity."""

    def build(self, X: np.ndarray, ids: np.ndarray) -> Graph:
        _require_self_exclusion(self.graph_cfg, "mutual_proximity")
        n = X.shape[0]
        if n < 3:
            raise ValueError(f"mutual_proximity needs at least 3 items, got {n}")
        k = _read_k(self.graph_cfg, "mutual_proximity", n)

        S = X @ X.T
        np.fill_diagonal(S, np.inf)
        neighbors = np.empty((n, k), dtype=np.int32)
        weights = np.empty((n, k), dtype=np.float32)
        for i in range(n):
            mp = _mutual_proximity_row(S, i)
            mp[i] = -np.inf
            # Primary key MP, then cosine; self sorts last (MP = -inf).
            order = np.lexsort((-S[i], -mp))[:k]
            neighbors[i] = order
            weights[i] = mp[order]
        return Graph(neighbors=neighbors, weights=weights, ids=np.asarray(ids))
