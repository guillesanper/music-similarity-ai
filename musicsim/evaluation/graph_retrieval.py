"""Retrieval metrics computed from a :class:`~musicsim.graphs.base.Graph`, so that graph
variants with rows of fewer than ``k`` neighbours (mutual kNN, thresholds) are comparable.

Ranking rule
------------
R@10 and nDCG@10 need ten ordered neighbours and MAP needs the whole ranking,
but a variant may give a query fewer neighbours than that. Without a common
rule the variants would be scored on different amounts of evidence, so every
query is ranked lexicographically: first the neighbours the graph gives it, in
the graph's own order (best first), then every remaining candidate by cosine
similarity, descending. The graph decides *who comes first*; the cosine only
fills the rest.

Exclusion variants
------------------
The ``group`` argument is the same exclusion group
:func:`musicsim.evaluation.retrieval.compute_per_query_metrics` uses (the query
alone for ``raw``, its duplicate group for ``dedup``, its artist for
``artist_filter``). A graph neighbour inside the query's group is dropped, not
ranked: it would otherwise take a top position and count as relevant, which is
exactly what the variant exists to prevent. Padding slots are skipped. The
dropped neighbours and the padding leave the same hole, and the cosine fills it.

Completion
----------
A query is *completed* when, after those exclusions, the graph gives it fewer
than ``completion_depth`` (10, the headline depth) neighbours, so that its
top-10 metrics partly come from the cosine fill. The flag therefore depends on
the variant: a canonical kNN with k = 10 is never completed under ``raw``, but
is under ``dedup`` or ``artist_filter`` whenever one of its neighbours is
excluded. MAP always reads the full ranking and is not what the flag measures.

With a graph that is exactly the cosine kNN, the ranking equals the cosine
ranking, so the metrics equal those of ``compute_per_query_metrics``: both go
through :func:`musicsim.evaluation.retrieval.metrics_from_order`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from musicsim.evaluation.retrieval import metrics_from_order
from musicsim.graphs.base import PADDING, Graph

__all__ = ["COMPLETION_DEPTH", "graph_per_query_metrics", "graph_ranking"]

#: Neighbours a query must get from the graph to not count as completed.
COMPLETION_DEPTH = 10

_DEFAULT_BLOCK_ROWS = 512


def graph_ranking(
    graph: Graph,
    X: np.ndarray,
    block: np.ndarray,
    group: np.ndarray,
    *,
    completion_depth: int = COMPLETION_DEPTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Full gallery ranking of every query in ``block`` and its "completed" flag.

    Returns ``(order, completed)``: ``order`` is ``(len(block), N)`` with the
    graph's neighbours first, then the rest by cosine, then the query's
    exclusion group last (as in the cosine ranking); ``completed`` is a boolean
    ``(len(block),)``. ``X`` must be L2-normalised and row-aligned with
    ``graph``.
    """
    group = np.asarray(group)
    S = (X[block] @ X.T).astype(np.float64)
    same_group = group[block][:, None] == group[None, :]
    S[same_group] = -np.inf

    neighbors = graph.neighbors[block].astype(np.int64)
    k = neighbors.shape[1]
    real = neighbors != PADDING
    safe = np.where(real, neighbors, 0)
    usable = real & (group[safe] != group[block][:, None])

    # A graph neighbour outranks every cosine (<= 1) and keeps the graph's own
    # order: the key falls by one per slot. Excluded items stay at -inf.
    boost = 2.0 + (k - np.arange(k, dtype=np.float64))
    r, c = np.nonzero(usable)
    S[r, neighbors[r, c]] = boost[c]

    order = np.argsort(-S, axis=1, kind="stable")
    completed = usable.sum(axis=1) < completion_depth
    return order, completed


def graph_per_query_metrics(
    graph: Graph,
    X: np.ndarray,
    ids: np.ndarray,
    rows: np.ndarray,
    group: np.ndarray,
    genre: np.ndarray,
    ks: Sequence[int],
    *,
    completion_depth: int = COMPLETION_DEPTH,
    block_rows: int = _DEFAULT_BLOCK_ROWS,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-query metrics of the ranking ``graph`` induces, as in ``per_query.csv``.

    Same arguments and output frame as
    :func:`musicsim.evaluation.retrieval.compute_per_query_metrics`, plus the
    graph. Also returns the boolean "completed" flag of every query in ``rows``
    (see the module docstring); its mean is the fraction of completed queries.
    ``graph`` must have been built over exactly the rows of ``X`` (the whole
    gallery, near-duplicates included): ``graph.ids`` must equal ``ids``.
    """
    ks = sorted({int(k) for k in ks})
    n = X.shape[0]
    ids = np.asarray(ids)
    if graph.n_nodes != n or not np.array_equal(graph.ids, ids):
        raise ValueError(
            f"graph has {graph.n_nodes} node(s) and X has {n} row(s); expected a graph "
            "built over exactly the ids of the gallery, in the same order"
        )
    if max(ks) >= n:
        raise ValueError(f"k={max(ks)} does not fit a gallery of {n} items")
    if block_rows < 1:
        raise ValueError(f"block_rows must be >= 1, got {block_rows}")
    group = np.asarray(group)

    frames: list[pd.DataFrame] = []
    flags: list[np.ndarray] = []
    for start in range(0, len(rows), block_rows):
        block = rows[start : start + block_rows]
        order, completed = graph_ranking(
            graph, X, block, group, completion_depth=completion_depth
        )
        frames.append(metrics_from_order(order, block, ids, group, genre, ks))
        flags.append(completed)
    return pd.concat(frames, ignore_index=True), np.concatenate(flags)
