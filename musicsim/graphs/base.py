"""The graph contract every builder implements (B.1.8): ``build_graph(X, ids,
graph_cfg) -> Graph``.

A builder is registered by name (``graph.builder`` in the configuration) in
:data:`musicsim.registry.GRAPH_BUILDERS`, exactly like an extractor or a
dataset. :data:`Graph` is the frozen result: a directed k-nearest-neighbour
graph, ``neighbors[i]`` and ``weights[i]`` being node ``i``'s ``k`` neighbours
ordered by decreasing similarity.

Rows may have fewer than ``k`` real neighbours (mutual kNN, thresholds,
alpha-cuts): the missing slots are filled with :data:`PADDING` so the arrays
keep their ``(N, k)`` shape and every consumer can index them without
branching on a ragged structure. A padded slot is marked by the neighbour
value (``-1`` can never be a row position) and carries weight ``0.0``; padding
is always a suffix of its row, which keeps "the first ``degree[i]`` entries are
the real neighbours, best first" true and makes ``rank`` well defined. A graph
without padding behaves exactly as a fixed-degree kNN graph.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["PADDING", "Graph", "GraphBuilder", "build_graph"]

#: Neighbour value of an unused slot in a row with fewer than ``k`` neighbours.
PADDING = -1


@dataclasses.dataclass(frozen=True, slots=True)
class Graph:
    """A directed k-nearest-neighbour graph over a set of items.

    ``neighbors`` and ``weights`` are ``(N, k)``: row ``i`` holds the ``k``
    neighbours of ``ids[i]``, as row positions into ``ids`` (never as
    identifiers), ordered by decreasing ``weights``. ``ids`` is ``(N,)``.
    A row with fewer than ``k`` neighbours ends in a run of :data:`PADDING`
    slots (``neighbors == PADDING``, ``weights == 0.0``).
    """

    neighbors: np.ndarray
    weights: np.ndarray
    ids: np.ndarray

    def __post_init__(self) -> None:
        n, k = self.neighbors.shape
        if self.weights.shape != (n, k):
            raise ValueError(
                f"neighbors {self.neighbors.shape} and weights {self.weights.shape} disagree"
            )
        if self.ids.shape != (n,):
            raise ValueError(f"ids {self.ids.shape} does not have {n} rows like neighbors")
        padded = self.neighbors == PADDING
        bad = ~padded & ((self.neighbors < 0) | (self.neighbors >= n))
        if bad.any():
            row = int(np.flatnonzero(bad.any(axis=1))[0])
            raise ValueError(
                f"row {row} has neighbor {int(self.neighbors[row][bad[row]][0])}; "
                f"expected a position in [0, {n}) or PADDING ({PADDING})"
            )
        loose = padded & (self.weights != 0.0)
        if loose.any():
            row = int(np.flatnonzero(loose.any(axis=1))[0])
            raise ValueError(
                f"row {row} has a nonzero weight {float(self.weights[row][loose[row]][0])} "
                "in a padding slot; expected 0.0"
            )
        gaps = padded[:, :-1] & ~padded[:, 1:]
        if gaps.any():
            row = int(np.flatnonzero(gaps.any(axis=1))[0])
            raise ValueError(
                f"row {row} has a real neighbor after a padding slot; "
                "padding must be a suffix of the row"
            )

    @property
    def n_nodes(self) -> int:
        return self.ids.shape[0]

    @property
    def k(self) -> int:
        return self.neighbors.shape[1]

    @property
    def degree(self) -> np.ndarray:
        """Real (non-padding) neighbours of every node, ``(N,)`` ``int64``."""
        return (self.neighbors != PADDING).sum(axis=1, dtype=np.int64)

    def to_edges_frame(self) -> pd.DataFrame:
        """The ``edges.csv`` of B.1.8: ``src_id, dst_id, weight, rank`` (rank 1-based).

        Padding slots are not edges and are dropped; ``rank`` is the position
        within the row, so it still starts at 1 for every source.
        """
        n, k = self.neighbors.shape
        real = (self.neighbors != PADDING).ravel()
        src = np.repeat(self.ids, k)[real]
        dst = self.ids[self.neighbors.ravel()[real]]
        rank = np.tile(np.arange(1, k + 1), n)[real]
        return pd.DataFrame(
            {
                "src_id": src,
                "dst_id": dst,
                "weight": self.weights.ravel()[real],
                "rank": rank,
            }
        )

    def write(self, directory: Path) -> None:
        """Write ``neighbors.npz`` and ``edges.csv`` under ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "neighbors.npz",
            neighbors=self.neighbors,
            weights=self.weights,
            ids=self.ids,
        )
        self.to_edges_frame().to_csv(directory / "edges.csv", index=False)

    @classmethod
    def read(cls, directory: Path) -> Graph:
        """Inverse of :meth:`write`, from ``neighbors.npz`` only (``edges.csv`` is derived)."""
        with np.load(directory / "neighbors.npz") as data:
            return cls(neighbors=data["neighbors"], weights=data["weights"], ids=data["ids"])


class GraphBuilder(ABC):
    """One way of turning an embedding matrix into a :class:`Graph`."""

    def __init__(self, graph_cfg: Mapping[str, Any]) -> None:
        self.graph_cfg = graph_cfg

    @abstractmethod
    def build(self, X: np.ndarray, ids: np.ndarray) -> Graph:
        """Build the graph over rows ``X[i]`` / ``ids[i]``."""


def build_graph(X: np.ndarray, ids: np.ndarray, graph_cfg: Mapping[str, Any]) -> Graph:
    """The free function of the B.1.8 contract: dispatches on ``graph_cfg["builder"]``."""
    from musicsim.registry import GRAPH_BUILDERS

    name = graph_cfg.get("builder", "knn")
    builder_cls = GRAPH_BUILDERS.get(name)
    return builder_cls(graph_cfg).build(X, ids)
