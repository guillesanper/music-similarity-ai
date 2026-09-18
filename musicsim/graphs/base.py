"""The graph contract every builder implements (B.1.8): ``build_graph(X, ids,
graph_cfg) -> Graph``.

A builder is registered by name (``graph.builder`` in the configuration) in
:data:`musicsim.registry.GRAPH_BUILDERS`, exactly like an extractor or a
dataset. :data:`Graph` is the frozen result: a directed k-nearest-neighbour
graph, ``neighbors[i]`` and ``weights[i]`` being node ``i``'s ``k`` neighbours
ordered by decreasing similarity.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["Graph", "GraphBuilder", "build_graph"]


@dataclasses.dataclass(frozen=True, slots=True)
class Graph:
    """A directed k-nearest-neighbour graph over a set of items.

    ``neighbors`` and ``weights`` are ``(N, k)``: row ``i`` holds the ``k``
    neighbours of ``ids[i]``, as row positions into ``ids`` (never as
    identifiers), ordered by decreasing ``weights``. ``ids`` is ``(N,)``.
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

    @property
    def n_nodes(self) -> int:
        return self.ids.shape[0]

    @property
    def k(self) -> int:
        return self.neighbors.shape[1]

    def to_edges_frame(self) -> pd.DataFrame:
        """The ``edges.csv`` of B.1.8: ``src_id, dst_id, weight, rank`` (rank 1-based)."""
        n, k = self.neighbors.shape
        src = np.repeat(self.ids, k)
        dst = self.ids[self.neighbors.ravel()]
        rank = np.tile(np.arange(1, k + 1), n)
        return pd.DataFrame(
            {
                "src_id": src,
                "dst_id": dst,
                "weight": self.weights.ravel(),
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
