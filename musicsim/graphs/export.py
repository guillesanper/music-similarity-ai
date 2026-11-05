"""Export of a graph with its node attributes, in formats readable outside the project.

:func:`export_graph` writes three files: ``nodes.csv`` (one row per node),
``edges.csv`` (the ``src_id, dst_id, weight, rank`` table of
:meth:`musicsim.graphs.base.Graph.to_edges_frame`) and ``graph.graphml`` (a
directed GraphML, openable in Gephi, Cytoscape or networkx). ``Graph.write``
already writes the same ``edges.csv``; the export rewrites it with identical
content rather than assuming the caller ran ``write`` first, so the directory it
produces is complete on its own. The cost is negligible next to building the
graph.

GraphML cannot store a missing value, so a node attribute that is null (``NaN``
or ``None``, e.g. a track without ``genre_top``) is *omitted* from that node
rather than written as an empty string or a sentinel: an empty string would be
indistinguishable from a real label when the file is read back, and a sentinel
number would silently join a numeric column. ``nodes.csv`` keeps the empty cell.
Numpy scalars are converted to native Python types, which is what the GraphML
writer accepts. Node ids in the GraphML are the ``item_id`` as text.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from musicsim.graphs.base import Graph

__all__ = ["export_graph"]


def _native(value: object) -> object:
    """A numpy scalar as the equivalent Python scalar; anything else unchanged."""
    return value.item() if isinstance(value, np.generic) else value


def export_graph(graph: Graph, nodes: pd.DataFrame, directory: Path) -> None:
    """Write ``nodes.csv``, ``edges.csv`` and ``graph.graphml`` under ``directory``.

    ``nodes`` has one row per node, in the order of ``graph.ids``, with
    ``item_id`` as its first column; every column becomes a node attribute.
    Raises ``ValueError`` if ``nodes`` does not line up with ``graph.ids``.
    """
    if len(nodes.columns) == 0 or nodes.columns[0] != "item_id":
        first = nodes.columns[0] if len(nodes.columns) else None
        raise ValueError(f"nodes must have 'item_id' as its first column, got {first!r}")
    if len(nodes) != graph.n_nodes or not np.array_equal(nodes["item_id"].to_numpy(), graph.ids):
        raise ValueError(
            f"nodes ({len(nodes)} rows) must list the {graph.n_nodes} graph ids in the same order"
        )

    directory.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(directory / "nodes.csv", index=False)
    edges = graph.to_edges_frame()
    edges.to_csv(directory / "edges.csv", index=False)

    digraph = nx.DiGraph()
    for record in nodes.to_dict("records"):
        attributes = {key: _native(value) for key, value in record.items() if not pd.isna(value)}
        digraph.add_node(str(attributes["item_id"]), **attributes)
    for src, dst, weight, rank in edges.itertuples(index=False):
        digraph.add_edge(str(_native(src)), str(_native(dst)), weight=float(weight), rank=int(rank))
    nx.write_graphml(digraph, directory / "graph.graphml")
