"""Similarity graphs over embeddings: kNN builders and near-duplicate detection.

Importing this package registers every graph builder (``knn``) in
:data:`musicsim.registry.GRAPH_BUILDERS`, which is what
:func:`musicsim.registry.load_plugins` relies on.
"""

from __future__ import annotations

from musicsim.graphs import knn  # noqa: F401  (registers "knn")
from musicsim.graphs.adjacency import to_csr
from musicsim.graphs.base import Graph, GraphBuilder, build_graph
from musicsim.graphs.duplicates import NEAR_DUP_SIM, duplicate_groups, near_duplicate_pairs
from musicsim.graphs.gallery import GalleryNodes, gallery_nodes, graph_dir

__all__ = [
    "NEAR_DUP_SIM",
    "GalleryNodes",
    "Graph",
    "GraphBuilder",
    "build_graph",
    "duplicate_groups",
    "gallery_nodes",
    "graph_dir",
    "near_duplicate_pairs",
    "to_csr",
]
