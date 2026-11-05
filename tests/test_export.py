"""Tests for graph export: a GraphML round trip and the alignment checks."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from musicsim.graphs.base import Graph
from musicsim.graphs.export import export_graph

# Node 2 has a single neighbour (padded row); ids are not row positions.
GRAPH = Graph(
    neighbors=np.array([[1, 2], [0, 2], [0, -1], [2, 1]]),
    weights=np.array([[0.9, 0.5], [0.8, 0.4], [0.7, 0.0], [0.6, 0.3]], dtype=np.float32),
    ids=np.array([10, 20, 30, 40], dtype=np.int64),
)


def _nodes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": GRAPH.ids,
            "genre_top": ["Rock", None, "Jazz", "Rock"],
            "artist_id": np.array([1, 2, 3, 4], dtype=np.int64),
            "k_occurrence": np.array([2, 1, 3, 0], dtype=np.int64),
            "score": np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
        }
    )


def test_round_trip_keeps_nodes_edges_and_attributes(tmp_path: Path) -> None:
    export_graph(GRAPH, _nodes(), tmp_path)

    read = nx.read_graphml(tmp_path / "graph.graphml")
    assert read.is_directed()
    assert sorted(read.nodes) == ["10", "20", "30", "40"]
    assert read.nodes["10"]["genre_top"] == "Rock"
    assert read.nodes["30"]["artist_id"] == 3
    assert isinstance(read.nodes["30"]["k_occurrence"], int)
    assert read.nodes["20"]["score"] == pytest.approx(1.5)
    assert read.nodes["10"]["item_id"] == 10

    edges = GRAPH.to_edges_frame()
    assert read.number_of_edges() == len(edges) == 7  # the padding slot is not an edge
    for src, dst, weight, rank in edges.itertuples(index=False):
        data = read.edges[str(src), str(dst)]
        assert data["weight"] == pytest.approx(float(weight))
        assert data["rank"] == rank
    assert not read.has_edge("30", "20")


def test_missing_value_is_omitted_in_graphml_and_empty_in_csv(tmp_path: Path) -> None:
    export_graph(GRAPH, _nodes(), tmp_path)

    read = nx.read_graphml(tmp_path / "graph.graphml")
    assert "genre_top" not in read.nodes["20"]
    assert read.nodes["30"]["genre_top"] == "Jazz"
    csv = pd.read_csv(tmp_path / "nodes.csv")
    assert csv["genre_top"].isna().tolist() == [False, True, False, False]


def test_edges_csv_is_the_graph_edges_frame_and_nodes_csv_is_the_input(tmp_path: Path) -> None:
    export_graph(GRAPH, _nodes(), tmp_path)

    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "edges.csv"), GRAPH.to_edges_frame(), check_dtype=False
    )
    columns = ["item_id", "artist_id", "k_occurrence", "score"]
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "nodes.csv")[columns], _nodes()[columns], check_dtype=False
    )


def test_graph_with_padding_exports_every_node(tmp_path: Path) -> None:
    isolated = Graph(
        neighbors=np.full((3, 2), -1), weights=np.zeros((3, 2), np.float32), ids=np.arange(3)
    )
    nodes = pd.DataFrame({"item_id": np.arange(3), "k_occurrence": np.zeros(3, dtype=np.int64)})
    export_graph(isolated, nodes, tmp_path)

    read = nx.read_graphml(tmp_path / "graph.graphml")
    assert read.number_of_nodes() == 3
    assert read.number_of_edges() == 0


def test_misaligned_nodes_raise(tmp_path: Path) -> None:
    shuffled = _nodes().iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="same order"):
        export_graph(GRAPH, shuffled, tmp_path)
    with pytest.raises(ValueError, match="same order"):
        export_graph(GRAPH, _nodes().iloc[:3], tmp_path)
    with pytest.raises(ValueError, match="first column"):
        export_graph(GRAPH, _nodes()[["genre_top", "item_id"]], tmp_path)
    assert not (tmp_path / "graph.graphml").exists()
