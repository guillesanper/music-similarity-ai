"""Tests for homophily, reciprocity, clustering and components, on graphs with
hand-computable answers and against dense / networkx references."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy import sparse

from musicsim.evaluation.retrieval import compute_per_query_metrics
from musicsim.evaluation.structure import (
    component_summary,
    label_homophily,
    mean_clustering,
    reciprocity,
)
from musicsim.graphs.adjacency import to_csr
from musicsim.graphs.base import PADDING, Graph, build_graph


def _graph(rows: list[list[int]], width: int | None = None) -> Graph:
    """A graph from neighbour lists, padded to ``width`` (default: the longest row)."""
    width = width or max(len(r) for r in rows)
    nb = np.array([r + [PADDING] * (width - len(r)) for r in rows], dtype=np.int64)
    return Graph(
        neighbors=nb,
        weights=np.where(nb == PADDING, 0.0, 0.5).astype(np.float32),
        ids=np.arange(len(rows)),
    )


def _cliques(bridge: bool) -> Graph:
    """Two 4-cliques (nodes 0-3 and 4-7), optionally joined by the edge 3 <-> 4."""
    rows = [
        [j for j in range(start, start + 4) if j != i]
        for start in (0, 4)
        for i in range(start, start + 4)
    ]
    if bridge:
        rows[3].append(4)
        rows[4].append(3)
    return _graph(rows)


CLIQUE_LABELS = np.array([0] * 4 + [1] * 4)

# 0 -> 1, 1 -> 0 (mutual); 2 -> 0, 3 -> 0 (not reciprocated).
STAR = _graph([[1], [0], [0], [0]])
EMPTY = _graph([[], [], []], width=2)


def test_bridged_cliques_by_hand() -> None:
    graph = _cliques(bridge=True)
    union = to_csr(graph, symmetrize="union")
    # 24 edges inside cliques and the two directions of the bridge, which crosses labels.
    assert label_homophily(graph, CLIQUE_LABELS) == pytest.approx(24 / 26)
    assert reciprocity(graph) == 1.0
    # Nodes 0-2 and 5-7: d=3, 3 triangles -> 1; nodes 3 and 4: d=4, 3 triangles -> 1/2.
    assert mean_clustering(union) == pytest.approx(0.875)
    assert component_summary(union) == {"n_components": 1.0, "giant_fraction": 1.0}


def test_cliques_without_bridge_split_in_two() -> None:
    graph = _cliques(bridge=False)
    union = to_csr(graph, symmetrize="union")
    assert label_homophily(graph, CLIQUE_LABELS) == 1.0
    assert mean_clustering(union) == 1.0
    assert component_summary(union) == {"n_components": 2.0, "giant_fraction": 0.5}


def test_reciprocity_counts_mutual_edges() -> None:
    assert reciprocity(STAR) == pytest.approx(0.5)


def test_label_homophily_is_the_share_of_same_label_edges() -> None:
    labels = np.array([0, 0, 1, 0])  # edges: 0-1 same, 1-0 same, 2-0 diff, 3-0 same
    assert label_homophily(STAR, labels) == pytest.approx(0.75)


def test_label_homophily_rejects_misaligned_labels() -> None:
    with pytest.raises(ValueError, match="one entry per node"):
        label_homophily(STAR, np.array([0, 1]))


def test_edgeless_graph_is_an_error() -> None:
    with pytest.raises(ValueError, match="3 nodes has no edges"):
        label_homophily(EMPTY, np.zeros(3, dtype=int))
    with pytest.raises(ValueError, match="3 nodes has no edges"):
        reciprocity(EMPTY)


def test_isolated_node_is_its_own_component() -> None:
    # {0, 1, 3, 4} are connected, node 2 has no neighbour and nobody points at it.
    graph = _graph([[1, 4], [0], [], [4, 0], [3]], width=2)
    summary = component_summary(to_csr(graph, symmetrize="union"))
    assert summary == {"n_components": 2.0, "giant_fraction": 0.8}
    assert component_summary(to_csr(EMPTY, symmetrize="union")) == {
        "n_components": 3.0,
        "giant_fraction": pytest.approx(1 / 3),
    }


def test_clustering_of_a_path_and_a_triangle() -> None:
    path = to_csr(_graph([[1], [0, 2], [1]]), symmetrize="union")
    triangle = to_csr(_graph([[1, 2], [0, 2], [0, 1]]), symmetrize="union")
    assert mean_clustering(path) == 0.0  # no node with two connected neighbours
    assert mean_clustering(triangle) == 1.0


def test_clustering_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        mean_clustering(sparse.csr_matrix(np.ones((2, 3))))


def _random_padded_graph(rng: np.random.Generator) -> Graph:
    n, k = int(rng.integers(5, 14)), int(rng.integers(1, 5))
    while True:
        rows = []
        for i in range(n):
            degree = int(rng.integers(0, k + 1))
            rows.append(rng.choice(np.delete(np.arange(n), i), size=degree, replace=False).tolist())
        if any(rows):
            return _graph(rows, width=k)


def test_matches_dense_and_networkx_references(rng: np.random.Generator) -> None:
    for _ in range(300):
        graph = _random_padded_graph(rng)
        n = graph.n_nodes
        labels = rng.integers(0, 3, n)

        A = np.zeros((n, n))
        for i, row in enumerate(graph.neighbors):
            A[i, row[row != PADDING]] = 1.0
        S = np.maximum(A, A.T)

        same = labels[:, None] == labels[None, :]
        assert label_homophily(graph, labels) == pytest.approx((A * same).sum() / A.sum())
        assert reciprocity(graph) == pytest.approx((A * A.T).sum() / A.sum())

        union = to_csr(graph, symmetrize="union")
        degree = S.sum(axis=1)
        triangles = np.diag(S @ S @ S) / 2
        local = np.divide(2 * triangles, degree * (degree - 1), out=np.zeros(n), where=degree > 1)
        assert mean_clustering(union) == pytest.approx(local.mean())

        reference = nx.from_numpy_array(S)
        assert mean_clustering(union) == pytest.approx(nx.average_clustering(reference))
        components = list(nx.connected_components(reference))
        assert component_summary(union) == {
            "n_components": float(len(components)),
            "giant_fraction": max(map(len, components)) / n,
        }


def test_homophily_equals_mean_precision_at_k(
    synthetic_embeddings: tuple[np.ndarray, np.ndarray], synthetic_index
) -> None:
    X, ids = synthetic_embeddings
    k = 10
    genre = synthetic_index["label"].astype("category").cat.codes.to_numpy()
    graph = build_graph(X, ids, {"builder": "knn", "k": k, "exclude_self": True})

    # Every row is a query in its own exclusion group: only the query itself is left out.
    per_query = compute_per_query_metrics(
        X, ids, np.arange(len(ids)), np.arange(len(ids)), genre, [k]
    )
    precision = per_query[(per_query["metric"] == "P") & (per_query["k"] == k)]["value"]
    assert len(precision) == len(ids)
    assert label_homophily(graph, genre) == pytest.approx(precision.mean(), abs=1e-12)
