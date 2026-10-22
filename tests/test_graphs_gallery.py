"""Gallery nodes: which rows a graph is built from, and where it is written."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from musicsim.graphs.gallery import gallery_nodes, graph_dir
from musicsim.representations import AlignedRepresentations

THRESHOLD = 0.9999


def _aligned(
    index: pd.DataFrame, X: np.ndarray, ids: np.ndarray, *, plant: tuple[int, int] | None = None
) -> AlignedRepresentations:
    X = X.copy()
    if plant is not None:  # make row `plant[1]` an exact copy of row `plant[0]`
        X[plant[1]] = X[plant[0]]
    return AlignedRepresentations(
        X={"ref": X, "other": X[:, :2]},
        ids=ids,
        index=index.set_index("item_id").loc[ids],
        seeds={"ref": 0, "other": 0},
    )


def test_test_gallery_nodes_are_all_test_items(synthetic_index, synthetic_embeddings):
    X, ids = synthetic_embeddings
    aligned = _aligned(synthetic_index, X, ids)

    nodes = gallery_nodes(aligned, "fma_test", reference="ref", threshold=THRESHOLD)

    expected = synthetic_index.loc[synthetic_index["split"] == "test", "item_id"].to_numpy()
    assert nodes.ids.tolist() == expected.tolist()
    assert nodes.rows.tolist() == expected.tolist()  # ids are 0..N-1, so rows equal ids
    assert nodes.n_collapsed == 0


def test_planted_duplicate_keeps_the_smallest_id(synthetic_index, synthetic_embeddings):
    X, ids = synthetic_embeddings
    # Items 26 and 57 are both test items; 57 becomes an exact copy of 26.
    aligned = _aligned(synthetic_index, X, ids, plant=(26, 57))

    nodes = gallery_nodes(aligned, "fma_test", reference="ref", threshold=THRESHOLD)

    full = gallery_nodes(
        aligned, "fma_test", reference="ref", threshold=THRESHOLD, collapse_duplicates=False
    )
    assert nodes.n_collapsed == 1
    assert 26 in nodes.ids and 57 not in nodes.ids
    assert nodes.ids.tolist() == [i for i in full.ids.tolist() if i != 57]
    assert nodes.rows.tolist() == nodes.ids.tolist()


def test_planted_duplicate_keeps_the_smaller_id_even_when_listed_second(
    synthetic_index, synthetic_embeddings
):
    X, ids = synthetic_embeddings
    # The copy now goes from the larger id onto the smaller one's row.
    aligned = _aligned(synthetic_index, X, ids, plant=(57, 26))

    nodes = gallery_nodes(aligned, "fma_test", reference="ref", threshold=THRESHOLD)

    assert nodes.n_collapsed == 1
    assert 26 in nodes.ids and 57 not in nodes.ids


def test_without_collapse_no_item_is_lost(synthetic_index, synthetic_embeddings):
    X, ids = synthetic_embeddings
    aligned = _aligned(synthetic_index, X, ids, plant=(26, 57))

    nodes = gallery_nodes(
        aligned, "fma_test", reference="ref", threshold=THRESHOLD, collapse_duplicates=False
    )

    n_test = int((synthetic_index["split"] == "test").sum())
    assert len(nodes.ids) == n_test
    assert 26 in nodes.ids and 57 in nodes.ids
    assert nodes.n_collapsed == 0


def test_duplicate_outside_the_gallery_is_ignored(synthetic_index, synthetic_embeddings):
    X, ids = synthetic_embeddings
    # Item 0 is a training item: a copy of a test item does not touch the test nodes.
    aligned = _aligned(synthetic_index, X, ids, plant=(26, 0))

    nodes = gallery_nodes(aligned, "fma_test", reference="ref", threshold=THRESHOLD)

    assert nodes.n_collapsed == 0


def test_unknown_reference_raises(synthetic_index, synthetic_embeddings):
    X, ids = synthetic_embeddings
    aligned = _aligned(synthetic_index, X, ids)

    with pytest.raises(ValueError, match="'nope'"):
        gallery_nodes(aligned, "fma_test", reference="nope", threshold=THRESHOLD)


def test_empty_gallery_raises(synthetic_index, synthetic_embeddings):
    X, ids = synthetic_embeddings
    keep = synthetic_index["split"].to_numpy() != "test"
    aligned = _aligned(synthetic_index[keep], X[keep], ids[keep])

    with pytest.raises(ValueError, match="fma_test"):
        gallery_nodes(aligned, "fma_test", reference="ref", threshold=THRESHOLD)


def test_graph_dir_layout_and_no_side_effect(tmp_path: Path):
    path = graph_dir(tmp_path, "mfcc", "fma_test", "knn_k10")

    assert path == tmp_path / "graphs" / "mfcc" / "fma_test" / "knn_k10"
    assert not (tmp_path / "graphs").exists()
