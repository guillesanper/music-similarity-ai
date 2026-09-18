"""Tests for :mod:`musicsim.graphs.duplicates`: near-duplicate pairs and groups."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from musicsim.graphs.duplicates import NEAR_DUP_SIM, duplicate_groups, near_duplicate_pairs


def _unit(vectors: list[tuple[float, float, float]]) -> np.ndarray:
    X = np.asarray(vectors, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_near_duplicate_pairs_finds_only_the_planted_pair() -> None:
    # Items 0/1 are an exact duplicate; every other pair is well below the
    # threshold (see the module docstring's angle math: 0.9999 is ~0.8 degrees).
    X = _unit(
        [
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),  # duplicate of item 0
            (1.0, 0.0, 3.0),
            (1.0, 0.0, 4.0),
            (0.0, 1.0, 7.0),
        ]
    )
    ids = np.array([10, 20, 30, 40, 50])

    pairs = near_duplicate_pairs(X, ids)

    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert (row["id_a"], row["id_b"]) == (10, 20)
    assert row["similarity"] >= NEAR_DUP_SIM


def test_near_duplicate_pairs_is_empty_without_duplicates() -> None:
    X = _unit([(1.0, 0.0, 3.0), (1.0, 0.0, 4.0), (0.0, 1.0, 7.0)])
    ids = np.array([1, 2, 3])
    pairs = near_duplicate_pairs(X, ids)
    assert len(pairs) == 0
    assert list(pairs.columns) == ["id_a", "id_b", "similarity"]


def test_near_duplicate_pairs_is_independent_of_block_rows() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(23, 4)).astype(np.float32)
    X[5] = X[2]  # plant one exact duplicate
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ids = np.arange(23) + 100

    whole = near_duplicate_pairs(X, ids, block_rows=1000)
    blocked = near_duplicate_pairs(X, ids, block_rows=4)
    pd.testing.assert_frame_equal(whole, blocked)
    assert (whole["id_a"] == 102).any() and (whole["id_b"] == 105).any()


def test_duplicate_groups_unions_transitively() -> None:
    # A~B and B~C: all three land in one group, D stays a singleton.
    ids = np.array([1, 2, 3, 4])
    pairs = pd.DataFrame({"id_a": [1, 2], "id_b": [2, 3]})

    groups = duplicate_groups(ids, pairs)

    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
    assert len(set(groups.tolist())) == 2


def test_duplicate_groups_without_pairs_is_all_singletons() -> None:
    ids = np.array([5, 6, 7])
    groups = duplicate_groups(ids, pd.DataFrame({"id_a": [], "id_b": []}))
    assert len(set(groups.tolist())) == 3


def test_duplicate_groups_rejects_an_id_outside_ids() -> None:
    ids = np.array([1, 2, 3])
    pairs = pd.DataFrame({"id_a": [1], "id_b": [999]})
    with pytest.raises(ValueError, match="999"):
        duplicate_groups(ids, pairs)


def test_duplicate_groups_accepts_a_plain_array_of_pairs() -> None:
    ids = np.array([1, 2, 3])
    groups = duplicate_groups(ids, np.array([[1, 2]]))
    assert groups[0] == groups[1] != groups[2]
