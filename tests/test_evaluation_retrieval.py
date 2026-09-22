"""Tests for :mod:`musicsim.evaluation.retrieval`'s pure metric functions,
checked against a case worked out by hand (see the module docstring for the
full-ranking MAP argument this relies on).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from musicsim.evaluation.retrieval import compute_per_query_metrics, query_rows


def _unit_row(angle_degrees: float) -> tuple[float, float]:
    angle = np.radians(angle_degrees)
    return (float(np.cos(angle)), float(np.sin(angle)))


@pytest.fixture
def hand_worked_case() -> dict:
    # Item 0 (query, genre A) at angle 0; items 1..5 at increasing angles, so
    # similarity to item 0 strictly decreases in that order: 1 > 2 > 3 > 4 > 5.
    X = np.asarray(
        [_unit_row(a) for a in (0, 10, 20, 30, 40, 50)], dtype=np.float32
    )
    ids = np.arange(6)
    genre = np.array([0, 0, 1, 0, 1, 1])  # A, A, B, A, B, B
    group = np.arange(6)  # "raw" variant: only the query itself is excluded
    return {"X": X, "ids": ids, "genre": genre, "group": group}


def test_compute_per_query_metrics_matches_the_hand_worked_case(hand_worked_case: dict) -> None:
    df = compute_per_query_metrics(
        hand_worked_case["X"],
        hand_worked_case["ids"],
        rows=np.array([0]),
        group=hand_worked_case["group"],
        genre=hand_worked_case["genre"],
        ks=[1, 3],
    )
    value = {(row.metric, row.k): row.value for row in df.itertuples()}

    # Ranking (by decreasing similarity): 1(A), 2(B), 3(A), 4(B), 5(B).
    # rel@3 = [1, 0, 1]; n_relevant = 2 (items 1 and 3, item 0's own genre A).
    assert value[("P", 1)] == pytest.approx(1.0)
    assert value[("P", 3)] == pytest.approx(2 / 3)
    assert value[("R", 1)] == pytest.approx(0.5)
    assert value[("R", 3)] == pytest.approx(1.0)
    assert value[("MAP", 0)] == pytest.approx(5 / 6)  # (1/1 + 2/3) / 2

    disc = 1.0 / np.log2(np.arange(2, 5))  # ranks 1, 2, 3
    expected_ndcg3 = (disc[0] + disc[2]) / (disc[0] + disc[1])
    assert value[("nDCG", 3)] == pytest.approx(expected_ndcg3)


def test_every_query_gets_every_metric_and_k(hand_worked_case: dict) -> None:
    rows = np.array([0, 1])
    df = compute_per_query_metrics(
        hand_worked_case["X"],
        hand_worked_case["ids"],
        rows=rows,
        group=hand_worked_case["group"],
        genre=hand_worked_case["genre"],
        ks=[1, 3],
    )
    # 2 queries * (3 metrics * 2 ks + 1 MAP row) = 14 rows.
    assert len(df) == 2 * (3 * 2 + 1)
    assert set(df["item_id"]) == {0, 1}


def test_compute_per_query_metrics_excludes_the_whole_group() -> None:
    # Items 0 and 1 are near-duplicates (same group); item 1 must vanish from
    # both the ranking and the relevant count for query 0.
    X = np.asarray([_unit_row(a) for a in (0, 0, 20, 30)], dtype=np.float32)
    ids = np.arange(4)
    genre = np.array([0, 0, 0, 1])  # items 0, 1, 2 share a genre with the query
    group = np.array([0, 0, 1, 2])  # 0 and 1 share a duplicate group

    df = compute_per_query_metrics(X, ids, rows=np.array([0]), group=group, genre=genre, ks=[1])
    p_at_1 = df.loc[(df["metric"] == "P") & (df["k"] == 1), "value"].iloc[0]
    # With item 1 excluded, item 2 (same genre) is the top real candidate.
    assert p_at_1 == pytest.approx(1.0)


def test_compute_per_query_metrics_rejects_a_query_with_no_relevant_candidate() -> None:
    X = np.asarray([_unit_row(a) for a in (0, 20, 40)], dtype=np.float32)
    ids = np.arange(3)
    genre = np.array([0, 1, 1])  # query's genre (0) appears nowhere else
    group = np.arange(3)

    with pytest.raises(ValueError, match="no relevant"):
        compute_per_query_metrics(X, ids, rows=np.array([0]), group=group, genre=genre, ks=[1])


def test_query_rows_all_returns_every_row() -> None:
    index = pd.DataFrame({"split": ["training", "test", "test"]})
    rows = query_rows(np.array([1, 2, 3]), index, "all")
    np.testing.assert_array_equal(rows, [0, 1, 2])


def test_query_rows_test_filters_by_split() -> None:
    index = pd.DataFrame({"split": ["training", "test", "test"]})
    rows = query_rows(np.array([1, 2, 3]), index, "test")
    np.testing.assert_array_equal(rows, [1, 2])


def test_query_rows_rejects_an_unknown_mode() -> None:
    index = pd.DataFrame({"split": ["training"]})
    with pytest.raises(ValueError, match="'all' or 'test'"):
        query_rows(np.array([1]), index, "validation")
