"""Tests for paired_query_deltas: known differences, exact zeros, shared resample indices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from musicsim.evaluation.deltas import paired_query_deltas

N_ITEMS = 40
ARTISTS = pd.Series(np.arange(N_ITEMS) // 4, index=np.arange(N_ITEMS))  # 10 artists of 4 items
KW = {"n_resamples": 200, "confidence": 0.95, "seed": 7}


def _table(**sides: np.ndarray) -> pd.DataFrame:
    frames = [
        pd.DataFrame(
            {
                "item_id": np.arange(N_ITEMS),
                "variant": name,
                "metric": "R",
                "value": values,
            }
        )
        for name, values in sides.items()
    ]
    return pd.concat(frames, ignore_index=True)


def _deltas(per_query: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return paired_query_deltas(
        per_query, by="variant", pairs=pairs, within=["metric"], artist_by_item=ARTISTS, **KW
    )


def test_difference_of_a_side_with_itself_is_exactly_zero() -> None:
    a = np.random.default_rng(0).uniform(size=N_ITEMS)
    out = _deltas(_table(a=a), [("a", "a")])
    row = out.iloc[0]
    assert (row["delta"], row["ci_low"], row["ci_high"]) == (0.0, 0.0, 0.0)
    assert (row["n_queries"], row["n_artists"]) == (N_ITEMS, 10)


def test_constant_difference_gives_that_difference_and_a_degenerate_interval() -> None:
    b = np.random.default_rng(1).uniform(size=N_ITEMS)
    out = _deltas(_table(a=b + 0.1, b=b), [("a", "b")])
    row = out.iloc[0]
    assert row["delta"] == pytest.approx(0.1)
    assert row["ci_low"] == pytest.approx(0.1)
    assert row["ci_high"] == pytest.approx(0.1)


def test_queries_that_do_not_match_raise() -> None:
    table = _table(a=np.zeros(N_ITEMS), b=np.zeros(N_ITEMS))
    table = table.drop(table.index[(table["variant"] == "b") & (table["item_id"] == 3)])
    with pytest.raises(ValueError, match="do not have the same queries"):
        _deltas(table, [("a", "b")])


def test_unknown_side_and_missing_column_raise() -> None:
    table = _table(a=np.zeros(N_ITEMS))
    with pytest.raises(ValueError, match="'z' is not in the table"):
        _deltas(table, [("a", "z")])
    with pytest.raises(ValueError, match="lacks column"):
        paired_query_deltas(
            table.drop(columns="metric"), by="variant", pairs=[("a", "a")], within=["metric"],
            artist_by_item=ARTISTS, **KW,
        )  # fmt: skip


def test_pairs_of_one_combination_share_resample_indices() -> None:
    """c = a + 0.1 exactly, so with the same artists resampled every interval shifts by 0.1."""
    rng = np.random.default_rng(2)
    a, b = rng.uniform(size=N_ITEMS), rng.uniform(size=N_ITEMS)
    out = _deltas(_table(a=a, b=b, c=a + 0.1), [("a", "b"), ("c", "b")]).set_index("a")
    for column in ("delta", "ci_low", "ci_high"):
        assert out.loc["c", column] - out.loc["a", column] == pytest.approx(0.1)


def test_seed_depends_on_within_values_not_on_the_pair_and_is_reproducible() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.uniform(size=N_ITEMS), rng.uniform(size=N_ITEMS)
    table = _table(a=a, b=b)
    first = _deltas(table, [("a", "b")])
    again = _deltas(table, [("a", "b"), ("b", "a")])
    pd.testing.assert_frame_equal(first, again.iloc[[0]].reset_index(drop=True))
    # The reversed pair is the exact mirror: same artists resampled.
    assert again.iloc[1]["delta"] == pytest.approx(-again.iloc[0]["delta"])
    assert again.iloc[1]["ci_low"] == pytest.approx(-again.iloc[0]["ci_high"])


def test_several_within_combinations_and_integer_columns() -> None:
    frames = []
    for k in (1, 10):
        t = _table(a=np.full(N_ITEMS, 0.5), b=np.full(N_ITEMS, 0.25))
        t["k"] = k
        frames.append(t)
    out = paired_query_deltas(
        pd.concat(frames, ignore_index=True), by="variant", pairs=[("a", "b")],
        within=["metric", "k"], artist_by_item=ARTISTS, **KW,
    )  # fmt: skip
    assert out["k"].tolist() == [1, 10]
    assert out["delta"].tolist() == [0.25, 0.25]
    assert list(out.columns) == [
        "metric", "k", "a", "b", "delta", "ci_low", "ci_high", "n_queries", "n_artists",
    ]  # fmt: skip
