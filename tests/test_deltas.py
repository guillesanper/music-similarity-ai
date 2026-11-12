"""Tests for :mod:`musicsim.evaluation.deltas`: paired differences from ``per_query.csv``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from musicsim.evaluation.bootstrap import holm_correction, paired_bootstrap_test
from musicsim.evaluation.deltas import HEADLINE_METRICS, paired_deltas

N_ARTISTS = 40
PER_ARTIST = 5
N_ITEMS = N_ARTISTS * PER_ARTIST
FAST = {"n_resamples": 400, "seed": 7}


def _artists() -> pd.Series:
    return pd.Series(np.arange(N_ITEMS) // PER_ARTIST, index=np.arange(N_ITEMS), name="artist_id")


def _frame(values: dict[str, dict[tuple[str, int], np.ndarray]], **extra: object) -> pd.DataFrame:
    """A ``per_query.csv`` with one block per representation and ``(metric, k)``."""
    blocks = []
    for representation, by_metric in values.items():
        for (metric, k), v in by_metric.items():
            blocks.append(
                pd.DataFrame(
                    {
                        "item_id": np.arange(N_ITEMS),
                        "metric": metric,
                        "k": k,
                        "value": v,
                        "representation": representation,
                        "gallery": "fma_test",
                        "variant": "dedup",
                        "queries": "all",
                        "seed": 42,
                    }
                )
            )
    return pd.concat(blocks, ignore_index=True)


def _base_values(rng: np.random.Generator) -> dict[tuple[str, int], np.ndarray]:
    return {key: rng.uniform(0.2, 0.8, size=N_ITEMS) for key in HEADLINE_METRICS}


def test_delta_of_a_representation_with_itself_is_exactly_zero() -> None:
    base = _base_values(np.random.default_rng(0))
    frame = _frame({"mfcc": base, "same": base})
    table = paired_deltas(
        frame, _artists(), candidates=["same"], reference="mfcc", gallery="fma_test", **FAST
    )
    assert len(table) == 3
    assert (table["delta"] == 0.0).all()
    assert (table["ci_low"] == 0.0).all()
    assert (table["ci_high"] == 0.0).all()
    assert (table["p_value"] == 1.0).all()
    assert not table["reject"].any()


def test_planted_difference_is_the_mean_gap_and_inside_its_interval() -> None:
    rng = np.random.default_rng(1)
    base = _base_values(rng)
    better = {key: v + 0.05 + rng.normal(scale=0.02, size=N_ITEMS) for key, v in base.items()}
    frame = _frame({"mfcc": base, "mert": better})
    table = paired_deltas(
        frame, _artists(), candidates=["mert"], reference="mfcc", gallery="fma_test", **FAST
    )
    for row in table.itertuples():
        expected = better[(row.metric, row.k)].mean() - base[(row.metric, row.k)].mean()
        assert row.delta == pytest.approx(expected, abs=1e-12)
        assert row.ci_low < 0.05 < row.ci_high
        assert row.ci_low > 0  # excludes zero
        assert row.p_holm < 0.05
        assert row.reject
        assert (row.n_queries, row.n_artists) == (N_ITEMS, N_ARTISTS)


def test_result_does_not_depend_on_row_order() -> None:
    rng = np.random.default_rng(2)
    base = _base_values(rng)
    other = {key: v + rng.normal(scale=0.05, size=N_ITEMS) for key, v in base.items()}
    frame = _frame({"mfcc": base, "mert": other})
    shuffled = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    kwargs = {"candidates": ["mert"], "reference": "mfcc", "gallery": "fma_test", **FAST}
    pd.testing.assert_frame_equal(
        paired_deltas(frame, _artists(), **kwargs), paired_deltas(shuffled, _artists(), **kwargs)
    )


def test_resampling_is_by_artist_not_by_query() -> None:
    # The gap is constant within an artist and varies across artists, so queries
    # of one artist are perfectly dependent: the clustered interval must be
    # wider than the one that wrongly treats the 200 queries as independent.
    rng = np.random.default_rng(3)
    base = {("R", 10): rng.uniform(0.3, 0.6, size=N_ITEMS)}
    gap = np.repeat(rng.normal(scale=0.1, size=N_ARTISTS), PER_ARTIST)
    frame = _frame({"mfcc": base, "mert": {("R", 10): base[("R", 10)] + gap}})
    clustered = paired_deltas(
        frame,
        _artists(),
        candidates=["mert"],
        reference="mfcc",
        gallery="fma_test",
        metrics=[("R", 10)],
        **FAST,
    ).iloc[0]
    naive = paired_bootstrap_test(base[("R", 10)] + gap, base[("R", 10)], n_resamples=400, seed=7)
    assert clustered.ci_high - clustered.ci_low > 1.5 * (naive.ci_high - naive.ci_low)


def test_holm_runs_over_every_candidate_and_metric_of_the_call() -> None:
    rng = np.random.default_rng(4)
    base = _base_values(rng)
    a = {key: v + rng.normal(scale=0.05, size=N_ITEMS) for key, v in base.items()}
    b = {key: v + 0.1 + rng.normal(scale=0.03, size=N_ITEMS) for key, v in base.items()}
    frame = _frame({"mfcc": base, "a": a, "b": b})
    table = paired_deltas(
        frame, _artists(), candidates=["a", "b"], reference="mfcc", gallery="fma_test", **FAST
    )
    assert len(table) == 6
    assert list(table["candidate"]) == ["a"] * 3 + ["b"] * 3
    np.testing.assert_array_equal(
        table["p_holm"].to_numpy(), holm_correction(table["p_value"].to_numpy()).adjusted_pvalues
    )
    assert (table["p_holm"] >= table["p_value"]).all()


def test_all_candidates_share_resamples_so_a_copy_has_identical_interval() -> None:
    rng = np.random.default_rng(5)
    base = _base_values(rng)
    other = {key: v + rng.normal(scale=0.05, size=N_ITEMS) for key, v in base.items()}
    frame = _frame({"mfcc": base, "x": other, "y": other})
    table = paired_deltas(
        frame, _artists(), candidates=["x", "y"], reference="mfcc", gallery="fma_test", **FAST
    )
    x, y = table[table["candidate"] == "x"], table[table["candidate"] == "y"]
    np.testing.assert_array_equal(
        x[["delta", "ci_low", "ci_high", "p_value"]].to_numpy(),
        y[["delta", "ci_low", "ci_high", "p_value"]].to_numpy(),
    )


def test_filters_select_gallery_variant_and_queries() -> None:
    rng = np.random.default_rng(6)
    base = _base_values(rng)
    other = {key: v + 0.1 for key, v in base.items()}
    frame = _frame({"mfcc": base, "mert": other})
    elsewhere = frame.assign(gallery="fma_all", value=0.0)
    mixed = pd.concat([frame, elsewhere], ignore_index=True)
    table = paired_deltas(
        mixed, _artists(), candidates=["mert"], reference="mfcc", gallery="fma_test", **FAST
    )
    assert table["delta"].to_numpy() == pytest.approx(0.1)
    with pytest.raises(ValueError, match="no rows"):
        paired_deltas(
            frame,
            _artists(),
            candidates=["mert"],
            reference="mfcc",
            gallery="fma_test",
            variant="raw",
            **FAST,
        )


def test_misuse_is_reported_with_the_received_and_expected_values() -> None:
    base = _base_values(np.random.default_rng(8))
    frame = _frame({"mfcc": base, "mert": base})
    kwargs = {"reference": "mfcc", "gallery": "fma_test", **FAST}

    with pytest.raises(ValueError, match="no rows for representation 'ghost'"):
        paired_deltas(frame, _artists(), candidates=["ghost"], **kwargs)
    with pytest.raises(ValueError, match="also a candidate"):
        paired_deltas(frame, _artists(), candidates=["mfcc"], **kwargs)
    with pytest.raises(ValueError, match="candidates is empty"):
        paired_deltas(frame, _artists(), candidates=[], **kwargs)

    partial = frame.drop(frame[(frame["representation"] == "mert") & (frame["item_id"] < 10)].index)
    with pytest.raises(ValueError, match="different queries"):
        paired_deltas(partial, _artists(), candidates=["mert"], **kwargs)

    twice = pd.concat([frame, frame[frame["representation"] == "mert"].assign(seed=43)])
    with pytest.raises(ValueError, match="several rows for the same item_id"):
        paired_deltas(twice, _artists(), candidates=["mert"], **kwargs)

    with pytest.raises(ValueError, match="no artist"):
        paired_deltas(frame, _artists().iloc[:-3], candidates=["mert"], **kwargs)
