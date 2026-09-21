"""Tests for :mod:`musicsim.evaluation.bootstrap`: the C.8 protocol primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from musicsim.evaluation.bootstrap import (
    bootstrap_ci,
    bootstrap_means,
    holm_correction,
    paired_bootstrap_ci,
    seed_variability_table,
)


def test_bootstrap_ci_of_a_constant_is_a_point() -> None:
    result = bootstrap_ci(np.full(50, 0.7), n_resamples=200, seed=0)
    assert result.mean == pytest.approx(0.7)
    assert result.ci_low == pytest.approx(0.7)
    assert result.ci_high == pytest.approx(0.7)


def test_bootstrap_ci_is_reproducible_with_the_same_seed() -> None:
    values = np.linspace(0.0, 1.0, 40)
    first = bootstrap_ci(values, n_resamples=500, seed=42)
    second = bootstrap_ci(values, n_resamples=500, seed=42)
    assert first == second


def test_bootstrap_ci_brackets_the_mean_for_varied_values() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(loc=0.5, scale=0.1, size=200)
    result = bootstrap_ci(values, n_resamples=2000, seed=1)
    assert result.ci_low < result.mean < result.ci_high


def test_clustered_bootstrap_weights_by_cluster_not_by_row() -> None:
    # One cluster of 99 rows at 0.0 and one cluster of 1 row at 1.0. Resampling
    # individual rows draws the singleton with probability ~1/100 per row, so
    # its contribution barely registers. Resampling the 2 *clusters* with
    # replacement draws that same singleton with probability 1/2 per pick:
    # with 2 picks, both land on it 1/4 of the time (mean 1.0), one does 1/2 of
    # the time (mean ~0.01), and neither does 1/4 of the time (mean 0) — an
    # expectation of 0.25*1.0 + 0.5*0.01 ~= 0.255, an order of magnitude above
    # the row-level bootstrap's.
    values = np.concatenate([np.zeros(99), np.ones(1)])
    groups = np.concatenate([np.zeros(99), np.ones(1)])

    row_means = bootstrap_means(values, n_resamples=2000, seed=0)
    cluster_means = bootstrap_means(values, n_resamples=2000, seed=0, groups=groups)

    assert row_means.mean() < 0.05
    assert cluster_means.mean() == pytest.approx(0.255, abs=0.05)


def test_paired_bootstrap_ci_of_identical_arrays_is_zero() -> None:
    values = np.linspace(0.0, 1.0, 30)
    result = paired_bootstrap_ci(values, values, n_resamples=300, seed=3)
    assert result.mean == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0


def test_paired_bootstrap_ci_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="row-aligned"):
        paired_bootstrap_ci(np.zeros(5), np.zeros(6))


def test_holm_correction_matches_a_hand_worked_example() -> None:
    # p = [0.01, 0.02, 0.03, 0.04, 0.5], alpha = 0.05 (already sorted ascending).
    # Multipliers (m - rank): 5, 4, 3, 2, 1.
    # Step-down: 0.01 <= 0.05/5 (reject), 0.02 <= 0.05/4=0.0125? no -> stop.
    result = holm_correction([0.01, 0.02, 0.03, 0.04, 0.5], alpha=0.05)
    assert result.reject.tolist() == [True, False, False, False, False]
    np.testing.assert_allclose(result.adjusted_pvalues, [0.05, 0.08, 0.09, 0.09, 0.5])


def test_holm_correction_is_less_conservative_than_bonferroni() -> None:
    # A p-value that a flat Bonferroni (alpha / m) would reject at rank > 1
    # can still be rejected by Holm once earlier hypotheses are rejected.
    result = holm_correction([0.001, 0.001, 0.02], alpha=0.05)
    assert result.reject.tolist() == [True, True, True]


def test_holm_correction_rejects_out_of_range_pvalues() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_correction([0.1, 1.5])


def test_holm_correction_handles_unsorted_input_by_original_order() -> None:
    ascending = holm_correction([0.01, 0.02, 0.03, 0.04, 0.5], alpha=0.05)
    # Same family, shuffled: index i of `shuffled` came from `mapping[i]` of `ascending`.
    shuffled_pvalues = [0.5, 0.01, 0.04, 0.02, 0.03]
    mapping = [4, 0, 3, 1, 2]
    shuffled = holm_correction(shuffled_pvalues, alpha=0.05)

    np.testing.assert_allclose(shuffled.adjusted_pvalues, ascending.adjusted_pvalues[mapping])
    assert shuffled.reject.tolist() == [ascending.reject[i] for i in mapping]


def test_seed_variability_table_reports_mean_and_std_per_group() -> None:
    per_seed = pd.DataFrame(
        {
            "model": ["mfcc", "mfcc", "mfcc", "siamese", "siamese", "siamese"],
            "seed": [0, 1, 2, 0, 1, 2],
            "value": [0.30, 0.32, 0.31, 0.50, 0.70, 0.60],
        }
    )
    table = seed_variability_table(per_seed, group_cols=["model"])
    table = table.set_index("model")

    assert table.loc["mfcc", "n_seeds"] == 3
    assert table.loc["mfcc", "mean"] == pytest.approx(0.31)
    assert table.loc["siamese", "std"] == pytest.approx(0.1, abs=1e-9)


def test_seed_variability_table_requires_the_declared_columns() -> None:
    with pytest.raises(ValueError, match="missing"):
        seed_variability_table(pd.DataFrame({"seed": [0]}), group_cols=["model"])
