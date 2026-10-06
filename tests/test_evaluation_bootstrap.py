"""Tests for :mod:`musicsim.evaluation.bootstrap`: the C.8 protocol primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from musicsim.evaluation.bootstrap import (
    bootstrap_ci,
    bootstrap_means,
    bootstrap_pvalue,
    holm_correction,
    paired_bootstrap_ci,
    paired_bootstrap_test,
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


def test_bootstrap_pvalue_matches_a_hand_worked_example() -> None:
    # B = 10 resampled means, null = 0.05: two are <= null (-0.2, -0.1) and
    # eight are >= null (0.1 ... 0.8), so
    #   tail_low = (2 + 1) / 11, tail_high = (8 + 1) / 11, p = 2 * 3/11 = 6/11.
    resampled = np.array([-0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert bootstrap_pvalue(resampled, null=0.05) == pytest.approx(6 / 11)
    # Ties count in both tails: null = 0.1 -> 3 values <= and 8 values >=, so
    # tail_low = 4/11, tail_high = 9/11, p = 8/11.
    assert bootstrap_pvalue(resampled, null=0.1) == pytest.approx(8 / 11)


def test_bootstrap_pvalue_is_clipped_to_one() -> None:
    # B = 3, null = 0 sits on the middle value: r_low = 2 and r_high = 2, so
    # both tails are 3/4 and 2 * 3/4 = 1.5 is clipped to 1.
    resampled = np.array([-1.0, 0.0, 1.0])
    assert bootstrap_pvalue(resampled, null=0.0) == 1.0


@pytest.mark.parametrize("bad", [np.array([]), np.zeros((3, 2))])
def test_bootstrap_pvalue_rejects_empty_or_non_1d_input(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="1-D"):
        bootstrap_pvalue(bad)


def test_paired_bootstrap_test_of_identical_arrays_is_zero_with_p_one() -> None:
    values = np.linspace(0.0, 1.0, 30)
    result = paired_bootstrap_test(values, values, n_resamples=300, seed=3)
    assert result.mean == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0
    assert result.p_value == 1.0


def test_paired_bootstrap_test_of_a_constant_gain_gives_the_minimum_p_value() -> None:
    a = np.linspace(0.2, 0.9, 40)
    b = a - 0.1
    n_resamples = 500
    result = paired_bootstrap_test(a, b, n_resamples=n_resamples, seed=5)
    # No resample of a constant positive difference reaches 0: r_low = 0 and
    # r_high = B, so p = 2 * 1 / (B + 1).
    assert result.p_value == pytest.approx(2 / (n_resamples + 1))
    assert result.mean == pytest.approx(0.1)
    assert 0.0 < result.ci_low <= result.ci_high


@pytest.mark.parametrize("use_groups", [False, True])
def test_paired_bootstrap_test_matches_paired_bootstrap_ci(use_groups: bool) -> None:
    rng = np.random.default_rng(11)
    a = rng.normal(0.6, 0.1, size=60)
    b = rng.normal(0.5, 0.1, size=60)
    groups = np.repeat(np.arange(12), 5) if use_groups else None

    test = paired_bootstrap_test(a, b, n_resamples=400, confidence=0.9, seed=9, groups=groups)
    ci = paired_bootstrap_ci(a, b, n_resamples=400, confidence=0.9, seed=9, groups=groups)

    assert test.mean == ci.mean
    assert test.ci_low == ci.ci_low
    assert test.ci_high == ci.ci_high


def test_paired_bootstrap_test_is_reproducible_with_the_same_seed() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=50)
    b = rng.normal(size=50)
    first = paired_bootstrap_test(a, b, n_resamples=300, seed=21)
    second = paired_bootstrap_test(a, b, n_resamples=300, seed=21)
    assert first == second


def test_paired_bootstrap_test_is_calibrated_under_the_null() -> None:
    # Both samples come from the same distribution, so the true difference is 0
    # and a well-calibrated test rejects at 0.05 about 5 % of the time. The
    # percentile bootstrap is slightly liberal at n = 50, hence the margin.
    rng = np.random.default_rng(123)
    n_repeats = 400
    rejections = 0
    for rep in range(n_repeats):
        a = rng.normal(size=50)
        b = rng.normal(size=50)
        rejections += paired_bootstrap_test(a, b, n_resamples=1000, seed=rep).p_value < 0.05
    assert 0.02 <= rejections / n_repeats <= 0.09


def test_paired_bootstrap_test_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="row-aligned"):
        paired_bootstrap_test(np.zeros(5), np.zeros(6))


def test_paired_bootstrap_test_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap_test(np.array([]), np.array([]))


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
