"""Uncertainty and multiple-comparison protocol (C.8): clustered bootstrap
confidence intervals, Holm-Bonferroni correction, and the per-seed variability
table the tutor asked for explicitly.

Every resampling function takes an explicit ``seed`` rather than drawing from
NumPy's global state (see :mod:`musicsim.seeding`), so that two calls with the
same ``(n_units, n_resamples, seed)`` draw the same resample indices — the
"idénticos índices de bootstrap para todas las representaciones" invariant of
B.1.8: comparing two representations means resampling the *same* queries (or
the same artists) in both, which is what makes :func:`paired_bootstrap_ci`
meaningful.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "BootstrapResult",
    "HolmResult",
    "bootstrap_ci",
    "bootstrap_means",
    "holm_correction",
    "paired_bootstrap_ci",
    "seed_variability_table",
]

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class BootstrapResult:
    """A point estimate with its percentile confidence interval."""

    mean: float
    ci_low: float
    ci_high: float


def bootstrap_means(
    values: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """The macro-average of ``n_resamples`` resamples with replacement.

    ``values`` is ``(Q,)`` or ``(Q, M)``; the result is ``(n_resamples,)`` or
    ``(n_resamples, M)``. Without ``groups`` each of the ``Q`` rows is its own
    resampling unit (the usual bootstrap); with ``groups`` ``(Q,)`` (e.g.
    artist id per row) whole groups are resampled together with every row they
    contain — the clustered bootstrap, needed because rows from the same
    artist are not independent observations.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim not in (1, 2):
        raise ValueError(f"values must be 1-D or 2-D, got shape {v.shape}")

    if len(v) == 0:
        raise ValueError("no resampling units: values is empty")
    if groups is None:
        unit = np.arange(len(v))
    else:
        unit = np.unique(np.asarray(groups), return_inverse=True)[1].ravel()
    n_units = int(unit.max()) + 1

    sums = np.zeros((n_units, *v.shape[1:]))
    np.add.at(sums, unit, v)
    sizes = np.bincount(unit, minlength=n_units).astype(np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_units, size=(n_resamples, n_units))
    # Multiplicity of each unit in each resample, via one bincount over a
    # flattened (resample, unit) index — avoids materialising values[idx].
    counts = np.bincount(
        (idx + n_units * np.arange(n_resamples)[:, None]).ravel(), minlength=n_resamples * n_units
    ).reshape(n_resamples, n_units).astype(np.float64)

    totals = counts @ sizes
    weighted_sums = counts @ sums.reshape(n_units, -1)
    means = weighted_sums / totals[:, None]
    return means if v.ndim == 2 else means.ravel()


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> BootstrapResult:
    """Percentile bootstrap confidence interval of the macro-average of ``values``."""
    v = np.asarray(values, dtype=np.float64)
    resampled = bootstrap_means(v, n_resamples=n_resamples, seed=seed, groups=groups)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(resampled, [alpha, 1 - alpha])
    return BootstrapResult(mean=float(v.mean()), ci_low=float(low), ci_high=float(high))


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> BootstrapResult:
    """Confidence interval of ``mean(a) - mean(b)``, resampling the same units in both.

    ``a`` and ``b`` must be row-aligned (row ``i`` of each is the same query in
    both representations being compared) — this is the caller's
    responsibility. The interval is of the *paired* difference, so it is
    typically tighter than combining the two individual intervals would
    suggest, and its ``ci_low``/``ci_high`` excluding 0 is the significance
    criterion of C.8.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"a and b must be row-aligned, got shapes {a.shape} and {b.shape}")
    return bootstrap_ci(
        a - b, n_resamples=n_resamples, confidence=confidence, seed=seed, groups=groups
    )


@dataclasses.dataclass(frozen=True, slots=True)
class HolmResult:
    """Holm-Bonferroni step-down correction over a declared family of tests."""

    reject: np.ndarray
    adjusted_pvalues: np.ndarray


def holm_correction(pvalues: Sequence[float], *, alpha: float = 0.05) -> HolmResult:
    """Holm-Bonferroni step-down correction (Holm, 1979).

    Controls the family-wise error rate over the whole declared family (C.8.4)
    at ``alpha``, less conservative than a flat Bonferroni. ``reject[i]`` is
    whether hypothesis ``i`` (in the input order) is rejected;
    ``adjusted_pvalues[i]`` is its Holm-adjusted p-value, monotone
    non-decreasing in rank so it can be compared against ``alpha`` directly.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("pvalues must be a non-empty 1-D sequence")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("every p-value must be in [0, 1]")

    m = len(p)
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    multipliers = m - np.arange(m)
    step_adjusted = np.minimum(sorted_p * multipliers, 1.0)
    # Enforce monotonicity: an adjusted p-value can never be smaller than an
    # earlier (more significant) one's.
    adjusted_sorted = np.maximum.accumulate(step_adjusted)

    reject_sorted = np.zeros(m, dtype=bool)
    for i in range(m):
        if sorted_p[i] <= alpha / multipliers[i]:
            reject_sorted[i] = True
        else:
            break  # step-down: stop at the first non-rejection

    reject = np.empty(m, dtype=bool)
    adjusted = np.empty(m, dtype=np.float64)
    reject[order] = reject_sorted
    adjusted[order] = adjusted_sorted
    return HolmResult(reject=reject, adjusted_pvalues=adjusted)


def seed_variability_table(
    per_seed: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    value_col: str = "value",
    seed_col: str = "seed",
) -> pd.DataFrame:
    """Mean and standard deviation across seeds, for each group in ``group_cols``.

    ``per_seed`` has one row per ``(*group_cols, seed)`` combination (already
    reduced to one scalar per seed, e.g. a metric's macro-average over
    queries). This is the "tabla propia de variabilidad entre semillas" of
    C.8.3: a model's headline number is not trustworthy on its own until this
    table shows it does not move much between seeds.
    """
    missing = [c for c in (*group_cols, value_col, seed_col) if c not in per_seed.columns]
    if missing:
        raise ValueError(f"per_seed is missing column(s): {missing}")

    grouped = per_seed.groupby(list(group_cols), dropna=False)[value_col]
    table = grouped.agg(mean="mean", std="std", n_seeds="count").reset_index()
    table["std"] = table["std"].fillna(0.0)
    return table
