"""Paired differences between representations, from ``per_query.csv`` (C.8).

:func:`paired_deltas` answers "is representation ``A`` better than the
reference ``B`` on the headline retrieval metrics, and is the difference
distinguishable from zero?": the mean difference ``A - B``, its percentile
interval from an artist-clustered paired bootstrap, the bootstrap p-value, and
the Holm-adjusted p-value over the whole declared family.

Decisions worth knowing:

* **Paired by item, resampled by artist.** Row ``i`` of ``A`` and of ``B`` is
  the same query; tracks by the same artist are not independent, so whole
  artists are resampled together (C.8.1). Both representations must cover
  exactly the same queries: retrieval already evaluates every representation
  on the intersection of their ids, so a mismatch means the two frames come
  from different runs and is an error rather than something to align quietly.
* **Same resample indices everywhere.** The seed depends on the gallery,
  variant and query mode but not on the candidate or the metric, so every
  comparison of one table draws the same artist resamples (the "identical
  bootstrap indices" invariant of the protocol) and the p-values within a
  family are computed on a common resampling.
* **The family is what the caller passes.** Holm runs over every
  ``(candidate, metric)`` pair of one call, so declare the family by calling
  once with all of its candidates. Controls such as ``random`` are not part of
  the confirmatory family and should be left out of the call.
* **Resamples.** The default is 10 000: with 1 000 the smallest attainable
  p-value (about 0.002) cannot survive Holm over a family of more than 25
  tests (see :func:`musicsim.evaluation.bootstrap.paired_bootstrap_test`).
* One representation, one row per query: a frame with several seeds of the same
  representation would silently average over duplicated ``item_id`` rows, so it
  is rejected; filter the seed first.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from musicsim.evaluation.bootstrap import holm_correction, paired_bootstrap_test
from musicsim.seeding import derive_seed

__all__ = ["HEADLINE_METRICS", "paired_deltas"]

#: Headline retrieval metrics as ``(metric, k)`` of ``per_query.csv``; ``MAP`` has no cutoff
#: and carries the sentinel ``k = 0``.
HEADLINE_METRICS: tuple[tuple[str, int], ...] = (("R", 10), ("MAP", 0), ("nDCG", 10))

_COLUMNS = (
    "gallery",
    "variant",
    "queries",
    "candidate",
    "reference",
    "metric",
    "k",
    "mean_candidate",
    "mean_reference",
    "delta",
    "ci_low",
    "ci_high",
    "p_value",
    "p_holm",
    "reject",
    "n_queries",
    "n_artists",
)


def _values(
    per_query: pd.DataFrame, representation: str, metric: str, k: int, *, where: pd.Series
) -> pd.Series:
    """Values of one representation and metric, indexed by ``item_id``."""
    mask = where & (per_query["representation"] == representation)
    mask &= (per_query["metric"] == metric) & (per_query["k"] == k)
    rows = per_query.loc[mask, ["item_id", "value"]]
    if rows.empty:
        raise ValueError(
            f"per_query has no rows for representation {representation!r}, metric {metric!r}, "
            f"k={k} in the requested gallery/variant/queries; expected at least one"
        )
    if rows["item_id"].duplicated().any():
        raise ValueError(
            f"representation {representation!r} has several rows for the same item_id "
            f"(metric {metric!r}, k={k}); expected one row per query, so filter a single seed first"
        )
    return rows.set_index("item_id")["value"].sort_index()


def paired_deltas(
    per_query: pd.DataFrame,
    artist_by_id: pd.Series,
    *,
    candidates: Sequence[str],
    reference: str,
    gallery: str,
    variant: str = "dedup",
    queries: str = "all",
    metrics: Sequence[tuple[str, int]] = HEADLINE_METRICS,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """``candidate - reference`` on each metric, one row per ``(candidate, metric)``.

    ``per_query`` has the ``per_query.csv`` schema (``item_id, metric, k, value,
    representation, gallery, variant, queries, ...``); ``artist_by_id`` maps
    ``item_id`` to artist. Columns: ``delta`` (mean paired difference),
    ``ci_low``/``ci_high`` (percentile interval at ``confidence``), ``p_value``
    (bootstrap, two-sided), ``p_holm`` (Holm-adjusted over every row of the
    result) and ``reject`` (``p_holm < alpha``). The interval excluding 0, the
    adjusted p-value below ``alpha`` and, for models with seeds, the same sign in
    every seed are the three conditions of the significance rule; this function
    provides the first two.
    """
    if not candidates:
        raise ValueError("candidates is empty; expected at least one representation")
    if reference in candidates:
        raise ValueError(f"reference {reference!r} is also a candidate; expected it to be apart")
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"candidates has repeated entries: {list(candidates)}")
    if not metrics:
        raise ValueError("metrics is empty; expected at least one (metric, k)")

    where = (
        (per_query["gallery"] == gallery)
        & (per_query["variant"] == variant)
        & (per_query["queries"] == queries)
    )
    stream = ("delta", gallery, variant, queries)
    records: list[dict] = []
    for metric, k in metrics:
        base = _values(per_query, reference, metric, k, where=where)
        missing = base.index.difference(artist_by_id.index)
        if len(missing):
            raise ValueError(
                f"artist_by_id has no artist for {len(missing)} item_id(s), e.g. {missing[0]!r}; "
                "expected every query to have one"
            )
        artists = artist_by_id.loc[base.index].to_numpy()
        for candidate in candidates:
            other = _values(per_query, candidate, metric, k, where=where)
            if not other.index.equals(base.index):
                raise ValueError(
                    f"{candidate!r} and {reference!r} cover different queries for metric "
                    f"{metric!r}, k={k} ({len(other)} vs {len(base)}); expected the same item_ids"
                )
            result = paired_bootstrap_test(
                other.to_numpy(),
                base.to_numpy(),
                n_resamples=n_resamples,
                confidence=confidence,
                seed=derive_seed(seed, *stream),
                groups=artists,
            )
            records.append(
                {
                    "gallery": gallery,
                    "variant": variant,
                    "queries": queries,
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "k": k,
                    "mean_candidate": float(other.mean()),
                    "mean_reference": float(base.mean()),
                    "delta": result.mean,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "p_value": result.p_value,
                    "n_queries": len(base),
                    "n_artists": int(np.unique(artists).size),
                }
            )

    table = pd.DataFrame.from_records(records)
    holm = holm_correction(table["p_value"].to_numpy(), alpha=alpha)
    table["p_holm"] = holm.adjusted_pvalues
    table["reject"] = table["p_holm"] < alpha
    table = table[list(_COLUMNS)]
    # Candidate-major order, metrics in the order given.
    order = {name: i for i, name in enumerate(candidates)}
    return table.sort_values("candidate", key=lambda s: s.map(order), kind="stable").reset_index(
        drop=True
    )
