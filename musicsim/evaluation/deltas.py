"""Paired per-query differences with an artist-clustered confidence interval.

One function for every table that compares two ways of scoring the same
queries (graph variants, similarity measures, fuzzy graphs, perturbed audio):
the difference of the mean of a per-query metric, with the percentile interval
of :func:`musicsim.evaluation.bootstrap.paired_bootstrap_ci` resampling whole
artists (C.8.1: tracks of one artist are not independent observations).

The resample indices of a table depend only on the values of the ``within``
columns, never on the pair being compared. Every comparison inside one
combination therefore resamples the same artists, which is what lets two
differences of the same table be compared with each other, and why a copy of a
representation under another label has a difference of exactly 0 with an
interval of exactly ``[0, 0]``.

No p-value is produced here: the interval is the evidence this table reports.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from musicsim.evaluation.bootstrap import paired_bootstrap_ci
from musicsim.seeding import derive_seed

__all__ = ["paired_query_deltas"]

_OUTPUT_COLUMNS = ["a", "b", "delta", "ci_low", "ci_high", "n_queries", "n_artists"]


def paired_query_deltas(
    per_query: pd.DataFrame,
    *,
    by: str,
    pairs: Sequence[tuple[str, str]],
    within: Sequence[str],
    artist_by_item: pd.Series,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """``mean(a) - mean(b)`` per ``(a, b)`` in ``pairs`` and per combination of ``within``.

    ``per_query`` is long format with ``item_id``, ``value``, the column ``by``
    (the thing compared: a graph, a measure, a perturbation) and the ``within``
    columns (what must match for two rows to be comparable: representation,
    gallery, metric, k...). Within a combination, the two sides of a pair must
    hold exactly the same queries, aligned by ``item_id``; otherwise
    ``ValueError``. ``artist_by_item`` maps ``item_id`` to artist and is the
    resampling unit.

    Returns the ``within`` columns followed by ``a, b, delta, ci_low, ci_high,
    n_queries, n_artists``. Each combination's resample seed is
    ``derive_seed(seed, "deltas", *values_of_within)``, independent of the pair.
    """
    within = list(within)
    missing = [c for c in ["item_id", "value", by, *within] if c not in per_query.columns]
    if missing:
        raise ValueError(f"per_query lacks column(s) {missing}; has {list(per_query.columns)}")

    groups = per_query.groupby(within, sort=False) if within else [((), per_query)]
    records: list[dict] = []
    for key, frame in groups:
        key = key if isinstance(key, tuple) else (key,)
        stream = tuple(_stream_part(value) for value in key)
        combo_seed = derive_seed(seed, "deltas", *stream)
        combo = dict(zip(within, key, strict=True))
        values_by_side = _values_by_side(frame, by, combo)
        for a, b in pairs:
            va, vb = _side(values_by_side, a, by, combo), _side(values_by_side, b, by, combo)
            if not va.index.equals(vb.index):
                only = va.index.symmetric_difference(vb.index)
                raise ValueError(
                    f"{by} {a!r} and {b!r} do not have the same queries for {combo}: "
                    f"{len(only)} item(s) on one side only, e.g. {only[:3].tolist()}"
                )
            artists = artist_by_item.loc[va.index].to_numpy()
            result = paired_bootstrap_ci(
                va.to_numpy(), vb.to_numpy(), n_resamples=n_resamples, confidence=confidence,
                seed=combo_seed, groups=artists,
            )  # fmt: skip
            records.append(
                {
                    **combo, "a": a, "b": b, "delta": result.mean, "ci_low": result.ci_low,
                    "ci_high": result.ci_high, "n_queries": len(va),
                    "n_artists": len(np.unique(artists)),
                }
            )  # fmt: skip
    return pd.DataFrame.from_records(records, columns=[*within, *_OUTPUT_COLUMNS])


def _stream_part(value: object) -> str | int:
    """A seed stream part: integers stay integers, everything else is its text."""
    return int(value) if isinstance(value, int | np.integer) else str(value)


def _values_by_side(frame: pd.DataFrame, by: str, combo: dict) -> dict[object, pd.Series]:
    out: dict[object, pd.Series] = {}
    for side, part in frame.groupby(by, sort=False):
        if part["item_id"].duplicated().any():
            dup = part.loc[part["item_id"].duplicated(), "item_id"].iloc[0]
            raise ValueError(f"{by} {side!r} has item_id {dup} more than once for {combo}")
        out[side] = part.set_index("item_id")["value"].sort_index()
    return out


def _side(values_by_side: dict[object, pd.Series], side: str, by: str, combo: dict) -> pd.Series:
    if side not in values_by_side:
        raise ValueError(
            f"{by} {side!r} is not in the table for {combo}; has {sorted(map(str, values_by_side))}"
        )
    return values_by_side[side]
