"""Triplet agreement: how often a representation agrees with human similarity judgements.

The ground truth is MagnaTagATune's set of similarity constraints, derived from
the odd-one-out votes of TagATune players (:mod:`musicsim.datasets.mtt_triplets`,
``Wolff2012``). A constraint ``(anchor, pos, neg)`` says that people heard
``anchor`` as closer to ``pos`` than to ``neg``. A representation satisfies it
when ``cos(anchor, pos) > cos(anchor, neg)``; an exact tie counts as 0.5, the
chance value, so that a representation cannot gain or lose anything by being
unable to tell two clips apart. The metric depends on neither genres nor a
gallery: it looks at three clips at a time.

Variants
--------
``unweighted`` (headline)
    the plain mean over the 860 constraints, the figure the literature reports
    (``Wolff2012``, ``Karamanolakis2016``: 63.6-64.6 % for an untrained MFCC
    bag of audio words).
``vote_weighted``
    each constraint weighted by its ``margin``, the votes left after subtracting
    the opposing ones (6898 in all): constraints people agreed on more count
    more.
``margin_ge_2``
    the unweighted mean over the 642 constraints with a margin of at least 2,
    leaving out those that rest on one net vote.

The variants are those of ``docs/ground-truth-study.md`` section 4.3.

Uncertainty
-----------
The constraints of one triplet of clips share their clips and their voters, so
they are not independent: every interval is a bootstrap clustered by
``triplet_id`` (337 clusters), and every difference between two
representations is the paired version over the same clusters (C.8). Every
variant uses one seed for every representation and every pair, so all of them
resample the same triplets.

The vote-weighted mean is resampled by repeating each constraint ``margin``
times before the clustered bootstrap. Resampling a triplet then brings in its
constraints with their weights, and the mean of a resample is the weighted
mean of the constraints drawn, so no weighted variant of the bootstrap is
needed. This is our own arrangement; it is exact because the margins are
whole numbers of votes.

Comparisons between representations come with Holm's correction over the
declared family: the headline variant of every pair under
``evaluation.triplets.comparisons``. The other variants are reported without
correction, as secondary readings. The correction is marked ``provisional``:
the family is not closed until every representation of the report is in it,
and the final tables need 10 000 resamples (see
:func:`musicsim.evaluation.bootstrap.paired_bootstrap_test`).

Outputs, under ``metrics/``:

``triplets.csv``
    ``representation, seed, variant, value, ci_low, ci_high, n_constraints,
    n_triplets`` — one row per representation and variant.
``triplets_per_constraint.csv``
    ``representation, seed, anchor, pos, neg, margin, triplet_id, hit`` — the
    hit of every constraint, from which every figure above is computed.
``triplets_deltas.csv``
    ``representation_a, representation_b, variant, delta, ci_low, ci_high,
    p_value, p_holm, reject_holm, holm_status`` — ``delta`` is ``a - b``. Holm
    columns are filled for the headline variant only; written only when
    ``comparisons`` lists a pair.

The constraints are read from the file the dataset class writes next to its
index (``triplet_constraints_path``), so they always come from the same build as
the index; the clips they use must be exactly the index's ``in_triplets`` clips.

The evaluator is off unless the experiment sets ``evaluation.triplets.enabled``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from musicsim import seeding
from musicsim.config import Config, ConfigError
from musicsim.evaluation.base import Evaluator
from musicsim.evaluation.bootstrap import (
    BootstrapResult,
    PairedBootstrapTest,
    bootstrap_ci,
    holm_correction,
    paired_bootstrap_test,
)
from musicsim.galleries import gallery_mask
from musicsim.registry import register_evaluator
from musicsim.representations import load_aligned_representations, stage_representations
from musicsim.runlog import RunLog

__all__ = [
    "HEADLINE_VARIANT",
    "VARIANTS",
    "TripletEvaluator",
    "agreement_ci",
    "constraint_hits",
    "paired_agreement_test",
    "triplet_agreement",
]

#: Every variant of the agreement, the headline first (see the module docstring).
VARIANTS: tuple[str, ...] = ("unweighted", "vote_weighted", "margin_ge_2")

#: The variant the literature reports, and the only one in the Holm family.
HEADLINE_VARIANT = "unweighted"

#: Smallest margin a constraint needs to enter ``margin_ge_2``.
_MIN_MARGIN = 2

#: Score of a tie: the chance value of a two-way choice.
_TIE_SCORE = 0.5

#: Columns every constraint table must have.
_CONSTRAINT_COLUMNS = ("anchor", "pos", "neg", "margin", "triplet_id")

#: Gallery whose clips must be exactly those the constraints use.
_TRIPLET_GALLERY = "mtt_triplets"

#: Holm status of the headline comparisons until the family is closed.
_PROVISIONAL = "provisional"

_SUMMARY_COLUMNS = [
    "representation",
    "seed",
    "variant",
    "value",
    "ci_low",
    "ci_high",
    "n_constraints",
    "n_triplets",
]

_DELTA_COLUMNS = [
    "representation_a",
    "representation_b",
    "variant",
    "delta",
    "ci_low",
    "ci_high",
    "p_value",
    "p_holm",
    "reject_holm",
    "holm_status",
]


def constraint_hits(X: np.ndarray, ids: np.ndarray, constraints: pd.DataFrame) -> np.ndarray:
    """1, 0 or 0.5 per constraint: whether ``cos(anchor, pos)`` beats ``cos(anchor, neg)``.

    ``X`` is ``(N, d)``, row-aligned with ``ids``; ``constraints`` has the
    columns ``anchor``, ``pos`` and ``neg`` (item ids). The rows of ``X`` are
    brought to unit norm here, so the comparison is a cosine whatever the
    embedding did. A clip missing from ``ids``, or a row of norm zero (no
    cosine is defined), raises ``ValueError``.
    """
    positions = {}
    lookup = pd.Index(ids)
    for column in ("anchor", "pos", "neg"):
        found = lookup.get_indexer(constraints[column].to_numpy())
        if (found < 0).any():
            missing = constraints[column].to_numpy()[found < 0]
            raise ValueError(
                f"{len(missing)} {column} clip(s) of the constraints have no embedding, "
                f"e.g. {missing[:3].tolist()}"
            )
        positions[column] = found

    used = np.unique(np.concatenate(list(positions.values())))
    norms = np.linalg.norm(X[used].astype(np.float64), axis=1)
    if (norms == 0).any():
        raise ValueError(
            f"{int((norms == 0).sum())} clip(s) have an all-zero embedding, so no cosine "
            f"is defined, e.g. {ids[used[norms == 0]][:3].tolist()}"
        )
    unit = np.zeros((len(ids), X.shape[1]))
    unit[used] = X[used] / norms[:, None]

    anchor = unit[positions["anchor"]]
    sim_pos = (anchor * unit[positions["pos"]]).sum(axis=1)
    sim_neg = (anchor * unit[positions["neg"]]).sum(axis=1)
    return np.where(sim_pos > sim_neg, 1.0, np.where(sim_pos < sim_neg, 0.0, _TIE_SCORE))


def _resampling_units(
    values: np.ndarray, margin: np.ndarray, triplet_id: np.ndarray, variant: str
) -> tuple[np.ndarray, np.ndarray]:
    """``(values, groups)`` whose plain mean is ``variant`` and whose clusters are the triplets.

    ``vote_weighted`` repeats each constraint ``margin`` times (see the module
    docstring); ``margin_ge_2`` keeps the constraints with a margin of at least
    2; ``unweighted`` returns its input.
    """
    values = np.asarray(values, dtype=np.float64)
    margin = np.asarray(margin)
    triplet_id = np.asarray(triplet_id)
    if not (values.shape == margin.shape == triplet_id.shape) or values.ndim != 1:
        raise ValueError(
            "values, margin and triplet_id must be 1-D and row-aligned, got shapes "
            f"{values.shape}, {margin.shape} and {triplet_id.shape}"
        )
    if len(values) == 0:
        raise ValueError("there are no constraints to score")
    if not np.issubdtype(margin.dtype, np.integer) or (margin < 1).any():
        raise ValueError(
            f"every margin must be a whole number of votes >= 1, got dtype {margin.dtype} "
            f"and minimum {margin.min()}"
        )

    if variant == "unweighted":
        return values, triplet_id
    if variant == "vote_weighted":
        return np.repeat(values, margin), np.repeat(triplet_id, margin)
    if variant == "margin_ge_2":
        keep = margin >= _MIN_MARGIN
        if not keep.any():
            raise ValueError(f"no constraint has a margin >= {_MIN_MARGIN}")
        return values[keep], triplet_id[keep]
    raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")


def _scored(margin: np.ndarray, variant: str) -> np.ndarray:
    """Which constraints ``variant`` scores (for the ``n_constraints`` column)."""
    if variant == "margin_ge_2":
        return margin >= _MIN_MARGIN
    return np.ones(margin.shape, dtype=bool)


def triplet_agreement(hits: np.ndarray, margin: np.ndarray, *, variant: str) -> float:
    """Agreement of ``hits`` (from :func:`constraint_hits`) under ``variant``.

    ``margin`` holds the net votes of each constraint, row-aligned with
    ``hits``. ``unweighted`` is the plain mean, ``vote_weighted`` the mean
    weighted by ``margin`` and ``margin_ge_2`` the plain mean over the
    constraints with a margin of at least 2.
    """
    margin = np.asarray(margin)
    values, _ = _resampling_units(hits, margin, np.zeros(margin.shape, dtype=int), variant)
    return float(values.mean())


def agreement_ci(
    hits: np.ndarray,
    margin: np.ndarray,
    triplet_id: np.ndarray,
    *,
    variant: str,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> BootstrapResult:
    """:func:`triplet_agreement` with its bootstrap interval, clustered by ``triplet_id``."""
    values, groups = _resampling_units(hits, margin, triplet_id, variant)
    return bootstrap_ci(
        values, n_resamples=n_resamples, confidence=confidence, seed=seed, groups=groups
    )


def paired_agreement_test(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    margin: np.ndarray,
    triplet_id: np.ndarray,
    *,
    variant: str,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> PairedBootstrapTest:
    """Agreement of ``a`` minus that of ``b``, with interval and p-value over the same triplets.

    ``hits_a`` and ``hits_b`` score the same constraints, in the same order.
    Every variant is linear in the hits, so its difference is the variant of
    ``hits_a - hits_b``; both are expanded the same way before the paired
    bootstrap.
    """
    hits_a = np.asarray(hits_a, dtype=np.float64)
    hits_b = np.asarray(hits_b, dtype=np.float64)
    if hits_a.shape != hits_b.shape:
        raise ValueError(
            f"hits_a and hits_b must score the same constraints, got shapes "
            f"{hits_a.shape} and {hits_b.shape}"
        )
    a, groups = _resampling_units(hits_a, margin, triplet_id, variant)
    b, _ = _resampling_units(hits_b, margin, triplet_id, variant)
    return paired_bootstrap_test(
        a, b, n_resamples=n_resamples, confidence=confidence, seed=seed, groups=groups
    )


def _parse_comparisons(raw: object, labels: Sequence[str]) -> list[tuple[str, str]]:
    """``evaluation.triplets.comparisons`` as validated ``(a, b)`` pairs."""
    if raw is None:
        return []
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise ConfigError(
            f"evaluation.triplets.comparisons must be a list of [a, b] pairs, got {raw!r}"
        )
    pairs: list[tuple[str, str]] = []
    for pair in raw:
        if isinstance(pair, str) or not isinstance(pair, Sequence) or len(pair) != 2:
            raise ConfigError(f"each comparison must be a pair [a, b], got {pair!r}")
        a, b = (str(label) for label in pair)
        unknown = [label for label in (a, b) if label not in labels]
        if unknown:
            raise ConfigError(
                f"comparison {[a, b]} names {unknown}, which is not among the "
                f"evaluated representations {list(labels)}"
            )
        if a == b:
            raise ConfigError(f"comparison {[a, b]} compares a representation with itself")
        pairs.append((a, b))
    return pairs


def _read_constraints(config: Config) -> pd.DataFrame:
    """The constraints the dataset class wrote next to its index."""
    from musicsim.registry import DATASETS

    directory = config.require("dataset.directory")
    dataset = DATASETS.get(directory)(config)
    path = getattr(dataset, "triplet_constraints_path", None)
    if path is None:
        raise ConfigError(
            f"dataset {directory!r} has no triplet constraints; the triplets evaluator "
            "needs a corpus with human similarity judgements, such as mtt"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `musicsim index --dataset {directory}` first"
        )
    constraints = pd.read_csv(path)
    missing = [column for column in _CONSTRAINT_COLUMNS if column not in constraints.columns]
    if missing:
        raise ValueError(f"{path} lacks column(s) {missing}; expected {list(_CONSTRAINT_COLUMNS)}")
    return constraints


def _check_clips(constraints: pd.DataFrame, index_common: pd.DataFrame) -> None:
    """The constraints must use exactly the index's ``in_triplets`` clips."""
    flagged = set(index_common.index[gallery_mask(index_common, _TRIPLET_GALLERY)].tolist())
    used = set(pd.unique(constraints[["anchor", "pos", "neg"]].to_numpy().ravel()).tolist())
    if flagged != used:
        raise ValueError(
            f"the constraints use {len(used)} clip(s) and the index flags {len(flagged)} "
            f"as in_triplets ({len(used - flagged)} unflagged, {len(flagged - used)} unused); "
            "the index and triplet_constraints.csv come from different builds, "
            "run `musicsim index` again"
        )


@register_evaluator("triplets")
class TripletEvaluator(Evaluator):
    """Agreement of every representation with the human similarity constraints.

    Reads ``evaluation.triplets`` (``comparisons``, ``baselines``) and the shared
    ``bootstrap`` section (``n_resamples``, ``confidence``); the top-level
    ``representations`` list works as it does for retrieval. ``seed`` in the
    outputs is each representation's own seed; the bootstrap uses the
    experiment's.
    """

    def evaluate(self, index: pd.DataFrame, run: RunLog) -> None:
        stage_cfg = self.config.section("evaluation").get("triplets", {})
        main_name = self.config.require("extractor.name")
        specs = stage_representations(self.config, stage_cfg, main_name, "triplets")
        labels = [spec.label for spec in specs]
        pairs = _parse_comparisons(stage_cfg.get("comparisons"), labels)

        constraints = _read_constraints(self.config)
        loaded = load_aligned_representations(self.config, index, specs)
        _check_clips(constraints, loaded.index)

        n_resamples = int(self.config.get("bootstrap.n_resamples", 1000))
        confidence = float(self.config.get("bootstrap.confidence", 0.95))
        seed = int(self.config.require("seed"))
        variant_seeds = {
            variant: seeding.derive_seed(seed, "triplets", variant) for variant in VARIANTS
        }
        margin = constraints["margin"].to_numpy()
        triplet_id = constraints["triplet_id"].to_numpy()

        hits = {
            label: constraint_hits(loaded.X[label], loaded.ids, constraints) for label in labels
        }

        per_constraint = pd.concat(
            [
                constraints[list(_CONSTRAINT_COLUMNS)].assign(
                    representation=label, seed=loaded.seeds[label], hit=hits[label]
                )
                for label in labels
            ],
            ignore_index=True,
        )
        per_constraint = per_constraint[["representation", "seed", *_CONSTRAINT_COLUMNS, "hit"]]
        per_constraint.to_csv(run.metrics_dir / "triplets_per_constraint.csv", index=False)

        records = []
        for label in labels:
            for variant in VARIANTS:
                result = agreement_ci(
                    hits[label],
                    margin,
                    triplet_id,
                    variant=variant,
                    n_resamples=n_resamples,
                    confidence=confidence,
                    seed=variant_seeds[variant],
                )
                kept = _scored(margin, variant)
                records.append(
                    {
                        "representation": label,
                        "seed": loaded.seeds[label],
                        "variant": variant,
                        "value": result.mean,
                        "ci_low": result.ci_low,
                        "ci_high": result.ci_high,
                        "n_constraints": int(kept.sum()),
                        "n_triplets": int(np.unique(triplet_id[kept]).size),
                    }
                )
        summary = pd.DataFrame.from_records(records, columns=_SUMMARY_COLUMNS)
        summary.to_csv(run.metrics_dir / "triplets.csv", index=False)

        if pairs:
            deltas = _deltas(
                hits,
                pairs,
                margin,
                triplet_id,
                variant_seeds,
                n_resamples=n_resamples,
                confidence=confidence,
            )
            deltas.to_csv(run.metrics_dir / "triplets_deltas.csv", index=False)

        headline = summary[summary["variant"] == HEADLINE_VARIANT].set_index("representation")
        run.logger.info(
            "triplets: %d constraint(s) over %d triplet(s); %s agreement %s; %d comparison(s)",
            len(constraints),
            int(np.unique(triplet_id).size),
            HEADLINE_VARIANT,
            ", ".join(f"{label} {headline.loc[label, 'value']:.3f}" for label in labels),
            len(pairs),
        )


def _deltas(
    hits: dict[str, np.ndarray],
    pairs: Sequence[tuple[str, str]],
    margin: np.ndarray,
    triplet_id: np.ndarray,
    variant_seeds: dict[str, int],
    *,
    n_resamples: int,
    confidence: float,
) -> pd.DataFrame:
    """One row per pair and variant; Holm over the headline variant of every pair."""
    records = []
    for a, b in pairs:
        for variant in VARIANTS:
            test = paired_agreement_test(
                hits[a],
                hits[b],
                margin,
                triplet_id,
                variant=variant,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=variant_seeds[variant],
            )
            records.append(
                {
                    "representation_a": a,
                    "representation_b": b,
                    "variant": variant,
                    "delta": test.mean,
                    "ci_low": test.ci_low,
                    "ci_high": test.ci_high,
                    "p_value": test.p_value,
                    "p_holm": np.nan,
                    "reject_holm": pd.NA,
                    "holm_status": pd.NA,
                }
            )
    deltas = pd.DataFrame.from_records(records, columns=_DELTA_COLUMNS)

    family = deltas["variant"] == HEADLINE_VARIANT
    holm = holm_correction(deltas.loc[family, "p_value"].to_numpy(), alpha=1 - confidence)
    deltas["reject_holm"] = deltas["reject_holm"].astype("boolean")
    deltas.loc[family, "p_holm"] = holm.adjusted_pvalues
    deltas.loc[family, "reject_holm"] = holm.reject
    deltas.loc[family, "holm_status"] = _PROVISIONAL
    return deltas
