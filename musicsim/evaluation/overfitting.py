"""Overfitting of the heads trained on human judgements: the gap and the validation curves.

The heads of :mod:`musicsim.extractors.human_siamese` learn from about 600 of
MagnaTagATune's 860 similarity constraints per fold, with hundreds of thousands
of parameters (the linear map alone has about a hundred thousand). The out-of-fold
agreement that the triplet evaluator reports says how well they generalise; this
module says how far that is from what they reached on the constraints they were
trained on, fold by fold, and how the two evolved during training. It reads the
files the cross-validation leaves in the checkpoint directories and trains
nothing.

**The three agreements of a fold**, all unweighted (a tie counts 0.5, as in
:mod:`musicsim.evaluation.triplets`), each taken from the fold's chosen grid
point:

``train_agreement``
    on the constraints of the inner training split, the only ones that produced
    gradients, at the chosen epoch. It is read from the fold's ``selection.csv``,
    which records it at the best epoch. The validator scores the model in
    evaluation mode, so the weights kept in ``best.pt`` give that same value and
    nothing needs recomputing (our choice).
``inner_agreement``
    on the inner validation fold, which chose the epoch and the grid point, so
    it is optimistic: it is the maximum over the grid of a noisy estimate.
``test_agreement``
    on the outer test fold, out of fold: the mean hit of the fold's constraints
    in the triplet evaluator's ``triplets_per_constraint.csv``, the fold of each
    constraint coming from the checkpoint's ``folds.csv``. The folds change with
    the seed (:mod:`musicsim.extractors.human_siamese`), so each seed is joined
    with its own.

``gap = train - test`` is the overfitting gap; ``optimism = inner - test`` is
what the selection on the inner fold promised beyond what the test fold gave.

**Validation curves.** For every fold, the per-epoch curve of the chosen grid
point (the fold's ``curve.csv``): agreement on the inner training split and on
the inner validation fold. The curves start after the first epoch, since the
training loop scores no epoch zero. The figure has one panel per variant, the
MLP head and the linear map, on the same vertical axis, with one thin line per
fold and seed and no mean: early stopping ends the curves at different epochs,
so a mean over them would change its membership along the axis. The chosen epoch
of each curve is marked, and two horizontal references are drawn: the mean
out-of-fold agreement of the variant and the agreement of the input
representation (``mert``) on all the constraints. The question it answers is the
tutor's (2026-11-20): whether the linear map is enough against the MLP.

**Running it.** ``python -m musicsim.evaluation.overfitting --run <run>`` reads
a finished run of ``configs/experiments/e12_genre_vs_human_mtt.yaml``. The
checkpoint directory of every ``human_siamese`` representation is derived from
the run's resolved configuration, exactly as the extractor derives it, never
written by hand. The analysis is a run of its own (:func:`musicsim.runlog.start_run`,
experiment :data:`EXPERIMENT`) whose configuration names the source run, so the
e12 run is never modified. It writes, under ``metrics/``:

:data:`GAP_FILE`
    ``representation, variant, seed, fold, point, best_epoch,
    n_training_constraints, n_inner_constraints, n_test_constraints,
    train_agreement, inner_agreement, test_agreement, gap, optimism``.
:data:`SUMMARY_FILE`
    ``variant, seed, quantity, mean, min, max`` over the folds, per variant and
    seed and per variant over every seed (``seed`` empty).
:data:`CURVES_FILE`
    ``representation, variant, seed, fold, epoch, train_loss, train_agreement,
    val_agreement, chosen_epoch``.

and the figure :data:`FIGURE_STEM` under ``figures/`` (PNG) and, as a PDF, in
``report/figures/generated/`` for the report.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

from musicsim import paths
from musicsim.config import Config, ConfigError
from musicsim.runlog import start_run

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "CURVES_FILE",
    "EXPERIMENT",
    "FIGURE_STEM",
    "GAP_FILE",
    "SUMMARY_FILE",
    "VARIANTS",
    "CrossValidationFiles",
    "HumanModel",
    "analyse",
    "curve_table",
    "fold_gaps",
    "gap_summary",
    "locate_checkpoints",
    "main",
    "plot_validation_curves",
    "read_cross_validation",
]

logger = logging.getLogger(__name__)

#: Experiment name of the analysis run.
EXPERIMENT = "e12_overfitting"

#: Per-fold agreements and gap.
GAP_FILE = "overfitting_gap.csv"

#: Mean, minimum and maximum over the folds.
SUMMARY_FILE = "overfitting_summary.csv"

#: Per-epoch curves of the chosen grid point of every fold.
CURVES_FILE = "validation_curves.csv"

#: File name, without suffix, of the figure.
FIGURE_STEM = "validation_curves"

#: Variants in panel order: the head of ``siamese_emb`` and the linear map.
VARIANTS = ("mlp", "linear")

#: Extractor whose checkpoints are analysed.
_EXTRACTOR = "human_siamese"

#: The input representation drawn as a reference; the reference of every comparison.
_REFERENCE = "mert"

#: Variant of the triplet agreement the reference line uses (the headline one).
_REFERENCE_VARIANT = "unweighted"

# File names written by the cross-validation. They repeat those of
# musicsim.extractors.human_siamese, musicsim.training.constraints and
# musicsim.training.train, which import torch; a test checks they agree.
_FOLDS_FILE = "folds.csv"
_CROSS_VALIDATION_SUMMARY = "cross_validation.json"
_SELECTION_FILE = "selection.csv"
_FOLD_SUMMARY = "fold.json"
_CURVE_FILE = "curve.csv"

#: Columns of the selection and the curves holding the two agreements.
_TRAIN = "train_agreement"
_VAL = "val_agreement"

#: Columns that identify a constraint in ``folds.csv`` and in the per-constraint hits.
_KEY = ["anchor", "pos", "neg", "triplet_id"]

#: Inputs, under the source run's ``metrics/``.
_PER_CONSTRAINT = "triplets_per_constraint.csv"
_TRIPLETS = "triplets.csv"

#: The resolved configuration of a run.
_RESOLVED_CONFIG = "config.resolved.yaml"

#: Quantities summarised over the folds.
_SUMMARISED = ("train_agreement", "inner_agreement", "test_agreement", "gap", "optimism")

# --- figure ------------------------------------------------------------------------
#: Size in inches: the text width of the report, two panels side by side.
_FIGURE_SIZE = (6.3, 2.8)
#: Resolution of the PNG copy.
_PNG_DPI = 200
#: Series colours: categorical slots 1 and 2 of a palette checked for colour
#: vision deficiency (the orange is below 3:1 against white, hence the legend).
_TRAIN_COLOUR = "#257de0"
_VAL_COLOUR = "#f47230"
#: Neutral ink of the reference lines, axes and grid.
_REFERENCE_COLOUR = "#4d4d4d"
_GRID_COLOUR = "#e0e0e0"
#: Thin, translucent curves: 30 per panel overlap.
_CURVE_WIDTH = 0.8
_CURVE_ALPHA = 0.35
#: Marker of the chosen epoch, in points.
_MARKER_SIZE = 3.0
_REFERENCE_WIDTH = 1.0
#: Distance in points between a reference line and its label.
_LABEL_OFFSET = 2
#: Panel titles of the variants.
_TITLES = {"mlp": "MLP head", "linear": "Linear map"}


@dataclass(frozen=True)
class HumanModel:
    """One ``human_siamese`` representation of a run and its checkpoint directory."""

    label: str
    variant: str
    seed: int
    directory: Path


@dataclass(frozen=True)
class CrossValidationFiles:
    """What the cross-validation of one configuration left in its directory."""

    seed: int
    n_parameters: int
    folds: pd.DataFrame
    selections: tuple[pd.DataFrame, ...]
    curves: tuple[pd.DataFrame, ...]
    summaries: tuple[dict[str, Any], ...]


# --- reading -----------------------------------------------------------------------
def read_cross_validation(directory: Path) -> CrossValidationFiles:
    """Read ``folds.csv``, ``cross_validation.json`` and every ``fold_<k>/`` of ``directory``."""
    directory = Path(directory)
    summary_path = directory / _CROSS_VALIDATION_SUMMARY
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"{summary_path} not found: the cross-validation of this configuration has not "
            "finished; train it first (README, 'Running experiments')"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selections, curves, summaries = [], [], []
    for fold in range(int(summary["n_folds"])):
        fold_dir = directory / f"fold_{fold}"
        selections.append(pd.read_csv(fold_dir / _SELECTION_FILE))
        curves.append(pd.read_csv(fold_dir / _CURVE_FILE))
        summaries.append(json.loads((fold_dir / _FOLD_SUMMARY).read_text(encoding="utf-8")))
    return CrossValidationFiles(
        seed=int(summary["seed"]),
        n_parameters=int(summary["n_parameters"]),
        folds=pd.read_csv(directory / _FOLDS_FILE),
        selections=tuple(selections),
        curves=tuple(curves),
        summaries=tuple(summaries),
    )


# --- the gap -----------------------------------------------------------------------
def fold_gaps(files: CrossValidationFiles, hits: pd.DataFrame) -> pd.DataFrame:
    """Per-fold train, inner and out-of-fold agreements of one configuration.

    ``hits`` holds the per-constraint hits of this configuration's
    representation (columns ``anchor, pos, neg, triplet_id, hit``); every
    constraint of ``files.folds`` must appear in it exactly once.
    """
    joined = _join_folds(files.folds, hits)
    rows = []
    for fold, (selection, summary) in enumerate(
        zip(files.selections, files.summaries, strict=True)
    ):
        chosen = _chosen_row(selection, fold)
        if summary["chosen"]["point"] != chosen["point"]:
            raise ValueError(
                f"fold {fold}: fold.json chose {summary['chosen']['point']!r} but "
                f"selection.csv marks {chosen['point']!r}"
            )
        test = joined.loc[joined["fold"] == fold, "hit"]
        if test.empty:
            raise ValueError(f"fold {fold} has no constraint in folds.csv")
        if len(test) != int(summary["n_test_constraints"]):
            raise ValueError(
                f"fold {fold}: {len(test)} test constraints in folds.csv, "
                f"{summary['n_test_constraints']} in fold.json"
            )
        train, inner, out = float(chosen[_TRAIN]), float(chosen[_VAL]), float(test.mean())
        rows.append(
            {
                "fold": fold,
                "point": chosen["point"],
                "best_epoch": int(chosen["best_epoch"]),
                "n_training_constraints": int(summary["n_training_constraints"]),
                "n_inner_constraints": int(summary["n_inner_constraints"]),
                "n_test_constraints": len(test),
                "train_agreement": train,
                "inner_agreement": inner,
                "test_agreement": out,
                "gap": train - out,
                "optimism": inner - out,
            }
        )
    return pd.DataFrame(rows)


def gap_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    """Mean, minimum and maximum over the folds, per variant and seed and per variant.

    ``gaps`` is the concatenation of :func:`fold_gaps` tables with ``variant``
    and ``seed`` columns; the rows pooled over every seed leave ``seed`` empty.
    """
    tables = []
    for keys, pooled in ((["variant", "seed"], False), (["variant"], True)):
        stats = gaps.groupby(keys, sort=False)[list(_SUMMARISED)].agg(["mean", "min", "max"])
        long = stats.stack(level=0, future_stack=True).rename_axis([*keys, "quantity"])
        long = long.reset_index()
        if pooled:
            long.insert(1, "seed", pd.NA)
        tables.append(long)
    summary = pd.concat(tables, ignore_index=True)
    summary["seed"] = summary["seed"].astype("Int64")
    return summary[["variant", "seed", "quantity", "mean", "min", "max"]]


def curve_table(files: CrossValidationFiles) -> pd.DataFrame:
    """The curves of every fold, stacked, with ``chosen_epoch`` marking the kept epoch."""
    tables = []
    for fold, (selection, curve) in enumerate(zip(files.selections, files.curves, strict=True)):
        best = int(_chosen_row(selection, fold)["best_epoch"])
        if best not in set(curve["epoch"]):
            raise ValueError(
                f"fold {fold}: the chosen epoch {best} is not in its curve "
                f"(epochs {curve['epoch'].min()}-{curve['epoch'].max()})"
            )
        table = curve.copy()
        table.insert(0, "fold", fold)
        table["chosen_epoch"] = table["epoch"] == best
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _join_folds(folds: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """``folds`` with the hit of each constraint, checked one to one."""
    joined = folds[[*_KEY, "fold"]].merge(hits[[*_KEY, "hit"]], on=_KEY, how="left")
    if len(joined) != len(folds) or joined["hit"].isna().any():
        missing = int(joined["hit"].isna().sum())
        raise ValueError(
            f"every constraint of folds.csv needs exactly one hit: {len(folds)} constraints, "
            f"{len(joined)} joined rows, {missing} without a hit"
        )
    return joined


def _chosen_row(selection: pd.DataFrame, fold: int) -> pd.Series:
    """The one row of ``selection`` marked as chosen."""
    chosen = selection[selection["chosen"].astype(bool)]
    if len(chosen) != 1:
        raise ValueError(
            f"fold {fold}: selection.csv marks {len(chosen)} chosen points, expected 1"
        )
    return chosen.iloc[0]


# --- the figure --------------------------------------------------------------------
def plot_validation_curves(
    curves: pd.DataFrame,
    gaps: pd.DataFrame,
    *,
    reference: float,
    n_parameters: Mapping[str, int],
) -> Figure:
    """One panel per variant of :data:`VARIANTS`: every curve, its chosen epoch, the references.

    ``curves`` is the concatenation of :func:`curve_table` tables with
    ``variant`` and ``seed`` columns, ``gaps`` that of :func:`fold_gaps`;
    ``reference`` is the agreement of the input representation and
    ``n_parameters`` the size of each variant, for the panel titles.
    """
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    missing = sorted(set(VARIANTS) - set(curves["variant"]))
    if missing:
        raise ValueError(f"no curves for the variant(s) {missing}; expected {list(VARIANTS)}")

    figure = Figure(figsize=_FIGURE_SIZE, layout="constrained")
    axes = figure.subplots(1, len(VARIANTS), sharey=True)
    for ax, variant in zip(axes, VARIANTS, strict=True):
        of_variant = curves[curves["variant"] == variant]
        for _, curve in of_variant.groupby(["seed", "fold"], sort=True):
            best = curve[curve["chosen_epoch"]]
            for column, colour in ((_TRAIN, _TRAIN_COLOUR), (_VAL, _VAL_COLOUR)):
                ax.plot(
                    curve["epoch"],
                    curve[column],
                    color=colour,
                    linewidth=_CURVE_WIDTH,
                    alpha=_CURVE_ALPHA,
                )
                ax.plot(best["epoch"], best[column], "o", color=colour, markersize=_MARKER_SIZE)
        out_of_fold = float(gaps.loc[gaps["variant"] == variant, "test_agreement"].mean())
        # The two references can be close: one label above its line, one below.
        for value, style, text, offset, align in (
            (out_of_fold, ":", f"out of fold {out_of_fold:.3f}", _LABEL_OFFSET, "bottom"),
            (reference, "--", f"{_REFERENCE} {reference:.3f}", -_LABEL_OFFSET, "top"),
        ):
            ax.axhline(value, color=_REFERENCE_COLOUR, linestyle=style, linewidth=_REFERENCE_WIDTH)
            ax.annotate(
                text,
                xy=(1.0, value),
                xycoords=("axes fraction", "data"),
                xytext=(-_LABEL_OFFSET, offset),
                textcoords="offset points",
                ha="right",
                va=align,
                fontsize="x-small",
                color=_REFERENCE_COLOUR,
            )
        ax.set_title(f"{_TITLES[variant]} ({n_parameters[variant]:,} parameters)", fontsize="small")
        ax.set_xlabel("epoch", fontsize="small")
        ax.tick_params(labelsize="x-small", color=_GRID_COLOUR)
        ax.grid(True, color=_GRID_COLOUR, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("unweighted agreement", fontsize="small")
    handles = [
        Line2D([], [], color=_TRAIN_COLOUR, label="inner training split"),
        Line2D([], [], color=_VAL_COLOUR, label="inner validation fold"),
        Line2D([], [], color=_REFERENCE_COLOUR, marker="o", linestyle="", label="chosen epoch"),
    ]
    figure.legend(
        handles=handles, loc="outside upper center", ncols=3, fontsize="x-small", frameon=False
    )
    return figure


# --- locating the checkpoints ------------------------------------------------------
def locate_checkpoints(config: Config) -> list[HumanModel]:
    """The ``human_siamese`` representations of ``config`` and their checkpoint directories.

    Each directory is the one the extractor itself would use
    (:meth:`~musicsim.extractors.human_siamese.HumanSiameseExtractor.checkpoint_dir`).
    Needs torch, which the extractor imports.
    """
    from musicsim.extractors.human_siamese import HumanSiameseExtractor
    from musicsim.representations import parse_representations, spec_config

    models = []
    for spec in parse_representations(config):
        if spec.extractor != _EXTRACTOR:
            continue
        extractor = HumanSiameseExtractor(spec_config(config, spec))
        hidden = extractor.config.require("extractor.hidden")
        models.append(
            HumanModel(
                label=spec.label,
                variant="linear" if len(hidden) == 0 else "mlp",
                seed=int(extractor.config.require("seed")),
                directory=extractor.checkpoint_dir(extractor.mtt_config()),
            )
        )
    if not models:
        raise ConfigError(f"the run has no {_EXTRACTOR!r} representation to analyse")
    return models


# --- the analysis ------------------------------------------------------------------
def analyse(run_dir: Path, *, report_dir: Path | None) -> Path:
    """Analyse the e12 run in ``run_dir``; return the directory of the analysis run.

    The PDF copy of the figure goes to ``report_dir`` unless it is ``None``.
    """
    run_dir = Path(run_dir)
    source = Config(
        data=yaml.safe_load((run_dir / _RESOLVED_CONFIG).read_text(encoding="utf-8")),
        source=run_dir / _RESOLVED_CONFIG,
    )
    models = locate_checkpoints(source)
    per_constraint = pd.read_csv(run_dir / "metrics" / _PER_CONSTRAINT)
    reference = _reference_agreement(pd.read_csv(run_dir / "metrics" / _TRIPLETS))

    config = Config(data={"experiment": EXPERIMENT, "source_run": paths.relative_to_repo(run_dir)})
    run = start_run(config)
    try:
        with run.stage("analyse"):
            gaps, curves, sizes = _collect(models, per_constraint)
            gaps.to_csv(run.metrics_dir / GAP_FILE, index=False)
            gap_summary(gaps).to_csv(run.metrics_dir / SUMMARY_FILE, index=False)
            curves.to_csv(run.metrics_dir / CURVES_FILE, index=False)
            figure = plot_validation_curves(curves, gaps, reference=reference, n_parameters=sizes)
            figure.savefig(run.figures_dir / f"{FIGURE_STEM}.png", dpi=_PNG_DPI)
            if report_dir is not None:
                report_dir.mkdir(parents=True, exist_ok=True)
                figure.savefig(Path(report_dir) / f"{FIGURE_STEM}.pdf")
            run.record["checkpoints"] = {
                model.label: paths.relative_to_repo(model.directory) for model in models
            }
    except BaseException:
        run.finish("failed")
        raise
    run.finish()
    return run.directory


def _collect(
    models: Sequence[HumanModel], per_constraint: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Gap and curve tables of every model, and the parameter count of each variant."""
    gaps, curves, sizes = [], [], {}
    for model in models:
        files = read_cross_validation(model.directory)
        if files.seed != model.seed:
            raise ValueError(
                f"{model.label}: {model.directory} was trained with seed {files.seed}, "
                f"the run asks for {model.seed}"
            )
        hits = per_constraint[per_constraint["representation"] == model.label]
        if hits.empty:
            raise ValueError(f"{model.label} has no row in {_PER_CONSTRAINT}")
        labels = {"representation": model.label, "variant": model.variant, "seed": model.seed}
        gaps.append(fold_gaps(files, hits).assign(**labels))
        curves.append(curve_table(files).assign(**labels))
        sizes[model.variant] = files.n_parameters
        logger.info("%s: %d folds from %s", model.label, len(files.selections), model.directory)
    front = ["representation", "variant", "seed"]
    gap_table = pd.concat(gaps, ignore_index=True)
    curve_frame = pd.concat(curves, ignore_index=True)
    return (
        gap_table[front + [c for c in gap_table.columns if c not in front]],
        curve_frame[front + [c for c in curve_frame.columns if c not in front]],
        sizes,
    )


def _reference_agreement(triplets: pd.DataFrame) -> float:
    """Headline agreement of the reference representation in a ``triplets.csv``."""
    row = triplets[
        (triplets["representation"] == _REFERENCE) & (triplets["variant"] == _REFERENCE_VARIANT)
    ]
    if len(row) != 1:
        raise ValueError(
            f"{_TRIPLETS} has {len(row)} {_REFERENCE_VARIANT!r} rows for {_REFERENCE!r}, expected 1"
        )
    return float(row["value"].iloc[0])


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: ``python -m musicsim.evaluation.overfitting --run <e12 run>``."""
    parser = argparse.ArgumentParser(prog="python -m musicsim.evaluation.overfitting")
    parser.add_argument("--run", type=Path, required=True, help="finished run of e12 on MTT")
    args = parser.parse_args(argv)
    analyse(args.run, report_dir=paths.report_figure_dir())
    return 0


if __name__ == "__main__":
    # Run the package's copy of the module, whose logger sits under the run's
    # ("musicsim...") and so reaches log.txt; this copy's would be "__main__".
    from musicsim.evaluation import overfitting

    raise SystemExit(overfitting.main())
