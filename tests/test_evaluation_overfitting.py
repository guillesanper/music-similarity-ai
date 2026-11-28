"""Tests for the overfitting gap and the validation curves of the heads trained on human judgements.

The synthetic cross-validation has two folds of two constraints each, written by
hand in the layout of :mod:`musicsim.extractors.human_siamese`:

* fold 0 holds triplet 0, ``(1, 2, 3)`` and ``(3, 2, 1)``, whose hits are 1 and
  0 (out of fold 0.5); its chosen point ``a`` has train 0.9 and inner 0.75 at
  epoch 2, so the gap is 0.4 and the optimism 0.25;
* fold 1 holds triplet 1, ``(4, 5, 6)`` and ``(6, 5, 4)``, both hits 1 (out of
  fold 1.0); its chosen point ``b`` has train 1.0 and inner 0.5 at epoch 1, so
  the gap is 0.0 and the optimism -0.5.

The tests that locate checkpoints through the extractor need torch and skip
themselves without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from musicsim import paths
from musicsim.config import Config, ConfigError, load_config
from musicsim.evaluation import overfitting
from musicsim.evaluation.overfitting import (
    CURVES_FILE,
    GAP_FILE,
    SUMMARY_FILE,
    VARIANTS,
    CrossValidationFiles,
    analyse,
    curve_table,
    fold_gaps,
    gap_summary,
    locate_checkpoints,
    plot_validation_curves,
    read_cross_validation,
)

E12_MTT = paths.REPO_ROOT / "configs" / "experiments" / "e12_genre_vs_human_mtt.yaml"
N_PARAMETERS = 1234

FOLDS = pd.DataFrame(
    {
        "anchor": [1, 3, 4, 6],
        "pos": [2, 2, 5, 5],
        "neg": [3, 1, 6, 4],
        "margin": [2, 1, 3, 1],
        "triplet_id": [0, 0, 1, 1],
        "fold": [0, 0, 1, 1],
    }
)
HITS = FOLDS[["anchor", "pos", "neg", "triplet_id"]].assign(hit=[1.0, 0.0, 1.0, 1.0])

# Per fold: the two grid points, the chosen one, and its epoch and agreements.
SELECTIONS = (
    pd.DataFrame(
        {
            "point": ["a", "b"],
            "best_epoch": [2, 1],
            "val_agreement": [0.75, 0.70],
            "train_agreement": [0.90, 0.80],
            "chosen": [True, False],
        }
    ),
    pd.DataFrame(
        {
            "point": ["a", "b"],
            "best_epoch": [3, 1],
            "val_agreement": [0.40, 0.50],
            "train_agreement": [0.95, 1.00],
            "chosen": [False, True],
        }
    ),
)
CURVES = (
    pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [0.3, 0.2, 0.1],
            "val_agreement": [0.70, 0.75, 0.72],
            "train_agreement": [0.80, 0.90, 0.95],
        }
    ),
    pd.DataFrame(
        {
            "epoch": [1, 2],
            "train_loss": [0.3, 0.2],
            "val_agreement": [0.50, 0.45],
            "train_agreement": [1.00, 1.00],
        }
    ),
)
CHOSEN = ("a", "b")


def _fold_summary(fold: int) -> dict:
    return {
        "chosen": {"point": CHOSEN[fold]},
        "n_training_constraints": 5,
        "n_inner_constraints": 1,
        "n_test_constraints": 2,
    }


def _files(seed: int = 0, **changes: object) -> CrossValidationFiles:
    values = {
        "seed": seed,
        "n_parameters": N_PARAMETERS,
        "folds": FOLDS,
        "selections": SELECTIONS,
        "curves": CURVES,
        "summaries": (_fold_summary(0), _fold_summary(1)),
    }
    values.update(changes)
    return CrossValidationFiles(**values)


def _write_cross_validation(directory: Path, seed: int) -> None:
    """The synthetic cross-validation, in the files the extractor writes."""
    directory.mkdir(parents=True)
    FOLDS.to_csv(directory / "folds.csv", index=False)
    (directory / "cross_validation.json").write_text(
        json.dumps({"n_folds": 2, "seed": seed, "n_parameters": N_PARAMETERS}), encoding="utf-8"
    )
    for fold in range(2):
        fold_dir = directory / f"fold_{fold}"
        fold_dir.mkdir()
        SELECTIONS[fold].to_csv(fold_dir / "selection.csv", index=False)
        CURVES[fold].to_csv(fold_dir / "curve.csv", index=False)
        (fold_dir / "fold.json").write_text(json.dumps(_fold_summary(fold)), encoding="utf-8")


def _gaps_with(variant: str, seed: int) -> pd.DataFrame:
    return fold_gaps(_files(seed), HITS).assign(variant=variant, seed=seed)


def _curves_with(variant: str, seed: int) -> pd.DataFrame:
    return curve_table(_files(seed)).assign(variant=variant, seed=seed)


# --- reading -------------------------------------------------------------------------
def test_read_cross_validation_round_trips(tmp_path: Path) -> None:
    _write_cross_validation(tmp_path / "cv", seed=7)
    files = read_cross_validation(tmp_path / "cv")
    assert (files.seed, files.n_parameters) == (7, N_PARAMETERS)
    pd.testing.assert_frame_equal(files.folds, FOLDS)
    for read, written in zip(files.curves, CURVES, strict=True):
        pd.testing.assert_frame_equal(read, written)
    assert [summary["chosen"]["point"] for summary in files.summaries] == list(CHOSEN)


def test_read_cross_validation_needs_a_finished_cross_validation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="has not finished"):
        read_cross_validation(tmp_path)


# --- the gap -------------------------------------------------------------------------
def test_fold_gaps_hand_computed() -> None:
    gaps = fold_gaps(_files(), HITS)
    assert gaps["point"].tolist() == ["a", "b"]
    assert gaps["best_epoch"].tolist() == [2, 1]
    assert gaps["n_test_constraints"].tolist() == [2, 2]
    np.testing.assert_allclose(gaps["train_agreement"], [0.9, 1.0])
    np.testing.assert_allclose(gaps["inner_agreement"], [0.75, 0.5])
    np.testing.assert_allclose(gaps["test_agreement"], [0.5, 1.0])
    np.testing.assert_allclose(gaps["gap"], [0.4, 0.0], atol=1e-12)
    np.testing.assert_allclose(gaps["optimism"], [0.25, -0.5])


def test_fold_gaps_ignores_hits_of_other_constraints() -> None:
    extra = pd.DataFrame({"anchor": [9], "pos": [8], "neg": [7], "triplet_id": [5], "hit": [0.0]})
    gaps = fold_gaps(_files(), pd.concat([HITS, extra], ignore_index=True))
    np.testing.assert_allclose(gaps["test_agreement"], [0.5, 1.0])


def test_fold_gaps_needs_a_hit_for_every_constraint() -> None:
    with pytest.raises(ValueError, match="1 without a hit"):
        fold_gaps(_files(), HITS.iloc[1:])


def test_fold_gaps_rejects_duplicated_hits() -> None:
    with pytest.raises(ValueError, match="5 joined rows"):
        fold_gaps(_files(), pd.concat([HITS, HITS.iloc[:1]], ignore_index=True))


@pytest.mark.parametrize("chosen", [[False, False], [True, True]])
def test_fold_gaps_needs_one_chosen_point(chosen: list[bool]) -> None:
    selections = (SELECTIONS[0].assign(chosen=chosen), SELECTIONS[1])
    with pytest.raises(ValueError, match=f"marks {sum(chosen)} chosen points"):
        fold_gaps(_files(selections=selections), HITS)


def test_fold_gaps_checks_the_chosen_point_against_fold_json() -> None:
    summaries = ({**_fold_summary(0), "chosen": {"point": "b"}}, _fold_summary(1))
    with pytest.raises(ValueError, match=r"fold.json chose 'b' but selection.csv marks 'a'"):
        fold_gaps(_files(summaries=summaries), HITS)


def test_fold_gaps_checks_the_test_size_against_fold_json() -> None:
    summaries = (_fold_summary(0), {**_fold_summary(1), "n_test_constraints": 3})
    with pytest.raises(ValueError, match=r"fold 1: 2 test constraints in folds.csv, 3"):
        fold_gaps(_files(summaries=summaries), HITS)


def test_fold_gaps_rejects_a_fold_without_constraints() -> None:
    folds = FOLDS.assign(fold=1)
    with pytest.raises(ValueError, match="fold 0 has no constraint"):
        fold_gaps(_files(folds=folds), HITS)


def test_gap_summary_per_seed_and_pooled() -> None:
    second = _gaps_with("mlp", 1)
    second["gap"] = [0.2, 0.6]
    summary = gap_summary(pd.concat([_gaps_with("mlp", 0), second], ignore_index=True))
    gap = summary[summary["quantity"] == "gap"].set_index("seed", drop=False)
    assert len(summary) == 3 * 5  # seeds 0 and 1 and pooled, five quantities
    np.testing.assert_allclose(gap.loc[0, ["mean", "min", "max"]].astype(float), [0.2, 0.0, 0.4])
    np.testing.assert_allclose(gap.loc[1, ["mean", "min", "max"]].astype(float), [0.4, 0.2, 0.6])
    pooled = summary[summary["seed"].isna() & (summary["quantity"] == "gap")]
    np.testing.assert_allclose(pooled[["mean", "min", "max"]].to_numpy()[0], [0.3, 0.0, 0.6])
    assert summary.columns.tolist() == ["variant", "seed", "quantity", "mean", "min", "max"]


# --- the curves ----------------------------------------------------------------------
def test_curve_table_marks_the_chosen_epoch() -> None:
    table = curve_table(_files())
    assert len(table) == 5
    chosen = table[table["chosen_epoch"]]
    assert list(zip(chosen["fold"], chosen["epoch"], strict=True)) == [(0, 2), (1, 1)]
    np.testing.assert_allclose(chosen["train_agreement"], [0.9, 1.0])


def test_curve_table_needs_the_chosen_epoch_in_the_curve() -> None:
    curves = (CURVES[0].iloc[:1], CURVES[1])
    with pytest.raises(ValueError, match="chosen epoch 2 is not in its curve"):
        curve_table(_files(curves=curves))


def test_plot_validation_curves_draws_every_curve_and_both_references() -> None:
    curves = pd.concat([_curves_with(v, 0) for v in VARIANTS], ignore_index=True)
    gaps = pd.concat(
        [_gaps_with("mlp", 0), _gaps_with("linear", 0).assign(test_agreement=[0.6, 0.7])]
    )
    figure = plot_validation_curves(
        curves, gaps, reference=0.68, n_parameters={"mlp": 10, "linear": 5}
    )
    axes = figure.get_axes()
    assert [ax.get_title() for ax in axes] == [
        "MLP head (10 parameters)",
        "Linear map (5 parameters)",
    ]
    for ax, out_of_fold in zip(axes, (0.75, 0.65), strict=True):
        # Two folds, each a line and a chosen-epoch marker for both agreements,
        # then the out-of-fold mean and the reference.
        lines = ax.get_lines()
        assert len(lines) == 2 * 4 + 2
        np.testing.assert_allclose(
            [lines[-2].get_ydata()[0], lines[-1].get_ydata()[0]], [out_of_fold, 0.68]
        )


def test_plot_validation_curves_needs_both_variants() -> None:
    with pytest.raises(ValueError, match=r"variant\(s\) \['linear'\]"):
        plot_validation_curves(
            _curves_with("mlp", 0), _gaps_with("mlp", 0), reference=0.5, n_parameters={"mlp": 1}
        )


def test_reference_agreement_needs_one_row() -> None:
    triplets = pd.DataFrame(
        {
            "representation": ["mert", "mert"],
            "variant": ["unweighted", "vote_weighted"],
            "value": [0.6, 0.7],
        }
    )
    assert overfitting._reference_agreement(triplets) == 0.6
    with pytest.raises(ValueError, match="has 0 'unweighted' rows for 'mert'"):
        overfitting._reference_agreement(triplets.iloc[1:])


# --- locating the checkpoints and the whole analysis (torch) -------------------------
def test_file_names_match_the_cross_validation() -> None:
    pytest.importorskip("torch")
    from musicsim.extractors import human_siamese
    from musicsim.training import constraints, train

    assert overfitting._FOLDS_FILE == human_siamese.FOLDS_FILE
    assert overfitting._CROSS_VALIDATION_SUMMARY == human_siamese.CROSS_VALIDATION_SUMMARY
    assert overfitting._SELECTION_FILE == constraints.SELECTION_FILE
    assert overfitting._FOLD_SUMMARY == constraints.FOLD_SUMMARY
    assert overfitting._CURVE_FILE == train.CURVE_FILE
    assert (overfitting._TRAIN, overfitting._VAL) == (
        constraints.TRAIN_AGREEMENT,
        constraints.VAL_AGREEMENT,
    )


def test_locate_checkpoints_of_e12(isolated_roots: dict[str, Path]) -> None:
    pytest.importorskip("torch")
    from musicsim.extractors.human_siamese import HumanSiameseExtractor
    from musicsim.representations import parse_representations, spec_config

    config = load_config(E12_MTT)
    models = locate_checkpoints(config)
    assert [(m.label, m.variant, m.seed) for m in models] == [
        (f"human_{variant}_s{seed}", variant, seed) for variant in VARIANTS for seed in range(3)
    ]
    assert len({m.directory for m in models}) == 6
    spec = next(s for s in parse_representations(config) if s.label == "human_linear_s1")
    extractor = HumanSiameseExtractor(spec_config(config, spec))
    assert models[4].directory == extractor.checkpoint_dir(extractor.mtt_config())
    assert models[4].directory.parent == isolated_roots["output"] / "checkpoints"


def test_locate_checkpoints_needs_a_human_siamese_representation() -> None:
    pytest.importorskip("torch")
    config = load_config(E12_MTT)
    data = config.to_dict()
    data["representations"] = ["mert", "mfcc"]
    with pytest.raises(ConfigError, match="no 'human_siamese' representation"):
        locate_checkpoints(Config(data=data))


def _e12_run(root: Path, *, seed_offset: int = 0) -> Path:
    """A finished e12 run on MTT whose checkpoints hold the synthetic cross-validation."""
    pytest.importorskip("torch")
    config = load_config(E12_MTT)
    run_dir = root / "e12_run"
    (run_dir / "metrics").mkdir(parents=True)
    config.write(run_dir / "config.resolved.yaml")
    models = locate_checkpoints(config)
    for model in models:
        _write_cross_validation(model.directory, seed=model.seed + seed_offset)
    hits = pd.concat([HITS.assign(representation=m.label) for m in models], ignore_index=True)
    hits.to_csv(run_dir / "metrics" / "triplets_per_constraint.csv", index=False)
    pd.DataFrame({"representation": ["mert"], "variant": ["unweighted"], "value": [0.68]}).to_csv(
        run_dir / "metrics" / "triplets.csv", index=False
    )
    return run_dir


def test_analyse_writes_its_own_run(isolated_roots: dict[str, Path], tmp_path: Path) -> None:
    run_dir = _e12_run(tmp_path)
    out = analyse(run_dir, report_dir=tmp_path / "report")
    assert out.parent == isolated_roots["output"] / "runs" / "e12_overfitting"

    gaps = pd.read_csv(out / "metrics" / GAP_FILE)
    assert len(gaps) == 6 * 2
    assert gaps.columns[:4].tolist() == ["representation", "variant", "seed", "fold"]
    np.testing.assert_allclose(gaps["gap"], [0.4, 0.0] * 6, atol=1e-12)
    assert len(pd.read_csv(out / "metrics" / SUMMARY_FILE)) == (6 + 2) * 5
    assert len(pd.read_csv(out / "metrics" / CURVES_FILE)) == 6 * 5
    assert (out / "figures" / "validation_curves.png").is_file()
    assert (tmp_path / "report" / "validation_curves.pdf").is_file()

    record = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "ok"
    assert sorted(record["checkpoints"]) == sorted(gaps["representation"].unique())
    assert (run_dir / "metrics").exists() and not (run_dir / "figures").exists()


def test_analyse_rejects_a_checkpoint_of_another_seed(
    isolated_roots: dict[str, Path], tmp_path: Path
) -> None:
    run_dir = _e12_run(tmp_path, seed_offset=1)
    with pytest.raises(ValueError, match="trained with seed 1, the run asks for 0"):
        analyse(run_dir, report_dir=None)
