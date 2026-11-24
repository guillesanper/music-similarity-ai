"""Tests for training on similarity constraints, with nested selection per fold.

The synthetic corpus has 12 triplets of clips, each its own connected component
with two constraints, ``(a, b, c)`` and ``(c, b, a)``: 36 clips and 24
constraints. Three outer folds hold 8 constraints each, so the training side of
a fold has 16, split into two inner folds of 8. Clip ids start at
:data:`FIRST_ID`, so they never coincide with row positions. Skipped as a whole
without the optional deep-learning extra.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402 (after the torch skip)

from musicsim.datasets.mtt_folds import constraint_folds  # noqa: E402
from musicsim.seeding import generator  # noqa: E402
from musicsim.training.constraints import (  # noqa: E402
    FOLD_SUMMARY,
    SELECTION_FILE,
    TRAIN_AGREEMENT,
    VAL_AGREEMENT,
    AgreementValidator,
    ConstraintTask,
    ConstraintTrainingSettings,
    GridPoint,
    constraint_triplet_loss,
    train_fold,
)
from musicsim.training.models import MlpHead, count_parameters  # noqa: E402
from musicsim.training.train import (  # noqa: E402
    BEST_CHECKPOINT,
    CURVE_FILE,
    TrainingSettings,
    train,
)

SEED = 5

N_TRIPLETS = 12
CLIPS_PER_TRIPLET = 3
N_CLIPS = N_TRIPLETS * CLIPS_PER_TRIPLET
N_CONSTRAINTS = 2 * N_TRIPLETS
N_DIMS = 8
FIRST_ID = 1000

N_FOLDS = 3
CONSTRAINTS_PER_FOLD = N_CONSTRAINTS // N_FOLDS
TEST_FOLD = 1

GRID = (GridPoint(margin=0.2, lr=1e-2, weight_decay=0.0), GridPoint(0.5, 1e-3, 1e-4))

#: Hand-made clips for known values: ids 10 -> (1, 0), 30 -> (0, 1), 20 -> (-1, 0).
HAND_IDS = np.array([20, 10, 30])
HAND_FEATURES = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
#: d(10, 30) = 1 and d(10, 20) = 2: the first constraint holds with a margin of 1,
#: the second is broken by 1.
HAND_CONSTRAINTS = pd.DataFrame({"anchor": [10, 10], "pos": [30, 20], "neg": [20, 30]})
#: Its loss with margin 0.5: (max(0, 1 - 2 + 0.5) + max(0, 2 - 1 + 0.5)) / 2.
HAND_LOSS = 0.75


# --- fixtures and helpers ------------------------------------------------------
@pytest.fixture
def constraints() -> pd.DataFrame:
    rows = []
    for t in range(N_TRIPLETS):
        a, b, c = FIRST_ID + CLIPS_PER_TRIPLET * t + np.arange(CLIPS_PER_TRIPLET)
        rows += [(a, b, c, t), (c, b, a, t)]
    return pd.DataFrame(rows, columns=["anchor", "pos", "neg", "triplet_id"])


@pytest.fixture
def folds(constraints: pd.DataFrame) -> pd.Series:
    return constraint_folds(
        constraints, n_folds=N_FOLDS, max_imbalance=0.1, generator=generator(SEED, "folds")
    )


@pytest.fixture
def ids() -> np.ndarray:
    return FIRST_ID + np.arange(N_CLIPS)


@pytest.fixture
def features(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(N_CLIPS, N_DIMS)).astype(np.float32)


def _settings(*, hidden: tuple[int, ...] = (16,), dropout: float = 0.0, grid=GRID):
    return ConstraintTrainingSettings(
        hidden=hidden,
        dim=4,
        dropout=dropout,
        grid=grid,
        batch_size=4,
        max_epochs=3,
        patience=10,
        inner_folds=2,
        max_imbalance=0.1,
    )


def _clips(table: pd.DataFrame) -> set[int]:
    return set(table[["anchor", "pos", "neg"]].to_numpy().ravel().tolist())


def _assert_same_state(first: dict, second: dict) -> None:
    assert list(first) == list(second)
    for name, tensor in first.items():
        assert torch.equal(tensor, second[name]), name


def _fold(features, ids, constraints, folds, out_dir: Path, **kwargs):
    return train_fold(
        features,
        ids,
        constraints,
        folds,
        test_fold=TEST_FOLD,
        settings=kwargs.pop("settings", _settings()),
        seed=kwargs.pop("seed", SEED),
        out_dir=out_dir,
    )


# --- loss --------------------------------------------------------------------
def test_loss_known_value():
    embeddings = torch.from_numpy(HAND_FEATURES[[1, 2, 0]])  # rows 10, 30, 20
    loss = constraint_triplet_loss(
        embeddings, np.array([0, 0]), np.array([1, 2]), np.array([2, 1]), margin=0.5
    )
    assert loss.item() == pytest.approx(HAND_LOSS)


def test_loss_is_zero_once_every_constraint_holds_by_the_margin():
    embeddings = torch.from_numpy(HAND_FEATURES[[1, 2, 0]])
    loss = constraint_triplet_loss(
        embeddings, np.array([0]), np.array([1]), np.array([2]), margin=1.0
    )
    assert loss.item() == 0.0


@pytest.mark.parametrize(
    ("anchor", "pos", "neg", "margin", "message"),
    [
        ([0], [1], [2], 0.0, "margin"),
        ([0], [1], [2], 2.5, "margin"),
        ([0, 1], [1], [2], 0.5, "aligned"),
        ([], [], [], 0.5, "non-empty"),
        ([0], [1], [3], 0.5, "lie in"),
        ([0], [0], [2], 0.5, "same clip twice"),
        ([0.0], [1.0], [2.0], 0.5, "integer positions"),
    ],
)
def test_loss_rejects_invalid_input(anchor, pos, neg, margin, message):
    embeddings = torch.from_numpy(HAND_FEATURES)
    with pytest.raises(ValueError, match=message):
        constraint_triplet_loss(
            embeddings, np.array(anchor), np.array(pos), np.array(neg), margin=margin
        )


def test_loss_rejects_a_bad_embedding_batch():
    with pytest.raises(ValueError, match="non-empty"):
        constraint_triplet_loss(torch.zeros(3), [0], [1], [2], margin=0.5)
    with pytest.raises(ValueError, match="floating point"):
        constraint_triplet_loss(torch.zeros(3, 2, dtype=torch.int64), [0], [1], [2], margin=0.5)


# --- task and validator ------------------------------------------------------------
def test_task_loss_looks_clips_up_by_id():
    task = ConstraintTask(HAND_FEATURES, HAND_IDS, HAND_CONSTRAINTS, margin=0.5, batch_size=2)
    loss = task.loss(nn.Identity(), np.array([0, 1]), generator(SEED, "test"))
    assert loss.item() == pytest.approx(HAND_LOSS)


def test_task_epoch_covers_every_constraint_once(constraints, features, ids):
    task = ConstraintTask(features, ids, constraints.head(10), margin=0.5, batch_size=4)

    batches = task.batches(generator(SEED, "test"))

    assert [len(batch) for batch in batches] == [4, 4, 2]
    assert sorted(np.concatenate(batches).tolist()) == list(range(10))
    again = task.batches(generator(SEED, "test"))
    assert all(np.array_equal(a, b) for a, b in zip(batches, again, strict=True))


def test_validator_scores_both_agreements():
    validator = AgreementValidator(
        HAND_FEATURES, HAND_IDS, HAND_CONSTRAINTS.iloc[[0]], HAND_CONSTRAINTS.iloc[[1]]
    )
    assert validator.columns == (VAL_AGREEMENT, TRAIN_AGREEMENT)
    assert validator.score(nn.Identity(), None) == {VAL_AGREEMENT: 0.0, TRAIN_AGREEMENT: 1.0}


@pytest.mark.parametrize(
    ("ids", "table", "kwargs", "message"),
    [
        (np.array([20, 10, 31]), HAND_CONSTRAINTS, {}, "no feature row"),
        (np.array([20, 10, 10]), HAND_CONSTRAINTS, {}, "unique"),
        (np.array([20, 10]), HAND_CONSTRAINTS, {}, "one entry per feature row"),
        (HAND_IDS, HAND_CONSTRAINTS.drop(columns="neg"), {}, "missing"),
        (HAND_IDS, HAND_CONSTRAINTS.iloc[:0], {}, "no training constraints"),
        (HAND_IDS, HAND_CONSTRAINTS, {"batch_size": 0}, "batch_size"),
        (HAND_IDS, HAND_CONSTRAINTS, {"batch_size": True}, "batch_size"),
        (HAND_IDS, HAND_CONSTRAINTS, {"margin": 3.0}, "margin"),
    ],
)
def test_task_rejects_invalid_input(ids, table, kwargs, message):
    arguments = {"margin": 0.5, "batch_size": 2, **kwargs}
    with pytest.raises(ValueError, match=message):
        ConstraintTask(HAND_FEATURES, ids, table, **arguments)


@pytest.mark.parametrize("empty", ["training", "validation"])
def test_validator_rejects_an_empty_side(empty):
    sides = {"training": HAND_CONSTRAINTS, "validation": HAND_CONSTRAINTS}
    sides[empty] = HAND_CONSTRAINTS.iloc[:0]
    with pytest.raises(ValueError, match=f"no {empty} constraints"):
        AgreementValidator(HAND_FEATURES, HAND_IDS, sides["training"], sides["validation"])


def test_overfits_a_tiny_set_of_constraints(constraints, features, ids, tmp_path):
    task = ConstraintTask(features, ids, constraints, margin=0.2, batch_size=8)
    validator = AgreementValidator(features, ids, constraints, constraints)
    model = MlpHead(N_DIMS, (32,), 8, dropout=0.0, generator=torch.Generator().manual_seed(SEED))

    train(
        model,
        task,
        validator,
        TrainingSettings(lr=1e-2, max_epochs=150, patience=150),
        tmp_path,
        seed=SEED,
    )

    curve = pd.read_csv(tmp_path / CURVE_FILE)
    assert curve[TRAIN_AGREEMENT].iloc[0] < 1.0
    assert curve[TRAIN_AGREEMENT].iloc[-1] == 1.0


# --- one fold ------------------------------------------------------------------
def test_test_fold_never_reaches_training_or_selection(constraints, folds, features, ids, tmp_path):
    test_rows = constraints.index[folds == TEST_FOLD]
    test_clips = _clips(constraints.loc[test_rows])
    poisoned = features.copy()
    # A test clip that reached the network would fail on these NaNs.
    poisoned[np.isin(ids, list(test_clips))] = np.nan

    result = _fold(poisoned, ids, constraints, folds, tmp_path)

    seen = np.concatenate([result.training_rows, result.inner_rows])
    assert len(result.inner_rows) == CONSTRAINTS_PER_FOLD
    assert sorted(seen.tolist()) == sorted(constraints.index[folds != TEST_FOLD].tolist())
    assert not set(seen.tolist()) & set(test_rows.tolist())
    assert not _clips(constraints.loc[seen]) & test_clips
    assert not _clips(constraints.loc[result.training_rows]) & _clips(
        constraints.loc[result.inner_rows]
    )


def test_fold_files_and_selection(constraints, folds, features, ids, tmp_path):
    result = _fold(features, ids, constraints, folds, tmp_path)

    selection = pd.read_csv(tmp_path / SELECTION_FILE)
    pd.testing.assert_frame_equal(selection, result.selection)
    assert selection["point"].tolist() == [point.label for point in GRID]
    best = selection[VAL_AGREEMENT].idxmax()  # the first maximum
    assert selection["chosen"].tolist() == [i == best for i in range(len(GRID))]
    assert result.chosen == GRID[best]

    chosen_dir = tmp_path / "inner" / result.chosen.label
    for name in (BEST_CHECKPOINT, CURVE_FILE):
        assert (tmp_path / name).read_bytes() == (chosen_dir / name).read_bytes()
    _assert_same_state(
        result.model.state_dict(), torch.load(tmp_path / BEST_CHECKPOINT, weights_only=True)
    )
    curve = pd.read_csv(chosen_dir / CURVE_FILE)
    row = selection.loc[best]
    assert row[VAL_AGREEMENT] == curve[VAL_AGREEMENT].max()
    assert row[TRAIN_AGREEMENT] == curve[TRAIN_AGREEMENT].iloc[row["best_epoch"] - 1]

    summary = json.loads((tmp_path / FOLD_SUMMARY).read_text(encoding="utf-8"))
    assert summary["test_fold"] == TEST_FOLD
    assert summary["chosen"]["point"] == result.chosen.label
    assert summary["n_parameters"] == count_parameters(result.model) == result.n_parameters
    assert summary["n_test_constraints"] == CONSTRAINTS_PER_FOLD
    assert summary["n_inner_constraints"] == CONSTRAINTS_PER_FOLD
    assert summary["n_training_constraints"] == CONSTRAINTS_PER_FOLD
    # 8 triplets of 3 clips outside the test fold.
    assert summary["n_training_clips"] == (N_TRIPLETS - N_TRIPLETS // N_FOLDS) * CLIPS_PER_TRIPLET


def test_every_grid_point_starts_from_the_same_weights(constraints, folds, features, ids, tmp_path):
    # With one epoch and the same initial weights, two points that only differ
    # in an lr too small to move anything give the same scores.
    grid = (GridPoint(0.2, 1e-12, 0.0), GridPoint(0.5, 1e-12, 0.0))
    result = _fold(features, ids, constraints, folds, tmp_path, settings=_settings(grid=grid))

    assert result.selection[VAL_AGREEMENT].nunique() == 1
    assert result.chosen == grid[0]  # the tie goes to the first point


@pytest.mark.parametrize("hidden", [(16,), ()], ids=["mlp", "linear"])
def test_same_seed_gives_the_same_fold(hidden, constraints, folds, features, ids, tmp_path):
    settings = _settings(hidden=hidden, dropout=0.2)
    first = _fold(features, ids, constraints, folds, tmp_path / "a", settings=settings)
    second = _fold(features, ids, constraints, folds, tmp_path / "b", settings=settings)

    pd.testing.assert_frame_equal(first.selection, second.selection)
    _assert_same_state(first.model.state_dict(), second.model.state_dict())
    assert np.array_equal(first.inner_rows, second.inner_rows)


def test_another_seed_gives_another_fold(constraints, folds, features, ids, tmp_path):
    first = _fold(features, ids, constraints, folds, tmp_path / "a")
    second = _fold(features, ids, constraints, folds, tmp_path / "b", seed=SEED + 1)

    weight = "layers.0.weight"
    assert not torch.equal(first.model.state_dict()[weight], second.model.state_dict()[weight])


def test_running_again_resumes_to_the_same_result(constraints, folds, features, ids, tmp_path):
    first = _fold(features, ids, constraints, folds, tmp_path)
    again = _fold(features, ids, constraints, folds, tmp_path)

    pd.testing.assert_frame_equal(first.selection, again.selection)
    _assert_same_state(first.model.state_dict(), again.model.state_dict())


def test_linear_variant_has_one_map(constraints, folds, features, ids, tmp_path):
    result = _fold(features, ids, constraints, folds, tmp_path, settings=_settings(hidden=()))
    assert result.n_parameters == N_DIMS * 4 + 4


# --- errors ----------------------------------------------------------------------
def test_rejects_an_absent_test_fold(constraints, folds, features, ids, tmp_path):
    with pytest.raises(ValueError, match="test_fold must be one of"):
        train_fold(
            features,
            ids,
            constraints,
            folds,
            test_fold=N_FOLDS,
            settings=_settings(),
            seed=SEED,
            out_dir=tmp_path,
        )


def test_rejects_misaligned_folds(constraints, folds, features, ids, tmp_path):
    with pytest.raises(ValueError, match="aligned"):
        _fold(features, ids, constraints, folds.iloc[:-1], tmp_path)


def test_rejects_folds_that_split_a_component(constraints, folds, features, ids, tmp_path):
    split = folds.copy()
    # The two constraints of the first triplet, one inside the test fold and one out.
    split.iloc[0], split.iloc[1] = TEST_FOLD, (TEST_FOLD + 1) % N_FOLDS
    with pytest.raises(ValueError, match="whole components"):
        _fold(features, ids, constraints, split, tmp_path)


@pytest.mark.parametrize(
    ("margin", "lr", "weight_decay", "message"),
    [(0.0, 1e-3, 0.0, "margin"), (0.5, 0.0, 0.0, "lr"), (0.5, 1e-3, -1.0, "weight_decay")],
)
def test_grid_point_rejects_invalid_values(margin, lr, weight_decay, message):
    with pytest.raises(ValueError, match=message):
        GridPoint(margin, lr, weight_decay)


@pytest.mark.parametrize(
    ("grid", "message"), [((), "at least one point"), ((GRID[0], GRID[0]), "repeats")]
)
def test_settings_reject_an_invalid_grid(grid, message):
    with pytest.raises(ValueError, match=message):
        _settings(grid=grid)


def test_grid_point_label():
    assert GridPoint(0.2, 1e-3, 1e-4).label == "margin=0.2_lr=0.001_wd=0.0001"
