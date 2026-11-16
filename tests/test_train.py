"""Tests for the training loop: overfitting, early stopping, determinism and resuming.

The training data are 64 synthetic tracks: 4 genres x 16 tracks, 4 artists of
4 tracks per genre, with random feature vectors, so the only way to a low loss
is to memorise them. The task is the contrastive one (PxK batches, balanced
pairs, a small :class:`MlpHead`). Validation scores that must be known in
advance use a single-genre validation set, where every other item is relevant
and Recall@10 is ``10 / (n - 1)`` whatever the network does. Skipped as a whole
without the optional deep-learning extra.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402 (after the torch skip)

from musicsim.seeding import derive_seed  # noqa: E402
from musicsim.training.data import PKSampler, contrastive_pairs  # noqa: E402
from musicsim.training.losses import contrastive_loss  # noqa: E402
from musicsim.training.models import MlpHead  # noqa: E402
from musicsim.training.train import (  # noqa: E402
    BEST_CHECKPOINT,
    CURVE_FILE,
    LAST_CHECKPOINT,
    PATIENCE,
    TrainingSettings,
    ValidationSet,
    train,
)

SEED = 11

#: The synthetic training set: 4 genres x 16 tracks, 4 tracks per artist.
N_GENRES = 4
TRACKS_PER_GENRE = 16
TRACKS_PER_ARTIST = 4
N_TRACKS = N_GENRES * TRACKS_PER_GENRE
N_DIMS = 16

#: One PxK batch holds the whole training set, so an epoch is one step.
P, K = N_GENRES, TRACKS_PER_GENRE

MARGIN = 0.5
LR = 1e-2

#: Mean epoch loss the network must reach on its own training tracks.
OVERFIT_LOSS = 1e-2
HIDDEN = (64,)

#: Single-genre validation set: Recall@10 = 10 / (21 - 1) = 0.5 for every query.
N_FLAT_VALIDATION = 21
FLAT_RECALL = 0.5


class ContrastiveTask:
    """Contrastive training of an MLP head on cached vectors, as a :class:`TrainingTask`."""

    def __init__(self, features: np.ndarray, validation_features: np.ndarray) -> None:
        self.features = features
        self.validation_features = validation_features
        self.labels = np.repeat(np.arange(N_GENRES), TRACKS_PER_GENRE)
        self.artists = np.arange(N_TRACKS) // TRACKS_PER_ARTIST

    def batches(self, rng: np.random.Generator) -> PKSampler:
        return PKSampler(self.labels, self.artists, p=P, k=K, generator=rng)

    def loss(self, model: nn.Module, batch: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        first, second, is_positive = contrastive_pairs(
            self.labels[batch], self.artists[batch], generator=rng
        )
        embeddings = model(torch.from_numpy(self.features[batch]))
        return contrastive_loss(embeddings, first, second, is_positive, margin=MARGIN)

    def embed_validation(self, model: nn.Module) -> np.ndarray:
        return model(torch.from_numpy(self.validation_features)).numpy()


# --- fixtures and helpers ------------------------------------------------------
@pytest.fixture
def features(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(N_TRACKS, N_DIMS)).astype(np.float32)


@pytest.fixture
def task(features: np.ndarray) -> ContrastiveTask:
    """Validation on the training tracks themselves, labelled by genre."""
    return ContrastiveTask(features, features)


@pytest.fixture
def validation(task: ContrastiveTask) -> ValidationSet:
    return ValidationSet(np.arange(N_TRACKS), np.arange(N_TRACKS), task.labels)


@pytest.fixture
def flat_task(features: np.ndarray) -> ContrastiveTask:
    return ContrastiveTask(features, features[:N_FLAT_VALIDATION])


@pytest.fixture
def flat_validation() -> ValidationSet:
    n = N_FLAT_VALIDATION
    return ValidationSet(np.arange(n), np.arange(n), np.zeros(n, dtype=np.int64))


def _model(*, dropout: float = 0.0, init: str = "init") -> MlpHead:
    generator = torch.Generator().manual_seed(derive_seed(SEED, "test-train", init))
    return MlpHead(N_DIMS, HIDDEN, dropout=dropout, generator=generator)


def _settings(max_epochs: int, *, patience: int = PATIENCE) -> TrainingSettings:
    return TrainingSettings(lr=LR, max_epochs=max_epochs, patience=patience)


def _curve(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / CURVE_FILE)


def _assert_close_states(first: dict, second: dict) -> None:
    assert list(first) == list(second)
    for name, tensor in first.items():
        torch.testing.assert_close(tensor, second[name], rtol=0, atol=1e-6, msg=name)


def _load(path: Path) -> dict:
    return torch.load(path, weights_only=True)


# --- learning ----------------------------------------------------------------------
def test_overfits_64_synthetic_tracks(task, validation, tmp_path):
    result = train(_model(), task, validation, _settings(100, patience=100), tmp_path, seed=SEED)

    curve = _curve(tmp_path)
    assert list(curve.columns) == ["epoch", "train_loss", "val_recall_at_10"]
    assert curve["epoch"].tolist() == list(range(1, 101))
    # Starts well above the target, so reaching it is learning, not luck.
    assert curve["train_loss"].iloc[0] > 5 * OVERFIT_LOSS
    assert curve["train_loss"].iloc[-1] < OVERFIT_LOSS
    assert result.epochs == 100
    assert not result.stopped_early
    assert result.best_recall == curve["val_recall_at_10"].max()
    assert result.best_epoch == curve["val_recall_at_10"].idxmax() + 1


def test_model_ends_with_the_best_epochs_weights(task, validation, tmp_path):
    model = _model()
    train(model, task, validation, _settings(5), tmp_path, seed=SEED)

    _assert_close_states(model.state_dict(), _load(tmp_path / BEST_CHECKPOINT))


# --- early stopping -----------------------------------------------------------------
def test_early_stopping_after_patience_epochs_without_improvement(
    flat_task, flat_validation, tmp_path
):
    # The flat validation score never strictly improves after epoch 1.
    result = train(_model(), flat_task, flat_validation, _settings(50), tmp_path, seed=SEED)

    assert result.stopped_early
    assert result.best_epoch == 1
    assert result.epochs == 1 + PATIENCE
    assert result.best_recall == FLAT_RECALL
    curve = _curve(tmp_path)
    assert len(curve) == 1 + PATIENCE
    assert (curve["val_recall_at_10"] == FLAT_RECALL).all()


def test_best_checkpoint_is_kept_from_the_best_epoch(flat_task, flat_validation, tmp_path):
    one_epoch, stopped = tmp_path / "one", tmp_path / "stopped"
    train(_model(), flat_task, flat_validation, _settings(1), one_epoch, seed=SEED)
    train(_model(), flat_task, flat_validation, _settings(50), stopped, seed=SEED)

    best_after_one = _load(one_epoch / BEST_CHECKPOINT)
    _assert_close_states(_load(stopped / BEST_CHECKPOINT), best_after_one)
    last_weights = _load(stopped / LAST_CHECKPOINT)["model"]
    assert not torch.equal(last_weights["layers.0.weight"], best_after_one["layers.0.weight"])


def test_stops_at_max_epochs_before_patience_runs_out(flat_task, flat_validation, tmp_path):
    result = train(_model(), flat_task, flat_validation, _settings(3), tmp_path, seed=SEED)

    assert result.epochs == 3
    assert not result.stopped_early
    assert len(_curve(tmp_path)) == 3


# --- determinism and resuming ----------------------------------------------------------
def test_same_seed_gives_the_same_weights(task, validation, tmp_path):
    first, second = _model(dropout=0.2), _model(dropout=0.2)
    train(first, task, validation, _settings(4), tmp_path / "a", seed=SEED)
    train(second, task, validation, _settings(4), tmp_path / "b", seed=SEED)

    _assert_close_states(first.state_dict(), second.state_dict())
    _assert_close_states(
        _load(tmp_path / "a" / LAST_CHECKPOINT)["model"],
        _load(tmp_path / "b" / LAST_CHECKPOINT)["model"],
    )
    pd.testing.assert_frame_equal(_curve(tmp_path / "a"), _curve(tmp_path / "b"))


def test_another_seed_gives_other_weights(task, validation, tmp_path):
    train(_model(dropout=0.2), task, validation, _settings(4), tmp_path / "a", seed=SEED)
    train(_model(dropout=0.2), task, validation, _settings(4), tmp_path / "b", seed=SEED + 1)

    first = _load(tmp_path / "a" / LAST_CHECKPOINT)["model"]
    second = _load(tmp_path / "b" / LAST_CHECKPOINT)["model"]
    assert not torch.allclose(first["layers.0.weight"], second["layers.0.weight"])


def test_resuming_reproduces_an_uninterrupted_run(task, validation, tmp_path):
    whole, split = tmp_path / "whole", tmp_path / "split"
    whole_result = train(_model(dropout=0.2), task, validation, _settings(6), whole, seed=SEED)
    train(_model(dropout=0.2), task, validation, _settings(3), split, seed=SEED)
    # A differently initialised model: the resumed weights must come from the checkpoint.
    resumed = _model(dropout=0.2, init="other")
    split_result = train(resumed, task, validation, _settings(6), split, seed=SEED, resume=True)

    assert split_result == whole_result
    _assert_close_states(
        _load(split / LAST_CHECKPOINT)["model"], _load(whole / LAST_CHECKPOINT)["model"]
    )
    _assert_close_states(_load(split / BEST_CHECKPOINT), _load(whole / BEST_CHECKPOINT))
    pd.testing.assert_frame_equal(_curve(split), _curve(whole))


def test_leaves_the_global_torch_state_as_it_was(task, validation, tmp_path):
    rng_state = torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()

    train(_model(dropout=0.2), task, validation, _settings(2), tmp_path, seed=SEED)

    assert torch.equal(torch.get_rng_state(), rng_state)
    assert torch.are_deterministic_algorithms_enabled() == deterministic


# --- errors ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": 0.0, "max_epochs": 5},
        {"lr": -1e-3, "max_epochs": 5},
        {"lr": LR, "max_epochs": 0},
        {"lr": LR, "max_epochs": 2.5},
        {"lr": LR, "max_epochs": True},
        {"lr": LR, "max_epochs": 5, "patience": 0},
        {"lr": LR, "max_epochs": 5, "patience": True},
        {"lr": LR, "max_epochs": 5, "weight_decay": -1e-4},
    ],
)
def test_settings_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrainingSettings(**kwargs)


@pytest.mark.parametrize(
    ("ids", "group", "genre", "message"),
    [
        (np.arange(20), np.arange(19), np.zeros(20, dtype=int), "aligned"),
        (np.arange(10), np.arange(10), np.zeros(10, dtype=int), "more than 10"),
        (np.arange(20), np.arange(20.0), np.zeros(20, dtype=int), "group"),
        (np.arange(20), np.arange(20), np.full(20, -1), "genre"),
    ],
)
def test_validation_set_rejects_invalid_arrays(ids, group, genre, message):
    with pytest.raises(ValueError, match=message):
        ValidationSet(ids, group, genre)


def test_rejects_a_model_outside_the_cpu(task, validation, tmp_path):
    with pytest.raises(ValueError, match="CPU only"):
        train(_model().to("meta"), task, validation, _settings(1), tmp_path, seed=SEED)


def test_fresh_run_refuses_a_directory_holding_a_run(task, validation, tmp_path):
    train(_model(), task, validation, _settings(1), tmp_path, seed=SEED)
    with pytest.raises(ValueError, match="resume=True"):
        train(_model(), task, validation, _settings(1), tmp_path, seed=SEED)


def test_resume_needs_a_saved_run(task, validation, tmp_path):
    with pytest.raises(FileNotFoundError, match=LAST_CHECKPOINT):
        train(_model(), task, validation, _settings(1), tmp_path, seed=SEED, resume=True)


@pytest.mark.parametrize(
    ("seed", "settings"),
    [
        (SEED + 1, TrainingSettings(lr=LR, max_epochs=2)),
        (SEED, TrainingSettings(lr=LR / 2, max_epochs=2)),
    ],
)
def test_resume_refuses_other_seed_or_settings(task, validation, tmp_path, seed, settings):
    train(_model(), task, validation, _settings(1), tmp_path, seed=SEED)
    with pytest.raises(ValueError, match="cannot resume"):
        train(_model(), task, validation, settings, tmp_path, seed=seed, resume=True)


class _NoBatches(ContrastiveTask):
    def batches(self, rng):
        return []


class _NanLoss(ContrastiveTask):
    def loss(self, model, batch, rng):
        return super().loss(model, batch, rng) * torch.tensor(float("nan"))


class _VectorLoss(ContrastiveTask):
    def loss(self, model, batch, rng):
        return super().loss(model, batch, rng).repeat(2)


@pytest.mark.parametrize(
    ("task_class", "message"),
    [(_NoBatches, "no batch"), (_NanLoss, "finite scalar"), (_VectorLoss, "finite scalar")],
)
def test_rejects_a_task_without_batches_or_with_a_bad_loss(
    task_class, message, features, validation, tmp_path
):
    with pytest.raises(ValueError, match=message):
        train(
            _model(), task_class(features, features), validation, _settings(1), tmp_path, seed=SEED
        )


class _ShortEmbedding(ContrastiveTask):
    def embed_validation(self, model):
        return super().embed_validation(model)[:-1]


class _NanEmbedding(ContrastiveTask):
    def embed_validation(self, model):
        embeddings = super().embed_validation(model)
        embeddings[0, 0] = np.nan
        return embeddings


@pytest.mark.parametrize(
    ("task_class", "message"),
    [(_ShortEmbedding, r"\(64, dims\)"), (_NanEmbedding, "NaN")],
)
def test_rejects_a_bad_validation_embedding(task_class, message, features, validation, tmp_path):
    with pytest.raises(ValueError, match=message):
        train(
            _model(), task_class(features, features), validation, _settings(1), tmp_path, seed=SEED
        )
