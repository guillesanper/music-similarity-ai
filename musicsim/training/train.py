"""The training loop of the learned similarity models, with early stopping.

**What the loop owns, and what it does not.** :func:`train` knows nothing about
which network, which loss or which input it trains: a :class:`TrainingTask`
supplies the batches of an epoch and the loss of a batch, and a
:class:`Validator` scores the network after every epoch. The loop owns
everything that has to be the same for every model: the optimiser, when
validation runs, early stopping, the checkpoints, the learning curve and the
random state. That way the MLP heads and the CNNs, trained on genres or on
human judgements, are trained and selected by exactly the same rules.

**Optimiser.** Adam with the learning rate of the caller and an L2 weight
decay of :data:`WEIGHT_DECAY` (``1e-4``) added to the gradient, as
:class:`torch.optim.Adam` does; the value is this project's choice, the usual
light regularisation for networks of this size.

**Validation and early stopping.** After every epoch the validator scores the
network, in evaluation mode and without gradients, on the columns it declares
(:attr:`Validator.columns`); the first column is the criterion, higher being
better, and the others are only recorded in the curve (the agreement on the
training constraints next to the validation one, for instance, which shows the
gap between the two). Model selection never sees the test split. Training stops
when :data:`PATIENCE` (10) epochs in a row have not *strictly* improved the best
criterion, or after ``max_epochs``; the weights of the best epoch are what the
run keeps. Both values are this project's choice.

:class:`ValidationSet` is the validator of the models trained on genres: the
network embeds the items of FMA's validation split, the embeddings are scaled to
unit norm, and the epoch is scored by the mean Recall@10 of
:func:`musicsim.evaluation.retrieval.compute_per_query_metrics`, the headline
metric of the retrieval evaluation, every validation item acting as a query
against the others. The exclusion groups (near-duplicates of each query) come
from the caller and stay fixed for the whole run: they are a property of the
audio, not of the embedding being learned, and fixing them keeps the scores of
different epochs comparable. The validator of the models trained on human
judgements is :class:`musicsim.training.constraints.AgreementValidator`.

**Files.** In the output directory (``paths.checkpoints_dir(<model>)`` for a
real run):

* :data:`BEST_CHECKPOINT`, the ``state_dict`` of the best epoch so far, the
  file an extractor loads;
* :data:`LAST_CHECKPOINT`, the full state at the end of the last epoch (weights,
  optimiser, counters, curve and both random states), from which a run is
  resumed;
* :data:`CURVE_FILE`, one row per epoch: ``epoch``, ``train_loss`` (the mean
  loss over the epoch's batches) and the validator's columns
  (``val_recall_at_10`` for :class:`ValidationSet`).

Each is written to a temporary file and renamed, so an interrupted run never
leaves a truncated checkpoint behind.

**Randomness and determinism.** The loop draws from exactly two streams:

* one NumPy generator, ``seeding.generator(seed, "training")``, handed to the
  task for everything it samples (PxK batches, contrastive pairs, crops);
* PyTorch's global CPU generator, which dropout reads and which accepts no
  generator of its own. The whole loop runs inside
  :func:`torch.random.fork_rng`, seeded with
  ``seeding.derive_seed(seed, "training", "torch")``, so the process's own
  global state is left as it was.

Both states are saved in :data:`LAST_CHECKPOINT` after every epoch, which is
what makes a resumed run reproduce an uninterrupted one bit for bit.
:func:`torch.use_deterministic_algorithms` is switched on for the duration of
the loop and restored afterwards. Training runs on the CPU only for now;
moving it to a GPU also means seeding and saving the CUDA generator.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd
import torch
from torch import nn

from musicsim.evaluation.retrieval import compute_per_query_metrics
from musicsim.seeding import derive_seed, generator

__all__ = [
    "BEST_CHECKPOINT",
    "CURVE_FILE",
    "LAST_CHECKPOINT",
    "PATIENCE",
    "VALIDATION_K",
    "WEIGHT_DECAY",
    "EmbeddingTask",
    "TrainingResult",
    "TrainingSettings",
    "TrainingTask",
    "ValidationSet",
    "Validator",
    "train",
]

logger = logging.getLogger(__name__)

#: L2 weight decay of Adam (this project's choice).
WEIGHT_DECAY = 1e-4

#: Epochs in a row without a strict improvement of the validation criterion
#: after which training stops (this project's choice).
PATIENCE = 10

#: Cutoff of the validation metric: Recall@10, the headline retrieval metric.
VALIDATION_K = 10

#: ``state_dict`` of the best epoch so far.
BEST_CHECKPOINT = "best.pt"

#: Full training state at the end of the last epoch, for resuming.
LAST_CHECKPOINT = "last.pt"

#: Learning curve, one row per epoch.
CURVE_FILE = "curve.csv"

#: Columns of :data:`CURVE_FILE` the loop itself fills; the validator's follow.
_LOOP_COLUMNS = ("epoch", "train_loss")

#: Settings that must match between a run and its resumption (``max_epochs``
#: may grow, which is how a finished run is continued). ``columns`` are the
#: validator's, so a run is never continued under another criterion.
_RESUME_KEYS = ("seed", "lr", "weight_decay", "patience", "columns")

#: Axes of an embedding matrix, ``(n_items, dims)``.
_EMBEDDING_NDIM = 2

#: Suffix of the temporary file a checkpoint is written to before renaming.
_PARTIAL_SUFFIX = ".partial"


# --- inputs ------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingSettings:
    """The optimisation settings of one run (from the model's configuration)."""

    lr: float
    max_epochs: int
    patience: int = PATIENCE
    weight_decay: float = WEIGHT_DECAY

    def __post_init__(self) -> None:
        if not self.lr > 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if isinstance(self.max_epochs, bool) or not isinstance(self.max_epochs, int):
            raise ValueError(f"max_epochs must be an integer, got {self.max_epochs!r}")
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be at least 1, got {self.max_epochs}")
        if isinstance(self.patience, bool) or not isinstance(self.patience, int):
            raise ValueError(f"patience must be an integer, got {self.patience!r}")
        if self.patience < 1:
            raise ValueError(f"patience must be at least 1, got {self.patience}")
        if not self.weight_decay >= 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")


class TrainingTask(Protocol):
    """What a model's training supplies to :func:`train`.

    ``rng`` is the loop's NumPy generator: every random draw of the task must
    come from it, or a resumed run will not reproduce an uninterrupted one.
    """

    def batches(self, rng: np.random.Generator) -> Iterable[np.ndarray]:
        """The batches of one epoch, as positions into the training items."""
        ...

    def loss(self, model: nn.Module, batch: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        """The scalar loss of ``model`` on one batch."""
        ...


class EmbeddingTask(TrainingTask, Protocol):
    """A :class:`TrainingTask` that also embeds the items of a :class:`ValidationSet`."""

    def embed_validation(self, model: nn.Module) -> np.ndarray:
        """``(n_validation, dims)`` embeddings of the :class:`ValidationSet` items, in order.

        Called in evaluation mode and without gradients; need not be normalised.
        """
        ...


class Validator(Protocol):
    """What scores the network after every epoch of :func:`train`.

    ``columns`` names the scores, which become columns of :data:`CURVE_FILE`;
    the first is the early-stopping criterion, higher being better.
    """

    columns: ClassVar[tuple[str, ...]]

    def score(self, model: nn.Module, task: Any) -> dict[str, float]:
        """One finite score per name of :attr:`columns`.

        Called in evaluation mode and without gradients, with the task being
        trained.
        """
        ...


@dataclass(frozen=True)
class ValidationSet:
    """The items each epoch is scored on, row-aligned with the task's validation embedding.

    ``group`` is the exclusion group of every item (its near-duplicate group,
    as in the ``dedup`` retrieval variant) and ``genre`` its relevance label
    code; both are non-negative integers, as
    :func:`~musicsim.evaluation.retrieval.compute_per_query_metrics` expects.
    It is the :class:`Validator` of an :class:`EmbeddingTask`, scored by
    Recall@10.
    """

    columns: ClassVar[tuple[str, ...]] = ("val_recall_at_10",)

    ids: np.ndarray
    group: np.ndarray
    genre: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.ids)
        group = np.asarray(self.group)
        genre = np.asarray(self.genre)
        if ids.ndim != 1 or group.shape != ids.shape or genre.shape != ids.shape:
            raise ValueError(
                "ids, group and genre must be 1-D and aligned per item, got shapes "
                f"{ids.shape}, {group.shape} and {genre.shape}"
            )
        if len(ids) <= VALIDATION_K:
            raise ValueError(
                f"Recall@{VALIDATION_K} needs more than {VALIDATION_K} validation items, "
                f"got {len(ids)}"
            )
        for name, codes in (("group", group), ("genre", genre)):
            if not np.issubdtype(codes.dtype, np.integer) or codes.min() < 0:
                raise ValueError(
                    f"{name} must hold non-negative integer codes, got dtype {codes.dtype}"
                )
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "genre", genre)

    def score(self, model: nn.Module, task: EmbeddingTask) -> dict[str, float]:
        """Mean Recall@10 of the validation items, each a query against the others."""
        embeddings = np.asarray(task.embed_validation(model), dtype=np.float32)
        n_items = len(self.ids)
        if embeddings.ndim != _EMBEDDING_NDIM or len(embeddings) != n_items:
            raise ValueError(
                f"embed_validation must return ({n_items}, dims) embeddings, "
                f"got shape {embeddings.shape}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("embed_validation returned NaN or infinite values")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        unit = embeddings / np.where(norms == 0, 1.0, norms)
        per_query = compute_per_query_metrics(
            unit, self.ids, np.arange(n_items), self.group, self.genre, ks=[VALIDATION_K]
        )
        recall = float(per_query.loc[per_query["metric"] == "R", "value"].mean())
        return {self.columns[0]: recall}


@dataclass(frozen=True)
class TrainingResult:
    """How a run ended. The model passed to :func:`train` holds the best epoch's weights.

    ``best_score`` is the validator's criterion (its first column) at ``best_epoch``.
    """

    best_epoch: int
    best_score: float
    epochs: int
    stopped_early: bool


@dataclass
class _State:
    """Everything that changes from one epoch to the next, besides the weights."""

    epoch: int = 0
    best_epoch: int = 0
    best_score: float = -math.inf
    since_best: int = 0
    curve: list[dict[str, float]] = field(default_factory=list)


# --- the loop ----------------------------------------------------------------
def train(
    model: nn.Module,
    task: TrainingTask,
    validation: Validator,
    settings: TrainingSettings,
    out_dir: Path,
    *,
    seed: int,
    resume: bool = False,
) -> TrainingResult:
    """Train ``model`` on ``task`` until early stopping or ``settings.max_epochs``.

    Writes :data:`BEST_CHECKPOINT`, :data:`LAST_CHECKPOINT` and
    :data:`CURVE_FILE` into ``out_dir`` after every epoch and leaves the best
    epoch's weights in ``model``. A fresh run refuses an ``out_dir`` that
    already holds one; ``resume=True`` continues the run saved there instead,
    which must have been started with the same seed, learning rate, weight
    decay, patience and validator columns.
    """
    _check_on_cpu(model)
    columns = _check_columns(validation.columns)
    out_dir = Path(out_dir)
    run_key: dict[str, Any] = {
        "seed": int(seed),
        "lr": settings.lr,
        "weight_decay": settings.weight_decay,
        "patience": settings.patience,
        "columns": list(columns),
    }
    if not resume:
        existing = [
            name for name in (BEST_CHECKPOINT, LAST_CHECKPOINT) if (out_dir / name).exists()
        ]
        if existing:
            raise ValueError(
                f"{out_dir} already holds a training run ({', '.join(existing)}); "
                "pass resume=True to continue it, or use an empty directory"
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay
    )
    rng = generator(seed, "training")
    state = _State()

    with _deterministic_algorithms(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(seed, "training", "torch"))
        if resume:
            state = _load_last(out_dir / LAST_CHECKPOINT, run_key, model, optimizer, rng)

        while state.epoch < settings.max_epochs and state.since_best < settings.patience:
            train_loss = _train_epoch(model, task, optimizer, rng)
            scores = _validation_scores(model, task, validation, columns)
            criterion = scores[columns[0]]
            state.epoch += 1
            state.curve.append({"epoch": state.epoch, "train_loss": train_loss, **scores})
            if criterion > state.best_score:
                state.best_epoch, state.best_score, state.since_best = state.epoch, criterion, 0
                _save_best(out_dir / BEST_CHECKPOINT, model)
            else:
                state.since_best += 1
            logger.info(
                "epoch %d: train loss %.4f, %s %.4f (best %.4f at epoch %d)",
                state.epoch,
                train_loss,
                columns[0],
                criterion,
                state.best_score,
                state.best_epoch,
            )
            _write_curve(out_dir / CURVE_FILE, state.curve, columns)
            _save_last(out_dir / LAST_CHECKPOINT, run_key, model, optimizer, rng, state)

    model.load_state_dict(torch.load(out_dir / BEST_CHECKPOINT, weights_only=True))
    stopped_early = state.since_best >= settings.patience
    if stopped_early:
        logger.info(
            "early stopping after epoch %d: no improvement for %d epochs",
            state.epoch,
            settings.patience,
        )
    return TrainingResult(
        best_epoch=state.best_epoch,
        best_score=state.best_score,
        epochs=state.epoch,
        stopped_early=stopped_early,
    )


def _train_epoch(
    model: nn.Module,
    task: TrainingTask,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
) -> float:
    """One pass over the task's batches; returns the mean batch loss."""
    model.train()
    losses: list[float] = []
    for batch in task.batches(rng):
        optimizer.zero_grad()
        loss = task.loss(model, batch, rng)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError(
                f"the task's loss must be a finite scalar, got shape {tuple(loss.shape)} "
                f"and value(s) {loss.detach().flatten()[:3].tolist()}"
            )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if not losses:
        raise ValueError("the task yielded no batch for this epoch")
    return float(np.mean(losses))


def _validation_scores(
    model: nn.Module, task: TrainingTask, validation: Validator, columns: tuple[str, ...]
) -> dict[str, float]:
    """The validator's scores, in evaluation mode and without gradients, checked."""
    model.eval()
    with torch.no_grad():
        scores = validation.score(model, task)
    model.train()

    if set(scores) != set(columns):
        raise ValueError(
            f"the validator must score exactly its columns {list(columns)}, got {sorted(scores)}"
        )
    values = {name: float(scores[name]) for name in columns}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"the validator returned a non-finite score: {values}")
    return values


def _check_columns(columns: tuple[str, ...]) -> tuple[str, ...]:
    """The validator's columns: at least one, all distinct and apart from the loop's own."""
    columns = tuple(columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError(f"a validator needs at least one column, all distinct; got {columns}")
    clashing = sorted(set(columns) & set(_LOOP_COLUMNS))
    if clashing:
        raise ValueError(f"validator columns {clashing} clash with the loop's {_LOOP_COLUMNS}")
    return columns


# --- checkpoints -------------------------------------------------------------
def _save_best(path: Path, model: nn.Module) -> None:
    state_dict = model.state_dict()
    _atomic_write(path, lambda partial: torch.save(state_dict, partial))


def _write_curve(path: Path, rows: list[dict[str, float]], columns: tuple[str, ...]) -> None:
    curve = pd.DataFrame(rows, columns=[*_LOOP_COLUMNS, *columns])
    _atomic_write(path, lambda partial: curve.to_csv(partial, index=False))


def _save_last(
    path: Path,
    run_key: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    state: _State,
) -> None:
    payload = {
        "run": run_key,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "numpy_rng": rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "epoch": state.epoch,
        "best_epoch": state.best_epoch,
        "best_score": state.best_score,
        "since_best": state.since_best,
        "curve": state.curve,
    }
    _atomic_write(path, lambda p: torch.save(payload, p))


def _load_last(
    path: Path,
    run_key: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
) -> _State:
    """Restore weights, optimiser and both random states; return the counters."""
    if not path.exists():
        raise FileNotFoundError(f"cannot resume: {path} does not exist")
    payload = torch.load(path, weights_only=True)
    saved = {key: payload["run"][key] for key in _RESUME_KEYS}
    if saved != run_key:
        raise ValueError(f"cannot resume a run started with {saved} using {run_key}")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    rng.bit_generator.state = payload["numpy_rng"]
    torch.set_rng_state(payload["torch_rng"])
    return _State(
        epoch=payload["epoch"],
        best_epoch=payload["best_epoch"],
        best_score=payload["best_score"],
        since_best=payload["since_best"],
        curve=payload["curve"],
    )


def _atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    """Write through a temporary file and rename it, so ``path`` is never half-written."""
    partial = path.with_name(path.name + _PARTIAL_SUFFIX)
    write(partial)
    partial.replace(path)


# --- environment -------------------------------------------------------------
@contextmanager
def _deterministic_algorithms() -> Iterator[None]:
    """Deterministic PyTorch kernels inside the block; the previous setting after it."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _check_on_cpu(model: nn.Module) -> None:
    devices = {p.device.type for p in model.parameters()} | {b.device.type for b in model.buffers()}
    if devices != {"cpu"}:
        raise ValueError(
            f"training runs on the CPU only for now; the model's tensors are on {sorted(devices)}"
        )
