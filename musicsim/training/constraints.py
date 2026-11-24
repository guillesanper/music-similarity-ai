"""Training on human similarity constraints, with nested model selection per fold.

The models trained on MagnaTagATune learn from its 860 similarity constraints
(:mod:`musicsim.datasets.mtt_triplets`): ``(anchor, pos, neg)`` says that people
heard ``anchor`` as closer to ``pos`` than to ``neg``. With so few constraints a
model is only ever tested on constraints it did not see, through the grouped
folds of :mod:`musicsim.datasets.mtt_folds`; this module trains the model of one
fold. Its input is a matrix of fixed clip vectors (a cached MERT layer), already
standardised by the caller.

**Loss.** The triplet loss on the cosine distance ``d = 1 - cos(u, v)``, the one
of :mod:`musicsim.training.losses` (FaceNet's, ``Schroff2015``, with the margin
halved), applied to the constraints themselves instead of mined triplets:
``max(0, d(anchor, pos) - d(anchor, neg) + m)``, averaged over the constraints
of a batch. Every constraint weighs the same, whatever its margin of votes: the
headline agreement (:mod:`musicsim.evaluation.triplets`) is unweighted, and so is
the selection criterion below (our choice). A batch holds
``batch_size`` constraints; an epoch is one pass over the training constraints in
a shuffled order, the last batch being smaller when they do not divide evenly.
The clips of a batch are embedded once each, however many of its constraints
share them.

**Validation.** :class:`AgreementValidator` scores every epoch by the unweighted
agreement (the share of constraints a representation satisfies, a tie counting
0.5) on the validation constraints, which is the early-stopping criterion, and
records the agreement on the training constraints next to it. The two together
are the validation curve of the run, and their gap shows how much the model
memorises.

**Nested selection** (decided before any result). Within the training
constraints of a fold, an *inner* split into ``inner_folds`` folds, again by
connected components of clips and balanced by
:func:`musicsim.datasets.mtt_folds.constraint_folds`. Inner fold
:data:`INNER_VALIDATION_FOLD` is held out for early stopping and for choosing
the grid point; the others are trained on. Every point of a small grid of
margin, learning rate and weight decay is trained that way. The point with the
best agreement on the inner fold wins, ties going to the first in grid order,
and *its* model is the fold's model: it is not retrained on the whole training
side, so the epoch that was validated is the one that is kept. The test fold of
the outer split takes no part in any of this. Its clips are not even handed to
the network: only the clips of the training constraints are. All of these
choices are our own.

**Randomness.** For the outer fold ``f`` and the run seed ``s``, the inner split
draws from ``seeding.generator(s, "folds", "inner", f)``, the initial weights
from ``seeding.derive_seed(s, "constraints", f, "init")``, and the training
loop runs with the seed ``seeding.derive_seed(s, "constraints", f)``. Every grid
point of a fold therefore starts from the same weights and sees the same
batches, and differs from the others only in its hyperparameters.

**Files**, in the fold's output directory:

* ``inner/<grid point>/``, the files of :func:`musicsim.training.train.train`
  for every grid point, its curve holding ``val_agreement`` and
  ``train_agreement`` per epoch;
* :data:`SELECTION_FILE`, one row per grid point: its settings, best epoch and
  agreements at that epoch, and whether it was chosen;
* :data:`~musicsim.training.train.BEST_CHECKPOINT` and
  :data:`~musicsim.training.train.CURVE_FILE`, copies of the chosen point's;
* :data:`FOLD_SUMMARY`, written last: the settings, the chosen point, the number
  of parameters and the sizes of every part of the split.

Running :func:`train_fold` again on the same directory resumes every grid point
from its last checkpoint and ends in the same result.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from musicsim.datasets.mtt_folds import constraint_folds
from musicsim.evaluation.triplets import constraint_hits
from musicsim.seeding import derive_seed, generator
from musicsim.training.data import VectorDataset
from musicsim.training.models import MlpHead, count_parameters
from musicsim.training.train import (
    BEST_CHECKPOINT,
    CURVE_FILE,
    LAST_CHECKPOINT,
    TrainingSettings,
    train,
)

__all__ = [
    "FOLD_SUMMARY",
    "INNER_VALIDATION_FOLD",
    "SELECTION_FILE",
    "TRAIN_AGREEMENT",
    "VAL_AGREEMENT",
    "AgreementValidator",
    "ConstraintTask",
    "ConstraintTrainingSettings",
    "FoldResult",
    "GridPoint",
    "constraint_triplet_loss",
    "train_fold",
]

logger = logging.getLogger(__name__)

#: Curve column of the agreement on the validation constraints (the criterion).
VAL_AGREEMENT = "val_agreement"

#: Curve column of the agreement on the training constraints.
TRAIN_AGREEMENT = "train_agreement"

#: Inner fold held out for early stopping and selection; any one would do.
INNER_VALIDATION_FOLD = 0

#: One row per grid point of a fold, in grid order.
SELECTION_FILE = "selection.csv"

#: Summary of a finished fold; written last, so its presence marks it finished.
FOLD_SUMMARY = "fold.json"

#: Subdirectory of the fold's output holding one training run per grid point.
_INNER_DIR = "inner"

#: The three clip columns of a constraint, in order.
_CLIP_COLUMNS = ("anchor", "pos", "neg")

#: Largest cosine distance, between opposite vectors; no margin can exceed it.
_MAX_COSINE_DISTANCE = 2.0

#: Number of axes of an embedding batch, ``(batch, dims)``.
_EMBEDDING_NDIM = 2

#: Suffix of the temporary file the summary is written to before renaming.
_PARTIAL_SUFFIX = ".partial"


# --- loss --------------------------------------------------------------------
def constraint_triplet_loss(
    embeddings: torch.Tensor,
    anchor: np.ndarray | torch.Tensor,
    pos: np.ndarray | torch.Tensor,
    neg: np.ndarray | torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Mean of ``max(0, d(anchor, pos) - d(anchor, neg) + margin)`` over the constraints.

    ``embeddings`` is ``(n_clips, dims)``; ``anchor``, ``pos`` and ``neg`` are
    aligned per constraint and index into its rows. ``d`` is the cosine
    distance, so ``margin`` lies in ``(0, 2]``.
    """
    if embeddings.ndim != _EMBEDDING_NDIM or len(embeddings) == 0:
        raise ValueError(
            f"embeddings must be a non-empty (n_clips, dims) tensor, got shape "
            f"{tuple(embeddings.shape)}"
        )
    if not embeddings.is_floating_point():
        raise ValueError(f"embeddings must be floating point, got dtype {embeddings.dtype}")
    _check_margin(margin)
    device = embeddings.device
    a, p, n = (torch.as_tensor(column, device=device) for column in (anchor, pos, neg))
    _check_triples(a, p, n, len(embeddings))

    unit = F.normalize(embeddings, dim=1)
    d_pos = 1.0 - (unit[a] * unit[p]).sum(dim=1)
    d_neg = 1.0 - (unit[a] * unit[n]).sum(dim=1)
    return torch.clamp(d_pos - d_neg + margin, min=0.0).mean()


# --- the training task and its validator ---------------------------------------
class ConstraintTask:
    """Triplet training of a head on fixed clip vectors, one batch of constraints at a time.

    It is the :class:`~musicsim.training.train.TrainingTask` of the models trained
    on human judgements. ``features`` is ``(n_clips, dims)``, row-aligned with
    ``ids``; ``constraints`` has the columns ``anchor``, ``pos`` and ``neg``
    (clip ids, all in ``ids``).
    """

    def __init__(
        self,
        features: np.ndarray,
        ids: np.ndarray,
        constraints: pd.DataFrame,
        *,
        margin: float,
        batch_size: int,
    ) -> None:
        self._features = VectorDataset(features)
        self._triples = _clip_positions(ids, constraints, len(self._features))
        if len(self._triples) == 0:
            raise ValueError("there are no training constraints")
        _check_margin(margin)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        self.margin = float(margin)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self._triples)

    def batches(self, rng: np.random.Generator) -> list[np.ndarray]:
        """One epoch: every constraint once, shuffled, in chunks of ``batch_size``."""
        order = rng.permutation(len(self._triples))
        return [
            order[start : start + self.batch_size]
            for start in range(0, len(order), self.batch_size)
        ]

    def loss(self, model: nn.Module, batch: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        """Triplet loss of the batch's constraints; each clip is embedded once."""
        triples = self._triples[batch]
        clips, inverse = np.unique(triples.ravel(), return_inverse=True)
        inverse = inverse.reshape(triples.shape)
        embeddings = model(torch.from_numpy(self._features[clips]))
        return constraint_triplet_loss(
            embeddings, inverse[:, 0], inverse[:, 1], inverse[:, 2], margin=self.margin
        )


class AgreementValidator:
    """The :class:`~musicsim.training.train.Validator` of the models trained on constraints.

    Scores the unweighted agreement on ``validation`` (the criterion) and on
    ``training``, both constraint tables over clips of ``ids``, by embedding the
    clips of ``features`` (row-aligned with ``ids``).
    """

    columns: ClassVar[tuple[str, ...]] = (VAL_AGREEMENT, TRAIN_AGREEMENT)

    def __init__(
        self,
        features: np.ndarray,
        ids: np.ndarray,
        training: pd.DataFrame,
        validation: pd.DataFrame,
    ) -> None:
        self._features = VectorDataset(features)
        self._ids = np.asarray(ids)
        for name, table in (("training", training), ("validation", validation)):
            if len(_clip_positions(self._ids, table, len(self._features))) == 0:
                raise ValueError(f"there are no {name} constraints to score")
        self._training = training
        self._validation = validation

    def score(self, model: nn.Module, task: Any) -> dict[str, float]:
        """Agreement on the validation and on the training constraints."""
        embeddings = _project(model, self._features[np.arange(len(self._features))])
        return {
            VAL_AGREEMENT: float(constraint_hits(embeddings, self._ids, self._validation).mean()),
            TRAIN_AGREEMENT: float(constraint_hits(embeddings, self._ids, self._training).mean()),
        }


# --- settings ------------------------------------------------------------------
@dataclass(frozen=True)
class GridPoint:
    """One point of the selection grid."""

    margin: float
    lr: float
    weight_decay: float

    def __post_init__(self) -> None:
        _check_margin(self.margin)
        if not self.lr > 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not self.weight_decay >= 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")

    @property
    def label(self) -> str:
        """Name of the point's run directory, e.g. ``margin=0.2_lr=0.001_wd=0.0001``."""
        return f"margin={self.margin:g}_lr={self.lr:g}_wd={self.weight_decay:g}"


@dataclass(frozen=True)
class ConstraintTrainingSettings:
    """Architecture, grid and loop settings of the model of every fold.

    ``hidden`` empty is the linear variant (one linear map, then the unit
    normalisation); ``dropout`` only acts after hidden layers.
    """

    hidden: tuple[int, ...]
    dim: int
    dropout: float
    grid: tuple[GridPoint, ...]
    batch_size: int
    max_epochs: int
    patience: int
    inner_folds: int
    max_imbalance: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden", tuple(self.hidden))
        object.__setattr__(self, "grid", tuple(self.grid))
        if not self.grid:
            raise ValueError("the selection grid needs at least one point")
        labels = [point.label for point in self.grid]
        if len(set(labels)) != len(labels):
            raise ValueError(f"the selection grid repeats a point: {labels}")


@dataclass(frozen=True)
class FoldResult:
    """The model of one fold and how it was chosen.

    ``training_rows`` and ``inner_rows`` are the index labels of the constraints
    the model was trained on and selected on; ``model`` holds the chosen point's
    best weights.
    """

    test_fold: int
    chosen: GridPoint
    selection: pd.DataFrame
    model: nn.Module
    n_parameters: int
    training_rows: np.ndarray
    inner_rows: np.ndarray


# --- one fold ------------------------------------------------------------------
def train_fold(
    features: np.ndarray,
    ids: np.ndarray,
    constraints: pd.DataFrame,
    folds: pd.Series,
    *,
    test_fold: int,
    settings: ConstraintTrainingSettings,
    seed: int,
    out_dir: Path,
) -> FoldResult:
    """Train every grid point on the training side of ``test_fold`` and keep the best.

    ``features`` is ``(n_clips, dims)``, row-aligned with ``ids``;
    ``constraints`` has the columns ``anchor``, ``pos``, ``neg`` and
    ``triplet_id``, and ``folds`` (from
    :func:`~musicsim.datasets.mtt_folds.constraint_folds`) is aligned with its
    index. See the module docstring for the selection, the seeds and the files.
    """
    outer_training, test = _outer_split(constraints, folds, test_fold)
    inner = constraint_folds(
        outer_training,
        n_folds=settings.inner_folds,
        max_imbalance=settings.max_imbalance,
        generator=generator(seed, "folds", "inner", test_fold),
    )
    is_inner = (inner == INNER_VALIDATION_FOLD).to_numpy()
    inner_training, inner_validation = outer_training[~is_inner], outer_training[is_inner]

    # Only the clips of the training side reach the network.
    clip_ids = np.unique(outer_training[list(_CLIP_COLUMNS)].to_numpy().ravel())
    clip_features = np.asarray(features)[_positions_of(np.asarray(ids), clip_ids)]
    task_seed = derive_seed(seed, "constraints", test_fold)
    validator = AgreementValidator(clip_features, clip_ids, inner_training, inner_validation)
    logger.info(
        "fold %d: %d training, %d inner validation and %d test constraints",
        test_fold,
        len(inner_training),
        len(inner_validation),
        len(test),
    )

    out_dir = Path(out_dir)
    rows: list[dict[str, Any]] = []
    models: list[nn.Module] = []
    for point in settings.grid:
        run_dir = out_dir / _INNER_DIR / point.label
        model = _build_model(clip_features.shape[1], settings, seed, test_fold)
        task = ConstraintTask(
            clip_features,
            clip_ids,
            inner_training,
            margin=point.margin,
            batch_size=settings.batch_size,
        )
        result = train(
            model,
            task,
            validator,
            TrainingSettings(
                lr=point.lr,
                max_epochs=settings.max_epochs,
                patience=settings.patience,
                weight_decay=point.weight_decay,
            ),
            run_dir,
            seed=task_seed,
            resume=(run_dir / LAST_CHECKPOINT).is_file(),
        )
        curve = pd.read_csv(run_dir / CURVE_FILE)
        rows.append(
            {
                "point": point.label,
                **asdict(point),
                "best_epoch": result.best_epoch,
                "epochs": result.epochs,
                "stopped_early": result.stopped_early,
                VAL_AGREEMENT: result.best_score,
                TRAIN_AGREEMENT: float(curve[TRAIN_AGREEMENT].iloc[result.best_epoch - 1]),
            }
        )
        logger.info(
            "fold %d, %s: inner agreement %.4f at epoch %d",
            test_fold,
            point.label,
            result.best_score,
            result.best_epoch,
        )
        models.append(model)

    selection = pd.DataFrame(rows)
    # argmax returns the first maximum: ties go to the first point in grid order.
    best = int(np.argmax(selection[VAL_AGREEMENT].to_numpy()))
    chosen, model = settings.grid[best], models[best]
    selection["chosen"] = np.arange(len(selection)) == best
    n_parameters = count_parameters(model)
    _write_fold(out_dir, chosen, selection)
    _write_summary(
        out_dir / FOLD_SUMMARY,
        {
            "test_fold": int(test_fold),
            "seed": int(seed),
            "settings": asdict(settings),
            "chosen": {"point": chosen.label, **asdict(chosen)},
            "n_parameters": n_parameters,
            "inner_validation_fold": INNER_VALIDATION_FOLD,
            "n_training_constraints": len(inner_training),
            "n_inner_constraints": len(inner_validation),
            "n_test_constraints": len(test),
            "n_training_clips": len(clip_ids),
        },
    )
    logger.info("fold %d: chose %s (%d parameters)", test_fold, chosen.label, n_parameters)
    return FoldResult(
        test_fold=int(test_fold),
        chosen=chosen,
        selection=selection,
        model=model,
        n_parameters=n_parameters,
        training_rows=inner_training.index.to_numpy(),
        inner_rows=inner_validation.index.to_numpy(),
    )


def _outer_split(
    constraints: pd.DataFrame, folds: pd.Series, test_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(training, test)`` constraints of ``test_fold``, checked to share no clip."""
    if not folds.index.equals(constraints.index):
        raise ValueError("folds must be aligned with the index of the constraints")
    present = sorted(int(fold) for fold in folds.unique())
    if test_fold not in present:
        raise ValueError(f"test_fold must be one of the folds {present}, got {test_fold!r}")
    is_test = (folds == test_fold).to_numpy()
    training, test = constraints[~is_test], constraints[is_test]
    shared = np.intersect1d(
        training[list(_CLIP_COLUMNS)].to_numpy().ravel(),
        test[list(_CLIP_COLUMNS)].to_numpy().ravel(),
    )
    if len(shared):
        raise ValueError(
            f"{len(shared)} clip(s) appear both in fold {test_fold} and outside it, "
            f"e.g. {shared[:3].tolist()}; the folds must keep whole components of clips"
        )
    return training, test


def _build_model(
    d_in: int, settings: ConstraintTrainingSettings, seed: int, test_fold: int
) -> MlpHead:
    init = torch.Generator().manual_seed(derive_seed(seed, "constraints", test_fold, "init"))
    return MlpHead(d_in, settings.hidden, settings.dim, dropout=settings.dropout, generator=init)


def _write_fold(out_dir: Path, chosen: GridPoint, selection: pd.DataFrame) -> None:
    """The selection table and copies of the chosen point's weights and curve."""
    chosen_dir = out_dir / _INNER_DIR / chosen.label
    selection.to_csv(out_dir / SELECTION_FILE, index=False)
    for name in (BEST_CHECKPOINT, CURVE_FILE):
        shutil.copyfile(chosen_dir / name, out_dir / name)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    """Write the summary through a temporary file, so its presence means it is complete."""
    partial = path.with_name(path.name + _PARTIAL_SUFFIX)
    partial.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


# --- helpers -------------------------------------------------------------------
def _project(model: nn.Module, features: np.ndarray) -> np.ndarray:
    """``model``'s ``float32`` embeddings of ``features``, in evaluation mode."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        embeddings = model(torch.from_numpy(np.asarray(features, dtype=np.float32))).numpy()
    model.train(was_training)
    return embeddings.astype(np.float32, copy=False)


def _positions_of(ids: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Positions of ``wanted`` in ``ids`` (unique), every one present."""
    lookup = pd.Index(ids)
    if not lookup.is_unique:
        raise ValueError("ids must be unique, one per row of features")
    found = lookup.get_indexer(wanted)
    if (found < 0).any():
        missing = np.asarray(wanted)[found < 0]
        raise ValueError(
            f"{len(missing)} clip(s) of the constraints have no feature row, "
            f"e.g. {missing[:3].tolist()}"
        )
    return found


def _clip_positions(ids: np.ndarray, constraints: pd.DataFrame, n_rows: int) -> np.ndarray:
    """``(n_constraints, 3)`` positions of the anchor, pos and neg clips in ``ids``."""
    ids = np.asarray(ids)
    if ids.shape != (n_rows,):
        raise ValueError(
            f"ids must be 1-D with one entry per feature row ({n_rows}), got {ids.shape}"
        )
    missing = [column for column in _CLIP_COLUMNS if column not in constraints.columns]
    if missing:
        raise ValueError(
            f"the constraints need the columns {list(_CLIP_COLUMNS)}; missing {missing}"
        )
    wanted = constraints[list(_CLIP_COLUMNS)].to_numpy()
    return _positions_of(ids, wanted.ravel()).reshape(wanted.shape).astype(np.int64)


def _check_margin(margin: float) -> None:
    if not 0.0 < margin <= _MAX_COSINE_DISTANCE:
        raise ValueError(f"margin must lie in (0, {_MAX_COSINE_DISTANCE}], got {margin}")


def _check_triples(a: torch.Tensor, p: torch.Tensor, n: torch.Tensor, n_rows: int) -> None:
    if a.ndim != 1 or len(a) == 0 or p.shape != a.shape or n.shape != a.shape:
        raise ValueError(
            "anchor, pos and neg must be non-empty 1-D and aligned per constraint, got shapes "
            f"{tuple(a.shape)}, {tuple(p.shape)} and {tuple(n.shape)}"
        )
    if a.dtype not in (torch.int32, torch.int64) or p.dtype != a.dtype or n.dtype != a.dtype:
        raise ValueError(
            "anchor, pos and neg must be integer positions of one dtype, got "
            f"{a.dtype}, {p.dtype} and {n.dtype}"
        )
    low = int(torch.stack([a, p, n]).min())
    high = int(torch.stack([a, p, n]).max())
    if low < 0 or high >= n_rows:
        raise ValueError(f"clip positions must lie in [0, {n_rows}), got {low}..{high}")
    if ((a == p) | (a == n) | (p == n)).any():
        raise ValueError("a constraint uses the same clip twice")
