"""A head trained on MagnaTagATune's human similarity judgements, applied out of fold.

The models of :mod:`musicsim.training.constraints` learn from the 860 similarity
constraints of MagnaTagATune (:mod:`musicsim.datasets.mtt_triplets`). This
extractor turns them into a representation the rest of the pipeline can
evaluate like any other. On MTT that representation is *out of fold*: every clip
with a constraint is embedded by a model that never saw it. On any other corpus
(FMA's test split, for the genre ground truth) it is the projection of one
*final* model trained on every constraint. Like ``siamese_emb``, it works on the
whole index at once (:meth:`HumanSiameseExtractor.extract_all`), and
:meth:`~HumanSiameseExtractor.extract_one` refuses to run.

**Input.** The cached vectors of another extractor, ``extractor.input`` (MERT's
titular layer), read or computed through :func:`musicsim.experiments.extract_stage`
exactly as :mod:`musicsim.extractors.vector_siamese` does, so the same head is
compared over the same input. Each corpus is standardised on its own (our choice,
fixed before any result). MTT's clips use the statistics of every MTT clip of the
input cache, like ``embedding.fit_on: all`` in ``configs/datasets/mtt.yaml``.
That is unsupervised, so it leaks nothing, and the FMA statistics would be skewed
by MTT's empty band above 8 kHz. Any other corpus uses its training split, as
everywhere else.

**Cross-validation.** The constraints are split into ``extractor.folds.n_folds``
folds of whole connected components of clips
(:func:`musicsim.datasets.mtt_folds.constraint_folds`, with the generator
``seeding.generator(seed, "folds")``). The split therefore changes with the run
seed, and the spread between seeds includes the luck of the partition, which the
bootstrap over triplets does not. The model of every fold comes from
:func:`musicsim.training.constraints.train_fold`, nested selection included; its
test fold takes no part in it. ``extractor.folds`` repeats the ``folds`` section
of the MTT configuration, and the two must be equal. The copy puts the folds into
this extractor's feature cache key, which only hashes the ``extractor`` section,
so changing them can never reuse a stale out-of-fold matrix. The check keeps the
two copies in step.

**Out of fold, on MTT.** Every clip of the constraints is in exactly one test
fold, and so are the three clips of each constraint. Each clip is embedded by
the model of that fold, so every constraint is judged by one model that saw none
of its clips. Clips without constraints have no fold and are not embedded. The
returned matrix holds only the constrained clips, in index order, and they are
not reported as failures: evaluators align representations on their common ids.
The agreement on these vectors comes from ten models, and its bootstrap interval
does not include the variability between them. That spread is reported
separately, from the files below.

**Final model, on another corpus.** One model trained on every constraint, for
the evaluation of family A. Its grid point is the one with the best inner
agreement averaged over the folds of the same seed, ties going to the first in
grid order (:func:`final_selection`). It is trained on the constraints outside
fold :data:`~musicsim.training.constraints.INNER_VALIDATION_FOLD` of a fresh
split of all of them (``extractor.folds.n_folds`` folds, generator
``seeding.generator(seed, "folds", "final")``), with early stopping on that fold,
just as the models of the folds are never retrained on their inner fold. It has
seen every constraint, so it must never be evaluated on them. Its initial weights
come from ``seeding.derive_seed(seed, "constraints", "final", "init")`` and its
loop from ``seeding.derive_seed(seed, "constraints", "final")``. All of these
choices are our own, fixed before any result.

**Variants.** ``extractor.hidden: []`` is the linear map (768 -> 128), the
reference of the learned metrics of Wolff et al. (2012, ``Wolff2012``); an
experiment lists it as an override of the same extractor. The output is already
unit-norm, and an experiment evaluating it should set ``embedding.standardize:
false``: standardising the vectors of ten models together would change the
cosine they were trained on.

**Checkpoints**, in ``paths.checkpoints_dir("<name>-<hash8>")``. The hash covers
this extractor's section (architecture, grid, folds, input), the run seed and
the name of MTT's input feature cache directory, which itself hashes the dataset,
audio and input settings. Running on MTT or on another corpus therefore shares
the same folds:

* :data:`~musicsim.extractors.vector_siamese.RESOLVED_CONFIG`, the MTT
  configuration the models were trained under, and :data:`FOLDS_FILE`, the
  constraints with their outer fold;
* ``fold_<k>/``, the files of :func:`~musicsim.training.constraints.train_fold`
  (inner runs with their validation curves, ``selection.csv``, ``fold.json``);
* :data:`CROSS_VALIDATION_SUMMARY`: fold sizes, the point chosen in every fold
  and the number of trainable parameters;
* :data:`FINAL_DIR`, only once another corpus has asked for it: the files of the
  training loop, the averaged selection and :data:`FINAL_SUMMARY`.

Everything is resumed: the training loop skips the epochs already run, so a
second extraction reloads every model and ends in the same vectors.

``torch`` is imported inside the methods, like in :mod:`musicsim.extractors.mert`,
so registering this extractor does not require the optional ``[dl]`` extra.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from musicsim import paths
from musicsim.config import Config, ConfigError, compose, config_hash
from musicsim.datasets.mtt_folds import constraint_folds
from musicsim.extractors.base import Extractor
from musicsim.extractors.vector_siamese import RESOLVED_CONFIG, standardize_on_training
from musicsim.registry import DATASETS, load_plugins, register_extractor
from musicsim.seeding import derive_seed, generator

if TYPE_CHECKING:
    from torch import nn

    from musicsim.training.constraints import (
        ConstraintTrainingSettings,
        FoldResult,
        GridPoint,
    )

__all__ = [
    "CROSS_VALIDATION_SUMMARY",
    "FINAL_DIR",
    "FINAL_SUMMARY",
    "FOLDS_FILE",
    "MTT_CONFIG",
    "HumanSiameseExtractor",
    "final_selection",
    "selection_grid",
]

logger = logging.getLogger(__name__)

#: The corpus configuration the models are trained under, read when this
#: extractor runs on another corpus.
MTT_CONFIG = paths.REPO_ROOT / "configs" / "datasets" / "mtt.yaml"

#: The constraints with the outer fold of each, next to the fold directories.
FOLDS_FILE = "folds.csv"

#: Summary of the cross-validation: fold sizes, chosen points, parameters.
CROSS_VALIDATION_SUMMARY = "cross_validation.json"

#: Subdirectory of the final model, trained on every constraint.
FINAL_DIR = "final"

#: Summary of the final model; written last.
FINAL_SUMMARY = "final.json"

#: Corpus directory of MagnaTagATune, the only one with similarity constraints.
_MTT_DIRECTORY = "mtt"

#: Sections taken from :data:`MTT_CONFIG` when running on another corpus.
_MTT_SECTIONS = ("dataset", "folds")

#: Index column and value of the split another corpus is standardised on.
_SPLIT_COLUMN = "split"
_TRAINING_SPLIT = "training"

#: The three clip columns of a constraint.
_CLIP_COLUMNS = ["anchor", "pos", "neg"]

#: Keys of ``extractor.training.grid``, from the slowest to the fastest varying.
_GRID_KEYS = ("margin", "lr", "weight_decay")

#: Suffix of the temporary file a summary is written to before renaming.
_PARTIAL_SUFFIX = ".partial"


# --- pure helpers --------------------------------------------------------------
def selection_grid(raw: Any) -> tuple[GridPoint, ...]:
    """Every combination of ``extractor.training.grid``, margin slowest and weight decay fastest.

    ``raw`` maps ``margin``, ``lr`` and ``weight_decay`` each to a non-empty list.
    The order matters: ties in the selection go to the first point.
    """
    from musicsim.training.constraints import GridPoint

    if not isinstance(raw, Mapping) or set(raw) != set(_GRID_KEYS):
        raise ConfigError(
            f"extractor.training.grid must map exactly {list(_GRID_KEYS)} to lists, got {raw!r}"
        )
    values = []
    for key in _GRID_KEYS:
        options = raw[key]
        if isinstance(options, str) or not isinstance(options, Sequence) or not options:
            raise ConfigError(
                f"extractor.training.grid.{key} must be a non-empty list, got {options!r}"
            )
        values.append([float(option) for option in options])
    return tuple(
        GridPoint(margin=margin, lr=lr, weight_decay=weight_decay)
        for margin, lr, weight_decay in itertools.product(*values)
    )


def final_selection(selections: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """The grid point of the final model, from the selection tables of the folds.

    Every table is a :attr:`~musicsim.training.constraints.FoldResult.selection`,
    with the same points in the same (grid) order. Returns one row per point:
    its settings, the mean of its inner agreement over the folds
    (``mean_val_agreement``), how many folds chose it (``n_chosen``) and whether
    it wins (``chosen``): the highest mean, ties going to the first point.
    """
    from musicsim.training.constraints import VAL_AGREEMENT

    if not selections:
        raise ValueError("there is no fold selection to choose the final point from")
    points = selections[0]["point"].tolist()
    for position, table in enumerate(selections):
        if table["point"].tolist() != points:
            raise ValueError(
                f"the selection of fold {position} lists the points "
                f"{table['point'].tolist()}, not {points}"
            )
    agreement = np.stack([table[VAL_AGREEMENT].to_numpy(dtype=float) for table in selections])
    chosen = np.stack([table["chosen"].to_numpy(dtype=bool) for table in selections])

    table = selections[0][["point", *_GRID_KEYS]].reset_index(drop=True)
    table[f"mean_{VAL_AGREEMENT}"] = agreement.mean(axis=0)
    table["n_chosen"] = chosen.sum(axis=0)
    # argmax returns the first maximum: ties go to the first point in grid order.
    table["chosen"] = np.arange(len(table)) == int(np.argmax(agreement.mean(axis=0)))
    return table


@dataclass(frozen=True)
class _CrossValidation:
    """The standardised MTT input, the constraints, their folds and the model of each."""

    features: np.ndarray
    ids: np.ndarray
    constraints: pd.DataFrame
    folds: pd.Series
    results: list[FoldResult]


# --- the extractor -----------------------------------------------------------------
@register_extractor("human_siamese")
class HumanSiameseExtractor(Extractor):
    """Trains (once per configuration) heads on MTT's constraints and applies them."""

    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        raise NotImplementedError(
            f"the {self.name!r} extractor embeds the whole index at once (its models "
            "are trained on every constraint of the corpus); use extract_all"
        )

    def extract_all(
        self,
        index: pd.DataFrame,
        *,
        n_jobs: int = -1,
        progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
        """``(X, ids, failures)`` as in :meth:`Extractor.extract_all`, X unit-norm rows.

        On MTT, the out-of-fold vectors of the constrained clips; on any other
        corpus, the final model's vectors of every item. ``n_jobs`` and
        ``progress`` are accepted for the common interface; the input extractor
        reads them from the configuration.
        """
        mtt = self.mtt_config()
        if self.config.require("dataset.directory") == _MTT_DIRECTORY:
            return self._out_of_fold(mtt, index)
        return self._final(mtt, index)

    # -- configuration ------------------------------------------------------------
    def mtt_config(self) -> Config:
        """The configuration the models are trained under, checked against ``extractor.folds``.

        On MTT it is this configuration; on another corpus, this one with its
        ``dataset`` and ``folds`` sections replaced by those of :data:`MTT_CONFIG`.
        """
        if self.config.require("dataset.directory") == _MTT_DIRECTORY:
            config = self.config
        else:
            corpus = compose(MTT_CONFIG)
            data = self.config.to_dict()
            for section in _MTT_SECTIONS:
                data.pop(section, None)
                if section in corpus:
                    data[section] = corpus[section]
            config = Config(data=data, source=self.config.source)

        own = self.config.to_dict()["extractor"].get("folds")
        if own is None or own != config.get("folds"):
            raise ConfigError(
                f"extractor.folds ({own!r}) must repeat the folds section of the MTT "
                f"configuration ({config.get('folds')!r})"
            )
        return config

    def checkpoint_dir(self, mtt: Config) -> Path:
        """``paths.checkpoints_dir("<name>-<hash8>")`` of this configuration.

        ``mtt`` is :meth:`mtt_config`; see the module docstring for the hash.
        """
        key = config_hash(
            {
                "extractor": self.config.to_dict()["extractor"],
                "seed": int(self.config.require("seed")),
                "input_features": _input_features_name(mtt),
            }
        )
        return paths.checkpoints_dir(f"{self.name}-{key[:8]}")

    def settings(self) -> ConstraintTrainingSettings:
        """Architecture, grid and loop settings of every fold, from ``extractor``."""
        from musicsim.training.constraints import ConstraintTrainingSettings

        return ConstraintTrainingSettings(
            hidden=tuple(int(width) for width in self.config.require("extractor.hidden")),
            dim=self.dim,
            dropout=float(self.config.require("extractor.dropout")),
            grid=selection_grid(self.config.require("extractor.training.grid")),
            batch_size=int(self.config.require("extractor.training.batch_size")),
            max_epochs=int(self.config.require("extractor.training.max_epochs")),
            patience=int(self.config.require("extractor.training.patience")),
            inner_folds=int(self.config.require("extractor.training.inner_folds")),
            max_imbalance=float(self.config.require("extractor.folds.max_imbalance")),
        )

    # -- cross-validation ---------------------------------------------------------
    def _cross_validate(self, mtt: Config, mtt_index: pd.DataFrame) -> _CrossValidation:
        """Split the constraints into folds and train (or reload) the model of each."""
        from musicsim.training import constraints as training

        seed = int(self.config.require("seed"))
        settings = self.settings()
        n_folds = int(self.config.require("extractor.folds.n_folds"))

        inputs, ids, _ = _cached_input(mtt, mtt_index)
        features = standardize_on_training(inputs, np.ones(len(ids), dtype=bool))
        constraints = _read_constraints(mtt)
        clips = np.unique(constraints[_CLIP_COLUMNS].to_numpy().ravel())
        missing = np.setdiff1d(clips, ids)
        if len(missing):
            raise ValueError(
                f"{len(missing)} clip(s) of the constraints have no input features, "
                f"e.g. {missing[:3].tolist()}"
            )
        folds = constraint_folds(
            constraints,
            n_folds=n_folds,
            max_imbalance=settings.max_imbalance,
            generator=generator(seed, "folds"),
        )

        out_dir = self.checkpoint_dir(mtt)
        out_dir.mkdir(parents=True, exist_ok=True)
        mtt.write(out_dir / RESOLVED_CONFIG)
        constraints.assign(fold=folds).to_csv(out_dir / FOLDS_FILE, index=False)
        results = [
            training.train_fold(
                features,
                ids,
                constraints,
                folds,
                test_fold=fold,
                settings=settings,
                seed=seed,
                out_dir=out_dir / f"fold_{fold}",
            )
            for fold in range(n_folds)
        ]

        clips_of_fold = _clip_folds(constraints, folds)
        _write_summary(
            out_dir / CROSS_VALIDATION_SUMMARY,
            {
                "seed": seed,
                "n_folds": n_folds,
                "max_imbalance": settings.max_imbalance,
                "n_constraints": len(constraints),
                "n_clips": len(clips),
                "n_parameters": results[0].n_parameters,
                "input_features": _input_features_name(mtt),
                "folds": [
                    {
                        "test_fold": result.test_fold,
                        "chosen": result.chosen.label,
                        "n_parameters": result.n_parameters,
                        "n_test_constraints": int((folds == result.test_fold).sum()),
                        "n_test_clips": int((clips_of_fold == result.test_fold).sum()),
                    }
                    for result in results
                ],
            },
        )
        logger.info(
            "%s: %d folds trained in %s (%d trainable parameters each)",
            self.name,
            n_folds,
            out_dir,
            results[0].n_parameters,
        )
        return _CrossValidation(features, ids, constraints, folds, results)

    def _out_of_fold(
        self, mtt: Config, index: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
        """Every constrained clip of ``index``, embedded by the model of its test fold."""
        cv = self._cross_validate(mtt, index)
        clips_of_fold = _clip_folds(cv.constraints, cv.folds)
        wanted = index["item_id"].to_numpy(dtype=np.int64)
        wanted = wanted[np.isin(wanted, clips_of_fold.index.to_numpy())]
        fold_of_wanted = clips_of_fold.loc[wanted].to_numpy()
        positions = pd.Index(cv.ids).get_indexer(wanted)

        parts: list[np.ndarray] = []
        rows: list[np.ndarray] = []
        for result in cv.results:
            in_fold = np.flatnonzero(fold_of_wanted == result.test_fold)
            parts.append(_project(result.model, cv.features[positions[in_fold]]))
            rows.append(in_fold)
        X = np.concatenate(parts)[np.argsort(np.concatenate(rows))]
        logger.info(
            "%s: %d constrained clips embedded out of fold; the other %d clips have no fold",
            self.name,
            len(wanted),
            len(index) - len(wanted),
        )
        return X, wanted, []

    # -- the final model ------------------------------------------------------------
    def _final(
        self, mtt: Config, index: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
        """Every item of ``index``, embedded by the model trained on every constraint."""
        mtt_index = pd.read_csv(paths.dataset_index(mtt.require("dataset.directory")))
        cv = self._cross_validate(mtt, mtt_index)
        model = self._train_final(cv, self.checkpoint_dir(mtt) / FINAL_DIR)

        if _SPLIT_COLUMN not in index.columns:
            raise ValueError(
                f"the index needs a {_SPLIT_COLUMN!r} column to standardise the input on "
                f"its {_TRAINING_SPLIT!r} items; it has {list(index.columns)}"
            )
        inputs, ids, failures = _cached_input(self.config, index)
        split = index.set_index("item_id").loc[ids, _SPLIT_COLUMN].to_numpy()
        features = standardize_on_training(inputs, split == _TRAINING_SPLIT)
        return _project(model, features), ids, failures

    def _train_final(self, cv: _CrossValidation, out_dir: Path) -> nn.Module:
        """Train (or resume) the final model in ``out_dir`` and write its summary."""
        import torch

        from musicsim.training.constraints import (
            INNER_VALIDATION_FOLD,
            SELECTION_FILE,
            TRAIN_AGREEMENT,
            AgreementValidator,
            ConstraintTask,
        )
        from musicsim.training.models import MlpHead, count_parameters
        from musicsim.training.train import CURVE_FILE, LAST_CHECKPOINT, TrainingSettings, train

        seed = int(self.config.require("seed"))
        settings = self.settings()
        selection = final_selection([result.selection for result in cv.results])
        point = settings.grid[int(np.flatnonzero(selection["chosen"].to_numpy())[0])]

        split = constraint_folds(
            cv.constraints,
            n_folds=int(self.config.require("extractor.folds.n_folds")),
            max_imbalance=settings.max_imbalance,
            generator=generator(seed, "folds", "final"),
        )
        is_held_out = (split == INNER_VALIDATION_FOLD).to_numpy()
        training, held_out = cv.constraints[~is_held_out], cv.constraints[is_held_out]
        clip_ids = np.unique(cv.constraints[_CLIP_COLUMNS].to_numpy().ravel())
        clip_features = cv.features[pd.Index(cv.ids).get_indexer(clip_ids)]

        init = torch.Generator().manual_seed(derive_seed(seed, "constraints", "final", "init"))
        model = MlpHead(
            clip_features.shape[1],
            settings.hidden,
            settings.dim,
            dropout=settings.dropout,
            generator=init,
        )
        result = train(
            model,
            ConstraintTask(
                clip_features,
                clip_ids,
                training,
                margin=point.margin,
                batch_size=settings.batch_size,
            ),
            AgreementValidator(clip_features, clip_ids, training, held_out),
            TrainingSettings(
                lr=point.lr,
                max_epochs=settings.max_epochs,
                patience=settings.patience,
                weight_decay=point.weight_decay,
            ),
            out_dir,
            seed=derive_seed(seed, "constraints", "final"),
            resume=(out_dir / LAST_CHECKPOINT).is_file(),
        )

        curve = pd.read_csv(out_dir / CURVE_FILE)
        selection.to_csv(out_dir / SELECTION_FILE, index=False)
        n_parameters = count_parameters(model)
        _write_summary(
            out_dir / FINAL_SUMMARY,
            {
                "seed": seed,
                "chosen": {"point": point.label, **asdict(point)},
                "n_parameters": n_parameters,
                "held_out_fold": INNER_VALIDATION_FOLD,
                "n_training_constraints": len(training),
                "n_held_out_constraints": len(held_out),
                "best_epoch": result.best_epoch,
                "epochs": result.epochs,
                "stopped_early": result.stopped_early,
                "held_out_agreement": result.best_score,
                "train_agreement": float(curve[TRAIN_AGREEMENT].iloc[result.best_epoch - 1]),
            },
        )
        logger.info(
            "%s: final model %s, held-out agreement %.4f at epoch %d (%d parameters)",
            self.name,
            point.label,
            result.best_score,
            result.best_epoch,
            n_parameters,
        )
        return model


# --- module helpers ---------------------------------------------------------------
def _input_config(config: Config) -> Config:
    """``config`` with the input extractor swapped in and its overrides applied."""
    from musicsim.representations import RepresentationSpec, spec_config

    name = str(config.require("extractor.input.extractor"))
    overrides = config.get("extractor.input.overrides") or {}
    spec = RepresentationSpec(
        label=name,
        extractor=name,
        overrides={f"extractor.{key}": value for key, value in overrides.items()},
    )
    return spec_config(config, spec)


def _input_features_name(config: Config) -> str:
    """Name of ``config``'s input feature cache directory, which hashes its settings."""
    from musicsim.embeddings import features_cache_dir

    return features_cache_dir(_input_config(config)).name


def _cached_input(
    config: Config, index: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
    """The input features of ``index``'s items, in index order, and the items without."""
    from musicsim.experiments import extract_stage

    input_config = _input_config(config)
    cache = extract_stage(input_config).cache_dir
    cached = np.load(cache / "features.npy")
    cached_ids = np.load(cache / "ids.npy")

    wanted = index["item_id"].to_numpy(dtype=np.int64)
    rows = pd.Index(cached_ids).get_indexer(wanted)
    input_name = input_config.require("extractor.name")
    failures = [
        (int(item_id), f"no {input_name!r} features for this item") for item_id in wanted[rows < 0]
    ]
    found = rows >= 0
    return cached[rows[found]], wanted[found], failures


def _read_constraints(config: Config) -> pd.DataFrame:
    """The constraints the MTT dataset class wrote next to its index."""
    load_plugins()
    directory = config.require("dataset.directory")
    path = DATASETS.get(directory)(config).triplet_constraints_path
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `musicsim index --dataset {directory}` first"
        )
    return pd.read_csv(path)


def _clip_folds(constraints: pd.DataFrame, folds: pd.Series) -> pd.Series:
    """The test fold of every clip of the constraints, indexed by clip id."""
    long = pd.DataFrame(
        {
            "clip": constraints[_CLIP_COLUMNS].to_numpy().ravel(),
            "fold": np.repeat(folds.to_numpy(), len(_CLIP_COLUMNS)),
        }
    ).drop_duplicates()
    if long["clip"].duplicated().any():
        raise ValueError("a clip of the constraints lies in more than one fold")
    return long.set_index("clip")["fold"].sort_index()


def _project(model: nn.Module, features: np.ndarray) -> np.ndarray:
    """``model``'s ``float32`` embeddings of ``features``, in evaluation mode."""
    import torch

    was_training = model.training
    model.eval()
    with torch.no_grad():
        embeddings = model(torch.from_numpy(np.asarray(features, dtype=np.float32))).numpy()
    model.train(was_training)
    return embeddings.astype(np.float32, copy=False)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    """Write the summary through a temporary file, so its presence means it is complete."""
    partial = path.with_name(path.name + _PARTIAL_SUFFIX)
    partial.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)
