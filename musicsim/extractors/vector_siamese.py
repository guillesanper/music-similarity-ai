"""A siamese MLP head trained on top of another extractor's cached vectors.

This extractor does not look at audio. Its input is the cached feature matrix of
another extractor (``extractor.input.extractor``, MERT's titular layer for
``siamese_emb``), and its output is the 128-dimensional embedding of an
:class:`~musicsim.training.models.MlpHead` trained with the contrastive loss
(``Hadsell2006``) to bring same-genre tracks by different artists together. It
therefore works on the whole index at once (:meth:`VectorSiameseExtractor.extract_all`)
rather than item by item, and :meth:`~VectorSiameseExtractor.extract_one` refuses
to run.

**Input.** The input extractor's configuration is the experiment's own with the
``extractor`` section swapped for ``configs/extractors/<input>.yaml`` and
``extractor.input.overrides`` applied on top
(:func:`musicsim.representations.spec_config`), and its features are computed or
reused through :func:`musicsim.experiments.extract_stage`, so a MERT cache that
already exists is read, never recomputed. The overrides pin the settings the head
was designed for (``mert_layer: 6``, the titular layer chosen on validation): they
are part of this extractor's own section, so changing them changes both its
feature cache and its checkpoint. Items the input extractor failed on are
recorded as failures here as well.

**Standardisation on training.** Each input dimension is centred and scaled by
the mean and standard deviation of the training split only, as
:mod:`musicsim.embeddings` does for every representation: statistics of
validation or test items must not shape what the network sees. A constant
dimension keeps a scale of one, the convention of scikit-learn's
``StandardScaler``.

**Training.** Positives are same-genre pairs by different artists and negatives
different-genre pairs, drawn as balanced pairs from PxK batches
(:mod:`musicsim.training.data`); the loss is the cosine contrastive loss with
margin ``extractor.training.margin`` (:mod:`musicsim.training.losses`). The loop,
early stopping on validation Recall@10 and the checkpoint files are those of
:func:`musicsim.training.train.train`. The validation items are FMA's validation
split, each excluded from its near-duplicates' rankings; those groups are computed
once on the standardised input, since they are a property of the audio, with the
threshold the layer selection also uses
(:data:`musicsim.graphs.duplicates.NEAR_DUP_SIM`). Test items are only projected.

**Checkpoints.** A trained head lives in
``paths.checkpoints_dir("<name>-<hash8>")``, the hash covering this extractor's
whole section (architecture, training settings, input and its overrides), the
run seed and the name of the input's feature cache directory, which itself hashes
the dataset, audio and input settings. Next to the files of the training loop it
holds ``config.resolved.yaml`` (the configuration the head was trained with) and
:data:`TRAINING_SUMMARY`, written last: its presence means the training finished,
and a later extraction only loads :data:`~musicsim.training.train.BEST_CHECKPOINT`.
A directory with :data:`~musicsim.training.train.LAST_CHECKPOINT` but no summary
is an interrupted run, and is resumed. The summary records the number of trainable
parameters, how the run ended and the collapse diagnostics below.

**Collapse diagnostics.** A metric loss can be satisfied by an embedding that uses
very few directions; recall on eight genres would not show it. After training,
the validation embeddings are summarised by their effective rank and mean cosine
(:mod:`musicsim.evaluation.representation`), and a warning is logged when the
effective rank is at most :data:`MIN_EFFECTIVE_RANK` or the mean cosine at least
:data:`MAX_MEAN_COSINE`. Both thresholds are this project's choice. A collapsed
head is reported, not rejected: whether to use it is the experiment's call.

``torch`` is imported inside the methods, like in :mod:`musicsim.extractors.mert`,
so registering this extractor does not require the optional ``[dl]`` extra.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from musicsim import paths
from musicsim.config import Config, config_hash
from musicsim.evaluation.representation import effective_rank, mean_cosine
from musicsim.extractors.base import Extractor
from musicsim.graphs.duplicates import NEAR_DUP_SIM, duplicate_groups, near_duplicate_pairs
from musicsim.registry import register_extractor
from musicsim.relevance.base import label_codes
from musicsim.seeding import derive_seed
from musicsim.training.data import PKSampler, VectorDataset, contrastive_pairs

if TYPE_CHECKING:
    import torch
    from torch import nn

__all__ = [
    "MAX_MEAN_COSINE",
    "MIN_EFFECTIVE_RANK",
    "RESOLVED_CONFIG",
    "TRAINING_SUMMARY",
    "VectorContrastiveTask",
    "VectorSiameseExtractor",
    "collapse_diagnostics",
    "standardize_on_training",
]

logger = logging.getLogger(__name__)

#: Summary of a finished training run; its presence marks the run as finished.
TRAINING_SUMMARY = "training.json"

#: The configuration the head was trained with, next to its checkpoints.
RESOLVED_CONFIG = "config.resolved.yaml"

#: An embedding whose effective rank is at most this is reported as collapsed
#: (this project's choice).
MIN_EFFECTIVE_RANK = 10.0

#: An embedding whose mean cosine between items is at least this is reported as
#: collapsed (this project's choice).
MAX_MEAN_COSINE = 0.9

#: Index columns the training reads: split, relevance label and artist.
_SPLIT_COLUMN = "split"
_GENRE_COLUMN = "genre_top"
_ARTIST_COLUMN = "artist_id"

#: Split the head and the input standardisation are fitted on.
_TRAINING_SPLIT = "training"

#: Split early stopping is scored on.
_VALIDATION_SPLIT = "validation"

#: Suffix of the temporary file the summary is written to before renaming.
_PARTIAL_SUFFIX = ".partial"


# --- pure helpers --------------------------------------------------------------
def standardize_on_training(features: np.ndarray, is_training: np.ndarray) -> np.ndarray:
    """``features`` centred and scaled per column by the training rows' statistics.

    The standard deviation is the population one (``ddof=0``); a column that is
    constant on the training rows is only centred. Returns ``float32``.
    """
    features = np.asarray(features, dtype=np.float64)
    is_training = np.asarray(is_training, dtype=bool)
    if features.ndim != 2 or is_training.shape != (len(features),):
        raise ValueError(
            "features must be (n_items, n_dims) and is_training (n_items,), got shapes "
            f"{features.shape} and {is_training.shape}"
        )
    if not is_training.any():
        raise ValueError("no training row to compute the standardisation statistics on")
    mean = features[is_training].mean(axis=0)
    std = features[is_training].std(axis=0)
    std[std == 0] = 1.0
    return ((features - mean) / std).astype(np.float32)


def collapse_diagnostics(embeddings: np.ndarray) -> dict[str, Any]:
    """Effective rank and mean cosine of ``embeddings``, and whether they signal collapse.

    Logs a warning when they do (see the module docstring for the thresholds).
    """
    rank = effective_rank(embeddings)
    cosine = mean_cosine(embeddings)
    collapsed = rank <= MIN_EFFECTIVE_RANK or cosine >= MAX_MEAN_COSINE
    if collapsed:
        logger.warning(
            "the embedding looks collapsed: effective rank %.2f (must exceed %.0f), "
            "mean cosine %.3f (must stay below %.2f)",
            rank,
            MIN_EFFECTIVE_RANK,
            cosine,
            MAX_MEAN_COSINE,
        )
    return {"effective_rank": rank, "mean_cosine": cosine, "collapsed": bool(collapsed)}


# --- the training task -----------------------------------------------------------
class VectorContrastiveTask:
    """Contrastive training of a head on fixed vectors.

    It is the :class:`~musicsim.training.train.TrainingTask` of this extractor.
    ``features``, ``labels`` and ``artists`` are aligned per training item;
    ``validation_features`` are the validation items, in the order of the
    :class:`~musicsim.training.train.ValidationSet`.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        artists: np.ndarray,
        validation_features: np.ndarray,
        *,
        margin: float,
        p: int,
        k: int,
    ) -> None:
        self._features = VectorDataset(features)
        self._validation = VectorDataset(validation_features)
        labels = np.asarray(labels)
        artists = np.asarray(artists)
        if labels.shape != (len(self._features),) or artists.shape != labels.shape:
            raise ValueError(
                f"labels and artists must be aligned with the {len(self._features)} training "
                f"items, got shapes {labels.shape} and {artists.shape}"
            )
        self._labels = labels
        self._artists = artists
        self.margin = float(margin)
        self.p = int(p)
        self.k = int(k)

    def batches(self, rng: np.random.Generator) -> PKSampler:
        """One epoch of PxK batches, drawn from the loop's generator."""
        return PKSampler(self._labels, self._artists, p=self.p, k=self.k, generator=rng)

    def loss(self, model: nn.Module, batch: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        """Contrastive loss of the batch's balanced positive and negative pairs."""
        import torch

        from musicsim.training.losses import contrastive_loss

        first, second, is_positive = contrastive_pairs(
            self._labels[batch], self._artists[batch], generator=rng
        )
        embeddings = model(torch.from_numpy(self._features[batch]))
        return contrastive_loss(embeddings, first, second, is_positive, margin=self.margin)

    def embed_validation(self, model: nn.Module) -> np.ndarray:
        """Embeddings of the validation items, in order."""
        return _project(model, self._validation[np.arange(len(self._validation))])


# --- the extractor -----------------------------------------------------------------
@register_extractor("siamese_emb")
class VectorSiameseExtractor(Extractor):
    """Trains (once per configuration) and applies a siamese head over cached vectors."""

    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        raise NotImplementedError(
            f"the {self.name!r} extractor embeds the whole index at once (its input "
            "standardisation and training need every item); use extract_all"
        )

    def extract_all(
        self,
        index: pd.DataFrame,
        *,
        n_jobs: int = -1,
        progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
        """``(X, ids, failures)`` as in :meth:`Extractor.extract_all`, X unit-norm rows.

        ``n_jobs`` and ``progress`` are accepted for the common interface; the
        input extractor reads them from the configuration, and the training loop
        logs its own progress once per epoch.
        """
        import torch

        from musicsim.training.models import count_parameters
        from musicsim.training.train import BEST_CHECKPOINT

        inputs, ids, failures = self._input_features(index)
        items = _item_columns(index, ids)
        features = standardize_on_training(
            inputs, items[_SPLIT_COLUMN].to_numpy() == _TRAINING_SPLIT
        )
        model = self._build_model(features.shape[1])
        logger.info("%s: %d trainable parameters", self.name, count_parameters(model))

        out_dir = self.checkpoint_dir()
        if (out_dir / TRAINING_SUMMARY).is_file():
            logger.info("%s: using the trained head in %s", self.name, out_dir)
            model.load_state_dict(torch.load(out_dir / BEST_CHECKPOINT, weights_only=True))
        else:
            self._train(model, features, ids, items, out_dir)
        return _project(model, features), ids, failures

    # -- input ----------------------------------------------------------------
    def input_config(self) -> Config:
        """The configuration of the input extractor, overrides applied."""
        from musicsim.representations import RepresentationSpec, spec_config

        name = str(self.config.require("extractor.input.extractor"))
        overrides = self.config.get("extractor.input.overrides") or {}
        spec = RepresentationSpec(
            label=name,
            extractor=name,
            overrides={f"extractor.{key}": value for key, value in overrides.items()},
        )
        return spec_config(self.config, spec)

    def _input_features(
        self, index: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
        """The input features of ``index``'s items, in index order, and the items without."""
        from musicsim.experiments import extract_stage

        input_config = self.input_config()
        cache = extract_stage(input_config).cache_dir
        cached = np.load(cache / "features.npy")
        cached_ids = np.load(cache / "ids.npy")

        wanted = index["item_id"].to_numpy(dtype=np.int64)
        rows = pd.Index(cached_ids).get_indexer(wanted)
        input_name = input_config.require("extractor.name")
        failures = [
            (int(item_id), f"no {input_name!r} features for this item")
            for item_id in wanted[rows < 0]
        ]
        found = rows >= 0
        return cached[rows[found]], wanted[found], failures

    def checkpoint_dir(self) -> Path:
        """``paths.checkpoints_dir("<name>-<hash8>")`` of this configuration.

        See the module docstring for what the hash covers.
        """
        key = config_hash(
            {
                "extractor": self.config.to_dict()["extractor"],
                "seed": int(self.config.require("seed")),
                "input_features": self._input_features_name(),
            }
        )
        return paths.checkpoints_dir(f"{self.name}-{key[:8]}")

    # -- model and training ---------------------------------------------------------
    def _build_model(self, d_in: int) -> nn.Module:
        import torch

        from musicsim.training.models import DROPOUT, MlpHead

        seed = int(self.config.require("seed"))
        generator = torch.Generator().manual_seed(derive_seed(seed, "extractor", self.name, "init"))
        return MlpHead(
            d_in,
            [int(width) for width in self.config.require("extractor.hidden")],
            self.dim,
            dropout=float(self.config.get("extractor.dropout", DROPOUT)),
            generator=generator,
        )

    def _train(
        self,
        model: nn.Module,
        features: np.ndarray,
        ids: np.ndarray,
        items: pd.DataFrame,
        out_dir: Path,
    ) -> None:
        """Train ``model`` (resuming an interrupted run) and write the summary."""
        from musicsim.training.models import count_parameters
        from musicsim.training.train import (
            LAST_CHECKPOINT,
            TrainingSettings,
            ValidationSet,
            train,
        )

        split = items[_SPLIT_COLUMN].to_numpy()
        train_rows = np.flatnonzero(split == _TRAINING_SPLIT)
        val_rows = np.flatnonzero(split == _VALIDATION_SPLIT)
        if len(val_rows) == 0:
            raise ValueError(
                f"no {_VALIDATION_SPLIT!r} item to select the epoch on; the index's "
                f"{_SPLIT_COLUMN!r} column holds {sorted(set(split))}"
            )

        genres = items[_GENRE_COLUMN]
        val_ids = ids[val_rows]
        val_features = features[val_rows]
        validation = ValidationSet(
            val_ids, _exclusion_groups(val_features, val_ids), label_codes(genres.iloc[val_rows])
        )
        task = VectorContrastiveTask(
            features[train_rows],
            label_codes(genres.iloc[train_rows]),
            items[_ARTIST_COLUMN].to_numpy()[train_rows],
            val_features,
            margin=float(self.config.require("extractor.training.margin")),
            p=int(self.config.require("extractor.training.p")),
            k=int(self.config.require("extractor.training.k")),
        )
        settings = TrainingSettings(
            lr=float(self.config.require("extractor.training.lr")),
            max_epochs=self.config.require("extractor.training.max_epochs"),
            patience=self.config.require("extractor.training.patience"),
            weight_decay=float(self.config.require("extractor.training.weight_decay")),
        )

        resume = (out_dir / LAST_CHECKPOINT).is_file()
        logger.info(
            "%s: %s training in %s (%d training, %d validation items)",
            self.name,
            "resuming" if resume else "starting",
            out_dir,
            len(train_rows),
            len(val_rows),
        )
        self.config.write(out_dir / RESOLVED_CONFIG)
        result = train(
            model,
            task,
            validation,
            settings,
            out_dir,
            seed=int(self.config.require("seed")),
            resume=resume,
        )

        diagnostics = collapse_diagnostics(_project(model, val_features))
        summary = {
            "n_parameters": count_parameters(model),
            "best_epoch": result.best_epoch,
            "best_recall_at_10": result.best_recall,
            "epochs": result.epochs,
            "stopped_early": result.stopped_early,
            "n_training_items": len(train_rows),
            "n_validation_items": len(val_rows),
            "input_features": self._input_features_name(),
            "validation_effective_rank": diagnostics["effective_rank"],
            "validation_mean_cosine": diagnostics["mean_cosine"],
            "collapsed": diagnostics["collapsed"],
        }
        _write_summary(out_dir / TRAINING_SUMMARY, summary)
        logger.info(
            "%s: best validation R@10 %.4f at epoch %d of %d",
            self.name,
            result.best_recall,
            result.best_epoch,
            result.epochs,
        )

    def _input_features_name(self) -> str:
        """Name of the input's feature cache directory, which hashes its settings."""
        from musicsim.embeddings import features_cache_dir

        return features_cache_dir(self.input_config()).name


# --- module helpers ---------------------------------------------------------------
def _item_columns(index: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    """Split, genre and artist of ``ids``, in order."""
    columns = [_SPLIT_COLUMN, _GENRE_COLUMN, _ARTIST_COLUMN]
    missing = [column for column in ["item_id", *columns] if column not in index.columns]
    if missing:
        raise ValueError(
            f"the index needs the columns item_id, {', '.join(columns)}; missing {missing}"
        )
    return index.set_index("item_id").loc[ids, columns].reset_index(drop=True)


def _exclusion_groups(features: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Near-duplicate group of every item, from the cosine of its input vector."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    unit = features / np.where(norms == 0, 1.0, norms)
    return duplicate_groups(ids, near_duplicate_pairs(unit, ids, threshold=NEAR_DUP_SIM))


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
