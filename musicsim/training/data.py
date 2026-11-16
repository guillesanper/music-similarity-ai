"""What a training loop draws from: PxK batches, contrastive pairs and datasets.

**PxK batches.** :class:`PKSampler` builds every batch from *P* labels (genres)
and *K* items of each, the layout of the batch-hard triplet setting of Hermans
et al. (2017, "In Defense of the Triplet Loss"; not yet in ``docs/sources.md``).
A random batch of 64 FMA tracks over 8 genres holds a handful of positives per
genre at best; a PxK batch guarantees ``P·K·(K-1)/2`` same-genre pairs, so both
the contrastive and the triplet losses have something to learn from in every
step.

**Positives never share an artist.** A positive pair is two items of the same
genre by *different* artists (a decision for every learned model of this
project). Same-artist pairs would reward the network for recognising an
artist's voice or production, which the artist-filtered retrieval evaluation
never rewards and which would inflate the training signal for nothing. Each
genre block of a batch therefore holds at least two artists
(:data:`MIN_ARTISTS_PER_LABEL`); otherwise it would contribute no positive.

**Balanced pairs.** :func:`contrastive_pairs` returns as many positives as
negatives. In a PxK batch negatives outnumber positives about 8 to 1 (with
``P = K = 8``: at most 224 positives against 1792 negatives), and the contrastive
loss of Hadsell et al. (``Hadsell2006``) would then be dominated by pushing
apart pairs that are already far. Keeping every positive and drawing the same
number of negatives is our own choice of the simplest rule that balances them.

**Epochs.** An epoch has ``N // (P·K)`` batches, ``N`` being the number of
items the sampler can draw from, so every item is seen once per epoch on
average (99 batches on FMA's training split). Items are drawn without
replacement *within* a batch only; a fixed number of steps per epoch keeps
learning curves comparable between models, which strict epoch coverage would
not, since the genres differ in size.

**Datasets.** :class:`VectorDataset` serves cached feature vectors (for heads
trained on top of MERT) and :class:`MelCropDataset` random fixed-length crops
of cached log-mel spectrograms (for the CNN). Both have the same small
interface, ``len(dataset)`` and ``dataset[positions] -> np.ndarray`` with one
row per position, so one loop can drive either. With the default spectrogram
settings (22 050 Hz, hop 512) a 3 s crop is 66 150 samples, and librosa's
centred framing gives ``1 + 66150 // 512 = 1 + 129 = 130`` frames
(:func:`frames_for_seconds`).

Everything here is plain NumPy, so it is tested and reused without the optional
deep-learning extra; the training loop turns the arrays into tensors. All
randomness comes from a :class:`numpy.random.Generator` the caller takes from
:func:`musicsim.seeding.generator` with a stream name of its own.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd

__all__ = [
    "MIN_ARTISTS_PER_LABEL",
    "MelCropDataset",
    "PKSampler",
    "VectorDataset",
    "contrastive_pairs",
    "frames_for_seconds",
]

logger = logging.getLogger(__name__)

#: Fewest distinct artists a label needs inside a batch so that its block holds
#: at least one positive pair (same label, different artist).
MIN_ARTISTS_PER_LABEL = 2

#: Number of spectrogram axes, ``(n_mels, n_frames)``.
_SPECTROGRAM_NDIM = 2


# --- batches -----------------------------------------------------------------
class PKSampler:
    """Batches of ``p`` labels x ``k`` items, each label block by ≥ 2 artists.

    ``labels`` and ``artists`` are aligned per item; a batch is an ``int64``
    array of ``p * k`` positions into them, laid out label block by label block
    (positions ``[b*k, (b+1)*k)`` share one label). The ``p`` labels of a batch
    are drawn without replacement, and the ``k`` items of a label too.

    A label with fewer than ``k`` items or fewer than two artists can never fill
    a block; it is left out with a warning, and its items are never drawn.
    Iterating the sampler again continues drawing from ``generator``, so two
    epochs differ while the whole sequence stays reproducible.
    """

    def __init__(
        self,
        labels: np.ndarray,
        artists: np.ndarray,
        *,
        p: int = 8,
        k: int = 8,
        generator: np.random.Generator,
    ) -> None:
        labels = _as_column(labels, "labels")
        artists = _as_column(artists, "artists")
        _check_aligned(labels, artists)
        if p < 1:
            raise ValueError(f"p must be at least 1, got {p}")
        if k < MIN_ARTISTS_PER_LABEL:
            raise ValueError(f"k must be at least {MIN_ARTISTS_PER_LABEL}, got {k}")

        blocks: list[np.ndarray] = []
        dropped: list[str] = []
        for label in np.unique(labels):
            positions = np.flatnonzero(labels == label)
            n_artists = len(np.unique(artists[positions]))
            if len(positions) >= k and n_artists >= MIN_ARTISTS_PER_LABEL:
                blocks.append(positions)
            else:
                dropped.append(f"{label} ({len(positions)} items, {n_artists} artist(s))")
        if dropped:
            logger.warning(
                "PKSampler leaves out %d label(s) that cannot fill a block of %d items "
                "by %d artists: %s",
                len(dropped),
                k,
                MIN_ARTISTS_PER_LABEL,
                ", ".join(dropped),
            )
        if len(blocks) < p:
            raise ValueError(
                f"p={p} labels per batch, but only {len(blocks)} label(s) have at least "
                f"k={k} items by {MIN_ARTISTS_PER_LABEL} or more artists"
            )

        self.p: int = int(p)
        self.k: int = int(k)
        self._blocks = blocks
        self._artists = artists
        self._generator = generator
        self._n_items = sum(len(block) for block in blocks)

    def __len__(self) -> int:
        """Batches per epoch: the drawable items over the batch size, rounded down."""
        return self._n_items // (self.p * self.k)

    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(len(self)):
            yield self._draw_batch()

    def _draw_batch(self) -> np.ndarray:
        chosen = self._generator.choice(len(self._blocks), size=self.p, replace=False)
        return np.concatenate([self._draw_block(self._blocks[c]) for c in chosen])

    def _draw_block(self, positions: np.ndarray) -> np.ndarray:
        """``k`` items of one label; if all share an artist, the last is swapped.

        The swap takes the first item of another artist in the same random
        permutation, so it is as random as the rest and needs no rejection loop.
        It always finds one, since the constructor kept only labels with two or
        more artists.
        """
        order = self._generator.permutation(positions)
        picked = order[: self.k].copy()
        first_artist = self._artists[picked[0]]
        if np.all(self._artists[picked] == first_artist):
            rest = order[self.k :]
            picked[-1] = rest[self._artists[rest] != first_artist][0]
        return picked.astype(np.int64, copy=False)


def contrastive_pairs(
    labels: np.ndarray, artists: np.ndarray, *, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balanced positive and negative pairs among the items of one batch.

    ``labels`` and ``artists`` are those of the batch items (``labels[batch]``),
    and the returned ``(i, j, is_positive)`` index into them, with ``i < j``.
    Positives are same-label pairs by different artists; negatives are
    different-label pairs; same-label same-artist pairs are neither and never
    appear. Every positive is kept and as many negatives are drawn without
    replacement (if, unusually, negatives are fewer, the positives are drawn
    down to their number instead), so the split is exactly 50/50. Positives
    come first, each half in ``(i, j)`` order.
    """
    labels = _as_column(labels, "labels")
    artists = _as_column(artists, "artists")
    _check_aligned(labels, artists)

    first, second = np.triu_indices(len(labels), k=1)
    same_label = labels[first] == labels[second]
    positive = np.flatnonzero(same_label & (artists[first] != artists[second]))
    negative = np.flatnonzero(~same_label)
    n_each = min(len(positive), len(negative))
    if n_each == 0:
        raise ValueError(
            f"a batch needs both kinds of pair, got {len(positive)} positive(s) "
            f"and {len(negative)} negative(s)"
        )

    picked = np.concatenate(
        [
            np.sort(generator.choice(positive, size=n_each, replace=False)),
            np.sort(generator.choice(negative, size=n_each, replace=False)),
        ]
    )
    is_positive = np.repeat(np.array([True, False]), n_each)
    return first[picked].astype(np.int64), second[picked].astype(np.int64), is_positive


# --- datasets ----------------------------------------------------------------
class VectorDataset:
    """Cached feature vectors, one row per item, served as ``float32`` rows."""

    def __init__(self, features: np.ndarray) -> None:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or len(features) == 0:
            raise ValueError(
                f"features must be a non-empty (n_items, n_dims) matrix, got shape {features.shape}"
            )
        if not np.isfinite(features).all():
            raise ValueError("features contain NaN or infinite values")
        self._features = features

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, positions: np.ndarray) -> np.ndarray:
        """Rows ``(len(positions), n_dims)`` of the items at ``positions``."""
        return self._features[_check_positions(positions, len(self))]


class MelCropDataset:
    """Random ``crop_frames``-long crops of the log-mel spectrogram of each item.

    ``load(position)`` returns the cached ``(n_mels, n_frames)`` spectrogram of
    the item at ``position`` (typically a closure over
    :func:`musicsim.audio.load_or_compute_melspec` and the index), so the
    spectrograms are read lazily instead of held in memory: FMA's training
    split alone is ~2 GB of float16. Every call draws a fresh start frame per
    item, uniformly over the positions where the crop fits entirely.
    """

    def __init__(
        self,
        load: Callable[[int], np.ndarray],
        n_items: int,
        *,
        crop_frames: int,
        generator: np.random.Generator,
    ) -> None:
        if n_items < 1:
            raise ValueError(f"n_items must be at least 1, got {n_items}")
        if crop_frames < 1:
            raise ValueError(f"crop_frames must be at least 1, got {crop_frames}")
        self.crop_frames: int = int(crop_frames)
        self._load = load
        self._n_items = int(n_items)
        self._generator = generator

    def __len__(self) -> int:
        return self._n_items

    def __getitem__(self, positions: np.ndarray) -> np.ndarray:
        """Crops ``(len(positions), n_mels, crop_frames)``, ``float32``."""
        positions = _check_positions(positions, len(self))
        crops = [self._crop(int(position)) for position in positions]
        shapes = {crop.shape for crop in crops}
        if len(shapes) > 1:
            raise ValueError(f"spectrograms of one batch differ in n_mels: {sorted(shapes)}")
        return np.stack(crops).astype(np.float32, copy=False)

    def _crop(self, position: int) -> np.ndarray:
        spectrogram = np.asarray(self._load(position))
        if spectrogram.ndim != _SPECTROGRAM_NDIM:
            raise ValueError(
                f"item at position {position}: expected an (n_mels, n_frames) spectrogram, "
                f"got shape {spectrogram.shape}"
            )
        n_frames = spectrogram.shape[1]
        if n_frames < self.crop_frames:
            raise ValueError(
                f"item at position {position}: {n_frames} frames, shorter than a "
                f"{self.crop_frames}-frame crop"
            )
        start = int(self._generator.integers(0, n_frames - self.crop_frames + 1))
        return spectrogram[:, start : start + self.crop_frames]


def frames_for_seconds(seconds: float, *, sample_rate: int, hop_length: int) -> int:
    """Frames librosa's centred STFT gives for ``seconds`` of audio.

    ``1 + n_samples // hop_length`` with ``n_samples = round(seconds * sample_rate)``:
    3 s at 22 050 Hz with hop 512 is ``1 + 66150 // 512 = 130`` frames, and the
    full 30 s clip ``1 + 661500 // 512 = 1292``.
    """
    if seconds <= 0 or sample_rate <= 0 or hop_length <= 0:
        raise ValueError(
            "seconds, sample_rate and hop_length must be positive, got "
            f"{seconds}, {sample_rate} and {hop_length}"
        )
    return 1 + round(seconds * sample_rate) // hop_length


# --- validation helpers ------------------------------------------------------
def _as_column(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {array.shape}")
    if pd.isna(array).any():
        raise ValueError(f"{name} contain missing values")
    return array


def _check_aligned(labels: np.ndarray, artists: np.ndarray) -> None:
    if len(labels) != len(artists):
        raise ValueError(
            f"labels and artists must be aligned per item, got {len(labels)} and {len(artists)}"
        )


def _check_positions(positions: np.ndarray, n_items: int) -> np.ndarray:
    """Positions as a 1-D integer array within ``[0, n_items)``.

    Negative positions are rejected rather than wrapped around as NumPy would,
    since a negative position here is always a bug upstream.
    """
    positions = np.asarray(positions)
    if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
        raise ValueError(
            f"positions must be a 1-D integer array, got shape {positions.shape} "
            f"and dtype {positions.dtype}"
        )
    if len(positions) and (positions.min() < 0 or positions.max() >= n_items):
        raise ValueError(
            f"positions must lie in [0, {n_items}), got {positions.min()}..{positions.max()}"
        )
    return positions
