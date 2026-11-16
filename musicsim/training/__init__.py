"""Training of the learned similarity models (siamese, triplet, CNN).

:mod:`musicsim.training.data` holds what a training loop draws from: the PxK
batch sampler, the balanced contrastive pairs of a batch and the two datasets
(cached vectors and random log-mel crops). Everything in it is plain NumPy, so
it can be tested and reused without the optional deep-learning extra; the loop
turns its arrays into tensors.
"""

from __future__ import annotations

from musicsim.training.data import (
    MelCropDataset,
    PKSampler,
    VectorDataset,
    contrastive_pairs,
    frames_for_seconds,
)

__all__ = [
    "MelCropDataset",
    "PKSampler",
    "VectorDataset",
    "contrastive_pairs",
    "frames_for_seconds",
]
