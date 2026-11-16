"""Networks of the learned similarity models: an MLP head and a small CNN.

**MlpHead** maps a fixed feature vector (a cached MERT layer, or the acoustic
descriptors) to a 128-dimensional embedding. Each hidden layer is
``Linear - BatchNorm - ReLU - Dropout(0.2)``, and a last ``Linear`` gives the
embedding, scaled to unit L2 norm inside :meth:`MlpHead.forward`. Retrieval
ranks by cosine distance, which only sees directions, so returning unit vectors
makes the head's output the embedding as it is scored, with nothing left for
the extractor to do; the losses normalise again, which changes nothing.

**CnnBackbone** maps a log-mel spectrogram to a 128-dimensional vector: five
``Conv3x3 - BatchNorm - ReLU - MaxPool2`` blocks of 32, 64, 128, 256 and 256
channels, a global average over the remaining frequency-time grid and a
``Linear(256 -> 128)``. The global average makes the output independent of the
input length, so a network trained on 3 s crops embeds a whole clip as well.
Its output is *not* normalised: :class:`CnnClassifier` puts a
``Linear(128 -> n_classes)`` on top of it, and a classifier fed unit vectors
would lose the norm as a confidence scale. Whoever takes embeddings from a
backbone normalises them there. The layer pattern and widths are this
project's choice, a common small design for 128-band spectrograms.

**No bias before BatchNorm.** A ``Linear`` or ``Conv2d`` followed by BatchNorm
has no bias: BatchNorm subtracts the batch mean, which cancels any constant
offset, and its own shift β plays the bias's role (Ioffe & Szegedy, 2015,
"Batch Normalization"; not yet in ``docs/sources.md``). The final layers,
which no BatchNorm follows, keep theirs.

**Seeded initialisation.** Every model takes a CPU :class:`torch.Generator`
(seeded from :func:`musicsim.seeding.derive_seed` with a stream name of its
own) and initialises its layers from it, never from PyTorch's global random
state. The draws are exactly those of PyTorch's default initialisation (He
uniform with ``a = √5`` for weights, ``U(±1/√fan_in)`` for biases), in the
same order, so a model is the one PyTorch would build after seeding the
global state, without touching that state. The layers are created inside
:func:`torch.random.fork_rng`, so their throw-away default initialisation does
not advance the global stream either.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import nn

__all__ = [
    "CNN_CHANNELS",
    "DROPOUT",
    "EMBEDDING_DIM",
    "CnnBackbone",
    "CnnClassifier",
    "MlpHead",
    "count_parameters",
]

#: Size of every embedding the learned models produce.
EMBEDDING_DIM = 128

#: Dropout probability after each hidden layer of :class:`MlpHead`.
DROPOUT = 0.2

#: Output channels of the five convolutional blocks of :class:`CnnBackbone`.
CNN_CHANNELS: tuple[int, ...] = (32, 64, 128, 256, 256)

#: Side of the square convolution kernels.
_KERNEL_SIZE = 3

#: Zero padding that keeps a 3x3 convolution's output the size of its input.
_PADDING = _KERNEL_SIZE // 2

#: Side of the max-pooling windows (and their stride): each block halves both axes.
_POOL_SIZE = 2

#: Smallest input side the five poolings leave at least one cell of: 2**5 = 32.
_MIN_SPECTROGRAM_SIDE = _POOL_SIZE ** len(CNN_CHANNELS)

#: Negative slope PyTorch's default weight initialisation passes to He uniform.
_DEFAULT_INIT_SLOPE = math.sqrt(5.0)

#: Axes of a batch of feature vectors, ``(batch, dims)``.
_VECTOR_NDIM = 2

#: Axes of a batch of spectrograms, ``(batch, n_mels, n_frames)``.
_SPECTROGRAM_NDIM = 3


# --- models ------------------------------------------------------------------
class MlpHead(nn.Module):
    """``(batch, d_in)`` feature vectors to ``(batch, d_out)`` unit-norm embeddings.

    ``hidden`` lists the widths of the hidden layers (``(512,)`` over a MERT
    layer of 768, ``(256,)`` over 141 acoustic descriptors); an empty
    ``hidden`` is a single linear map followed by the normalisation.
    """

    def __init__(
        self,
        d_in: int,
        hidden: Sequence[int],
        d_out: int = EMBEDDING_DIM,
        *,
        dropout: float = DROPOUT,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        hidden = tuple(hidden)
        widths = (d_in, *hidden, d_out)
        if any(isinstance(w, bool) or not isinstance(w, int) or w <= 0 for w in widths):
            raise ValueError(
                f"d_in, hidden and d_out must be positive integers, got "
                f"d_in={d_in!r}, hidden={hidden!r}, d_out={d_out!r}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {dropout}")
        _check_generator(generator)

        self.d_in = d_in
        with torch.random.fork_rng(devices=[]):
            blocks: list[nn.Module] = []
            for width_in, width_out in pairwise(widths[:-1]):
                blocks += [
                    nn.Linear(width_in, width_out, bias=False),
                    nn.BatchNorm1d(width_out),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            blocks.append(nn.Linear(widths[-2], d_out))
            self.layers = nn.Sequential(*blocks)
        _reset_parameters(self, generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != _VECTOR_NDIM or x.shape[1] != self.d_in:
            raise ValueError(f"expected a (batch, {self.d_in}) tensor, got shape {tuple(x.shape)}")
        return nn.functional.normalize(self.layers(x), dim=1)


class CnnBackbone(nn.Module):
    """``(batch, n_mels, n_frames)`` log-mel spectrograms to ``(batch, 128)`` vectors.

    The input is what :class:`musicsim.training.data.MelCropDataset` returns,
    one spectrogram per row; the single input channel is added here. Both
    spectrogram axes need at least 32 cells to survive the five poolings.
    """

    def __init__(self, *, generator: torch.Generator) -> None:
        super().__init__()
        _check_generator(generator)
        with torch.random.fork_rng(devices=[]):
            blocks: list[nn.Module] = []
            channels_in = 1
            for channels_out in CNN_CHANNELS:
                blocks += [
                    nn.Conv2d(
                        channels_in, channels_out, _KERNEL_SIZE, padding=_PADDING, bias=False
                    ),
                    nn.BatchNorm2d(channels_out),
                    nn.ReLU(),
                    nn.MaxPool2d(_POOL_SIZE),
                ]
                channels_in = channels_out
            self.blocks = nn.Sequential(*blocks)
            self.projection = nn.Linear(CNN_CHANNELS[-1], EMBEDDING_DIM)
        _reset_parameters(self, generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != _SPECTROGRAM_NDIM or min(x.shape[1:]) < _MIN_SPECTROGRAM_SIDE:
            raise ValueError(
                f"expected a (batch, n_mels, n_frames) tensor with both n_mels and "
                f"n_frames >= {_MIN_SPECTROGRAM_SIDE}, got shape {tuple(x.shape)}"
            )
        features = self.blocks(x.unsqueeze(1))
        return self.projection(features.mean(dim=(2, 3)))


class CnnClassifier(nn.Module):
    """A :class:`CnnBackbone` with a ``Linear(128 -> n_classes)`` on top.

    :meth:`forward` returns logits for the cross-entropy loss; the embedding of
    a trained classifier is the output of :attr:`backbone`.
    """

    def __init__(self, n_classes: int, *, generator: torch.Generator) -> None:
        super().__init__()
        if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
            raise ValueError(f"n_classes must be an integer >= 2, got {n_classes!r}")
        _check_generator(generator)
        self.backbone = CnnBackbone(generator=generator)
        with torch.random.fork_rng(devices=[]):
            self.classifier = nn.Linear(EMBEDDING_DIM, n_classes)
        _reset_parameters(self.classifier, generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (BatchNorm's running statistics are not)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- initialisation ----------------------------------------------------------
def _reset_parameters(model: nn.Module, generator: torch.Generator) -> None:
    """PyTorch's default initialisation of every Linear and Conv2d, drawn from ``generator``.

    Layers are visited in registration order, the order in which PyTorch draws
    their default initialisation, weight before bias. BatchNorm's
    initialisation is constant (scale 1, shift 0) and needs no draws.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear | nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=_DEFAULT_INIT_SLOPE, generator=generator)
            if module.bias is not None:
                fan_in = module.weight[0].numel()
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound, generator=generator)


def _check_generator(generator: torch.Generator) -> None:
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError(
            f"generator must be a CPU torch.Generator (models are built on the CPU), "
            f"got {generator!r}"
        )
