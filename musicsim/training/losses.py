"""Losses of the learned similarity models: contrastive, triplet and cross-entropy.

**Cosine distance everywhere.** Both metric losses measure ``d = 1 - cos(u, v)``
between embeddings, the distance the retrieval evaluation ranks by, so what
the network is trained to separate is exactly what is scored afterwards. It
ignores the norm of the embeddings, lies in ``[0, 2]``, and for unit vectors
equals half the squared Euclidean distance FaceNet uses (``Schroff2015``), so
the triplet loss below is FaceNet's with the margin halved. Margins therefore
lie in ``(0, 2]``: a larger one could never be met.

**Contrastive loss** (``Hadsell2006``), over the pairs the caller chose (in
practice the balanced pairs of :func:`musicsim.training.data.contrastive_pairs`):
``½·d²`` for a positive pair and ``½·max(0, m - d)²`` for a negative one,
averaged over the pairs. Positives are pulled together; negatives are pushed
apart only while closer than the margin ``m``.

**Triplet loss with semi-hard negatives** (``Schroff2015``), mined inside the
PxK batch. Every ordered anchor-positive pair ``(a, p)`` gets one negative ``n``
and the term ``max(0, d(a, p) - d(a, n) + m)``; the loss is the mean over all
anchor-positive pairs, those already within the margin counting as zero. The
negative is *semi-hard*: the closest one still farther from the anchor than the
positive, ``d(a, n) > d(a, p)``. FaceNet reports that the hardest negatives
early in training lead to a collapsed model, which semi-hard ones avoid. FaceNet
does not say what to do when no negative is farther than the positive; we take
the farthest negative then, the convention of TensorFlow Addons'
``triplet_semihard_loss`` (our choice, as the gentlest of the hard ones).

Positives and negatives follow the rule of every learned model of this project
(see :mod:`musicsim.training.data`): a positive is the same label by a
*different* artist, a negative a different label, and same-label same-artist
pairs are neither.

**Cross-entropy** trains the genre classifier CNN, whose penultimate layer is
the embedding; it is :func:`torch.nn.functional.cross_entropy` with the input
checks of the other two losses.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["contrastive_loss", "cross_entropy_loss", "triplet_semihard_loss"]

#: Largest cosine distance, between opposite vectors; no margin can exceed it.
_MAX_COSINE_DISTANCE = 2.0

#: Number of axes of an embedding batch, ``(batch, dims)``.
_EMBEDDING_NDIM = 2


# --- losses ------------------------------------------------------------------
def contrastive_loss(
    embeddings: torch.Tensor,
    first: np.ndarray | torch.Tensor,
    second: np.ndarray | torch.Tensor,
    is_positive: np.ndarray | torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Mean contrastive loss of the pairs ``(first[q], second[q])`` of a batch.

    ``embeddings`` is ``(batch, dims)``; ``first``, ``second`` and
    ``is_positive`` are aligned per pair and index into its rows, as returned by
    :func:`musicsim.training.data.contrastive_pairs`.
    """
    _check_embeddings(embeddings)
    _check_margin(margin)
    device = embeddings.device
    first = torch.as_tensor(first, device=device)
    second = torch.as_tensor(second, device=device)
    is_positive = torch.as_tensor(is_positive, device=device)
    _check_pairs(first, second, is_positive, len(embeddings))

    unit = _unit_rows(embeddings)
    distance = 1.0 - (unit[first] * unit[second]).sum(dim=1)
    pull = 0.5 * distance.square()
    push = 0.5 * torch.clamp(margin - distance, min=0.0).square()
    return torch.where(is_positive, pull, push).mean()


def triplet_semihard_loss(
    embeddings: torch.Tensor,
    labels: np.ndarray,
    artists: np.ndarray,
    *,
    margin: float,
) -> torch.Tensor:
    """Mean triplet loss over every anchor-positive pair, with semi-hard negatives.

    ``labels`` and ``artists`` are those of the batch items (``labels[batch]``
    for a PxK batch), aligned with the rows of ``embeddings``.
    """
    _check_embeddings(embeddings)
    _check_margin(margin)
    positive, negative = _pair_masks(labels, artists, len(embeddings))
    if not positive.any():
        raise ValueError(
            "the batch has no positive pair (same label, different artist); "
            "a PxK batch with two artists per label always has one"
        )
    # With two labels in the batch every item has a negative, so this one check
    # guarantees each anchor one.
    if not negative.any():
        raise ValueError("the batch has no negative pair: every item shares one label")

    distance = _cosine_distances(embeddings)
    positive_t = torch.as_tensor(positive, device=embeddings.device)
    negative_t = torch.as_tensor(negative, device=embeddings.device)
    with torch.no_grad():
        chosen = _semihard_negatives(distance, positive_t, negative_t)

    anchor, pos = torch.nonzero(positive_t, as_tuple=True)
    neg = chosen[anchor, pos]
    terms = distance[anchor, pos] - distance[anchor, neg] + margin
    return torch.clamp(terms, min=0.0).mean()


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy of ``(batch, n_classes)`` logits against class indices."""
    if logits.ndim != _EMBEDDING_NDIM or len(logits) == 0:
        raise ValueError(
            f"logits must be a non-empty (batch, n_classes) tensor, got shape {tuple(logits.shape)}"
        )
    if targets.shape != logits.shape[:1] or targets.dtype != torch.int64:
        raise ValueError(
            f"targets must be int64 of shape ({len(logits)},), got dtype "
            f"{targets.dtype} and shape {tuple(targets.shape)}"
        )
    n_classes = logits.shape[1]
    if targets.min() < 0 or targets.max() >= n_classes:
        raise ValueError(
            f"targets must lie in [0, {n_classes}), got {int(targets.min())}..{int(targets.max())}"
        )
    return F.cross_entropy(logits, targets)


# --- mining ------------------------------------------------------------------
def _semihard_negatives(
    distance: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
) -> torch.Tensor:
    """For every pair ``(a, p)``, the position of its negative (``(batch, batch)`` int64).

    The closest negative of ``a`` with ``d(a, n) > d(a, p)``; if there is none,
    the farthest negative of ``a``. Entries where ``(a, p)`` is not a positive
    pair are meaningless. The cube ``(a, p, n)`` is ``batch³`` booleans, 262 144
    for a 64-item batch, so it is built whole.
    """
    beyond = negative[:, None, :] & (distance[:, None, :] > distance[:, :, None])
    closest_beyond = torch.where(beyond, distance[:, None, :], torch.inf).argmin(dim=2)
    farthest = torch.where(negative, distance, -torch.inf).argmax(dim=1)
    fallback = farthest[:, None].expand_as(closest_beyond)
    return torch.where(beyond.any(dim=2), closest_beyond, fallback)


def _pair_masks(
    labels: np.ndarray, artists: np.ndarray, n_items: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(positive, negative)`` boolean ``(n, n)`` masks of the batch items."""
    labels = np.asarray(labels)
    artists = np.asarray(artists)
    if labels.shape != (n_items,) or artists.shape != (n_items,):
        raise ValueError(
            f"labels and artists must be 1-D with one entry per embedding ({n_items}), "
            f"got shapes {labels.shape} and {artists.shape}"
        )
    same_label = labels[:, None] == labels[None, :]
    positive = same_label & (artists[:, None] != artists[None, :])
    return positive, ~same_label


def _cosine_distances(embeddings: torch.Tensor) -> torch.Tensor:
    unit = _unit_rows(embeddings)
    return 1.0 - unit @ unit.T


def _unit_rows(embeddings: torch.Tensor) -> torch.Tensor:
    """Rows scaled to unit norm (an all-zero row stays zero, at distance 1 from all)."""
    return F.normalize(embeddings, dim=1)


# --- validation helpers ------------------------------------------------------
def _check_embeddings(embeddings: torch.Tensor) -> None:
    if embeddings.ndim != _EMBEDDING_NDIM or len(embeddings) == 0:
        raise ValueError(
            f"embeddings must be a non-empty (batch, dims) tensor, got shape "
            f"{tuple(embeddings.shape)}"
        )
    if not embeddings.is_floating_point():
        raise ValueError(f"embeddings must be floating point, got dtype {embeddings.dtype}")


def _check_margin(margin: float) -> None:
    if not 0.0 < margin <= _MAX_COSINE_DISTANCE:
        raise ValueError(f"margin must lie in (0, {_MAX_COSINE_DISTANCE}], got {margin}")


def _check_pairs(
    first: torch.Tensor, second: torch.Tensor, is_positive: torch.Tensor, n_items: int
) -> None:
    n_pairs = len(first)
    if (
        first.ndim != 1
        or n_pairs == 0
        or second.shape != first.shape
        or is_positive.shape != first.shape
    ):
        raise ValueError(
            "first, second and is_positive must be non-empty 1-D and aligned per pair, "
            f"got shapes {tuple(first.shape)}, {tuple(second.shape)} and "
            f"{tuple(is_positive.shape)}"
        )
    if first.dtype not in (torch.int32, torch.int64) or second.dtype != first.dtype:
        raise ValueError(
            f"first and second must be integer positions, got dtypes {first.dtype} "
            f"and {second.dtype}"
        )
    if is_positive.dtype != torch.bool:
        raise ValueError(f"is_positive must be boolean, got dtype {is_positive.dtype}")
    low = int(torch.minimum(first, second).min())
    high = int(torch.maximum(first, second).max())
    if low < 0 or high >= n_items:
        raise ValueError(f"pair positions must lie in [0, {n_items}), got {low}..{high}")
    if (first == second).any():
        raise ValueError("a pair joins an item with itself (first == second)")
