"""Tests for the training losses: contrastive, triplet with semi-hard mining, cross-entropy.

The embeddings are 2-D unit vectors at chosen angles, so every cosine distance
is ``1 - cos(angle between them)`` and each expected loss is worked out by hand
in the test. Skipped as a whole without the optional deep-learning extra.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from musicsim.seeding import derive_seed  # noqa: E402 (after the torch skip)
from musicsim.training.losses import (  # noqa: E402
    _pair_masks,
    _semihard_negatives,
    contrastive_loss,
    cross_entropy_loss,
    triplet_semihard_loss,
)

SEED = 7

#: Cosine distance between unit vectors 30° apart.
D30 = 1.0 - math.sqrt(3.0) / 2.0


def _at_angles(*degrees: float, dtype=torch.float32) -> torch.Tensor:
    radians = torch.tensor(degrees, dtype=torch.float64).deg2rad()
    return torch.stack([radians.cos(), radians.sin()], dim=1).to(dtype)


def _random(*shape: int, stream: str) -> torch.Tensor:
    gen = torch.Generator().manual_seed(derive_seed(SEED, "test-losses", stream))
    return torch.randn(*shape, generator=gen, dtype=torch.float64)


# --- contrastive -------------------------------------------------------------
def test_contrastive_matches_hand_computed_value() -> None:
    emb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
    first = np.array([0, 0, 1, 0])
    second = np.array([2, 1, 2, 3])
    is_positive = np.array([True, False, False, False])

    loss = contrastive_loss(emb, first, second, is_positive, margin=0.5)

    d45 = 1.0 - 1.0 / math.sqrt(2.0)
    # (0,2) positive at 45°: ½·d²; (0,1) at d = 1 and (0,3) at d = 2 are beyond
    # the margin: 0; (1,2) negative at 45°: ½·(0.5 - d)².
    expected = (0.5 * d45**2 + 0.0 + 0.5 * (0.5 - d45) ** 2 + 0.0) / 4
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_contrastive_ignores_the_norm_of_the_embeddings() -> None:
    emb = _at_angles(0, 60, 90)
    pairs = (np.array([0, 0]), np.array([1, 2]), np.array([True, False]))

    unit = contrastive_loss(emb, *pairs, margin=1.0)
    scaled = contrastive_loss(emb * torch.tensor([[3.0], [0.5], [7.0]]), *pairs, margin=1.0)

    # (0,1) positive at 60°: ½·0.5²; (0,2) negative at 90°: d = 1 = margin, 0.
    assert unit.item() == pytest.approx(0.125 / 2, rel=1e-6)
    assert scaled.item() == pytest.approx(unit.item(), rel=1e-6)


def test_contrastive_is_zero_when_every_margin_holds() -> None:
    emb = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    first = torch.tensor([0, 0, 1])
    second = torch.tensor([1, 2, 2])
    is_positive = torch.tensor([True, False, False])

    assert contrastive_loss(emb, first, second, is_positive, margin=1.0).item() == 0.0


def test_contrastive_gradient_matches_finite_differences() -> None:
    emb = _random(6, 4, stream="contrastive-grad").requires_grad_()
    first, second = np.array([0, 1, 2, 0, 3]), np.array([1, 2, 4, 5, 5])
    is_positive = np.array([True, True, False, False, False])

    assert torch.autograd.gradcheck(
        lambda x: contrastive_loss(x, first, second, is_positive, margin=1.5), (emb,)
    )


@pytest.mark.parametrize(
    ("emb", "first", "second", "is_positive", "margin", "message"),
    [
        (torch.zeros(3), [0], [1], [True], 1.0, "non-empty"),
        (torch.zeros(0, 2), [0], [1], [True], 1.0, "non-empty"),
        (torch.zeros(3, 2, dtype=torch.int64), [0], [1], [True], 1.0, "floating point"),
        (torch.ones(3, 2), [0], [1], [True], 0.0, "margin"),
        (torch.ones(3, 2), [0], [1], [True], 2.5, "margin"),
        (torch.ones(3, 2), [0, 1], [1], [True], 1.0, "aligned"),
        (torch.ones(3, 2), [], [], [], 1.0, "aligned"),
        (torch.ones(3, 2), [0.0], [1.0], [True], 1.0, "integer"),
        (torch.ones(3, 2), [0], [1], [1], 1.0, "boolean"),
        (torch.ones(3, 2), [0], [3], [True], 1.0, r"\[0, 3\)"),
        (torch.ones(3, 2), [-1], [1], [True], 1.0, r"\[0, 3\)"),
        (torch.ones(3, 2), [1], [1], [True], 1.0, "itself"),
    ],
)
def test_contrastive_rejects_bad_input(emb, first, second, is_positive, margin, message) -> None:
    with pytest.raises(ValueError, match=message):
        contrastive_loss(
            emb, np.array(first), np.array(second), np.array(is_positive), margin=margin
        )


# --- triplet -----------------------------------------------------------------
@pytest.fixture
def four_items() -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Items at 0°, 60° (label a) and 90°, 30° (label b), each by its own artist."""
    return _at_angles(0, 60, 90, 30), np.array(["a", "a", "b", "b"]), np.arange(4)


def test_pair_masks_exclude_same_artist_from_both_kinds() -> None:
    positive, negative = _pair_masks(
        np.array(["a", "a", "a", "b"]), np.array([0, 0, 1, 2]), n_items=4
    )

    np.testing.assert_array_equal(
        positive,
        [[0, 0, 1, 0], [0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
    )
    np.testing.assert_array_equal(
        negative,
        [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 1, 0]],
    )


def test_semihard_picks_the_closest_negative_beyond_the_positive(four_items) -> None:
    emb, labels, artists = four_items
    positive, negative = (torch.as_tensor(m) for m in _pair_masks(labels, artists, 4))
    distance = 1.0 - emb @ emb.T

    chosen = _semihard_negatives(distance, positive, negative)

    # Anchor 0, positive 1 at d = 0.5: negative 3 (d ≈ 0.134) is the hardest
    # but closer than the positive; negative 2 (d = 1) is the semi-hard one.
    assert chosen[0, 1].item() == 2
    # Anchor 2, positive 3 at d = 0.5: negative 1 at d ≈ 0.134, negative 0 at d = 1.
    assert chosen[2, 3].item() == 0


def test_semihard_falls_back_to_the_farthest_negative() -> None:
    # Positive at 90° (d = 1); negatives at 20° and 45°, both closer.
    emb = _at_angles(0, 90, 20, 45)
    positive, negative = (
        torch.as_tensor(m) for m in _pair_masks(np.array([0, 0, 1, 1]), np.arange(4), 4)
    )

    chosen = _semihard_negatives(1.0 - emb @ emb.T, positive, negative)

    assert chosen[0, 1].item() == 3


def test_triplet_matches_hand_computed_value(four_items) -> None:
    emb, labels, artists = four_items

    loss = triplet_semihard_loss(emb, labels, artists, margin=0.6)

    # Every anchor-positive pair is 60° apart (d = 0.5).
    # (0,1): semi-hard negative 2 at d = 1       -> 0.5 - 1 + 0.6 = 0.1
    # (1,0): both negatives at 30° (d = D30), no semi-hard: farthest -> 1.1 - D30
    # (2,3): semi-hard negative 0 at d = 1       -> 0.1
    # (3,2): both negatives at 30°, no semi-hard -> 1.1 - D30
    expected = (0.1 + (1.1 - D30) + 0.1 + (1.1 - D30)) / 4
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_triplet_is_zero_when_every_margin_holds() -> None:
    emb = _at_angles(0, 0, 180, 180)

    loss = triplet_semihard_loss(emb, np.array([0, 0, 1, 1]), np.arange(4), margin=1.0)

    # d(a, p) = 0 and d(a, n) = 2: 0 - 2 + 1 < 0 for every pair.
    assert loss.item() == 0.0


def test_triplet_gradient_matches_finite_differences() -> None:
    emb = _random(8, 4, stream="triplet-grad").requires_grad_()
    labels = np.repeat([0, 1], 4)
    artists = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    assert torch.autograd.gradcheck(
        lambda x: triplet_semihard_loss(x, labels, artists, margin=0.5), (emb,)
    )


def test_gradients_stay_finite_with_zero_and_repeated_rows() -> None:
    emb = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    labels, artists = np.array([0, 0, 1, 1]), np.arange(4)

    loss = triplet_semihard_loss(emb, labels, artists, margin=1.0) + contrastive_loss(
        emb, np.array([1, 0]), np.array([2, 3]), np.array([True, False]), margin=1.0
    )
    loss.backward()

    assert torch.isfinite(emb.grad).all()


@pytest.mark.parametrize(
    ("labels", "artists", "message"),
    [
        (["a", "a", "b"], [0, 1, 2, 3], "one entry per embedding"),
        (["a", "a", "b", "b"], [0, 0, 1, 1], "no positive pair"),
        (["a", "a", "a", "a"], [0, 1, 2, 3], "no negative pair"),
    ],
)
def test_triplet_rejects_bad_batches(labels, artists, message) -> None:
    with pytest.raises(ValueError, match=message):
        triplet_semihard_loss(
            _at_angles(0, 30, 60, 90), np.array(labels), np.array(artists), margin=0.5
        )


def test_triplet_rejects_a_bad_margin(four_items) -> None:
    emb, labels, artists = four_items
    with pytest.raises(ValueError, match="margin"):
        triplet_semihard_loss(emb, labels, artists, margin=-0.1)


# --- cross-entropy -----------------------------------------------------------
def test_cross_entropy_matches_hand_computed_value() -> None:
    logits = torch.tensor([[0.0, 0.0], [math.log(3.0), 0.0]])

    loss = cross_entropy_loss(logits, torch.tensor([0, 1]))

    # Softmaxes (1/2, 1/2) and (3/4, 1/4): -log(1/2) and -log(1/4).
    assert loss.item() == pytest.approx(1.5 * math.log(2.0), rel=1e-6)


@pytest.mark.parametrize(
    ("logits", "targets", "message"),
    [
        (torch.zeros(3), torch.tensor([0, 1, 0]), "non-empty"),
        (torch.zeros(2, 3), torch.tensor([0, 1, 0]), "int64"),
        (torch.zeros(2, 3), torch.tensor([0, 1], dtype=torch.int32), "int64"),
        (torch.zeros(2, 3), torch.tensor([0, 3]), r"\[0, 3\)"),
        (torch.zeros(2, 3), torch.tensor([-1, 0]), r"\[0, 3\)"),
    ],
)
def test_cross_entropy_rejects_bad_input(logits, targets, message) -> None:
    with pytest.raises(ValueError, match=message):
        cross_entropy_loss(logits, targets)
