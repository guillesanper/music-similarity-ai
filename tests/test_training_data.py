"""Tests for the training data: PxK sampler, contrastive pairs and datasets.

Every case is built by hand so the expected batch layout, pair counts and crop
offsets are known in advance.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from musicsim.seeding import generator
from musicsim.training.data import (
    MelCropDataset,
    PKSampler,
    VectorDataset,
    contrastive_pairs,
    frames_for_seconds,
)

SEED = 7


def _gen(*stream: str) -> np.random.Generator:
    return generator(SEED, "test-training-data", *stream)


@pytest.fixture
def three_genres() -> tuple[np.ndarray, np.ndarray]:
    """Three genres of 6 items each, two items per artist (3 artists per genre)."""
    labels = np.repeat(np.array(["a", "b", "c"]), 6)
    artists = np.arange(18) // 2
    return labels, artists


# --- PKSampler ---------------------------------------------------------------
def test_sampler_batches_have_exact_pk_layout(three_genres) -> None:
    labels, artists = three_genres
    sampler = PKSampler(labels, artists, p=2, k=3, generator=_gen("layout"))
    # 18 items over batches of 2 * 3 = 6 -> 3 batches per epoch.
    assert len(sampler) == 3

    batches = list(sampler)
    assert len(batches) == 3
    for batch in batches:
        assert batch.dtype == np.int64
        assert batch.shape == (6,)
        assert len(np.unique(batch)) == 6
        blocks = batch.reshape(2, 3)
        block_labels = [set(labels[block]) for block in blocks]
        assert all(len(found) == 1 for found in block_labels)
        assert block_labels[0] != block_labels[1]
        assert all(len(np.unique(artists[block])) >= 2 for block in blocks)


def test_sampler_swaps_in_a_second_artist_when_a_block_has_one() -> None:
    # Five items by artist 0 and one (item 5) by artist 1, k = 5: one random
    # permutation in six leaves item 5 out of the first five, and the swap must
    # bring it back. So item 5 is in every block of "a", and item 11 in every
    # block of "b".
    labels = np.array(["a"] * 6 + ["b"] * 6)
    artists = np.array([0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 3])
    sampler = PKSampler(labels, artists, p=2, k=5, generator=_gen("swap"))
    for _ in range(50):
        for batch in sampler:
            for block in batch.reshape(2, 5):
                lone = 5 if labels[block[0]] == "a" else 11
                assert lone in block
                assert len(np.unique(block)) == 5


def test_sampler_leaves_out_labels_that_cannot_fill_a_block(three_genres, caplog) -> None:
    labels, artists = three_genres
    # "d": only 2 items (< k = 3); "e": 4 items by a single artist.
    labels = np.concatenate([labels, ["d", "d", "e", "e", "e", "e"]])
    artists = np.concatenate([artists, [100, 101, 200, 200, 200, 200]])
    with caplog.at_level(logging.WARNING, logger="musicsim.training.data"):
        sampler = PKSampler(labels, artists, p=2, k=3, generator=_gen("drop"))
    assert "d (2 items, 2 artist(s))" in caplog.text
    assert "e (4 items, 1 artist(s))" in caplog.text
    # Only the 18 items of a, b, c count: 18 // 6 = 3 batches.
    assert len(sampler) == 3
    drawn = np.concatenate([batch for _ in range(20) for batch in sampler])
    assert set(labels[drawn]) == {"a", "b", "c"}


def test_sampler_is_deterministic_per_stream(three_genres) -> None:
    labels, artists = three_genres

    def epochs(stream: str) -> list[np.ndarray]:
        sampler = PKSampler(labels, artists, p=2, k=3, generator=_gen(stream))
        return [batch for _ in range(4) for batch in sampler]

    first, again, other = epochs("same"), epochs("same"), epochs("other")
    assert all(np.array_equal(x, y) for x, y in zip(first, again, strict=True))
    assert not all(np.array_equal(x, y) for x, y in zip(first, other, strict=True))
    # Two epochs of one sampler differ: iterating continues the generator.
    assert not all(np.array_equal(x, y) for x, y in zip(first[:3], first[3:6], strict=True))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"p": 0}, "p must be at least 1, got 0"),
        ({"k": 1}, "k must be at least 2, got 1"),
        ({"p": 4}, "p=4 labels per batch, but only 3 label"),
        ({"k": 7}, "p=2 labels per batch, but only 0 label"),
    ],
)
def test_sampler_rejects_bad_sizes(three_genres, kwargs, message) -> None:
    labels, artists = three_genres
    options = {"p": 2, "k": 3, **kwargs}
    with pytest.raises(ValueError, match=message):
        PKSampler(labels, artists, generator=_gen("errors"), **options)


@pytest.mark.parametrize(
    ("labels", "artists", "message"),
    [
        (np.array(["a", "b"]), np.array([1, 2, 3]), "aligned per item, got 2 and 3"),
        (np.array([], dtype=object), np.array([]), "labels must be a non-empty 1-D"),
        (np.array([["a"], ["b"]]), np.array([1, 2]), r"labels must be .* shape \(2, 1\)"),
        (np.array(["a", None], dtype=object), np.array([1, 2]), "labels contain missing"),
        (np.array(["a", "b"]), np.array([1.0, np.nan]), "artists contain missing"),
    ],
)
def test_sampler_rejects_bad_columns(labels, artists, message) -> None:
    with pytest.raises(ValueError, match=message):
        PKSampler(labels, artists, p=1, k=2, generator=_gen("columns"))


# --- contrastive_pairs -------------------------------------------------------
def test_pairs_by_hand() -> None:
    # Items 0, 1, 2 are genre a (artists 1, 1, 2); item 3 is genre b.
    # (0, 1) shares an artist: neither kind. Positives (0, 2), (1, 2);
    # negatives (0, 3), (1, 3), (2, 3), of which two are drawn.
    labels = np.array(["a", "a", "a", "b"])
    artists = np.array([1, 1, 2, 3])
    i, j, is_positive = contrastive_pairs(labels, artists, generator=_gen("hand"))
    assert is_positive.tolist() == [True, True, False, False]
    assert list(zip(i[:2], j[:2], strict=True)) == [(0, 2), (1, 2)]
    negatives = set(zip(i[2:].tolist(), j[2:].tolist(), strict=True))
    assert len(negatives) == 2
    assert negatives <= {(0, 3), (1, 3), (2, 3)}
    assert (i < j).all()


def test_pairs_draw_positives_down_when_negatives_are_fewer() -> None:
    # Four items of genre a by four artists: 6 positives; one item of b: 4 negatives.
    labels = np.array(["a", "a", "a", "a", "b"])
    artists = np.arange(5)
    i, j, is_positive = contrastive_pairs(labels, artists, generator=_gen("down"))
    assert is_positive.sum() == 4
    assert (~is_positive).sum() == 4
    assert set(zip(i[4:].tolist(), j[4:].tolist(), strict=True)) == {
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
    }


def test_pairs_on_a_pk_batch_are_balanced_and_artist_free(three_genres) -> None:
    labels, artists = three_genres
    sampler = PKSampler(labels, artists, p=3, k=4, generator=_gen("pk"))
    for batch in sampler:
        lab, art = labels[batch], artists[batch]
        i, j, is_positive = contrastive_pairs(lab, art, generator=_gen("pk-pairs"))
        assert is_positive.sum() == (~is_positive).sum()
        assert (lab[i[is_positive]] == lab[j[is_positive]]).all()
        assert (art[i[is_positive]] != art[j[is_positive]]).all()
        assert (lab[i[~is_positive]] != lab[j[~is_positive]]).all()
        # Every valid positive is kept: per block of 4 items by 2-3 artists,
        # C(4, 2) minus the same-artist pairs.
        expected = 0
        for block in art.reshape(3, 4):
            _, counts = np.unique(block, return_counts=True)
            expected += 6 - int((counts * (counts - 1) // 2).sum())
        assert is_positive.sum() == expected


def test_pairs_are_deterministic_per_stream(three_genres) -> None:
    labels, artists = three_genres
    first = contrastive_pairs(labels, artists, generator=_gen("det"))
    again = contrastive_pairs(labels, artists, generator=_gen("det"))
    assert all(np.array_equal(x, y) for x, y in zip(first, again, strict=True))


@pytest.mark.parametrize(
    ("labels", "artists", "message"),
    [
        (np.array(["a", "a"]), np.array([1, 2]), "1 positive.* and 0 negative"),
        (np.array(["a", "a", "b"]), np.array([1, 1, 2]), "0 positive.* and 2 negative"),
        (np.array(["a", "b"]), np.array([1]), "aligned per item, got 2 and 1"),
    ],
)
def test_pairs_reject_degenerate_batches(labels, artists, message) -> None:
    with pytest.raises(ValueError, match=message):
        contrastive_pairs(labels, artists, generator=_gen("degenerate"))


# --- VectorDataset -----------------------------------------------------------
def test_vector_dataset_serves_float32_rows() -> None:
    features = np.arange(12, dtype=np.float64).reshape(4, 3)
    dataset = VectorDataset(features)
    assert len(dataset) == 4
    rows = dataset[np.array([3, 0, 3])]
    assert rows.dtype == np.float32
    np.testing.assert_array_equal(rows, [[9, 10, 11], [0, 1, 2], [9, 10, 11]])


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (np.zeros(3), r"non-empty \(n_items, n_dims\) matrix, got shape \(3,\)"),
        (np.zeros((0, 3)), r"got shape \(0, 3\)"),
        (np.array([[1.0, np.nan]]), "NaN or infinite"),
        (np.array([[1.0, np.inf]]), "NaN or infinite"),
    ],
)
def test_vector_dataset_rejects_bad_features(features, message) -> None:
    with pytest.raises(ValueError, match=message):
        VectorDataset(features)


@pytest.mark.parametrize(
    ("positions", "message"),
    [
        (np.array([-1]), r"lie in \[0, 4\), got -1..-1"),
        (np.array([0, 4]), r"lie in \[0, 4\), got 0..4"),
        (np.array([0.0]), "1-D integer array"),
        (np.array([[0]]), "1-D integer array"),
    ],
)
def test_vector_dataset_rejects_bad_positions(positions, message) -> None:
    dataset = VectorDataset(np.zeros((4, 2)))
    with pytest.raises(ValueError, match=message):
        dataset[positions]


# --- MelCropDataset ----------------------------------------------------------
def _ramp(n_mels: int, n_frames: int) -> np.ndarray:
    """``S[m, t] = 1000 m + t``: a crop's first column tells where it starts."""
    return (1000 * np.arange(n_mels)[:, None] + np.arange(n_frames)[None, :]).astype(np.float16)


def test_crops_stay_inside_the_spectrogram() -> None:
    lengths = [5, 7, 12]  # the first one is exactly the crop length
    spectrograms = [_ramp(3, n) for n in lengths]
    dataset = MelCropDataset(
        spectrograms.__getitem__, len(spectrograms), crop_frames=5, generator=_gen("crop")
    )
    starts: dict[int, set[int]] = {0: set(), 1: set(), 2: set()}
    for _ in range(200):
        crops = dataset[np.array([0, 1, 2])]
        assert crops.shape == (3, 3, 5)
        assert crops.dtype == np.float32
        for position, crop in enumerate(crops):
            start = int(crop[0, 0])
            np.testing.assert_array_equal(crop, _ramp(3, lengths[position])[:, start : start + 5])
            starts[position].add(start)
    # Every start where the crop fits is drawn, and no other.
    assert starts == {0: {0}, 1: {0, 1, 2}, 2: set(range(8))}


def test_crops_are_deterministic_per_stream() -> None:
    spectrograms = [_ramp(2, 40) for _ in range(4)]

    def draw(stream: str) -> np.ndarray:
        dataset = MelCropDataset(spectrograms.__getitem__, 4, crop_frames=6, generator=_gen(stream))
        return np.stack([dataset[np.arange(4)] for _ in range(3)])

    np.testing.assert_array_equal(draw("same"), draw("same"))
    assert not np.array_equal(draw("same"), draw("other"))


@pytest.mark.parametrize(
    ("spectrograms", "message"),
    [
        ([_ramp(3, 4)], "position 0: 4 frames, shorter than a 5-frame crop"),
        ([np.zeros((1, 3, 10))], r"position 0: expected an \(n_mels, n_frames\)"),
        ([_ramp(3, 10), _ramp(4, 10)], "differ in n_mels"),
    ],
)
def test_crops_reject_bad_spectrograms(spectrograms, message) -> None:
    dataset = MelCropDataset(
        spectrograms.__getitem__, len(spectrograms), crop_frames=5, generator=_gen("bad")
    )
    with pytest.raises(ValueError, match=message):
        dataset[np.arange(len(spectrograms))]


@pytest.mark.parametrize(
    ("n_items", "crop_frames", "message"),
    [(0, 5, "n_items must be at least 1, got 0"), (1, 0, "crop_frames must be at least 1, got 0")],
)
def test_crop_dataset_rejects_bad_sizes(n_items, crop_frames, message) -> None:
    with pytest.raises(ValueError, match=message):
        MelCropDataset(lambda _: _ramp(1, 5), n_items, crop_frames=crop_frames, generator=_gen())


def test_crop_dataset_rejects_out_of_range_positions() -> None:
    dataset = MelCropDataset(lambda _: _ramp(1, 5), 2, crop_frames=5, generator=_gen())
    with pytest.raises(ValueError, match=r"lie in \[0, 2\), got 2..2"):
        dataset[np.array([2])]


# --- frames_for_seconds -------------------------------------------------------
@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(3.0, 130), (30.0, 1292)],  # 1 + 66150 // 512, 1 + 661500 // 512
)
def test_frames_for_seconds_matches_librosa_framing(seconds, expected) -> None:
    assert frames_for_seconds(seconds, sample_rate=22050, hop_length=512) == expected


@pytest.mark.parametrize(
    ("seconds", "sample_rate", "hop_length"), [(0.0, 22050, 512), (3.0, 0, 512), (3.0, 22050, 0)]
)
def test_frames_for_seconds_rejects_non_positive(seconds, sample_rate, hop_length) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        frames_for_seconds(seconds, sample_rate=sample_rate, hop_length=hop_length)
