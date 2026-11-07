"""Tests for :mod:`musicsim.galleries`: the FMA and MagnaTagATune gallery masks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from musicsim.evaluation import retrieval
from musicsim.galleries import GALLERIES, gallery_mask


def test_masks_select_their_split(synthetic_index: pd.DataFrame) -> None:
    split = synthetic_index["split"].to_numpy()
    assert np.array_equal(gallery_mask(synthetic_index, "fma_test"), split == "test")
    assert np.array_equal(gallery_mask(synthetic_index, "fma_validation"), split == "validation")
    assert gallery_mask(synthetic_index, "fma_all").all()


def test_split_masks_are_disjoint_and_all_is_their_union(synthetic_index: pd.DataFrame) -> None:
    test = gallery_mask(synthetic_index, "fma_test")
    validation = gallery_mask(synthetic_index, "fma_validation")
    training = synthetic_index["split"].to_numpy() == "training"
    assert test.any() and validation.any()
    assert not (test & validation).any()
    assert np.array_equal(test | validation | training, gallery_mask(synthetic_index, "fma_all"))


def test_unknown_gallery_names_the_valid_ones(synthetic_index: pd.DataFrame) -> None:
    with pytest.raises(ValueError) as excinfo:
        gallery_mask(synthetic_index, "fma_train")
    message = str(excinfo.value)
    assert "fma_train" in message
    assert all(name in message for name in GALLERIES)


def test_retrieval_still_exports_the_gallery_names() -> None:
    assert retrieval.GALLERIES is GALLERIES
    assert retrieval.gallery_mask is gallery_mask


def _mtt_index(in_triplets: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [10, 11, 12, 13],
            "split": ["training", "training", "validation", "test"],
            "in_triplets": in_triplets,
        }
    )


def test_mtt_galleries_select_every_clip_and_the_triplet_clips() -> None:
    index = _mtt_index([True, False, False, True])
    assert gallery_mask(index, "mtt_all").tolist() == [True, True, True, True]
    assert gallery_mask(index, "mtt_triplets").tolist() == [True, False, False, True]


@pytest.mark.parametrize("gallery", ["mtt_all", "mtt_triplets"])
def test_mtt_galleries_without_in_triplets_name_the_column(
    synthetic_index: pd.DataFrame, gallery: str
) -> None:
    with pytest.raises(ValueError, match=r"in_triplets.*missing"):
        gallery_mask(synthetic_index, gallery)


def test_mtt_triplets_refuses_a_non_boolean_column() -> None:
    # As strings, "False" would count as true if the column were coerced.
    index = _mtt_index(["True", "False", "False", "True"])
    with pytest.raises(ValueError, match=r"in_triplets.*of dtype"):
        gallery_mask(index, "mtt_triplets")
