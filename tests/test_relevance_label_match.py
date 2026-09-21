"""Tests for :mod:`musicsim.relevance.label_match` and :func:`label_codes`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from musicsim.config import Config
from musicsim.relevance.base import label_codes
from musicsim.relevance.label_match import LabelMatchRelevance


def test_label_codes_gives_the_same_code_to_equal_values() -> None:
    codes = label_codes(pd.Series(["rock", "pop", "rock", "pop"]))
    assert codes[0] == codes[2]
    assert codes[1] == codes[3]
    assert codes[0] != codes[1]


def test_label_codes_never_matches_two_missing_values() -> None:
    codes = label_codes(pd.Series(["rock", np.nan, np.nan, "rock"]))
    assert codes[0] == codes[3]
    assert codes[1] != codes[2]  # two NaNs must not count as "the same label"
    assert codes[1] != codes[0]


def test_label_match_relevance_reads_genre_top_by_default() -> None:
    index = pd.DataFrame(
        {
            "item_id": [1, 2, 3],
            "genre_top": ["Rock", "Pop", "Rock"],
        }
    )
    relevance = LabelMatchRelevance(Config(data={}))
    codes = relevance.label_codes(index, np.array([3, 1, 2]))
    assert codes[0] == codes[1]  # item 3 and item 1 are both Rock
    assert codes[2] != codes[0]


def test_label_match_relevance_can_use_another_column() -> None:
    index = pd.DataFrame({"item_id": [1, 2], "mood": ["happy", "sad"]})
    relevance = LabelMatchRelevance(Config(data={"relevance": {"column": "mood"}}))
    codes = relevance.label_codes(index, np.array([1, 2]))
    assert codes[0] != codes[1]


def test_label_match_relevance_rejects_an_unknown_column() -> None:
    index = pd.DataFrame({"item_id": [1], "genre_top": ["Rock"]})
    relevance = LabelMatchRelevance(Config(data={"relevance": {"column": "nope"}}))
    with pytest.raises(ValueError, match="nope"):
        relevance.label_codes(index, np.array([1]))
