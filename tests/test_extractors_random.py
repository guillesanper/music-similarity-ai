"""Tests for the `random` extractor: the negative control must be reproducible."""

from __future__ import annotations

import numpy as np
import pytest

from musicsim.config import Config
from musicsim.extractors.random import RandomExtractor


def _config(seed: int = 42, dim: int = 80) -> Config:
    return Config(
        data={
            "seed": seed,
            "extractor": {
                "name": "random",
                "dim": dim,
                "distribution": "normal",
                "seed_stream": "extractor/random",
            },
        }
    )


def test_random_extractor_is_deterministic_for_the_same_seed_and_item() -> None:
    extractor = RandomExtractor(_config())
    v1 = extractor.extract_one(123, "unused.mp3")
    v2 = extractor.extract_one(123, "unused.mp3")
    assert np.array_equal(v1, v2)


def test_random_extractor_differs_across_items() -> None:
    extractor = RandomExtractor(_config())
    assert not np.array_equal(extractor.extract_one(1, "a"), extractor.extract_one(2, "a"))


def test_random_extractor_differs_across_seeds() -> None:
    v1 = RandomExtractor(_config(seed=1)).extract_one(1, "a")
    v2 = RandomExtractor(_config(seed=2)).extract_one(1, "a")
    assert not np.array_equal(v1, v2)


def test_random_extractor_shape_and_dtype() -> None:
    vector = RandomExtractor(_config(dim=80)).extract_one(1, "a")
    assert vector.shape == (80,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_random_extractor_rejects_an_unknown_distribution() -> None:
    config = Config(
        data={
            "seed": 42,
            "extractor": {"name": "random", "dim": 4, "distribution": "uniform"},
        }
    )
    with pytest.raises(ValueError, match="distribution"):
        RandomExtractor(config).extract_one(1, "a")
