"""Tests for the shared `Extractor.extract_all` driver: parallelism and failures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from musicsim.config import Config
from musicsim.extractors.base import Extractor


class _FlakyExtractor(Extractor):
    """Fails deterministically for a fixed set of ids, to exercise failure handling."""

    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        if item_id in {2, 4}:
            raise ValueError(f"boom on item {item_id}")
        return np.full(self.dim, float(item_id), dtype=np.float32)


class _WrongShapeExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        return np.zeros(self.dim + 1, dtype=np.float32)


class _NonFiniteExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        return np.full(self.dim, np.nan, dtype=np.float32)


def _config(dim: int = 3) -> Config:
    return Config(data={"extractor": {"name": "flaky", "dim": dim}})


def _index(n: int) -> pd.DataFrame:
    ids = list(range(1, n + 1))
    return pd.DataFrame({"item_id": ids, "path": [f"{i}.wav" for i in ids]})


def test_extract_all_skips_failures_without_aborting() -> None:
    extractor = _FlakyExtractor(_config())
    X, ids, failures = extractor.extract_all(_index(5), n_jobs=1, progress=False)
    assert list(ids) == [1, 3, 5]
    assert X.shape == (3, 3)
    assert {item_id for item_id, _ in failures} == {2, 4}


def test_extract_all_rows_stay_aligned_with_ids() -> None:
    extractor = _FlakyExtractor(_config())
    X, ids, _ = extractor.extract_all(_index(5), n_jobs=1, progress=False)
    for row, item_id in zip(X, ids, strict=True):
        assert np.all(row == float(item_id))


def test_extract_all_on_an_empty_index_returns_empty_arrays() -> None:
    extractor = _FlakyExtractor(_config())
    X, ids, failures = extractor.extract_all(_index(0), n_jobs=1, progress=False)
    assert X.shape == (0, 3)
    assert ids.shape == (0,)
    assert failures == []


def test_wrong_shape_vector_is_recorded_as_a_failure() -> None:
    extractor = _WrongShapeExtractor(_config())
    _X, ids, failures = extractor.extract_all(_index(2), n_jobs=1, progress=False)
    assert len(ids) == 0
    assert len(failures) == 2
    assert "shape" in failures[0][1]


def test_non_finite_vector_is_recorded_as_a_failure() -> None:
    extractor = _NonFiniteExtractor(_config())
    _X, ids, failures = extractor.extract_all(_index(1), n_jobs=1, progress=False)
    assert len(ids) == 0
    assert len(failures) == 1
    assert "NaN" in failures[0][1] or "inf" in failures[0][1]


def test_extract_all_runs_with_multiple_jobs_too() -> None:
    extractor = _FlakyExtractor(_config())
    _X, ids, failures = extractor.extract_all(_index(6), n_jobs=2, progress=False)
    assert list(ids) == [1, 3, 5, 6]
    assert len(failures) == 2
