"""Tests for musicsim.embeddings: standardize (training only) -> PCA -> L2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from musicsim.config import Config
from musicsim.embeddings import EmbeddingError, build_embeddings


def _index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [1, 2, 3, 4, 5, 6],
            "split": ["training", "training", "training", "training", "validation", "test"],
        }
    )


def _features() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(loc=5.0, scale=2.0, size=(6, 4)).astype(np.float32)


def _ids() -> np.ndarray:
    return np.array([1, 2, 3, 4, 5, 6])


def _config(**embedding_overrides: object) -> Config:
    data: dict = {
        "seed": 42,
        "embedding": {
            "fit_on": "training",
            "standardize": True,
            "l2_normalize": True,
            "pca": {"enabled": False, "n_components": None},
        },
    }
    data["embedding"].update(embedding_overrides)
    return Config(data=data)


def test_build_embeddings_l2_normalizes_every_row() -> None:
    Z, info = build_embeddings(_features(), _ids(), _index(), _config())
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-4)
    assert info["n_fit"] == 4


def test_standardize_uses_only_the_training_split_statistics() -> None:
    X = _features()
    scaler = StandardScaler().fit(X[:4].astype(np.float64))
    expected = scaler.transform(X.astype(np.float64))
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)

    Z, _ = build_embeddings(X, _ids(), _index(), _config())
    np.testing.assert_allclose(Z, expected.astype(np.float32), rtol=1e-4, atol=1e-4)


def test_pca_reduces_dimensionality_and_reports_explained_variance() -> None:
    config = _config(pca={"enabled": True, "n_components": 2})
    Z, info = build_embeddings(_features(), _ids(), _index(), config)
    assert Z.shape == (6, 2)
    assert info["explained_variance_ratio"] is not None
    assert len(info["explained_variance_ratio"]) == 2


def test_pca_without_n_components_raises() -> None:
    config = _config(pca={"enabled": True, "n_components": None})
    with pytest.raises(EmbeddingError, match="n_components"):
        build_embeddings(_features(), _ids(), _index(), config)


def test_pca_with_too_many_components_raises() -> None:
    config = _config(pca={"enabled": True, "n_components": 99})
    with pytest.raises(EmbeddingError):
        build_embeddings(_features(), _ids(), _index(), config)


def test_mismatched_rows_and_ids_raises() -> None:
    with pytest.raises(EmbeddingError):
        build_embeddings(_features(), np.array([1, 2, 3]), _index(), _config())


def test_duplicate_ids_raises() -> None:
    with pytest.raises(EmbeddingError):
        build_embeddings(_features(), np.array([1, 1, 3, 4, 5, 6]), _index(), _config())


def test_no_item_in_the_fit_split_raises() -> None:
    index = _index().copy()
    index["split"] = "test"
    with pytest.raises(EmbeddingError):
        build_embeddings(_features(), _ids(), index, _config())


def test_ids_missing_from_the_index_raises() -> None:
    with pytest.raises(EmbeddingError):
        build_embeddings(_features(), np.array([1, 2, 3, 4, 5, 999]), _index(), _config())


def test_standardize_and_pca_can_both_be_disabled() -> None:
    config = _config(standardize=False, l2_normalize=False)
    Z, _ = build_embeddings(_features(), _ids(), _index(), config)
    np.testing.assert_allclose(Z, _features(), rtol=1e-5, atol=1e-5)
