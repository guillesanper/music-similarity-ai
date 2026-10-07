"""Tests for musicsim.embeddings: standardize (training only) -> PCA -> L2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from musicsim.config import Config
from musicsim.embeddings import (
    EmbeddingError,
    build_embeddings,
    embeddings_cache_dir,
    features_cache_dir,
)


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


def _full_config(**overrides: object) -> Config:
    data: dict = {
        "experiment": "a",
        "stages": ["extract"],
        "seed": 42,
        "dataset": {"directory": "toy"},
        "audio": {"sample_rate": 22050},
        "spectrogram": {"n_mels": 128},
        "extractor": {"name": "mfcc", "n_mfcc": 20},
        "embedding": {"standardize": True},
        "graph": {"k": 10},
        "evaluation": {"retrieval": {"enabled": True}},
        "bootstrap": {"n_resamples": 100},
        "runtime": {"n_jobs": 1, "cache": True},
        "outputs": {"figures": True},
    }
    data.update(overrides)
    return Config(data=data)


def test_unrelated_sections_do_not_change_either_cache_directory() -> None:
    base = _full_config()
    other = _full_config(
        experiment="b",
        stages=["evaluate"],
        graph={"k": 20},
        evaluation={"structure": {"enabled": True}},
        bootstrap={"n_resamples": 5},
        runtime={"n_jobs": 8, "cache": False},
        outputs={"figures": False},
    )
    assert features_cache_dir(base) == features_cache_dir(other)
    assert embeddings_cache_dir(base) == embeddings_cache_dir(other)


@pytest.mark.parametrize(
    "override",
    [
        {"dataset": {"directory": "toy2"}},
        {"audio": {"sample_rate": 16000}},
        {"spectrogram": {"n_mels": 64}},
        {"extractor": {"name": "mfcc", "n_mfcc": 13}},
        {"seed": 7},
    ],
)
def test_feature_relevant_sections_change_the_features_directory(override: dict) -> None:
    assert features_cache_dir(_full_config()) != features_cache_dir(_full_config(**override))


def test_the_embedding_section_changes_only_the_embeddings_directory() -> None:
    base = _full_config()
    other = _full_config(embedding={"standardize": False})
    assert features_cache_dir(base) == features_cache_dir(other)
    assert embeddings_cache_dir(base) != embeddings_cache_dir(other)
