"""Raw features -> embeddings: standardize -> optional PCA -> L2 normalize.

The scaler and the PCA are fit on the ``embedding.fit_on`` split only (training
by default) and then applied to every item, so normalisation statistics never
see validation/test — the same rule later experiments that train models on
``training`` (siamese, CNN, ...) need anyway.

With every row L2-normalised, ``cos(a, b) == a @ b``, so the similarity graph
is a plain matrix product.

Cache keys are deliberately narrow. Features depend only on the ``dataset``,
``audio``, ``spectrogram``, ``extractor`` and ``seed`` sections (the only ones
extractors read), embeddings additionally on ``embedding``. Keying on the whole
configuration would re-extract every item whenever an unrelated key (a graph
parameter, an evaluator option, a new default in ``base.yaml``) changes.
``runtime`` is left out on purpose: ``n_jobs``, ``progress`` and ``cache`` do
not change any value. ``seed`` always enters, even though only some extractors
use it, because that is simpler and never reuses a cache wrongly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from musicsim import paths, seeding
from musicsim.config import Config, ConfigError, config_hash

__all__ = [
    "EmbeddingError",
    "build_embeddings",
    "embeddings_cache_dir",
    "features_cache_dir",
    "load_embeddings",
]

#: Configuration sections that determine the raw features.
_FEATURE_SECTIONS: tuple[str, ...] = ("dataset", "audio", "spectrogram", "extractor", "seed")

#: Tolerance on |norm - 1| after L2-normalising and casting to float32.
_NORM_TOL = 1e-4


class EmbeddingError(ValueError):
    """The feature matrix and the item index are inconsistent, or a step failed."""


def features_cache_dir(config: Config, *, create: bool = False) -> Path:
    """Directory of the cached raw features of ``config``'s dataset + extractor.

    The single place that decides where features live; every stage and test
    reaches the cache through here instead of rebuilding the path by hand.
    """
    dataset = config.require("dataset.directory")
    extractor_name = config.require("extractor.name")
    key = _section_hash(config, _FEATURE_SECTIONS)
    return paths.cache_dir(dataset, f"features-{extractor_name}", key, create=create)


def embeddings_cache_dir(config: Config, *, create: bool = False) -> Path:
    """Directory of the cached embeddings of ``config``'s dataset + extractor."""
    dataset = config.require("dataset.directory")
    extractor_name = config.require("extractor.name")
    key = _section_hash(config, (*_FEATURE_SECTIONS, "embedding"))
    return paths.cache_dir(dataset, f"embeddings-{extractor_name}", key, create=create)


def _section_hash(config: Config, sections: tuple[str, ...]) -> str:
    """Hash of the named top-level sections as resolved; an absent one counts as ``{}``."""
    resolved = config.to_dict()
    return config_hash({name: resolved.get(name, {}) for name in sections})


def build_embeddings(
    X: np.ndarray, ids: np.ndarray, index: pd.DataFrame, config: Config
) -> tuple[np.ndarray, dict]:
    """Return ``(Z, info)``, ``Z`` aligned row-for-row with ``X`` / ``ids``.

    ``info`` carries ``n_fit`` (items the scaler/PCA were fit on) and
    ``explained_variance_ratio`` (``None`` unless PCA is enabled).
    """
    if len(X) != len(ids):
        raise EmbeddingError(f"rows ({len(X)}) and ids ({len(ids)}) are misaligned")
    if pd.Index(ids).has_duplicates:
        raise EmbeddingError("duplicate ids in the feature matrix")

    fit_split = config.get("embedding.fit_on", "training")
    split_by_id = index.set_index("item_id")["split"]
    missing = pd.Index(ids).difference(split_by_id.index)
    if len(missing):
        raise EmbeddingError(
            f"{len(missing)} id(s) not present in the index, e.g. {list(missing[:5])}"
        )

    fit_mask = split_by_id.loc[ids].to_numpy() == fit_split
    if not fit_mask.any():
        raise EmbeddingError(f"no item belongs to the '{fit_split}' split; nothing to fit on")

    Z = X.astype(np.float64)
    info: dict = {"n_fit": int(fit_mask.sum()), "explained_variance_ratio": None}

    if config.get("embedding.standardize", True):
        scaler = StandardScaler().fit(Z[fit_mask])
        Z = scaler.transform(Z)

    if config.get("embedding.pca.enabled", False):
        n_components = config.get("embedding.pca.n_components")
        if not n_components:
            raise EmbeddingError(
                "embedding.pca.enabled is true but embedding.pca.n_components is not set"
            )
        if not 0 < n_components <= Z.shape[1]:
            raise EmbeddingError(
                f"embedding.pca.n_components must be in [1, {Z.shape[1]}], got {n_components}"
            )
        random_state = seeding.derive_seed(config.require("seed"), "embedding", "pca")
        pca = PCA(n_components=n_components, random_state=random_state).fit(Z[fit_mask])
        Z = pca.transform(Z)
        info["explained_variance_ratio"] = pca.explained_variance_ratio_

    if config.get("embedding.l2_normalize", True):
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        if (norms == 0).any():
            raise EmbeddingError(
                f"{int((norms == 0).sum())} row(s) have zero norm; cannot L2-normalise"
            )
        Z = Z / norms

    Z = Z.astype(np.float32)
    if config.get("embedding.l2_normalize", True):
        final_norms = np.linalg.norm(Z.astype(np.float64), axis=1)
        max_err = float(np.abs(final_norms - 1).max())
        if max_err > _NORM_TOL or not np.isfinite(Z).all():
            raise EmbeddingError(
                f"L2 norms out of tolerance ({_NORM_TOL}) after casting to float32 "
                f"(max |norm - 1| = {max_err:.2e}), or non-finite values"
            )

    return Z, info


def load_embeddings(config: Config) -> tuple[np.ndarray, np.ndarray]:
    """``(X, ids)`` of the cached embeddings ``config``'s dataset + extractor produced.

    This is the ``load_embeddings`` half of the B.1.8 interface contract: any
    stage past ``embed`` reaches the cache only through this function, never
    by rebuilding ``paths.cache_dir(...)`` by hand. Raises :class:`ConfigError`
    (a usage error, not a crash) when the embed stage has not run yet.
    """
    extractor_name = config.require("extractor.name")
    embed_dir = embeddings_cache_dir(config)
    embeddings_path = embed_dir / "embeddings.npy"
    ids_path = embed_dir / "ids.npy"
    if not embeddings_path.is_file() or not ids_path.is_file():
        raise ConfigError(
            f"no cached embeddings for extractor {extractor_name!r} at "
            f"{paths.relative_to_repo(embed_dir)}; run the embed stage first"
        )
    return np.load(embeddings_path), np.load(ids_path)
