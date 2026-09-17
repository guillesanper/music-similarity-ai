"""Raw features -> embeddings: standardize -> optional PCA -> L2 normalize.

The scaler and the PCA are fit on the ``embedding.fit_on`` split only (training
by default) and then applied to every item, so normalisation statistics never
see validation/test — the same rule later experiments that train models on
``training`` (siamese, CNN, ...) need anyway.

With every row L2-normalised, ``cos(a, b) == a @ b``, so the similarity graph
(phase P3) is a plain matrix product.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from musicsim import seeding
from musicsim.config import Config

__all__ = ["EmbeddingError", "build_embeddings"]

#: Tolerance on |norm - 1| after L2-normalising and casting to float32.
_NORM_TOL = 1e-4


class EmbeddingError(ValueError):
    """The feature matrix and the item index are inconsistent, or a step failed."""


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
