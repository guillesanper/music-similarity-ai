"""The extractor contract: an item (its id and audio path) -> one feature vector.

Adding an extractor is writing one class here, decorating it with
``@register_extractor("name")``, and adding a YAML file under
``configs/extractors/``, following the same pattern as
:mod:`musicsim.datasets.base`. :meth:`Extractor.extract_all` is the shared
driver every extractor gets for free: it parallelises over items with
``joblib``, and turns a per-item exception into a recorded failure rather than
aborting the whole run — one corrupt file must not lose 7993 good ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from musicsim.config import Config

__all__ = ["Extractor"]


class Extractor(ABC):
    """One feature extractor: how to turn one item into a fixed-width vector."""

    def __init__(self, config: Config) -> None:
        self.config = config

    @property
    def name(self) -> str:
        """The extractor's configuration name, e.g. ``mfcc``."""
        return self.config.require("extractor.name")

    @property
    def dim(self) -> int:
        """Width of the feature vector this extractor produces."""
        return int(self.config.require("extractor.dim"))

    @abstractmethod
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        """Return the ``(dim,)`` feature vector for one item.

        Implementations raise on failure (a bad file, an unexpected shape)
        rather than returning a sentinel; :meth:`extract_all` is what turns
        that exception into a recorded, non-fatal failure.
        """

    def extract_all(
        self,
        index: pd.DataFrame,
        *,
        n_jobs: int = -1,
        progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
        """Extract every item of ``index`` (columns ``item_id``, ``path``).

        Returns ``(X, ids, failures)``: ``X`` is ``(N, dim)`` float32 and
        ``ids[i]`` is the item of row ``X[i]`` — the pipeline's usual
        row-alignment invariant. ``failures`` is a list of ``(item_id, reason)``
        for items that were skipped.
        """
        tasks = (
            delayed(self._safe_extract_one)(int(item_id), str(path))
            for item_id, path in zip(index["item_id"], index["path"], strict=True)
        )
        # return_as="generator" preserves input order, so results line up with
        # the index without having to sort them back afterwards.
        results = Parallel(n_jobs=n_jobs, return_as="generator")(tasks)
        if progress:
            results = tqdm(results, total=len(index), desc=self.name, unit="item")

        vectors: list[np.ndarray] = []
        ids: list[int] = []
        failures: list[tuple[int, str]] = []
        for item_id, vector, error in results:
            if error is not None:
                failures.append((item_id, error))
                continue
            vectors.append(vector)
            ids.append(item_id)

        if vectors:
            X = np.stack(vectors).astype(np.float32, copy=False)
        else:
            X = np.empty((0, self.dim), np.float32)
        return X, np.asarray(ids, dtype=np.int64), failures

    def _safe_extract_one(
        self, item_id: int, path: str
    ) -> tuple[int, np.ndarray | None, str | None]:
        """Worker: never raises, returns the failure reason as text instead."""
        try:
            vector = np.asarray(self.extract_one(item_id, path), dtype=np.float32)
        except Exception as exc:
            return item_id, None, f"{type(exc).__name__}: {exc}"
        if vector.shape != (self.dim,):
            return item_id, None, f"expected shape ({self.dim},), got {vector.shape}"
        if not np.isfinite(vector).all():
            return item_id, None, "vector contains NaN or inf"
        return item_id, vector, None
