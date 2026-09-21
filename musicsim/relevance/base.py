"""The relevance contract: which candidates count as relevant to a query.

A relevance function is registered by name (``retrieval.relevance`` in the
configuration) in :data:`musicsim.registry.RELEVANCES`, exactly like an
extractor. So far there is one implementation
(:mod:`musicsim.relevance.label_match`, binary relevance from a shared label);
graded relevance (``hierarchy_graded``, ``tag_facet_jaccard``) is a future
addition under the same contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from musicsim.config import Config

__all__ = ["RelevanceFunction", "label_codes"]


def label_codes(values: pd.Series | np.ndarray) -> np.ndarray:
    """Integer code per value, each missing value getting its own code.

    Two candidates are "the same label" exactly when their code matches, so a
    missing label (``NaN``) must not accidentally match another missing one:
    :func:`pandas.factorize` groups every ``NaN`` together, which this
    corrects by handing each one a fresh, otherwise-unused code.
    """
    codes, _ = pd.factorize(pd.Series(values).to_numpy(), use_na_sentinel=True)
    missing = codes < 0
    if missing.any():
        codes = codes.copy()
        codes[missing] = codes.max(initial=-1) + 1 + np.arange(missing.sum())
    return codes.astype(np.int64)


class RelevanceFunction(ABC):
    """One way of deciding which gallery items are relevant to which query."""

    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def label_codes(self, index: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
        """Integer label code for each of ``ids`` (row-aligned); equal code = relevant.

        ``index`` is the dataset's item index (``item_id`` among its columns);
        ``ids`` need not be in the same order or the same set as
        ``index["item_id"]``, so implementations look items up by id.
        """
