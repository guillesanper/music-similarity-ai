"""Binary relevance from a shared label: two items are relevant to each other
exactly when they carry the same value of one index column (``genre_top`` by
default). This is the family-A proxy of C.10bis: a genre match, not a
judgement of musical similarity, and every table that uses it must say so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from musicsim.registry import register_relevance
from musicsim.relevance.base import RelevanceFunction
from musicsim.relevance.base import label_codes as _label_codes

__all__ = ["LabelMatchRelevance"]


@register_relevance("label_match")
class LabelMatchRelevance(RelevanceFunction):
    """Relevant iff ``index[column]`` matches; ``column`` defaults to ``genre_top``."""

    def label_codes(self, index: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
        column = self.config.get("relevance.column", "genre_top")
        if column not in index.columns:
            raise ValueError(f"relevance.column {column!r} is not a column of the item index")
        values = index.set_index("item_id").loc[np.asarray(ids), column]
        return _label_codes(values)
