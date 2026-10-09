"""Galleries: which items take part in an evaluation, queries and candidates alike.

A gallery is a named subset of the item index. It lives in its own module so
that every evaluator (retrieval, graph structure, similarity measures, ...)
restricts its items the same way, instead of each one re-deriving the rule from
the split column.

``fma_test``
    the test split only: artists no model has been fitted on. The main
    evaluation gallery.
``fma_validation``
    the validation split only. Every hyper-parameter selection is made here and
    never on ``fma_test``, so the test figures stay untouched by tuning.
``fma_all``
    every item, no restriction. A secondary analysis and a regression check
    against the archived pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["GALLERIES", "gallery_mask"]

#: Valid gallery names (see the module docstring).
GALLERIES: tuple[str, ...] = ("fma_test", "fma_validation", "fma_all")

_SPLIT_OF: dict[str, str] = {"fma_test": "test", "fma_validation": "validation"}


def gallery_mask(index_common: pd.DataFrame, gallery: str) -> np.ndarray:
    """Boolean mask, row-aligned with ``index_common``, selecting ``gallery``.

    Evaluators apply this mask to every representation's embeddings before
    ranking, computing relevance, detecting near-duplicates or filtering by
    artist, so nothing outside the gallery can be a neighbour, relevant, or the
    reason a query gets excluded.
    """
    if gallery == "fma_all":
        return np.ones(len(index_common), dtype=bool)
    if gallery in _SPLIT_OF:
        return index_common["split"].to_numpy() == _SPLIT_OF[gallery]
    raise ValueError(f"gallery must be one of {GALLERIES}, got {gallery!r}")
