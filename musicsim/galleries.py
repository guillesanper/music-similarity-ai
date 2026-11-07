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
``mtt_all``
    every MagnaTagATune clip, for analyses of the whole external corpus.
``mtt_triplets``
    the MagnaTagATune clips flagged ``in_triplets``: those with at least one
    similarity constraint derived from the human odd-one-out votes, the only
    clips the triplet agreement is measured on.

The two MagnaTagATune galleries require the boolean ``in_triplets`` column the
MagnaTagATune index carries, even ``mtt_all``, which does not read it: asking
for an MTT gallery on another corpus's index is a configuration mistake, and it
is better reported than silently answered with every item.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["GALLERIES", "gallery_mask"]

#: Valid gallery names (see the module docstring).
GALLERIES: tuple[str, ...] = (
    "fma_test",
    "fma_validation",
    "fma_all",
    "mtt_all",
    "mtt_triplets",
)

_SPLIT_OF: dict[str, str] = {"fma_test": "test", "fma_validation": "validation"}

#: Index column that marks the MagnaTagATune clips with a triplet constraint.
_IN_TRIPLETS = "in_triplets"


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
    if gallery == "mtt_all":
        _in_triplets(index_common, gallery)  # only to check this is an MTT index
        return np.ones(len(index_common), dtype=bool)
    if gallery == "mtt_triplets":
        return _in_triplets(index_common, gallery)
    raise ValueError(f"gallery must be one of {GALLERIES}, got {gallery!r}")


def _in_triplets(index_common: pd.DataFrame, gallery: str) -> np.ndarray:
    """The boolean ``in_triplets`` column as an array; a ``ValueError`` if there is none.

    A non-boolean column (strings, or a column with missing values) is refused
    rather than coerced: ``"False"`` would otherwise count as true.
    """
    column = index_common.get(_IN_TRIPLETS)
    if column is None or not pd.api.types.is_bool_dtype(column):
        found = "missing" if column is None else f"of dtype {column.dtype}"
        raise ValueError(
            f"gallery {gallery!r} needs a boolean {_IN_TRIPLETS!r} column in the index "
            f"(the MagnaTagATune index has one); the column is {found}"
        )
    return column.to_numpy(dtype=bool)
