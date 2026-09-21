"""Relevance functions: which gallery items count as relevant to a query.

Importing this package registers every relevance function (``label_match``)
in :data:`musicsim.registry.RELEVANCES`.
"""

from __future__ import annotations

from musicsim.relevance import label_match  # noqa: F401  (registers "label_match")
from musicsim.relevance.base import RelevanceFunction, label_codes

__all__ = ["RelevanceFunction", "label_codes"]
