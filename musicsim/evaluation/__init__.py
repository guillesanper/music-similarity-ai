"""Evaluators: score a run's representation(s) and write metrics under it.

Importing this package registers every evaluator (``retrieval``, ``structure``) in
:data:`musicsim.registry.EVALUATORS`. The pure metric functions live in
:mod:`musicsim.evaluation.hubness` and :mod:`musicsim.evaluation.structure`.
"""

from __future__ import annotations

from musicsim.evaluation import (
    graph_evaluator,  # noqa: F401  (registers "structure")
    retrieval,  # noqa: F401  (registers "retrieval")
)
from musicsim.evaluation.base import Evaluator

__all__ = ["Evaluator"]
