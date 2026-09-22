"""Evaluators: score a run's representation(s) and write metrics under it.

Importing this package registers every evaluator (``retrieval``) in
:data:`musicsim.registry.EVALUATORS`.
"""

from __future__ import annotations

from musicsim.evaluation import retrieval  # noqa: F401  (registers "retrieval")
from musicsim.evaluation.base import Evaluator

__all__ = ["Evaluator"]
