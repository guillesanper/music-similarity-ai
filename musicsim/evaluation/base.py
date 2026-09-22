"""The evaluator contract: read a run's cached artefacts, write its metrics.

An evaluator is registered by name in :data:`musicsim.registry.EVALUATORS`,
exactly like an extractor or a dataset. Each one owns a subsection of
``config["evaluation"]`` (e.g. ``evaluation.retrieval``) that says whether it
runs and with which parameters, and writes its CSVs under
``run.metrics_dir`` — never anywhere else, so a run directory is always the
complete record of what a report number came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from musicsim.config import Config
from musicsim.runlog import RunLog

__all__ = ["Evaluator"]


class Evaluator(ABC):
    """One evaluation protocol: how to score a run's representation(s)."""

    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def evaluate(self, index: pd.DataFrame, run: RunLog) -> None:
        """Compute metrics for ``config`` and write them under ``run.metrics_dir``.

        ``index`` is the dataset's item index. Implementations load whichever
        cached embeddings they need through
        :func:`musicsim.embeddings.load_embeddings`, never by rebuilding a
        cache path by hand.
        """
