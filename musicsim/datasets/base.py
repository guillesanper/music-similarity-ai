"""The download / build-index contract every corpus implements.

A dataset is registered by name (its ``dataset.directory`` value, e.g.
``"fma"``) in :data:`musicsim.registry.DATASETS`, exactly like an extractor or
a graph builder: adding a corpus means adding one class here and one YAML file
under ``configs/datasets/``, never editing the CLI.

Every corpus produces the same index shape (:data:`INDEX_COLUMNS`) so that
downstream code (feature extraction, the graph, the evaluation) never branches
on which dataset it is looking at.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from musicsim import paths
from musicsim.config import Config

__all__ = ["INDEX_COLUMNS", "Dataset", "DatasetError"]

#: Every dataset index has exactly these columns, in this order.
#:
#: ``group_id`` identifies near-duplicate items (a query and its almost-exact
#: partner should count as one item for retrieval purposes). At index-build
#: time no such grouping is known yet — it is only recoverable from the
#: embeddings, which :mod:`musicsim.graphs.duplicates` computes — so every
#: dataset's :meth:`Dataset.build_index` sets ``group_id`` equal to
#: ``item_id`` (each item its own group) as a placeholder. Code that already
#: has embeddings must not read this column as ground truth for duplicates.
INDEX_COLUMNS: tuple[str, ...] = (
    "item_id",
    "path",
    "split",
    "artist_id",
    "group_id",
    "genre_top",
    "genres_all",
)


class DatasetError(ValueError):
    """Downloading a corpus failed, or its built index fails a consistency check."""


class Dataset(ABC):
    """One corpus: how to fetch its files and how to build its item index."""

    def __init__(self, config: Config) -> None:
        self.config = config

    # -- identity ----------------------------------------------------------
    @property
    def name(self) -> str:
        """The dataset's configuration name, e.g. ``fma_small``."""
        return self.config.require("dataset.name")

    @property
    def directory_name(self) -> str:
        """Name of the corpus directory under the data root, e.g. ``fma``."""
        return self.config.require("dataset.directory")

    @property
    def raw_dir(self) -> Path:
        """Where the downloaded, never-versioned files of this corpus live."""
        return paths.raw_dir(self.directory_name)

    @property
    def index_path(self) -> Path:
        """Where :meth:`write_index` writes the item index."""
        return paths.dataset_index(self.directory_name)

    # -- the two operations every corpus implements -------------------------
    @abstractmethod
    def download(self, *, force: bool = False) -> None:
        """Fetch, verify and unpack the corpus's files under :attr:`raw_dir`.

        ``force`` re-downloads and re-extracts even when a verified copy is
        already present.
        """

    @abstractmethod
    def build_index(self) -> pd.DataFrame:
        """Return the item index (see :data:`INDEX_COLUMNS`), validated.

        Implementations must raise :class:`DatasetError` themselves for any
        corpus-specific consistency check (expected counts, disjoint splits,
        files present on disk, ...) before returning: a silently wrong index
        would make every later number wrong too.
        """

    def write_index(self) -> Path:
        """Build the index and write it to :attr:`index_path`; return that path."""
        frame = self.build_index()
        missing = [column for column in INDEX_COLUMNS if column not in frame.columns]
        if missing:
            raise DatasetError(f"{self.name}: build_index() is missing columns {missing}")
        frame = frame[list(INDEX_COLUMNS)]
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.index_path, index=False)
        return self.index_path
