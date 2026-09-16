"""Tests for :mod:`musicsim.datasets.base`.

Exercises the shared ``write_index`` contract through a minimal concrete
:class:`Dataset`, independent of any real corpus.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from musicsim import paths
from musicsim.config import Config
from musicsim.datasets.base import INDEX_COLUMNS, Dataset, DatasetError


class _ToyDataset(Dataset):
    def __init__(self, config: Config, frame: pd.DataFrame) -> None:
        super().__init__(config)
        self._frame = frame

    def download(self, *, force: bool = False) -> None:
        raise NotImplementedError

    def build_index(self) -> pd.DataFrame:
        return self._frame


def _config(directory: str = "toy") -> Config:
    return Config(data={"dataset": {"name": directory, "directory": directory}})


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": [1, 2],
            "path": ["a.wav", "b.wav"],
            "split": ["training", "test"],
            "artist_id": [10, 20],
            "group_id": [1, 2],
            "genre_top": ["rock", "pop"],
            "genres_all": ["[1]", "[2]"],
        }
    )


def test_write_index_writes_the_standard_columns_in_order(
    isolated_roots: dict[str, Path],
) -> None:
    dataset = _ToyDataset(_config(), _valid_frame())
    path = dataset.write_index()

    assert path == paths.dataset_index("toy")
    on_disk = pd.read_csv(path)
    assert list(on_disk.columns) == list(INDEX_COLUMNS)
    assert len(on_disk) == 2


def test_write_index_reorders_extra_columns_out(isolated_roots: dict[str, Path]) -> None:
    frame = _valid_frame()
    frame["extra"] = ["x", "y"]
    dataset = _ToyDataset(_config(), frame)
    path = dataset.write_index()
    assert list(pd.read_csv(path).columns) == list(INDEX_COLUMNS)


def test_write_index_refuses_a_frame_missing_a_standard_column(
    isolated_roots: dict[str, Path],
) -> None:
    frame = _valid_frame().drop(columns=["artist_id"])
    dataset = _ToyDataset(_config(), frame)
    with pytest.raises(DatasetError, match="missing columns"):
        dataset.write_index()


def test_name_and_directory_name_read_the_dataset_section() -> None:
    dataset = _ToyDataset(
        Config(data={"dataset": {"name": "toy_name", "directory": "toy_dir"}}), _valid_frame()
    )
    assert dataset.name == "toy_name"
    assert dataset.directory_name == "toy_dir"
