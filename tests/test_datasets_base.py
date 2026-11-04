"""Tests for :mod:`musicsim.datasets.base`.

Exercises the shared ``write_index`` contract through a minimal concrete
:class:`Dataset`, independent of any real corpus, plus one byte-level check
that FMA's index (which declares no extra columns) is written exactly as
before extra columns existed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from musicsim import paths
from musicsim.config import Config
from musicsim.datasets.base import INDEX_COLUMNS, Dataset, DatasetError
from musicsim.datasets.fma import FmaDataset


class _ToyDataset(Dataset):
    def __init__(self, config: Config, frame: pd.DataFrame) -> None:
        super().__init__(config)
        self._frame = frame

    def download(self, *, force: bool = False) -> None:
        raise NotImplementedError

    def build_index(self) -> pd.DataFrame:
        return self._frame


class _FlaggedToyDataset(_ToyDataset):
    extra_columns: ClassVar[tuple[str, ...]] = ("flag",)


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


def test_write_index_refuses_an_undeclared_column(isolated_roots: dict[str, Path]) -> None:
    frame = _valid_frame()
    frame["extra"] = ["x", "y"]
    dataset = _ToyDataset(_config(), frame)
    with pytest.raises(DatasetError, match=r"undeclared columns \['extra'\]"):
        dataset.write_index()
    assert not paths.dataset_index("toy").exists()


def test_write_index_appends_declared_extra_columns_after_the_standard_ones(
    isolated_roots: dict[str, Path],
) -> None:
    # The extra column comes first in the frame; the file must still put it last.
    frame = _valid_frame()
    frame.insert(0, "flag", [True, False])
    dataset = _FlaggedToyDataset(_config(), frame)
    on_disk = pd.read_csv(dataset.write_index())

    assert list(on_disk.columns) == [*INDEX_COLUMNS, "flag"]
    assert len(on_disk.columns) == 8
    assert on_disk["flag"].tolist() == [True, False]


def test_write_index_refuses_a_declared_extra_column_that_is_absent(
    isolated_roots: dict[str, Path],
) -> None:
    dataset = _FlaggedToyDataset(_config(), _valid_frame())
    with pytest.raises(DatasetError, match=r"missing columns \['flag'\]"):
        dataset.write_index()


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


def test_fma_index_bytes_are_unchanged_by_extra_column_support(
    isolated_roots: dict[str, Path],
) -> None:
    # Smallest tracks.csv FmaDataset accepts: one item per split, two genres,
    # one artist per item. The expected file is written out by hand.
    dataset_dir = paths.dataset_dir("fma")
    tracks = pd.DataFrame(
        [
            ("small", "training", 100, "rock", "[10]"),
            ("small", "validation", 200, "pop", "[20]"),
            ("small", "test", 300, "rock", "[10]"),
        ],
        index=pd.Index([2, 3, 4], name="track_id"),
        columns=pd.MultiIndex.from_tuples(
            [
                ("set", "subset"),
                ("set", "split"),
                ("artist", "id"),
                ("track", "genre_top"),
                ("track", "genres_all"),
            ]
        ),
    )
    tracks_path = dataset_dir / "raw" / "fma_metadata" / "tracks.csv"
    tracks_path.parent.mkdir(parents=True)
    tracks.to_csv(tracks_path)
    audio_dir = dataset_dir / "raw" / "fma_small"
    audio = {item_id: audio_dir / "000" / f"{item_id:06d}.mp3" for item_id in (2, 3, 4)}
    audio[2].parent.mkdir(parents=True)
    for path in audio.values():
        path.write_bytes(b"")
    config = Config(
        data={
            "dataset": {
                "name": "toy_fma_small",
                "directory": "fma",
                "subset": "small",
                "metadata": {"tracks": "raw/fma_metadata/tracks.csv"},
                "audio_dir": "raw/fma_small",
                "expected_items": 3,
                "expected_genres": 2,
                "exclude_items": [],
            }
        }
    )

    written = FmaDataset(config).write_index().read_bytes()

    # pandas ends CSV rows with os.linesep, as it did before this change.
    expected = os.linesep.join(
        [
            "item_id,path,split,artist_id,group_id,genre_top,genres_all",
            f"2,{audio[2]},training,100,2,rock,[10]",
            f"3,{audio[3]},validation,200,3,pop,[20]",
            f"4,{audio[4]},test,300,4,rock,[10]",
            "",
        ]
    )
    assert written == expected.encode("utf-8")
