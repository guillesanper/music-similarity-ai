"""Tests for :mod:`musicsim.datasets.fma`.

No network and no real audio: ``tracks.csv`` is synthesised with the same
two-row header FMA ships, and audio files are empty stand-ins whose only
tested property is that they exist. What matters here is the filtering logic
and, above all, that every consistency check listed in the project plan stops
the process with a message naming what was expected, what was found, and
where to look.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from musicsim import paths
from musicsim.config import Config
from musicsim.datasets.base import INDEX_COLUMNS, DatasetError
from musicsim.datasets.fma import FmaDataset
from musicsim.registry import DATASETS

TRACKS_RELATIVE = "raw/fma_metadata/tracks.csv"
AUDIO_RELATIVE = "raw/fma_small"

# 8 valid items, 2 genres x 4, artist-disjoint splits (6 training, 1 validation, 1 test).
_VALID_ROWS = [
    # item_id, subset, split,       artist_id, genre_top, genres_all
    (2, "small", "training", 100, "rock", "[10]"),
    (3, "small", "training", 101, "rock", "[10]"),
    (4, "small", "training", 102, "rock", "[10]"),
    (5, "small", "training", 103, "pop", "[20]"),
    (6, "small", "training", 104, "pop", "[20]"),
    (7, "small", "training", 105, "pop", "[20]"),
    (8, "small", "validation", 200, "pop", "[20]"),
    (9, "small", "test", 300, "rock", "[10]"),
]


def _write_tracks_csv(path: Path, rows: list[tuple]) -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("set", "subset"),
            ("set", "split"),
            ("artist", "id"),
            ("track", "genre_top"),
            ("track", "genres_all"),
        ]
    )
    ids = [row[0] for row in rows]
    data = [row[1:] for row in rows]
    frame = pd.DataFrame(data, index=pd.Index(ids, name="track_id"), columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def _touch_audio_files(audio_dir: Path, item_ids: list[int]) -> None:
    for item_id in item_ids:
        tid = f"{item_id:06d}"
        file_path = audio_dir / tid[:3] / f"{tid}.mp3"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"")


def _make_dataset(
    isolated_roots: dict[str, Path],
    rows: list[tuple],
    *,
    expected_items: int,
    expected_genres: int,
    exclude_items: list[int] | None = None,
    skip_ids: list[int] | None = None,
) -> FmaDataset:
    dataset_dir = paths.dataset_dir("fma")
    _write_tracks_csv(dataset_dir / TRACKS_RELATIVE, rows)
    present_ids = [row[0] for row in rows if row[0] not in (skip_ids or [])]
    _touch_audio_files(dataset_dir / AUDIO_RELATIVE, present_ids)

    config = Config(
        data={
            "dataset": {
                "name": "toy_fma_small",
                "directory": "fma",
                "subset": "small",
                "metadata": {"tracks": TRACKS_RELATIVE},
                "audio_dir": AUDIO_RELATIVE,
                "expected_items": expected_items,
                "expected_genres": expected_genres,
                "exclude_items": exclude_items or [],
            }
        }
    )
    return FmaDataset(config)


# --- registration --------------------------------------------------------------
def test_fma_is_registered_under_its_directory_name() -> None:
    assert DATASETS.get("fma") is FmaDataset


# --- the happy path --------------------------------------------------------------
def test_build_index_is_deterministic_and_has_the_standard_columns(
    isolated_roots: dict[str, Path],
) -> None:
    dataset = _make_dataset(isolated_roots, _VALID_ROWS, expected_items=8, expected_genres=2)

    first = dataset.build_index()
    second = dataset.build_index()

    assert list(first.columns) == list(INDEX_COLUMNS)
    assert first["item_id"].tolist() == sorted(first["item_id"].tolist())
    pd.testing.assert_frame_equal(first, second)


def test_group_id_defaults_to_item_id(isolated_roots: dict[str, Path]) -> None:
    dataset = _make_dataset(isolated_roots, _VALID_ROWS, expected_items=8, expected_genres=2)
    frame = dataset.build_index()
    assert (frame["group_id"] == frame["item_id"]).all()


def test_write_index_writes_a_csv_with_the_standard_columns(
    isolated_roots: dict[str, Path],
) -> None:
    dataset = _make_dataset(isolated_roots, _VALID_ROWS, expected_items=8, expected_genres=2)
    path = dataset.write_index()
    assert path == paths.dataset_index("fma")
    on_disk = pd.read_csv(path)
    assert list(on_disk.columns) == list(INDEX_COLUMNS)
    assert len(on_disk) == 8


def test_excluded_items_are_removed_before_counting(isolated_roots: dict[str, Path]) -> None:
    # Item 6 (training, pop) is excluded; all three splits and both genres
    # still have at least one representative among the remaining 7 items.
    dataset = _make_dataset(
        isolated_roots,
        _VALID_ROWS,
        expected_items=8,
        expected_genres=2,
        exclude_items=[6],
    )
    frame = dataset.build_index()
    assert 6 not in frame["item_id"].tolist()
    assert len(frame) == 7


# --- assertions that must stop the process --------------------------------------
def test_wrong_item_count_is_refused(isolated_roots: dict[str, Path]) -> None:
    dataset = _make_dataset(isolated_roots, _VALID_ROWS, expected_items=9, expected_genres=2)
    with pytest.raises(DatasetError, match="expected 9 items"):
        dataset.build_index()


def test_wrong_genre_count_is_refused(isolated_roots: dict[str, Path]) -> None:
    dataset = _make_dataset(isolated_roots, _VALID_ROWS, expected_items=8, expected_genres=3)
    with pytest.raises(DatasetError, match="expected 3 distinct genre_top"):
        dataset.build_index()


def test_a_missing_split_is_refused(isolated_roots: dict[str, Path]) -> None:
    rows = [row for row in _VALID_ROWS if row[2] != "test"]  # drop the only test item
    dataset = _make_dataset(isolated_roots, rows, expected_items=7, expected_genres=2)
    with pytest.raises(DatasetError, match="expected splits"):
        dataset.build_index()


def test_a_shared_artist_across_splits_is_refused(isolated_roots: dict[str, Path]) -> None:
    rows = list(_VALID_ROWS)
    # Item 9 (test, artist 300) becomes item 9 with artist 100, already in training.
    rows[-1] = (9, "small", "test", 100, "rock", "[10]")
    dataset = _make_dataset(isolated_roots, rows, expected_items=8, expected_genres=2)
    with pytest.raises(DatasetError, match="artist"):
        dataset.build_index()


def test_a_missing_genre_top_is_refused(isolated_roots: dict[str, Path]) -> None:
    rows = list(_VALID_ROWS)
    rows[0] = (2, "small", "training", 100, float("nan"), "[10]")
    dataset = _make_dataset(isolated_roots, rows, expected_items=8, expected_genres=2)
    with pytest.raises(DatasetError, match="no genre_top"):
        dataset.build_index()


def test_a_missing_audio_file_is_refused(isolated_roots: dict[str, Path]) -> None:
    dataset = _make_dataset(
        isolated_roots, _VALID_ROWS, expected_items=8, expected_genres=2, skip_ids=[9]
    )
    with pytest.raises(DatasetError, match="missing on disk"):
        dataset.build_index()


def test_duplicate_item_ids_are_refused(isolated_roots: dict[str, Path]) -> None:
    rows = [*_VALID_ROWS, (2, "small", "test", 999, "rock", "[10]")]
    dataset = _make_dataset(isolated_roots, rows, expected_items=9, expected_genres=2)
    with pytest.raises(DatasetError, match="duplicate item_id"):
        dataset.build_index()


def test_a_subset_without_a_verified_checksum_is_refused(isolated_roots: dict[str, Path]) -> None:
    dataset_dir = paths.dataset_dir("fma")
    _write_tracks_csv(dataset_dir / TRACKS_RELATIVE, _VALID_ROWS)
    config = Config(
        data={
            "dataset": {
                "name": "toy_fma_medium",
                "directory": "fma",
                "subset": "medium",
                "metadata": {"tracks": TRACKS_RELATIVE},
                "audio_dir": AUDIO_RELATIVE,
                "expected_items": 8,
                "expected_genres": 2,
                "exclude_items": [],
            }
        }
    )
    dataset = FmaDataset(config)
    with pytest.raises(NotImplementedError, match="no verified checksum"):
        dataset.download()


def test_build_index_without_a_download_names_the_missing_file(
    isolated_roots: dict[str, Path],
) -> None:
    config = Config(
        data={
            "dataset": {
                "name": "toy_fma_small",
                "directory": "fma",
                "subset": "small",
                "metadata": {"tracks": TRACKS_RELATIVE},
                "audio_dir": AUDIO_RELATIVE,
                "expected_items": 8,
                "expected_genres": 2,
                "exclude_items": [],
            }
        }
    )
    dataset = FmaDataset(config)
    with pytest.raises(FileNotFoundError, match="musicsim download"):
        dataset.build_index()
