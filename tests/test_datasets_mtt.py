"""Tests for :mod:`musicsim.datasets.mtt`.

No network: the download tests serve synthetic bytes through a patched
``urlopen`` and build a real zip of empty stand-in mp3 files, split into three
parts as MTT ships it. The index tests write tab-separated mini-CSVs with the
real column names; the constraint derivation itself is tested in
``test_mtt_triplets.py``. Only the ``slow`` test reads the real corpus.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from musicsim import paths
from musicsim.config import Config, load_config
from musicsim.datasets import mtt
from musicsim.datasets.base import INDEX_COLUMNS, DatasetError
from musicsim.datasets.download import DownloadError, DownloadSpec
from musicsim.datasets.mtt import MttDataset
from musicsim.registry import DATASETS

CLIP_INFO_RELATIVE = "raw/clip_info_final.csv"
COMPARISONS_RELATIVE = "raw/comparisons_final.csv"
AUDIO_RELATIVE = "raw/audio"

# clip_id, artist, mp3_path. Artists are out of alphabetical order on purpose;
# clip 4 has no mp3_path, as 5519 rows of the real clip_info_final.csv.
_CLIP_ROWS = [
    (7, "Zeta", "f/zeta-07.mp3"),
    (2, "Alpha", "0/alpha-02.mp3"),
    (5, "Mid", "c/mid-05.mp3"),
    (3, "Alpha", "b/alpha-03.mp3"),
    (4, "Mid", None),
]

# One odd-one-out row over clips 2, 5 and 7: 3 votes for 2 as the outlier, 1 for
# 5. Edges: (5,7,2) 3, (7,5,2) 3, (2,7,5) 1, (7,2,5) 1; (7,2,5) cancels against
# (7,5,2), leaving the three constraints below. Clip 3 is in no comparison.
_COMPARISON_ROWS = [(2, 5, 7, 3, 1, 0)]
_CONSTRAINTS = [(5, 7, 2, 3, 0), (7, 5, 2, 2, 0), (2, 7, 5, 1, 0)]
_COUNTS = {
    "rows": 1,
    "votes": 4,
    "edges": 8,
    "unique_edges": 4,
    "constraints": 3,
    "weight": 6,
    "triplets": 1,
    "clips": 3,
}


def _write_clip_info(path: Path, rows: list[tuple]) -> None:
    frame = pd.DataFrame(rows, columns=["clip_id", "artist", "mp3_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def _write_comparisons(path: Path, rows: list[tuple]) -> None:
    columns = [
        "clip1_id",
        "clip2_id",
        "clip3_id",
        "clip1_numvotes",
        "clip2_numvotes",
        "clip3_numvotes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, sep="\t", index=False)


def _config(*, expected_items: int, expected_constraints: dict[str, int] = _COUNTS) -> Config:
    return Config(
        data={
            "dataset": {
                "name": "toy_mtt",
                "directory": "mtt",
                "metadata": {"clip_info": CLIP_INFO_RELATIVE, "comparisons": COMPARISONS_RELATIVE},
                "audio_dir": AUDIO_RELATIVE,
                "expected_items": expected_items,
                "expected_constraints": expected_constraints,
            }
        }
    )


def _make_dataset(
    rows: list[tuple] = _CLIP_ROWS,
    *,
    expected_items: int = 4,
    expected_constraints: dict[str, int] = _COUNTS,
    comparisons: list[tuple] = _COMPARISON_ROWS,
    skip_audio: tuple[str, ...] = (),
) -> MttDataset:
    dataset_dir = paths.dataset_dir("mtt")
    _write_clip_info(dataset_dir / CLIP_INFO_RELATIVE, rows)
    _write_comparisons(dataset_dir / COMPARISONS_RELATIVE, comparisons)
    for _, _, mp3_path in rows:
        if mp3_path is None or mp3_path in skip_audio:
            continue
        file_path = dataset_dir / AUDIO_RELATIVE / mp3_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"")
    return MttDataset(
        _config(expected_items=expected_items, expected_constraints=expected_constraints)
    )


# --- registration --------------------------------------------------------------
def test_mtt_is_registered_under_its_directory_name() -> None:
    assert DATASETS.get("mtt") is MttDataset


# --- the index -------------------------------------------------------------------
def test_build_index_maps_directories_artists_and_votes(isolated_roots: dict[str, Path]) -> None:
    frame = _make_dataset().build_index()

    assert list(frame.columns) == [*INDEX_COLUMNS, "in_triplets"]
    assert frame["item_id"].tolist() == [2, 3, 5, 7]  # sorted; clip 4 has no audio
    assert frame["split"].tolist() == ["training", "training", "validation", "test"]
    # Alphabetical: Alpha 0, Mid 1, Zeta 2.
    assert frame["artist_id"].tolist() == [0, 0, 1, 2]
    assert frame["group_id"].tolist() == [2, 3, 5, 7]
    assert frame["in_triplets"].tolist() == [True, False, True, True]
    assert frame["genre_top"].isna().all()
    assert frame["genres_all"].isna().all()
    assert frame["path"].iloc[0] == str(
        paths.dataset_dir("mtt") / AUDIO_RELATIVE / "0/alpha-02.mp3"
    )


def test_artist_ids_do_not_depend_on_row_order(isolated_roots: dict[str, Path]) -> None:
    forward = _make_dataset().build_index()
    backward = _make_dataset(list(reversed(_CLIP_ROWS))).build_index()
    pd.testing.assert_frame_equal(forward, backward)


def test_write_index_writes_the_eight_columns(isolated_roots: dict[str, Path]) -> None:
    path = _make_dataset().write_index()
    on_disk = pd.read_csv(path)
    assert path == paths.dataset_index("mtt")
    assert list(on_disk.columns) == [*INDEX_COLUMNS, "in_triplets"]
    assert on_disk["in_triplets"].tolist() == [True, False, True, True]


def test_write_index_writes_the_constraints_next_to_the_index(
    isolated_roots: dict[str, Path],
) -> None:
    dataset = _make_dataset()
    dataset.write_index()
    on_disk = pd.read_csv(dataset.triplet_constraints_path)
    assert dataset.triplet_constraints_path == paths.dataset_dir("mtt") / "triplet_constraints.csv"
    assert list(on_disk.columns) == ["anchor", "pos", "neg", "margin", "triplet_id"]
    assert sorted(on_disk.itertuples(index=False, name=None)) == sorted(_CONSTRAINTS)


def test_a_compared_clip_that_loses_its_constraints_is_not_flagged(
    isolated_roots: dict[str, Path],
) -> None:
    # One vote for each clip of (3, 5, 7) as the outlier: every edge meets its
    # opposite with the same weight, so clip 3 is compared but constrained nowhere.
    comparisons = [*_COMPARISON_ROWS, (3, 5, 7, 1, 1, 1)]
    counts = {**_COUNTS, "rows": 2, "votes": 7, "edges": 14, "unique_edges": 10}
    frame = _make_dataset(comparisons=comparisons, expected_constraints=counts).build_index()
    assert frame["in_triplets"].tolist() == [True, False, True, True]


def test_wrong_clip_count_is_refused(isolated_roots: dict[str, Path]) -> None:
    with pytest.raises(DatasetError, match="expected 5 clips"):
        _make_dataset(expected_items=5).build_index()


def test_duplicate_clip_ids_are_refused(isolated_roots: dict[str, Path]) -> None:
    rows = [*_CLIP_ROWS, (7, "Zeta", "f/zeta-07-copy.mp3")]
    with pytest.raises(DatasetError, match=r"duplicate item_id values in the index: \[7\]"):
        _make_dataset(rows, expected_items=5).build_index()


def test_an_unknown_directory_is_refused(isolated_roots: dict[str, Path]) -> None:
    rows = [*_CLIP_ROWS, (9, "Zeta", "g/zeta-09.mp3")]
    with pytest.raises(DatasetError, match=r"directories \['g'\]"):
        _make_dataset(rows, expected_items=5).build_index()


def test_a_missing_split_is_refused(isolated_roots: dict[str, Path]) -> None:
    rows = [row for row in _CLIP_ROWS if row[0] != 5]  # the only validation clip
    with pytest.raises(DatasetError, match="expected splits"):
        _make_dataset(rows, expected_items=3).build_index()


def test_wrong_constraint_figures_are_refused(isolated_roots: dict[str, Path]) -> None:
    with pytest.raises(DatasetError, match="votes: expected 5, got 4"):
        _make_dataset(expected_constraints={**_COUNTS, "votes": 5}).build_index()


def test_a_constrained_clip_without_audio_is_refused(isolated_roots: dict[str, Path]) -> None:
    # Clip 4 has no mp3_path, so it is not indexed, yet it carries constraints.
    with pytest.raises(
        DatasetError, match="expected 3 indexed clips with a triplet constraint, got 2"
    ):
        _make_dataset(comparisons=[(2, 4, 7, 3, 1, 0)]).build_index()


def test_a_missing_audio_file_is_refused(isolated_roots: dict[str, Path]) -> None:
    with pytest.raises(DatasetError, match="missing on disk"):
        _make_dataset(skip_audio=("c/mid-05.mp3",)).build_index()


def test_build_index_without_a_download_names_the_missing_file(
    isolated_roots: dict[str, Path],
) -> None:
    dataset = MttDataset(_config(expected_items=4))
    with pytest.raises(FileNotFoundError, match="musicsim download"):
        dataset.build_index()


# --- the download ----------------------------------------------------------------
_MP3_NAMES = ("0/a.mp3", "c/b.mp3", "f/c.mp3")


def _audio_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in _MP3_NAMES:
            archive.writestr(name, f"fake mp3 {name}")
    return buffer.getvalue()


class _FakeResponse(io.BytesIO):
    status = 200

    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, object]]:
    """Patch the six downloads to synthetic files; count requests per URL."""
    archive = _audio_archive()
    third = len(archive) // 3
    bodies = {
        "clip_info_final.csv": b"clip_id\tartist\tmp3_path\n",
        "annotations_final.csv": b"clip_id\n",
        "comparisons_final.csv": b"clip1_id\n",
        "mp3.zip.001": archive[:third],
        "mp3.zip.002": archive[third : 2 * third],
        "mp3.zip.003": archive[2 * third :],
    }
    specs = tuple(
        DownloadSpec(
            url=f"http://mtt.invalid/{name}",
            filename=name,
            sha1=hashlib.sha1(body).hexdigest(),
            size_bytes=len(body),
        )
        for name, body in bodies.items()
    )
    requests: list[str] = []

    def _urlopen(request: object, timeout: float) -> _FakeResponse:
        url = request.full_url  # type: ignore[attr-defined]
        requests.append(url)
        return _FakeResponse(bodies[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(mtt, "_DOWNLOADS", specs)
    monkeypatch.setattr(mtt, "ensure_free_space", lambda directory, required: None)
    monkeypatch.setattr("musicsim.datasets.download.urllib.request.urlopen", _urlopen)
    monkeypatch.setattr("musicsim.datasets.download.time.sleep", lambda seconds: None)
    yield {"archive": archive, "bodies": bodies, "requests": requests}


def _download_dataset(expected_items: int = len(_MP3_NAMES)) -> MttDataset:
    return MttDataset(_config(expected_items=expected_items))


def test_concatenate_joins_the_parts_byte_for_byte(tmp_path: Path) -> None:
    parts = []
    for number, body in enumerate((b"abc", b"", b"\x00\xffxyz"), start=1):
        part = tmp_path / f"part.{number}"
        part.write_bytes(body)
        parts.append(part)

    mtt._concatenate(parts, tmp_path / "whole", chunk_size=2)

    assert (tmp_path / "whole").read_bytes() == b"abc\x00\xffxyz"
    assert not (tmp_path / "whole.part").exists()


def test_download_reassembles_and_extracts_the_archive(
    isolated_roots: dict[str, Path], served: dict[str, object]
) -> None:
    dataset = _download_dataset()
    dataset.download()

    assert (dataset.raw_dir / "mp3.zip").read_bytes() == served["archive"]
    for name in _MP3_NAMES:
        assert (dataset.audio_dir / name).read_text() == f"fake mp3 {name}"
    assert len(served["requests"]) == 6  # type: ignore[arg-type]


def test_a_second_download_fetches_nothing(
    isolated_roots: dict[str, Path], served: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _download_dataset()
    dataset.download()
    served["requests"].clear()  # type: ignore[attr-defined]

    def _must_not_extract(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a completed extraction must not be repeated")

    monkeypatch.setattr(mtt, "_concatenate", _must_not_extract)
    dataset.download()
    assert served["requests"] == []


def test_force_downloads_and_extracts_again(
    isolated_roots: dict[str, Path], served: dict[str, object]
) -> None:
    dataset = _download_dataset()
    dataset.download()
    (dataset.audio_dir / _MP3_NAMES[0]).write_text("tampered")
    served["requests"].clear()  # type: ignore[attr-defined]

    dataset.download(force=True)

    assert len(served["requests"]) == 6  # type: ignore[arg-type]
    assert (dataset.audio_dir / _MP3_NAMES[0]).read_text() == f"fake mp3 {_MP3_NAMES[0]}"


def test_a_wrong_checksum_is_refused(
    isolated_roots: dict[str, Path], served: dict[str, object]
) -> None:
    served["bodies"]["mp3.zip.002"] = b"corrupted"  # type: ignore[index]
    with pytest.raises(DownloadError, match=r"mp3\.zip\.002: checksum mismatch"):
        _download_dataset().download()


def test_a_wrong_mp3_count_is_refused(
    isolated_roots: dict[str, Path], served: dict[str, object]
) -> None:
    dataset = _download_dataset(expected_items=4)
    with pytest.raises(DatasetError, match=r"expected 4 mp3 files .* found 3"):
        dataset.download()
    assert not (dataset.raw_dir / ".mp3.zip.extracted").exists()


# --- the real corpus -------------------------------------------------------------
@pytest.mark.slow
def test_the_real_index_has_every_clip_and_every_constrained_clip() -> None:
    config = load_config(paths.REPO_ROOT / "configs" / "datasets" / "mtt.yaml")
    dataset = MttDataset(config)
    if not dataset.clip_info_csv.is_file():
        pytest.skip("MTT not downloaded; run `musicsim download --dataset mtt` first")

    frame = dataset.build_index()

    assert len(frame) == 25863
    assert int(frame["in_triplets"].sum()) == 993
    assert frame["artist_id"].nunique() == 230
    assert frame["split"].value_counts().to_dict() == {
        "training": 18709,
        "validation": 1825,
        "test": 5329,
    }
