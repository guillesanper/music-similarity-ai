"""MagnaTagATune (Law et al., 2009): download, reassemble, unpack and index.

Source: <https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset> (key
``Law2009`` in ``docs/sources.md``). The corpus is used for evaluation only;
see ``datasets/mtt/README.md``.

**No official checksums.** MTT publishes none, so the SHA-1 values in
:data:`_DOWNLOADS` were computed locally from the first download (2026-11-04).
They detect a corrupt or changed file; they do not certify that the first copy
matched what the authors uploaded.

**The audio is one zip split in three.** ``mp3.zip.001``-``003`` are byte
ranges of a single archive, not three archives: they are concatenated into
``mp3.zip`` by streaming (never loaded into memory) before extraction.

**Which rows are clips.** ``clip_info_final.csv`` lists 31 382 clip ids, but
5 519 of them have an empty ``mp3_path`` and no audio in the archive (none of
them appears in ``annotations_final.csv`` either). The index keeps the 25 863
rows that have a path, the published size of the corpus. All three CSVs are
tab-separated and quoted.

**Split.** MTT ships no split file. The tagging literature uses the archive's
16 directories (``0``-``f``) in a 12 : 1 : 3 ratio: ``0``-``b`` training,
``c`` validation, ``d``-``f`` test (the protocol of ``Castellon2021``, whose
MTT figures this project compares against). Nothing is trained for similarity
on MTT; only the label probes use this split. The directories do not keep
artists apart, and this is documented, not corrected.

**Artist ids.** ``clip_info_final.csv`` gives artist names. They are numbered in
alphabetical order, so the same CSV always yields the same ids (our own
choice; any deterministic order would do).

**in_triplets.** True for every clip that keeps at least one consistent
similarity constraint (993 clips), not for every clip that appears in
``comparisons_final.csv`` (1019): the other 26 lose all their constraints when
opposing votes cancel, so no evaluation can use them. The derivation lives in
:mod:`musicsim.datasets.mtt_triplets`; :meth:`MttDataset.write_index` writes its
result next to the index, as ``triplet_constraints.csv``, so that the index and
the constraints always come from the same build.

MTT has no genre labels: ``genre_top`` and ``genres_all`` are left empty.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from musicsim import paths
from musicsim.datasets.base import Dataset, DatasetError
from musicsim.datasets.download import DownloadSpec, download_file, ensure_free_space
from musicsim.datasets.mtt_triplets import (
    check_constraint_counts,
    constraint_counts,
    derive_constraints,
)
from musicsim.registry import register_dataset

__all__ = ["MttDataset"]

logger = logging.getLogger(__name__)

_MIRROR = "https://mirg.city.ac.uk/datasets/magnatagatune"

#: Every file the corpus is made of. The SHA-1 values are local, not official:
#: computed on 2026-11-04 from the first download (MTT publishes no checksums).
#: The sizes are exact, as served on that date.
_DOWNLOADS: tuple[DownloadSpec, ...] = (
    DownloadSpec(
        url=f"{_MIRROR}/clip_info_final.csv",
        filename="clip_info_final.csv",
        sha1="f2c884791b61413054bec12c5fd970eaec528da3",
        size_bytes=8_608_397,
    ),
    DownloadSpec(
        url=f"{_MIRROR}/annotations_final.csv",
        filename="annotations_final.csv",
        sha1="1c61c804ee2ee037c8f06c2dadc3a1b7b5aa31be",
        size_bytes=21_517_373,
    ),
    DownloadSpec(
        url=f"{_MIRROR}/comparisons_final.csv",
        filename="comparisons_final.csv",
        sha1="3755c22179fa59474986b38a709433af657fd266",
        size_bytes=131_947,
    ),
    DownloadSpec(
        url=f"{_MIRROR}/mp3.zip.001",
        filename="mp3.zip.001",
        sha1="42d6167bd4934ee917c3aef3d238e50b17eaee15",
        size_bytes=1_100_000_000,
    ),
    DownloadSpec(
        url=f"{_MIRROR}/mp3.zip.002",
        filename="mp3.zip.002",
        sha1="f63d51856e7915457bd6d193564c65b1296f4b2c",
        size_bytes=1_100_000_000,
    ),
    DownloadSpec(
        url=f"{_MIRROR}/mp3.zip.003",
        filename="mp3.zip.003",
        sha1="0e0d23498d33ae616a67901749ac4dc3fc051d2d",
        size_bytes=772_769_864,
    ),
)

#: The parts of the audio archive, in the order they must be concatenated.
_ARCHIVE_PARTS: tuple[str, ...] = ("mp3.zip.001", "mp3.zip.002", "mp3.zip.003")

#: Name of the reassembled archive under the raw directory.
_ARCHIVE_NAME = "mp3.zip"

#: The parts (2.97 GB), the reassembled zip (2.97 GB) and the extracted mp3
#: files (about 3 GB, they are stored uncompressed), rounded up.
_REQUIRED_FREE_BYTES = 10 * 1024**3

#: Archive directory (first character of ``mp3_path``) -> split. The 12 : 1 : 3
#: split of the tagging literature; see the module docstring.
_SPLIT_BY_DIRECTORY: dict[str, str] = {
    **dict.fromkeys("0123456789ab", "training"),
    "c": "validation",
    **dict.fromkeys("def", "test"),
}

#: File name of the derived constraints, next to ``index.csv``.
_CONSTRAINTS_NAME = "triplet_constraints.csv"

#: MTT's CSV files are tab-separated.
_CSV_SEPARATOR = "\t"


@register_dataset("mtt")
class MttDataset(Dataset):
    """MagnaTagATune: download the CSVs and the three-part audio archive, build the index."""

    extra_columns = ("in_triplets",)

    # -- configuration -------------------------------------------------------
    @property
    def clip_info_csv(self) -> Path:
        return paths.dataset_dir(self.directory_name) / self.config.require(
            "dataset.metadata.clip_info"
        )

    @property
    def comparisons_csv(self) -> Path:
        return paths.dataset_dir(self.directory_name) / self.config.require(
            "dataset.metadata.comparisons"
        )

    @property
    def audio_dir(self) -> Path:
        return paths.dataset_dir(self.directory_name) / self.config.require("dataset.audio_dir")

    @property
    def triplet_constraints_path(self) -> Path:
        """Where :meth:`write_index` writes the constraints of :meth:`build_constraints`."""
        return paths.dataset_dir(self.directory_name) / _CONSTRAINTS_NAME

    # -- download --------------------------------------------------------------
    def download(self, *, force: bool = False) -> None:
        """Download and verify the six files, reassemble the audio archive and extract it.

        Idempotent: verified files are not fetched again and a completed
        extraction (marked by a hidden file) is not repeated, unless ``force``.
        """
        ensure_free_space(paths.dataset_dir(self.directory_name), _REQUIRED_FREE_BYTES)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        archive = self.raw_dir / _ARCHIVE_NAME
        marker = self.raw_dir / f".{_ARCHIVE_NAME}.extracted"
        if force:
            for spec in _DOWNLOADS:
                (self.raw_dir / spec.filename).unlink(missing_ok=True)
            archive.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)

        for spec in _DOWNLOADS:
            download_file(spec, self.raw_dir / spec.filename)

        if marker.is_file():
            return
        _concatenate([self.raw_dir / part for part in _ARCHIVE_PARTS], archive)
        with zipfile.ZipFile(archive) as archive_file:
            archive_file.extractall(self.audio_dir)
        self._check_audio_count()
        marker.write_text("ok\n", encoding="utf-8")

    def _check_audio_count(self) -> None:
        expected = self.config.require("dataset.expected_items")
        found = sum(1 for _ in self.audio_dir.rglob("*.mp3"))
        if found != expected:
            raise DatasetError(
                f"expected {expected} mp3 files under {self.audio_dir} after extraction, "
                f"found {found}; run `musicsim download --dataset {self.name}` with force"
            )

    # -- triplet constraints -----------------------------------------------
    def build_constraints(self) -> pd.DataFrame:
        """Derive the consistent constraints and check every figure of the derivation.

        The expected figures come from ``dataset.expected_constraints``; any
        difference raises :class:`DatasetError`.
        """
        if not self.comparisons_csv.is_file():
            raise FileNotFoundError(
                f"{self.comparisons_csv} not found; "
                f"run `musicsim download --dataset {self.name}` first"
            )
        comparisons = pd.read_csv(self.comparisons_csv, sep=_CSV_SEPARATOR)
        constraints = derive_constraints(comparisons)
        counts = constraint_counts(comparisons, constraints)
        check_constraint_counts(counts, self.config.require("dataset.expected_constraints"))
        logger.info("%s: triplet constraints %s", self.name, counts)
        return constraints

    # -- index -------------------------------------------------------------
    def build_index(self) -> pd.DataFrame:
        """Index every clip with audio, flag the clips with a constraint, validate."""
        for csv in (self.clip_info_csv, self.comparisons_csv):
            if not csv.is_file():
                raise FileNotFoundError(
                    f"{csv} not found; run `musicsim download --dataset {self.name}` first"
                )

        clips = pd.read_csv(self.clip_info_csv, sep=_CSV_SEPARATOR)
        without_audio = clips["mp3_path"].isna()
        logger.info(
            "%s: %d of %d rows of %s have no mp3_path and are left out",
            self.name,
            int(without_audio.sum()),
            len(clips),
            self.clip_info_csv.name,
        )
        clips = clips[~without_audio]

        constraints = self.build_constraints()
        constrained = pd.unique(constraints[["anchor", "pos", "neg"]].to_numpy().ravel())

        artists = sorted(clips["artist"].unique())
        item_ids = clips["clip_id"].astype(int).to_numpy()
        mp3_paths = clips["mp3_path"].astype(str)
        frame = pd.DataFrame(
            {
                "item_id": item_ids,
                "path": [str(self.audio_dir / mp3_path) for mp3_path in mp3_paths],
                "split": mp3_paths.str[0].map(_SPLIT_BY_DIRECTORY).to_numpy(),
                "artist_id": pd.Categorical(clips["artist"], categories=artists).codes,
                # Placeholder, as for every corpus; see INDEX_COLUMNS in
                # musicsim.datasets.base.
                "group_id": item_ids,
                "genre_top": pd.NA,
                "genres_all": pd.NA,
                "in_triplets": pd.Series(item_ids).isin(constrained).to_numpy(),
            }
        )
        frame = frame.sort_values("item_id").reset_index(drop=True)

        self._validate(frame, mp3_paths)
        return frame

    def write_index(self) -> Path:
        """Write ``index.csv`` and, next to it, ``triplet_constraints.csv``; return the index path.

        The constraints are derived again rather than passed from
        :meth:`build_index`, which the base class calls: the derivation is cheap
        and both calls read the same CSV.
        """
        index_path = super().write_index()
        self.build_constraints().to_csv(self.triplet_constraints_path, index=False)
        return index_path

    def _validate(self, frame: pd.DataFrame, mp3_paths: pd.Series) -> None:
        """Raise :class:`DatasetError` with an actionable message on any mismatch."""
        expected = self.config.require("dataset.expected_items")
        if len(frame) != expected:
            raise DatasetError(
                f"expected {expected} clips with an mp3_path in {self.clip_info_csv.name}, "
                f"got {len(frame)}. Check that the file is the unmodified download."
            )

        duplicated = frame.loc[frame["item_id"].duplicated(), "item_id"].tolist()
        if duplicated:
            raise DatasetError(f"duplicate item_id values in the index: {duplicated[:5]}")

        unknown = sorted(set(mp3_paths.str[0]) - set(_SPLIT_BY_DIRECTORY))
        if unknown:
            raise DatasetError(
                f"mp3_path directories {unknown} are not archive directories; "
                f"expected one of {sorted(_SPLIT_BY_DIRECTORY)}"
            )

        split_sizes = frame["split"].value_counts().to_dict()
        if set(split_sizes) != set(_SPLIT_BY_DIRECTORY.values()):
            raise DatasetError(
                f"expected splits {sorted(set(_SPLIT_BY_DIRECTORY.values()))}, "
                f"got {sorted(split_sizes)}"
            )
        # No split size is published for MTT, so the sizes are reported, not asserted.
        logger.info("%s: clips per split %s", self.name, split_sizes)

        # Equal to the "clips" figure of the derivation unless a constrained
        # clip is missing from the index.
        expected_constrained = self.config.require("dataset.expected_constraints.clips")
        n_constrained = int(frame["in_triplets"].sum())
        if n_constrained != expected_constrained:
            raise DatasetError(
                f"expected {expected_constrained} indexed clips with a triplet constraint, "
                f"got {n_constrained}; a constrained clip has no mp3_path in "
                f"{self.clip_info_csv.name}"
            )

        missing_files = [path for path in frame["path"] if not Path(path).is_file()]
        if missing_files:
            raise DatasetError(
                f"{len(missing_files)} audio file(s) listed in the index are missing on disk, "
                f"e.g. {missing_files[0]}; run `musicsim download --dataset {self.name}` again"
            )


def _concatenate(parts: list[Path], destination: Path, *, chunk_size: int = 1 << 20) -> None:
    """Write the bytes of ``parts``, in order, to ``destination``, streaming."""
    partial = destination.with_name(destination.name + ".part")
    with partial.open("wb") as out:
        for part in parts:
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, out, chunk_size)
    partial.replace(destination)
