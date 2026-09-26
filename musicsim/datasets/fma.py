"""FMA (Defferrard et al., 2017): download, unpack and index.

Source: <https://github.com/mdeff/fma>. Only the ``small`` subset (8000
excerpts of 30 s, 8 balanced genres) is wired up with a verified checksum;
see :data:`_SUBSET_SPECS`.

``tracks.csv`` has a two-row column header (``header=[0, 1]`` when read with
pandas) and already carries a ``genres_all`` column (every genre in the
excerpt's hierarchy, not just the leaf), so no separate read of
``genres.csv`` is needed to fill the index's ``genres_all`` column.

Six excerpts are excluded even though their files are on disk, per
``configs/datasets/fma_small.yaml`` and the `mdeff/fma wiki
<https://github.com/mdeff/fma/wiki>`_: ``99134``, ``108925``, ``133297`` are
about 1 KB with no decodable audio, and ``98565``, ``98567``, ``98569`` decode
but last far less than 30 s.
"""

from __future__ import annotations

import itertools
import zipfile
from pathlib import Path

import pandas as pd

from musicsim import paths
from musicsim.datasets.base import Dataset, DatasetError
from musicsim.datasets.download import DownloadSpec, download_file, ensure_free_space
from musicsim.registry import register_dataset

__all__ = ["FmaDataset"]

_MIRROR = "https://os.unil.cloud.switch.ch/fma"

#: SHA-1 copied verbatim from the "Checksums" section of
#: https://github.com/mdeff/fma/blob/master/README.md (2017-05-09 release).
_METADATA_SPEC = DownloadSpec(
    url=f"{_MIRROR}/fma_metadata.zip",
    filename="fma_metadata.zip",
    sha1="f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
    size_bytes=342 * 1024**2,
)

#: One entry per subset with a checksum verified against the source above.
#: A subset not listed here has no verified checksum and downloading it is
#: refused rather than silently skipping verification.
_SUBSET_SPECS: dict[str, DownloadSpec] = {
    "small": DownloadSpec(
        url=f"{_MIRROR}/fma_small.zip",
        filename="fma_small.zip",
        sha1="ade154f733639d52e35e32f5593efe5be76c6d70",
        size_bytes=int(7.2 * 1024**3),
    ),
}

#: Metadata archive (~0.34 GB) + one subset archive (~7.15 GB for "small"),
#: plus room to extract both, rounded up.
_REQUIRED_FREE_BYTES = 25 * 1024**3

#: FMA's official train/validation/test split names.
_EXPECTED_SPLITS = frozenset({"training", "validation", "test"})


@register_dataset("fma")
class FmaDataset(Dataset):
    """FMA: download the metadata and audio archives, build the item index."""

    # -- configuration -------------------------------------------------------
    @property
    def subset(self) -> str:
        return self.config.require("dataset.subset")

    @property
    def tracks_csv(self) -> Path:
        return paths.dataset_dir(self.directory_name) / self.config.require(
            "dataset.metadata.tracks"
        )

    @property
    def audio_dir(self) -> Path:
        return paths.dataset_dir(self.directory_name) / self.config.require("dataset.audio_dir")

    @property
    def exclude_items(self) -> frozenset[int]:
        return frozenset(int(item) for item in self.config.get("dataset.exclude_items", ()))

    # -- download --------------------------------------------------------------
    def download(self, *, force: bool = False) -> None:
        """Download, verify and extract the metadata archive and this subset's audio."""
        subset = self.subset
        if subset not in _SUBSET_SPECS:
            available = ", ".join(sorted(_SUBSET_SPECS))
            raise NotImplementedError(
                f"no verified checksum for fma_{subset}.zip; only {available} "
                "is wired up. Copy the SHA-1 for it from "
                "https://github.com/mdeff/fma#checksums before adding it here."
            )

        dataset_dir = paths.dataset_dir(self.directory_name)
        ensure_free_space(dataset_dir, _REQUIRED_FREE_BYTES)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        for spec in (_METADATA_SPEC, _SUBSET_SPECS[subset]):
            self._download_and_extract(spec, force=force)

    def _download_and_extract(self, spec: DownloadSpec, *, force: bool) -> None:
        archive = self.raw_dir / spec.filename
        marker = self.raw_dir / f".{spec.filename}.extracted"
        if force:
            archive.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)

        download_file(spec, archive)

        if not marker.is_file():
            with zipfile.ZipFile(archive) as archive_file:
                archive_file.extractall(self.raw_dir)
            marker.write_text("ok\n", encoding="utf-8")

    # -- index -------------------------------------------------------------
    def build_index(self) -> pd.DataFrame:
        """Filter ``tracks.csv`` to this subset, exclude known-bad items, validate."""
        if not self.tracks_csv.is_file():
            raise FileNotFoundError(
                f"{self.tracks_csv} not found; run `musicsim download --dataset {self.name}` first"
            )

        tracks = pd.read_csv(self.tracks_csv, index_col=0, header=[0, 1], low_memory=False)
        tracks.index.name = "item_id"

        subset = tracks[tracks[("set", "subset")] == self.subset]
        subset = subset[~subset.index.isin(self.exclude_items)]

        item_ids = subset.index.astype(int).to_numpy()
        # Built from bare numpy/list values, never from the Series above, so the
        # result gets a fresh default index instead of inheriting the "item_id"
        # index name from `subset` (which would collide with the column below).
        frame = pd.DataFrame(
            {
                "item_id": item_ids,
                "path": [str(self._track_path(item_id)) for item_id in item_ids],
                "split": subset[("set", "split")].astype(str).to_numpy(),
                "artist_id": subset[("artist", "id")].astype("Int64").to_numpy(),
                # Placeholder, recomputed from the embeddings by
                # musicsim.graphs.duplicates; see the docstring of
                # INDEX_COLUMNS in musicsim.datasets.base.
                "group_id": item_ids,
                "genre_top": subset[("track", "genre_top")].to_numpy(),
                "genres_all": subset[("track", "genres_all")].astype(str).to_numpy(),
            }
        )
        frame = frame.sort_values("item_id").reset_index(drop=True)

        self._validate(frame)
        return frame

    def _track_path(self, item_id: int) -> Path:
        tid = f"{int(item_id):06d}"
        return self.audio_dir / tid[:3] / f"{tid}.mp3"

    def _validate(self, frame: pd.DataFrame) -> None:
        """Raise :class:`DatasetError` with an actionable message on any mismatch."""
        expected = self.config.require("dataset.expected_items") - len(self.exclude_items)
        if len(frame) != expected:
            raise DatasetError(
                f"expected {expected} items in the '{self.subset}' subset after excluding "
                f"{len(self.exclude_items)} known-bad tracks, got {len(frame)}. Check that "
                f"{self.tracks_csv} came from an unmodified fma_metadata.zip."
            )

        duplicated = frame.loc[frame["item_id"].duplicated(), "item_id"].tolist()
        if duplicated:
            raise DatasetError(f"duplicate item_id values in the index: {duplicated}")

        if frame["genre_top"].isna().any():
            missing = frame.loc[frame["genre_top"].isna(), "item_id"].tolist()
            raise DatasetError(
                f"{len(missing)} item(s) have no genre_top, e.g. {missing[:5]}; "
                f"the '{self.subset}' subset is expected to have none"
            )

        n_genres = frame["genre_top"].nunique()
        expected_genres = self.config.require("dataset.expected_genres")
        if n_genres != expected_genres:
            raise DatasetError(
                f"expected {expected_genres} distinct genre_top values, got {n_genres}: "
                f"{sorted(frame['genre_top'].unique())}"
            )

        got_splits = frozenset(frame["split"].unique())
        if got_splits != _EXPECTED_SPLITS:
            raise DatasetError(
                f"expected splits {sorted(_EXPECTED_SPLITS)}, got {sorted(got_splits)}"
            )

        artists_by_split = {
            split: frozenset(group["artist_id"]) for split, group in frame.groupby("split")
        }
        for left, right in itertools.combinations(sorted(artists_by_split), 2):
            shared = artists_by_split[left] & artists_by_split[right]
            if shared:
                raise DatasetError(
                    f"artist(s) {sorted(shared)[:5]} appear in both '{left}' and '{right}'; "
                    "the retrieval evaluation assumes splits are artist-disjoint"
                )

        missing_files = [path for path in frame["path"] if not Path(path).is_file()]
        if missing_files:
            raise DatasetError(
                f"{len(missing_files)} audio file(s) listed in the index are missing on disk, "
                f"e.g. {missing_files[0]}; run `musicsim download --dataset {self.name}` again"
            )
