"""Generic resumable, checksum-verified HTTP downloads.

Every corpus's download code (currently only :mod:`musicsim.datasets.fma`) is
built on :func:`download_file`: it resumes a partial transfer with an HTTP
``Range`` request, retries on network errors, and only renames the file into
place once its SHA-1 matches the value published by the corpus's source. A
file that already matches its checksum is left untouched, so re-running
``musicsim download`` after a previous success does no network I/O at all.

Only the standard library and ``tqdm`` are used here (both already project
dependencies), so verifying a multi-gigabyte download does not require adding
``requests`` as a new dependency.
"""

from __future__ import annotations

import hashlib
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

__all__ = ["DownloadError", "DownloadSpec", "download_file", "ensure_free_space", "sha1_of"]


class DownloadError(RuntimeError):
    """A file could not be downloaded or its checksum could not be verified."""


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    """One file to fetch: where from, what it must hash to, and how big it is.

    ``size_bytes`` is only an estimate used for the free-space check before the
    transfer starts; the checksum, not the size, is what decides success.
    """

    url: str
    filename: str
    sha1: str
    size_bytes: int


def ensure_free_space(directory: Path, required_bytes: int) -> None:
    """Raise :class:`DownloadError` unless ``directory``'s filesystem has room.

    Checked once before any transfer starts: failing 6 GB into a 7 GB download
    because the disk filled up is a worse experience than failing immediately.
    """
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if free < required_bytes:
        raise DownloadError(
            f"not enough free space under {directory}: "
            f"{free / 1e9:.1f} GB free, {required_bytes / 1e9:.1f} GB required"
        )


def sha1_of(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-1 hex digest of a file's contents, read in fixed-size chunks."""
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    spec: DownloadSpec,
    destination: Path,
    *,
    chunk_size: int = 1 << 20,
    max_retries: int = 5,
    retry_delay: float = 10.0,
    timeout: float = 30.0,
    progress: bool = True,
) -> Path:
    """Download ``spec`` to ``destination``, resuming and verifying it.

    The transfer is written to a ``.part`` sibling of ``destination`` so an
    interrupted attempt resumes instead of restarting; the file is only
    renamed into place once its SHA-1 matches ``spec.sha1``. A checksum
    mismatch after a byte-complete transfer means the source or the network
    corrupted something that cannot be fixed by resuming, so the partial file
    is discarded and the next attempt starts from scratch.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha1_of(destination) == spec.sha1:
        return destination

    part = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            _download_once(spec, part, chunk_size=chunk_size, timeout=timeout, progress=progress)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_delay)
            continue

        digest = sha1_of(part)
        if digest == spec.sha1:
            part.replace(destination)
            return destination

        part.unlink(missing_ok=True)
        last_error = DownloadError(
            f"{spec.filename}: checksum mismatch (expected {spec.sha1}, got {digest})"
        )
        if attempt < max_retries:
            time.sleep(retry_delay)

    raise DownloadError(
        f"failed to download {spec.filename} after {max_retries} attempt(s): {last_error}"
    ) from last_error


def _download_once(
    spec: DownloadSpec, part: Path, *, chunk_size: int, timeout: float, progress: bool
) -> None:
    """One attempt: open the connection (resuming if ``part`` already exists), stream it."""
    resume_from = part.stat().st_size if part.is_file() else 0
    request = urllib.request.Request(spec.url)
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")

    with urllib.request.urlopen(request, timeout=timeout) as response:
        # A server that ignores Range answers 200 with the full body; resuming
        # then means overwriting rather than appending.
        resumed = resume_from and response.status == 206
        mode = "ab" if resumed else "wb"
        initial = resume_from if resumed else 0
        content_length = int(response.headers.get("Content-Length", 0))
        total = (initial + content_length) or None

        bar = tqdm(
            total=total,
            initial=initial,
            unit="B",
            unit_scale=True,
            desc=spec.filename,
            disable=not progress,
        )
        try:
            with part.open(mode) as handle:
                while chunk := response.read(chunk_size):
                    handle.write(chunk)
                    bar.update(len(chunk))
        finally:
            bar.close()
