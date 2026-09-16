"""Tests for :mod:`musicsim.datasets.download`.

A tiny local HTTP server stands in for the real FMA mirror. Python's
``http.server`` does not honour the ``Range`` header (it always answers 200
with the full body), so the "resume" test below exercises the graceful
fallback path (restart from scratch when the server ignores ``Range``)
rather than a true partial transfer; the retry/backoff and checksum logic
around it is what actually matters and is covered independently.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from musicsim.datasets.download import (
    DownloadError,
    DownloadSpec,
    download_file,
    ensure_free_space,
    sha1_of,
)

CONTENT = b"0123456789" * 100_000
SHA1 = hashlib.sha1(CONTENT).hexdigest()


@pytest.fixture
def served_url(tmp_path: Path) -> Iterator[str]:
    """A background HTTP server exposing one file at ``/file.bin``."""
    serve_dir = tmp_path / "served"
    serve_dir.mkdir()
    (serve_dir / "file.bin").write_bytes(CONTENT)

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(serve_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/file.bin"
    finally:
        server.shutdown()
        thread.join()


def _spec(url: str, *, sha1: str = SHA1) -> DownloadSpec:
    return DownloadSpec(url=url, filename="file.bin", sha1=sha1, size_bytes=len(CONTENT))


def test_download_file_fetches_and_verifies(tmp_path: Path, served_url: str) -> None:
    destination = tmp_path / "out" / "file.bin"
    result = download_file(_spec(served_url), destination, progress=False)
    assert result == destination
    assert destination.read_bytes() == CONTENT
    assert not destination.with_name("file.bin.part").exists()


def test_download_file_is_idempotent_once_verified(
    tmp_path: Path, served_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "file.bin"
    download_file(_spec(served_url), destination, progress=False)

    def _must_not_be_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a verified file must not trigger a new request")

    monkeypatch.setattr("musicsim.datasets.download.urllib.request.urlopen", _must_not_be_called)
    download_file(_spec(served_url), destination, progress=False)  # no error means no request


def test_download_file_restarts_when_a_partial_file_is_present(
    tmp_path: Path, served_url: str
) -> None:
    destination = tmp_path / "file.bin"
    part = destination.with_name("file.bin.part")
    part.write_bytes(CONTENT[:500_000])

    download_file(_spec(served_url), destination, progress=False)
    assert destination.read_bytes() == CONTENT
    assert not part.exists()


def test_download_file_rejects_a_wrong_checksum(tmp_path: Path, served_url: str) -> None:
    destination = tmp_path / "file.bin"
    with pytest.raises(DownloadError, match="checksum mismatch"):
        download_file(
            _spec(served_url, sha1="0" * 40),
            destination,
            max_retries=1,
            retry_delay=0.0,
            progress=False,
        )
    assert not destination.exists()


def test_download_file_gives_up_after_max_retries_on_a_connection_error(tmp_path: Path) -> None:
    bad_spec = DownloadSpec(
        url="http://127.0.0.1:1/nope", filename="file.bin", sha1=SHA1, size_bytes=1
    )
    with pytest.raises(DownloadError, match="after 2 attempt"):
        download_file(
            bad_spec, tmp_path / "file.bin", max_retries=2, retry_delay=0.0, progress=False
        )


def test_ensure_free_space_passes_when_there_is_room(tmp_path: Path) -> None:
    ensure_free_space(tmp_path, required_bytes=1)


def test_ensure_free_space_fails_when_there_is_not(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="not enough free space"):
        ensure_free_space(tmp_path, required_bytes=10**18)


def test_sha1_of_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(CONTENT)
    assert sha1_of(path) == SHA1
