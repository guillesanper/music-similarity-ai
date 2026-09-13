"""The single source of truth for every path the project reads or writes.

No other module builds a path by hand. Two environment variables move the two
roots without touching any code, which is what lets the university server read
the corpora from a different disk:

``MUSICSIM_DATA_DIR``
    Root of the dataset directories. Default: ``<repo>/datasets``.
``MUSICSIM_OUTPUT_DIR``
    Root of everything the code produces. Default: ``<repo>/outputs``.

Layout under the data root (one directory per dataset, as required by the
repository guidelines)::

    <data>/fma/raw/fma_small/000/000002.mp3
    <data>/fma/raw/fma_metadata/tracks.csv
    <data>/fma/index.csv
    <data>/mtt/raw/...
    <data>/mtt/index.csv

Layout under the output root::

    <out>/cache/<dataset>/<artifact>-<hash8>/      content-addressed artefacts
    <out>/runs/<experiment>/<timestamp>_<hash8>/   one directory per run
    <out>/checkpoints/<model>/                     trained weights

Nothing under the output root is versioned; see ``.gitignore``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = [
    "ENV_DATA_DIR",
    "ENV_OUTPUT_DIR",
    "REPORT_DIR",
    "REPO_ROOT",
    "cache_dir",
    "checkpoints_dir",
    "data_root",
    "dataset_dir",
    "dataset_index",
    "output_root",
    "raw_dir",
    "relative_to_repo",
    "report_figure_dir",
    "report_table_dir",
    "run_dir",
    "runs_root",
    "slugify",
]

#: Repository root, i.e. the directory that contains ``pyproject.toml``.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where the LaTeX report lives. Exported figures and tables land under it.
REPORT_DIR = REPO_ROOT / "report"

#: Environment variables that move the two roots without touching any code.
ENV_DATA_DIR = "MUSICSIM_DATA_DIR"
ENV_OUTPUT_DIR = "MUSICSIM_OUTPUT_DIR"

_SLUG_RE = re.compile(r"[^0-9a-zA-Z._-]+")


def slugify(name: str) -> str:
    """Return ``name`` reduced to a safe single path segment.

    Used for experiment names and artefact names so that a configuration value
    can never escape its directory or produce an unwritable file name.
    """
    slug = _SLUG_RE.sub("-", str(name).strip()).strip("-._")
    if not slug:
        raise ValueError(f"name {name!r} does not contain any usable character")
    return slug


def data_root() -> Path:
    """Root of the dataset directories (``$MUSICSIM_DATA_DIR``)."""
    return Path(os.environ.get(ENV_DATA_DIR) or REPO_ROOT / "datasets").expanduser().resolve()


def output_root() -> Path:
    """Root of everything the code writes (``$MUSICSIM_OUTPUT_DIR``)."""
    return Path(os.environ.get(ENV_OUTPUT_DIR) or REPO_ROOT / "outputs").expanduser().resolve()


def dataset_dir(dataset: str) -> Path:
    """Directory of one dataset, e.g. ``<data>/fma``."""
    return data_root() / slugify(dataset)


def raw_dir(dataset: str) -> Path:
    """Where the downloaded, never-versioned files of a dataset live."""
    return dataset_dir(dataset) / "raw"


def dataset_index(dataset: str) -> Path:
    """CSV index of a dataset: one row per item, standard columns plus labels."""
    return dataset_dir(dataset) / "index.csv"


def cache_dir(dataset: str, artifact: str, config_hash: str, *, create: bool = False) -> Path:
    """Content-addressed cache directory for an expensive artefact.

    The hash is the stable hash of the configuration that produced the
    artefact, so a changed parameter yields a different directory and an
    unchanged one is reused instead of recomputed.
    """
    path = output_root() / "cache" / slugify(dataset) / f"{slugify(artifact)}-{_short(config_hash)}"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def runs_root() -> Path:
    """Root of the per-run directories."""
    return output_root() / "runs"


def run_dir(experiment: str, timestamp: str, config_hash: str, *, create: bool = False) -> Path:
    """Directory of a single run: ``<out>/runs/<experiment>/<timestamp>_<hash8>``."""
    path = runs_root() / slugify(experiment) / f"{slugify(timestamp)}_{_short(config_hash)}"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoints_dir(model: str, *, create: bool = False) -> Path:
    """Where training writes the weights of one model."""
    path = output_root() / "checkpoints" / slugify(model)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def report_figure_dir(*, generated: bool = True) -> Path:
    """``report/figures/generated`` (exported by code) or ``.../static`` (hand-made)."""
    return REPORT_DIR / "figures" / ("generated" if generated else "static")


def report_table_dir() -> Path:
    """``report/tables/generated``: LaTeX tables written by ``musicsim export``."""
    return REPORT_DIR / "tables" / "generated"


def relative_to_repo(path: Path | str) -> str:
    """Path relative to the repository root when it is inside it, for log messages."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    return str(resolved)


def _short(config_hash: str) -> str:
    """First 8 hexadecimal characters of a configuration hash."""
    text = str(config_hash)
    if not re.fullmatch(r"[0-9a-fA-F]{8,}", text):
        raise ValueError(f"{config_hash!r} is not a hexadecimal hash of at least 8 characters")
    return text[:8].lower()
