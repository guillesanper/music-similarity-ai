"""Run directories and provenance.

Every execution leaves a directory behind::

    outputs/runs/<experiment>/<timestamp>_<hash8>/
        config.resolved.yaml   the configuration after composition and overrides
        run.json               provenance: git commit, dirty flag, versions,
                               seed, platform, timings, stage status
        log.txt                everything the run logged
        metrics/*.csv          the numbers
        figures/*.png|pdf      the plots

The point of ``run.json`` is that a number in the report can be traced back to
the exact code and configuration that produced it. In particular ``git.dirty``
records whether the working tree had uncommitted changes: a result produced from
a dirty tree is not reproducible from the commit alone, and the report must not
claim otherwise.
"""

from __future__ import annotations

import contextlib
import json
import logging
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from musicsim import paths
from musicsim.config import Config

__all__ = ["RunLog", "git_state", "package_versions", "platform_state", "start_run"]

LOGGER_NAME = "musicsim"

#: Packages whose exact version is recorded in every run. A missing one is
#: recorded as ``null`` rather than omitted, so the absence is visible.
TRACKED_PACKAGES: tuple[str, ...] = (
    "musicsim",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "librosa",
    "soundfile",
    "numba",
    "joblib",
    "matplotlib",
    "networkx",
    "PyYAML",
    "torch",
    "torchaudio",
    "transformers",
)


@dataclass
class RunLog:
    """The directory of one run, plus the provenance collected while it runs."""

    directory: Path
    config: Config
    started_at: float
    record: dict[str, Any]
    logger: logging.Logger
    _handler: logging.Handler | None = field(default=None, repr=False)

    # -- sub-directories -------------------------------------------------
    @property
    def metrics_dir(self) -> Path:
        """``metrics/`` inside the run directory, created on first use."""
        return self._subdir("metrics")

    @property
    def figures_dir(self) -> Path:
        """``figures/`` inside the run directory, created on first use."""
        return self._subdir("figures")

    def _subdir(self, name: str) -> Path:
        path = self.directory / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- timings ---------------------------------------------------------
    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one stage, log its start and end, and record the outcome.

        A stage that raises is recorded as failed, with the exception type and
        message, before the exception propagates: a crashed run still leaves a
        ``run.json`` that says how far it got.
        """
        self.logger.info("stage %s: start", name)
        began = time.perf_counter()
        entry: dict[str, Any] = {"stage": name, "status": "running"}
        self.record.setdefault("stages", []).append(entry)
        try:
            yield
        except BaseException as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["seconds"] = round(time.perf_counter() - began, 3)
            self.logger.error("stage %s: failed after %.1fs", name, entry["seconds"])
            self.write()
            raise
        else:
            entry["status"] = "ok"
            entry["seconds"] = round(time.perf_counter() - began, 3)
            self.logger.info("stage %s: done in %.1fs", name, entry["seconds"])
            self.write()

    # -- writing ---------------------------------------------------------
    def write(self) -> Path:
        """Write ``run.json``. Called after every stage so a crash keeps the trace."""
        self.record["total_seconds"] = round(time.perf_counter() - self.started_at, 3)
        path = self.directory / "run.json"
        path.write_text(
            json.dumps(self.record, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    def finish(self, status: str = "ok") -> Path:
        """Close the run: record the status, flush the log, detach the handler."""
        self.record["status"] = status
        self.record["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        path = self.write()
        if self._handler is not None:
            self.logger.removeHandler(self._handler)
            self._handler.close()
            self._handler = None
        return path


def start_run(
    config: Config,
    *,
    seed_record: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    level: int = logging.INFO,
) -> RunLog:
    """Create the run directory, write ``config.resolved.yaml``, start logging.

    The directory name carries the timestamp and the first eight characters of
    the configuration hash, so two runs of the same configuration sort together
    and two different configurations can never collide in one directory.
    """
    config_hash = config.hash()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = paths.run_dir(config.name, timestamp, config_hash, create=True)

    config.write(directory / "config.resolved.yaml")

    logger = _configure_logger(directory / "log.txt", level=level)

    record: dict[str, Any] = {
        "experiment": config.name,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_hash": config_hash,
        "config_source": paths.relative_to_repo(config.source) if config.source else None,
        "command": " ".join(sys.argv),
        "git": git_state(),
        "packages": package_versions(),
        "platform": platform_state(),
        "seeding": seed_record or {},
        "data_root": str(paths.data_root()),
        "output_root": str(paths.output_root()),
        "stages": [],
    }
    if extra:
        record.update(extra)

    run = RunLog(
        directory=directory,
        config=config,
        started_at=time.perf_counter(),
        record=record,
        logger=logger,
        _handler=logger.handlers[-1] if logger.handlers else None,
    )
    logger.info("run directory: %s", paths.relative_to_repo(directory))
    if record["git"].get("dirty"):
        logger.warning(
            "the working tree has uncommitted changes; this run is not reproducible "
            "from commit %s alone",
            record["git"].get("commit"),
        )
    run.write()
    return run


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def git_state() -> dict[str, Any]:
    """Current commit, branch and dirty flag, or ``available: false`` without git."""
    state: dict[str, Any] = {"available": False, "commit": None, "branch": None, "dirty": None}
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return state
    state["available"] = True
    state["commit"] = commit
    state["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    state["dirty"] = bool(status) if status is not None else None
    return state


def package_versions() -> dict[str, str | None]:
    """Installed version of every tracked package, ``None`` when not installed."""
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def platform_state() -> dict[str, Any]:
    """Interpreter and machine, enough to explain a numeric difference later."""
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
    }


def _git(*args: str) -> str | None:
    """Run a git command in the repository; return its output, or ``None``."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=paths.REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _configure_logger(log_file: Path, *, level: int) -> logging.Logger:
    """Log to the console and to ``log.txt`` inside the run directory."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console = logging.StreamHandler(stream=sys.stderr)
        console.setFormatter(formatter)
        logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
