"""Tests for :mod:`musicsim.paths`.

Two things are being protected here. First, that the roots really do follow the
environment variables, because that is what lets the university server read the
corpora from another disk without a code change. Second, that names coming from
a configuration cannot escape their directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from musicsim import paths

HASH = "a694603e26f87a3e7547e195ce0feb7e4e3ba458e7f81aa446454f7d5ebe3ad0"


# --- roots -------------------------------------------------------------------
def test_the_defaults_live_inside_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(paths.ENV_DATA_DIR, raising=False)
    monkeypatch.delenv(paths.ENV_OUTPUT_DIR, raising=False)
    assert paths.data_root() == paths.REPO_ROOT / "datasets"
    assert paths.output_root() == paths.REPO_ROOT / "outputs"


def test_the_environment_variables_move_the_roots(isolated_roots: dict[str, Path]) -> None:
    assert paths.data_root() == isolated_roots["data"]
    assert paths.output_root() == isolated_roots["output"]
    # Everything derived follows, which is the whole point of the indirection.
    assert paths.dataset_dir("fma").is_relative_to(isolated_roots["data"])
    assert paths.runs_root().is_relative_to(isolated_roots["output"])
    assert paths.cache_dir("fma", "mfcc", HASH).is_relative_to(isolated_roots["output"])


def test_an_empty_environment_variable_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(paths.ENV_DATA_DIR, "")
    assert paths.data_root() == paths.REPO_ROOT / "datasets"


def test_the_repository_root_is_the_one_holding_pyproject() -> None:
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
    assert (paths.REPO_ROOT / "musicsim" / "__init__.py").is_file()


# --- dataset layout -----------------------------------------------------------
def test_each_dataset_gets_its_own_directory(isolated_roots: dict[str, Path]) -> None:
    fma = paths.dataset_dir("fma")
    mtt = paths.dataset_dir("mtt")
    assert fma != mtt
    assert fma.parent == mtt.parent == isolated_roots["data"]
    assert paths.raw_dir("fma") == fma / "raw"
    assert paths.dataset_index("fma") == fma / "index.csv"


# --- cache and runs ------------------------------------------------------------
def test_the_cache_directory_carries_eight_hash_characters(
    isolated_roots: dict[str, Path],
) -> None:
    directory = paths.cache_dir("fma", "mfcc", HASH)
    assert directory.name == f"mfcc-{HASH[:8]}"
    assert directory.parent.name == "fma"


def test_different_configurations_get_different_cache_directories(
    isolated_roots: dict[str, Path],
) -> None:
    other = "b" + HASH[1:]
    assert paths.cache_dir("fma", "mfcc", HASH) != paths.cache_dir("fma", "mfcc", other)
    # The same configuration reuses the directory, which is what makes the
    # cache a cache rather than a growing pile of directories.
    assert paths.cache_dir("fma", "mfcc", HASH) == paths.cache_dir("fma", "mfcc", HASH)


def test_a_run_directory_carries_the_timestamp_and_the_hash(
    isolated_roots: dict[str, Path],
) -> None:
    directory = paths.run_dir("e01_mfcc_baseline", "20260912-140000", HASH)
    assert directory.name == f"20260912-140000_{HASH[:8]}"
    assert directory.parent.name == "e01_mfcc_baseline"
    assert directory.parent.parent == paths.runs_root()


def test_create_makes_the_directory_and_omitting_it_does_not(
    isolated_roots: dict[str, Path],
) -> None:
    assert not paths.cache_dir("fma", "mfcc", HASH).exists()
    created = paths.cache_dir("fma", "mfcc", HASH, create=True)
    assert created.is_dir()


@pytest.mark.parametrize("bad", ["", "short", "not-hex-at-all!!", "ZZZZZZZZ"])
def test_a_value_that_is_not_a_hash_is_rejected(bad: str, isolated_roots: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="hexadecimal"):
        paths.cache_dir("fma", "mfcc", bad)


# --- slugs ---------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("fma_small", "fma_small"),
        ("e01 mfcc baseline", "e01-mfcc-baseline"),
        ("MFCC/PCA 32", "MFCC-PCA-32"),
        ("  padded  ", "padded"),
    ],
)
def test_slugify_keeps_readable_names(raw: str, expected: str) -> None:
    assert paths.slugify(raw) == expected


@pytest.mark.parametrize("bad", ["", "   ", "/", "..", "///"])
def test_slugify_rejects_names_that_would_escape_the_directory(bad: str) -> None:
    with pytest.raises(ValueError):
        paths.slugify(bad)


def test_a_traversal_attempt_stays_inside_the_data_root(isolated_roots: dict[str, Path]) -> None:
    # A dataset name is configuration data, so it must not be able to walk out.
    directory = paths.dataset_dir("../../etc")
    assert directory.is_relative_to(isolated_roots["data"])


# --- report exports --------------------------------------------------------------
def test_generated_and_static_figures_are_kept_apart() -> None:
    generated = paths.report_figure_dir(generated=True)
    static = paths.report_figure_dir(generated=False)
    assert generated.name == "generated"
    assert static.name == "static"
    assert generated.parent == static.parent == paths.REPORT_DIR / "figures"
    assert paths.report_table_dir() == paths.REPORT_DIR / "tables" / "generated"


def test_relative_to_repo_shortens_inside_paths_and_leaves_outside_ones_alone(
    tmp_path: Path,
) -> None:
    inside = paths.REPO_ROOT / "configs" / "base.yaml"
    assert paths.relative_to_repo(inside) == "configs/base.yaml"
    outside = (tmp_path / "elsewhere.txt").resolve()
    assert paths.relative_to_repo(outside) == str(outside)
