"""Tests for the ``download`` and ``index`` subcommands of :mod:`musicsim.cli`.

A throwaway dataset class stands in for FMA so these run without network
access; :mod:`tests.test_datasets_fma` covers the real corpus's own rules.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from musicsim import paths
from musicsim.cli import main
from musicsim.datasets.base import Dataset
from musicsim.registry import register_dataset


@register_dataset("toy")
class _ToyDataset(Dataset):
    downloaded: bool = False

    def download(self, *, force: bool = False) -> None:
        _ToyDataset.downloaded = True

    def build_index(self) -> pd.DataFrame:
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


@pytest.fixture
def toy_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``musicsim.paths.REPO_ROOT`` at a temp tree with configs/datasets/toy.yaml."""
    (tmp_path / "configs" / "datasets").mkdir(parents=True)
    (tmp_path / "configs" / "datasets" / "toy.yaml").write_text(
        "dataset:\n  name: toy\n  directory: toy\n", encoding="utf-8"
    )
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    return tmp_path


def test_download_dashdataset_resolves_the_config_by_convention(
    isolated_roots: dict[str, Path], toy_config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _ToyDataset.downloaded = False
    assert main(["download", "--dataset", "toy"]) == 0
    assert _ToyDataset.downloaded is True
    assert "toy" in capsys.readouterr().out


def test_index_dashdataset_writes_the_index_and_reports_the_count(
    isolated_roots: dict[str, Path], toy_config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["index", "--dataset", "toy"]) == 0
    out = capsys.readouterr().out
    assert "2 items" in out
    assert paths.dataset_index("toy").is_file()
    assert len(pd.read_csv(paths.dataset_index("toy"))) == 2


def test_index_with_config_bypasses_the_dataset_convention(
    isolated_roots: dict[str, Path], tmp_path: Path
) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text("dataset:\n  name: toy\n  directory: toy\n", encoding="utf-8")
    assert main(["index", "--config", str(config_path)]) == 0
    assert paths.dataset_index("toy").is_file()


def test_neither_config_nor_dataset_is_a_usage_error(isolated_roots: dict[str, Path]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["index"])
    assert excinfo.value.code == 2


def test_an_unknown_dataset_name_is_a_usage_error(isolated_roots: dict[str, Path]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["index", "--dataset", "does-not-exist"])
    assert excinfo.value.code == 2


def test_an_unregistered_directory_lists_whats_available(
    isolated_roots: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "configs" / "datasets").mkdir(parents=True)
    (tmp_path / "configs" / "datasets" / "ghost.yaml").write_text(
        "dataset:\n  name: ghost\n  directory: does_not_exist\n", encoding="utf-8"
    )
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(["index", "--dataset", "ghost"])
    assert excinfo.value.code == 2
