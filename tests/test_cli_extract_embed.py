"""Tests for the `extract` and `embed` subcommands of musicsim.cli.

A throwaway "toy" extractor stands in for MFCC so these run without real
audio; :mod:`tests.test_extractors_mfcc` covers MFCC's own numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from musicsim import paths
from musicsim.cli import main
from musicsim.extractors.base import Extractor
from musicsim.registry import register_extractor

_CALLS: list[int] = []


@register_extractor("cli_toy")
class _ToyExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        _CALLS.append(item_id)
        return np.array([float(item_id), float(item_id) * 2.0], dtype=np.float32)


@pytest.fixture
def toy_config(tmp_path: Path, isolated_roots: dict[str, Path]) -> Path:
    _CALLS.clear()
    paths.dataset_dir("toy").mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame(
        {
            "item_id": [1, 2, 3, 4],
            "path": ["a.wav", "b.wav", "c.wav", "d.wav"],
            "split": ["training", "training", "validation", "test"],
            "artist_id": [1, 2, 3, 4],
            "group_id": [1, 2, 3, 4],
            "genre_top": ["rock", "pop", "rock", "pop"],
            "genres_all": ["[1]", "[2]", "[1]", "[2]"],
        }
    )
    index.to_csv(paths.dataset_index("toy"), index=False)

    config_path = tmp_path / "e00_toy.yaml"
    config_path.write_text(
        "seed: 42\n"
        "dataset:\n  directory: toy\n"
        "extractor:\n  name: cli_toy\n  dim: 2\n"
        "embedding:\n"
        "  fit_on: training\n"
        "  standardize: true\n"
        "  l2_normalize: true\n"
        "  pca:\n    enabled: false\n    n_components: null\n"
        "runtime:\n  n_jobs: 1\n  progress: false\n  cache: true\n",
        encoding="utf-8",
    )
    return config_path


def test_extract_writes_features_and_reports_the_count(
    toy_config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["extract", "--config", str(toy_config)]) == 0
    out = capsys.readouterr().out
    assert "extracted 4 items (0 failed)" in out
    assert len(_CALLS) == 4


def test_extract_second_run_uses_the_cache(toy_config: Path) -> None:
    assert main(["extract", "--config", str(toy_config)]) == 0
    n_calls_after_first_run = len(_CALLS)
    assert main(["extract", "--config", str(toy_config)]) == 0
    assert len(_CALLS) == n_calls_after_first_run  # no new extract_one calls


def test_embed_without_extract_first_is_a_usage_error(toy_config: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["embed", "--config", str(toy_config)])
    assert excinfo.value.code == 2


def test_embed_writes_l2_normalized_embeddings(toy_config: Path) -> None:
    assert main(["extract", "--config", str(toy_config)]) == 0
    assert main(["embed", "--config", str(toy_config)]) == 0

    from musicsim.config import load_config

    config = load_config(toy_config)
    embed_dir = paths.cache_dir("toy", "embeddings-cli_toy", config.hash())
    Z = np.load(embed_dir / "embeddings.npy")
    ids = np.load(embed_dir / "ids.npy")

    assert Z.shape == (4, 2)
    assert set(ids.tolist()) == {1, 2, 3, 4}
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-3)


def test_extract_records_failures_without_aborting(
    tmp_path: Path, isolated_roots: dict[str, Path]
) -> None:
    @register_extractor("flaky_toy")
    class _FlakyToyExtractor(Extractor):
        def extract_one(self, item_id: int, path: str) -> np.ndarray:
            if item_id == 2:
                raise ValueError("boom")
            return np.zeros(self.dim, dtype=np.float32)

    paths.dataset_dir("toy").mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame(
        {
            "item_id": [1, 2, 3],
            "path": ["a.wav", "b.wav", "c.wav"],
            "split": ["training", "training", "test"],
            "artist_id": [1, 2, 3],
            "group_id": [1, 2, 3],
            "genre_top": ["rock", "pop", "rock"],
            "genres_all": ["[1]", "[2]", "[1]"],
        }
    )
    index.to_csv(paths.dataset_index("toy"), index=False)

    config_path = tmp_path / "e00_flaky.yaml"
    config_path.write_text(
        "seed: 42\n"
        "dataset:\n  directory: toy\n"
        "extractor:\n  name: flaky_toy\n  dim: 2\n"
        "runtime:\n  n_jobs: 1\n  progress: false\n  cache: true\n",
        encoding="utf-8",
    )

    assert main(["extract", "--config", str(config_path)]) == 0

    from musicsim.config import load_config

    config = load_config(config_path)
    cache_dir = paths.cache_dir("toy", "features-flaky_toy", config.hash())
    import json

    failures = json.loads((cache_dir / "failures.json").read_text(encoding="utf-8"))
    assert [f["item_id"] for f in failures] == [2]
    assert np.load(cache_dir / "ids.npy").shape == (2,)
