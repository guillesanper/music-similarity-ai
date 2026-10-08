"""Tests for musicsim.representations: parsing, validation and aligned loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import musicsim.extractors  # noqa: F401  (registers "random")
from musicsim import paths
from musicsim.config import Config, ConfigError
from musicsim.extractors.base import Extractor
from musicsim.registry import register_extractor
from musicsim.representations import (
    RepresentationSpec,
    load_aligned_representations,
    parse_representations,
    spec_config,
)

_N = 10


def _config(representations: object = None) -> Config:
    data: dict = {
        "seed": 3,
        "dataset": {"directory": "toy_rep"},
        "extractor": {"name": "toy_rep_main", "dim": 4},
        "embedding": {
            "fit_on": "training",
            "standardize": False,
            "l2_normalize": True,
            "pca": {"enabled": False, "n_components": None},
        },
        "runtime": {"n_jobs": 1, "progress": False, "cache": True},
    }
    if representations is not None:
        data["representations"] = representations
    return Config(data=data)


# --- parsing -------------------------------------------------------------------
def test_without_the_key_the_only_representation_is_the_main_extractor() -> None:
    assert parse_representations(_config()) == [
        RepresentationSpec(label="toy_rep_main", extractor="toy_rep_main", overrides={})
    ]


def test_entries_by_name_and_by_mapping_are_accepted() -> None:
    specs = parse_representations(
        _config(
            [
                "random",
                {"label": "random_s1", "extractor": "random", "overrides": {"seed": 1}},
                {"label": "plain", "extractor": "random"},
            ]
        )
    )
    assert [s.label for s in specs] == ["random", "random_s1", "plain"]
    assert specs[1].overrides == {"seed": 1}
    assert specs[2].overrides == {}


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (["random", "random"], "unique"),
        ([{"label": "Bad-Label", "extractor": "random"}], "match"),
        (
            [{"label": "x", "extractor": "random", "overrides": {"dataset.directory": "z"}}],
            "dataset",
        ),
        ([{"label": "x", "extractor": "random", "overrides": {"representations": []}}], "list"),
        ([{"label": "x"}], "extractor"),
        ([], "empty"),
    ],
)
def test_invalid_specifications_are_rejected(entries: list, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_representations(_config(entries))


def test_spec_config_applies_nested_overrides_on_top_of_the_swapped_extractor() -> None:
    spec = RepresentationSpec(
        "r", "random", {"seed": 9, "embedding.pca.enabled": True, "embedding.pca.n_components": 5}
    )
    out = spec_config(_config(), spec)
    assert out.require("extractor.name") == "random"
    assert out.require("seed") == 9
    assert out.require("embedding.pca.n_components") == 5
    assert out.require("dataset.directory") == "toy_rep"


# --- aligned loading -----------------------------------------------------------
@register_extractor("toy_rep_main")
class _Main(Extractor):
    """Fails on the last item, so the other representations have one more id."""

    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        if item_id == _N:
            raise RuntimeError("planted failure")
        return np.array([1.0, item_id, item_id**2 % 7, 2.0], dtype=np.float32)


@pytest.fixture
def toy_index(isolated_roots: dict[str, Path]) -> pd.DataFrame:
    paths.dataset_dir("toy_rep").mkdir(parents=True, exist_ok=True)
    ids = list(range(1, _N + 1))
    index = pd.DataFrame(
        {
            "item_id": ids,
            "path": [f"s/{i}.wav" for i in ids],
            "split": ["training"] * _N,
            "artist_id": ids,
            "group_id": ids,
            "genre_top": ["A"] * _N,
            "genres_all": ["[]"] * _N,
        }
    )
    index.to_csv(paths.dataset_index("toy_rep"), index=False)
    return index


def test_two_seeds_of_random_differ_and_report_their_seeds(toy_index: pd.DataFrame) -> None:
    config = _config(
        [
            {"label": "random_s1", "extractor": "random", "overrides": {"seed": 1}},
            {"label": "random_s2", "extractor": "random", "overrides": {"seed": 2}},
        ]
    )
    aligned = load_aligned_representations(config, toy_index, parse_representations(config))
    assert aligned.seeds == {"random_s1": 1, "random_s2": 2}
    assert not np.allclose(aligned.X["random_s1"], aligned.X["random_s2"])
    assert aligned.ids.tolist() == list(range(1, _N + 1))
    assert aligned.index.index.tolist() == aligned.ids.tolist()


def test_ids_are_the_intersection_and_rows_stay_aligned_by_id(toy_index: pd.DataFrame) -> None:
    config = _config(["toy_rep_main", "random"])
    aligned = load_aligned_representations(config, toy_index, parse_representations(config))
    assert aligned.ids.tolist() == list(range(1, _N))  # the main extractor lost the last item
    assert all(len(X) == _N - 1 for X in aligned.X.values())
    assert aligned.index.index.tolist() == aligned.ids.tolist()

    raw = np.array([1.0, 4, 16 % 7, 2.0])
    row = int(np.flatnonzero(aligned.ids == 4)[0])
    np.testing.assert_allclose(aligned.X["toy_rep_main"][row], raw / np.linalg.norm(raw), rtol=1e-6)
