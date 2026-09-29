"""Tests for the retrieval gallery: ``fma_test`` restricts both queries and
candidates to the test split (the evaluation principal of C.5); ``fma_all``
is the explicit, unrestricted override kept for the secondary analysis and
the archived pipeline's regression figure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import musicsim.evaluation
import musicsim.graphs
import musicsim.relevance  # noqa: F401  (registers the "label_match" relevance function)
from musicsim import paths
from musicsim.config import Config
from musicsim.evaluation.retrieval import gallery_mask
from musicsim.experiments import run_experiment
from musicsim.extractors.base import Extractor
from musicsim.registry import register_extractor

# item_id -> (angle_degrees, genre, split, artist_id). Item 11 (training) sits
# only 1 degree from item 1, closer than item 1's real test-split neighbour
# (item 2, 10 degrees away) — a plant that only leaks into the ranking if the
# gallery restriction has a hole in it.
_ITEMS: dict[int, tuple[float, str, str, int]] = {
    1: (0.0, "A", "test", 101),
    2: (10.0, "A", "test", 102),
    3: (170.0, "B", "test", 103),
    4: (180.0, "B", "test", 104),
    11: (1.0, "B", "training", 111),
}


@register_extractor("toy_gallery_ret")
class _ToyGalleryExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        angle = np.radians(_ITEMS[item_id][0])
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)


@pytest.fixture
def gallery_config(isolated_roots: dict[str, Path]) -> Config:
    paths.dataset_dir("toy_gallery").mkdir(parents=True, exist_ok=True)
    item_ids = sorted(_ITEMS)
    index = pd.DataFrame(
        {
            "item_id": item_ids,
            "path": [f"synthetic/{i:02d}.wav" for i in item_ids],
            "split": [_ITEMS[i][2] for i in item_ids],
            "artist_id": [_ITEMS[i][3] for i in item_ids],
            "group_id": item_ids,
            "genre_top": [_ITEMS[i][1] for i in item_ids],
            "genres_all": ["[]"] * len(item_ids),
        }
    )
    index.to_csv(paths.dataset_index("toy_gallery"), index=False)

    return Config(
        data={
            "experiment": "e00_toy_gallery",
            "seed": 0,
            "dataset": {"directory": "toy_gallery"},
            "extractor": {"name": "toy_gallery_ret", "dim": 2},
            "embedding": {
                "fit_on": "training",
                "standardize": False,
                "l2_normalize": True,
                "pca": {"enabled": False, "n_components": None},
            },
            "graph": {
                "builder": "knn",
                "k": 1,
                "exclude_self": True,
                "near_duplicate_threshold": 0.9999,
            },
            "retrieval": {
                "ks": [1],
                "gallery": "fma_test",
                "queries": "all",
                # "dedup" is included (not just "raw") so retrieval_per_genre.csv,
                # whose headline variant is "dedup", is non-empty to check below.
                "variants": ["raw", "dedup"],
                "relevance": "label_match",
            },
            "bootstrap": {"n_resamples": 50, "confidence": 0.95},
            "evaluation": {
                "retrieval": {
                    "enabled": True,
                    "galleries": ["fma_test", "fma_all"],
                    "queries": ["all"],
                    "baselines": [],
                },
            },
            "runtime": {"n_jobs": 1, "progress": False, "cache": True},
            "stages": ["extract", "embed", "graph", "evaluate"],
        }
    )


# --- gallery_mask --------------------------------------------------------------
def test_gallery_mask_selects_the_test_split() -> None:
    index_common = pd.DataFrame({"split": ["training", "test", "test", "validation"]})
    np.testing.assert_array_equal(
        gallery_mask(index_common, "fma_test"), [False, True, True, False]
    )


def test_gallery_mask_fma_all_selects_everything() -> None:
    index_common = pd.DataFrame({"split": ["training", "test", "validation"]})
    np.testing.assert_array_equal(gallery_mask(index_common, "fma_all"), [True, True, True])


def test_gallery_mask_rejects_an_unknown_gallery() -> None:
    with pytest.raises(ValueError, match="fma_test"):
        gallery_mask(pd.DataFrame({"split": ["test"]}), "validation")


# --- end-to-end gallery restriction --------------------------------------------
def test_fma_test_gallery_excludes_training_items_entirely(gallery_config: Config) -> None:
    run = run_experiment(gallery_config)
    per_query = pd.read_csv(run.metrics_dir / "per_query.csv")

    test_rows = per_query[per_query["gallery"] == "fma_test"]
    # No query and no candidate reachable through it can come from outside the
    # test split: every item_id under this gallery is one of the four test
    # items, dedup included.
    assert set(test_rows["item_id"]) == {1, 2, 3, 4}

    # Item 11 (training) sits closer to item 1 than item 1's real test-split
    # neighbour (item 2). It cannot leak in: item 2 (same genre) is the top-1
    # neighbour, so P@1 is exactly 1.
    p1_test = test_rows[
        (test_rows["item_id"] == 1) & (test_rows["metric"] == "P") & (test_rows["k"] == 1)
    ]["value"].iloc[0]
    assert p1_test == pytest.approx(1.0)


def test_fma_all_gallery_lets_the_closer_training_item_leak_in(gallery_config: Config) -> None:
    run = run_experiment(gallery_config)
    per_query = pd.read_csv(run.metrics_dir / "per_query.csv")

    all_rows = per_query[per_query["gallery"] == "fma_all"]
    assert 11 in set(all_rows["item_id"])  # opted in explicitly: no restriction

    # Under fma_all, item 11 (genre B, closer than item 2) becomes item 1's
    # top-1 neighbour, which is *not* relevant: P@1 drops to 0. This is
    # exactly the leak fma_test exists to prevent, and why fma_all stays an
    # explicit override rather than the default.
    p1_all = all_rows[
        (all_rows["item_id"] == 1) & (all_rows["metric"] == "P") & (all_rows["k"] == 1)
    ]["value"].iloc[0]
    assert p1_all == pytest.approx(0.0)


def test_retrieval_outputs_carry_a_gallery_column(gallery_config: Config) -> None:
    run = run_experiment(gallery_config)
    retrieval = pd.read_csv(run.metrics_dir / "retrieval.csv")
    per_genre = pd.read_csv(run.metrics_dir / "retrieval_per_genre.csv")
    per_query = pd.read_csv(run.metrics_dir / "per_query.csv")

    assert set(retrieval["gallery"]) == {"fma_test", "fma_all"}
    assert set(per_query["gallery"]) == {"fma_test", "fma_all"}
    # retrieval_per_genre.csv only ever reports the headline P@10/dedup slice
    # (see _summarize_per_genre): this toy gallery is too small for k=10, so
    # it is empty here, but the column itself must still be declared.
    assert "gallery" in per_genre.columns


def test_fma_test_gallery_item_count_and_artists_match_the_test_split(
    gallery_config: Config,
) -> None:
    run = run_experiment(gallery_config)
    per_query = pd.read_csv(run.metrics_dir / "per_query.csv")
    test_rows = per_query[per_query["gallery"] == "fma_test"]

    index = pd.read_csv(paths.dataset_index("toy_gallery"))
    test_ids = set(index.loc[index["split"] == "test", "item_id"])
    assert set(test_rows["item_id"]) == test_ids

    test_artists = set(index.loc[index["split"] == "test", "artist_id"])
    other_artists = set(index.loc[index["split"] != "test", "artist_id"])
    assert not (test_artists & other_artists)
