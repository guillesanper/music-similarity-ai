"""Tests for ``graph_metrics`` and the structure evaluator, on graphs small enough
to have hand-computable answers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import musicsim.evaluation
import musicsim.graphs
import musicsim.relevance  # noqa: F401  (registers "label_match")
from musicsim import paths
from musicsim.config import Config
from musicsim.evaluation.graph_evaluator import graph_metrics
from musicsim.experiments import run_experiment
from musicsim.extractors.base import Extractor
from musicsim.graphs.base import Graph
from musicsim.registry import register_extractor


def _graph(neighbors: list[list[int]]) -> Graph:
    nb = np.array(neighbors, dtype=np.int32)
    return Graph(
        neighbors=nb,
        weights=np.full(nb.shape, 0.5, dtype=np.float32),
        ids=np.arange(len(nb)),
    )


# 0 -> 1, 1 -> 0 (mutual); 2 -> 0, 3 -> 0 (point at node 0); node 2 and 3 unreferenced.
STAR = _graph([[1], [0], [0], [0]])


def test_graph_metrics_on_star() -> None:
    # N_k = [3, 1, 0, 0]: mean 1, std sqrt(1.5), skewness 1 / sqrt(1.5); threshold 2 -> one hub.
    # Union edges are 0-1, 0-2, 0-3: a star without triangles.
    metrics = graph_metrics(STAR, np.array([0, 0, 1, 0]))
    assert metrics == pytest.approx(
        {
            "k_occurrence_skewness": 1 / np.sqrt(1.5),
            "label_homophily": 0.75,
            "antihub_fraction": 0.5,
            "hub_fraction": 0.25,
            "max_k_occurrence": 3.0,
            "reciprocity": 0.5,
            "mean_clustering": 0.0,
            "giant_fraction": 1.0,
            "n_components": 1.0,
        }
    )


def test_graph_metrics_rejects_edgeless_graph() -> None:
    empty = Graph(
        neighbors=np.full((3, 2), -1), weights=np.zeros((3, 2), dtype=np.float32), ids=np.arange(3)
    )
    with pytest.raises(ValueError, match="no edges"):
        graph_metrics(empty, np.zeros(3, dtype=int))


# --- end to end ----------------------------------------------------------------
_ANGLES = {1: 0.0, 2: 10.0, 3: 20.0, 4: 170.0, 5: 180.0, 6: 190.0}
_GENRE = {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B"}


@register_extractor("toy_structure")
class _ToyStructureExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        angle = np.radians(_ANGLES[item_id])
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)


@pytest.fixture
def structure_config(isolated_roots: dict[str, Path]) -> Config:
    paths.dataset_dir("toy_structure").mkdir(parents=True, exist_ok=True)
    ids = sorted(_ANGLES)
    pd.DataFrame(
        {
            "item_id": ids,
            "path": [f"synthetic/{i}.wav" for i in ids],
            "split": ["test"] * len(ids),
            "artist_id": [100 + i for i in ids],
            "group_id": ids,
            "genre_top": [_GENRE[i] for i in ids],
            "genres_all": ["[]"] * len(ids),
        }
    ).to_csv(paths.dataset_index("toy_structure"), index=False)
    return Config(
        data={
            "experiment": "e00_toy_structure",
            "seed": 0,
            "dataset": {"directory": "toy_structure"},
            "extractor": {"name": "toy_structure", "dim": 2},
            "embedding": {
                "fit_on": "test",
                "standardize": False,
                "l2_normalize": True,
                "pca": {"enabled": False, "n_components": None},
            },
            "graph": {
                "builder": "knn",
                "k": 2,
                "exclude_self": True,
                "near_duplicate_threshold": 0.9999,
            },
            "retrieval": {"gallery": "fma_all", "relevance": "label_match"},
            "evaluation": {
                "retrieval": {"enabled": False},
                "structure": {"enabled": True, "galleries": ["fma_all"], "baselines": []},
            },
            "runtime": {"n_jobs": 1, "progress": False, "cache": True},
            "stages": ["extract", "embed", "graph", "evaluate"],
        }
    )


def test_evaluator_writes_structure_and_hubness(structure_config: Config) -> None:
    run = run_experiment(structure_config)
    structure = pd.read_csv(run.metrics_dir / "structure.csv")
    values = structure.set_index("metric")["value"]
    # Two well-separated 3-item clusters with k=2: every edge stays in its genre,
    # and the graph splits into the two clusters.
    assert values["label_homophily"] == pytest.approx(1.0)
    assert values["n_components"] == 2.0
    assert values["giant_fraction"] == 0.5

    hubness = pd.read_csv(run.metrics_dir / "hubness.csv")
    assert len(hubness) == 6
    assert hubness["k_occurrence"].sum() == 6 * 2  # every node lists k=2 neighbours


def test_evaluator_is_off_by_default(structure_config: Config) -> None:
    data = structure_config.to_dict()
    del data["evaluation"]["structure"]
    run = run_experiment(Config(data=data, source=structure_config.source))
    assert not (run.metrics_dir / "structure.csv").exists()
