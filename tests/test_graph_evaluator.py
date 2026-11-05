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
from musicsim.config import Config, ConfigError
from musicsim.evaluation.graph_evaluator import METRIC_VIEWS, PAIR_METRICS, graph_metrics
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
    metrics = graph_metrics(STAR, np.array([0, 0, 1, 0]), seed=0, n_louvain_seeds=1, resolution=1.0)
    # A star has a single optimal Louvain partition (everything in one community),
    # so its modularity is 0 and its NMI with a non-trivial labelling is 0.
    assert np.isnan(metrics.pop("louvain_stability"))  # one seed: no pair to compare
    assert metrics == pytest.approx(
        {
            "nmi_louvain_genre": 0.0,
            "modularity": 0.0,
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
    # The vote metrics need artists, which graph_metrics does not take; the evaluator adds them.
    vote = {"vote_accuracy", "vote_macro_f1", "vote_abstention_fraction"}
    assert set(metrics) | {"louvain_stability"} | vote == set(METRIC_VIEWS)
    assert {METRIC_VIEWS[name] for name in vote} == {"directed"}


def test_graph_metrics_rejects_edgeless_graph() -> None:
    empty = Graph(
        neighbors=np.full((3, 2), -1), weights=np.zeros((3, 2), dtype=np.float32), ids=np.arange(3)
    )
    with pytest.raises(ValueError, match="no edges"):
        graph_metrics(empty, np.zeros(3, dtype=int), seed=0, n_louvain_seeds=1, resolution=1.0)


# --- end to end ----------------------------------------------------------------
_ANGLES = {1: 0.0, 2: 10.0, 3: 20.0, 4: 170.0, 5: 180.0, 6: 190.0}
_GENRE = {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B", 7: "A"}
# Item 7 is an exact copy of item 1: a near-duplicate, collapsed when it is present.
_ALL_ANGLES = {**_ANGLES, 7: 0.0}


@register_extractor("toy_structure")
class _ToyStructureExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        angle = np.radians(_ALL_ANGLES[item_id])
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)


def _config(ids: list[int], **structure: object) -> Config:
    paths.dataset_dir("toy_structure").mkdir(parents=True, exist_ok=True)
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
                "structure": {
                    "enabled": True,
                    "galleries": ["fma_all"],
                    "baselines": [],
                    **structure,
                },
            },
            "runtime": {"n_jobs": 1, "progress": False, "cache": True},
            "stages": ["extract", "embed", "graph", "evaluate"],
        }
    )


@pytest.fixture
def structure_config(isolated_roots: dict[str, Path]) -> Config:
    return _config(sorted(_ANGLES))


def _with_representations(config: Config, representations: list) -> Config:
    data = config.to_dict()
    data["representations"] = representations
    del data["evaluation"]["structure"]["baselines"]
    return Config(data=data, source=config.source)


def test_evaluator_writes_structure_and_hubness(structure_config: Config) -> None:
    run = run_experiment(structure_config)
    structure = pd.read_csv(run.metrics_dir / "structure.csv")
    values = structure.set_index("metric")["value"]
    # Two well-separated 3-item clusters with k=2: every edge stays in its genre,
    # and the graph splits into the two clusters.
    assert values["label_homophily"] == pytest.approx(1.0)
    assert values["n_components"] == 2.0
    assert values["giant_fraction"] == 0.5
    # Louvain recovers the two clusters, which are the two genres.
    assert values["nmi_louvain_genre"] == pytest.approx(1.0)
    assert values["louvain_stability"] == pytest.approx(1.0)
    assert values["modularity"] == pytest.approx(0.5)
    # Every artist is distinct, so no neighbour is excluded and each node's two
    # neighbours (both of its own genre) vote for it.
    assert values["vote_accuracy"] == 1.0
    assert values["vote_macro_f1"] == 1.0
    assert values["vote_abstention_fraction"] == 0.0

    hubness = pd.read_csv(run.metrics_dir / "hubness.csv")
    assert len(hubness) == 6
    assert hubness["k_occurrence"].sum() == 6 * 2  # every node lists k=2 neighbours


def test_evaluator_is_off_by_default(structure_config: Config) -> None:
    data = structure_config.to_dict()
    del data["evaluation"]["structure"]
    run = run_experiment(Config(data=data, source=structure_config.source))
    assert not (run.metrics_dir / "structure.csv").exists()


def test_output_schema_and_views(structure_config: Config) -> None:
    run = run_experiment(structure_config)
    structure = pd.read_csv(run.metrics_dir / "structure.csv")
    assert list(structure.columns) == [
        "representation",
        "seed",
        "gallery",
        "graph",
        "view",
        "metric",
        "value",
        "ci_low",
        "ci_high",
        "n_replicates",
    ]
    assert sorted(structure["metric"]) == sorted(METRIC_VIEWS)  # one row per metric
    assert (structure["view"] == structure["metric"].map(METRIC_VIEWS)).all()
    assert set(structure["view"]) == {"directed", "union"}
    assert structure["ci_low"].isna().all() and structure["ci_high"].isna().all()
    assert (structure["n_replicates"] == 0).all()
    assert set(structure["graph"]) == {"knn2"} and set(structure["gallery"]) == {"fma_all"}

    hubness = pd.read_csv(run.metrics_dir / "hubness.csv")
    assert list(hubness.columns) == [
        "item_id",
        "representation",
        "seed",
        "gallery",
        "graph",
        "k_occurrence",
    ]


def test_graphs_are_written_and_can_be_read_back(structure_config: Config) -> None:
    run = run_experiment(structure_config)
    folder = run.directory / "graphs" / "toy_structure" / "fma_all" / "knn2"
    assert (folder / "neighbors.npz").is_file() and (folder / "edges.csv").is_file()
    graph = Graph.read(folder)
    hubness = pd.read_csv(run.metrics_dir / "hubness.csv")
    np.testing.assert_array_equal(graph.ids, hubness["item_id"])
    assert len(pd.read_csv(folder / "edges.csv")) == 6 * 2


def test_planted_duplicate_is_collapsed(
    isolated_roots: dict[str, Path], structure_config: Config
) -> None:
    clean = pd.read_csv(run_experiment(structure_config).metrics_dir / "hubness.csv")
    run = run_experiment(_config([*sorted(_ANGLES), 7]))
    hubness = pd.read_csv(run.metrics_dir / "hubness.csv")
    assert len(hubness) == len(clean) == 6  # 7 items, one collapsed
    assert 7 not in set(hubness["item_id"])  # the copy with the larger id is the one dropped


def test_same_representation_under_two_labels_gives_identical_values(
    structure_config: Config,
) -> None:
    config = _with_representations(
        structure_config,
        ["toy_structure", {"label": "copy", "extractor": "toy_structure"}],
    )
    structure = pd.read_csv(run_experiment(config).metrics_dir / "structure.csv")
    by_label = {
        label: frame.set_index("metric")["value"]
        for label, frame in structure.groupby("representation")
    }
    pd.testing.assert_series_equal(by_label["toy_structure"], by_label["copy"], check_names=False)


def test_seed_column_is_the_seed_of_each_representation(structure_config: Config) -> None:
    config = _with_representations(
        structure_config,
        [
            "toy_structure",
            {"label": "random_s1", "extractor": "random", "overrides": {"seed": 1}},
            {"label": "random_s2", "extractor": "random", "overrides": {"seed": 2}},
        ],
    )
    run = run_experiment(config)
    expected = {"toy_structure": [0], "random_s1": [1], "random_s2": [2]}
    for name in ("structure.csv", "hubness.csv"):
        frame = pd.read_csv(run.metrics_dir / name)
        assert frame.groupby("representation")["seed"].unique().map(list).to_dict() == expected


def test_unknown_reference_is_a_config_error(isolated_roots: dict[str, Path]) -> None:
    with pytest.raises(ConfigError, match="reference is 'mert'"):
        run_experiment(_config(sorted(_ANGLES), reference="mert"))


# --- subsampled replicates ---------------------------------------------------------
_SUBSAMPLE = {"subsample": {"n_replicates": 6, "fraction": 0.8, "galleries": ["fma_all"]}}


def _copy_config(structure_config: Config) -> Config:
    data = structure_config.to_dict()
    data["evaluation"]["structure"]["subsample"] = _SUBSAMPLE["subsample"]
    data["evaluation"]["structure"]["reference"] = "toy_structure"
    data["representations"] = [
        "toy_structure",
        {"label": "copy", "extractor": "toy_structure"},
        {"label": "random_s1", "extractor": "random", "overrides": {"seed": 1}},
    ]
    del data["evaluation"]["structure"]["baselines"]
    return Config(data=data, source=structure_config.source)


def test_copy_of_reference_has_exactly_zero_delta(structure_config: Config) -> None:
    run = run_experiment(_copy_config(structure_config))
    deltas = pd.read_csv(run.metrics_dir / "structure_deltas.csv")
    assert list(deltas.columns) == [
        "representation", "seed", "reference", "gallery", "graph", "metric",
        "delta", "ci_low", "ci_high", "n_replicates",
    ]  # fmt: skip
    assert set(deltas["representation"]) == {"copy", "random_s1"}  # one block per non-reference
    copy = deltas[deltas["representation"] == "copy"]
    assert len(copy) == len(METRIC_VIEWS)
    assert (copy["n_replicates"] == 6).all()
    # louvain_stability is nan with a single replicate seed only; here it has 3.
    assert (copy[["delta", "ci_low", "ci_high"]] == 0.0).all().all()

    structure = pd.read_csv(run.metrics_dir / "structure.csv")
    assert (structure["n_replicates"] == 6).all()
    assert (structure["ci_low"] <= structure["ci_high"]).all()
    replicates = pd.read_csv(run.metrics_dir / "structure_replicates.csv")
    assert len(replicates) == 3 * len(METRIC_VIEWS) * 6
    assert list(replicates.columns) == [
        "representation", "seed", "gallery", "graph", "metric", "replicate", "value",
    ]  # fmt: skip


def test_interval_is_percentile_of_replicates(structure_config: Config) -> None:
    run = run_experiment(_copy_config(structure_config))
    structure = pd.read_csv(run.metrics_dir / "structure.csv")
    replicates = pd.read_csv(run.metrics_dir / "structure_replicates.csv")
    row = structure[
        (structure["representation"] == "random_s1") & (structure["metric"] == "reciprocity")
    ]
    values = replicates[
        (replicates["representation"] == "random_s1") & (replicates["metric"] == "reciprocity")
    ]["value"]
    assert row["ci_low"].iloc[0] == pytest.approx(np.quantile(values, 0.025))
    assert row["ci_high"].iloc[0] == pytest.approx(np.quantile(values, 0.975))


def test_replicates_use_the_same_masks_and_are_deterministic(
    structure_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import musicsim.graphs.base as graph_base

    real = graph_base.build_graph
    seen: list[tuple[int, ...]] = []

    def spy(X: np.ndarray, ids: np.ndarray, cfg: object) -> Graph:
        seen.append(tuple(int(i) for i in ids))
        return real(X, ids, cfg)

    monkeypatch.setattr(graph_base, "build_graph", spy)
    config = _copy_config(structure_config)
    first = run_experiment(config).metrics_dir
    seen.pop(0)  # the graph stage builds the main representation's graph once before evaluating
    # Three full graphs, then 6 replicates x 3 representations (replicate-major):
    # replicate r of every representation must keep the same nodes.
    assert len(seen) == 3 + 6 * 3
    replicate_nodes = [seen[3 + 3 * r : 6 + 3 * r] for r in range(6)]
    assert all(len({*nodes}) == 1 for nodes in replicate_nodes)
    assert all(len(nodes[0]) == 5 for nodes in replicate_nodes)  # round(0.8 * 6) artists
    assert len({nodes[0] for nodes in replicate_nodes}) > 1  # and replicates differ

    monkeypatch.setattr(graph_base, "build_graph", real)
    second = run_experiment(config).metrics_dir
    for name in (
        "structure.csv", "structure_replicates.csv", "structure_deltas.csv", "graph_pairs.csv",
    ):  # fmt: skip
        pd.testing.assert_frame_equal(pd.read_csv(first / name), pd.read_csv(second / name))


def test_subsample_gallery_must_be_evaluated(structure_config: Config) -> None:
    data = structure_config.to_dict()
    data["evaluation"]["structure"]["subsample"] = {"galleries": ["fma_test"]}
    with pytest.raises(ConfigError, match=r"subsample\.galleries"):
        run_experiment(Config(data=data, source=structure_config.source))


# --- pairs of graphs ---------------------------------------------------------------
# Twelve points on an arc with irregular gaps, so the k-occurrence is not constant
# (it is in the two-cluster toy above, which makes the Spearman correlation undefined).
_ARC = [0, 3, 7, 15, 30, 31, 50, 90, 95, 130, 170, 175]


@register_extractor("toy_pairs")
class _ToyPairsExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        angle = np.radians(_ARC[item_id - 1])
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)


def _pairs_config(isolated: object, **structure: object) -> Config:
    ids = list(range(1, len(_ARC) + 1))
    paths.dataset_dir("toy_pairs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "item_id": ids,
            "path": [f"synthetic/{i}.wav" for i in ids],
            "split": ["test"] * len(ids),
            "artist_id": [100 + i for i in ids],
            "group_id": ids,
            "genre_top": ["A" if i <= 6 else "B" for i in ids],
            "genres_all": ["[]"] * len(ids),
        }
    ).to_csv(paths.dataset_index("toy_pairs"), index=False)
    return Config(
        data={
            "experiment": "e00_toy_pairs",
            "seed": 0,
            "dataset": {"directory": "toy_pairs"},
            "extractor": {"name": "toy_pairs", "dim": 2},
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
            "representations": [
                "toy_pairs",
                {"label": "copy", "extractor": "toy_pairs"},
                {"label": "random_s1", "extractor": "random", "overrides": {"seed": 1}},
            ],
            "evaluation": {
                "retrieval": {"enabled": False},
                "structure": {
                    "enabled": True,
                    "galleries": ["fma_all"],
                    "reference": "toy_pairs",
                    "louvain": {"n_seeds": 4, "n_seeds_replicate": 2},
                    **structure,
                },
            },
            "runtime": {"n_jobs": 1, "progress": False, "cache": True},
            "stages": ["extract", "embed", "graph", "evaluate"],
        }
    )


def test_pair_of_a_representation_with_its_copy_is_one(isolated_roots: dict[str, Path]) -> None:
    config = _pairs_config(
        isolated_roots, subsample={"n_replicates": 8, "fraction": 0.8, "galleries": ["fma_all"]}
    )
    pairs = pd.read_csv(run_experiment(config).metrics_dir / "graph_pairs.csv")
    assert list(pairs.columns) == [
        "representation_a", "seed_a", "representation_b", "seed_b", "gallery", "graph",
        "metric", "value", "ci_low", "ci_high", "n_replicates",
    ]  # fmt: skip
    # C(3, 2) pairs x 4 metrics x 1 gallery.
    assert len(pairs) == 3 * len(PAIR_METRICS) * 1
    assert set(pairs["metric"]) == set(PAIR_METRICS)
    copy = pairs[(pairs["representation_a"] == "toy_pairs") & (pairs["representation_b"] == "copy")]
    assert sorted(copy["metric"]) == sorted(PAIR_METRICS)
    for column in ("value", "ci_low", "ci_high"):
        assert copy[column].to_numpy() == pytest.approx(1.0, abs=1e-12)
    assert (copy["n_replicates"] == 8).all()
    # The seed columns are those of each representation.
    mixed = pairs[pairs["representation_b"] == "random_s1"]
    assert (mixed["seed_b"] == 1).all() and (mixed["seed_a"] == 0).all()


def test_pairs_without_subsampling_have_no_interval(isolated_roots: dict[str, Path]) -> None:
    pairs = pd.read_csv(
        run_experiment(_pairs_config(isolated_roots)).metrics_dir / "graph_pairs.csv"
    )
    assert len(pairs) == 3 * len(PAIR_METRICS)
    assert pairs["ci_low"].isna().all() and pairs["ci_high"].isna().all()
    assert (pairs["n_replicates"] == 0).all()


def test_pair_values_match_the_compare_functions(isolated_roots: dict[str, Path]) -> None:
    from musicsim.graphs.compare import neighbor_jaccard

    run = run_experiment(_pairs_config(isolated_roots))
    root = run.directory / "graphs"
    a = Graph.read(root / "toy_pairs" / "fma_all" / "knn2")
    b = Graph.read(root / "random_s1" / "fma_all" / "knn2")
    pairs = pd.read_csv(run.metrics_dir / "graph_pairs.csv")
    row = pairs[
        (pairs["representation_a"] == "toy_pairs")
        & (pairs["representation_b"] == "random_s1")
        & (pairs["metric"] == "neighbor_jaccard")
    ]
    assert row["value"].iloc[0] == pytest.approx(neighbor_jaccard(a, b, k=10))
