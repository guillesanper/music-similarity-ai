"""Graph-structure evaluation: hubness, reciprocity, clustering, connectivity and
genre homophily of the kNN graph each representation induces.

For every gallery the nodes are its items with near-duplicates collapsed on the
``reference`` representation (see :func:`musicsim.graphs.gallery.gallery_nodes`),
so every representation's graph has the same nodes. Each representation's kNN
graph is built *inside* that gallery, written under
``graphs/<representation>/<gallery>/<builder><k>/``, and scored. Two files come
out under ``metrics/``:

``structure.csv``
    ``representation, seed, gallery, graph, view, metric, value, ci_low,
    ci_high, n_replicates`` — one row per statistic of :func:`graph_metrics`.
    ``view`` says which edge set the statistic is computed on: ``directed`` is
    the kNN graph as built, ``union`` its undirected version. The three
    ``vote_*`` metrics (neighbour-vote accuracy, macro F1 and abstention, with
    the node's own artist never voting) are transductive descriptions of the
    graph, see :mod:`musicsim.evaluation.graph_classification`. ``value`` is the
    point estimate on the whole gallery. In the galleries listed in
    ``evaluation.structure.subsample.galleries``, ``ci_low`` / ``ci_high`` are the
    95 % percentile interval of the metric over the subsample replicates and
    ``n_replicates`` their number; elsewhere the interval is empty and
    ``n_replicates`` is 0.
``structure_replicates.csv``
    ``representation, seed, gallery, graph, metric, replicate, value`` — the
    metric of every replicate (only in subsampled galleries).
``structure_deltas.csv``
    ``representation, seed, reference, gallery, graph, metric, delta, ci_low,
    ci_high, n_replicates`` — each representation minus the reference. ``delta``
    is the difference of the whole-gallery values; the interval is the
    percentile interval of the replicate-by-replicate differences.
``graph_pairs.csv``
    ``representation_a, seed_a, representation_b, seed_b, gallery, graph, metric,
    value, ci_low, ci_high, n_replicates`` — how much the graphs of every
    unordered pair of representations agree: ``neighbor_jaccard`` (k = 10, the
    headline), ``edge_overlap``, ``k_occurrence_spearman`` and ``partition_ari``
    (mean over Louvain seeds, seed ``i`` of one graph against seed ``i`` of the
    other). In subsampled galleries the interval is the percentile interval of the
    pair's value over the replicates (the value itself, not a difference). The
    pairs among the ``random`` draws are the noise floor of a graph that carries
    no structure; it is *not* the noise floor between trained models that the
    project plan asks for, which needs those models.
``hubness.csv``
    ``item_id, representation, seed, gallery, graph, k_occurrence`` — the
    k-occurrence of every node, which is what the hubness figures are drawn from.

Next to ``neighbors.npz`` and ``edges.csv`` every graph directory also gets
``nodes.csv`` and ``graph.graphml`` (:func:`musicsim.graphs.export.export_graph`),
with ``item_id, genre_top, artist_id, split, k_occurrence`` and ``community``
(the first Louvain seed) as node attributes.

For every gallery three figures go to ``figures/`` (see :mod:`musicsim.plotting`),
each as ``.png`` and ``.pdf``: ``k_occurrence_<gallery>`` and
``community_sizes_<gallery>`` with one line per representation, and
``community_genre_<gallery>_<reference>`` with the genre make-up of the reference
representation's communities. Communities are those of the *first* Louvain seed
(the same ones exported as the ``community`` node attribute); the other seeds are
used by the numbers, not by the figures. Genre names are read from ``genre_top``,
which is what the default ``label_match`` relevance codes, because a relevance
function only returns integer codes; a different ``relevance.column`` is rejected
rather than drawn with the wrong names.

``seed`` is the seed of each representation's own configuration, so several
draws of the ``random`` extractor stay distinguishable.

Replicates are paired subsamples of the gallery's artists (see
:func:`musicsim.evaluation.bootstrap.paired_artist_subsample`): the masks are
drawn once per gallery and applied to every representation, and in each
replicate the graph is *rebuilt* from the surviving nodes rather than cropped.
They are subsamples without replacement, so their interval describes
sensitivity to the artist mix, not a confidence interval, and no p-value is
derived from it. Inside the replicates Louvain runs with
``louvain.n_seeds_replicate`` seeds (default 3) to bound the cost; the point
estimate keeps ``louvain.n_seeds``. Louvain seeds come from the experiment
seed, never from the representation label, in the replicates too.

The replicates run replicate-major: in each one every representation's graph is
rebuilt and scored, then all pairs are compared on those graphs, and the graphs
are dropped. Nothing is kept across replicates but the numbers, so memory does
not grow with ``subsample.n_replicates``. The Louvain partitions are computed once
per graph and shared by the structure metrics and ``partition_ari``. A pair
statistic that is undefined (``nan``: a constant k-occurrence, no edge) stays
``nan`` in the replicates and so does its interval.

The evaluator is off unless the experiment sets ``evaluation.structure.enabled``
(:attr:`~musicsim.evaluation.base.Evaluator.default_enabled`); it does not check
that itself.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from musicsim.config import ConfigError
from musicsim.evaluation.base import Evaluator
from musicsim.evaluation.bootstrap import paired_artist_subsample, percentile_interval
from musicsim.evaluation.communities import (
    louvain_partitions,
    partition_modularity,
    partition_nmi,
    partition_stability,
)
from musicsim.evaluation.graph_classification import neighbor_vote, vote_metrics
from musicsim.evaluation.hubness import hubness_summary, k_occurrence
from musicsim.evaluation.structure import (
    component_summary,
    label_homophily,
    mean_clustering,
    reciprocity,
)
from musicsim.graphs.adjacency import to_csr
from musicsim.graphs.base import Graph
from musicsim.graphs.compare import (
    edge_overlap,
    k_occurrence_spearman,
    neighbor_jaccard,
    partition_ari,
)
from musicsim.graphs.export import export_graph
from musicsim.graphs.gallery import gallery_nodes, graph_dir
from musicsim.plotting import (
    plot_community_genre_matrix,
    plot_community_sizes,
    plot_k_occurrence,
    save_figure,
)
from musicsim.registry import register_evaluator
from musicsim.representations import load_aligned_representations, stage_representations
from musicsim.runlog import RunLog
from musicsim.seeding import derive_seed

__all__ = ["METRIC_VIEWS", "PAIR_METRICS", "GraphStructureEvaluator", "graph_metrics"]

#: The edge set each metric is computed on.
METRIC_VIEWS = {
    "k_occurrence_skewness": "directed",
    "antihub_fraction": "directed",
    "hub_fraction": "directed",
    "max_k_occurrence": "directed",
    "label_homophily": "directed",
    "nmi_louvain_genre": "union",
    "modularity": "union",
    "louvain_stability": "union",
    "reciprocity": "directed",
    "mean_clustering": "union",
    "giant_fraction": "union",
    "n_components": "union",
    "vote_accuracy": "directed",
    "vote_macro_f1": "directed",
    "vote_abstention_fraction": "directed",
}

#: The statistics compared between two graphs, in output order.
PAIR_METRICS = ("neighbor_jaccard", "edge_overlap", "k_occurrence_spearman", "partition_ari")

#: Neighbours per node that ``neighbor_jaccard`` compares.
_PAIR_K = 10

_PAIR_COLUMNS = [
    "representation_a",
    "seed_a",
    "representation_b",
    "seed_b",
    "gallery",
    "graph",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n_replicates",
]

_STRUCTURE_COLUMNS = [
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

_REPLICATE_COLUMNS = [
    "representation",
    "seed",
    "gallery",
    "graph",
    "metric",
    "replicate",
    "value",
]

_DELTA_COLUMNS = [
    "representation",
    "seed",
    "reference",
    "gallery",
    "graph",
    "metric",
    "delta",
    "ci_low",
    "ci_high",
    "n_replicates",
]


def graph_metrics(
    graph: Graph,
    genre_codes: np.ndarray,
    *,
    seed: int,
    n_louvain_seeds: int,
    resolution: float,
) -> dict[str, float]:
    """Every structure metric of ``graph`` as one flat ``{metric}: value`` mapping.

    ``genre_codes`` holds one integer genre code per node. The headline metrics
    are ``k_occurrence_skewness``, ``nmi_louvain_genre`` and ``label_homophily``;
    the others are ``antihub_fraction``, ``hub_fraction``, ``max_k_occurrence``,
    ``modularity``, ``louvain_stability``, ``reciprocity``, ``mean_clustering``,
    ``giant_fraction`` and ``n_components``.

    Louvain runs on the binary union adjacency with ``n_louvain_seeds`` seeds
    derived from ``seed`` alone, so identical graphs get identical partitions
    whatever their label; NMI and modularity are means over those seeds, and
    stability is their mean pairwise ARI (``nan`` with a single seed). Raises
    ``ValueError`` if the graph has no edge.
    """
    return _metrics_and_partitions(
        graph, genre_codes, seed=seed, n_louvain_seeds=n_louvain_seeds, resolution=resolution
    )[0]


def _metrics_and_partitions(
    graph: Graph,
    genre_codes: np.ndarray,
    *,
    seed: int,
    n_louvain_seeds: int,
    resolution: float,
    artist_codes: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """:func:`graph_metrics` plus the ``(n_louvain_seeds, N)`` partitions it used.

    The partitions are what ``partition_ari`` compares between graphs; returning
    them avoids running Louvain a second time on the same graph. With
    ``artist_codes`` it also adds the neighbour-vote metrics
    (:func:`musicsim.evaluation.graph_classification.neighbor_vote`), in which a
    node's own artist never votes; :func:`graph_metrics` leaves them out because
    it has no artists to exclude.
    """
    hubness = hubness_summary(k_occurrence(graph))
    homophily = label_homophily(graph, genre_codes)  # rejects an edgeless graph first
    union = to_csr(graph, symmetrize="union")
    partitions = louvain_partitions(
        union, n_seeds=n_louvain_seeds, seed=seed, resolution=resolution
    )
    metrics = {
        "k_occurrence_skewness": hubness["skewness"],
        "nmi_louvain_genre": partition_nmi(partitions, genre_codes)[0],
        "label_homophily": homophily,
        "antihub_fraction": hubness["antihub_fraction"],
        "hub_fraction": hubness["hub_fraction"],
        "max_k_occurrence": hubness["max_k_occurrence"],
        "modularity": partition_modularity(union, partitions)[0],
        "louvain_stability": partition_stability(partitions),
        "reciprocity": reciprocity(graph),
        "mean_clustering": mean_clustering(union),
        **component_summary(union),
    }
    if artist_codes is not None:
        votes = neighbor_vote(graph, genre_codes, exclude_groups=artist_codes)
        metrics.update(vote_metrics(votes, genre_codes))
    return metrics, partitions


def _genre_names(genre_top: pd.Series, genre_codes: np.ndarray) -> list[str]:
    """Name of every genre code ``0..max``: the ``genre_top`` of the nodes carrying it.

    A code whose nodes have no genre (missing labels get a code each) is named
    ``unknown``.
    """
    first = pd.Series(genre_top.to_numpy()).groupby(genre_codes).first()
    names = first.reindex(range(int(genre_codes.max()) + 1)).fillna("unknown")
    return [str(name) for name in names]


def _pair_metrics(
    graph_a: Graph, graph_b: Graph, partitions_a: np.ndarray, partitions_b: np.ndarray
) -> dict[str, float]:
    """The :data:`PAIR_METRICS` of two graphs over the same nodes.

    ``partition_ari`` is the mean over Louvain seeds, seed ``i`` of one graph
    against seed ``i`` of the other.
    """
    return {
        "neighbor_jaccard": neighbor_jaccard(graph_a, graph_b, k=_PAIR_K),
        "edge_overlap": edge_overlap(graph_a, graph_b),
        "k_occurrence_spearman": k_occurrence_spearman(graph_a, graph_b),
        "partition_ari": float(
            np.mean(
                [partition_ari(pa, pb) for pa, pb in zip(partitions_a, partitions_b, strict=True)]
            )
        ),
    }


@register_evaluator("structure")
class GraphStructureEvaluator(Evaluator):
    """Structure of the kNN graph of the main extractor plus its baselines, and the
    agreement between the graphs of every pair of representations.

    Reads ``evaluation.structure`` (``galleries``, ``baselines``, ``reference``,
    ``louvain.n_seeds``, ``louvain.n_seeds_replicate``, ``louvain.resolution``,
    ``subsample.n_replicates``, ``subsample.fraction``, ``subsample.galleries``);
    ``representations`` at the top level works as it does for retrieval, and
    ``graph.k`` / ``graph.builder`` / ``graph.near_duplicate_threshold`` come
    from the shared graph section. ``reference`` is the representation on which
    near-duplicates are detected (default: the main extractor) and must be one
    of the evaluated labels.
    """

    def evaluate(self, index: pd.DataFrame, run: RunLog) -> None:
        # Deferred: these modules import the evaluation package back (circular otherwise).
        from musicsim.graphs.base import build_graph
        from musicsim.registry import RELEVANCES

        stage_cfg = self.config.section("evaluation").get("structure", {})
        main_name = self.config.require("extractor.name")
        specs = stage_representations(self.config, stage_cfg, main_name, "structure")
        labels = [spec.label for spec in specs]
        reference = stage_cfg.get("reference", main_name)
        if reference not in labels:
            raise ConfigError(
                f"evaluation.structure.reference is {reference!r}; expected one of {labels}"
            )

        loaded = load_aligned_representations(self.config, index, specs)
        column = self.config.get("relevance.column", "genre_top")
        if column != "genre_top":
            raise ConfigError(
                f"relevance.column is {column!r}; the structure figures name genres from "
                "'genre_top', so expected 'genre_top'"
            )
        relevance = RELEVANCES.get(self.config.get("retrieval.relevance", "label_match"))(
            self.config
        )
        louvain = stage_cfg.get("louvain", {})
        n_louvain_seeds = int(louvain.get("n_seeds", 10))
        resolution = float(louvain.get("resolution", 1.0))
        experiment_seed = int(self.config.require("seed"))
        threshold = float(self.config.get("graph.near_duplicate_threshold", 0.9999))

        galleries = stage_cfg.get("galleries", self.config.get("retrieval.gallery", "fma_test"))
        if isinstance(galleries, str):
            galleries = [galleries]

        n_replicate_seeds = int(louvain.get("n_seeds_replicate", 3))
        subsample = stage_cfg.get("subsample", {})
        n_replicates = int(subsample.get("n_replicates", 200))
        fraction = float(subsample.get("fraction", 0.8))
        subsampled = subsample.get("galleries", [])
        if isinstance(subsampled, str):
            subsampled = [subsampled]
        unknown = [name for name in subsampled if name not in galleries]
        if unknown:
            raise ConfigError(
                f"evaluation.structure.subsample.galleries has {unknown}; "
                f"expected a subset of the evaluated galleries {list(galleries)}"
            )

        graph_cfg = self.config.section("graph")
        graph_name = f"{graph_cfg.get('builder', 'knn')}{int(graph_cfg['k'])}"
        rows: list[dict] = []
        replicate_frames: list[pd.DataFrame] = []
        delta_rows: list[dict] = []
        hubness_frames: list[pd.DataFrame] = []
        pair_rows: list[dict] = []
        label_pairs = list(itertools.combinations(labels, 2))
        for gallery_name in galleries:
            nodes = gallery_nodes(loaded, gallery_name, reference=reference, threshold=threshold)
            run.logger.info(
                "structure: gallery %s keeps %d node(s), %d near-duplicate item(s) collapsed",
                gallery_name,
                len(nodes.ids),
                nodes.n_collapsed,
            )
            genre_codes = relevance.label_codes(index, nodes.ids)
            artist_codes = loaded.index.loc[nodes.ids, "artist_id"].to_numpy()
            point: dict[str, dict[str, float]] = {}
            graphs: dict[str, Graph] = {}
            partitions: dict[str, np.ndarray] = {}
            for spec in specs:
                graph = build_graph(loaded.X[spec.label][nodes.rows], nodes.ids, graph_cfg)
                directory = graph_dir(run.directory, spec.label, gallery_name, graph_name)
                graph.write(directory)
                graphs[spec.label] = graph
                point[spec.label], partitions[spec.label] = _metrics_and_partitions(
                    graph,
                    genre_codes,
                    seed=experiment_seed,
                    n_louvain_seeds=n_louvain_seeds,
                    resolution=resolution,
                    artist_codes=artist_codes,
                )
                node_table = loaded.index.loc[
                    nodes.ids, ["genre_top", "artist_id", "split"]
                ].reset_index()
                node_table["k_occurrence"] = k_occurrence(graph)
                node_table["community"] = partitions[spec.label][0]
                export_graph(graph, node_table, directory)
                hubness_frames.append(
                    pd.DataFrame(
                        {
                            "item_id": nodes.ids,
                            "representation": spec.label,
                            "seed": loaded.seeds[spec.label],
                            "gallery": gallery_name,
                            "graph": graph_name,
                            "k_occurrence": k_occurrence(graph),
                        }
                    )
                )
            first_seed = {label: parts[0] for label, parts in partitions.items()}
            figures = run.figures_dir
            save_figure(
                plot_k_occurrence(
                    {label: k_occurrence(graph) for label, graph in graphs.items()},
                    int(graph_cfg["k"]),
                ),
                figures / f"k_occurrence_{gallery_name}",
            )
            save_figure(
                plot_community_sizes(
                    {label: np.bincount(parts) for label, parts in first_seed.items()}
                ),
                figures / f"community_sizes_{gallery_name}",
            )
            save_figure(
                plot_community_genre_matrix(
                    first_seed[reference],
                    genre_codes,
                    _genre_names(loaded.index.loc[nodes.ids, "genre_top"], genre_codes),
                ),
                figures / f"community_genre_{gallery_name}_{reference}",
            )
            point_pairs = {
                (a, b): _pair_metrics(graphs[a], graphs[b], partitions[a], partitions[b])
                for a, b in label_pairs
            }

            # replicate[label][metric] -> (n_replicates,) values; empty when not subsampled.
            replicate: dict[str, dict[str, np.ndarray]] = {}
            pair_replicate: dict[tuple[str, str], dict[str, np.ndarray]] = {}
            if gallery_name in subsampled:
                masks = paired_artist_subsample(
                    loaded.index.loc[nodes.ids, "artist_id"].to_numpy(),
                    n_replicates=n_replicates,
                    fraction=fraction,
                    seed=derive_seed(experiment_seed, "structure", "subsample", gallery_name),
                )
                X = {spec.label: loaded.X[spec.label][nodes.rows] for spec in specs}
                metric_runs: dict[str, list[dict[str, float]]] = {label: [] for label in labels}
                pair_runs: dict[tuple[str, str], list[dict[str, float]]] = {
                    pair: [] for pair in label_pairs
                }
                for r, keep in enumerate(masks):
                    built: dict[str, tuple[Graph, np.ndarray]] = {}
                    for label in labels:
                        graph = build_graph(X[label][keep], nodes.ids[keep], graph_cfg)
                        values, parts = _metrics_and_partitions(
                            graph,
                            genre_codes[keep],
                            seed=experiment_seed,
                            n_louvain_seeds=n_replicate_seeds,
                            resolution=resolution,
                            artist_codes=artist_codes[keep],
                        )
                        metric_runs[label].append(values)
                        built[label] = (graph, parts)
                    for a, b in label_pairs:
                        pair_runs[(a, b)].append(
                            _pair_metrics(built[a][0], built[b][0], built[a][1], built[b][1])
                        )
                    if (r + 1) % 50 == 0 or r + 1 == len(masks):
                        run.logger.info(
                            "structure: %s: %d/%d replicate(s) done",
                            gallery_name,
                            r + 1,
                            len(masks),
                        )
                for spec in specs:
                    replicate[spec.label] = {
                        metric: np.array([values[metric] for values in metric_runs[spec.label]])
                        for metric in METRIC_VIEWS
                    }
                    replicate_frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "representation": spec.label,
                                    "seed": loaded.seeds[spec.label],
                                    "gallery": gallery_name,
                                    "graph": graph_name,
                                    "metric": metric,
                                    "replicate": r,
                                    "value": value,
                                }
                                for metric, series in replicate[spec.label].items()
                                for r, value in enumerate(series)
                            ],
                            columns=_REPLICATE_COLUMNS,
                        )
                    )
                for pair, runs in pair_runs.items():
                    pair_replicate[pair] = {
                        metric: np.array([values[metric] for values in runs])
                        for metric in PAIR_METRICS
                    }

            for a, b in label_pairs:
                for metric in PAIR_METRICS:
                    low = high = np.nan
                    if (a, b) in pair_replicate:
                        low, high = percentile_interval(pair_replicate[(a, b)][metric])
                    pair_rows.append(
                        {
                            "representation_a": a,
                            "seed_a": loaded.seeds[a],
                            "representation_b": b,
                            "seed_b": loaded.seeds[b],
                            "gallery": gallery_name,
                            "graph": graph_name,
                            "metric": metric,
                            "value": point_pairs[(a, b)][metric],
                            "ci_low": low,
                            "ci_high": high,
                            "n_replicates": n_replicates if (a, b) in pair_replicate else 0,
                        }
                    )

            for spec in specs:
                seed = loaded.seeds[spec.label]
                for metric, value in point[spec.label].items():
                    low = high = np.nan
                    if spec.label in replicate:
                        low, high = percentile_interval(replicate[spec.label][metric])
                    rows.append(
                        {
                            "representation": spec.label,
                            "seed": seed,
                            "gallery": gallery_name,
                            "graph": graph_name,
                            "view": METRIC_VIEWS[metric],
                            "metric": metric,
                            "value": value,
                            "ci_low": low,
                            "ci_high": high,
                            "n_replicates": n_replicates if spec.label in replicate else 0,
                        }
                    )
                if spec.label == reference or spec.label not in replicate:
                    continue
                for metric, value in point[spec.label].items():
                    low, high = percentile_interval(
                        replicate[spec.label][metric] - replicate[reference][metric]
                    )
                    delta_rows.append(
                        {
                            "representation": spec.label,
                            "seed": seed,
                            "reference": reference,
                            "gallery": gallery_name,
                            "graph": graph_name,
                            "metric": metric,
                            "delta": value - point[reference][metric],
                            "ci_low": low,
                            "ci_high": high,
                            "n_replicates": n_replicates,
                        }
                    )

        pd.DataFrame(rows, columns=_STRUCTURE_COLUMNS).to_csv(
            run.metrics_dir / "structure.csv", index=False
        )
        pd.concat(hubness_frames, ignore_index=True).to_csv(
            run.metrics_dir / "hubness.csv", index=False
        )
        if pair_rows:
            pd.DataFrame(pair_rows, columns=_PAIR_COLUMNS).to_csv(
                run.metrics_dir / "graph_pairs.csv", index=False
            )
        if replicate_frames:
            pd.concat(replicate_frames, ignore_index=True).to_csv(
                run.metrics_dir / "structure_replicates.csv", index=False
            )
            pd.DataFrame(delta_rows, columns=_DELTA_COLUMNS).to_csv(
                run.metrics_dir / "structure_deltas.csv", index=False
            )
        run.logger.info(
            "structure: %d representation(s), %d galler(y/ies)", len(specs), len(galleries)
        )
