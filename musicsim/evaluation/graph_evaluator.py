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
    the kNN graph as built, ``union`` its undirected version. The interval
    columns are empty (``n_replicates`` is 0): these are point estimates, and
    the intervals arrive with the subsampling step.
``hubness.csv``
    ``item_id, representation, seed, gallery, graph, k_occurrence`` — the
    k-occurrence of every node, which is what the hubness figures are drawn from.

``seed`` is the seed of each representation's own configuration, so several
draws of the ``random`` extractor stay distinguishable.

The evaluator is off unless the experiment sets ``evaluation.structure.enabled``
(:attr:`~musicsim.evaluation.base.Evaluator.default_enabled`); it does not check
that itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from musicsim.config import ConfigError
from musicsim.evaluation.base import Evaluator
from musicsim.evaluation.hubness import hubness_summary, k_occurrence
from musicsim.evaluation.structure import (
    component_summary,
    label_homophily,
    mean_clustering,
    reciprocity,
)
from musicsim.graphs.adjacency import to_csr
from musicsim.graphs.base import Graph
from musicsim.graphs.gallery import gallery_nodes, graph_dir
from musicsim.registry import register_evaluator
from musicsim.representations import load_aligned_representations, stage_representations
from musicsim.runlog import RunLog

__all__ = ["METRIC_VIEWS", "GraphStructureEvaluator", "graph_metrics"]

#: The edge set each metric is computed on.
METRIC_VIEWS = {
    "k_occurrence_skewness": "directed",
    "antihub_fraction": "directed",
    "hub_fraction": "directed",
    "max_k_occurrence": "directed",
    "label_homophily": "directed",
    "reciprocity": "directed",
    "mean_clustering": "union",
    "giant_fraction": "union",
    "n_components": "union",
}

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


def graph_metrics(graph: Graph, genre_codes: np.ndarray) -> dict[str, float]:
    """Every structure metric of ``graph`` as one flat ``{metric: value}`` mapping.

    ``genre_codes`` holds one integer genre code per node. The headline metrics
    are ``k_occurrence_skewness`` and ``label_homophily``; the others are
    ``antihub_fraction``, ``hub_fraction``, ``max_k_occurrence``,
    ``reciprocity``, ``mean_clustering``, ``giant_fraction`` and
    ``n_components``. Raises ``ValueError`` if the graph has no edge.
    """
    hubness = hubness_summary(k_occurrence(graph))
    union = to_csr(graph, symmetrize="union")
    return {
        "k_occurrence_skewness": hubness["skewness"],
        "label_homophily": label_homophily(graph, genre_codes),
        "antihub_fraction": hubness["antihub_fraction"],
        "hub_fraction": hubness["hub_fraction"],
        "max_k_occurrence": hubness["max_k_occurrence"],
        "reciprocity": reciprocity(graph),
        "mean_clustering": mean_clustering(union),
        **component_summary(union),
    }


@register_evaluator("structure")
class GraphStructureEvaluator(Evaluator):
    """Structure of the kNN graph of the main extractor plus its baselines.

    Reads ``evaluation.structure`` (``galleries``, ``baselines``, ``reference``);
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
        relevance = RELEVANCES.get(self.config.get("retrieval.relevance", "label_match"))(
            self.config
        )
        threshold = float(self.config.get("graph.near_duplicate_threshold", 0.9999))

        galleries = stage_cfg.get("galleries", self.config.get("retrieval.gallery", "fma_test"))
        if isinstance(galleries, str):
            galleries = [galleries]

        graph_cfg = self.config.section("graph")
        graph_name = f"{graph_cfg.get('builder', 'knn')}{int(graph_cfg['k'])}"
        rows: list[dict] = []
        hubness_frames: list[pd.DataFrame] = []
        for gallery_name in galleries:
            nodes = gallery_nodes(loaded, gallery_name, reference=reference, threshold=threshold)
            run.logger.info(
                "structure: gallery %s keeps %d node(s), %d near-duplicate item(s) collapsed",
                gallery_name,
                len(nodes.ids),
                nodes.n_collapsed,
            )
            genre_codes = relevance.label_codes(index, nodes.ids)
            for spec in specs:
                graph = build_graph(loaded.X[spec.label][nodes.rows], nodes.ids, graph_cfg)
                graph.write(graph_dir(run.directory, spec.label, gallery_name, graph_name))
                seed = loaded.seeds[spec.label]
                rows.extend(
                    {
                        "representation": spec.label,
                        "seed": seed,
                        "gallery": gallery_name,
                        "graph": graph_name,
                        "view": METRIC_VIEWS[metric],
                        "metric": metric,
                        "value": value,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n_replicates": 0,
                    }
                    for metric, value in graph_metrics(graph, genre_codes).items()
                )
                hubness_frames.append(
                    pd.DataFrame(
                        {
                            "item_id": nodes.ids,
                            "representation": spec.label,
                            "seed": seed,
                            "gallery": gallery_name,
                            "graph": graph_name,
                            "k_occurrence": k_occurrence(graph),
                        }
                    )
                )

        pd.DataFrame(rows, columns=_STRUCTURE_COLUMNS).to_csv(
            run.metrics_dir / "structure.csv", index=False
        )
        pd.concat(hubness_frames, ignore_index=True).to_csv(
            run.metrics_dir / "hubness.csv", index=False
        )
        run.logger.info(
            "structure: %d representation(s), %d galler(y/ies)", len(specs), len(galleries)
        )
