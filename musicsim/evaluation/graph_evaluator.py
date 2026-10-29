"""Graph-structure evaluation: hubness, reciprocity, clustering, connectivity and
genre homophily of the kNN graph each representation induces.

For every representation and gallery it builds the canonical kNN graph
(``graph.k``, ``graph.metric`` as configured) *inside the gallery*, so nodes
and neighbours never reach outside it, and writes:

``structure.csv``
    ``representation, gallery, metric, value`` — one row per statistic of
    :func:`graph_metrics`.
``hubness.csv``
    ``representation, gallery, item_id, k_occurrence`` — the k-occurrence of
    every node, which is what the hubness figures are drawn from.

Statistics are graph-level (one graph, one number), so no bootstrap interval is
attached; whether two representations differ significantly on them is the
business of the paired comparison of a later phase.

The evaluator is off unless the experiment sets ``evaluation.structure.enabled``
(:attr:`~musicsim.evaluation.base.Evaluator.default_enabled`); it does not check
that itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from musicsim.evaluation.base import Evaluator
from musicsim.evaluation.hubness import hubness_summary, k_occurrence
from musicsim.evaluation.retrieval import _representation_specs
from musicsim.evaluation.structure import (
    component_summary,
    label_homophily,
    mean_clustering,
    reciprocity,
)
from musicsim.galleries import gallery_mask
from musicsim.graphs.adjacency import to_csr
from musicsim.graphs.base import Graph
from musicsim.registry import register_evaluator
from musicsim.representations import load_aligned_representations
from musicsim.runlog import RunLog

__all__ = ["GraphStructureEvaluator", "graph_metrics"]


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

    Reads ``evaluation.structure`` (``galleries``, ``baselines``);
    ``representations`` at the top level works as it does for retrieval, and
    ``graph.k`` / ``graph.builder`` come from the shared graph section.
    """

    def evaluate(self, index: pd.DataFrame, run: RunLog) -> None:
        # Deferred: these modules import the evaluation package back (circular otherwise).
        from musicsim.graphs.base import build_graph
        from musicsim.registry import RELEVANCES

        stage_cfg = self.config.section("evaluation").get("structure", {})
        main_name = self.config.require("extractor.name")
        specs = _representation_specs(self.config, stage_cfg, main_name)
        loaded = load_aligned_representations(self.config, index, specs)
        relevance = RELEVANCES.get(self.config.get("retrieval.relevance", "label_match"))(
            self.config
        )

        galleries = stage_cfg.get("galleries", self.config.get("retrieval.gallery", "fma_test"))
        if isinstance(galleries, str):
            galleries = [galleries]

        graph_cfg = self.config.section("graph")
        rows: list[dict] = []
        hubness_frames: list[pd.DataFrame] = []
        for gallery_name in galleries:
            mask = gallery_mask(loaded.index, gallery_name)
            gallery_ids = loaded.ids[mask]
            if len(gallery_ids) == 0:
                raise ValueError(f"gallery {gallery_name!r} selects no item")
            labels = relevance.label_codes(index, gallery_ids)
            for spec in specs:
                graph = build_graph(loaded.X[spec.label][mask], gallery_ids, graph_cfg)
                rows.extend(
                    {
                        "representation": spec.label,
                        "gallery": gallery_name,
                        "metric": metric,
                        "value": value,
                    }
                    for metric, value in graph_metrics(graph, labels).items()
                )
                hubness_frames.append(
                    pd.DataFrame(
                        {
                            "representation": spec.label,
                            "gallery": gallery_name,
                            "item_id": gallery_ids,
                            "k_occurrence": k_occurrence(graph),
                        }
                    )
                )

        pd.DataFrame(rows).to_csv(run.metrics_dir / "structure.csv", index=False)
        pd.concat(hubness_frames, ignore_index=True).to_csv(
            run.metrics_dir / "hubness.csv", index=False
        )
        run.logger.info(
            "structure: %d representation(s), %d galler(y/ies)", len(specs), len(galleries)
        )
