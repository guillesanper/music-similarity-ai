"""Graph-structure evaluation: hubness, reciprocity, connectivity and genre
homophily of the kNN graph each representation induces.

For every representation and gallery it builds the canonical kNN graph
(``graph.k``, ``graph.metric`` as configured) *inside the gallery*, so nodes
and neighbours never reach outside it, and writes:

``graph_structure.csv``
    ``representation, gallery, metric, value`` — one row per statistic of
    :func:`musicsim.graphs.structure.structure_summary`.
``in_degree.csv``
    ``representation, gallery, item_id, in_degree`` — the k-occurrence of every
    node, which is what the hubness figures are drawn from.

Statistics are graph-level (one graph, one number), so no bootstrap interval is
attached; whether two representations differ significantly on them is the
business of the paired comparison of a later phase.
"""

from __future__ import annotations

import pandas as pd

from musicsim.evaluation.base import Evaluator
from musicsim.evaluation.retrieval import _representation_specs
from musicsim.galleries import gallery_mask
from musicsim.registry import register_evaluator
from musicsim.representations import load_aligned_representations
from musicsim.runlog import RunLog

__all__ = ["GraphStructureEvaluator"]


@register_evaluator("graph_structure")
class GraphStructureEvaluator(Evaluator):
    """Structural statistics of the kNN graph of the main extractor plus its baselines.

    Reads ``evaluation.graph_structure`` (``enabled``, ``galleries``,
    ``baselines``); ``representations`` at the top level works as it does for
    retrieval, and ``graph.k`` / ``graph.builder`` come from the shared graph
    section. Evaluation is off unless the experiment turns it on.
    """

    def evaluate(self, index: pd.DataFrame, run: RunLog) -> None:
        from musicsim.graphs.base import build_graph
        from musicsim.graphs.structure import in_degree, structure_summary
        from musicsim.registry import RELEVANCES

        stage_cfg = self.config.section("evaluation").get("graph_structure", {})
        if not stage_cfg.get("enabled", False):
            return

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
        degree_frames: list[pd.DataFrame] = []
        for gallery_name in galleries:
            mask = gallery_mask(loaded.index, gallery_name)
            gallery_ids = loaded.ids[mask]
            if len(gallery_ids) == 0:
                raise ValueError(f"gallery {gallery_name!r} selects no item")
            labels = relevance.label_codes(index, gallery_ids)
            for spec in specs:
                graph = build_graph(loaded.X[spec.label][mask], gallery_ids, graph_cfg)
                summary = structure_summary(graph, labels)
                rows.extend(
                    {
                        "representation": spec.label,
                        "gallery": gallery_name,
                        "metric": metric,
                        "value": value,
                    }
                    for metric, value in summary.items()
                )
                degree_frames.append(
                    pd.DataFrame(
                        {
                            "representation": spec.label,
                            "gallery": gallery_name,
                            "item_id": gallery_ids,
                            "in_degree": in_degree(graph),
                        }
                    )
                )

        pd.DataFrame(rows).to_csv(run.metrics_dir / "graph_structure.csv", index=False)
        pd.concat(degree_frames, ignore_index=True).to_csv(
            run.metrics_dir / "in_degree.csv", index=False
        )
        run.logger.info(
            "graph_structure: %d representation(s), %d galler(y/ies)", len(specs), len(galleries)
        )
