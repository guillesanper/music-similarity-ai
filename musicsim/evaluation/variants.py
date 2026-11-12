"""Graph-variant evaluation: does a conclusion about representations survive a change of graph?

For every gallery, representation and graph variant the evaluator builds the
graph over *all* the rows of the gallery (near-duplicates included, not
collapsed: :func:`musicsim.evaluation.graph_retrieval.graph_per_query_metrics`
needs the graph to cover the rows it ranks) and scores the ranking that graph
induces under the three exclusion variants of retrieval (``dedup`` is the
headline one, see :mod:`musicsim.evaluation.retrieval`).

The variants are a closed list (C.5): ``knn`` with k in {5, 10, 20, 50},
``mutual_knn``, ``epsilon`` (edge count matched to the kNN-10) and
``mutual_proximity``, the last three with k = 10. Their ``graph`` names are
``knn5 .. knn50``, ``mutual_knn10``, ``epsilon`` and ``mutual_proximity10``.
A graph row with fewer than ten neighbours is completed with the cosine order
(see the ranking rule of :mod:`musicsim.evaluation.graph_retrieval`); with
k = 5 every query is completed by construction, and that shows in the
completed fraction rather than being hidden.

Three files come out under ``metrics/``:

``variants_per_query.csv``
    ``item_id, representation, seed, gallery, graph, variant, metric, k, value,
    completed``. ``variant`` is the exclusion variant; ``graph`` is the graph
    variant. ``completed`` says whether the query's top-10 partly came from the
    cosine fill.
``variants_summary.csv``
    Mean per ``(representation, seed, gallery, graph, variant, metric, k)``,
    plus ``completed_fraction`` and ``isolated_fraction`` (nodes of degree 0 in
    the graph, which is what separates a threshold graph from a kNN).
``variants_deltas.csv``
    The result of the experiment: one row per ``(gallery, graph, comparison,
    metric)``, each representation minus the reference, ``dedup`` variant,
    headline metrics only (R@10, MAP, nDCG@10, the last two with their usual
    ``k``). ``sign`` is +1 / -1 when the 95 % interval excludes 0, else 0. The
    interval is the artist-clustered paired bootstrap; the resample indices
    depend on ``(gallery, graph, metric)`` but not on the representation, so
    every comparison resamples the same artists. There is no Holm correction
    and no p-value: this is a sensitivity analysis, read by sign (C.5): a
    conclusion is robust if the sign of a delta is the same for every variant.
    :func:`sign_consistency` summarises exactly that and the evaluator logs it.

Mutual proximity is cubic in the number of nodes, so the evaluator refuses a
gallery larger than ``evaluation.variants.mutual_proximity_max_nodes`` (default
2000) instead of running for hours. Only the test gallery (800 items) is meant
for this experiment.

The evaluator is off unless ``evaluation.variants.enabled`` is set
(:attr:`~musicsim.evaluation.base.Evaluator.default_enabled`). It reads
``evaluation.variants`` (``galleries``, ``baselines``, ``reference``,
``mutual_proximity_max_nodes``); ``representations`` at the top level works as
for retrieval. Every item of the gallery acts as a query. ``reference`` is the
representation the others are compared with and on which near-duplicates are
detected (default: the main extractor).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from musicsim.config import ConfigError
from musicsim.evaluation.base import Evaluator
from musicsim.evaluation.bootstrap import paired_bootstrap_ci
from musicsim.evaluation.graph_retrieval import graph_per_query_metrics
from musicsim.evaluation.retrieval import DEFAULT_KS
from musicsim.graphs.base import Graph, build_graph
from musicsim.graphs.duplicates import duplicate_groups, near_duplicate_pairs
from musicsim.graphs.gallery import gallery_nodes
from musicsim.registry import register_evaluator
from musicsim.relevance.base import label_codes
from musicsim.representations import load_aligned_representations, stage_representations
from musicsim.runlog import RunLog
from musicsim.seeding import derive_seed

__all__ = [
    "GRAPH_VARIANTS",
    "HEADLINE_METRICS",
    "GraphVariantsEvaluator",
    "sign_consistency",
]

#: ``graph`` name -> overrides of the ``graph`` section. Closed list (C.5).
GRAPH_VARIANTS: dict[str, dict[str, Any]] = {
    "knn5": {"builder": "knn", "k": 5},
    "knn10": {"builder": "knn", "k": 10},
    "knn20": {"builder": "knn", "k": 20},
    "knn50": {"builder": "knn", "k": 50},
    "mutual_knn10": {"builder": "mutual_knn", "k": 10},
    "epsilon": {"builder": "epsilon", "k": 10, "epsilon": "matched"},
    "mutual_proximity10": {"builder": "mutual_proximity", "k": 10},
}

#: ``(metric, k)`` of the headline retrieval metrics (MAP has the sentinel k = 0).
HEADLINE_METRICS: tuple[tuple[str, int], ...] = (("R", 10), ("MAP", 0), ("nDCG", 10))

_HEADLINE_VARIANT = "dedup"
_DEFAULT_MP_MAX_NODES = 2000

_PER_QUERY_COLUMNS = [
    "item_id", "representation", "seed", "gallery", "graph", "variant",
    "metric", "k", "value", "completed",
]  # fmt: skip
_SUMMARY_COLUMNS = [
    "representation", "seed", "gallery", "graph", "variant", "metric", "k",
    "value", "completed_fraction", "isolated_fraction",
]  # fmt: skip
_DELTA_COLUMNS = [
    "representation", "seed", "reference", "gallery", "graph", "metric", "k",
    "delta", "ci_low", "ci_high", "sign", "n_resamples",
]  # fmt: skip


def _sign(ci_low: float, ci_high: float) -> int:
    """+1 / -1 when the interval excludes 0, else 0."""
    if ci_low > 0:
        return 1
    if ci_high < 0:
        return -1
    return 0


def sign_consistency(deltas: pd.DataFrame) -> pd.DataFrame:
    """Per comparison and metric, whether the sign of the delta is the same in every graph.

    ``deltas`` is the frame of ``variants_deltas.csv``. Returns one row per
    ``(gallery, representation, seed, reference, metric, k)`` with ``n_graphs``,
    ``same_sign`` (a single sign across all of them, 0 included) and ``sign``
    (that sign, or ``nan`` when it differs).
    """
    keys = ["gallery", "representation", "seed", "reference", "metric", "k"]
    rows = []
    for key, frame in deltas.groupby(keys, sort=False):
        signs = set(frame["sign"].tolist())
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_graphs": len(frame),
                "same_sign": len(signs) == 1,
                "sign": next(iter(signs)) if len(signs) == 1 else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=[*keys, "n_graphs", "same_sign", "sign"])


def _isolated_fraction(graph: Graph) -> float:
    return float((graph.degree == 0).mean())


@register_evaluator("variants")
class GraphVariantsEvaluator(Evaluator):
    """Retrieval quality of every representation under every graph variant."""

    def evaluate(self, index: pd.DataFrame, run: RunLog) -> None:
        from musicsim.registry import RELEVANCES

        stage_cfg = self.config.section("evaluation").get("variants", {})
        main_name = self.config.require("extractor.name")
        specs = stage_representations(self.config, stage_cfg, main_name, "variants")
        labels = [spec.label for spec in specs]
        reference = stage_cfg.get("reference", main_name)
        if reference not in labels:
            raise ConfigError(
                f"evaluation.variants.reference is {reference!r}; expected one of {labels}"
            )
        max_nodes = int(stage_cfg.get("mutual_proximity_max_nodes", _DEFAULT_MP_MAX_NODES))

        loaded = load_aligned_representations(self.config, index, specs)
        relevance = RELEVANCES.get(self.config.get("retrieval.relevance", "label_match"))(
            self.config
        )
        ks = sorted({int(k) for k in self.config.get("retrieval.ks", DEFAULT_KS)})
        for metric, k in HEADLINE_METRICS:
            if k and k not in ks:
                raise ConfigError(
                    f"retrieval.ks is {ks}; the headline metric {metric}@{k} needs {k}"
                )
        threshold = float(self.config.get("graph.near_duplicate_threshold", 0.9999))
        n_resamples = int(self.config.get("bootstrap.n_resamples", 1000))
        confidence = float(self.config.get("bootstrap.confidence", 0.95))
        experiment_seed = int(self.config.require("seed"))
        graph_section = dict(self.config.section("graph"))

        galleries = stage_cfg.get("galleries", self.config.get("retrieval.gallery", "fma_test"))
        if isinstance(galleries, str):
            galleries = [galleries]

        per_query_frames: list[pd.DataFrame] = []
        summary_rows: list[dict] = []
        delta_rows: list[dict] = []
        for gallery_name in galleries:
            nodes = gallery_nodes(
                loaded, gallery_name, reference=reference, threshold=threshold,
                collapse_duplicates=False,
            )  # fmt: skip
            ids, n = nodes.ids, len(nodes.ids)
            if n > max_nodes:
                raise ConfigError(
                    f"gallery {gallery_name!r} has {n} items; mutual_proximity is cubic and "
                    f"evaluation.variants.mutual_proximity_max_nodes is {max_nodes}, so "
                    f"expected a gallery of at most {max_nodes} items"
                )
            gallery_index = loaded.index.loc[ids]
            artists = gallery_index["artist_id"].to_numpy()
            genre = relevance.label_codes(index, ids)
            pairs = near_duplicate_pairs(loaded.X[reference][nodes.rows], ids, threshold=threshold)
            groups = {
                "dedup": duplicate_groups(ids, pairs),
                "raw": np.arange(n, dtype=np.int64),
                "artist_filter": label_codes(gallery_index["artist_id"]),
            }
            rows = np.arange(n)

            # dedup values per (graph, representation, metric, k), item order of ``ids``.
            headline: dict[tuple[str, str, str, int], np.ndarray] = {}
            for label in labels:
                X = loaded.X[label][nodes.rows]
                seed = loaded.seeds[label]
                for graph_name, overrides in GRAPH_VARIANTS.items():
                    graph = build_graph(X, ids, {**graph_section, **overrides})
                    isolated = _isolated_fraction(graph)
                    for variant, group in groups.items():
                        frame, completed = graph_per_query_metrics(
                            graph, X, ids, rows, group, genre, ks
                        )
                        frame["completed"] = (
                            pd.Series(completed, index=ids[rows]).loc[frame["item_id"]].to_numpy()
                        )
                        frame["representation"] = label
                        frame["seed"] = seed
                        frame["gallery"] = gallery_name
                        frame["graph"] = graph_name
                        frame["variant"] = variant
                        per_query_frames.append(frame[_PER_QUERY_COLUMNS])

                        summary = frame.groupby(["metric", "k"], sort=False).agg(
                            value=("value", "mean"), completed_fraction=("completed", "mean")
                        )
                        for (metric, k), row in summary.iterrows():
                            summary_rows.append(
                                {
                                    "representation": label,
                                    "seed": seed,
                                    "gallery": gallery_name,
                                    "graph": graph_name,
                                    "variant": variant,
                                    "metric": metric,
                                    "k": k,
                                    "value": row["value"],
                                    "completed_fraction": row["completed_fraction"],
                                    "isolated_fraction": isolated,
                                }
                            )
                        if variant == _HEADLINE_VARIANT:
                            for metric, k in HEADLINE_METRICS:
                                sel = frame[(frame["metric"] == metric) & (frame["k"] == k)]
                                headline[(graph_name, label, metric, k)] = (
                                    sel.set_index("item_id")["value"].loc[ids].to_numpy()
                                )
                run.logger.info("variants: %s: %s done", gallery_name, label)

            for graph_name in GRAPH_VARIANTS:
                for metric, k in HEADLINE_METRICS:
                    # Same seed for every comparison of this (graph, metric): same artists.
                    seed = derive_seed(
                        experiment_seed, "variants", gallery_name, graph_name, metric, k
                    )
                    ref_values = headline[(graph_name, reference, metric, k)]
                    for label in labels:
                        if label == reference:
                            continue
                        result = paired_bootstrap_ci(
                            headline[(graph_name, label, metric, k)],
                            ref_values,
                            n_resamples=n_resamples,
                            confidence=confidence,
                            seed=seed,
                            groups=artists,
                        )
                        delta_rows.append(
                            {
                                "representation": label,
                                "seed": loaded.seeds[label],
                                "reference": reference,
                                "gallery": gallery_name,
                                "graph": graph_name,
                                "metric": metric,
                                "k": k,
                                "delta": result.mean,
                                "ci_low": result.ci_low,
                                "ci_high": result.ci_high,
                                "sign": _sign(result.ci_low, result.ci_high),
                                "n_resamples": n_resamples,
                            }
                        )

        pd.concat(per_query_frames, ignore_index=True).to_csv(
            run.metrics_dir / "variants_per_query.csv", index=False
        )
        pd.DataFrame(summary_rows, columns=_SUMMARY_COLUMNS).to_csv(
            run.metrics_dir / "variants_summary.csv", index=False
        )
        deltas = pd.DataFrame(delta_rows, columns=_DELTA_COLUMNS)
        deltas.to_csv(run.metrics_dir / "variants_deltas.csv", index=False)

        self._log_sign_consistency(run, deltas)

    @staticmethod
    def _log_sign_consistency(run: RunLog, deltas: pd.DataFrame) -> None:
        for row in sign_consistency(deltas).itertuples(index=False):
            outcome = f"same sign ({int(row.sign):+d})" if row.same_sign else "sign changes"
            run.logger.info(
                "variants: %s: %s vs %s, %s@%d: %s over %d graph variant(s)",
                row.gallery, row.representation, row.reference, row.metric, row.k,
                outcome, row.n_graphs,
            )  # fmt: skip
