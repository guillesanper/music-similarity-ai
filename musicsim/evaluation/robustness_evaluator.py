"""Robustness evaluation: does a clip's neighbourhood change when its audio is degraded?

Protocol
--------
The gallery is the *original* clips of a gallery (``fma_test`` by default); a
query is the perturbed clip (:mod:`musicsim.perturbations`). The original of a
query and its near-duplicates are excluded from its candidates, so what is
measured is whether the neighbourhood moves, not whether a clip finds itself
(:mod:`musicsim.evaluation.robustness`).

For each representation the scaler and the PCA are fitted once, on the
``embedding.fit_on`` split of the original features
(:func:`musicsim.embeddings.fit_embedding`), and the features of every
perturbed clip go through those same fitted parameters.

The comparison baseline is the ``identity`` condition, never the cached
original features. The cached originals may come from a log-mel spectrogram
re-read from cache in float16 while every perturbed clip is computed in float32;
comparing against ``identity`` makes the whole difference measured due to the
perturbation. That float16 vs float32 gap is reported on its own, as the
control row ``jaccard@10_vs_cached_original`` of the ``identity`` condition: the
Jaccard@10 between the neighbours of the ``identity`` query and those of the
cached original (expected to be very close to 1).

Outputs under ``metrics/``
--------------------------
``robustness.csv``
    ``representation, seed, gallery, condition, level, metric, value, ci_low,
    ci_high``. Metrics: ``jaccard@10`` (mean over queries of the Jaccard index of
    the top-10 of the ``identity`` query and of the perturbed one; interval by
    artist); ``delta_R@10``, ``delta_MAP``, ``delta_nDCG@10`` (perturbed minus
    ``identity``, with the artist-clustered paired interval of
    :func:`musicsim.evaluation.deltas.paired_query_deltas`); ``self_rank_median``
    and ``self_rank_top1_fraction`` (position of the original over the whole
    gallery, without interval); and the control row. The ``identity`` condition
    must give Jaccard@10 = 1 and every delta exactly 0.
``robustness_per_query.csv``
    ``item_id, representation, seed, gallery, condition, level, metric, value``
    with the per-query ``jaccard@10``, ``R@10``, ``MAP``, ``nDCG@10`` and
    ``self_rank`` the summary is built from.

Retrieval metrics come from :func:`musicsim.evaluation.retrieval.metrics_from_order`
on the ranking of each perturbed query over the gallery (candidates in the
query's exclusion group last), so they are the project's R@10, MAP and nDCG@10.

Configuration (``evaluation.robustness``): ``gallery``, ``reference``
(representation on which near-duplicates are detected, default the main
extractor) and ``conditions``, a list of ``{condition, level}`` that must
include ``identity``. The evaluator is off unless ``enabled`` is set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from musicsim.config import Config, ConfigError
from musicsim.embeddings import features_cache_dir, fit_embedding
from musicsim.evaluation.base import Evaluator
from musicsim.evaluation.bootstrap import bootstrap_ci
from musicsim.evaluation.deltas import paired_query_deltas
from musicsim.evaluation.retrieval import metrics_from_order
from musicsim.evaluation.robustness import neighbor_set_jaccard, query_neighbors, self_rank
from musicsim.graphs.duplicates import duplicate_groups, near_duplicate_pairs
from musicsim.graphs.gallery import gallery_nodes
from musicsim.perturbations import CONDITIONS, perturbed_features
from musicsim.registry import register_evaluator
from musicsim.representations import (
    load_aligned_representations,
    spec_config,
    stage_representations,
)
from musicsim.runlog import RunLog
from musicsim.seeding import derive_seed

__all__ = ["RobustnessEvaluator", "parse_conditions"]

_DEPTH = 10  # neighbourhood size of every headline metric
_IDENTITY = "identity"
_PER_QUERY_COLUMNS = [
    "item_id", "representation", "seed", "gallery", "condition", "level", "metric", "value",
]  # fmt: skip
_SUMMARY_COLUMNS = [
    "representation", "seed", "gallery", "condition", "level", "metric", "value", "ci_low",
    "ci_high",
]  # fmt: skip
_RAW_METRICS = {("R", _DEPTH): "R@10", ("MAP", 0): "MAP", ("nDCG", _DEPTH): "nDCG@10"}


def parse_conditions(spec: object) -> list[tuple[str, float]]:
    """``[(condition, level)]`` from the configured list; ``identity`` first.

    Each entry is a mapping ``{condition, level}``; ``level`` may be omitted for
    ``identity`` (it is ignored there and recorded as 0). The list must contain
    ``identity``, the baseline of every comparison, and no repeated pair.
    """
    if isinstance(spec, str) or not isinstance(spec, Sequence) or not spec:
        raise ConfigError(
            f"evaluation.robustness.conditions must be a non-empty list of "
            f"{{condition, level}} mappings, got {spec!r}"
        )
    parsed: list[tuple[str, float]] = []
    for entry in spec:
        if not isinstance(entry, Mapping) or "condition" not in entry:
            raise ConfigError(
                f"a condition is a mapping with 'condition' and 'level', got {entry!r}"
            )
        condition = entry["condition"]
        if condition not in CONDITIONS:
            raise ConfigError(
                f"unknown condition {condition!r}; expected one of {list(CONDITIONS)}"
            )
        if condition != _IDENTITY and "level" not in entry:
            raise ConfigError(f"condition {condition!r} needs a 'level', got {entry!r}")
        level = 0.0 if condition == _IDENTITY else float(entry["level"])
        parsed.append((str(condition), level))
    if len(set(parsed)) != len(parsed):
        raise ConfigError(f"conditions repeat: {parsed}")
    if (_IDENTITY, 0.0) not in parsed:
        raise ConfigError(
            f"conditions must include {_IDENTITY!r}, the baseline of every comparison; got {parsed}"
        )
    parsed.remove((_IDENTITY, 0.0))
    return [(_IDENTITY, 0.0), *parsed]


def _full_order(Q: np.ndarray, Z: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Gallery ranking of every query, best first, its exclusion group in the last positions."""
    S = (Q @ Z.T).astype(np.float64)
    S[group[:, None] == group[None, :]] = -np.inf
    return np.argsort(-S, axis=1, kind="stable")


def _variant(condition: str, level: float) -> str:
    return f"{condition}|{level:g}"


@register_evaluator("robustness")
class RobustnessEvaluator(Evaluator):
    """Stability of the neighbourhoods of every representation under audio perturbations."""

    def evaluate(self, index: pd.DataFrame, run: RunLog) -> None:
        from musicsim.registry import RELEVANCES

        stage_cfg = self.config.section("evaluation").get("robustness", {})
        main_name = self.config.require("extractor.name")
        specs = stage_representations(self.config, stage_cfg, main_name, "robustness")
        labels = [spec.label for spec in specs]
        reference = stage_cfg.get("reference", main_name)
        if reference not in labels:
            raise ConfigError(
                f"evaluation.robustness.reference is {reference!r}; expected one of {labels}"
            )
        conditions = parse_conditions(stage_cfg.get("conditions"))
        gallery_name = stage_cfg.get("gallery", self.config.get("retrieval.gallery", "fma_test"))

        loaded = load_aligned_representations(self.config, index, specs)
        relevance = RELEVANCES.get(self.config.get("retrieval.relevance", "label_match"))(
            self.config
        )
        threshold = float(self.config.get("graph.near_duplicate_threshold", 0.9999))
        n_resamples = int(self.config.get("bootstrap.n_resamples", 1000))
        confidence = float(self.config.get("bootstrap.confidence", 0.95))
        experiment_seed = int(self.config.require("seed"))

        nodes = gallery_nodes(
            loaded, gallery_name, reference=reference, threshold=threshold,
            collapse_duplicates=False,
        )  # fmt: skip
        ids, n = nodes.ids, len(nodes.ids)
        artists = loaded.index.loc[ids, "artist_id"].to_numpy()
        genre = relevance.label_codes(index, ids)
        pairs = near_duplicate_pairs(loaded.X[reference][nodes.rows], ids, threshold=threshold)
        group = duplicate_groups(ids, pairs)
        gallery_frame = loaded.index.loc[ids].reset_index()[["item_id", "path"]]
        rows = np.arange(n)

        per_query_frames: list[pd.DataFrame] = []
        summary_rows: list[dict] = []
        for spec in specs:
            label, seed = spec.label, loaded.seeds[spec.label]
            rep_config = spec_config(self.config, spec)
            Z = loaded.X[label][nodes.rows]
            transform = _fit_transform(rep_config, index)

            neighbours_original = query_neighbors(Z, Z, group, _DEPTH)
            queries: dict[tuple[str, float], np.ndarray] = {}
            for condition, level in conditions:
                X_raw, got_ids, failures = perturbed_features(
                    rep_config, gallery_frame, condition, level
                )
                if failures:
                    raise ConfigError(
                        f"{len(failures)} item(s) could not be extracted under "
                        f"{condition}={level:g} for {label!r}, e.g. {failures[:3]}; every "
                        "condition must cover the whole gallery"
                    )
                if not np.array_equal(got_ids, ids):
                    raise ConfigError(f"perturbed features of {label!r} do not cover the gallery")
                queries[(condition, level)] = transform.transform(X_raw)
            neighbours_identity = query_neighbors(queries[(_IDENTITY, 0.0)], Z, group, _DEPTH)

            for condition, level in conditions:
                Q = queries[(condition, level)]
                neighbours = query_neighbors(Q, Z, group, _DEPTH)
                frame = metrics_from_order(
                    _full_order(Q, Z, group), rows, ids, group, genre, [_DEPTH]
                )
                values = {
                    "jaccard@10": neighbor_set_jaccard(neighbours_identity, neighbours),
                    "self_rank": self_rank(Q, Z).astype(np.float64),
                }
                for (metric, k), name in _RAW_METRICS.items():
                    values[name] = frame[(frame["metric"] == metric) & (frame["k"] == k)][
                        "value"
                    ].to_numpy()
                for metric, vector in values.items():
                    per_query_frames.append(
                        pd.DataFrame(
                            {
                                "item_id": ids, "representation": label, "seed": seed,
                                "gallery": gallery_name, "condition": condition, "level": level,
                                "metric": metric, "value": vector,
                            }
                        )[_PER_QUERY_COLUMNS]
                    )  # fmt: skip

                summary_rows.append(
                    self._interval_row(
                        values["jaccard@10"], "jaccard@10", label, seed, gallery_name,
                        condition, level, artists, n_resamples, confidence, experiment_seed,
                    )
                )  # fmt: skip
                ranks = values["self_rank"]
                for metric, value in (
                    ("self_rank_median", float(np.median(ranks))),
                    ("self_rank_top1_fraction", float(np.mean(ranks == 1))),
                ):
                    summary_rows.append(
                        {
                            "representation": label, "seed": seed, "gallery": gallery_name,
                            "condition": condition, "level": level, "metric": metric,
                            "value": value, "ci_low": np.nan, "ci_high": np.nan,
                        }
                    )  # fmt: skip
                if condition == _IDENTITY:
                    summary_rows.append(
                        self._interval_row(
                            neighbor_set_jaccard(neighbours_original, neighbours),
                            "jaccard@10_vs_cached_original", label, seed, gallery_name,
                            condition, level, artists, n_resamples, confidence, experiment_seed,
                        )
                    )  # fmt: skip
            run.logger.info(
                "robustness: %s: %s: %d condition(s)", gallery_name, label, len(conditions)
            )

        per_query = pd.concat(per_query_frames, ignore_index=True)
        per_query.to_csv(run.metrics_dir / "robustness_per_query.csv", index=False)
        summary_rows += self._delta_rows(
            per_query, conditions, loaded.index["artist_id"], n_resamples, confidence,
            experiment_seed,
        )  # fmt: skip
        pd.DataFrame(summary_rows, columns=_SUMMARY_COLUMNS).to_csv(
            run.metrics_dir / "robustness.csv", index=False
        )

    @staticmethod
    def _interval_row(
        values: np.ndarray, metric: str, label: str, seed: int, gallery: str, condition: str,
        level: float, artists: np.ndarray, n_resamples: int, confidence: float,
        experiment_seed: int,
    ) -> dict:  # fmt: skip
        # The seed does not depend on the representation: same artists resampled for all.
        result = bootstrap_ci(
            values, n_resamples=n_resamples, confidence=confidence,
            seed=derive_seed(experiment_seed, "robustness", gallery, condition, str(level), metric),
            groups=artists,
        )  # fmt: skip
        return {
            "representation": label, "seed": seed, "gallery": gallery, "condition": condition,
            "level": level, "metric": metric, "value": result.mean, "ci_low": result.ci_low,
            "ci_high": result.ci_high,
        }  # fmt: skip

    @staticmethod
    def _delta_rows(
        per_query: pd.DataFrame, conditions: Sequence[tuple[str, float]], artist_by_item: pd.Series,
        n_resamples: int, confidence: float, experiment_seed: int,
    ) -> list[dict]:  # fmt: skip
        raw = per_query[per_query["metric"].isin(_RAW_METRICS.values())].copy()
        raw["variant"] = [
            _variant(c, lv) for c, lv in zip(raw["condition"], raw["level"], strict=True)
        ]
        baseline = _variant(_IDENTITY, 0.0)
        deltas = paired_query_deltas(
            raw, by="variant", pairs=[(_variant(c, lv), baseline) for c, lv in conditions],
            within=["representation", "gallery", "seed", "metric"], artist_by_item=artist_by_item,
            n_resamples=n_resamples, confidence=confidence, seed=experiment_seed,
        )  # fmt: skip
        level_of = {_variant(c, lv): (c, lv) for c, lv in conditions}
        rows = []
        for row in deltas.itertuples(index=False):
            condition, level = level_of[row.a]
            rows.append(
                {
                    "representation": row.representation, "seed": row.seed, "gallery": row.gallery,
                    "condition": condition, "level": level,
                    "metric": f"delta_{row.metric}", "value": row.delta,
                    "ci_low": row.ci_low, "ci_high": row.ci_high,
                }
            )  # fmt: skip
        return rows


def _fit_transform(rep_config: Config, index: pd.DataFrame):
    """The embedding transform fitted on the original features of ``rep_config``."""
    directory = features_cache_dir(rep_config)
    transform, _ = fit_embedding(
        np.load(directory / "features.npy"), np.load(directory / "ids.npy"), index, rep_config
    )
    return transform
