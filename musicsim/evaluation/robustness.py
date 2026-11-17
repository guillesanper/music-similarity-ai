"""Robustness evaluation: does a clip's neighbourhood change when its audio is degraded?

Protocol
--------
The gallery is the *original* clips of a gallery (``fma_test`` by default); a
query is a perturbed clip (:mod:`musicsim.audio_perturbations`). The question is
whether the neighbourhood moves, not whether a clip finds itself, so the
original of a query is excluded from the candidates together with its
near-duplicate group (the ``dedup`` rule of retrieval). The original's own
neighbourhood is ranked under the same exclusion, so both top-10 lists are drawn
from the same candidates and an unchanged clip has Jaccard@10 = 1.

Per ``(representation, condition)`` and per query:

``jaccard``
    size of the intersection over size of the union of the top-10 of the original
    and of the perturbed clip.
``dR``, ``dMAP``, ``dnDCG``
    R@10, MAP and nDCG@10 of the perturbed query minus those of the original
    one, both scored by :func:`musicsim.evaluation.retrieval.metrics_from_order`
    (genre relevance, ``dedup`` exclusion).
``original_rank``
    1-based position of the original clip in the ranking of its perturbed
    version. The original is *not* excluded here (that would make the position
    undefined) but its other near-duplicates are, so a duplicate cannot hide it.

Design decisions
----------------
* **The null condition runs through the same path as the others.** Gallery and
  queries are both computed from the decoded waveform. Embeddings from the
  retrieval cache come from float16 spectrograms and would differ from a fresh
  float32 computation by ~1e-3, so a null perturbation would not give
  Jaccard@10 = 1. The null (written once, as ``perturbation=none, level=0``) is
  the control that proves the pipeline is exact; every delta is relative to it,
  which is why deltas are not exactly the figures of the retrieval experiment.
* **The scaler and the PCA are the ones of the retrieval pipeline.** The
  features of every item (from the feature cache) are stacked with the gallery
  and perturbed vectors, under synthetic ids whose split is ``perturbed``, and
  :func:`musicsim.embeddings.build_embeddings` runs once. Since it fits on
  ``embedding.fit_on`` only, the transform is fitted on exactly the same rows
  as in retrieval and the perturbed vectors never influence it.
* **Cache: features, not audio.** A perturbed 30 s clip is 2.6 MB of float32,
  ~2 GB per condition for 800 clips; its feature vector is a few hundred bytes.
  Vectors are cached per condition, keyed by the feature sections of the
  configuration and the ``(perturbation, level)``; the item ids stored next to
  them must match the gallery or the condition is recomputed.
* **Bootstrap.** Intervals resample artists; the resample indices depend on
  ``(gallery, condition, metric)`` but not on the representation, so
  representations are compared on the same artists.

The evaluator is off unless ``evaluation.robustness.enabled`` is set. It reads
``evaluation.robustness``: ``galleries``, ``reference`` (representation on which
near-duplicates are detected, default the main extractor) and ``conditions``:
``default`` and, per representation label, either ``all`` (the 13 conditions of
:data:`musicsim.audio_perturbations.PERTURBATIONS`) or a list of one-key
mappings such as ``{noise_snr_db: 20}``. The null condition is always run.

Outputs under ``metrics/``: ``robustness.csv`` (``representation, seed, gallery,
perturbation, level, metric, k, value, ci_low, ci_high, n_queries``; metrics
``jaccard``, ``dR``, ``dMAP``, ``dnDCG``, ``original_rank`` (mean) and
``original_top10`` (fraction with rank <= 10)) and ``robustness_per_query.csv``
(``item_id, representation, seed, gallery, perturbation, level, metric, k,
value``, without ``original_top10``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from musicsim import paths
from musicsim.audio_perturbations import NULL_LEVELS, PERTURBATIONS
from musicsim.config import Config, ConfigError, config_hash
from musicsim.embeddings import (
    _FEATURE_SECTIONS,
    EmbeddingError,
    build_embeddings,
    features_cache_dir,
)
from musicsim.evaluation.base import Evaluator
from musicsim.evaluation.bootstrap import bootstrap_ci
from musicsim.evaluation.retrieval import metrics_from_order
from musicsim.graphs.duplicates import duplicate_groups, near_duplicate_pairs
from musicsim.graphs.gallery import gallery_nodes
from musicsim.registry import EXTRACTORS, register_evaluator
from musicsim.representations import (
    load_aligned_representations,
    spec_config,
    stage_representations,
)
from musicsim.runlog import RunLog
from musicsim.seeding import derive_seed

__all__ = [
    "NULL_CONDITION",
    "RobustnessEvaluator",
    "jaccard_at_k",
    "original_rank",
    "parse_conditions",
    "rank_gallery",
]

#: ``(perturbation, level)`` label of the control condition, written once for all perturbations.
NULL_CONDITION: tuple[str, float] = ("none", 0.0)

#: The null condition as :func:`musicsim.audio_perturbations.perturb` understands it.
_NULL_EXTRACTION: tuple[str, float] = ("gain_db", NULL_LEVELS["gain_db"])

_DEPTH = 10  # the headline neighbourhood size: Jaccard@10, R@10, nDCG@10
_PER_QUERY_COLUMNS = [
    "item_id", "representation", "seed", "gallery", "perturbation", "level", "metric", "k", "value",
]  # fmt: skip
_SUMMARY_COLUMNS = [
    "representation", "seed", "gallery", "perturbation", "level", "metric", "k", "value",
    "ci_low", "ci_high", "n_queries",
]  # fmt: skip


def rank_gallery(Q: np.ndarray, G: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Full gallery ranking, best first, of every query ``Q[i]`` (a stand-in for gallery row ``i``).

    ``Q`` and ``G`` are ``(N, d)`` and L2-normalised, so the similarity is a dot
    product. Candidates in the exclusion group of row ``i`` (itself included)
    are pushed to the end, which is where
    :func:`~musicsim.evaluation.retrieval.metrics_from_order` expects them.
    """
    S = (Q @ G.T).astype(np.float64)
    S[group[:, None] == group[None, :]] = -np.inf
    return np.argsort(-S, axis=1, kind="stable")


def jaccard_at_k(order_a: np.ndarray, order_b: np.ndarray, k: int) -> np.ndarray:
    """Per-row Jaccard index of the top-``k`` of two row-aligned rankings."""
    if order_a.shape != order_b.shape:
        raise ValueError(
            f"rankings must have the same shape, got {order_a.shape} and {order_b.shape}"
        )
    a, b = order_a[:, :k], order_b[:, :k]
    shared = (a[:, :, None] == b[:, None, :]).any(axis=2).sum(axis=1)
    return shared / (2 * k - shared)


def original_rank(Q: np.ndarray, G: np.ndarray, group: np.ndarray) -> np.ndarray:
    """1-based rank of gallery row ``i`` in the ranking of query ``Q[i]``.

    Only the *other* members of the group of ``i`` are removed from the
    ranking; ``i`` itself stays. Ties do not count against the original.
    """
    S = (Q @ G.T).astype(np.float64)
    own = np.diag(S).copy()
    excluded = (group[:, None] == group[None, :]) & ~np.eye(len(G), dtype=bool)
    return 1 + ((own[:, None] < S) & ~excluded).sum(axis=1)


def parse_conditions(spec: object, label: str) -> list[tuple[str, float]]:
    """Expand a ``conditions`` entry (``all`` or a list of ``{name: level}``) of ``label``."""
    if spec == "all":
        return [(name, float(level)) for name, levels in PERTURBATIONS.items() for level in levels]
    if isinstance(spec, str) or not isinstance(spec, Sequence):
        raise ConfigError(
            f"evaluation.robustness.conditions for {label!r} must be 'all' or a list of "
            f"{{perturbation: level}} mappings, got {spec!r}"
        )
    parsed: list[tuple[str, float]] = []
    for entry in spec:
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise ConfigError(
                f"a condition of {label!r} is a mapping with one key like {{noise_snr_db: 20}}, "
                f"got {entry!r}"
            )
        ((name, level),) = entry.items()
        if name not in PERTURBATIONS or float(level) not in PERTURBATIONS[name]:
            raise ConfigError(
                f"condition {name}: {level!r} of {label!r} is not in the closed list; "
                f"expected a perturbation of {sorted(PERTURBATIONS)} with a level of "
                f"{ {n: list(v) for n, v in PERTURBATIONS.items()} }"
            )
        parsed.append((str(name), float(level)))
    if len(set(parsed)) != len(parsed):
        raise ConfigError(f"conditions of {label!r} repeat: {parsed}")
    return parsed


def _conditions_for(stage_cfg: Mapping, label: str) -> list[tuple[str, float]]:
    conditions = stage_cfg.get("conditions", {})
    if not isinstance(conditions, Mapping):
        raise ConfigError(f"evaluation.robustness.conditions must be a mapping, got {conditions!r}")
    return parse_conditions(conditions.get(label, conditions.get("default", "all")), label)


def _condition_cache_dir(rep_config: Config, condition: tuple[str, float], *, create: bool):
    resolved = rep_config.to_dict()
    key = config_hash(
        {
            **{name: resolved.get(name, {}) for name in _FEATURE_SECTIONS},
            "condition": [condition[0], condition[1]],
        }
    )
    extractor = rep_config.require("extractor.name")
    return paths.cache_dir(
        rep_config.require("dataset.directory"), f"robustness-{extractor}", key, create=create
    )


def _condition_vectors(
    rep_config: Config, gallery_frame: pd.DataFrame, conditions: Sequence[tuple[str, float]]
) -> dict[tuple[str, float], np.ndarray]:
    """Raw ``(N, dim)`` vectors of the gallery under every condition, from cache or computed."""
    ids = gallery_frame["item_id"].to_numpy()
    cache = bool(rep_config.get("runtime.cache", True))
    vectors: dict[tuple[str, float], np.ndarray] = {}
    missing: list[tuple[str, float]] = []
    for condition in conditions:
        directory = _condition_cache_dir(rep_config, condition, create=False)
        cached = (directory / "features.npy").is_file() and (directory / "ids.npy").is_file()
        if cache and cached and np.array_equal(np.load(directory / "ids.npy"), ids):
            vectors[condition] = np.load(directory / "features.npy")
            continue
        missing.append(condition)
    if not missing:
        return vectors

    extractor = EXTRACTORS.get(rep_config.require("extractor.name"))(rep_config)
    X, got_ids, failures = extractor.extract_perturbed(
        gallery_frame,
        [_NULL_EXTRACTION if c == NULL_CONDITION else c for c in missing],
        n_jobs=int(rep_config.get("runtime.n_jobs", -1)),
        progress=bool(rep_config.get("runtime.progress", True)),
    )
    if failures:
        raise ConfigError(
            f"{len(failures)} item(s) could not be extracted under the perturbations of "
            f"{rep_config.require('extractor.name')!r}, e.g. {failures[:3]}; "
            "every representation must be evaluated on the same gallery"
        )
    for position, condition in enumerate(missing):
        vectors[condition] = X[position]
        if cache:
            directory = _condition_cache_dir(rep_config, condition, create=True)
            np.save(directory / "features.npy", X[position])
            np.save(directory / "ids.npy", got_ids)
    return vectors


def _embed(
    rep_config: Config,
    index: pd.DataFrame,
    vectors: Mapping[tuple[str, float], np.ndarray],
    conditions: Sequence[tuple[str, float]],
) -> dict[tuple[str, float], np.ndarray]:
    """Embeddings of the gallery vectors with the scaler and PCA of the retrieval pipeline."""
    features_dir = features_cache_dir(rep_config)
    raw = np.load(features_dir / "features.npy")
    raw_ids = np.load(features_dir / "ids.npy")

    n = len(next(iter(vectors.values())))
    first_id = int(index["item_id"].max()) + 1
    extra_ids = first_id + np.arange(n * len(conditions))
    stacked = np.concatenate([raw, *(vectors[c] for c in conditions)])
    stacked_ids = np.concatenate([raw_ids, extra_ids])
    extended = pd.concat(
        [
            index[["item_id", "split"]],
            pd.DataFrame({"item_id": extra_ids, "split": "perturbed"}),
        ],
        ignore_index=True,
    )
    try:
        Z, _ = build_embeddings(stacked, stacked_ids, extended, rep_config)
    except EmbeddingError as exc:
        raise ConfigError(str(exc)) from exc
    Z = Z[len(raw) :]
    return {c: Z[i * n : (i + 1) * n] for i, c in enumerate(conditions)}


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
        loaded = load_aligned_representations(self.config, index, specs)
        relevance = RELEVANCES.get(self.config.get("retrieval.relevance", "label_match"))(
            self.config
        )
        threshold = float(self.config.get("graph.near_duplicate_threshold", 0.9999))
        n_resamples = int(self.config.get("bootstrap.n_resamples", 1000))
        confidence = float(self.config.get("bootstrap.confidence", 0.95))
        experiment_seed = int(self.config.require("seed"))
        galleries = stage_cfg.get("galleries", self.config.get("retrieval.gallery", "fma_test"))
        if isinstance(galleries, str):
            galleries = [galleries]

        per_query_frames: list[pd.DataFrame] = []
        summary_rows: list[dict] = []
        for gallery_name in galleries:
            nodes = gallery_nodes(
                loaded, gallery_name, reference=reference, threshold=threshold,
                collapse_duplicates=False,
            )  # fmt: skip
            ids, n = nodes.ids, len(nodes.ids)
            artists = loaded.index.loc[ids, "artist_id"].to_numpy()
            genre = relevance.label_codes(index, ids)
            pairs = near_duplicate_pairs(loaded.X[reference][nodes.rows], ids, threshold=threshold)
            group = duplicate_groups(ids, pairs)
            candidates = n - np.bincount(group)[group]
            if candidates.min() < _DEPTH:
                raise ConfigError(
                    f"gallery {gallery_name!r} leaves a query with {int(candidates.min())} "
                    f"candidate(s) after exclusion; expected at least {_DEPTH}"
                )
            gallery_frame = loaded.index.loc[ids].reset_index()[["item_id", "path"]]
            rows = np.arange(n)

            for spec in specs:
                label = spec.label
                rep_config = spec_config(self.config, spec)
                conditions = [NULL_CONDITION, *_conditions_for(stage_cfg, label)]
                vectors = _condition_vectors(rep_config, gallery_frame, conditions)
                embedded = _embed(rep_config, index, vectors, conditions)
                G = embedded[NULL_CONDITION]
                order_original = rank_gallery(G, G, group)
                original = _headline(
                    metrics_from_order(order_original, rows, ids, group, genre, [_DEPTH])
                )
                seed = loaded.seeds[label]

                for condition in conditions:
                    Q = embedded[condition]
                    order = rank_gallery(Q, G, group)
                    perturbed = _headline(
                        metrics_from_order(order, rows, ids, group, genre, [_DEPTH])
                    )
                    per_query = {
                        ("jaccard", _DEPTH): jaccard_at_k(order_original, order, _DEPTH),
                        ("dR", _DEPTH): perturbed["R"] - original["R"],
                        ("dMAP", 0): perturbed["MAP"] - original["MAP"],
                        ("dnDCG", _DEPTH): perturbed["nDCG"] - original["nDCG"],
                        ("original_rank", 0): original_rank(Q, G, group).astype(np.float64),
                    }
                    name, level = condition
                    for (metric, k), values in per_query.items():
                        per_query_frames.append(
                            pd.DataFrame(
                                {
                                    "item_id": ids, "representation": label, "seed": seed,
                                    "gallery": gallery_name, "perturbation": name, "level": level,
                                    "metric": metric, "k": k, "value": values,
                                }
                            )[_PER_QUERY_COLUMNS]
                        )  # fmt: skip
                    summarised = {
                        **per_query,
                        ("original_top10", _DEPTH): (
                            per_query[("original_rank", 0)] <= _DEPTH
                        ).astype(np.float64),
                    }
                    for (metric, k), values in summarised.items():
                        result = bootstrap_ci(
                            values, n_resamples=n_resamples, confidence=confidence,
                            seed=derive_seed(
                                experiment_seed, "robustness", gallery_name, name, level, metric, k
                            ),
                            groups=artists,
                        )  # fmt: skip
                        summary_rows.append(
                            {
                                "representation": label, "seed": seed, "gallery": gallery_name,
                                "perturbation": name, "level": level, "metric": metric, "k": k,
                                "value": result.mean, "ci_low": result.ci_low,
                                "ci_high": result.ci_high, "n_queries": n,
                            }
                        )  # fmt: skip
                run.logger.info(
                    "robustness: %s: %s: %d condition(s) done", gallery_name, label, len(conditions)
                )

        pd.concat(per_query_frames, ignore_index=True).to_csv(
            run.metrics_dir / "robustness_per_query.csv", index=False
        )
        pd.DataFrame(summary_rows, columns=_SUMMARY_COLUMNS).to_csv(
            run.metrics_dir / "robustness.csv", index=False
        )


def _headline(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """``R``, ``MAP`` and ``nDCG`` at the headline depth, one value per query, in query order."""
    out: dict[str, np.ndarray] = {}
    for metric, k in (("R", _DEPTH), ("MAP", 0), ("nDCG", _DEPTH)):
        out[metric] = frame[(frame["metric"] == metric) & (frame["k"] == k)]["value"].to_numpy()
    return out
