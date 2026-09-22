"""Retrieval evaluation: Recall@K, MAP, nDCG@K and P@K, per query, with the
significance protocol of C.8 layered on top.

Ranking and relevance
----------------------
For a query, the gallery is ranked by cosine similarity to every other item
(the full gallery, not just the ``k`` neighbours the "graph" stage persists —
MAP needs the whole ranking, see below). A candidate is relevant when it
shares its ``genre_top`` with the query (:mod:`musicsim.relevance.label_match`,
the family-A proxy of C.10bis).

Variants (which candidates are removed from *both* the ranking and the
relevant set, alongside the query itself):

``dedup``
    the query's near-duplicate group (:mod:`musicsim.graphs.duplicates`).
    Headline variant: without it, the near-duplicate partner of a query is
    the trivial best "neighbour" and inflates every number.
``raw``
    nothing beyond the query itself.
``artist_filter``
    every track by the query's artist — an early look at the artist-disjoint
    split later phases evaluate on properly.

Full-depth metrics
------------------
Every query's *entire* gallery ranking is computed (not truncated to
``max(ks)``), which is what lets **MAP** be a true full-ranking average
precision (C.7) rather than a ``MAP@k`` approximation: with the exclusion
group forced to the bottom of the ranking (similarity ``-inf``), a fixed
``AP = sum_i precision_at_i * rel_i / n_relevant`` summed over the whole
gallery is exactly equal to summing only over the query's real candidates,
because every excluded or irrelevant position contributes zero to the sum.
**P@k**, **R@k** and **nDCG@k** then read off the same ranking's first ``k``
entries — one ranking, every metric, no separate depth bookkeeping.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from musicsim.evaluation.base import Evaluator
from musicsim.registry import register_evaluator
from musicsim.runlog import RunLog

__all__ = ["DEFAULT_KS", "RetrievalEvaluator", "compute_per_query_metrics", "query_rows"]

#: k = 10 is the headline neighbourhood size (R@10, nDCG@10); the rest are
#: secondary, kept for continuity with the archived pipeline's P@k figures.
DEFAULT_KS: tuple[int, ...] = (1, 5, 10, 20)

#: Query rows processed per block: memory is O(block_rows * gallery), not
#: O(gallery**2).
_DEFAULT_BLOCK_ROWS = 512


def query_rows(ids: np.ndarray, index_common: pd.DataFrame, queries: str) -> np.ndarray:
    """Row positions (into ``ids``) that act as queries; the gallery is always all of them."""
    if queries == "all":
        return np.arange(len(ids))
    if queries == "test":
        return np.flatnonzero(index_common["split"].to_numpy() == "test")
    raise ValueError(f"retrieval.queries must be 'all' or 'test', got {queries!r}")


def compute_per_query_metrics(
    X: np.ndarray,
    ids: np.ndarray,
    rows: np.ndarray,
    group: np.ndarray,
    genre: np.ndarray,
    ks: Sequence[int],
    *,
    block_rows: int = _DEFAULT_BLOCK_ROWS,
) -> pd.DataFrame:
    """Long-format per-query metrics: columns ``item_id, metric, k, value``.

    ``X`` (L2-normalised, ``(N, d)``), ``ids``, ``group`` (exclusion group of
    each gallery row) and ``genre`` (relevance label code of each gallery row)
    are row-aligned over the whole gallery; ``rows`` selects which of them act
    as queries. ``MAP`` is reported once per query with ``k=0`` (a sentinel
    for "the full ranking", since it is not tied to a cutoff); ``P``, ``R``
    and ``nDCG`` are reported once per ``(query, k)`` pair.
    """
    ks = sorted({int(k) for k in ks})
    n = X.shape[0]
    if max(ks) >= n:
        raise ValueError(f"k={max(ks)} does not fit a gallery of {n} items")

    group = np.asarray(group)
    n_groups = int(group.max()) + 1
    group_sizes = np.bincount(group, minlength=n_groups)
    disc = 1.0 / np.log2(np.arange(2, max(ks) + 2))
    cum_disc = np.cumsum(disc)
    positions = np.arange(n)

    blocks: list[pd.DataFrame] = []
    for start in range(0, len(rows), block_rows):
        block = rows[start : start + block_rows]
        S = X[block] @ X.T
        same_group = group[block][:, None] == group[None, :]
        S = np.where(same_group, -np.inf, S)
        order = np.argsort(-S, axis=1, kind="stable")

        rel_full = genre[order] == genre[block][:, None]
        valid_len = n - group_sizes[group[block]]
        rel = rel_full & (positions[None, :] < valid_len[:, None])
        n_rel = rel.sum(axis=1)
        if (n_rel < 1).any():
            bad = ids[block[n_rel < 1]]
            raise ValueError(
                f"{len(bad)} quer{'y has' if len(bad) == 1 else 'ies have'} no relevant "
                f"candidate outside their own exclusion group, e.g. {bad[:3].tolist()}"
            )

        item_ids = ids[block]
        out: dict[str, list] = {"item_id": [], "metric": [], "k": [], "value": []}

        for k in ks:
            rel_k = rel[:, :k]
            _append(out, item_ids, "P", k, rel_k.mean(axis=1))
            _append(out, item_ids, "R", k, rel_k.sum(axis=1) / n_rel)
            dcg = (rel_k.astype(np.float64) * disc[:k][None, :]).sum(axis=1)
            idcg = cum_disc[np.minimum(k, n_rel) - 1]
            _append(out, item_ids, "nDCG", k, dcg / idcg)

        cum_rel = np.cumsum(rel, axis=1).astype(np.float64)
        precision_curve = cum_rel / np.arange(1, n + 1)
        average_precision = (precision_curve * rel).sum(axis=1) / n_rel
        _append(out, item_ids, "MAP", 0, average_precision)

        blocks.append(pd.DataFrame(out))

    return pd.concat(blocks, ignore_index=True)


def _append(
    out: dict[str, list], item_ids: np.ndarray, metric: str, k: int, values: np.ndarray
) -> None:
    out["item_id"].extend(item_ids.tolist())
    out["metric"].extend([metric] * len(values))
    out["k"].extend([k] * len(values))
    out["value"].extend(values.tolist())


def _align(X: np.ndarray, ids: np.ndarray, common_ids: np.ndarray) -> np.ndarray:
    """Rows of ``X`` reordered to ``common_ids``, dropping any id not in it."""
    position = pd.Series(np.arange(len(ids)), index=ids)
    return X[position.loc[common_ids].to_numpy()]


@register_evaluator("retrieval")
class RetrievalEvaluator(Evaluator):
    """Retrieval evaluation for the configured extractor plus its baselines.

    Reads ``evaluation.retrieval`` (``enabled``, ``queries``, ``baselines``)
    and the shared defaults under ``retrieval`` (``ks``, ``variants``,
    ``relevance``). Every representation (the main one and every baseline) is
    extracted, embedded and ranked through the exact same code path, so a
    negative control like ``random`` is "just another representation" rather
    than a special case.
    """

    def evaluate(self, index: pd.DataFrame, run: RunLog) -> None:
        # Deferred imports: musicsim.experiments orchestrates the stages
        # (including this evaluator, via the registry), so importing it at
        # module load time would be circular.
        from musicsim import seeding
        from musicsim.embeddings import load_embeddings
        from musicsim.evaluation.bootstrap import bootstrap_ci
        from musicsim.experiments import embed_stage, extract_stage, representation_config
        from musicsim.graphs.duplicates import duplicate_groups, near_duplicate_pairs
        from musicsim.registry import RELEVANCES
        from musicsim.relevance.base import label_codes

        stage_cfg = self.config.section("evaluation").get("retrieval", {})
        if not stage_cfg.get("enabled", True):
            return

        main_name = self.config.require("extractor.name")
        baselines = [name for name in stage_cfg.get("baselines", []) if name != main_name]
        names = [main_name, *baselines]

        embeddings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in names:
            rep_config = representation_config(self.config, name)
            extract_stage(rep_config)
            embed_stage(rep_config)
            embeddings[name] = load_embeddings(rep_config)

        common_ids = embeddings[main_name][1]
        for name in names[1:]:
            common_ids = np.intersect1d(common_ids, embeddings[name][1])
        common_ids = np.sort(common_ids)
        if len(common_ids) == 0:
            raise ValueError("no item id is common to every representation being evaluated")

        index_common = index.set_index("item_id").loc[common_ids]
        aligned = {name: _align(X, ids, common_ids) for name, (X, ids) in embeddings.items()}

        threshold = float(self.config.get("graph.near_duplicate_threshold", 0.9999))
        dup_pairs = near_duplicate_pairs(aligned[main_name], common_ids, threshold=threshold)
        dup_groups = duplicate_groups(common_ids, dup_pairs)

        relevance_name = self.config.get("retrieval.relevance", "label_match")
        relevance = RELEVANCES.get(relevance_name)(self.config)
        genre_codes = relevance.label_codes(index, common_ids)

        variant_groups = {
            "dedup": dup_groups,
            "raw": np.arange(len(common_ids), dtype=np.int64),
            "artist_filter": label_codes(index_common["artist_id"]),
        }
        variants = [
            name
            for name in self.config.get("retrieval.variants", list(variant_groups))
            if name in variant_groups
        ]
        ks = sorted({int(k) for k in self.config.get("retrieval.ks", DEFAULT_KS)})

        queries_modes = stage_cfg.get("queries", self.config.get("retrieval.queries", "all"))
        if isinstance(queries_modes, str):
            queries_modes = [queries_modes]

        seed = int(self.config.require("seed"))
        frames = []
        for queries_mode in queries_modes:
            rows = query_rows(common_ids, index_common, queries_mode)
            for name in names:
                X = aligned[name]
                for variant in variants:
                    frame = compute_per_query_metrics(
                        X, common_ids, rows, variant_groups[variant], genre_codes, ks
                    )
                    frame["representation"] = name
                    frame["variant"] = variant
                    frame["queries"] = queries_mode
                    frame["seed"] = seed
                    frames.append(frame)
        per_query = pd.concat(frames, ignore_index=True)
        per_query.to_csv(run.metrics_dir / "per_query.csv", index=False)

        summary = _summarize(per_query, index_common, self.config, bootstrap_ci, seeding)
        summary.to_csv(run.metrics_dir / "retrieval.csv", index=False)

        per_genre = _summarize_per_genre(per_query, index_common, main_name)
        per_genre.to_csv(run.metrics_dir / "retrieval_per_genre.csv", index=False)

        run.logger.info(
            "retrieval: %d representation(s), %d near-duplicate pair(s), "
            "%d row(s) of per_query.csv",
            len(names),
            len(dup_pairs),
            len(per_query),
        )


_SUMMARY_COLUMNS = (
    "representation",
    "variant",
    "queries",
    "genre",
    "metric",
    "k",
    "value",
    "ci_low",
    "ci_high",
    "ci_low_artist",
    "ci_high_artist",
)


def _summarize(per_query, index_common, config, bootstrap_ci, seeding) -> pd.DataFrame:
    """One row per ``(representation, variant, queries, metric, k)``, with two CIs.

    ``ci_low``/``ci_high`` resample queries; ``ci_low_artist``/``ci_high_artist``
    resample artists (with all of their queries together), which is the
    honest interval given that tracks by the same artist are not independent
    (C.8.1). Both use the same resample indices across every group sharing a
    ``(queries, metric, k)`` triple only incidentally — a real cross-model
    comparison is :func:`musicsim.evaluation.bootstrap.paired_bootstrap_ci`.
    """
    n_resamples = int(config.get("bootstrap.n_resamples", 1000))
    confidence = float(config.get("bootstrap.confidence", 0.95))
    seed = int(config.require("seed"))
    artist_by_id = index_common["artist_id"]

    records = []
    group_cols = ["representation", "variant", "queries", "metric", "k"]
    for key, group_df in per_query.groupby(group_cols, sort=False):
        representation, variant, queries_mode, metric, k = key
        values = group_df["value"].to_numpy()
        artists = artist_by_id.loc[group_df["item_id"].to_numpy()].to_numpy()
        stream = ("retrieval", representation, variant, queries_mode, metric, k)

        by_query = bootstrap_ci(
            values, n_resamples=n_resamples, confidence=confidence,
            seed=seeding.derive_seed(seed, *stream, "query"),
        )
        by_artist = bootstrap_ci(
            values, n_resamples=n_resamples, confidence=confidence,
            seed=seeding.derive_seed(seed, *stream, "artist"), groups=artists,
        )
        records.append(
            {
                "representation": representation,
                "variant": variant,
                "queries": queries_mode,
                "genre": "all",
                "metric": metric,
                "k": k,
                "value": by_query.mean,
                "ci_low": by_query.ci_low,
                "ci_high": by_query.ci_high,
                "ci_low_artist": by_artist.ci_low,
                "ci_high_artist": by_artist.ci_high,
            }
        )
    return pd.DataFrame.from_records(records, columns=list(_SUMMARY_COLUMNS))


def _summarize_per_genre(
    per_query: pd.DataFrame,
    index_common: pd.DataFrame,
    main_name: str,
    *,
    headline_k: int = 10,
    headline_variant: str = "dedup",
) -> pd.DataFrame:
    """``P@10`` of the main representation (``dedup``), broken down by genre."""
    mask = (
        (per_query["representation"] == main_name)
        & (per_query["metric"] == "P")
        & (per_query["k"] == headline_k)
        & (per_query["variant"] == headline_variant)
    )
    sub = per_query.loc[mask, ["queries", "item_id", "value"]].copy()
    sub["genre"] = index_common["genre_top"].loc[sub["item_id"].to_numpy()].to_numpy()
    return (
        sub.groupby(["queries", "genre"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": f"P@{headline_k}"})
    )
