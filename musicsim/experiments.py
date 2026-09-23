"""The experiment runner: resolves a configuration, runs its ``stages`` in
order, writes a run directory (:mod:`musicsim.runlog`).

Each stage function takes the resolved :class:`~musicsim.config.Config` (and,
past ``extract``/``embed``, the :class:`~musicsim.runlog.RunLog` it writes
into) and does exactly what the corresponding CLI subcommand does — this
module *is* that code, not a copy of it: :mod:`musicsim.cli` calls
``extract_stage``/``embed_stage`` too, so a stage behaves identically whether
it runs on its own (``musicsim extract ...``) or as part of ``musicsim run``.

:func:`representation_config` is what makes a "baseline" (e.g. ``random``
alongside ``mfcc`` in ``evaluation.retrieval.baselines``) just another
representation: it is the same configuration with only its ``extractor``
section swapped for another one under ``configs/extractors/``, so
``extract_stage``/``embed_stage`` run on it exactly as they would on the main
configuration.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from musicsim import paths, seeding
from musicsim.config import Config, ConfigError, compose
from musicsim.embeddings import EmbeddingError, build_embeddings
from musicsim.registry import EVALUATORS, EXTRACTORS, load_plugins
from musicsim.runlog import RunLog, start_run

__all__ = [
    "DEFAULT_STAGES",
    "EmbedResult",
    "ExtractResult",
    "embed_stage",
    "evaluate_stage",
    "extract_stage",
    "graph_stage",
    "representation_config",
    "run_experiment",
]

#: Stages run by ``musicsim run`` when a configuration does not list its own.
DEFAULT_STAGES: tuple[str, ...] = ("extract", "embed", "graph", "evaluate")


@dataclasses.dataclass(frozen=True, slots=True)
class ExtractResult:
    """What :func:`extract_stage` did, for the CLI to report and for tests."""

    cache_dir: Path
    n_items: int
    n_failed: int
    cached: bool


def extract_stage(config: Config) -> ExtractResult:
    """Compute or reuse cached raw features for ``config``'s dataset + extractor."""
    load_plugins()
    dataset = config.require("dataset.directory")
    extractor_name = config.require("extractor.name")
    try:
        extractor_cls = EXTRACTORS.get(extractor_name)
    except KeyError as exc:
        raise ConfigError(str(exc)) from exc
    extractor = extractor_cls(config)

    index_path = paths.dataset_index(dataset)
    if not index_path.is_file():
        raise ConfigError(f"no index at {index_path}; run `musicsim index --dataset ...` first")
    index = pd.read_csv(index_path)

    cache_dir = paths.cache_dir(dataset, f"features-{extractor.name}", config.hash(), create=True)
    features_path = cache_dir / "features.npy"
    ids_path = cache_dir / "ids.npy"

    if config.get("runtime.cache", True) and features_path.is_file() and ids_path.is_file():
        n_items = int(np.load(ids_path).shape[0])
        return ExtractResult(cache_dir=cache_dir, n_items=n_items, n_failed=0, cached=True)

    X, ids, failures = extractor.extract_all(
        index,
        n_jobs=int(config.get("runtime.n_jobs", -1)),
        progress=bool(config.get("runtime.progress", True)),
    )
    np.save(features_path, X)
    np.save(ids_path, ids)
    failures_payload = [{"item_id": item_id, "reason": reason} for item_id, reason in failures]
    failures_path = cache_dir / "failures.json"
    failures_path.write_text(json.dumps(failures_payload, indent=2), encoding="utf-8")

    return ExtractResult(
        cache_dir=cache_dir, n_items=len(ids), n_failed=len(failures), cached=False
    )


@dataclasses.dataclass(frozen=True, slots=True)
class EmbedResult:
    """What :func:`embed_stage` did, for the CLI to report and for tests."""

    cache_dir: Path
    n_items: int
    n_dims: int
    n_fit: int


def embed_stage(config: Config) -> EmbedResult:
    """Standardise, optionally reduce and L2-normalise the cached feature matrix."""
    load_plugins()
    dataset = config.require("dataset.directory")
    extractor_name = config.require("extractor.name")

    features_dir = paths.cache_dir(dataset, f"features-{extractor_name}", config.hash())
    features_path = features_dir / "features.npy"
    ids_path = features_dir / "ids.npy"
    if not features_path.is_file() or not ids_path.is_file():
        raise ConfigError(
            f"no cached features for extractor {extractor_name!r} at "
            f"{paths.relative_to_repo(features_dir)}; run the extract stage first"
        )

    X = np.load(features_path)
    ids = np.load(ids_path)
    index = pd.read_csv(paths.dataset_index(dataset))

    try:
        Z, info = build_embeddings(X, ids, index, config)
    except EmbeddingError as exc:
        raise ConfigError(str(exc)) from exc

    embed_dir = paths.cache_dir(dataset, f"embeddings-{extractor_name}", config.hash(), create=True)
    np.save(embed_dir / "embeddings.npy", Z)
    np.save(embed_dir / "ids.npy", ids)

    return EmbedResult(
        cache_dir=embed_dir, n_items=Z.shape[0], n_dims=Z.shape[1], n_fit=info["n_fit"]
    )


def representation_config(config: Config, extractor_name: str) -> Config:
    """``config`` with its ``extractor`` section replaced by another one's.

    Looks up ``configs/extractors/<extractor_name>.yaml`` (composing its own
    ``defaults:``, though the extractor files have none today) and splices its
    ``extractor:`` mapping into a copy of ``config``. Everything else —
    dataset, embedding post-processing, graph, retrieval parameters, seed —
    stays the same, so the baseline is evaluated under identical conditions.
    """
    if extractor_name == config.get("extractor.name"):
        return config
    extractor_path = paths.REPO_ROOT / "configs" / "extractors" / f"{extractor_name}.yaml"
    if not extractor_path.is_file():
        raise ConfigError(
            f"no configuration for extractor {extractor_name!r}: {extractor_path} not found"
        )
    extractor_data = compose(extractor_path)
    if "extractor" not in extractor_data:
        raise ConfigError(f"{extractor_path} has no top-level 'extractor:' section")

    data = config.to_dict()
    data["extractor"] = extractor_data["extractor"]
    return Config(data=data, source=config.source)


def graph_stage(config: Config, run: RunLog) -> None:
    """Build the canonical kNN graph and near-duplicate pairs, written into ``run``."""
    from musicsim.embeddings import load_embeddings
    from musicsim.graphs.base import build_graph
    from musicsim.graphs.duplicates import near_duplicate_pairs

    load_plugins()
    name = config.require("extractor.name")
    X, ids = load_embeddings(config)

    graph_cfg = config.section("graph")
    try:
        graph = build_graph(X, ids, graph_cfg)
    except KeyError as exc:
        raise ConfigError(str(exc)) from exc
    out_dir = run.directory / "graphs" / name
    graph.write(out_dir)

    threshold = float(graph_cfg.get("near_duplicate_threshold", 0.9999))
    pairs = near_duplicate_pairs(X, ids, threshold=threshold)
    pairs.to_csv(out_dir / "duplicates.csv", index=False)

    run.logger.info(
        "graph: %d nodes, k=%d, %d near-duplicate pair(s) (threshold %.4f)",
        graph.n_nodes,
        graph.k,
        len(pairs),
        threshold,
    )


def evaluate_stage(config: Config, run: RunLog) -> None:
    """Run every evaluator whose ``evaluation.<name>.enabled`` is not ``false``."""
    load_plugins()
    dataset = config.require("dataset.directory")
    index_path = paths.dataset_index(dataset)
    if not index_path.is_file():
        raise ConfigError(f"no index at {index_path}; run `musicsim index --dataset ...` first")
    index = pd.read_csv(index_path)

    ran_any = False
    for name in EVALUATORS.names():
        if not config.get(f"evaluation.{name}.enabled", True):
            continue
        evaluator = EVALUATORS.get(name)(config)
        evaluator.evaluate(index, run)
        ran_any = True

    if not ran_any:
        run.logger.info("evaluate: no evaluator is enabled for this configuration")


_STAGE_FUNCS = {
    "extract": lambda config, run: extract_stage(config),
    "embed": lambda config, run: embed_stage(config),
    "graph": graph_stage,
    "evaluate": evaluate_stage,
}


def run_experiment(
    config: Config, *, stages: Sequence[str] | None = None, verbose: bool = False
) -> RunLog:
    """Run ``stages`` (default: ``config["stages"]``), writing a run directory.

    Used both by ``musicsim run`` (every configured stage) and by the
    standalone ``musicsim graph``/``musicsim evaluate`` subcommands (one
    stage, in its own run directory) — never a different code path.
    """
    stages = list(stages) if stages is not None else list(config.get("stages", DEFAULT_STAGES))
    unknown = [name for name in stages if name not in _STAGE_FUNCS]
    if unknown:
        raise ConfigError(f"unknown stage(s): {unknown}; known: {list(_STAGE_FUNCS)}")

    seed_record = seeding.seed_everything(config.require("seed"))
    level = logging.DEBUG if verbose else logging.INFO
    run = start_run(config, seed_record=seed_record, level=level)
    try:
        for name in stages:
            with run.stage(name):
                _STAGE_FUNCS[name](config, run)
    except Exception:
        run.finish(status="failed")
        raise
    run.finish(status="ok")
    return run
