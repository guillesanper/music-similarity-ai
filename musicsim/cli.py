"""Command line interface.

One subcommand per pipeline stage, plus ``run`` for a whole experiment::

    python -m musicsim download  --dataset fma_small
    python -m musicsim index     --dataset fma_small
    python -m musicsim extract   --config configs/extractors/mfcc.yaml
    python -m musicsim embed     --config configs/extractors/mfcc.yaml
    python -m musicsim graph     --config configs/experiments/e01_mfcc_baseline.yaml
    python -m musicsim evaluate  --config configs/experiments/e01_mfcc_baseline.yaml
    python -m musicsim run       configs/experiments/e01_mfcc_baseline.yaml
    python -m musicsim export    --run outputs/runs/e01_mfcc_baseline/<dir>

``run`` is the one that matters: it resolves a configuration, runs every stage
the configuration asks for and writes a run directory. The individual stage
subcommands exist to re-run one step during development; they share the same
configuration files, never a different code path.

Every subcommand accepts ``-o key=value`` overrides, applied on top of the
composed configuration, and every usage error exits with code 2 through
``parser.error`` rather than a traceback.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from musicsim import __version__, paths
from musicsim.config import Config, ConfigError, load_config
from musicsim.registry import (
    DATASETS,
    EVALUATORS,
    EXTRACTORS,
    GRAPH_BUILDERS,
    RELEVANCES,
    load_plugins,
)

__all__ = ["build_parser", "main"]

#: Stage subcommands, in pipeline order, with the phase that implements each.
#: Listing them here (rather than only in the parser) keeps ``--help`` honest
#: about what already works and what is still to come.
STAGES: tuple[tuple[str, str, str], ...] = (
    ("download", "fetch a dataset and verify its checksums", "P1"),
    ("index", "build the item index of a dataset", "P1"),
    ("extract", "compute raw features for every item", "P2"),
    ("embed", "standardise, optionally reduce and L2-normalise the features", "P2"),
    ("graph", "build the k-nearest-neighbour similarity graph", "P3"),
    ("evaluate", "run the evaluations the configuration asks for", "P3+"),
    ("export", "copy figures and tables of a run into report/", "P11"),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser, including every subcommand."""
    parser = argparse.ArgumentParser(
        prog="musicsim",
        description=(
            "Music similarity experiments: audio to embeddings to similarity graphs "
            "(crisp and fuzzy), comparing siamese networks, classical techniques and "
            "MERT, evaluated with retrieval, structure, probing and human triplet "
            "judgements."
        ),
        epilog=(
            "Every experiment is one YAML file under configs/experiments/. "
            "See README.md for the experiment table."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"musicsim {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    # -- run -------------------------------------------------------------
    run = subparsers.add_parser(
        "run",
        help="run a whole experiment and write a run directory",
        description="Resolve an experiment configuration, run its stages, write outputs/runs/...",
    )
    run.add_argument("config", type=Path, help="path to an experiment YAML file")
    _add_common(run)
    run.set_defaults(handler=_cmd_run)

    # -- stages ----------------------------------------------------------
    for name, help_text, phase in STAGES:
        stage = subparsers.add_parser(
            name,
            help=f"{help_text} [phase {phase}]",
            description=f"{help_text.capitalize()}. Implemented in phase {phase}.",
        )
        stage.add_argument(
            "--config",
            type=Path,
            required=(name not in {"download", "index", "export"}),
            help="path to a YAML configuration file",
        )
        if name in {"download", "index"}:
            stage.add_argument(
                "--dataset",
                help="dataset name, e.g. fma_small or mtt (alternative to --config)",
            )
        if name == "export":
            stage.add_argument("--run", type=Path, help="run directory to export from")
        _add_common(stage)
        stage.set_defaults(handler=_cmd_stage, stage=name, phase=phase)

    # -- introspection ---------------------------------------------------
    show = subparsers.add_parser(
        "show",
        help="print a resolved configuration and its hash, without running anything",
        description="Compose a configuration, apply overrides and print the result.",
    )
    show.add_argument("config", type=Path, help="path to a YAML configuration file")
    _add_common(show)
    show.set_defaults(handler=_cmd_show)

    registries = subparsers.add_parser(
        "registries",
        help="list the registered datasets, extractors, graph builders and evaluators",
        description="List every name a configuration may refer to.",
    )
    registries.set_defaults(handler=_cmd_registries)

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Options every configuration-driven subcommand shares."""
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a configuration value, e.g. -o seed=0 -o extractor.name=random "
        "(repeatable; the value is parsed as YAML)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=f"dataset root; overrides $MUSICSIM_DATA_DIR (default: {paths.data_root()})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"output root; overrides $MUSICSIM_OUTPUT_DIR (default: {paths.output_root()})",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log debug messages")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _cmd_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    config = _load(args, parser, args.config)
    try:
        from musicsim.experiments import run_experiment
    except ModuleNotFoundError:
        parser.error(
            "the experiment runner is not implemented yet (phase P3). "
            f"'musicsim show {args.config}' already resolves this configuration."
        )
    return int(run_experiment(config, verbose=args.verbose))


def _cmd_stage(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if getattr(args, "config", None) is not None:
        _load(args, parser, args.config)
    parser.error(
        f"stage '{args.stage}' is not implemented yet; it arrives in phase {args.phase}. "
        "See the phase table in README.md."
    )
    return 2  # unreachable: parser.error exits with code 2


def _cmd_show(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    config = _load(args, parser, args.config)
    sys.stdout.write(config.to_yaml())
    sys.stdout.write(f"\n# experiment: {config.name}\n# config hash: {config.hash()}\n")
    return 0


def _cmd_registries(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    load_plugins()
    for registry in (DATASETS, EXTRACTORS, GRAPH_BUILDERS, RELEVANCES, EVALUATORS):
        names = ", ".join(registry.names()) or "(none registered yet)"
        sys.stdout.write(f"{registry.kind:>16}: {names}\n")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load(args: argparse.Namespace, parser: argparse.ArgumentParser, path: Path) -> Config:
    """Load a configuration, turning any configuration error into a usage error."""
    try:
        return load_config(path, args.override)
    except ConfigError as exc:
        parser.error(str(exc))
        raise  # unreachable: parser.error exits


def _apply_roots(args: argparse.Namespace) -> None:
    """Let ``--data-dir`` / ``--output-dir`` win over the environment variables."""
    if getattr(args, "data_dir", None) is not None:
        os.environ[paths.ENV_DATA_DIR] = str(Path(args.data_dir).expanduser().resolve())
    if getattr(args, "output_dir", None) is not None:
        os.environ[paths.ENV_OUTPUT_DIR] = str(Path(args.output_dir).expanduser().resolve())


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_roots(args)
    try:
        return int(args.handler(args, parser))
    except ConfigError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through __main__.py
    raise SystemExit(main())
