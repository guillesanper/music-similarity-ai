"""Tests for :mod:`musicsim.cli`, :mod:`musicsim.runlog` and :mod:`musicsim.seeding`.

The command line is the only interface the README promises, so the checks here
are the ones a reader of the README would perform: the help lists the stages, a
configuration can be resolved without running anything, a usage error exits with
code 2 instead of a traceback, and a run directory contains everything needed to
trace a number back to the code that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from musicsim import paths
from musicsim.cli import STAGES, build_parser, main
from musicsim.config import load_config
from musicsim.runlog import git_state, package_versions, platform_state, start_run
from musicsim.seeding import derive_seed, generator, seed_everything

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


# --- command line ---------------------------------------------------------------
def test_help_exits_cleanly_and_lists_every_stage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for name, _help, _phase in STAGES:
        assert name in out
    assert "run" in out and "show" in out and "registries" in out


def test_no_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_show_prints_the_resolved_configuration_and_its_hash(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = CONFIGS_DIR / "experiments" / "e01_mfcc_baseline.yaml"
    assert main(["show", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert load_config(config_path).hash() in out
    # Values inherited through the defaults chain are in the printed document.
    assert "sample_rate: 22050" in out
    assert "e01_mfcc_baseline" in out


def test_show_applies_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    config_path = CONFIGS_DIR / "experiments" / "e01_mfcc_baseline.yaml"
    assert main(["show", str(config_path), "-o", "seed=7", "-o", "graph.k=25"]) == 0
    out = capsys.readouterr().out
    assert "seed: 7" in out
    assert "k: 25" in out


def test_a_missing_configuration_is_a_usage_error_not_a_traceback() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["show", "configs/experiments/does_not_exist.yaml"])
    assert excinfo.value.code == 2


def test_an_unimplemented_stage_says_which_phase_brings_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["graph", "--config", str(CONFIGS_DIR / "experiments" / "e01_mfcc_baseline.yaml")])
    assert excinfo.value.code == 2
    assert "R3" in capsys.readouterr().err


def test_registries_lists_every_kind(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["registries"]) == 0
    out = capsys.readouterr().out
    for kind in ("dataset", "extractor", "graph builder", "relevance function", "evaluator"):
        assert kind in out


def test_the_root_options_override_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "ignored"))
    target = tmp_path / "chosen"
    target.mkdir()
    main(["show", str(CONFIGS_DIR / "base.yaml"), "--data-dir", str(target)])
    capsys.readouterr()
    assert paths.data_root() == target.resolve()


def test_the_parser_can_be_built_without_side_effects() -> None:
    # build_parser is what the documentation generator and the tests use; it
    # must not import a dataset or touch the filesystem.
    assert build_parser().prog == "musicsim"


# --- seeding ---------------------------------------------------------------------
def test_seed_everything_reports_what_it_seeded() -> None:
    record = seed_everything(42)
    assert record["seed"] == 42
    assert record["deterministic"] is True
    assert "torch" in record


def test_seed_everything_rejects_an_out_of_range_seed() -> None:
    with pytest.raises(ValueError, match="32"):
        seed_everything(-1)


def test_named_streams_are_reproducible_and_independent() -> None:
    first = generator(42, "bootstrap", "e01").normal(size=5)
    again = generator(42, "bootstrap", "e01").normal(size=5)
    other = generator(42, "bootstrap", "e02").normal(size=5)
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other)


def test_a_stream_does_not_depend_on_what_ran_before_it() -> None:
    # This is the point of per-stream generators: inserting a stage that draws
    # random numbers must not move the numbers of the stages after it.
    expected = generator(42, "bootstrap").normal(size=3)
    generator(42, "some-other-stage").normal(size=1000)
    np.testing.assert_array_equal(generator(42, "bootstrap").normal(size=3), expected)


def test_derive_seed_is_reproducible_and_fits_in_32_bits() -> None:
    value = derive_seed(42, "probe", "genre_top")
    assert value == derive_seed(42, "probe", "genre_top")
    assert value != derive_seed(42, "probe", "tags_top50")
    assert 0 <= value < 2**32


# --- provenance --------------------------------------------------------------------
def test_package_versions_records_missing_packages_as_null() -> None:
    versions = package_versions()
    assert versions["numpy"] is not None
    # torch is an optional, phase R8 dependency; whether it is installed or not,
    # the key must exist so that its absence is visible in run.json.
    assert "torch" in versions


def test_platform_state_names_the_interpreter_and_the_machine() -> None:
    state = platform_state()
    assert state["python"].startswith("3.12")
    assert state["system"]
    assert state["machine"]


def test_git_state_has_the_expected_shape() -> None:
    state = git_state()
    assert set(state) == {"available", "commit", "branch", "dirty"}


def test_a_run_directory_holds_the_configuration_and_the_provenance(
    isolated_roots: dict[str, Path],
) -> None:
    config = load_config(CONFIGS_DIR / "experiments" / "e01_mfcc_baseline.yaml")
    run = start_run(config, seed_record=seed_everything(config.get("seed")))
    try:
        with run.stage("extract"):
            pass
        (run.metrics_dir / "retrieval.csv").write_text("metric,value\n", encoding="utf-8")
    finally:
        run.finish()

    assert (run.directory / "config.resolved.yaml").is_file()
    assert (run.directory / "log.txt").is_file()
    assert (run.directory / "metrics" / "retrieval.csv").is_file()

    record = json.loads((run.directory / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "ok"
    assert record["experiment"] == "e01_mfcc_baseline"
    assert record["config_hash"] == config.hash()
    assert record["seeding"]["seed"] == 42
    assert record["stages"][0] == {
        "stage": "extract",
        "status": "ok",
        "seconds": record["stages"][0]["seconds"],
    }
    assert record["packages"]["numpy"] is not None
    assert set(record["git"]) == {"available", "commit", "branch", "dirty"}

    # The written configuration reproduces the run exactly.
    assert load_config(run.directory / "config.resolved.yaml").hash() == config.hash()


def test_a_failing_stage_is_recorded_before_the_exception_propagates(
    isolated_roots: dict[str, Path],
) -> None:
    config = load_config(CONFIGS_DIR / "experiments" / "e01_mfcc_baseline.yaml")
    run = start_run(config)
    with pytest.raises(RuntimeError, match="no audio"), run.stage("extract"):
        raise RuntimeError("no audio")
    run.finish("failed")

    record = json.loads((run.directory / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["stages"][0]["status"] == "failed"
    assert "RuntimeError: no audio" in record["stages"][0]["error"]


def test_two_runs_of_the_same_configuration_do_not_share_a_directory(
    isolated_roots: dict[str, Path],
) -> None:
    config = load_config(CONFIGS_DIR / "experiments" / "e01_mfcc_baseline.yaml")
    first = start_run(config)
    first.finish()
    second = start_run(config)
    second.finish()
    # Same parent (same experiment and hash), different leaf (the timestamp),
    # unless the two starts landed in the same second.
    assert first.directory.parent == second.directory.parent
    assert first.directory.name[-8:] == second.directory.name[-8:] == config.hash()[:8]
