"""Tests for :mod:`musicsim.config`.

The properties that matter are the ones an experiment depends on: composition
order, deep merge, the fact that lists replace rather than append, override
parsing, and a configuration hash that is stable and sensitive to every value.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from musicsim.config import (
    Config,
    ConfigError,
    apply_overrides,
    compose,
    config_hash,
    deep_merge,
    load_config,
    parse_override,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


# --- composition -------------------------------------------------------------
def test_defaults_are_merged_and_the_including_file_wins(config_tree: Path) -> None:
    config = load_config(config_tree / "experiments" / "e00_toy.yaml")

    # Inherited from base.yaml, untouched by anyone.
    assert config.get("audio.mono") is True
    assert config.get("seed") == 42
    # toy.yaml overrides base.yaml, and the experiment does not touch it.
    assert config.get("audio.sample_rate") == 16000
    # The experiment file wins over everything it inherits.
    assert config.get("retrieval.ks") == (10,)
    # Keys the experiment does not mention survive the merge.
    assert config.get("retrieval.queries") == "all"
    assert config.get("dataset.name") == "toy"


def test_a_diamond_does_not_apply_the_same_file_twice(config_tree: Path) -> None:
    # e00_toy.yaml lists base.yaml directly *and* through toy.yaml. The value of
    # sample_rate shows which one landed last: toy.yaml must win, because
    # re-applying base.yaml would push 22050 back on top.
    config = load_config(config_tree / "experiments" / "e00_toy.yaml")
    assert config.get("audio.sample_rate") == 16000


def test_lists_replace_instead_of_appending() -> None:
    merged = deep_merge({"ks": [1, 5, 10], "a": {"b": 1}}, {"ks": [10], "a": {"c": 2}})
    assert merged["ks"] == [10]
    # Mappings, on the other hand, merge key by key.
    assert merged["a"] == {"b": 1, "c": 2}


def test_a_circular_defaults_chain_is_reported(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("defaults: [b.yaml]\nx: 1\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("defaults: [a.yaml]\ny: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="circular"):
        compose(tmp_path / "a.yaml")


def test_a_missing_file_is_reported_as_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_a_top_level_list_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        compose(path)


def test_an_empty_file_composes_to_an_empty_mapping(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert compose(path) == {}


# --- reading -----------------------------------------------------------------
def test_require_names_the_missing_key(config_tree: Path) -> None:
    config = load_config(config_tree / "experiments" / "e00_toy.yaml")
    assert config.require("seed") == 42
    with pytest.raises(ConfigError, match=r"graph\.k"):
        config.require("graph.k")


def test_get_returns_the_default_for_a_missing_path() -> None:
    config = Config({"a": {"b": 1}})
    assert config.get("a.b") == 1
    assert config.get("a.z", "fallback") == "fallback"
    assert config.get("z.y.x") is None
    # A dotted path through a non-mapping must not raise.
    assert config.get("a.b.c", "fallback") == "fallback"


def test_contains_uses_dotted_paths() -> None:
    config = Config({"a": {"b": 1}})
    assert "a.b" in config
    assert "a.c" not in config


def test_a_config_is_read_only() -> None:
    config = Config({"a": {"b": 1}})
    with pytest.raises(TypeError):
        config.data["a"] = 2  # type: ignore[index]
    # to_dict hands out a copy, so mutating it cannot reach the config.
    copy = config.to_dict()
    copy["a"]["b"] = 99
    assert config.get("a.b") == 1


def test_the_name_falls_back_to_the_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "e99_unnamed.yaml"
    path.write_text("seed: 1\n", encoding="utf-8")
    assert load_config(path).name == "e99_unnamed"


# --- overrides ---------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected_path", "expected_value"),
    [
        ("seed=0", ["seed"], 0),
        ("graph.k=20", ["graph", "k"], 20),
        ("embedding.pca.enabled=true", ["embedding", "pca", "enabled"], True),
        ("retrieval.ks=[1,5,10]", ["retrieval", "ks"], [1, 5, 10]),
        ("extractor.name=mfcc", ["extractor", "name"], "mfcc"),
        ("embedding.pca.n_components=null", ["embedding", "pca", "n_components"], None),
    ],
)
def test_override_values_are_parsed_as_yaml(
    text: str, expected_path: list[str], expected_value: object
) -> None:
    assert parse_override(text) == (expected_path, expected_value)


@pytest.mark.parametrize("text", ["seed", "=1", "a..b=1"])
def test_malformed_overrides_are_rejected(text: str) -> None:
    with pytest.raises(ConfigError):
        parse_override(text)


def test_overrides_create_missing_sections() -> None:
    result = apply_overrides({"seed": 42}, ["graph.k=20", "seed=0"])
    assert result == {"seed": 0, "graph": {"k": 20}}


def test_overrides_reach_load_config(config_tree: Path) -> None:
    config = load_config(
        config_tree / "experiments" / "e00_toy.yaml",
        ["seed=7", "audio.sample_rate=44100"],
    )
    assert config.get("seed") == 7
    assert config.get("audio.sample_rate") == 44100


# --- hashing -----------------------------------------------------------------
def test_the_hash_ignores_key_order() -> None:
    assert config_hash({"a": 1, "b": {"c": 2, "d": 3}}) == config_hash(
        {"b": {"d": 3, "c": 2}, "a": 1}
    )


def test_the_hash_changes_with_any_value() -> None:
    base = {"seed": 42, "graph": {"k": 10}}
    assert config_hash(base) != config_hash({"seed": 42, "graph": {"k": 11}})
    assert config_hash(base) != config_hash({"seed": 43, "graph": {"k": 10}})
    # An added key changes it too, even when every shared value is identical.
    assert config_hash(base) != config_hash({**base, "extra": None})


def test_the_hash_is_hexadecimal_and_long_enough_to_name_a_directory() -> None:
    digest = config_hash({"seed": 42})
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_a_frozen_config_hashes_like_its_plain_mapping() -> None:
    data = {"seed": 42, "ks": [1, 5, 10]}
    assert Config(data).hash() == config_hash(data)


# --- round trip --------------------------------------------------------------
def test_resolved_yaml_reloads_to_the_same_hash(config_tree: Path, tmp_path: Path) -> None:
    # This is the promise of config.resolved.yaml: re-running the file written
    # into a run directory reproduces that run exactly.
    config = load_config(config_tree / "experiments" / "e00_toy.yaml", ["seed=7"])
    written = config.write(tmp_path / "config.resolved.yaml")
    assert load_config(written).hash() == config.hash()


# --- the real configuration files ---------------------------------------------
def test_every_shipped_config_composes() -> None:
    files = sorted(CONFIGS_DIR.rglob("*.yaml"))
    assert files, "no configuration files found"
    for path in files:
        config = load_config(path)
        assert isinstance(config.to_dict(), dict)
        assert len(config.hash()) == 64


def test_every_experiment_declares_a_name_matching_its_file_name() -> None:
    for path in sorted((CONFIGS_DIR / "experiments").glob("*.yaml")):
        config = load_config(path)
        assert config.get("experiment") == path.stem, path.name


def test_every_experiment_inherits_base_dataset_and_extractor() -> None:
    for path in sorted((CONFIGS_DIR / "experiments").glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        parents = raw.get("defaults", [])
        assert any(p.endswith("base.yaml") for p in parents), path.name
        assert any("/datasets/" in p for p in parents), path.name
        assert any("/extractors/" in p for p in parents), path.name
        config = load_config(path)
        assert config.get("seed") is not None
        assert config.get("dataset.name") is not None
        assert config.get("extractor.name") is not None
