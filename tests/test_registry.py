"""Tests for :mod:`musicsim.registry`.

The registry is what turns "add a model" into "add a class and a YAML file". The
behaviour worth pinning down is therefore: a name resolves to exactly one
implementation, a duplicate name is an error rather than a silent overwrite, and
an unknown name produces a message that lists the alternatives.
"""

from __future__ import annotations

import pytest

from musicsim.registry import (
    DATASETS,
    EVALUATORS,
    EXTRACTORS,
    GRAPH_BUILDERS,
    RELEVANCES,
    Registry,
    load_plugins,
    register_extractor,
)

ALL_REGISTRIES = (DATASETS, EXTRACTORS, GRAPH_BUILDERS, RELEVANCES, EVALUATORS)


# --- basics ------------------------------------------------------------------
def test_register_and_get_round_trip() -> None:
    registry: Registry = Registry("thing")

    class Thing:
        pass

    assert registry.register("thing-a", Thing) is Thing
    assert registry.get("thing-a") is Thing
    assert "thing-a" in registry
    assert len(registry) == 1


def test_the_decorator_returns_the_class_unchanged() -> None:
    @register_extractor("toy")
    class ToyExtractor:
        pass

    # The decorated class must still be usable as a class, not a wrapper.
    assert EXTRACTORS.get("toy") is ToyExtractor
    assert ToyExtractor().__class__ is ToyExtractor


def test_names_are_sorted_and_iterable() -> None:
    registry: Registry = Registry("thing")
    for name in ("zulu", "alpha", "mike"):
        registry.register(name, object())
    assert registry.names() == ["alpha", "mike", "zulu"]
    assert list(registry) == ["alpha", "mike", "zulu"]


# --- errors ------------------------------------------------------------------
def test_a_duplicate_name_is_refused() -> None:
    registry: Registry = Registry("extractor")
    registry.register("mfcc", object())
    with pytest.raises(KeyError, match="already registered"):
        registry.register("mfcc", object())


def test_re_registering_the_same_object_is_allowed() -> None:
    # Importing one module twice under different names must not break.
    registry: Registry = Registry("extractor")
    entry = object()
    registry.register("mfcc", entry)
    registry.register("mfcc", entry)
    assert len(registry) == 1


def test_an_unknown_name_lists_the_available_ones() -> None:
    registry: Registry = Registry("extractor")
    registry.register("mfcc", object())
    registry.register("random", object())
    with pytest.raises(KeyError) as excinfo:
        registry.get("mffc")
    message = str(excinfo.value)
    assert "mffc" in message
    assert "mfcc" in message and "random" in message


def test_an_empty_registry_says_so_instead_of_showing_an_empty_list() -> None:
    registry: Registry = Registry("evaluator")
    with pytest.raises(KeyError, match="none registered"):
        registry.get("anything")


@pytest.mark.parametrize("bad", ["", None, 3])
def test_a_name_must_be_a_non_empty_string(bad: object) -> None:
    registry: Registry = Registry("thing")
    with pytest.raises(ValueError, match="non-empty string"):
        registry.register(bad, object())  # type: ignore[arg-type]


# --- the module-level registries ----------------------------------------------
def test_the_registries_are_distinct_namespaces() -> None:
    # The same name may mean an extractor and a relevance function; they must
    # not collide.
    EXTRACTORS.register("shared-name", object())
    RELEVANCES.register("shared-name", object())
    assert EXTRACTORS.get("shared-name") is not RELEVANCES.get("shared-name")


def test_every_registry_reports_its_kind_in_its_errors() -> None:
    for registry in ALL_REGISTRIES:
        with pytest.raises(KeyError) as excinfo:
            registry.get("definitely-not-registered")
        assert registry.kind in str(excinfo.value)


# --- plugin loading ------------------------------------------------------------
def test_load_plugins_skips_modules_that_do_not_exist_yet() -> None:
    # Phases R1 onwards add these sub-packages one at a time; until then the
    # command line still has to work.
    assert load_plugins(("musicsim.definitely_not_a_module",)) == []


def test_load_plugins_imports_what_is_there() -> None:
    assert load_plugins(("musicsim.config",)) == ["musicsim.config"]


def test_a_real_import_error_is_not_swallowed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A module that exists but imports something missing is a bug, not an
    # unimplemented phase, and must not be hidden.
    package = tmp_path / "brokenpkg"
    package.mkdir()
    (package / "__init__.py").write_text("import a_module_that_does_not_exist\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError, match="a_module_that_does_not_exist"):
        load_plugins(("brokenpkg",))
