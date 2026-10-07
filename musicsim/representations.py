"""Several representations of the same items, extracted, embedded and aligned.

An experiment names its representations in a top-level ``representations:``
list. Each entry is either an extractor name (label = the name, no overrides)
or a mapping ``{label, extractor, overrides}``, where ``overrides`` are dotted
configuration keys applied on top of the experiment's configuration. One
mechanism therefore covers a baseline (``random``), several seeds of the same
extractor, a PCA sweep (``embedding.pca.n_components``) and a no-L2 variant
(``embedding.l2_normalize: false``), without a dedicated key for each.

Overrides may not touch ``dataset``: every representation must describe the
same items, and aligning them relies on that. They may not touch
``representations`` either, which would make the list recursive.

:func:`load_aligned_representations` runs the extract and embed stages for each
specification and aligns the results on the sorted intersection of their item
ids. Alignment is always by id, never by position.

``musicsim.experiments`` is imported inside the functions that need it: it
orchestrates the evaluators, which import this module, so a module-level import
would be circular (the same arrangement as in the retrieval evaluator).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from musicsim.config import Config, ConfigError

__all__ = [
    "AlignedRepresentations",
    "RepresentationSpec",
    "load_aligned_representations",
    "parse_representations",
    "spec_config",
]

_LABEL = re.compile(r"[a-z0-9_]+")
_ENTRY_KEYS = frozenset({"label", "extractor", "overrides"})


@dataclasses.dataclass(frozen=True, slots=True)
class RepresentationSpec:
    """One representation to evaluate."""

    label: str  # goes to the `representation` column and to directory names
    extractor: str  # a name under configs/extractors/
    overrides: Mapping[str, Any]  # dotted keys -> values, applied on top of the experiment config


@dataclasses.dataclass(frozen=True, slots=True)
class AlignedRepresentations:
    """Embeddings of several representations over the same items, in the same order."""

    X: Mapping[str, np.ndarray]  # label -> (N, d_label), rows aligned with `ids`
    ids: np.ndarray  # (N,), sorted item ids common to every representation
    index: pd.DataFrame  # the item index restricted to `ids`, indexed by item_id, same order
    seeds: Mapping[str, int]  # label -> the `seed` its config resolved to


def parse_representations(config: Config) -> list[RepresentationSpec]:
    """The specifications under ``representations:``, validated.

    Without that key the only representation is the main extractor.
    """
    raw = config.get("representations")
    if raw is None:
        name = str(config.require("extractor.name"))
        return [RepresentationSpec(label=name, extractor=name, overrides={})]
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise ConfigError(f"'representations' must be a list, got {raw!r}")

    specs = [_parse_entry(entry) for entry in raw]
    if not specs:
        raise ConfigError("'representations' is empty; list at least one representation")
    labels = [spec.label for spec in specs]
    duplicated = sorted({label for label in labels if labels.count(label) > 1})
    if duplicated:
        raise ConfigError(f"representation labels must be unique, repeated: {duplicated}")
    return specs


def _parse_entry(entry: Any) -> RepresentationSpec:
    if isinstance(entry, str):
        label, extractor, overrides = entry, entry, {}
    elif isinstance(entry, Mapping):
        unknown = sorted(set(entry) - _ENTRY_KEYS)
        if unknown:
            raise ConfigError(
                f"representation entry has unknown key(s) {unknown}; "
                f"expected {sorted(_ENTRY_KEYS)}"
            )
        for key in ("label", "extractor"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise ConfigError(
                    f"representation entry needs a non-empty string {key!r}: {dict(entry)!r}"
                )
        label, extractor = entry["label"], entry["extractor"]
        overrides = entry.get("overrides") or {}
        if not isinstance(overrides, Mapping):
            raise ConfigError(f"'overrides' of {label!r} must be a mapping, got {overrides!r}")
    else:
        raise ConfigError(
            f"a representation is an extractor name or a mapping, got {type(entry).__name__}"
        )

    if not _LABEL.fullmatch(label):
        raise ConfigError(f"representation label {label!r} must match [a-z0-9_]+")
    for key in overrides:
        if key == "dataset" or key.startswith("dataset.") or key.startswith("representations"):
            raise ConfigError(
                f"override {key!r} of representation {label!r} is not allowed: "
                "every representation must share the dataset and cannot redefine the list"
            )
    return RepresentationSpec(label=label, extractor=extractor, overrides=dict(overrides))


def spec_config(config: Config, spec: RepresentationSpec) -> Config:
    """``config`` with ``spec``'s extractor swapped in and its overrides applied."""
    from musicsim.experiments import representation_config

    swapped = representation_config(config, spec.extractor)
    if not spec.overrides:
        return swapped
    data = swapped.to_dict()
    for dotted, value in spec.overrides.items():
        node = data
        *parents, leaf = dotted.split(".")
        for part in parents:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[leaf] = value
    return Config(data=data, source=swapped.source)


def load_aligned_representations(
    config: Config, index: pd.DataFrame, specs: Sequence[RepresentationSpec]
) -> AlignedRepresentations:
    """Extract, embed and align every specification on the common item ids."""
    from musicsim.embeddings import load_embeddings
    from musicsim.experiments import embed_stage, extract_stage

    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    seeds: dict[str, int] = {}
    for spec in specs:
        rep_config = spec_config(config, spec)
        extract_stage(rep_config)
        embed_stage(rep_config)
        loaded[spec.label] = load_embeddings(rep_config)
        seeds[spec.label] = int(rep_config.require("seed"))

    common: np.ndarray | None = None
    for _, ids in loaded.values():
        common = ids if common is None else np.intersect1d(common, ids)
    if common is None or len(common) == 0:
        raise ValueError("no item id is common to every representation being evaluated")
    common = np.sort(common)

    aligned: dict[str, np.ndarray] = {}
    for label, (X, ids) in loaded.items():
        position = pd.Series(np.arange(len(ids)), index=ids)
        aligned[label] = X[position.loc[common].to_numpy()]

    index_common = index.set_index("item_id").loc[common]
    return AlignedRepresentations(X=aligned, ids=common, index=index_common, seeds=seeds)
