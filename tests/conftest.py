"""Shared fixtures: synthetic data and isolated roots.

No test touches a real corpus. The fixtures below build a small clustered
embedding space whose structure is known in advance, so that a retrieval or a
clustering metric computed on it has an answer that can be asserted rather than
eyeballed:

* ``n_groups`` well-separated Gaussian clusters play the role of genres;
* items carry a ``group_id`` (the near-duplicate / same-song group) and an
  ``artist_id``, so the exclusion rules and the clustered bootstrap have
  something to bite on;
* a fixed generator makes every value reproducible.

Path fixtures point :mod:`musicsim.paths` at a temporary directory through the
environment variables, so a test can never write into ``datasets/`` or
``outputs/`` of the working copy.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from musicsim import paths

# --- constants of the synthetic corpus --------------------------------------
N_ITEMS = 120
N_GROUPS = 4
N_DIMS = 8
CLUSTER_SPREAD = 0.35
SEED = 20260912


@pytest.fixture
def rng() -> np.random.Generator:
    """A generator with a fixed seed, so every test draws the same numbers."""
    return np.random.default_rng(SEED)


@pytest.fixture
def synthetic_index(rng: np.random.Generator) -> pd.DataFrame:
    """An item index with the standard columns the real datasets produce.

    Columns match :class:`musicsim.datasets.base.Dataset`: ``item_id``,
    ``path``, ``split``, ``artist_id``, ``group_id``, plus one label column.
    Every item of an artist shares the artist label, and consecutive pairs share
    a ``group_id``, which is what a near-duplicate pair looks like.
    """
    labels = np.repeat(np.arange(N_GROUPS), N_ITEMS // N_GROUPS)
    item_ids = np.arange(N_ITEMS)
    # Six items per artist, so an artist never spans two labels.
    artist_ids = item_ids // 6
    # Items 0/1, 2/3, ... belong to the same group: a query and its duplicate.
    group_ids = item_ids // 2
    splits = np.array(["training"] * N_ITEMS, dtype=object)
    splits[artist_ids % 5 == 3] = "validation"
    splits[artist_ids % 5 == 4] = "test"

    return pd.DataFrame(
        {
            "item_id": item_ids,
            "path": [f"synthetic/{i:04d}.wav" for i in item_ids],
            "split": splits,
            "artist_id": artist_ids,
            "group_id": group_ids,
            "label": [f"label_{value}" for value in labels],
        }
    )


@pytest.fixture
def synthetic_embeddings(
    synthetic_index: pd.DataFrame, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalised embeddings whose clusters follow the label column.

    Returns ``(X, ids)`` with rows aligned to ``synthetic_index`` row by row, the
    invariant the whole pipeline relies on: features, identifiers and index rows
    always share an order, and any cross-reference goes through the identifier.
    """
    labels = synthetic_index["label"].to_numpy()
    unique = np.unique(labels)
    centres = rng.normal(size=(len(unique), N_DIMS))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    index_of = {label: i for i, label in enumerate(unique)}
    offsets = rng.normal(scale=CLUSTER_SPREAD, size=(len(labels), N_DIMS))
    features = centres[[index_of[label] for label in labels]] + offsets
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return features.astype(np.float32), synthetic_index["item_id"].to_numpy()


# --- isolated roots ----------------------------------------------------------
@pytest.fixture
def isolated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Path]]:
    """Point the data and output roots at a temporary directory."""
    data = tmp_path / "datasets"
    output = tmp_path / "outputs"
    data.mkdir()
    output.mkdir()
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(data))
    monkeypatch.setenv(paths.ENV_OUTPUT_DIR, str(output))
    yield {"data": data.resolve(), "output": output.resolve()}


@pytest.fixture
def config_tree(tmp_path: Path) -> Path:
    """A miniature configs/ tree: a base file, a dataset file and an experiment.

    Used to exercise the ``defaults:`` composition without depending on the real
    configuration files, whose content changes as phases are added.
    """
    root = tmp_path / "configs"
    (root / "datasets").mkdir(parents=True)
    (root / "experiments").mkdir(parents=True)

    (root / "base.yaml").write_text(
        "seed: 42\n"
        "audio:\n  sample_rate: 22050\n  mono: true\n"
        "retrieval:\n  ks: [1, 5, 10]\n  queries: all\n",
        encoding="utf-8",
    )
    (root / "datasets" / "toy.yaml").write_text(
        "defaults:\n  - ../base.yaml\n"
        "dataset:\n  name: toy\n  directory: toy\n"
        "audio:\n  sample_rate: 16000\n",
        encoding="utf-8",
    )
    (root / "experiments" / "e00_toy.yaml").write_text(
        "defaults:\n  - ../base.yaml\n  - ../datasets/toy.yaml\n"
        "experiment: e00_toy\n"
        "retrieval:\n  ks: [10]\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture(autouse=True)
def _clean_registries() -> Iterator[None]:
    """Undo any registration a test performs, so tests stay independent."""
    from musicsim import registry

    registries = (
        registry.DATASETS,
        registry.EXTRACTORS,
        registry.GRAPH_BUILDERS,
        registry.RELEVANCES,
        registry.EVALUATORS,
    )
    saved = [dict(reg._entries) for reg in registries]
    yield
    for reg, entries in zip(registries, saved, strict=True):
        reg._entries.clear()
        reg._entries.update(entries)
