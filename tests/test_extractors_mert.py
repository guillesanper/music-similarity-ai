"""Tests for the `mert` extractor and its layer-selection helper.

None of these need the real model or network access: `MertExtractor.compute_layers`
is monkeypatched with a deterministic stand-in, so the tests exercise the
contract (shape, dtype, caching, layer selection) rather than the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from musicsim.config import Config
from musicsim.extractors.mert import MertExtractor, select_titular_layer


def _config(**overrides: dict) -> Config:
    data: dict = {
        "dataset": {"directory": "toy"},
        "audio": {
            "sample_rate": 8000,
            "clip_seconds": 1.0,
            "clip_samples": 8000,
            "min_seconds": 0.1,
        },
        "extractor": {
            "name": "mert",
            "model_name": "unused/for-these-tests",
            "window_seconds": 1.0,
            "n_layers": 13,
            "hidden_size": 768,
            "mert_layer": 6,
            "dim": 768,
        },
        "runtime": {"cache": True},
    }
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return Config(data=data)


def _fake_layers(item_id: int, n_layers: int = 13, hidden_size: int = 768) -> np.ndarray:
    """A deterministic, item-dependent (n_layers, hidden_size) float16 array."""
    rng = np.random.default_rng(item_id)
    return rng.normal(size=(n_layers, hidden_size)).astype(np.float16)


def test_extract_one_shape_and_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = MertExtractor(_config())
    monkeypatch.setattr(extractor, "compute_layers", lambda path: _fake_layers(1))

    vector = extractor.extract_one(1, "unused.mp3")

    assert vector.shape == (768,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_extract_one_selects_the_configured_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    layers = _fake_layers(1)
    extractor = MertExtractor(_config(extractor={"mert_layer": 3}))
    monkeypatch.setattr(extractor, "compute_layers", lambda path: layers)

    vector = extractor.extract_one(1, "unused.mp3")

    np.testing.assert_array_equal(vector, layers[3].astype(np.float32))


def test_extract_one_rejects_an_out_of_range_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = MertExtractor(_config(extractor={"mert_layer": 13}))
    monkeypatch.setattr(extractor, "compute_layers", lambda path: _fake_layers(1))

    with pytest.raises(ValueError, match="mert_layer"):
        extractor.extract_one(1, "unused.mp3")


def test_repeated_extraction_is_identical_and_the_model_runs_once(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict
) -> None:
    """Caching (musicsim.extractors.mert.load_or_compute_mert_layers) means a
    second call for the same item never re-invokes `compute_layers`.
    """
    calls = []

    def _tracked_compute_layers(path: str) -> np.ndarray:
        calls.append(path)
        return _fake_layers(1)

    extractor = MertExtractor(_config())
    monkeypatch.setattr(extractor, "compute_layers", _tracked_compute_layers)

    v1 = extractor.extract_one(1, "clip.mp3")
    v2 = extractor.extract_one(1, "clip.mp3")

    np.testing.assert_array_equal(v1, v2)
    assert len(calls) == 1


def test_extraction_without_cache_is_still_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """With caching disabled, repeating extraction with the same (deterministic)
    model must still produce bit-identical vectors.
    """
    extractor = MertExtractor(_config(runtime={"cache": False}))
    monkeypatch.setattr(extractor, "compute_layers", lambda path: _fake_layers(1))

    v1 = extractor.extract_one(1, "clip.mp3")
    v2 = extractor.extract_one(1, "clip.mp3")

    np.testing.assert_array_equal(v1, v2)


def test_extract_all_forces_a_single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever `runtime.n_jobs` says, MERT extraction never spreads across
    worker processes (see MertExtractor.extract_all's docstring).
    """
    seen_n_jobs = []
    extractor = MertExtractor(_config())
    monkeypatch.setattr(extractor, "compute_layers", lambda path: _fake_layers(1))

    import musicsim.extractors.base as base_module

    original = base_module.Extractor.extract_all

    def _spy(self, index, *, n_jobs=-1, progress=True):
        seen_n_jobs.append(n_jobs)
        return original(self, index, n_jobs=n_jobs, progress=progress)

    monkeypatch.setattr(base_module.Extractor, "extract_all", _spy)

    index = pd.DataFrame({"item_id": [1], "path": ["clip.mp3"]})
    extractor.extract_all(index, n_jobs=8, progress=False)

    assert seen_n_jobs == [1]


# ---------------------------------------------------------------------------
# select_titular_layer
# ---------------------------------------------------------------------------
def _layer_selection_corpus() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """4 genres x 3 splits x a handful of artists; layer 2 is deliberately the
    only one whose clusters follow genre, so the best layer is unambiguous.
    """
    rng = np.random.default_rng(0)
    n_per_split = 40
    splits = ["training", "validation", "test"]
    genres = ["rock", "jazz", "pop", "folk"]

    rows = []
    item_id = 0
    for split in splits:
        for i in range(n_per_split):
            rows.append(
                {
                    "item_id": item_id,
                    "split": split,
                    "genre_top": genres[i % len(genres)],
                    "artist_id": item_id,  # one artist per item: no accidental exclusion
                }
            )
            item_id += 1
    index = pd.DataFrame(rows)

    n_items = len(index)
    n_layers = 4
    hidden_size = 16
    layers = rng.normal(scale=1.0, size=(n_items, n_layers, hidden_size)).astype(np.float32)

    genre_code = {g: i for i, g in enumerate(genres)}
    centres = rng.normal(size=(len(genres), hidden_size)) * 10.0
    for row_idx, row in index.iterrows():
        noise = rng.normal(scale=1.0, size=hidden_size)
        layers[row_idx, 2] = centres[genre_code[row["genre_top"]]] + noise

    ids = index["item_id"].to_numpy()
    return layers, ids, index


def test_select_titular_layer_picks_the_genre_clustered_layer() -> None:
    layers, ids, index = _layer_selection_corpus()

    layer, scores = select_titular_layer(layers, ids, index, k=10)

    assert layer == 2
    assert scores.shape == (4,)
    assert scores[2] == scores.max()


def test_select_titular_layer_never_uses_the_test_split() -> None:
    layers, ids, index = _layer_selection_corpus()

    # Corrupt every test-split row's layer-2 vector: if the function touched
    # test, this would change the outcome (it would not even run, since the
    # corrupted rows are excluded from scoring either way).
    corrupted = layers.copy()
    test_mask = index["split"].to_numpy() == "test"
    corrupted[test_mask] = 0.0

    layer_clean, scores_clean = select_titular_layer(layers, ids, index, k=10)
    layer_corrupted, scores_corrupted = select_titular_layer(corrupted, ids, index, k=10)

    assert layer_clean == layer_corrupted
    np.testing.assert_array_equal(scores_clean, scores_corrupted)


def test_select_titular_layer_requires_a_validation_split() -> None:
    layers, ids, index = _layer_selection_corpus()
    index = index[index["split"] != "validation"].reset_index(drop=True)
    mask = np.isin(ids, index["item_id"].to_numpy())

    with pytest.raises(ValueError, match="validation"):
        select_titular_layer(layers[mask], ids[mask], index, k=10)
