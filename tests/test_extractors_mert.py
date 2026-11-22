"""Tests for the `mert` extractor and its layer-selection helper.

None of these need the real model or network access: `MertExtractor.compute_layers`
is monkeypatched with a deterministic stand-in, so the tests exercise the
contract (shape, dtype, caching, layer selection) rather than the model. The
device and batching logic is tested the same way, with the audio loader and
the model's forward pass replaced by deterministic fakes, so it needs neither
torch nor a GPU; the few tests that do need them skip themselves.
"""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from musicsim import paths
from musicsim.audio import AudioError
from musicsim.config import Config, ConfigError
from musicsim.extractors import mert as mert_module
from musicsim.extractors.mert import (
    MertExtractor,
    _resolve_device,
    _save_layers,
    select_titular_layer,
)


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
            # Pinned so that a machine with a GPU still runs these on the CPU path.
            "device": "cpu",
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


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def _fake_torch(monkeypatch: pytest.MonkeyPatch, *, cuda: bool) -> None:
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda))
    monkeypatch.setitem(sys.modules, "torch", fake)


@pytest.mark.parametrize(
    ("requested", "cuda", "expected"),
    [
        ("auto", False, "cpu"),
        ("auto", True, "cuda"),
        ("cpu", True, "cpu"),
        ("cuda", True, "cuda"),
        ("cuda:1", True, "cuda:1"),
        ("AUTO", True, "cuda"),
    ],
)
def test_resolve_device(
    monkeypatch: pytest.MonkeyPatch, requested: str, cuda: bool, expected: str
) -> None:
    _fake_torch(monkeypatch, cuda=cuda)

    assert _resolve_device(_config(extractor={"device": requested})) == expected


def test_auto_device_without_torch_falls_back_to_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` raises ImportError

    assert _resolve_device(_config(extractor={"device": "auto"})) == "cpu"


def test_default_device_is_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_torch(monkeypatch, cuda=True)
    data = _config().to_dict()
    del data["extractor"]["device"]

    assert _resolve_device(Config(data=data)) == "cuda"


def test_explicit_cuda_without_a_gpu_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_torch(monkeypatch, cuda=False)

    with pytest.raises(ConfigError, match="no CUDA"):
        _resolve_device(_config(extractor={"device": "cuda"}))


def test_explicit_cuda_without_torch_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ConfigError, match="needs torch"):
        _resolve_device(_config(extractor={"device": "cuda"}))


def test_unknown_device_is_an_error() -> None:
    with pytest.raises(ConfigError, match=r"extractor.device"):
        _resolve_device(_config(extractor={"device": "tpu"}))


def test_cpu_device_never_imports_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    assert _resolve_device(_config(extractor={"device": "cpu"})) == "cpu"


# ---------------------------------------------------------------------------
# Batched extraction (the GPU path, with the model and the audio faked out)
# ---------------------------------------------------------------------------
def _fake_forward(batch: np.ndarray) -> np.ndarray:
    """Deterministic stand-in for the model: ``(B, samples)`` -> ``(B, 13, 768)``.

    Every output row depends only on its own window, like the real model's.
    """
    energy = batch.sum(axis=1)
    layer_scale = np.arange(13, dtype=np.float32)[None, :, None] + 1.0
    offset = np.arange(768, dtype=np.float32)[None, None, :] * 1e-3
    return (energy[:, None, None] * layer_scale + offset).astype(np.float32)


def _expected_layers(windows: np.ndarray) -> np.ndarray:
    return _fake_forward(windows).mean(axis=0).astype(np.float16)


def _expected_titular(windows: np.ndarray) -> np.ndarray:
    return _expected_layers(windows)[6].astype(np.float32)


def _tracks(n_windows: list[int]) -> tuple[pd.DataFrame, dict[str, np.ndarray | Exception]]:
    """An index of ``len(n_windows)`` items and the windows each one decodes to."""
    rng = np.random.default_rng(7)
    windows: dict[str, np.ndarray | Exception] = {
        f"t{i}.mp3": rng.normal(scale=0.1, size=(n, 16)).astype(np.float32)
        for i, n in enumerate(n_windows)
    }
    index = pd.DataFrame(
        {"item_id": [100 + i for i in range(len(n_windows))], "path": list(windows)}
    )
    return index, windows


class _Rig:
    """A ``MertExtractor`` on a pretend GPU, with every call recorded."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        windows: dict[str, np.ndarray | Exception],
        *,
        device: str = "cuda",
        **extractor: object,
    ) -> None:
        self.extractor = MertExtractor(_config(extractor={"device": device, **extractor}))
        self.extractor._device = device
        self.forward_sizes: list[int] = []
        self.loaded: list[str] = []
        self.lock = threading.Lock()
        self.windows = windows
        monkeypatch.setattr(self.extractor, "_ensure_loaded", lambda: None)
        monkeypatch.setattr(self.extractor, "_release_cuda_cache", lambda: None)
        monkeypatch.setattr(self.extractor, "_load_windows", self._load)
        monkeypatch.setattr(self.extractor, "_forward_pooled", self._forward)

    def _load(self, path: str) -> np.ndarray:
        with self.lock:
            self.loaded.append(path)
        value = self.windows[path]
        if isinstance(value, Exception):
            raise value
        return value

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        self.forward_sizes.append(len(batch))
        return _fake_forward(batch)

    def run(self, index: pd.DataFrame):
        return self.extractor.extract_all(index, progress=False)


def test_batched_extraction_matches_the_per_item_result(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict
) -> None:
    """Items with different numbers of windows, batches that straddle items."""
    index, windows = _tracks([6, 6, 3, 6, 1, 6, 6])
    rig = _Rig(monkeypatch, windows, batch_windows=10)

    X, ids, failures = rig.run(index)

    assert failures == []
    np.testing.assert_array_equal(ids, index["item_id"].to_numpy())
    for row, path in enumerate(index["path"]):
        np.testing.assert_array_equal(X[row], _expected_titular(windows[path]))
    assert X.dtype == np.float32
    assert max(rig.forward_sizes) <= 10
    assert max(rig.forward_sizes) > 6  # it really grouped windows of different items


@pytest.mark.parametrize("batch_windows", [1, 5, 100])
def test_batch_size_does_not_change_the_result(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict, batch_windows: int
) -> None:
    index, windows = _tracks([6, 4, 6, 2])
    rig = _Rig(monkeypatch, windows, batch_windows=batch_windows)

    X, _, _ = rig.run(index)

    expected = np.stack([_expected_titular(windows[p]) for p in index["path"]])
    np.testing.assert_array_equal(X, expected)


def test_a_cpu_extractor_runs_one_window_per_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    _, windows = _tracks([5])
    rig = _Rig(monkeypatch, windows, device="cpu", batch_windows=24)

    layers = rig.extractor.compute_layers("t0.mp3")

    assert rig.forward_sizes == [1] * 5
    np.testing.assert_array_equal(layers, _expected_layers(windows["t0.mp3"]))


def test_compute_layers_on_a_gpu_uses_batch_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    _, windows = _tracks([6])
    rig = _Rig(monkeypatch, windows, batch_windows=4)

    layers = rig.extractor.compute_layers("t0.mp3")

    assert rig.forward_sizes == [4, 2]
    np.testing.assert_array_equal(layers, _expected_layers(windows["t0.mp3"]))


def test_each_item_is_cached_and_a_second_run_never_touches_the_model(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict
) -> None:
    index, windows = _tracks([6, 6, 6])
    rig = _Rig(monkeypatch, windows, batch_windows=12)
    X1, _, _ = rig.run(index)

    cache_dir = mert_module._mert_layers_cache_dir("toy", rig.extractor.config)
    files = sorted(p.name for p in cache_dir.iterdir())
    assert files == ["000100.npy", "000101.npy", "000102.npy"]  # and no *.tmp leftovers
    np.testing.assert_array_equal(
        np.load(cache_dir / "000101.npy"), _expected_layers(windows["t1.mp3"])
    )

    second = _Rig(monkeypatch, windows, batch_windows=12)
    X2, _, failures = second.run(index)

    assert failures == []
    assert second.loaded == []
    assert second.forward_sizes == []
    np.testing.assert_array_equal(X1, X2)


def test_a_cache_written_on_the_cpu_is_reused_by_the_gpu_driver(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict
) -> None:
    index, windows = _tracks([6, 6])
    cpu = _Rig(monkeypatch, windows, device="cpu")
    cpu_vectors = [
        cpu.extractor.extract_one(int(i), p)
        for i, p in zip(index["item_id"], index["path"], strict=True)
    ]

    gpu = _Rig(monkeypatch, windows)
    X, _, _ = gpu.run(index)

    assert gpu.loaded == []
    np.testing.assert_array_equal(X, np.stack(cpu_vectors))


def test_only_uncached_items_are_computed(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict
) -> None:
    index, windows = _tracks([6, 6, 6])
    first = _Rig(monkeypatch, windows)
    first.extractor.extract_one(100, "t0.mp3")

    rig = _Rig(monkeypatch, windows)
    X, ids, _ = rig.run(index)

    assert sorted(rig.loaded) == ["t1.mp3", "t2.mp3"]
    np.testing.assert_array_equal(ids, [100, 101, 102])
    assert X.shape == (3, 768)


def test_a_file_that_cannot_be_decoded_is_a_failure_not_a_crash(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict
) -> None:
    index, windows = _tracks([6, 6, 6, 6])
    windows["t1.mp3"] = AudioError("could not decode t1.mp3")
    windows["t3.mp3"] = AudioError("could not decode t3.mp3")
    rig = _Rig(monkeypatch, windows, batch_windows=12)

    X, ids, failures = rig.run(index)

    np.testing.assert_array_equal(ids, [100, 102])
    assert X.shape == (2, 768)
    assert [item for item, _ in failures] == [101, 103]
    assert "AudioError" in failures[0][1]
    np.testing.assert_array_equal(X[1], _expected_titular(windows["t2.mp3"]))


def test_a_non_finite_vector_is_a_failure(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict
) -> None:
    index, windows = _tracks([2, 2])
    windows["t0.mp3"] = np.full((2, 16), np.nan, dtype=np.float32)
    rig = _Rig(monkeypatch, windows)

    X, ids, failures = rig.run(index)

    np.testing.assert_array_equal(ids, [101])
    assert [item for item, _ in failures] == [100]
    assert "NaN" in failures[0][1]
    assert X.shape == (1, 768)


def test_decoded_audio_read_ahead_is_bounded(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict
) -> None:
    """With 2 windows per item, prefetch 3 and batches of 4 windows, no more than
    (prefetch + 1) items plus one batch of windows may be alive at once --
    nowhere near the 80 windows of the index.
    """
    index, windows = _tracks([2] * 40)
    rig = _Rig(monkeypatch, windows, batch_windows=4, prefetch=3, decode_workers=2)
    outstanding = 0
    peak = 0

    def load(path: str) -> np.ndarray:
        nonlocal outstanding, peak
        with rig.lock:
            outstanding += 2
            peak = max(peak, outstanding)
        return windows[path]

    def forward(batch: np.ndarray) -> np.ndarray:
        nonlocal outstanding
        with rig.lock:
            outstanding -= len(batch)
        return _fake_forward(batch)

    monkeypatch.setattr(rig.extractor, "_load_windows", load)
    monkeypatch.setattr(rig.extractor, "_forward_pooled", forward)

    X, _, failures = rig.run(index)

    assert failures == []
    assert X.shape == (40, 768)
    assert peak <= (3 + 1) * 2 + 4


def test_the_gpu_driver_does_not_use_worker_processes(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict
) -> None:
    import musicsim.extractors.base as base_module

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("joblib must not be used on the GPU path")

    monkeypatch.setattr(base_module, "Parallel", _boom)
    index, windows = _tracks([6, 6])
    rig = _Rig(monkeypatch, windows)

    X, _, _ = rig.extractor.extract_all(index, n_jobs=8, progress=False)

    assert X.shape == (2, 768)


def test_an_out_of_range_titular_layer_fails_before_any_work(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict
) -> None:
    index, windows = _tracks([6])
    rig = _Rig(monkeypatch, windows, mert_layer=13)

    with pytest.raises(ValueError, match="mert_layer"):
        rig.run(index)

    assert rig.loaded == []


@pytest.mark.parametrize(
    "bad", [{"batch_windows": 0}, {"decode_workers": 0}, {"prefetch": 0}, {"precision": "int8"}]
)
def test_invalid_execution_settings_are_rejected(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict, bad: dict
) -> None:
    index, windows = _tracks([6])
    rig = _Rig(monkeypatch, windows, **bad)

    with pytest.raises(ConfigError):
        rig.run(index)


def test_an_empty_index_returns_empty_arrays(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict
) -> None:
    rig = _Rig(monkeypatch, {})

    X, ids, failures = rig.run(pd.DataFrame({"item_id": [], "path": []}))

    assert X.shape == (0, 768)
    assert ids.shape == (0,)
    assert failures == []


# -- running out of memory ---------------------------------------------------
def test_an_out_of_memory_forward_is_retried_with_a_smaller_batch(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict
) -> None:
    index, windows = _tracks([6, 6, 6])
    rig = _Rig(monkeypatch, windows, batch_windows=12)
    attempts: list[int] = []

    def forward(batch: np.ndarray) -> np.ndarray:
        attempts.append(len(batch))
        if len(batch) > 3:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return _fake_forward(batch)

    monkeypatch.setattr(rig.extractor, "_forward_pooled", forward)

    X, _, failures = rig.run(index)

    assert failures == []
    expected = np.stack([_expected_titular(windows[p]) for p in index["path"]])
    np.testing.assert_array_equal(X, expected)
    assert attempts[:2] == [12, 6]  # 12 fails, 6 fails, then 3 works
    assert rig.extractor._oom_limit == 3
    assert max(attempts[2:]) == 3  # the limit is remembered, no more failed attempts


def test_a_forward_that_fails_for_another_reason_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, isolated_roots: dict
) -> None:
    index, windows = _tracks([6])
    rig = _Rig(monkeypatch, windows)

    def forward(batch: np.ndarray) -> np.ndarray:
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(rig.extractor, "_forward_pooled", forward)

    with pytest.raises(RuntimeError, match="device-side"):
        rig.run(index)


def test_out_of_memory_on_a_single_window_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _, windows = _tracks([2])
    rig = _Rig(monkeypatch, windows, batch_windows=4)

    def forward(batch: np.ndarray) -> np.ndarray:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(rig.extractor, "_forward_pooled", forward)

    with pytest.raises(RuntimeError, match="out of memory"):
        rig.extractor.compute_layers("t0.mp3")


# -- atomic cache writes -------------------------------------------------------
def test_save_layers_leaves_no_temporary_file(tmp_path) -> None:
    layers = _expected_layers(np.ones((2, 16), dtype=np.float32))
    target = tmp_path / "000001.npy"

    _save_layers(target, layers)

    assert [p.name for p in tmp_path.iterdir()] == ["000001.npy"]
    np.testing.assert_array_equal(np.load(target), layers)


def test_an_interrupted_save_never_leaves_a_truncated_cache_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    target = tmp_path / "000001.npy"

    def _crash(handle: object, array: object) -> None:
        handle.write(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(np, "save", _crash)

    with pytest.raises(KeyboardInterrupt):
        _save_layers(target, np.zeros((13, 768), dtype=np.float16))

    assert not target.exists()


def test_cache_key_ignores_the_execution_settings(isolated_roots: dict) -> None:
    """A cache filled on one device / batch size is the cache of every other."""
    base = mert_module._mert_layers_cache_dir("toy", _config())
    other = mert_module._mert_layers_cache_dir(
        "toy",
        _config(
            extractor={
                "device": "cuda",
                "batch_windows": 7,
                "precision": "float16",
                "decode_workers": 1,
                "prefetch": 2,
            }
        ),
    )

    assert base == other
    assert paths.output_root() in base.parents


# ---------------------------------------------------------------------------
# The real forward pass: needs torch (skipped without it), a GPU only for the
# tests marked `gpu`
# ---------------------------------------------------------------------------
def _tiny_model():
    """A stand-in with the model's interface: 13 hidden states ``(B, T, 768)``."""
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.conv = torch.nn.Conv1d(1, 768, kernel_size=400, stride=320)

        def forward(self, x, output_hidden_states=False):
            h = self.conv(x.unsqueeze(1)).transpose(1, 2)
            return SimpleNamespace(hidden_states=tuple(h * (i + 1) for i in range(13)))

    return Tiny().eval()


def _tiny_extractor(device: str, **extractor: object) -> MertExtractor:
    ext = MertExtractor(_config(extractor={"device": device, **extractor}))
    ext._device = device
    ext._model = _tiny_model().to(device)
    ext._processor = SimpleNamespace(sampling_rate=16000)
    return ext


def test_real_forward_pools_on_the_device_and_windows_are_independent() -> None:
    torch = pytest.importorskip("torch")
    ext = _tiny_extractor("cpu")
    batch = np.random.default_rng(1).normal(size=(5, 16000)).astype(np.float32)

    with torch.no_grad():
        together = ext._forward_pooled(batch)
        apart = np.concatenate([ext._forward_pooled(batch[i : i + 1]) for i in range(5)])

    assert together.shape == (5, 13, 768)
    assert together.dtype == np.float32
    np.testing.assert_allclose(together, apart, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
def test_forward_on_cuda_agrees_with_the_cpu() -> None:
    torch = pytest.importorskip("torch")
    batch = np.random.default_rng(2).normal(size=(6, 16000)).astype(np.float32)
    cpu = _tiny_extractor("cpu")
    gpu = _tiny_extractor("cuda")

    with torch.no_grad():
        np.testing.assert_allclose(
            gpu._forward_pooled(batch), cpu._forward_pooled(batch), rtol=1e-3, atol=1e-3
        )


@pytest.mark.gpu
def test_half_precision_forward_stays_close_to_float32() -> None:
    torch = pytest.importorskip("torch")
    batch = np.random.default_rng(3).normal(size=(6, 16000)).astype(np.float32)
    full = _tiny_extractor("cuda", precision="float32")
    half = _tiny_extractor("cuda", precision="float16")

    with torch.no_grad():
        a = full._forward_pooled(batch)
        b = half._forward_pooled(batch)

    cosine = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    assert cosine.min() > 0.999


@pytest.mark.slow
@pytest.mark.gpu
def test_real_mert_on_gpu_agrees_with_the_cpu_per_layer() -> None:
    """Three real FMA tracks through the real model, CPU against CUDA: the
    13 layers must agree to float tolerance (cosine >= 0.9999 each).
    Needs the FMA index and downloads the weights on first use.
    """
    pytest.importorskip("transformers")
    from musicsim.config import load_config

    index_path = paths.dataset_index("fma")
    if not index_path.is_file():
        pytest.skip("fma index not built; run `musicsim index --dataset fma_small` first")
    index = pd.read_csv(index_path).head(3)

    base = load_config(paths.REPO_ROOT / "configs" / "experiments" / "e01_mert_baseline.yaml")
    data = base.to_dict()
    data["runtime"]["cache"] = False  # compare fresh forward passes, never the cache

    def layers_on(device: str) -> np.ndarray:
        data["extractor"]["device"] = device
        ext = MertExtractor(Config(data=data))
        return np.stack([ext.compute_layers(p) for p in index["path"]]).astype(np.float32)

    cpu = layers_on("cpu")
    gpu = layers_on("cuda")

    cosine = (cpu * gpu).sum(-1) / (np.linalg.norm(cpu, axis=-1) * np.linalg.norm(gpu, axis=-1))
    assert cosine.min() >= 0.9999
