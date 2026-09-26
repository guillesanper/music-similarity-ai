"""Tests for the `mfcc` extractor, including the regression check against the
archived repository (C:\\Users\\guill\\Desktop\\tfg, never modified by this project).
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf

from musicsim import audio, paths
from musicsim.config import Config, load_config
from musicsim.extractors.mfcc import MfccExtractor

#: Archived pre-refactor pipeline; read-only reference, see the module docstring.
_OLD_REPO = Path(r"C:\Users\guill\Desktop\tfg")
_OLD_MFCC = _OLD_REPO / "data" / "features" / "mfcc.npy"
_OLD_IDS = _OLD_REPO / "data" / "features" / "mfcc_ids.npy"


def _config(**section_overrides: dict) -> Config:
    data: dict = {
        "dataset": {"directory": "toy"},
        "audio": {
            "sample_rate": 8000,
            "clip_seconds": 1.0,
            "clip_samples": 8000,
            "min_seconds": 0.1,
            "mono": True,
        },
        "spectrogram": {"n_fft": 256, "hop_length": 64, "n_mels": 32, "fmin": 0, "fmax": 4000},
        "extractor": {"name": "mfcc", "n_mfcc": 13, "drop_c0": True, "delta_width": 9, "dim": 52},
        "runtime": {"cache": True},
    }
    for section, values in section_overrides.items():
        data.setdefault(section, {}).update(values)
    return Config(data=data)


def _write_wav(path: Path, seconds: float, sr: int = 8000, freq: float = 440.0) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * freq * 3 * t)
    sf.write(str(path), y.astype(np.float32), sr)


def test_mfcc_extractor_matches_the_direct_librosa_computation(
    tmp_path: Path, isolated_roots: dict[str, Path]
) -> None:
    """"By hand" reference: librosa.feature.mfcc(S=..., n_mfcc=14)[1:] + its own deltas.

    Caching disabled: a second `load_or_compute_melspec` call below must not
    hit the (lossy, float16) cache written by the extractor's own call.
    """
    path = tmp_path / "clip.wav"
    _write_wav(path, seconds=1.0, sr=8000)
    config = _config(runtime={"cache": False})

    vector = MfccExtractor(config).extract_one(1, str(path))

    S = audio.load_or_compute_melspec(path, 1, config)
    ref_mfcc = librosa.feature.mfcc(S=S, n_mfcc=14)[1:]
    ref_delta = librosa.feature.delta(ref_mfcc, width=9, axis=-1)
    ref_vector = np.concatenate(
        [ref_mfcc.mean(axis=1), ref_mfcc.std(axis=1), ref_delta.mean(axis=1), ref_delta.std(axis=1)]
    ).astype(np.float32)

    assert vector.shape == (52,)
    np.testing.assert_allclose(vector, ref_vector, rtol=1e-4, atol=1e-4)


def test_mfcc_extractor_drop_c0_changes_the_output(
    tmp_path: Path, isolated_roots: dict[str, Path]
) -> None:
    """Both configurations keep dim=52 (n_coeffs stays n_mfcc either way), but c0
    (frame energy) shifts every pooled coefficient, so the vectors must differ.
    """
    path = tmp_path / "clip.wav"
    _write_wav(path, seconds=1.0, sr=8000)

    vec_drop = MfccExtractor(_config()).extract_one(1, str(path))
    vec_keep = MfccExtractor(_config(extractor={"drop_c0": False})).extract_one(1, str(path))

    assert vec_drop.shape == vec_keep.shape == (52,)
    assert not np.allclose(vec_drop, vec_keep)


@pytest.mark.slow
def test_mfcc_matches_the_archived_repository_reference() -> None:
    """Regression check against the archived pipeline's ``mfcc.npy``.

    Requires the real FMA index (`musicsim index`) and a `musicsim extract` run for the
    mfcc extractor (``musicsim extract --config
    configs/experiments/e01_mfcc_baseline.yaml``) to already have populated the
    cache — this test does not itself pay for the ~7994-item extraction.
    """
    if not _OLD_MFCC.is_file() or not _OLD_IDS.is_file():
        pytest.skip(f"archived reference not available: {_OLD_MFCC}")

    index_path = paths.dataset_index("fma")
    if not index_path.is_file():
        pytest.skip("fma index not built; run `musicsim index --dataset fma_small` first")

    config = load_config(paths.REPO_ROOT / "configs" / "experiments" / "e01_mfcc_baseline.yaml")
    cache_dir = paths.cache_dir("fma", "features-mfcc", config.hash())
    features_path = cache_dir / "features.npy"
    ids_path = cache_dir / "ids.npy"
    if not features_path.is_file() or not ids_path.is_file():
        pytest.skip(
            "no cached mfcc features; run `musicsim extract --config "
            "configs/experiments/e01_mfcc_baseline.yaml` first"
        )

    X = np.load(features_path)
    ids = np.load(ids_path)
    old_X = np.load(_OLD_MFCC)
    old_ids = np.load(_OLD_IDS)

    assert len(ids) == len(old_ids), f"item counts differ: {len(ids)} vs {len(old_ids)}"
    assert set(ids.tolist()) == set(old_ids.tolist()), "item sets differ between the two pipelines"

    new_order = np.argsort(ids)
    old_order = np.argsort(old_ids)
    assert np.array_equal(ids[new_order], old_ids[old_order])

    np.testing.assert_allclose(X[new_order], old_X[old_order], rtol=1e-3, atol=1e-3)
