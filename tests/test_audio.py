"""Tests for musicsim.audio: decode, resample, trim/pad, log-mel, caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from musicsim import audio
from musicsim.config import Config


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
        "spectrogram": {"n_fft": 256, "hop_length": 64, "n_mels": 16, "fmin": 0, "fmax": 4000},
        "runtime": {"cache": True},
    }
    for section, values in section_overrides.items():
        data.setdefault(section, {}).update(values)
    return Config(data=data)


def _write_wav(path: Path, seconds: float, sr: int = 8000, freq: float = 440.0) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), y, sr)


def test_load_clip_pads_short_clips_with_zeros(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    _write_wav(path, seconds=0.5, sr=8000)
    y = audio.load_clip(path, _config())
    assert y.shape == (8000,)
    assert np.allclose(y[4000:], 0.0)


def test_load_clip_trims_long_clips(tmp_path: Path) -> None:
    path = tmp_path / "long.wav"
    _write_wav(path, seconds=2.0, sr=8000)
    y = audio.load_clip(path, _config())
    assert y.shape == (8000,)


def test_load_clip_resamples_to_the_configured_rate(tmp_path: Path) -> None:
    path = tmp_path / "high_sr.wav"
    _write_wav(path, seconds=1.0, sr=44100)
    y = audio.load_clip(path, _config())
    assert y.shape == (8000,)


def test_load_clip_raises_for_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(audio.AudioError):
        audio.load_clip(tmp_path / "missing.wav", _config())


def test_load_clip_raises_for_a_too_short_clip(tmp_path: Path) -> None:
    path = tmp_path / "tiny.wav"
    _write_wav(path, seconds=0.01, sr=8000)
    with pytest.raises(audio.AudioError):
        audio.load_clip(path, _config(audio={"min_seconds": 0.5}))


def test_log_mel_spectrogram_shape_and_dtype(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    _write_wav(path, seconds=1.0, sr=8000)
    config = _config()
    y = audio.load_clip(path, config)
    S = audio.log_mel_spectrogram(y, config)
    assert S.shape[0] == 16
    assert S.dtype == np.float32
    assert np.isfinite(S).all()


def test_melspec_cache_hit_skips_decoding(
    tmp_path: Path, isolated_roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "clip.wav"
    _write_wav(path, seconds=1.0, sr=8000)
    config = _config()

    S1 = audio.load_or_compute_melspec(path, 1, config)
    cache_path = audio.melspec_cache_dir("toy", config) / "000001.npy"
    assert cache_path.is_file()

    calls: list[int] = []
    original = audio.load_clip

    def spy(*args: object, **kwargs: object) -> np.ndarray:
        calls.append(1)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(audio, "load_clip", spy)
    S2 = audio.load_or_compute_melspec(path, 1, config)

    assert calls == []  # decoding was skipped entirely on the cache hit
    # float16 round-trip on disk, so a cache hit is close but not bit-identical.
    assert np.allclose(S1, S2, atol=0.5)


def test_melspec_cache_disabled_never_writes_to_disk(
    tmp_path: Path, isolated_roots: dict[str, Path]
) -> None:
    path = tmp_path / "clip.wav"
    _write_wav(path, seconds=1.0, sr=8000)
    config = _config(runtime={"cache": False})
    audio.load_or_compute_melspec(path, 1, config)
    cache_path = audio.melspec_cache_dir("toy", config) / "000001.npy"
    assert not cache_path.is_file()


def test_melspec_cache_is_shared_across_extractors_with_the_same_spectrogram_settings(
    tmp_path: Path, isolated_roots: dict[str, Path]
) -> None:
    config_a = _config(extractor={"name": "mfcc", "dim": 80})
    config_b = _config(extractor={"name": "other", "dim": 40})
    assert audio.melspec_cache_dir("toy", config_a) == audio.melspec_cache_dir("toy", config_b)
