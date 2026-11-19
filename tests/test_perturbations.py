"""Tests for musicsim.perturbations, on synthetic signals only."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from musicsim import audio
from musicsim import perturbations as pt
from musicsim.config import Config, ConfigError
from musicsim.extractors.base import Extractor
from musicsim.registry import register_extractor

SR = 8000


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _noise(seconds: float = 2.0) -> np.ndarray:
    return _rng(5).normal(scale=0.1, size=int(SR * seconds)).astype(np.float32)


def _sine(freq: float = 440.0, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))


def test_the_condition_list_is_the_closed_one() -> None:
    assert pt.CONDITIONS == ("identity", "noise", "gain", "pitch", "tempo", "crop")


def test_identity_is_an_equal_independent_copy() -> None:
    y = _noise()
    out = pt.perturb(y, SR, "identity", 123.0, rng=_rng())
    assert out.dtype == np.float32 and np.array_equal(out, y) and out is not y


@pytest.mark.parametrize("snr", [30.0, 20.0, 10.0])
def test_noise_reaches_the_requested_snr(snr: float) -> None:
    y = _sine()
    out = pt.perturb(y, SR, "noise", snr, rng=_rng())
    residual = out.astype(np.float64) - y.astype(np.float64)
    measured = 10 * np.log10(np.mean(y.astype(np.float64) ** 2) / np.mean(residual**2))
    assert measured == pytest.approx(snr, abs=0.1)


def test_gain_plus_six_db_scales_the_rms_without_clipping() -> None:
    y = _sine()
    out = pt.perturb(y, SR, "gain", 6.0, rng=_rng())
    assert _rms(out) == pytest.approx(_rms(y) * 10**0.3, rel=1e-6)
    loud = pt.perturb(y, SR, "gain", 20.0, rng=_rng())
    assert np.abs(loud).max() > 1.0


@pytest.mark.parametrize(
    ("condition", "level"),
    [
        ("noise", 20.0),
        ("gain", -6.0),
        ("pitch", 1.0),
        ("tempo", 0.9),
        ("tempo", 1.1),
        ("crop", 1.0),
    ],
)
def test_every_output_has_the_input_length_and_dtype(condition: str, level: float) -> None:
    y = _sine()
    out = pt.perturb(y, SR, condition, level, rng=_rng())
    assert out.shape == y.shape and out.dtype == np.float32


def test_same_generator_state_gives_the_same_noise_and_other_states_do_not() -> None:
    y = _sine()
    a = pt.perturb(y, SR, "noise", 20.0, rng=_rng(1))
    assert np.array_equal(a, pt.perturb(y, SR, "noise", 20.0, rng=_rng(1)))
    assert not np.array_equal(a, pt.perturb(y, SR, "noise", 20.0, rng=_rng(2)))


def test_only_noise_consumes_the_generator() -> None:
    y = _sine()
    rng = _rng(3)
    for condition, level in [("gain", 3.0), ("pitch", 1.0), ("tempo", 1.1), ("crop", 1.0)]:
        pt.perturb(y, SR, condition, level, rng=rng)
    assert rng.standard_normal() == _rng(3).standard_normal()


def test_crop_keeps_the_central_window_and_zeroes_the_rest() -> None:
    y = _noise(2.0)
    out = pt.perturb(y, SR, "crop", 1.0, rng=_rng())
    start = (len(y) - SR) // 2
    assert np.array_equal(out[start : start + SR], y[start : start + SR])
    assert not out[:start].any() and not out[start + SR :].any()


def test_pitch_shift_moves_the_spectral_peak_up() -> None:
    y = _sine(440.0, 1.0)
    up = pt.perturb(y, SR, "pitch", 12.0, rng=_rng())

    def peak(x: np.ndarray) -> float:
        return float(np.argmax(np.abs(np.fft.rfft(x))) * SR / len(x))

    assert peak(up) == pytest.approx(880.0, rel=0.05)


def test_tempo_pads_with_zeros_when_the_clip_gets_shorter() -> None:
    y = _noise(2.0)
    fast = pt.perturb(y, SR, "tempo", 1.1, rng=_rng())
    assert len(fast) == len(y) and not fast[-100:].any()


def test_a_silent_clip_gets_no_noise() -> None:
    silent = np.zeros(SR, dtype=np.float32)
    assert not pt.perturb(silent, SR, "noise", 10.0, rng=_rng()).any()


@pytest.mark.parametrize(
    ("condition", "level"),
    [
        ("reverb", 1.0),
        ("noise", float("inf")),
        ("noise", float("nan")),
        ("gain", float("inf")),
        ("tempo", 0.0),
        ("tempo", -1.0),
        ("crop", 0.0),
        ("crop", 3.0),  # longer than the 2 s clip
    ],
)
def test_senseless_conditions_and_levels_raise(condition: str, level: float) -> None:
    with pytest.raises(ValueError, match=condition if condition != "reverb" else "unknown"):
        pt.perturb(_sine(), SR, condition, level, rng=_rng())


def test_a_stereo_waveform_is_rejected() -> None:
    with pytest.raises(ValueError, match="mono"):
        pt.perturb(np.zeros((2, 100), np.float32), SR, "identity", 0.0, rng=_rng())


# --- features of perturbed audio ---------------------------------------------------
N_ITEMS = 6
N_BANDS = 8


def _band_features(y: np.ndarray) -> np.ndarray:
    power = np.abs(np.fft.rfft(y.astype(np.float64))) ** 2
    energy = np.array([band.sum() for band in np.array_split(power, N_BANDS)])
    return np.log(energy / energy.sum() + 1e-6).astype(np.float32)


@register_extractor("toy_perturbed")
class _ToyExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        return _band_features(audio.load_clip(path, self.config))


def _config(extractor: dict | None = None, **runtime: object) -> Config:
    return Config(
        data={
            "seed": 3,
            "dataset": {"directory": "toy_perturbed"},
            "audio": {
                "sample_rate": SR,
                "clip_seconds": 1.0,
                "clip_samples": SR,
                "min_seconds": 0.1,
                "mono": True,
            },
            "spectrogram": {
                "n_fft": 256,
                "hop_length": 64,
                "n_mels": 32,
                "fmin": 0,
                "fmax": 4000,
            },
            "extractor": extractor or {"name": "toy_perturbed", "dim": N_BANDS},
            "runtime": {"n_jobs": 1, "progress": False, "cache": True, **runtime},
        }
    )


def _mfcc_config(**runtime: object) -> Config:
    return _config(
        {"name": "mfcc", "n_mfcc": 13, "drop_c0": True, "delta_width": 9, "dim": 52}, **runtime
    )


@pytest.fixture
def clips(tmp_path: Path, isolated_roots: dict[str, Path], monkeypatch: pytest.MonkeyPatch):
    """An index of tone clips written as float WAV; toy extractor and mfcc admitted."""
    monkeypatch.setattr(pt, "CACHE_SAFE_EXTRACTORS", frozenset({"mfcc", "toy_perturbed"}))
    folder = tmp_path / "originals"
    folder.mkdir()
    paths_ = []
    for item in range(N_ITEMS):
        path = folder / f"{item}.wav"
        sf.write(str(path), _sine(300.0 + 150.0 * item, 1.0), SR, subtype="FLOAT")
        paths_.append(str(path))
    return pd.DataFrame({"item_id": np.arange(N_ITEMS), "path": paths_})


def test_float_wav_round_trip_is_lossless(tmp_path: Path) -> None:
    y = _noise(1.0)
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), y, SR, subtype="FLOAT")
    assert np.abs(audio.load_clip(wav, _config()) - y).max() == 0.0


def test_identity_features_equal_those_of_the_original_audio(clips: pd.DataFrame) -> None:
    config = _config()
    X, ids, failures = pt.perturbed_features(config, clips, "identity", 0.0)
    extractor = _ToyExtractor(config)
    expected = np.stack(
        [extractor.extract_one(i, p) for i, p in zip(clips["item_id"], clips["path"], strict=True)]
    )
    assert failures == [] and ids.tolist() == list(range(N_ITEMS))
    assert np.array_equal(X, expected)


def test_a_perturbation_changes_the_features_and_is_reproducible(clips: pd.DataFrame) -> None:
    config = _config(cache=False)
    identity, _, _ = pt.perturbed_features(config, clips, "identity", 0.0)
    noisy, _, _ = pt.perturbed_features(config, clips, "noise", 10.0)
    again, _, _ = pt.perturbed_features(config, clips, "noise", 10.0)
    assert not np.array_equal(noisy, identity)
    assert np.array_equal(noisy, again)


def test_noise_of_a_clip_does_not_depend_on_which_other_clips_are_extracted(
    clips: pd.DataFrame,
) -> None:
    config = _config(cache=False)
    full, _, _ = pt.perturbed_features(config, clips, "noise", 10.0)
    subset, ids, _ = pt.perturbed_features(config, clips.iloc[[4, 1]], "noise", 10.0)
    assert ids.tolist() == [4, 1]
    assert np.array_equal(subset, full[[4, 1]])


def test_the_second_call_reuses_the_cache(
    clips: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    first, _, _ = pt.perturbed_features(config, clips, "gain", 6.0)

    def boom(self, item_id, path):
        raise AssertionError("extracted although the condition was cached")

    monkeypatch.setattr(_ToyExtractor, "extract_one", boom)
    second, _, failures = pt.perturbed_features(config, clips, "gain", 6.0)
    assert failures == [] and np.array_equal(first, second)


def test_no_wav_is_left_behind(
    clips: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    pt.perturbed_features(_config(cache=False), clips, "noise", 20.0)
    assert list(scratch.iterdir()) == []


def test_an_extractor_that_is_not_admitted_is_refused(
    clips: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pt, "CACHE_SAFE_EXTRACTORS", frozenset({"mfcc"}))
    with pytest.raises(ConfigError, match=r"not verified to respect runtime\.cache"):
        pt.perturbed_features(_config(), clips, "identity", 0.0)


def test_an_unknown_condition_is_refused_before_any_work(clips: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown condition"):
        pt.perturbed_features(_config(), clips, "reverb", 1.0)


def test_an_unreadable_clip_is_a_failure_not_a_crash(clips: pd.DataFrame) -> None:
    broken = clips.copy()
    broken.loc[2, "path"] = str(Path(broken.loc[2, "path"]).with_name("missing.wav"))
    X, ids, failures = pt.perturbed_features(_config(cache=False), broken, "identity", 0.0)
    assert [item for item, _ in failures] == [2]
    assert ids.tolist() == [0, 1, 3, 4, 5] and X.shape == (5, N_BANDS)


def test_mfcc_identity_matches_the_original_and_leaves_the_log_mel_cache_alone(
    clips: pd.DataFrame,
) -> None:
    from musicsim.extractors.mfcc import MfccExtractor

    config = _mfcc_config()
    melspec_dir = audio.melspec_cache_dir("toy_perturbed", config)
    X, _, failures = pt.perturbed_features(config, clips, "identity", 0.0)
    X_noisy, _, _ = pt.perturbed_features(config, clips, "noise", 10.0)

    reference = MfccExtractor(_mfcc_config(cache=False))
    expected = np.stack(
        [reference.extract_one(i, p) for i, p in zip(clips["item_id"], clips["path"], strict=True)]
    )
    assert failures == [] and np.array_equal(X, expected)
    assert not np.array_equal(X_noisy, X)
    assert not melspec_dir.exists()  # nothing written, so nothing a perturbed clip could poison
