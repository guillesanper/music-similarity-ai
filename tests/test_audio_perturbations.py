"""Tests for musicsim.audio_perturbations, on synthetic signals only."""

from __future__ import annotations

import numpy as np
import pytest

from musicsim import audio_perturbations as ap

SR = 22050
KW = {"item_id": 7, "seed": 42, "sample_rate": SR}


def _sine(freq: float = 440.0, seconds: float = 2.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noisy(seconds: float = 2.0) -> np.ndarray:
    return np.random.default_rng(0).normal(scale=0.1, size=int(SR * seconds)).astype(np.float32)


def test_registry_has_the_thirteen_conditions() -> None:
    assert sum(len(levels) for levels in ap.PERTURBATIONS.values()) == 13
    assert set(ap.NULL_LEVELS) == set(ap.PERTURBATIONS)


@pytest.mark.parametrize("name", sorted(ap.PERTURBATIONS))
def test_null_level_is_bit_identical(name: str) -> None:
    y = _noisy()
    out = ap.perturb(y, name, ap.NULL_LEVELS[name], **KW)
    assert out.dtype == y.dtype
    assert np.array_equal(out, y)
    assert out is not y


@pytest.mark.parametrize("snr", ap.PERTURBATIONS["noise_snr_db"])
def test_noise_reaches_the_requested_snr(snr: float) -> None:
    y = _noisy()
    out = ap.perturb(y, "noise_snr_db", snr, **KW)
    noise = out.astype(np.float64) - y.astype(np.float64)
    measured = 10 * np.log10(np.mean(y.astype(np.float64) ** 2) / np.mean(noise**2))
    assert measured == pytest.approx(snr, abs=0.1)


def test_noise_on_a_silent_waveform_raises() -> None:
    with pytest.raises(ValueError, match="silent"):
        ap.perturb(np.zeros(1000, np.float32), "noise_snr_db", 20.0, **KW)


@pytest.mark.parametrize("db", ap.PERTURBATIONS["gain_db"])
def test_gain_scales_energy_without_clipping(db: float) -> None:
    y = _sine(amp=0.9)
    out = ap.perturb(y, "gain_db", db, **KW)
    ratio = np.sum(out.astype(np.float64) ** 2) / np.sum(y.astype(np.float64) ** 2)
    assert ratio == pytest.approx(10 ** (db / 10), rel=1e-5)
    if db > 0:
        assert np.abs(out).max() > 1.0  # documented: no saturation


@pytest.mark.parametrize("semitones", ap.PERTURBATIONS["pitch_semitones"])
def test_pitch_shift_moves_the_spectral_peak(semitones: float) -> None:
    y = _sine(440.0)
    out = ap.perturb(y, "pitch_semitones", semitones, **KW)
    assert len(out) == len(y)
    spectrum = np.abs(np.fft.rfft(out))
    peak = np.argmax(spectrum) * SR / len(out)
    bin_hz = SR / len(out)
    assert abs(peak - 440.0 * 2 ** (semitones / 12)) <= bin_hz


@pytest.mark.parametrize("rate", ap.PERTURBATIONS["tempo_ratio"])
def test_time_stretch_duration_before_length_fix(rate: float) -> None:
    y = _sine()
    assert abs(len(ap._time_stretch(y, rate)) - len(y) / rate) <= 1


@pytest.mark.parametrize("rate", ap.PERTURBATIONS["tempo_ratio"])
def test_time_stretch_output_keeps_input_length(rate: float) -> None:
    y = _sine()
    assert len(ap.perturb(y, "tempo_ratio", rate, **KW)) == len(y)


@pytest.mark.parametrize("seconds", ap.PERTURBATIONS["crop_seconds"])
def test_crop_keeps_exactly_the_central_window(seconds: float) -> None:
    y = _noisy(seconds=20.0)
    out = ap.perturb(y, "crop_seconds", seconds, **KW)
    n_keep = round(seconds * SR)
    start = (len(y) - n_keep) // 2
    assert len(out) == len(y)
    assert np.array_equal(out[start : start + n_keep], y[start : start + n_keep])
    assert not out[:start].any()
    assert not out[start + n_keep :].any()


def test_same_arguments_give_identical_output() -> None:
    y = _noisy()
    a = ap.perturb(y, "noise_snr_db", 20.0, **KW)
    b = ap.perturb(y, "noise_snr_db", 20.0, **KW)
    assert np.array_equal(a, b)


def test_other_seed_or_item_gives_other_noise() -> None:
    y = _noisy()
    base = ap.perturb(y, "noise_snr_db", 20.0, **KW)
    assert not np.array_equal(base, ap.perturb(y, "noise_snr_db", 20.0, **{**KW, "seed": 43}))
    assert not np.array_equal(base, ap.perturb(y, "noise_snr_db", 20.0, **{**KW, "item_id": 8}))


def test_level_outside_the_list_raises_with_value_and_allowed() -> None:
    with pytest.raises(ValueError, match=r"15\.0.*'noise_snr_db'.*\(30\.0, 20\.0, 10\.0\)"):
        ap.perturb(_noisy(), "noise_snr_db", 15.0, **KW)


def test_unknown_perturbation_raises() -> None:
    with pytest.raises(ValueError, match="reverb"):
        ap.perturb(_noisy(), "reverb", 1.0, **KW)
