"""Deterministic perturbations of a decoded waveform, to test how stable a
representation's neighbourhoods are when the audio is degraded.

The list of perturbations and their levels is closed (:data:`PERTURBATIONS`):
adding a condition after seeing results would be a new analysis, not a
variant of this one. Every perturbation is a pure function of
``(waveform, name, level, item_id, seed)``; the only random one, additive
noise, draws from ``seeding.generator(seed, "perturbation", name, level,
item_id)`` so the same clip always gets the same noise and two clips never
share it.

Design decisions worth knowing:

* **Output length equals input length.** Time stretching and cropping change
  how many samples there are; the result is trimmed at the end or zero-padded
  at the end (stretch, as :func:`musicsim.audio.load_clip` does) or kept as the
  central window with silence around it (crop), so the log-mel spectrogram
  keeps its number of frames and pooled vectors stay comparable with the
  originals. The price is that padding adds silence, which is part of what a
  crop does to a clip and is not hidden.
* **Every perturbation has a null level** (:data:`NULL_LEVELS`) that returns
  the input unchanged, bit for bit. It is the control that makes an unchanged
  neighbourhood (Jaccard@10 = 1) a testable property of the pipeline.
* **Gain does not clip.** A boost can push samples beyond ±1; the waveform is
  float and the feature extractors work in float, so nothing overflows, and
  saturating would make the output energy differ from the requested gain.
* **Noise is scaled to the requested SNR exactly**, using the realised noise
  power rather than its expectation, so the measured SNR matches the request
  up to float32 rounding.
"""

from __future__ import annotations

import math

import librosa
import numpy as np

from musicsim import seeding

__all__ = ["NULL_LEVELS", "PERTURBATIONS", "perturb"]

#: The closed set of conditions: perturbation name -> allowed levels (13 in all).
PERTURBATIONS: dict[str, tuple[float, ...]] = {
    "noise_snr_db": (30.0, 20.0, 10.0),
    "gain_db": (-6.0, 6.0),
    "pitch_semitones": (-2.0, -1.0, 1.0, 2.0),
    "tempo_ratio": (0.9, 1.1),
    "crop_seconds": (15.0, 10.0),
}

#: Per-perturbation level that leaves the waveform untouched.
NULL_LEVELS: dict[str, float] = {
    "noise_snr_db": math.inf,
    "gain_db": 0.0,
    "pitch_semitones": 0.0,
    "tempo_ratio": 1.0,
    "crop_seconds": math.inf,
}


def perturb(
    waveform: np.ndarray,
    name: str,
    level: float,
    *,
    item_id: int,
    seed: int,
    sample_rate: int,
) -> np.ndarray:
    """Apply one perturbation at one level; the result has ``len(waveform)`` samples.

    ``level`` must be one of ``PERTURBATIONS[name]`` or ``NULL_LEVELS[name]``;
    anything else raises ``ValueError``. The null level returns a copy of the
    input, identical bit for bit.
    """
    if name not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {name!r}; expected one of {sorted(PERTURBATIONS)}")
    level = float(level)
    allowed = PERTURBATIONS[name]
    if level != NULL_LEVELS[name] and level not in allowed:
        raise ValueError(
            f"level {level!r} is not allowed for {name!r}; "
            f"expected one of {allowed} or the null level {NULL_LEVELS[name]}"
        )
    if waveform.ndim != 1:
        raise ValueError(f"expected a mono waveform of shape (n,), got shape {waveform.shape}")

    if level == NULL_LEVELS[name]:
        return waveform.copy()

    y = waveform.astype(np.float32, copy=False)
    if name == "noise_snr_db":
        return _add_noise(y, level, item_id=item_id, seed=seed)
    if name == "gain_db":
        return (y * np.float32(10.0 ** (level / 20.0))).astype(np.float32, copy=False)
    if name == "pitch_semitones":
        shifted = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=level)
        return shifted.astype(np.float32, copy=False)
    if name == "tempo_ratio":
        return librosa.util.fix_length(_time_stretch(y, level), size=len(y))
    return _central_crop(y, round(level * sample_rate))


def _add_noise(y: np.ndarray, snr_db: float, *, item_id: int, seed: int) -> np.ndarray:
    """White Gaussian noise whose power is exactly ``power(y) / 10**(snr_db/10)``."""
    signal_power = float(np.mean(y.astype(np.float64) ** 2))
    if signal_power == 0.0:
        raise ValueError("cannot add noise at a given SNR to a silent waveform")
    rng = seeding.generator(seed, "perturbation", "noise_snr_db", snr_db, item_id)
    noise = rng.standard_normal(len(y))
    noise *= math.sqrt(signal_power / 10.0 ** (snr_db / 10.0) / float(np.mean(noise**2)))
    return (y + noise).astype(np.float32)


def _time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Phase-vocoder stretch without changing pitch; length is ``len(y) / rate``."""
    return librosa.effects.time_stretch(y, rate=rate).astype(np.float32, copy=False)


def _central_crop(y: np.ndarray, n_keep: int) -> np.ndarray:
    """Keep the central ``n_keep`` samples and zero everything around them."""
    out = np.zeros_like(y)
    if n_keep >= len(y):
        return y.copy()
    start = (len(y) - n_keep) // 2
    out[start : start + n_keep] = y[start : start + n_keep]
    return out
