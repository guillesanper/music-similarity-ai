"""Deterministic perturbations of a decoded waveform, to test how stable a
representation's neighbourhoods are when the audio is degraded.

The conditions are a closed list (:data:`CONDITIONS`): adding one after seeing
results would be a new analysis, not a variant of this one. Every perturbation
is a pure function of its arguments; the only random one, additive noise, draws
from the generator the caller passes in, so the same clip always gets the same
noise when the caller seeds it per clip.

Design decisions worth knowing:

* **Output length equals input length.** Time stretching and cropping change
  how many samples there are; the result is trimmed at the end or zero-padded
  at the end (tempo) or kept as the central window with silence around it
  (crop), so the log-mel spectrogram keeps its number of frames and pooled
  vectors stay comparable with the originals. The price is that padding adds
  silence, which is part of what a crop does to a clip and is not hidden.
* **``identity`` is the control.** It returns a copy of the input, so an
  unchanged neighbourhood (Jaccard@10 = 1) is a testable property of the
  pipeline.
* **Gain does not clip.** A boost can push samples beyond ±1; the waveform is
  float and the feature extractors work in float, so nothing overflows, and
  saturating would make the output energy differ from the requested gain.
* **Noise is scaled to the requested SNR exactly**, using the realised noise
  power rather than its expectation, so the measured SNR matches the request.
  A silent clip has no signal power to be a fraction of and is returned as is.

Features of perturbed audio (:func:`perturbed_features`)
--------------------------------------------------------
Extractors read audio from a path, so the perturbed clip is written to a
temporary 32-bit float WAV (lossless for a float32 waveform) and extracted with
``runtime.cache: false``. That is not an optimisation: the log-mel cache is
indexed by ``item_id`` and would hand back the *original* clip's spectrogram.
Only extractors verified to honour ``runtime.cache`` are admitted
(:data:`CACHE_SAFE_EXTRACTORS`); any other might keep a cache of its own indexed
by ``item_id`` and silently return unperturbed features.

The WAVs of a condition (about 2.6 MB per clip) live in a temporary directory
and are removed when the condition is done. The resulting feature vectors are
cached, per condition and level, under the key of the raw features.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed

from musicsim import audio, paths, seeding
from musicsim.config import Config, ConfigError, config_hash
from musicsim.embeddings import features_cache_dir
from musicsim.registry import EXTRACTORS, load_plugins

__all__ = ["CACHE_SAFE_EXTRACTORS", "CONDITIONS", "perturb", "perturbed_features"]

#: Extractors verified to respect ``runtime.cache: false`` (see the module docstring).
CACHE_SAFE_EXTRACTORS: frozenset[str] = frozenset({"mfcc"})

#: The closed set of conditions (``identity`` is the control).
CONDITIONS: tuple[str, ...] = ("identity", "noise", "gain", "pitch", "tempo", "crop")


def perturb(
    y: np.ndarray, sample_rate: int, condition: str, level: float, *, rng: np.random.Generator
) -> np.ndarray:
    """A perturbed copy of ``y``, float32, always the same length as ``y``.

    ``level`` is the SNR in dB for ``noise``, dB for ``gain``, semitones for
    ``pitch``, a rate factor for ``tempo`` and seconds for ``crop``; it is
    ignored by ``identity``. Only ``noise`` consumes ``rng``.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {list(CONDITIONS)}")
    if y.ndim != 1:
        raise ValueError(f"expected a mono waveform of shape (n,), got shape {y.shape}")
    y = y.astype(np.float32, copy=True)
    if condition == "identity":
        return y

    level = float(level)
    if condition == "noise":
        if not math.isfinite(level):
            raise ValueError(f"noise level (SNR in dB) must be finite, got {level!r}")
        return _add_noise(y, level, rng)
    if condition == "gain":
        if not math.isfinite(level):
            raise ValueError(f"gain level (dB) must be finite, got {level!r}")
        return (y * np.float32(10.0 ** (level / 20.0))).astype(np.float32, copy=False)
    if condition == "pitch":
        if not math.isfinite(level):
            raise ValueError(f"pitch level (semitones) must be finite, got {level!r}")
        shifted = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=level)
        return shifted.astype(np.float32, copy=False)
    if condition == "tempo":
        if not math.isfinite(level) or level <= 0:
            raise ValueError(f"tempo level (rate factor) must be positive, got {level!r}")
        stretched = librosa.effects.time_stretch(y, rate=level).astype(np.float32, copy=False)
        return librosa.util.fix_length(stretched, size=len(y))
    seconds = level
    if not math.isfinite(seconds) or seconds <= 0 or seconds * sample_rate > len(y):
        raise ValueError(
            f"crop level (seconds) must be in (0, {len(y) / sample_rate:g}] for this clip, "
            f"got {level!r}"
        )
    return _central_crop(y, round(seconds * sample_rate))


def _add_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise whose power is exactly ``power(y) / 10**(snr_db/10)``."""
    signal_power = float(np.mean(y.astype(np.float64) ** 2))
    if signal_power == 0.0:
        return y
    noise = rng.standard_normal(len(y))
    noise *= math.sqrt(signal_power / 10.0 ** (snr_db / 10.0) / float(np.mean(noise**2)))
    return (y + noise).astype(np.float32)


def _central_crop(y: np.ndarray, n_keep: int) -> np.ndarray:
    """Keep the central ``n_keep`` samples and zero everything around them."""
    out = np.zeros_like(y)
    start = (len(y) - n_keep) // 2
    out[start : start + n_keep] = y[start : start + n_keep]
    return out


def perturbed_features(
    config: Config, index: pd.DataFrame, condition: str, level: float
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
    """``(X, ids, failures)`` of the items of ``index`` (``item_id``, ``path``), perturbed.

    ``X`` is ``(N, dim)`` raw features, ``ids[i]`` the item of row ``X[i]``, in
    the order of ``index``; ``failures`` lists ``(item_id, reason)`` for items
    that could not be decoded, perturbed or extracted. Each clip is perturbed
    with ``seeding.generator(seed, "perturbation", condition, level, item_id)``,
    so the result does not depend on the order of ``index`` or on what else is
    in it. Results without failures are cached per condition and level.
    """
    name = str(config.require("extractor.name"))
    if name not in CACHE_SAFE_EXTRACTORS:
        raise ConfigError(
            f"extractor {name!r} is not verified to respect runtime.cache=false; it may cache "
            f"features by item_id and return the original clip's. Admitted: "
            f"{sorted(CACHE_SAFE_EXTRACTORS)}"
        )
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {list(CONDITIONS)}")

    ids = index["item_id"].to_numpy(dtype=np.int64)
    use_cache = bool(config.get("runtime.cache", True))
    cache_dir = paths.cache_dir(
        config.require("dataset.directory"),
        f"perturbed-{name}",
        config_hash(
            {
                "features": features_cache_dir(config).name,
                "condition": condition,
                "level": float(level),
            }
        ),
    )
    features_path, ids_path = cache_dir / "features.npy", cache_dir / "ids.npy"
    cached = features_path.is_file() and ids_path.is_file()
    if use_cache and cached and np.array_equal(np.load(ids_path), ids):
        return np.load(features_path), ids, []

    load_plugins()
    seed = int(config.require("seed"))
    sample_rate = int(config.require("audio.sample_rate"))
    n_jobs = int(config.get("runtime.n_jobs", -1))
    data = config.to_dict()
    data["runtime"] = {**data.get("runtime", {}), "cache": False}
    extractor = EXTRACTORS.get(name)(Config(data=data, source=config.source))

    with tempfile.TemporaryDirectory(prefix="musicsim-perturbed-") as scratch:
        written = Parallel(n_jobs=n_jobs)(
            delayed(_write_perturbed)(
                int(item_id),
                str(path),
                Path(scratch),
                config,
                condition,
                float(level),
                seed,
                sample_rate,
            )
            for item_id, path in zip(index["item_id"], index["path"], strict=True)
        )
        failures = [(item, error) for item, wav, error in written if wav is None]
        derived = pd.DataFrame(
            {
                "item_id": [item for item, wav, _ in written if wav is not None],
                "path": [wav for _, wav, _ in written if wav is not None],
            }
        )
        X, got_ids, extract_failures = extractor.extract_all(
            derived, n_jobs=n_jobs, progress=bool(config.get("runtime.progress", True))
        )
    failures += extract_failures

    if use_cache and not failures:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(features_path, X)
        np.save(ids_path, got_ids)
    return X, got_ids, failures


def _write_perturbed(
    item_id: int,
    path: str,
    directory: Path,
    config: Config,
    condition: str,
    level: float,
    seed: int,
    sample_rate: int,
) -> tuple[int, str | None, str | None]:
    """Worker: perturb one clip and write it as float WAV; the failure as text, never raised."""
    try:
        y = audio.load_clip(path, config)
        rng = seeding.generator(seed, "perturbation", condition, str(level), item_id)
        out = perturb(y, sample_rate, condition, level, rng=rng)
        wav = directory / f"{item_id:08d}.wav"
        sf.write(str(wav), out, sample_rate, subtype="FLOAT")
    except Exception as exc:
        return item_id, None, f"{type(exc).__name__}: {exc}"
    return item_id, str(wav), None
