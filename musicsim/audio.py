"""Audio decoding and the log-mel spectrogram every timbral extractor shares.

Every clip is decoded mono at ``audio.sample_rate`` and trimmed or zero-padded
to exactly ``audio.clip_samples`` (:func:`load_clip`), so that every
spectrogram downstream has the same number of frames and pooled vectors stay
comparable across items. The log-mel spectrogram (:func:`log_mel_spectrogram`)
is kept separate from any particular extractor because a future CNN extractor
takes the same spectrogram as input; MFCC is just a DCT and a pooling step on
top of it.

Spectrograms are cached in float16 under the output root
(:func:`load_or_compute_melspec`), keyed only by the ``audio`` and
``spectrogram`` configuration sections so that every extractor that shares
those settings shares the cache too. The float16 quantisation only affects a
*cache hit*: the very first computation of a spectrogram returns the full
float32 array it was computed in, and only the copy written to disk is
quantised. A cache hit therefore returns values that differ from a fresh
computation at roughly float16 precision (~1e-3 relative) — an accepted
trade-off for the ~2x disk saving, not a bug.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from musicsim import paths
from musicsim.config import Config, config_hash

__all__ = [
    "AudioError",
    "load_clip",
    "load_or_compute_melspec",
    "log_mel_spectrogram",
    "melspec_cache_dir",
]


class AudioError(ValueError):
    """A clip could not be decoded, or decoded to something unusable."""


def load_clip(path: str | Path, config: Config) -> np.ndarray:
    """Decode ``path`` to mono at the configured rate, fixed to ``clip_samples``.

    No ``librosa.effects.trim`` here: it would trim a variable amount depending
    on content and break comparability between clips. FMA excerpts are already
    cut from the middle of the track.
    """
    audio_cfg = config.section("audio")
    sample_rate = int(audio_cfg["sample_rate"])
    clip_seconds = float(audio_cfg["clip_seconds"])
    clip_samples = int(audio_cfg["clip_samples"])
    min_seconds = float(audio_cfg["min_seconds"])
    mono = bool(audio_cfg.get("mono", True))

    try:
        y, _ = librosa.load(str(path), sr=sample_rate, mono=mono, duration=clip_seconds)
    except Exception as exc:
        raise AudioError(f"could not decode {path}: {type(exc).__name__}: {exc}") from exc

    if len(y) < min_seconds * sample_rate:
        raise AudioError(
            f"{path}: decodes to {len(y) / sample_rate:.2f}s, below the {min_seconds}s minimum"
        )

    return librosa.util.fix_length(y, size=clip_samples)


def log_mel_spectrogram(y: np.ndarray, config: Config) -> np.ndarray:
    """Log-mel spectrogram ``(n_mels, n_frames)`` in dB, float32."""
    audio_cfg = config.section("audio")
    spec_cfg = config.section("spectrogram")
    S = librosa.feature.melspectrogram(
        y=y,
        sr=int(audio_cfg["sample_rate"]),
        n_fft=int(spec_cfg["n_fft"]),
        hop_length=int(spec_cfg["hop_length"]),
        n_mels=int(spec_cfg["n_mels"]),
        fmin=float(spec_cfg["fmin"]),
        fmax=float(spec_cfg["fmax"]),
        power=2.0,
    )
    return librosa.power_to_db(S).astype(np.float32, copy=False)


def melspec_cache_dir(dataset: str, config: Config, *, create: bool = False) -> Path:
    """Cache directory for log-mel spectrograms.

    Keyed only by the ``audio`` and ``spectrogram`` sections (not the whole
    configuration) so that MFCC and, later, the CNN reuse the same cached
    spectrograms instead of each recomputing them under their own hash.
    """
    resolved = config.to_dict()
    key = config_hash(
        {"audio": resolved.get("audio", {}), "spectrogram": resolved.get("spectrogram", {})}
    )
    return paths.cache_dir(dataset, "melspec", key, create=create)


def load_or_compute_melspec(path: str | Path, item_id: int, config: Config) -> np.ndarray:
    """Log-mel spectrogram of one item, float32, cached in float16 on disk."""
    dataset = config.require("dataset.directory")
    cache = bool(config.get("runtime.cache", True))
    cache_dir = melspec_cache_dir(dataset, config, create=cache)
    cache_path = cache_dir / f"{int(item_id):06d}.npy"

    if cache and cache_path.is_file():
        return np.load(cache_path).astype(np.float32)

    y = load_clip(path, config)
    S = log_mel_spectrogram(y, config)

    if cache:
        np.save(cache_path, S.astype(np.float16))

    return S
