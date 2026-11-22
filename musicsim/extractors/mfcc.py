"""MFCC baseline: log-mel -> DCT -> deltas -> mean/std pooling, 80 dims.

Equivalent to ``librosa.feature.mfcc(S=log_mel, n_mfcc=n_mfcc + 1)[1:]``, written
out explicitly with :func:`scipy.fft.dct` because that is what the archived
pipeline did and this extractor is checked against its reference output (see
``tests/test_extractors_mfcc.py``).
"""

from __future__ import annotations

import librosa
import numpy as np
import scipy.fft

from musicsim import audio
from musicsim.extractors.base import Extractor
from musicsim.registry import register_extractor

__all__ = ["MfccExtractor"]


@register_extractor("mfcc")
class MfccExtractor(Extractor):
    """20 MFCCs (c0 dropped) + their deltas, pooled to mean/std -> 80 dims."""

    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        S = audio.load_or_compute_melspec(path, item_id, self.config)

        n_mfcc = int(self.config.require("extractor.n_mfcc"))
        drop_c0 = bool(self.config.get("extractor.drop_c0", True))
        delta_width = int(self.config.require("extractor.delta_width"))

        n_coeffs = n_mfcc + 1 if drop_c0 else n_mfcc
        mfcc = scipy.fft.dct(S, axis=0, type=2, norm="ortho")[:n_coeffs]
        if drop_c0:
            mfcc = mfcc[1:]

        delta = librosa.feature.delta(mfcc, width=delta_width, axis=-1)
        vec = np.concatenate(
            [mfcc.mean(axis=1), mfcc.std(axis=1), delta.mean(axis=1), delta.std(axis=1)]
        )
        return vec.astype(np.float32, copy=False)
