"""Negative control: a fixed-seed vector carrying no audio information.

Every metric reported for a real extractor is reported for this one too, so
that a reader can see "better than chance" directly. It never touches
``path`` — the vector is derived purely from the run seed and the item id
through :func:`musicsim.seeding.generator`, so it is reproducible without
decoding any audio.
"""

from __future__ import annotations

import numpy as np

from musicsim import seeding
from musicsim.extractors.base import Extractor
from musicsim.registry import register_extractor

__all__ = ["RandomExtractor"]


@register_extractor("random")
class RandomExtractor(Extractor):
    """Draws each item's vector from ``N(0, 1)``, seeded by run seed + item id."""

    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        distribution = self.config.get("extractor.distribution", "normal")
        if distribution != "normal":
            raise ValueError(f"unknown extractor.distribution {distribution!r}")

        seed = self.config.require("seed")
        stream = self.config.get("extractor.seed_stream", "extractor/random")
        rng = seeding.generator(seed, stream, item_id)
        return rng.normal(size=self.dim).astype(np.float32)
