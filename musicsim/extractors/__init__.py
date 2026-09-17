"""Feature extractors. Importing this package registers every one of them.

Adding an extractor means adding one module here (decorated with
``@register_extractor``) and listing it below, following
:mod:`musicsim.registry`'s plugin convention.
"""

from __future__ import annotations

from musicsim.extractors import mfcc, random  # noqa: F401

__all__: list[str] = []
