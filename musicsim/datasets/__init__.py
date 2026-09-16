"""Corpus implementations, registered by name in :mod:`musicsim.registry`.

Importing this package is what populates :data:`musicsim.registry.DATASETS`;
:func:`musicsim.registry.load_plugins` does this automatically, and the ``download``
and ``index`` subcommands of :mod:`musicsim.cli` call it before looking a dataset up.
"""

from __future__ import annotations

from musicsim.datasets import fma  # noqa: F401 - imported for its registration side effect

__all__: list[str] = []
