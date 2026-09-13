"""Entry point for ``python -m musicsim``.

Kept separate from :mod:`musicsim.cli` so that importing the CLI module for a
test never runs it.
"""

from __future__ import annotations

from musicsim.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
