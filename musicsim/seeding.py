"""Seeding and determinism.

Two rules hold everywhere in this project:

* every source of randomness is seeded from a single integer that the
  configuration carries and the run log records, and
* a stage that needs randomness asks for its own generator through
  :func:`generator`, instead of drawing from the global NumPy state.

The second rule is what makes a run reproducible when stages are added,
reordered or skipped: the bootstrap of experiment ``e01`` draws the same
numbers whether or not a plot was produced before it.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np

__all__ = ["derive_seed", "generator", "seed_everything", "torch_is_available"]

#: Environment variable that CPython reads for hash randomisation. Set before
#: the interpreter starts; recorded so the run log can show what it was.
HASHSEED_ENV = "PYTHONHASHSEED"


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed the standard library, NumPy and, when installed, PyTorch.

    ``deterministic`` additionally asks PyTorch for deterministic kernels and
    switches cuDNN benchmarking off. That costs speed and is worth it here: an
    experiment whose numbers move between runs cannot be reported.

    Returns a small dictionary describing what was seeded, which
    :mod:`musicsim.runlog` stores in ``run.json``.
    """
    seed = int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed must fit in 32 unsigned bits, got {seed}")

    random.seed(seed)
    np.random.seed(seed)

    record: dict[str, Any] = {
        "seed": seed,
        "deterministic": bool(deterministic),
        "python_hash_seed": os.environ.get(HASHSEED_ENV),
        "torch": None,
    }

    torch = _import_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        record["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        }

    return record


def generator(seed: int, *stream: str | int) -> np.random.Generator:
    """A NumPy generator for one named stream, independent of every other one.

    ``generator(42, "bootstrap", "e01")`` always returns the same sequence, and
    a different one from ``generator(42, "bootstrap", "e02")``. Stages take
    their generator from here so that their numbers do not depend on what ran
    before them in the same process.
    """
    if not stream:
        return np.random.default_rng(int(seed))
    label = "|".join(str(part) for part in stream).encode("utf-8")
    digest = hashlib.sha256(label).digest()[:8]
    return np.random.default_rng([int(seed), int.from_bytes(digest, "big")])


def derive_seed(seed: int, *stream: str | int) -> int:
    """A reproducible 32-bit seed for a named stream, for libraries that want an int.

    ``scikit-learn`` estimators take ``random_state=int``; this gives each of
    them a distinct but reproducible value derived from the run seed.
    """
    label = "|".join(str(part) for part in (seed, *stream)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(label).digest()[:4], "big")


def torch_is_available() -> bool:
    """Whether PyTorch can be imported (it is an optional, phase P10 dependency)."""
    return _import_torch() is not None


def _import_torch() -> Any:
    """Import PyTorch, or return ``None`` when the deep-learning extra is absent."""
    try:
        import torch
    except ImportError:
        return None
    return torch
