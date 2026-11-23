"""Grouped cross-validation folds over the MagnaTagATune similarity constraints.

A model trained on the 860 constraints of :mod:`musicsim.datasets.mtt_triplets`
can only be tested on constraints it did not see, so they are split into folds
for cross-validation, ten as in Wolff et al. (2012; key ``Wolff2012`` in
``docs/sources.md``). The unit of the split is not the constraint, nor the
triplet, but the **connected component of clips**: two triplets that share a
clip are linked, and splitting them would let that clip reach the test fold
through training. Every clip therefore lands in exactly one fold, and so do the
constraints of a triplet, which share all three clips. On the real CSV there are
319 components, the largest of 7 clips and 8 constraints, so no giant component
swallows the corpus.

**Component ids.** Components are numbered from 0 in increasing order of the
smallest ``triplet_id`` they contain, so the same constraints always yield the
same ids, whatever the row order (our own choice; any deterministic order would
do).

**Balancing.** The folds are balanced by number of constraints, since that is
what each fold trains and tests on. Components are shuffled with the caller's
generator and then dealt, largest first (stable sort, so the shuffle orders
components of equal size), each to the fold with the fewest constraints so far;
ties go to the lowest fold. This is the classic longest-first greedy rule for
balancing loads (our own choice). The shuffle is what makes folds differ between
seeds.

**Balance criterion.** Fixed before looking at any result: no fold may deviate
from the mean size ``n_constraints / n_folds`` by more than ``max_imbalance``
(10 % in ``configs/datasets/mtt.yaml``). A split that misses it, or a component
with more constraints than a mean fold, is an explicit :class:`DatasetError`
that asks for fewer folds: the number of folds is never lowered silently.

Everything here is pure; the generator comes from :mod:`musicsim.seeding`
under the caller's own stream name.
"""

from __future__ import annotations

import logging
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from musicsim.datasets.base import DatasetError

__all__ = [
    "COMPONENT_COLUMN",
    "FOLD_COLUMN",
    "MIN_FOLDS",
    "constraint_components",
    "constraint_folds",
]

logger = logging.getLogger(__name__)

#: Name of the series returned by :func:`constraint_components`.
COMPONENT_COLUMN = "component"

#: Name of the series returned by :func:`constraint_folds`.
FOLD_COLUMN = "fold"

#: Fewest folds a cross-validation can have: one to test, at least one to train.
MIN_FOLDS = 2

#: The columns of a constraints frame that the split reads.
_CLIP_COLUMNS: tuple[str, ...] = ("anchor", "pos", "neg")
_REQUIRED_COLUMNS: tuple[str, ...] = (*_CLIP_COLUMNS, "triplet_id")


def constraint_components(constraints: pd.DataFrame) -> pd.Series:
    """The connected component of clips of every constraint, aligned with ``constraints``.

    ``constraints`` has at least the columns ``anchor, pos, neg, triplet_id``
    of :func:`musicsim.datasets.mtt_triplets.derive_constraints`. Two clips are
    connected when they appear in the same constraint. Ids are numbered from 0
    by the smallest ``triplet_id`` of each component. Raises
    :class:`DatasetError` if a column is missing.
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in constraints.columns]
    if missing:
        raise DatasetError(
            f"constraints are missing columns {missing}; expected {list(_REQUIRED_COLUMNS)}"
        )
    if constraints.empty:
        return pd.Series([], index=constraints.index, dtype=np.int64, name=COMPONENT_COLUMN)

    clips = constraints[list(_CLIP_COLUMNS)].to_numpy(dtype=np.int64)
    unique_clips, codes = np.unique(clips, return_inverse=True)
    codes = codes.reshape(clips.shape)
    sources = np.concatenate([codes[:, 0], codes[:, 0]])
    targets = np.concatenate([codes[:, 1], codes[:, 2]])
    graph = sparse.coo_matrix(
        (np.ones(len(sources), dtype=np.int8), (sources, targets)),
        shape=(len(unique_clips), len(unique_clips)),
    )
    _, clip_labels = connected_components(graph, directed=False)
    raw = clip_labels[codes[:, 0]]

    first_triplet = pd.Series(constraints["triplet_id"].to_numpy(dtype=np.int64)).groupby(raw).min()
    ordered = first_triplet.sort_values(kind="stable").index.to_numpy()
    renumber = np.empty(len(ordered), dtype=np.int64)
    renumber[ordered] = np.arange(len(ordered))
    return pd.Series(renumber[raw], index=constraints.index, name=COMPONENT_COLUMN)


def constraint_folds(
    constraints: pd.DataFrame,
    *,
    n_folds: int,
    max_imbalance: float,
    generator: np.random.Generator,
) -> pd.Series:
    """The fold, 0 to ``n_folds - 1``, of every constraint, aligned with ``constraints``.

    Whole components go to one fold (:func:`constraint_components`), balanced
    by number of constraints as the module docstring explains. Raises
    :class:`ValueError` if ``n_folds`` is not an integer between
    :data:`MIN_FOLDS` and the number of components, or ``max_imbalance`` is not
    in (0, 1); :class:`DatasetError` if a component is larger than a mean fold
    or the folds deviate from the mean size by more than ``max_imbalance``.
    """
    components = constraint_components(constraints)
    n_components = int(components.nunique())
    if (
        isinstance(n_folds, bool)
        or not isinstance(n_folds, Integral)
        or not MIN_FOLDS <= n_folds <= n_components
    ):
        raise ValueError(
            f"n_folds must be an integer between {MIN_FOLDS} and the number of "
            f"components ({n_components}), got {n_folds!r}"
        )
    if not 0 < max_imbalance < 1:
        raise ValueError(f"max_imbalance must be in (0, 1), got {max_imbalance!r}")

    labels = components.to_numpy()
    sizes = np.bincount(labels, minlength=n_components)
    mean_size = len(constraints) / n_folds
    largest = int(sizes.argmax())
    if sizes[largest] > mean_size:
        raise DatasetError(
            f"component {largest} has {int(sizes[largest])} constraints, more than a "
            f"mean fold of {mean_size:.1f} with n_folds={n_folds}; no grouped split can "
            "be balanced, lower n_folds"
        )

    fold_of_component = _deal(sizes, n_folds, generator)
    fold_sizes = np.bincount(fold_of_component, weights=sizes, minlength=n_folds).astype(np.int64)
    deviation = float(np.abs(fold_sizes - mean_size).max() / mean_size)
    if deviation > max_imbalance:
        raise DatasetError(
            f"the {n_folds} folds have {fold_sizes.tolist()} constraints, up to "
            f"{deviation:.1%} away from the mean {mean_size:.1f}; the criterion allows "
            f"{max_imbalance:.1%}, lower n_folds"
        )
    logger.info(
        "%d constraints in %d components split into %d folds of %s constraints "
        "(largest deviation from the mean %.1f%%)",
        len(constraints),
        n_components,
        n_folds,
        fold_sizes.tolist(),
        100 * deviation,
    )
    return pd.Series(fold_of_component[labels], index=constraints.index, name=FOLD_COLUMN)


def _deal(sizes: np.ndarray, n_folds: int, generator: np.random.Generator) -> np.ndarray:
    """Fold of each component: shuffled, then largest first to the emptiest fold."""
    shuffled = generator.permutation(len(sizes))
    order = shuffled[np.argsort(-sizes[shuffled], kind="stable")]
    loads = np.zeros(n_folds, dtype=np.int64)
    fold_of_component = np.empty(len(sizes), dtype=np.int64)
    for component in order:
        fold = int(loads.argmin())
        fold_of_component[component] = fold
        loads[fold] += sizes[component]
    return fold_of_component
