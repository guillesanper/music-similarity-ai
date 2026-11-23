"""Tests for :mod:`musicsim.datasets.mtt_folds`.

The hand-built constraints are written triplet by triplet: ``_triplet(clips, n,
triplet_id)`` gives ``n`` constraints over one set of three clips, so the size of
each component in constraints is known by construction. Only the ``slow`` test
reads the real constraints.
"""

from __future__ import annotations

import pandas as pd
import pytest

from musicsim import paths
from musicsim.config import load_config
from musicsim.datasets.base import DatasetError
from musicsim.datasets.mtt import MttDataset
from musicsim.datasets.mtt_folds import (
    COMPONENT_COLUMN,
    FOLD_COLUMN,
    constraint_components,
    constraint_folds,
)
from musicsim.seeding import generator

_COLUMNS = ["anchor", "pos", "neg", "margin", "triplet_id"]

#: The project seed of ``configs/base.yaml``; ``mtt.yaml`` alone does not carry it.
_PROJECT_SEED = 42


def _triplet(clips: tuple[int, int, int], n: int, triplet_id: int) -> list[tuple[int, ...]]:
    """``n`` constraints (anchor, pos, neg, margin 1, triplet_id) over three clips."""
    a, b, c = clips
    orders = [(a, b, c), (b, a, c), (a, c, b), (c, a, b), (b, c, a), (c, b, a)]
    return [(*order, 1, triplet_id) for order in orders[:n]]


def _frame(rows: list[tuple[int, ...]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_COLUMNS)


def _disjoint(sizes: list[int]) -> pd.DataFrame:
    """One single-triplet component per entry of ``sizes``, triplet ids in order."""
    rows: list[tuple[int, ...]] = []
    for triplet_id, n in enumerate(sizes):
        first = 3 * triplet_id
        rows += _triplet((first, first + 1, first + 2), n, triplet_id)
    return _frame(rows)


def _linked() -> pd.DataFrame:
    """30 components: 24 lone triplets and 6 pairs of triplets that share a clip."""
    rows: list[tuple[int, ...]] = []
    triplet_id = 0
    clip = 0
    for component in range(30):
        n_triplets = 2 if component % 5 == 0 else 1
        for _ in range(n_triplets):
            rows += _triplet((clip, clip + 1, clip + 2), 2 + triplet_id % 2, triplet_id)
            triplet_id += 1
            clip += 2  # the next triplet of the component reuses the last clip
        clip += 1
    return _frame(rows)


def _clips_by(constraints: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Every distinct label of each clip, over the three clip columns."""
    long = pd.concat(
        [pd.DataFrame({"clip": constraints[c], "label": labels}) for c in ("anchor", "pos", "neg")]
    )
    return long.groupby("clip")["label"].nunique()


# --- constraint_components -------------------------------------------------------
def test_triplets_sharing_a_clip_form_one_component_numbered_by_triplet_id() -> None:
    # Clip 3 is the negative of triplet 1 and the anchor of triplet 2.
    constraints = _frame(
        [
            *_triplet((6, 7, 8), 2, 0),
            *_triplet((1, 2, 3), 2, 1),
            *_triplet((3, 4, 5), 3, 2),
            *_triplet((9, 10, 11), 2, 3),
        ]
    )
    components = constraint_components(constraints)

    assert components.name == COMPONENT_COLUMN
    assert components.tolist() == [0, 0, 1, 1, 1, 1, 1, 2, 2]


def test_component_ids_do_not_depend_on_row_order() -> None:
    constraints = _linked()
    shuffled = constraints.sample(frac=1.0, random_state=0)
    pd.testing.assert_series_equal(
        constraint_components(shuffled).sort_index(), constraint_components(constraints)
    )


def test_no_constraints_give_no_components() -> None:
    components = constraint_components(_frame([]))
    assert components.empty
    assert components.name == COMPONENT_COLUMN


def test_a_missing_column_is_refused() -> None:
    constraints = _disjoint([2, 2]).drop(columns="triplet_id")
    with pytest.raises(DatasetError, match=r"missing columns \['triplet_id'\]"):
        constraint_components(constraints)


# --- constraint_folds ------------------------------------------------------------
def test_no_clip_component_or_triplet_is_split_across_folds() -> None:
    constraints = _linked()
    folds = constraint_folds(
        constraints, n_folds=5, max_imbalance=0.1, generator=generator(0, "folds")
    )
    components = constraint_components(constraints)

    assert folds.name == FOLD_COLUMN
    assert set(folds) == set(range(5))
    assert (_clips_by(constraints, folds) == 1).all()
    assert (folds.groupby(components).nunique() == 1).all()
    assert (folds.groupby(constraints["triplet_id"]).nunique() == 1).all()


def test_the_folds_are_balanced_in_a_hand_case() -> None:
    # Sizes 3, 3, 2, 2, 2, 2 into 2 folds: 3 and 3 open the folds, then the
    # four 2s alternate, so both end with 7 constraints.
    folds = constraint_folds(
        _disjoint([3, 3, 2, 2, 2, 2]), n_folds=2, max_imbalance=0.1, generator=generator(0, "f")
    )
    assert folds.value_counts().sort_index().tolist() == [7, 7]


def test_the_folds_keep_the_index_of_the_constraints() -> None:
    constraints = _disjoint([2, 2, 2, 2]).set_axis(list("abcdefgh"))
    folds = constraint_folds(constraints, n_folds=2, max_imbalance=0.1, generator=generator(0))
    assert folds.index.tolist() == list("abcdefgh")


def test_the_same_seed_gives_the_same_folds_and_another_seed_others() -> None:
    constraints = _linked()

    def split(seed: int) -> pd.Series:
        return constraint_folds(
            constraints, n_folds=5, max_imbalance=0.1, generator=generator(seed, "folds")
        )

    pd.testing.assert_series_equal(split(0), split(0))
    assert not split(0).equals(split(1))


@pytest.mark.parametrize("n_folds", [1, 0, 5, 2.0, True])
def test_n_folds_out_of_range_is_refused(n_folds: object) -> None:
    # Four components: 5 folds is one too many.
    with pytest.raises(ValueError, match=r"between 2 and the number of components \(4\)"):
        constraint_folds(
            _disjoint([2, 2, 2, 2]), n_folds=n_folds, max_imbalance=0.1, generator=generator(0)
        )


@pytest.mark.parametrize("max_imbalance", [0.0, 1.0, -0.1])
def test_max_imbalance_out_of_range_is_refused(max_imbalance: float) -> None:
    with pytest.raises(ValueError, match=r"max_imbalance must be in \(0, 1\)"):
        constraint_folds(
            _disjoint([2, 2, 2, 2]), n_folds=2, max_imbalance=max_imbalance, generator=generator(0)
        )


def test_a_giant_component_is_an_error() -> None:
    # Three triplets chained into one component of 6 constraints, plus two of
    # 2: a mean fold of 10 / 2 = 5 cannot hold it.
    constraints = _frame(
        [
            *_triplet((0, 1, 2), 2, 0),
            *_triplet((2, 3, 4), 2, 1),
            *_triplet((4, 5, 6), 2, 2),
            *_triplet((10, 11, 12), 2, 3),
            *_triplet((20, 21, 22), 2, 4),
        ]
    )
    with pytest.raises(DatasetError, match=r"component 0 has 6 constraints.*lower n_folds"):
        constraint_folds(constraints, n_folds=2, max_imbalance=0.1, generator=generator(0))


def test_a_split_that_misses_the_criterion_is_an_error() -> None:
    # Sizes 3, 3, 3, 2 into 3 folds (mean 11 / 3): the folds end with 5, 3 and
    # 3 constraints, 36.4 % away from the mean.
    with pytest.raises(DatasetError, match=r"\[5, 3, 3\] constraints, up to 36.4%"):
        constraint_folds(
            _disjoint([3, 3, 3, 2]), n_folds=3, max_imbalance=0.1, generator=generator(0)
        )


# --- the real constraints --------------------------------------------------------
@pytest.mark.slow
def test_the_real_constraints_split_into_balanced_folds() -> None:
    config = load_config(paths.REPO_ROOT / "configs" / "datasets" / "mtt.yaml")
    dataset = MttDataset(config)
    if not dataset.triplet_constraints_path.is_file():
        pytest.skip("MTT not indexed; run `musicsim index --dataset mtt` first")
    constraints = pd.read_csv(dataset.triplet_constraints_path)

    components = constraint_components(constraints)
    long = pd.concat(
        [
            pd.DataFrame({"clip": constraints[c], "component": components})
            for c in ("anchor", "pos", "neg")
        ]
    )
    clips_per_component = long.drop_duplicates().groupby("component").size()
    assert components.nunique() == 319
    assert clips_per_component.max() == 7

    n_folds = config.require("folds.n_folds")
    max_imbalance = config.require("folds.max_imbalance")
    folds = constraint_folds(
        constraints,
        n_folds=n_folds,
        max_imbalance=max_imbalance,
        generator=generator(_PROJECT_SEED, "folds"),
    )
    sizes = folds.value_counts()
    mean_size = len(constraints) / n_folds
    assert len(sizes) == n_folds
    assert int(sizes.sum()) == 860
    assert float((sizes - mean_size).abs().max() / mean_size) <= max_imbalance
    assert (_clips_by(constraints, folds) == 1).all()
