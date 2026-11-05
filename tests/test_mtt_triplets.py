"""Tests for :mod:`musicsim.datasets.mtt_triplets`.

The hand-worked case has two triplets. Over clips (1, 2, 3), three rows give
2 votes for 3 as the outlier, 1 for 1 and 1 for 2; the edges are

    (1,2,3) 2   (2,1,3) 2   (2,3,1) 1   (3,2,1) 1   (1,3,2) 1   (3,1,2) 1

so (1,2,3) and (2,1,3) keep a margin of 2 - 1 = 1, and (3,2,1) and (3,1,2) tie
and cancel. Over clips (4, 5, 6), two rows list the clips in opposite orders
and both vote for 6, so (4,5,6) and (5,4,6) add up to 2 with no opposite edge.
Only the ``slow`` test reads the real CSV.
"""

from __future__ import annotations

import pandas as pd
import pytest

from musicsim import paths
from musicsim.config import ConfigError, load_config
from musicsim.datasets.base import DatasetError
from musicsim.datasets.mtt import MttDataset
from musicsim.datasets.mtt_triplets import (
    CLIP_COLUMNS,
    CONSTRAINT_COLUMNS,
    VOTE_COLUMNS,
    check_constraint_counts,
    constraint_counts,
    derive_constraints,
)

# clip1, clip2, clip3, votes for clip1, clip2, clip3 as the outlier.
_ROWS = [
    (1, 2, 3, 0, 0, 2),
    (1, 2, 3, 1, 0, 0),
    (1, 2, 3, 0, 1, 0),
    (4, 5, 6, 0, 0, 1),
    (6, 5, 4, 1, 0, 0),
]

# anchor, pos, neg, margin, triplet_id.
_EXPECTED = [
    (1, 2, 3, 1, 0),
    (2, 1, 3, 1, 0),
    (4, 5, 6, 2, 1),
    (5, 4, 6, 2, 1),
]

_COUNTS = {
    "rows": 5,
    "votes": 6,
    "edges": 12,
    "unique_edges": 8,
    "constraints": 4,
    "weight": 6,
    "triplets": 2,
    "clips": 6,
}


def _comparisons(rows: list[tuple[int, ...]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[*CLIP_COLUMNS, *VOTE_COLUMNS])


def _as_rows(frame: pd.DataFrame) -> list[tuple[int, ...]]:
    return list(frame.itertuples(index=False, name=None))


# --- derive_constraints ----------------------------------------------------------
def test_opposing_votes_cancel_and_leave_the_margin() -> None:
    constraints = derive_constraints(_comparisons(_ROWS))

    assert list(constraints.columns) == list(CONSTRAINT_COLUMNS)
    assert _as_rows(constraints) == _EXPECTED


def test_a_row_without_votes_adds_nothing() -> None:
    with_silent_row = derive_constraints(_comparisons([*_ROWS, (7, 8, 9, 0, 0, 0)]))
    pd.testing.assert_frame_equal(with_silent_row, derive_constraints(_comparisons(_ROWS)))


def test_the_margin_is_the_vote_surplus() -> None:
    # 5 votes for 3 as the outlier against 2 for 2: (1,2,3) 5 - 2 = 3.
    constraints = derive_constraints(_comparisons([(1, 2, 3, 0, 2, 5)]))
    margins = {(a, p, n): m for a, p, n, m, _ in _as_rows(constraints)}
    assert margins == {(1, 2, 3): 3, (2, 1, 3): 5, (3, 1, 2): 2}


def test_a_full_tie_leaves_no_constraint() -> None:
    constraints = derive_constraints(_comparisons([(1, 2, 3, 1, 1, 1)]))
    assert list(constraints.columns) == list(CONSTRAINT_COLUMNS)
    assert constraints.empty


def test_triplet_ids_do_not_depend_on_row_order() -> None:
    forward = derive_constraints(_comparisons(_ROWS))
    backward = derive_constraints(_comparisons(list(reversed(_ROWS))))
    pd.testing.assert_frame_equal(forward, backward)


def test_a_missing_column_is_refused() -> None:
    comparisons = _comparisons(_ROWS).drop(columns="clip2_numvotes")
    with pytest.raises(DatasetError, match=r"missing columns \['clip2_numvotes'\]"):
        derive_constraints(comparisons)


# --- the figures -----------------------------------------------------------------
def test_constraint_counts_of_the_hand_case() -> None:
    comparisons = _comparisons(_ROWS)
    assert constraint_counts(comparisons, derive_constraints(comparisons)) == _COUNTS


def test_every_wrong_figure_is_named() -> None:
    expected = {**_COUNTS, "votes": 7, "clips": 5}
    with pytest.raises(DatasetError, match="votes: expected 7, got 6; clips: expected 5, got 6"):
        check_constraint_counts(_COUNTS, expected)


def test_expected_figures_must_name_every_count() -> None:
    expected = {name: value for name, value in _COUNTS.items() if name != "weight"}
    with pytest.raises(ConfigError, match="must have the keys"):
        check_constraint_counts(_COUNTS, expected)


# --- the real corpus -------------------------------------------------------------
@pytest.mark.slow
def test_the_real_votes_reproduce_wolff_et_al() -> None:
    config = load_config(paths.REPO_ROOT / "configs" / "datasets" / "mtt.yaml")
    dataset = MttDataset(config)
    if not dataset.comparisons_csv.is_file():
        pytest.skip("MTT not downloaded; run `musicsim download --dataset mtt` first")

    constraints = dataset.build_constraints()
    comparisons = pd.read_csv(dataset.comparisons_csv, sep="\t")

    assert constraint_counts(comparisons, constraints) == {
        "rows": 533,
        "votes": 7650,
        "edges": 15300,
        "unique_edges": 1598,
        "constraints": 860,
        "weight": 6898,
        "triplets": 337,
        "clips": 993,
    }
    assert int((constraints["margin"] >= 2).sum()) == 642
