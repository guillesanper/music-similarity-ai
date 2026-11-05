"""Consistent similarity constraints from the odd-one-out votes of MagnaTagATune.

``comparisons_final.csv`` records the bonus mode of TagATune: players hear
three clips and vote for the one most different from the other two. This module
turns those votes into relative-similarity constraints, following Wolff et al.
(2012, §3.2; key ``Wolff2012`` in ``docs/sources.md``):

- a vote for the outlier *k* of a triplet *(i, j, k)* says the other two are
  more alike, and yields two ordered edges, ``d(i, j) < d(i, k)`` (anchor *i*)
  and ``d(j, i) < d(j, k)`` (anchor *j*);
- the votes are summed per ordered edge ``(anchor, pos, neg)`` over every row,
  since one set of three clips can be presented in several rows;
- each edge loses the weight of its opposite ``(anchor, neg, pos)``, which
  removes the two-cycles, and only the edges with a positive remainder survive.
  That remainder is the constraint's ``margin``; an exact tie cancels both.

On the real CSV this yields 15 300 edges, 1598 of them distinct, and 860
consistent constraints of total weight 6898 over 337 triplets and 993 clips:
Wolff et al.'s figures exactly. Because every triplet result of the report
rests on them, :func:`check_constraint_counts` turns any deviation into an
error rather than a warning.

**Triplet ids.** A triplet is the set of three clips of a constraint, and it is
the cluster of the bootstrap of the agreement metric (``docs/ground-truth-study.md``
§4.3). The sets are numbered from 0 in lexicographic order of their sorted clip
ids, so the same CSV always yields the same ids (our own choice; any
deterministic order would do).

Everything here is pure: reading and writing the files is
:class:`musicsim.datasets.mtt.MttDataset`'s job.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from musicsim.config import ConfigError
from musicsim.datasets.base import DatasetError

__all__ = [
    "CLIP_COLUMNS",
    "CONSTRAINT_COLUMNS",
    "COUNT_NAMES",
    "VOTE_COLUMNS",
    "check_constraint_counts",
    "constraint_counts",
    "derive_constraints",
]

#: The columns of ``comparisons_final.csv`` that hold the three clip ids, and
#: the votes each clip received as the outlier, in the same order.
CLIP_COLUMNS: tuple[str, ...] = ("clip1_id", "clip2_id", "clip3_id")
VOTE_COLUMNS: tuple[str, ...] = ("clip1_numvotes", "clip2_numvotes", "clip3_numvotes")

#: Columns of the frame returned by :func:`derive_constraints`.
CONSTRAINT_COLUMNS: tuple[str, ...] = ("anchor", "pos", "neg", "margin", "triplet_id")

#: The figures :func:`constraint_counts` reports, in the order of the derivation.
COUNT_NAMES: tuple[str, ...] = (
    "rows",
    "votes",
    "edges",
    "unique_edges",
    "constraints",
    "weight",
    "triplets",
    "clips",
)


def _vote_edges(comparisons: pd.DataFrame) -> pd.DataFrame:
    """``anchor, pos, neg, weight``: the votes summed per distinct ordered edge."""
    missing = [c for c in (*CLIP_COLUMNS, *VOTE_COLUMNS) if c not in comparisons.columns]
    if missing:
        raise DatasetError(
            f"comparisons are missing columns {missing}; expected {[*CLIP_COLUMNS, *VOTE_COLUMNS]}"
        )
    clips = comparisons[list(CLIP_COLUMNS)].to_numpy(dtype=np.int64)
    votes = comparisons[list(VOTE_COLUMNS)].to_numpy(dtype=np.int64)

    parts = []
    for outlier in range(len(CLIP_COLUMNS)):
        first, second = (position for position in range(len(CLIP_COLUMNS)) if position != outlier)
        for anchor, pos in ((first, second), (second, first)):
            parts.append(
                pd.DataFrame(
                    {
                        "anchor": clips[:, anchor],
                        "pos": clips[:, pos],
                        "neg": clips[:, outlier],
                        "weight": votes[:, outlier],
                    }
                )
            )
    edges = pd.concat(parts, ignore_index=True)
    edges = edges[edges["weight"] > 0]
    return edges.groupby(["anchor", "pos", "neg"], as_index=False)["weight"].sum()


def derive_constraints(comparisons: pd.DataFrame) -> pd.DataFrame:
    """anchor, pos, neg, margin, triplet_id — one row per consistent constraint (Wolff et al. 2012).

    ``comparisons`` has the columns :data:`CLIP_COLUMNS` and
    :data:`VOTE_COLUMNS` of ``comparisons_final.csv``. Each row of the result
    states ``d(anchor, pos) < d(anchor, neg)``; ``margin`` is the vote surplus
    over the opposite edge (at least 1). Rows are sorted by ``triplet_id``,
    then ``anchor, pos, neg``. Raises :class:`DatasetError` if a column is
    missing.
    """
    edges = _vote_edges(comparisons)
    opposite = edges.rename(columns={"pos": "neg", "neg": "pos", "weight": "opposite"})
    edges = edges.merge(opposite, on=["anchor", "pos", "neg"], how="left")
    edges["margin"] = edges["weight"] - edges["opposite"].fillna(0).astype(np.int64)
    constraints = edges.loc[edges["margin"] > 0, ["anchor", "pos", "neg", "margin"]]

    members = np.sort(constraints[["anchor", "pos", "neg"]].to_numpy(dtype=np.int64), axis=1)
    _, triplet_ids = np.unique(members, axis=0, return_inverse=True)
    constraints = constraints.assign(triplet_id=triplet_ids.reshape(-1).astype(np.int64))
    constraints = constraints.sort_values(["triplet_id", "anchor", "pos", "neg"])
    return constraints[list(CONSTRAINT_COLUMNS)].reset_index(drop=True)


def constraint_counts(comparisons: pd.DataFrame, constraints: pd.DataFrame) -> dict[str, int]:
    """The figures of the derivation, keyed by :data:`COUNT_NAMES`.

    ``edges`` counts two per vote; ``unique_edges`` the distinct ordered edges
    before the cancellation; ``weight`` is the sum of the margins.
    """
    edges = _vote_edges(comparisons)
    return {
        "rows": len(comparisons),
        "votes": int(comparisons[list(VOTE_COLUMNS)].to_numpy().sum()),
        "edges": int(edges["weight"].sum()),
        "unique_edges": len(edges),
        "constraints": len(constraints),
        "weight": int(constraints["margin"].sum()),
        "triplets": int(constraints["triplet_id"].nunique()),
        "clips": len(np.unique(constraints[["anchor", "pos", "neg"]].to_numpy())),
    }


def check_constraint_counts(counts: Mapping[str, int], expected: Mapping[str, int]) -> None:
    """Raise :class:`DatasetError` naming every figure that differs from ``expected``.

    ``expected`` must have exactly the keys :data:`COUNT_NAMES`; otherwise
    :class:`ConfigError`, since a figure left out would go unchecked.
    """
    if set(expected) != set(COUNT_NAMES):
        raise ConfigError(
            f"expected constraint counts must have the keys {list(COUNT_NAMES)}, "
            f"got {sorted(expected)}"
        )
    wrong = [
        f"{name}: expected {expected[name]}, got {counts[name]}"
        for name in COUNT_NAMES
        if counts[name] != expected[name]
    ]
    if wrong:
        raise DatasetError(
            "the triplet constraints do not reproduce the expected figures ("
            + "; ".join(wrong)
            + "). Check that comparisons_final.csv is the unmodified download."
        )
