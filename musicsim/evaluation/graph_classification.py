"""Neighbour-vote classification: how much of a node's label its graph neighbourhood
carries.

Each node is assigned the label that its neighbours vote for, a vote being the
edge weight. This is **transductive**: the neighbours' labels come from the same
gallery as the node being classified, so the numbers describe the graph, not a
model. They are *not* comparable with a classifier fitted on a training split and
scored on unseen items (the inductive setting of the classical-classifier
baselines), and the report must not put them in the same table.

Votes add edge weights instead of counting edges, and a kNN graph built from
cosine similarity can carry zero or negative weights. Sums are therefore compared
as signed numbers over the labels that actually received a vote; a label with a
negative total can still win if every other voted label is more negative. A node
with no neighbour allowed to vote gets :data:`NO_VOTE` (``-1``), which is why
label codes must be non-negative.

Ties on the weight sum go to the smallest label code, which makes the result
independent of the order of a node's neighbours.
"""

from __future__ import annotations

import numpy as np

from musicsim.graphs.base import PADDING, Graph

__all__ = ["NO_VOTE", "neighbor_vote", "vote_metrics"]

#: Prediction of a node none of whose neighbours may vote.
NO_VOTE = -1


def neighbor_vote(
    graph: Graph, labels: np.ndarray, *, exclude_groups: np.ndarray | None = None
) -> np.ndarray:
    """int64 (N,): the label voted by each node's neighbours; -1 when no neighbour may vote.

    ``labels`` holds one non-negative integer code per node. With
    ``exclude_groups`` (one group code per node, e.g. the artist), neighbours of
    the same group as the node do not vote. The winner is the label with the
    largest sum of edge weights; ties go to the smallest code.
    """
    labels = np.asarray(labels)
    n = graph.n_nodes
    if labels.shape != (n,):
        raise ValueError(f"labels {labels.shape} does not have one entry per node ({n})")
    if n and labels.min() < 0:
        raise ValueError(f"labels contain {int(labels.min())}; expected non-negative codes")
    if exclude_groups is not None:
        exclude_groups = np.asarray(exclude_groups)
        if exclude_groups.shape != (n,):
            raise ValueError(
                f"exclude_groups {exclude_groups.shape} does not have one entry per node ({n})"
            )

    allowed = graph.neighbors != PADDING
    if exclude_groups is not None:
        # Padding slots hold -1, which indexes the last node; they are masked right after.
        allowed &= exclude_groups[graph.neighbors] != exclude_groups[:, None]
    voter, slot = np.nonzero(allowed)
    votes = np.full(n, NO_VOTE, dtype=np.int64)
    if voter.size == 0:
        return votes

    n_labels = int(labels.max()) + 1
    key = voter.astype(np.int64) * n_labels + labels[graph.neighbors[voter, slot]]
    unique_keys, inverse = np.unique(key, return_inverse=True)
    totals = np.bincount(
        inverse.ravel(), weights=graph.weights[voter, slot].astype(np.float64)
    )
    rows, codes = np.divmod(unique_keys, n_labels)
    # Per row: largest total first, then smallest code; the first entry of each row wins.
    order = np.lexsort((codes, -totals, rows))
    first = np.ones(order.size, dtype=bool)
    first[1:] = rows[order][1:] != rows[order][:-1]
    winners = order[first]
    votes[rows[winners]] = codes[winners]
    return votes


def vote_metrics(votes: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """``vote_accuracy``, ``vote_macro_f1`` and ``vote_abstention_fraction`` of ``votes``.

    A node without a vote (:data:`NO_VOTE`) is a miss for accuracy and an
    incorrect prediction for F1: it is a false negative of its true class and a
    true positive of none. Macro F1 averages the per-class F1 over the classes
    present in ``labels``; a class that is only predicted, never true, is not
    averaged in. ``vote_abstention_fraction`` is the share of nodes with no vote.
    """
    votes = np.asarray(votes)
    labels = np.asarray(labels)
    if votes.shape != labels.shape or votes.ndim != 1 or votes.size == 0:
        raise ValueError(
            f"votes {votes.shape} and labels {labels.shape} must be equal non-empty 1-D shapes"
        )
    f1 = []
    for code in np.unique(labels):
        true = labels == code
        predicted = votes == code
        tp = int((true & predicted).sum())
        f1.append(2 * tp / (int(true.sum()) + int(predicted.sum())))
    return {
        "vote_accuracy": float((votes == labels).mean()),
        "vote_macro_f1": float(np.mean(f1)),
        "vote_abstention_fraction": float((votes == NO_VOTE).mean()),
    }
