"""Tests for the neighbour vote, on graphs small enough to have hand-computed answers."""

from __future__ import annotations

import numpy as np
import pytest

from musicsim.evaluation.graph_classification import NO_VOTE, neighbor_vote, vote_metrics
from musicsim.graphs.base import Graph

LABELS = np.array([0, 0, 1, 1, 2, 2])

# Row i: neighbours and weights, best first. Node 3 has a single neighbour (padded row);
# node 4 has only negative weights.
SIX = Graph(
    neighbors=np.array([[1, 2], [2, 4], [3, 0], [4, -1], [0, 2], [4, 3]]),
    weights=np.array(
        [[0.9, 0.8], [0.5, 0.7], [0.5, 0.5], [0.4, 0.0], [-0.1, -0.5], [0.3, 0.2]],
        dtype=np.float32,
    ),
    ids=np.arange(6),
)


def test_vote_by_weight_by_code_and_with_negative_weights() -> None:
    # node 0: label 0 (0.9) beats label 1 (0.8)
    # node 1: label 2 (0.7) beats label 1 (0.5): decided by weight, not by code
    # node 2: labels 1 and 0 tie at 0.5: the smaller code wins
    # node 3: one real neighbour, the padding slot does not vote
    # node 4: -0.1 (label 0) beats -0.5 (label 1): signed sums, not counts
    # node 5: label 2 (0.3) beats label 1 (0.2)
    expected = [0, 2, 0, 2, 0, 2]
    np.testing.assert_array_equal(neighbor_vote(SIX, LABELS), expected)
    assert neighbor_vote(SIX, LABELS).dtype == np.int64


def test_sums_are_over_all_neighbours_of_a_label() -> None:
    # Two neighbours of label 1 (0.3 each) outweigh one of label 0 (0.5).
    graph = Graph(
        neighbors=np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]),
        weights=np.array([[0.5, 0.3, 0.3]] * 4, dtype=np.float32),
        ids=np.arange(4),
    )
    votes = neighbor_vote(graph, np.array([0, 0, 1, 1]))
    assert votes[0] == 1  # neighbours 1 (label 0, .5), 2 and 3 (label 1, .3 + .3)


def test_excluded_groups_do_not_vote() -> None:
    groups = np.array([0, 0, 1, 1, 2, 2])
    # Same-artist neighbours drop out: node 0 loses neighbour 1, node 2 loses 3, node 5 loses 4.
    np.testing.assert_array_equal(
        neighbor_vote(SIX, LABELS, exclude_groups=groups), [1, 2, 0, 2, 0, 1]
    )


def test_node_whose_neighbours_are_all_its_own_artist_gets_no_vote() -> None:
    graph = Graph(
        neighbors=np.array([[1, 2], [0, 2], [0, 1]]),
        weights=np.full((3, 2), 0.5, dtype=np.float32),
        ids=np.arange(3),
    )
    labels = np.array([0, 1, 1])
    np.testing.assert_array_equal(
        neighbor_vote(graph, labels, exclude_groups=np.zeros(3)), [NO_VOTE] * 3
    )
    # Node 2 is the only one of another artist: nodes 0 and 1 hear it, node 2 hears no one.
    np.testing.assert_array_equal(
        neighbor_vote(graph, labels, exclude_groups=np.array([0, 0, 1])), [1, 1, 0]
    )


def test_vote_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="one entry per node"):
        neighbor_vote(SIX, LABELS[:3])
    with pytest.raises(ValueError, match="non-negative"):
        neighbor_vote(SIX, LABELS - 1)
    with pytest.raises(ValueError, match="exclude_groups"):
        neighbor_vote(SIX, LABELS, exclude_groups=np.zeros(2))


def test_vote_metrics_by_hand() -> None:
    votes = np.array([0, 2, 0, 2, 0, 2])
    # Correct: nodes 0 and 5. Class 0: tp 1, true 2, predicted 3 -> 0.4; class 1: never
    # predicted -> 0; class 2: tp 1, true 2, predicted 3 -> 0.4.
    assert vote_metrics(votes, LABELS) == pytest.approx(
        {
            "vote_accuracy": 2 / 6,
            "vote_macro_f1": 0.8 / 3,
            "vote_abstention_fraction": 0.0,
        }
    )


def test_abstention_is_a_miss_and_an_incorrect_prediction() -> None:
    votes = np.array([0, NO_VOTE, 1, 1, 2, 2])
    # Class 0: tp 1, true 2, predicted 1 -> 2/3; classes 1 and 2 are perfect.
    assert vote_metrics(votes, LABELS) == pytest.approx(
        {
            "vote_accuracy": 5 / 6,
            "vote_macro_f1": (2 / 3 + 2) / 3,
            "vote_abstention_fraction": 1 / 6,
        }
    )
