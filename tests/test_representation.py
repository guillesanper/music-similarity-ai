"""Tests for the representation diagnostics, on matrices small enough to work out by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from musicsim.evaluation.representation import effective_rank, linear_cka, mean_cosine

# -- effective_rank ------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 5, 32])
def test_effective_rank_of_identity_is_d(d: int) -> None:
    assert effective_rank(np.eye(d)) == pytest.approx(d, rel=1e-12)


def test_effective_rank_of_two_unequal_singular_values() -> None:
    # Singular values 3 and 1: p = [3/4, 1/4].
    entropy = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert effective_rank(np.diag([3.0, 1.0])) == pytest.approx(math.exp(entropy), rel=1e-12)


def test_effective_rank_ignores_zero_singular_values() -> None:
    # Singular values 2, 2, 0: p = [1/2, 1/2] once 0 log 0 = 0, so exp(log 2) = 2.
    assert effective_rank(np.diag([2.0, 2.0, 0.0])) == pytest.approx(2.0, rel=1e-12)


def test_effective_rank_of_collapse_to_one_point_is_one() -> None:
    # Every row the same non-zero vector: rank one, a single non-zero singular value.
    X = np.tile([0.6, 0.8, 0.0], (10, 1))
    assert effective_rank(X) == pytest.approx(1.0, rel=1e-12)


def test_effective_rank_is_not_centred() -> None:
    # Centring the identity would leave rank d - 1; the uncentred definition keeps d.
    assert effective_rank(np.eye(4)) == pytest.approx(4.0, rel=1e-12)
    assert effective_rank(np.eye(4) - np.eye(4).mean(axis=0)) == pytest.approx(3.0, rel=1e-12)


def test_effective_rank_is_scale_invariant(rng: np.random.Generator) -> None:
    X = rng.normal(size=(20, 6))
    assert effective_rank(-7.5 * X) == pytest.approx(effective_rank(X), rel=1e-12)


def test_effective_rank_accepts_float32() -> None:
    assert effective_rank(np.eye(3, dtype=np.float32)) == pytest.approx(3.0, rel=1e-12)


def test_effective_rank_rejects_all_zeros() -> None:
    with pytest.raises(ValueError, match="all zeros"):
        effective_rank(np.zeros((3, 2)))


# -- mean_cosine ---------------------------------------------------------------


def test_mean_cosine_of_identical_vectors_is_one() -> None:
    X = np.tile([1.0, -2.0, 0.5], (6, 1))
    assert mean_cosine(X) == pytest.approx(1.0, abs=1e-12)


def test_mean_cosine_of_orthonormal_rows_is_zero() -> None:
    # Only pairs i != j count; with the diagonal this would be 1/4.
    assert mean_cosine(np.eye(4)) == pytest.approx(0.0, abs=1e-12)


def test_mean_cosine_of_opposite_vectors_is_minus_one() -> None:
    assert mean_cosine(np.array([[1.0, 0.0], [-3.0, 0.0]])) == pytest.approx(-1.0, abs=1e-12)


def test_mean_cosine_by_hand() -> None:
    # Pairs: cos(e1, e2) = 0, cos(e1, e1+e2) = cos(e2, e1+e2) = 1/sqrt(2). Mean over 3 pairs.
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert mean_cosine(X) == pytest.approx(math.sqrt(2) / 3, rel=1e-12)


def test_mean_cosine_equals_brute_force_over_pairs(rng: np.random.Generator) -> None:
    X = rng.normal(size=(15, 4)) + 0.5
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = U @ U.T
    off_diagonal = S[~np.eye(len(X), dtype=bool)]
    assert mean_cosine(X) == pytest.approx(off_diagonal.mean(), rel=1e-12)


def test_mean_cosine_ignores_row_norms(rng: np.random.Generator) -> None:
    X = rng.normal(size=(8, 3))
    scales = np.arange(1.0, 9.0)[:, None]
    assert mean_cosine(scales * X) == pytest.approx(mean_cosine(X), rel=1e-12)


def test_mean_cosine_rejects_zero_row() -> None:
    with pytest.raises(ValueError, match=r"1 zero row\(s\) \(first at row 1\)"):
        mean_cosine(np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]))


def test_mean_cosine_rejects_single_row() -> None:
    with pytest.raises(ValueError, match="at least 2 row"):
        mean_cosine(np.ones((1, 3)))


# -- linear_cka ----------------------------------------------------------------


def test_linear_cka_with_itself_is_one(rng: np.random.Generator) -> None:
    X = rng.normal(size=(30, 5))
    assert linear_cka(X, X) == pytest.approx(1.0, rel=1e-12)


def test_linear_cka_invariant_to_orthogonal_transform(rng: np.random.Generator) -> None:
    X = rng.normal(size=(30, 5))
    Y = rng.normal(size=(30, 4)) + X[:, :4]
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    assert linear_cka(X @ Q, Y) == pytest.approx(linear_cka(X, Y), rel=1e-10)
    assert linear_cka(X @ Q, X) == pytest.approx(1.0, rel=1e-12)


def test_linear_cka_invariant_to_isotropic_scale(rng: np.random.Generator) -> None:
    X = rng.normal(size=(30, 5))
    Y = rng.normal(size=(30, 3))
    assert linear_cka(4.0 * X, 0.1 * Y) == pytest.approx(linear_cka(X, Y), rel=1e-12)


def test_linear_cka_invariant_to_translation(rng: np.random.Generator) -> None:
    X = rng.normal(size=(30, 5))
    Y = rng.normal(size=(30, 3))
    assert linear_cka(X + 10.0, Y - 3.0) == pytest.approx(linear_cka(X, Y), rel=1e-10)


def test_linear_cka_of_one_dimensional_inputs_by_hand() -> None:
    # Centred x = [-1, 0, 1] and y = [-1, 1, 0]: (x.y)^2 / (|x|^2 |y|^2) = 1 / (2 * 2).
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[1.0], [3.0], [2.0]])
    assert linear_cka(X, Y) == pytest.approx(0.25, rel=1e-12)
    assert linear_cka(Y, X) == pytest.approx(0.25, rel=1e-12)


def test_linear_cka_of_uncorrelated_inputs_is_zero() -> None:
    # Centred x = [-1, 1, -1, 1] and y = [-1, -1, 1, 1] are orthogonal.
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    Y = np.array([[0.0], [0.0], [1.0], [1.0]])
    assert linear_cka(X, Y) == pytest.approx(0.0, abs=1e-12)


def test_linear_cka_rejects_different_row_counts() -> None:
    with pytest.raises(ValueError, match="X has 3 rows and Y has 4"):
        linear_cka(np.ones((3, 2)), np.ones((4, 2)))


@pytest.mark.parametrize("constant", ["X", "Y"])
def test_linear_cka_rejects_constant_representation(constant: str) -> None:
    varying = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    flat = np.tile([5.0, -1.0], (3, 1))
    X, Y = (flat, varying) if constant == "X" else (varying, flat)
    with pytest.raises(ValueError, match=f"{constant} is the same vector for every item"):
        linear_cka(X, Y)


# -- input validation shared by the three --------------------------------------


@pytest.mark.parametrize("bad", [np.ones(3), np.ones((2, 2, 2)), np.ones((0, 3)), np.ones((3, 0))])
@pytest.mark.parametrize(
    "call",
    [effective_rank, mean_cosine, lambda X: linear_cka(X, np.ones((3, 2)))],
    ids=["effective_rank", "mean_cosine", "linear_cka"],
)
def test_rejects_badly_shaped_input(call, bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="expected a 2-D array"):
        call(bad)


def test_linear_cka_rejects_badly_shaped_y() -> None:
    with pytest.raises(ValueError, match="Y has shape"):
        linear_cka(np.ones((3, 2)), np.ones(3))


@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.parametrize(
    "call",
    [effective_rank, mean_cosine, lambda X: linear_cka(np.eye(3), X)],
    ids=["effective_rank", "mean_cosine", "linear_cka"],
)
def test_rejects_non_finite_input(call, value: float) -> None:
    X = np.eye(3)
    X[1, 2] = value
    with pytest.raises(ValueError, match="non-finite"):
        call(X)
