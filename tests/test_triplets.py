"""Tests for the triplet agreement: hand-computed hits, the three variants, the
clustered and paired bootstraps, and the evaluator end to end on a toy corpus."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import musicsim.evaluation  # noqa: F401  (registers "triplets")
from musicsim import paths
from musicsim.config import Config, ConfigError
from musicsim.datasets.base import Dataset
from musicsim.evaluation.triplets import (
    VARIANTS,
    agreement_ci,
    constraint_hits,
    paired_agreement_test,
    triplet_agreement,
)
from musicsim.experiments import run_experiment
from musicsim.extractors.base import Extractor
from musicsim.registry import register_dataset, register_extractor

# --- hits and variants, by hand ------------------------------------------------
# Clip 1 points along x, clip 2 along y, clip 3 halfway (with norm 2*sqrt(2), to
# show that the rows are brought to unit norm). cos(1, 2) = 0 and
# cos(1, 3) = cos(2, 3) = 1/sqrt(2).
_IDS = np.array([1, 2, 3])
_X = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
_CONSTRAINTS = pd.DataFrame(
    {
        # (1, 2, 3): cos(1, 2) = 0 < cos(1, 3) -> 0.  (3, 1, 2): a tie -> 0.5.
        "anchor": [1, 3],
        "pos": [2, 1],
        "neg": [3, 2],
        "margin": [1, 3],
        "triplet_id": [0, 0],
    }
)
_HITS = np.array([0.0, 0.5])


def test_constraint_hits_by_hand() -> None:
    np.testing.assert_array_equal(constraint_hits(_X, _IDS, _CONSTRAINTS), _HITS)


def test_constraint_hits_counts_a_win() -> None:
    swapped = _CONSTRAINTS.assign(pos=[3, 1], neg=[2, 2])  # (1, 3, 2): cos(1, 3) > cos(1, 2)
    assert constraint_hits(_X, _IDS, swapped)[0] == 1.0


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("unweighted", 0.25),  # (0 + 0.5) / 2
        ("vote_weighted", 0.375),  # (0 * 1 + 0.5 * 3) / 4
        ("margin_ge_2", 0.5),  # only the second constraint
    ],
)
def test_triplet_agreement_variants(variant: str, expected: float) -> None:
    assert triplet_agreement(_HITS, _CONSTRAINTS["margin"], variant=variant) == expected


def test_constraint_hits_rejects_a_clip_without_embedding() -> None:
    # Clip 3 is the anchor of the second constraint, checked before pos and neg.
    with pytest.raises(ValueError, match=r"1 anchor clip\(s\).*\[3\]"):
        constraint_hits(_X[:2], _IDS[:2], _CONSTRAINTS)


def test_constraint_hits_rejects_an_all_zero_embedding() -> None:
    X = _X.copy()
    X[1] = 0.0
    with pytest.raises(ValueError, match=r"all-zero embedding.*\[2\]"):
        constraint_hits(X, _IDS, _CONSTRAINTS)


def test_triplet_agreement_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="variant must be one of"):
        triplet_agreement(_HITS, _CONSTRAINTS["margin"], variant="median")


def test_triplet_agreement_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="row-aligned"):
        triplet_agreement(_HITS, np.array([1, 2, 3]), variant="unweighted")


def test_triplet_agreement_rejects_no_constraint() -> None:
    with pytest.raises(ValueError, match="no constraints"):
        triplet_agreement(np.array([]), np.array([], dtype=int), variant="unweighted")


@pytest.mark.parametrize("margin", [np.array([1.5, 2.0]), np.array([0, 2])])
def test_triplet_agreement_rejects_margins_that_are_not_votes(margin: np.ndarray) -> None:
    with pytest.raises(ValueError, match="whole number of votes"):
        triplet_agreement(_HITS, margin, variant="unweighted")


def test_margin_ge_2_without_such_constraint() -> None:
    with pytest.raises(ValueError, match="no constraint has a margin >= 2"):
        triplet_agreement(_HITS, np.array([1, 1]), variant="margin_ge_2")


# --- bootstrap -----------------------------------------------------------------
def test_vote_weighted_interval_resamples_whole_weighted_triplets() -> None:
    # Triplet 0: hits (1, 0) with margins (3, 1), weighted mean 0.75. Triplet 1:
    # hit 0 with margin 2. A resample holds two triplets: both 0 (0.75), both 1
    # (0), or one of each (3 / 6 = 0.5), the first two with probability 1/4
    # each, so the 95 % interval is exactly [0, 0.75]. A per-row resampling of
    # the repeated constraints would not give these endpoints.
    hits = np.array([1.0, 0.0, 0.0])
    margin = np.array([3, 1, 2])
    triplet_id = np.array([0, 0, 1])
    result = agreement_ci(
        hits, margin, triplet_id, variant="vote_weighted",
        n_resamples=1000, confidence=0.95, seed=0,
    )  # fmt: skip
    assert result.mean == 0.5
    assert (result.ci_low, result.ci_high) == (0.0, 0.75)


def test_paired_test_of_a_representation_with_itself() -> None:
    for variant in VARIANTS:
        test = paired_agreement_test(
            _HITS, _HITS, _CONSTRAINTS["margin"], _CONSTRAINTS["triplet_id"], variant=variant,
            n_resamples=200, confidence=0.95, seed=0,
        )  # fmt: skip
        assert (test.mean, test.ci_low, test.ci_high, test.p_value) == (0.0, 0.0, 0.0, 1.0)


def test_paired_test_rejects_different_constraint_sets() -> None:
    with pytest.raises(ValueError, match="same constraints"):
        paired_agreement_test(
            _HITS, _HITS[:1], _CONSTRAINTS["margin"], _CONSTRAINTS["triplet_id"],
            variant="unweighted", n_resamples=10, confidence=0.95, seed=0,
        )  # fmt: skip


def test_random_vectors_agree_by_chance(rng: np.random.Generator) -> None:
    # 4000 constraints over 500 random 32-d clips: the agreement is a mean of
    # 4000 near-independent fair coins (sd ~0.008), so it lands within 0.03 of 0.5.
    n_clips, n_constraints = 500, 4000
    X = rng.standard_normal((n_clips, 32))
    ids = np.arange(n_clips)
    triples = np.array([rng.choice(n_clips, size=3, replace=False) for _ in range(n_constraints)])
    constraints = pd.DataFrame(triples, columns=["anchor", "pos", "neg"])
    hits = constraint_hits(X, ids, constraints)
    margin = np.ones(n_constraints, dtype=int)
    assert triplet_agreement(hits, margin, variant="unweighted") == pytest.approx(0.5, abs=0.03)


# --- the evaluator end to end ----------------------------------------------------
# Clips 1-6 on the unit circle; clip 7 has no constraint. Constraints:
#   triplet 0: (1, 2, 4) hit, (2, 1, 4) hit           margins 2, 1
#   triplet 1: (4, 5, 1) hit, (4, 1, 5) miss          margins 3, 1
#   triplet 2: (3, 6, 2) miss                          margin 1
# Hits 1, 1, 1, 0, 0: unweighted 3/5, vote-weighted 6/8, margin >= 2 1.
_ANGLES = {1: 0.0, 2: 10.0, 3: 20.0, 4: 170.0, 5: 180.0, 6: 190.0, 7: 90.0}
_TOY_CONSTRAINTS = pd.DataFrame(
    {
        "anchor": [1, 2, 4, 4, 3],
        "pos": [2, 1, 5, 1, 6],
        "neg": [4, 4, 1, 5, 2],
        "margin": [2, 1, 3, 1, 1],
        "triplet_id": [0, 0, 1, 1, 2],
    }
)
_TOY_DIRECTORY = "toy_triplets"


@register_extractor("toy_triplets")
class _ToyTripletExtractor(Extractor):
    def extract_one(self, item_id: int, path: str) -> np.ndarray:
        angle = np.radians(_ANGLES[item_id])
        return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)


class _ToyTripletDataset(Dataset):
    extra_columns = ("in_triplets",)

    @property
    def triplet_constraints_path(self) -> Path:
        return paths.dataset_dir(self.directory_name) / "triplet_constraints.csv"

    def download(self, *, force: bool = False) -> None:
        raise NotImplementedError("the toy corpus is written by the test")

    def build_index(self) -> pd.DataFrame:
        ids = sorted(_ANGLES)
        return pd.DataFrame(
            {
                "item_id": ids,
                "path": [f"synthetic/{i}.wav" for i in ids],
                "split": ["test"] * len(ids),
                "artist_id": ids,
                "group_id": ids,
                "genre_top": pd.NA,
                "genres_all": pd.NA,
                "in_triplets": [i != 7 for i in ids],
            }
        )


def _config(triplets: dict | None = None, **extra: object) -> Config:
    return Config(
        data={
            "experiment": "e00_toy_triplets",
            "seed": 0,
            "dataset": {"name": _TOY_DIRECTORY, "directory": _TOY_DIRECTORY},
            "extractor": {"name": "toy_triplets", "dim": 2},
            "embedding": {
                "fit_on": "all",
                "standardize": False,
                "l2_normalize": True,
                "pca": {"enabled": False, "n_components": None},
            },
            "bootstrap": {"n_resamples": 200, "confidence": 0.95},
            "evaluation": {
                "retrieval": {"enabled": False},
                **({} if triplets is None else {"triplets": {"enabled": True, **triplets}}),
            },
            "runtime": {"n_jobs": 1, "progress": False, "cache": True},
            "stages": ["evaluate"],
            **extra,
        }
    )


@pytest.fixture
def toy_corpus(isolated_roots: dict[str, Path]) -> _ToyTripletDataset:
    register_dataset(_TOY_DIRECTORY)(_ToyTripletDataset)
    dataset = _ToyTripletDataset(_config())
    dataset.write_index()
    _TOY_CONSTRAINTS.to_csv(dataset.triplet_constraints_path, index=False)
    return dataset


# The second representation is the same extractor under another seed: it scores
# every constraint exactly as the main one does.
_TWO_REPRESENTATIONS = [
    "toy_triplets",
    {"label": "copy", "extractor": "toy_triplets", "overrides": {"seed": 5}},
]


def test_evaluator_writes_agreement_per_variant(toy_corpus: _ToyTripletDataset) -> None:
    run = run_experiment(_config({}))
    summary = pd.read_csv(run.metrics_dir / "triplets.csv")
    assert list(summary.columns) == [
        "representation",
        "seed",
        "variant",
        "value",
        "ci_low",
        "ci_high",
        "n_constraints",
        "n_triplets",
    ]
    rows = summary.set_index("variant")
    assert rows.loc["unweighted", "value"] == pytest.approx(3 / 5)
    assert rows.loc["vote_weighted", "value"] == pytest.approx(6 / 8)
    assert rows.loc["margin_ge_2", "value"] == 1.0
    assert rows["n_constraints"].to_dict() == {
        "unweighted": 5,
        "vote_weighted": 5,
        "margin_ge_2": 2,
    }
    assert rows["n_triplets"].to_dict() == {"unweighted": 3, "vote_weighted": 3, "margin_ge_2": 2}
    assert (summary["ci_low"] <= summary["value"]).all()
    assert (summary["value"] <= summary["ci_high"]).all()

    per_constraint = pd.read_csv(run.metrics_dir / "triplets_per_constraint.csv")
    assert list(per_constraint.columns) == [
        "representation",
        "seed",
        "anchor",
        "pos",
        "neg",
        "margin",
        "triplet_id",
        "hit",
    ]
    assert per_constraint["hit"].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert not (run.metrics_dir / "triplets_deltas.csv").exists()


def test_evaluator_is_off_by_default(toy_corpus: _ToyTripletDataset) -> None:
    run = run_experiment(_config())
    assert not (run.metrics_dir / "triplets.csv").exists()


def test_evaluator_writes_paired_deltas_with_holm(toy_corpus: _ToyTripletDataset) -> None:
    config = _config(
        {"comparisons": [["toy_triplets", "copy"], ["copy", "toy_triplets"]]},
        representations=_TWO_REPRESENTATIONS,
    )
    run = run_experiment(config)
    summary = pd.read_csv(run.metrics_dir / "triplets.csv")
    assert summary["seed"].tolist() == [0] * 3 + [5] * 3
    # Same hits, same seed per variant: identical intervals for both representations.
    by_rep = summary.pivot(index="variant", columns="representation", values="ci_low")
    np.testing.assert_array_equal(by_rep["toy_triplets"], by_rep["copy"])

    deltas = pd.read_csv(run.metrics_dir / "triplets_deltas.csv")
    assert list(deltas.columns) == [
        "representation_a",
        "representation_b",
        "variant",
        "delta",
        "ci_low",
        "ci_high",
        "p_value",
        "p_holm",
        "reject_holm",
        "holm_status",
    ]
    assert len(deltas) == 2 * len(VARIANTS)
    assert (deltas["delta"] == 0).all() and (deltas["p_value"] == 1).all()
    headline = deltas["variant"] == "unweighted"
    assert (deltas.loc[headline, "p_holm"] == 1).all()
    assert (~deltas.loc[headline, "reject_holm"].astype(bool)).all()
    assert (deltas.loc[headline, "holm_status"] == "provisional").all()
    assert deltas.loc[~headline, ["p_holm", "reject_holm", "holm_status"]].isna().all().all()


@pytest.mark.parametrize(
    ("comparisons", "match"),
    [
        ("toy_triplets,copy", "list of \\[a, b\\] pairs"),
        ([["toy_triplets"]], "must be a pair"),
        ([["toy_triplets", "mert"]], "names \\['mert'\\]"),
        ([["copy", "copy"]], "with itself"),
    ],
)
def test_evaluator_rejects_bad_comparisons(
    toy_corpus: _ToyTripletDataset, comparisons: object, match: str
) -> None:
    config = _config({"comparisons": comparisons}, representations=_TWO_REPRESENTATIONS)
    with pytest.raises(ConfigError, match=match):
        run_experiment(config)


class _PlainDataset(_ToyTripletDataset):
    """The same corpus, without similarity judgements (like FMA)."""

    triplet_constraints_path = None


def test_evaluator_needs_a_dataset_with_constraints(isolated_roots: dict[str, Path]) -> None:
    register_dataset("toy_plain")(_PlainDataset)
    config = _config({})
    data = config.to_dict()
    data["dataset"] = {"name": "toy_plain", "directory": "toy_plain"}
    config = Config(data=data, source=config.source)
    _PlainDataset(config).write_index()
    with pytest.raises(ConfigError, match="dataset 'toy_plain' has no triplet constraints"):
        run_experiment(config)


def test_evaluator_needs_the_constraints_file(toy_corpus: _ToyTripletDataset) -> None:
    toy_corpus.triplet_constraints_path.unlink()
    with pytest.raises(FileNotFoundError, match="musicsim index --dataset toy_triplets"):
        run_experiment(_config({}))


def test_evaluator_rejects_constraints_without_a_column(toy_corpus: _ToyTripletDataset) -> None:
    _TOY_CONSTRAINTS.drop(columns="margin").to_csv(toy_corpus.triplet_constraints_path, index=False)
    with pytest.raises(ValueError, match=r"lacks column\(s\) \['margin'\]"):
        run_experiment(_config({}))


def test_evaluator_rejects_constraints_from_another_build(
    toy_corpus: _ToyTripletDataset,
) -> None:
    # Clip 7 is not flagged in_triplets but a constraint now uses it.
    _TOY_CONSTRAINTS.assign(neg=[4, 4, 1, 5, 7]).to_csv(
        toy_corpus.triplet_constraints_path, index=False
    )
    with pytest.raises(ValueError, match=r"(1 unflagged, 0 unused)"):
        run_experiment(_config({}))
