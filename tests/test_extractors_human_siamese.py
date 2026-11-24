"""Tests for the head trained on MTT's human constraints (the ``human_siamese`` extractor).

The input extractor is ``random`` (8 normal numbers per clip, no audio). The
synthetic MTT corpus has 12 triplets, each its own connected component with two
constraints, ``(a, b, c)`` and ``(c, b, a)``: 36 constrained clips and 24
constraints, plus 4 clips without any. Three outer folds hold 8 constraints and
12 clips each; the training side of a fold (16 constraints) splits into two inner
folds of 8. Clip ids start at :data:`FIRST_ID`, so they never coincide with row
positions. Some tests replace the training of a fold by hand-made models (a
constant one-hot of the fold, or the identity) to check the out-of-fold assembly
against known values; the others train for real, for two epochs. Skipped as a
whole without the optional deep-learning extra.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402 (after the torch skip)

from musicsim import paths  # noqa: E402
from musicsim.config import Config, ConfigError  # noqa: E402
from musicsim.evaluation.triplets import constraint_hits  # noqa: E402
from musicsim.experiments import extract_stage  # noqa: E402
from musicsim.extractors import human_siamese  # noqa: E402
from musicsim.extractors.human_siamese import (  # noqa: E402
    CROSS_VALIDATION_SUMMARY,
    FINAL_DIR,
    FINAL_SUMMARY,
    FOLDS_FILE,
    HumanSiameseExtractor,
    final_selection,
    selection_grid,
)
from musicsim.extractors.vector_siamese import (  # noqa: E402
    RESOLVED_CONFIG,
    standardize_on_training,
)
from musicsim.registry import EXTRACTORS, load_plugins  # noqa: E402
from musicsim.representations import RepresentationSpec, spec_config  # noqa: E402
from musicsim.training import constraints as training  # noqa: E402
from musicsim.training import train as train_module  # noqa: E402
from musicsim.training.constraints import (  # noqa: E402
    FOLD_SUMMARY,
    SELECTION_FILE,
    VAL_AGREEMENT,
    FoldResult,
    GridPoint,
)
from musicsim.training.models import MlpHead  # noqa: E402
from musicsim.training.train import BEST_CHECKPOINT  # noqa: E402

SEED = 3

N_TRIPLETS = 12
CLIPS_PER_TRIPLET = 3
N_CLIPS = N_TRIPLETS * CLIPS_PER_TRIPLET
N_CONSTRAINTS = 2 * N_TRIPLETS
FIRST_ID = 1000
#: Clips of the corpus that no constraint uses.
UNCONSTRAINED_IDS = [2000, 2001, 2002, 2003]

N_FOLDS = 3
FOLDS = {"n_folds": N_FOLDS, "max_imbalance": 0.1}
CONSTRAINTS_PER_FOLD = N_CONSTRAINTS // N_FOLDS
CLIPS_PER_FOLD = N_CLIPS // N_FOLDS

#: The head: 8 random inputs -> 16 hidden -> 4 dimensions.
D_IN = 8
HIDDEN = 16
DIM = 4
#: Trainable parameters, by hand: Linear(8->16, no bias) 128 + BatchNorm(16)
#: scale and shift 32 + Linear(16->4) 64 weights and 4 biases.
N_PARAMETERS = 8 * 16 + 2 * 16 + 16 * 4 + 4
#: The linear variant: Linear(8->4) 32 weights and 4 biases.
N_LINEAR_PARAMETERS = 8 * 4 + 4

GRID = {"margin": [0.2, 0.5], "lr": [1e-2], "weight_decay": [0.0]}

#: A second corpus the final model is applied to: 6 training and 4 test items.
TOY_IDS = list(range(1, 11))
TOY_SPLITS = ["training"] * 6 + ["test"] * 4


# --- fixtures and helpers --------------------------------------------------------
def _config(
    *,
    seed: int = SEED,
    hidden: tuple[int, ...] = (HIDDEN,),
    directory: str = "mtt",
    folds: dict | None = None,
    corpus_folds: dict | None = None,
) -> Config:
    data = {
        "seed": seed,
        "dataset": {"name": directory, "directory": directory},
        "runtime": {"n_jobs": 1, "progress": False, "cache": True},
        "extractor": {
            "name": "human_siamese",
            "input": {"extractor": "random", "overrides": {"dim": D_IN}},
            "hidden": list(hidden),
            "dropout": 0.0,
            "dim": DIM,
            "folds": dict(FOLDS if folds is None else folds),
            "training": {
                "grid": GRID,
                "inner_folds": 2,
                "batch_size": 4,
                "max_epochs": 2,
                "patience": 10,
            },
        },
    }
    if directory == "mtt":
        data["folds"] = dict(FOLDS if corpus_folds is None else corpus_folds)
    return Config(data=data)


def _constraints() -> pd.DataFrame:
    rows = []
    for t in range(N_TRIPLETS):
        a, b, c = (FIRST_ID + CLIPS_PER_TRIPLET * t + np.arange(CLIPS_PER_TRIPLET)).tolist()
        rows += [(a, b, c, 1, t), (c, b, a, 1, t)]
    return pd.DataFrame(rows, columns=["anchor", "pos", "neg", "margin", "triplet_id"])


def _mtt_index(ids: list[int]) -> pd.DataFrame:
    constrained = set(_constraints()[["anchor", "pos", "neg"]].to_numpy().ravel().tolist())
    return pd.DataFrame(
        {
            "item_id": ids,
            "path": [f"unused/{item_id}.mp3" for item_id in ids],
            "split": "training",
            "artist_id": ids,
            "in_triplets": [item_id in constrained for item_id in ids],
        }
    )


def _write(directory: str, index: pd.DataFrame, constraints: pd.DataFrame | None) -> None:
    path = paths.dataset_index(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(path, index=False)
    if constraints is not None:
        constraints.to_csv(paths.dataset_dir(directory) / "triplet_constraints.csv", index=False)


#: Every MTT clip, the unconstrained ones first, so index order is not id order.
ALL_IDS = [*UNCONSTRAINED_IDS, *range(FIRST_ID + N_CLIPS - 1, FIRST_ID - 1, -1)]


@pytest.fixture
def mtt(isolated_roots) -> pd.DataFrame:
    index = _mtt_index(ALL_IDS)
    _write("mtt", index, _constraints())
    return index


@pytest.fixture
def toy(mtt, tmp_path, monkeypatch) -> pd.DataFrame:
    """A second corpus, with the MTT configuration file pointed at a test copy."""
    corpus = tmp_path / "mtt.yaml"
    corpus.write_text(
        json.dumps({"dataset": {"name": "mtt", "directory": "mtt"}, "folds": FOLDS}),
        encoding="utf-8",
    )
    monkeypatch.setattr(human_siamese, "MTT_CONFIG", corpus)
    index = pd.DataFrame(
        {
            "item_id": TOY_IDS,
            "path": [f"unused/{item_id}.mp3" for item_id in TOY_IDS],
            "split": TOY_SPLITS,
            "artist_id": TOY_IDS,
        }
    )
    _write("toy", index, None)
    return index


def _standardised_input(config: Config, index: pd.DataFrame, is_fit: np.ndarray) -> np.ndarray:
    """The random input of ``index``'s items, standardised on the ``is_fit`` rows."""
    spec = RepresentationSpec("random", "random", {"extractor.dim": D_IN})
    cache = extract_stage(spec_config(config, spec)).cache_dir
    features = pd.DataFrame(np.load(cache / "features.npy"), index=np.load(cache / "ids.npy"))
    return standardize_on_training(features.loc[index["item_id"]].to_numpy(), is_fit)


def _head(directory, *, hidden=(HIDDEN,)) -> MlpHead:
    model = MlpHead(D_IN, hidden, DIM, dropout=0.0, generator=torch.Generator().manual_seed(0))
    model.load_state_dict(torch.load(directory / BEST_CHECKPOINT, weights_only=True))
    return model.eval()


def _json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class _OneHot(nn.Module):
    """Maps every input to the one-hot vector of its fold."""

    def __init__(self, fold: int) -> None:
        super().__init__()
        self.register_buffer("vector", torch.eye(DIM)[fold])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vector.expand(len(x), -1).clone()


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x, dim=1)


def _fake_folds(monkeypatch, make_model) -> list[int]:
    """Replace the training of a fold by ``make_model(test_fold)``; returns the folds asked."""
    asked: list[int] = []

    def fake(features, ids, constraints, folds, *, test_fold, settings, seed, out_dir):
        asked.append(test_fold)
        selection = pd.DataFrame(
            {"point": [point.label for point in settings.grid], VAL_AGREEMENT: 0.5}
        )
        return FoldResult(
            test_fold=test_fold,
            chosen=settings.grid[0],
            selection=selection,
            model=make_model(test_fold),
            n_parameters=0,
            training_rows=np.array([]),
            inner_rows=np.array([]),
        )

    monkeypatch.setattr(training, "train_fold", fake)
    return asked


def _constrained_in_index_order() -> list[int]:
    return [item_id for item_id in ALL_IDS if item_id not in UNCONSTRAINED_IDS]


# --- registration and configuration -------------------------------------------------
def test_is_registered_as_human_siamese() -> None:
    load_plugins()
    assert EXTRACTORS.get("human_siamese") is HumanSiameseExtractor


def test_extract_one_refuses_to_embed_a_single_item() -> None:
    with pytest.raises(NotImplementedError, match="extract_all"):
        HumanSiameseExtractor(_config()).extract_one(1, "unused.mp3")


def test_selection_grid_varies_the_margin_slowest() -> None:
    grid = selection_grid({"margin": [0.1, 0.2], "lr": [1e-3, 1e-4], "weight_decay": [0]})

    assert grid == (
        GridPoint(0.1, 1e-3, 0.0),
        GridPoint(0.1, 1e-4, 0.0),
        GridPoint(0.2, 1e-3, 0.0),
        GridPoint(0.2, 1e-4, 0.0),
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"margin": [0.1], "lr": [1e-3]},
        {"margin": [0.1], "lr": [1e-3], "weight_decay": [0.0], "batch_size": [4]},
        {"margin": [], "lr": [1e-3], "weight_decay": [0.0]},
        {"margin": "0.1", "lr": [1e-3], "weight_decay": [0.0]},
        [0.1, 1e-3, 0.0],
    ],
)
def test_selection_grid_rejects_a_malformed_grid(raw) -> None:
    with pytest.raises(ConfigError, match=r"extractor\.training\.grid"):
        selection_grid(raw)


def test_the_fold_settings_must_repeat_the_corpus_ones(mtt) -> None:
    config = _config(folds={"n_folds": 4, "max_imbalance": 0.1})
    with pytest.raises(ConfigError, match="must repeat the folds section"):
        HumanSiameseExtractor(config).extract_all(mtt)


def test_the_fold_settings_are_required(mtt) -> None:
    data = _config().to_dict()
    del data["extractor"]["folds"]
    with pytest.raises(ConfigError, match="must repeat the folds section"):
        HumanSiameseExtractor(Config(data=data)).extract_all(mtt)


def test_checkpoints_depend_on_the_seed_and_the_variant() -> None:
    def directory(config: Config):
        extractor = HumanSiameseExtractor(config)
        return extractor.checkpoint_dir(extractor.mtt_config())

    assert directory(_config()) == directory(_config())
    assert directory(_config(seed=SEED + 1)) != directory(_config())
    assert directory(_config(hidden=())) != directory(_config())
    assert directory(_config()).name.startswith("human_siamese-")


# --- the final point ----------------------------------------------------------------
def _selection(values: list[float], chosen: int) -> pd.DataFrame:
    points = [GridPoint(0.1, 1e-3, 0.0), GridPoint(0.2, 1e-3, 0.0), GridPoint(0.4, 1e-3, 0.0)]
    points = points[: len(values)]
    return pd.DataFrame(
        {
            "point": [point.label for point in points],
            "margin": [point.margin for point in points],
            "lr": [point.lr for point in points],
            "weight_decay": [point.weight_decay for point in points],
            VAL_AGREEMENT: values,
            "chosen": [position == chosen for position in range(len(values))],
        }
    )


def test_final_selection_takes_the_best_mean_over_the_folds() -> None:
    # Means 0.625, 0.5625 and 0.4375: the first point wins although each fold
    # chose another one once.
    table = final_selection([_selection([0.5, 0.75, 0.25], 1), _selection([0.75, 0.375, 0.625], 0)])

    assert table[f"mean_{VAL_AGREEMENT}"].tolist() == [0.625, 0.5625, 0.4375]
    assert table["n_chosen"].tolist() == [1, 1, 0]
    assert table["chosen"].tolist() == [True, False, False]
    assert table["point"].tolist() == _selection([0, 0, 0], 0)["point"].tolist()


def test_final_selection_breaks_ties_by_grid_order() -> None:
    table = final_selection([_selection([0.5, 0.75], 1), _selection([0.75, 0.5], 0)])

    assert table[f"mean_{VAL_AGREEMENT}"].tolist() == [0.625, 0.625]
    assert table["chosen"].tolist() == [True, False]


def test_final_selection_needs_folds_with_the_same_points() -> None:
    with pytest.raises(ValueError, match="no fold selection"):
        final_selection([])
    with pytest.raises(ValueError, match="lists the points"):
        final_selection([_selection([0.5, 0.75], 0), _selection([0.5, 0.75, 0.25], 0)])


# --- out of fold, with hand-made models -----------------------------------------
def test_every_clip_is_embedded_by_the_model_of_its_test_fold(mtt, monkeypatch) -> None:
    asked = _fake_folds(monkeypatch, _OneHot)
    config = _config()

    X, ids, failures = HumanSiameseExtractor(config).extract_all(mtt)

    assert asked == list(range(N_FOLDS))
    assert ids.tolist() == _constrained_in_index_order()
    assert failures == []
    extractor = HumanSiameseExtractor(config)
    folds = pd.read_csv(extractor.checkpoint_dir(config) / FOLDS_FILE)
    long = folds.melt(id_vars="fold", value_vars=["anchor", "pos", "neg"], value_name="clip")
    fold_of_clip = long.drop_duplicates(["clip", "fold"]).set_index("clip")["fold"]
    assert fold_of_clip.index.is_unique
    np.testing.assert_array_equal(X, np.eye(DIM, dtype=np.float32)[fold_of_clip.loc[ids]])
    assert np.bincount(fold_of_clip.to_numpy()).tolist() == [CLIPS_PER_FOLD] * N_FOLDS


def test_with_identity_models_the_agreement_is_that_of_the_input(mtt, monkeypatch) -> None:
    _fake_folds(monkeypatch, lambda fold: _Identity())
    config = _config()

    X, ids, _ = HumanSiameseExtractor(config).extract_all(mtt)

    # The input is standardised on every MTT clip, the unconstrained ones included.
    standardised = _standardised_input(config, mtt, np.ones(len(mtt), dtype=bool))
    expected = constraint_hits(standardised, mtt["item_id"].to_numpy(), _constraints())
    np.testing.assert_array_equal(constraint_hits(X, ids, _constraints()), expected)


def test_rows_follow_the_index_by_id(mtt, monkeypatch) -> None:
    _fake_folds(monkeypatch, lambda fold: _Identity())
    config = _config()
    X, ids, _ = HumanSiameseExtractor(config).extract_all(mtt)

    reordered = mtt.iloc[::-1].reset_index(drop=True)
    _write("mtt", reordered, _constraints())
    X_reordered, ids_reordered, _ = HumanSiameseExtractor(config).extract_all(reordered)

    assert ids_reordered.tolist() == ids[::-1].tolist()
    np.testing.assert_allclose(X_reordered, X[::-1], atol=1e-6)


def test_a_constrained_clip_without_input_is_an_error(isolated_roots) -> None:
    index = _mtt_index([item_id for item_id in ALL_IDS if item_id != FIRST_ID])
    _write("mtt", index, _constraints())
    with pytest.raises(ValueError, match="1 clip\\(s\\) of the constraints have no input"):
        HumanSiameseExtractor(_config()).extract_all(index)


def test_missing_constraints_are_an_error(isolated_roots) -> None:
    index = _mtt_index(ALL_IDS)
    _write("mtt", index, None)
    with pytest.raises(FileNotFoundError, match=r"triplet_constraints\.csv"):
        HumanSiameseExtractor(_config()).extract_all(index)


# --- out of fold, trained for real ----------------------------------------------
def test_extract_stage_caches_the_out_of_fold_vectors(mtt) -> None:
    config = _config()
    result = extract_stage(config)

    X = np.load(result.cache_dir / "features.npy")
    ids = np.load(result.cache_dir / "ids.npy")
    assert X.shape == (N_CLIPS, DIM)
    assert X.dtype == np.float32
    assert ids.tolist() == _constrained_in_index_order()
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)
    assert result.n_failed == 0


def test_trained_vectors_come_from_the_saved_model_of_each_fold(mtt) -> None:
    config = _config()
    X, ids, _ = HumanSiameseExtractor(config).extract_all(mtt)

    out_dir = HumanSiameseExtractor(config).checkpoint_dir(config)
    folds = pd.read_csv(out_dir / FOLDS_FILE)
    standardised = pd.DataFrame(
        _standardised_input(config, mtt, np.ones(len(mtt), dtype=bool)), index=mtt["item_id"]
    )
    for fold in range(N_FOLDS):
        clips = np.unique(folds.loc[folds["fold"] == fold, ["anchor", "pos", "neg"]])
        rows = np.flatnonzero(np.isin(ids, clips))
        assert len(rows) == CLIPS_PER_FOLD
        with torch.no_grad():
            expected = _head(out_dir / f"fold_{fold}")(
                torch.from_numpy(standardised.loc[ids[rows]].to_numpy())
            ).numpy()
        np.testing.assert_allclose(X[rows], expected, atol=1e-6)


def test_the_cross_validation_files_sit_next_to_the_folds(mtt) -> None:
    config = _config()
    HumanSiameseExtractor(config).extract_all(mtt)
    out_dir = HumanSiameseExtractor(config).checkpoint_dir(config)
    summary = _json(out_dir / CROSS_VALIDATION_SUMMARY)

    assert summary["n_folds"] == N_FOLDS
    assert summary["n_constraints"] == N_CONSTRAINTS
    assert summary["n_clips"] == N_CLIPS
    assert summary["n_parameters"] == N_PARAMETERS
    assert summary["seed"] == SEED
    assert summary["input_features"].startswith("features-random-")
    assert [fold["test_fold"] for fold in summary["folds"]] == list(range(N_FOLDS))
    for fold in summary["folds"]:
        fold_summary = _json(out_dir / f"fold_{fold['test_fold']}" / FOLD_SUMMARY)
        assert fold["chosen"] == fold_summary["chosen"]["point"]
        assert fold["n_parameters"] == N_PARAMETERS
        assert fold["n_test_constraints"] == CONSTRAINTS_PER_FOLD
        assert fold["n_test_clips"] == CLIPS_PER_FOLD
    folds = pd.read_csv(out_dir / FOLDS_FILE)
    assert len(folds) == N_CONSTRAINTS
    assert folds["fold"].value_counts().sort_index().tolist() == [CONSTRAINTS_PER_FOLD] * 3
    assert (out_dir / RESOLVED_CONFIG).read_text(encoding="utf-8") == config.to_yaml()


def test_the_linear_variant_has_one_map(mtt) -> None:
    config = _config(hidden=())
    X, _, _ = HumanSiameseExtractor(config).extract_all(mtt)

    out_dir = HumanSiameseExtractor(config).checkpoint_dir(config)
    assert _json(out_dir / CROSS_VALIDATION_SUMMARY)["n_parameters"] == N_LINEAR_PARAMETERS
    assert X.shape == (N_CLIPS, DIM)


def test_a_second_extraction_reloads_every_model(mtt, monkeypatch) -> None:
    config = _config()
    first, _, _ = HumanSiameseExtractor(config).extract_all(mtt)

    def no_training(*args, **kwargs):
        raise AssertionError("an epoch was trained again")

    monkeypatch.setattr(train_module, "_train_epoch", no_training)
    second, _, _ = HumanSiameseExtractor(config).extract_all(mtt)
    np.testing.assert_array_equal(second, first)


# --- the final model, on another corpus -------------------------------------------
def test_another_corpus_gets_the_final_model_on_its_training_statistics(toy) -> None:
    config = _config(directory="toy")
    extractor = HumanSiameseExtractor(config)

    X, ids, failures = extractor.extract_all(toy)

    assert ids.tolist() == TOY_IDS
    assert failures == []
    final_dir = extractor.checkpoint_dir(extractor.mtt_config()) / FINAL_DIR
    standardised = _standardised_input(config, toy, np.array(TOY_SPLITS) == "training")
    with torch.no_grad():
        expected = _head(final_dir)(torch.from_numpy(standardised)).numpy()
    np.testing.assert_allclose(X, expected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)


def test_the_final_point_is_the_best_mean_over_the_folds(toy) -> None:
    config = _config(directory="toy")
    extractor = HumanSiameseExtractor(config)
    extractor.extract_all(toy)

    out_dir = extractor.checkpoint_dir(extractor.mtt_config())
    selections = [pd.read_csv(out_dir / f"fold_{fold}" / SELECTION_FILE) for fold in range(3)]
    means = np.mean([table[VAL_AGREEMENT].to_numpy() for table in selections], axis=0)
    summary = _json(out_dir / FINAL_DIR / FINAL_SUMMARY)

    assert summary["chosen"]["point"] == selections[0]["point"].iloc[int(np.argmax(means))]
    assert summary["n_parameters"] == N_PARAMETERS
    assert summary["held_out_fold"] == 0
    assert summary["n_training_constraints"] == N_CONSTRAINTS - CONSTRAINTS_PER_FOLD
    assert summary["n_held_out_constraints"] == CONSTRAINTS_PER_FOLD
    saved = pd.read_csv(out_dir / FINAL_DIR / SELECTION_FILE)
    np.testing.assert_allclose(saved[f"mean_{VAL_AGREEMENT}"], means)


def test_the_final_model_shares_the_folds_of_the_mtt_run(toy) -> None:
    on_toy = HumanSiameseExtractor(_config(directory="toy"))
    on_mtt = HumanSiameseExtractor(_config())

    assert on_toy.checkpoint_dir(on_toy.mtt_config()) == on_mtt.checkpoint_dir(_config())


def test_another_corpus_needs_a_split_column(toy) -> None:
    with pytest.raises(ValueError, match="'split' column"):
        HumanSiameseExtractor(_config(directory="toy")).extract_all(toy.drop(columns="split"))
