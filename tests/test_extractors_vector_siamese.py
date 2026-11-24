"""Tests for the siamese head over cached vectors (the ``siamese_emb`` extractor).

The input extractor is ``random`` (16 normal numbers per item, no audio), so the
whole path runs for real: input features through ``extract_stage``,
standardisation on training, training with early stopping, checkpoints, summary
and projection. The toy dataset has 64 training tracks (4 genres x 16, 4 tracks
per artist), so one PxK batch of 4 x 16 is the whole training split and an epoch
is one step, plus 24 validation and 8 test tracks. The training corpus
(``TRAINING_CORPUS``) is pointed at that toy dataset for every test; a second toy
corpus without genres stands for MagnaTagATune. Skipped as a whole without the
optional deep-learning extra.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from musicsim import paths  # noqa: E402 (after the torch skip)
from musicsim.config import Config  # noqa: E402
from musicsim.embeddings import features_cache_dir  # noqa: E402
from musicsim.experiments import extract_stage  # noqa: E402
from musicsim.extractors import vector_siamese  # noqa: E402
from musicsim.extractors.vector_siamese import (  # noqa: E402
    RESOLVED_CONFIG,
    TRAINING_SUMMARY,
    VectorContrastiveTask,
    VectorSiameseExtractor,
    collapse_diagnostics,
    standardize_on_training,
)
from musicsim.registry import EXTRACTORS, load_plugins  # noqa: E402
from musicsim.training import train as train_module  # noqa: E402
from musicsim.training.models import MlpHead  # noqa: E402
from musicsim.training.train import BEST_CHECKPOINT, CURVE_FILE  # noqa: E402

SEED = 7
DATASET = "toy"

#: The other corpus: 12 clips without genres, the first 6 in its training split.
OTHER_DATASET = "toy_other"
OTHER_IDS = list(range(501, 513))
OTHER_TRAINING = 6

#: Toy dataset: 4 genres; per genre 16 training tracks (4 artists x 4),
#: 6 validation tracks and 2 test tracks.
N_GENRES = 4
TRAINING_PER_GENRE = 16
TRACKS_PER_ARTIST = 4
VALIDATION_PER_GENRE = 6
TEST_PER_GENRE = 2
N_TRAINING = N_GENRES * TRAINING_PER_GENRE
N_VALIDATION = N_GENRES * VALIDATION_PER_GENRE
N_ITEMS = N_GENRES * (TRAINING_PER_GENRE + VALIDATION_PER_GENRE + TEST_PER_GENRE)

#: The head: 16 random inputs -> 32 hidden -> 8 dimensions.
D_IN = 16
HIDDEN = 32
DIM = 8
#: Trainable parameters, by hand: Linear(16->32, no bias) 512 + BatchNorm(32)
#: scale and shift 64 + Linear(32->8) 256 weights and 8 biases.
N_PARAMETERS = 16 * 32 + 2 * 32 + 32 * 8 + 8

#: Mean epoch loss the head must reach on its own training tracks.
OVERFIT_LOSS = 1e-2


# --- fixtures and helpers --------------------------------------------------------
@pytest.fixture(autouse=True)
def training_corpus(tmp_path, monkeypatch):
    """The head is trained on the toy dataset, as it is on FMA small for real."""
    path = tmp_path / "toy.yaml"
    path.write_text(f"dataset:\n  directory: {DATASET}\n", encoding="utf-8")
    monkeypatch.setattr(vector_siamese, "TRAINING_CORPUS", path)
    return path


def _other_config(*, fit_on: str | None = "all") -> Config:
    data = _config().to_dict()
    data["dataset"] = {"directory": OTHER_DATASET}
    if fit_on is not None:
        data["embedding"] = {"fit_on": fit_on}
    return Config(data=data)


def _other_index(*, with_split: bool = True) -> pd.DataFrame:
    index = pd.DataFrame(
        {
            "item_id": OTHER_IDS,
            "path": [f"unused/{item_id}.mp3" for item_id in OTHER_IDS],
            "split": ["training"] * OTHER_TRAINING + ["test"] * (len(OTHER_IDS) - OTHER_TRAINING),
            "artist_id": OTHER_IDS,
            "genre_top": "",
        }
    )
    if not with_split:
        index = index.drop(columns="split")
    path = paths.dataset_index(OTHER_DATASET)
    path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(path, index=False)
    return pd.read_csv(path)


def _expected_other(config: Config, fit_rows: np.ndarray) -> np.ndarray:
    """The trained head applied by hand to the other corpus, standardised on ``fit_rows``."""
    extractor = VectorSiameseExtractor(config)
    cache = extract_stage(extractor.input_config()).cache_dir
    assert np.load(cache / "ids.npy").tolist() == OTHER_IDS
    features = standardize_on_training(np.load(cache / "features.npy"), fit_rows)
    head = MlpHead(D_IN, [HIDDEN], DIM, dropout=0.0, generator=torch.Generator().manual_seed(0))
    head.load_state_dict(
        torch.load(extractor.checkpoint_dir() / BEST_CHECKPOINT, weights_only=True)
    )
    head.eval()
    with torch.no_grad():
        return head(torch.from_numpy(features)).numpy()


def _config(*, seed: int = SEED, input_dim: int = D_IN, **training: object) -> Config:
    settings = {
        "margin": 0.5,
        "lr": 1e-2,
        "weight_decay": 1e-4,
        "p": N_GENRES,
        "k": TRAINING_PER_GENRE,
        "max_epochs": 3,
        "patience": 10,
    }
    settings.update(training)
    return Config(
        data={
            "seed": seed,
            "dataset": {"directory": DATASET},
            "runtime": {"n_jobs": 1, "progress": False, "cache": True},
            "extractor": {
                "name": "siamese_emb",
                "input": {"extractor": "random", "overrides": {"dim": input_dim}},
                "hidden": [HIDDEN],
                "dropout": 0.0,
                "dim": DIM,
                "training": settings,
            },
        }
    )


def _toy_index() -> pd.DataFrame:
    rows = []
    item_id = 1
    for genre in range(N_GENRES):
        for split, n_tracks in (
            ("training", TRAINING_PER_GENRE),
            ("validation", VALIDATION_PER_GENRE),
            ("test", TEST_PER_GENRE),
        ):
            for track in range(n_tracks):
                # Training artists hold 4 tracks; every other track has its own artist.
                if split == "training":
                    artist = 1000 * genre + track // TRACKS_PER_ARTIST
                else:
                    artist = 100_000 + item_id
                rows.append(
                    {
                        "item_id": item_id,
                        "path": f"unused/{item_id}.mp3",
                        "split": split,
                        "artist_id": artist,
                        "genre_top": f"genre{genre}",
                    }
                )
                item_id += 1
    return pd.DataFrame(rows)


def _write_index(index: pd.DataFrame) -> pd.DataFrame:
    path = paths.dataset_index(DATASET)
    path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(path, index=False)
    return index


@pytest.fixture
def index(isolated_roots) -> pd.DataFrame:
    return _write_index(_toy_index())


def _extract(config: Config, index: pd.DataFrame):
    return VectorSiameseExtractor(config).extract_all(index)


def _summary(extractor: VectorSiameseExtractor) -> dict:
    return json.loads((extractor.checkpoint_dir() / TRAINING_SUMMARY).read_text(encoding="utf-8"))


# --- registration and pipeline ------------------------------------------------------
def test_is_registered_as_siamese_emb() -> None:
    load_plugins()
    assert EXTRACTORS.get("siamese_emb") is VectorSiameseExtractor


def test_extract_stage_caches_one_unit_vector_per_item(index) -> None:
    result = extract_stage(_config())

    X = np.load(result.cache_dir / "features.npy")
    ids = np.load(result.cache_dir / "ids.npy")
    assert X.shape == (N_ITEMS, DIM)
    assert ids.tolist() == index["item_id"].tolist()
    assert result.n_failed == 0


def test_extract_one_refuses_to_embed_a_single_item() -> None:
    with pytest.raises(NotImplementedError, match="extract_all"):
        VectorSiameseExtractor(_config()).extract_one(1, "unused.mp3")


# --- training ---------------------------------------------------------------------------
def test_overfits_the_training_tracks(index) -> None:
    config = _config(max_epochs=100, patience=100)
    _extract(config, index)

    curve = pd.read_csv(VectorSiameseExtractor(config).checkpoint_dir() / CURVE_FILE)
    assert len(curve) == 100
    # Starts well above the target, so reaching it is learning, not luck.
    assert curve["train_loss"].iloc[0] > 5 * OVERFIT_LOSS
    assert curve["train_loss"].iloc[-1] < OVERFIT_LOSS


def test_output_is_unit_norm_float32_in_index_order(index) -> None:
    X, ids, failures = _extract(_config(), index)

    assert X.dtype == np.float32
    assert X.shape == (N_ITEMS, DIM)
    assert ids.tolist() == index["item_id"].tolist()
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)
    assert failures == []


def test_summary_and_resolved_config_sit_next_to_the_checkpoint(index) -> None:
    config = _config()
    _extract(config, index)
    extractor = VectorSiameseExtractor(config)
    summary = _summary(extractor)
    curve = pd.read_csv(extractor.checkpoint_dir() / CURVE_FILE)

    assert summary["n_parameters"] == N_PARAMETERS
    assert summary["epochs"] == 3
    assert summary["stopped_early"] is False
    assert summary["best_recall_at_10"] == curve["val_recall_at_10"].max()
    assert summary["best_epoch"] == curve["val_recall_at_10"].idxmax() + 1
    assert summary["n_training_items"] == N_TRAINING
    assert summary["n_validation_items"] == N_VALIDATION
    assert summary["input_features"] == features_cache_dir(extractor.input_config()).name
    assert set(summary) >= {"validation_effective_rank", "validation_mean_cosine", "collapsed"}
    resolved = (extractor.checkpoint_dir() / RESOLVED_CONFIG).read_text(encoding="utf-8")
    assert resolved == config.to_yaml()


def test_a_trained_head_is_loaded_not_retrained(index, monkeypatch) -> None:
    config = _config()
    first, _, _ = _extract(config, index)

    def no_training(*args, **kwargs):
        raise AssertionError("the head was trained again")

    monkeypatch.setattr(train_module, "train", no_training)
    second, _, _ = _extract(config, index)
    np.testing.assert_array_equal(second, first)


def test_an_interrupted_run_is_resumed(index, monkeypatch) -> None:
    config = _config()
    first, _, _ = _extract(config, index)
    extractor = VectorSiameseExtractor(config)
    best_epoch = _summary(extractor)["best_epoch"]
    # Without the summary the directory looks like a run stopped after its last epoch.
    (extractor.checkpoint_dir() / TRAINING_SUMMARY).unlink()

    calls: list[bool] = []
    real_train = train_module.train

    def recording_train(*args, resume: bool, **kwargs):
        calls.append(resume)
        return real_train(*args, resume=resume, **kwargs)

    monkeypatch.setattr(train_module, "train", recording_train)
    second, _, _ = _extract(config, index)

    assert calls == [True]
    np.testing.assert_allclose(second, first, rtol=0, atol=1e-6)
    assert _summary(extractor)["best_epoch"] == best_epoch


def test_checkpoint_directory_follows_section_seed_and_input() -> None:
    reference = VectorSiameseExtractor(_config()).checkpoint_dir()

    assert reference.parent == paths.checkpoints_dir("x").parent
    assert re.fullmatch(r"siamese_emb-[0-9a-f]{8}", reference.name)
    assert VectorSiameseExtractor(_config()).checkpoint_dir() == reference
    assert VectorSiameseExtractor(_config(seed=SEED + 1)).checkpoint_dir() != reference
    assert VectorSiameseExtractor(_config(margin=0.25)).checkpoint_dir() != reference
    assert VectorSiameseExtractor(_config(input_dim=D_IN + 1)).checkpoint_dir() != reference


def test_input_config_applies_the_overrides() -> None:
    input_config = VectorSiameseExtractor(_config(input_dim=12)).input_config()

    assert input_config.require("extractor.name") == "random"
    assert input_config.require("extractor.dim") == 12
    assert input_config.require("seed") == SEED


# --- another corpus ---------------------------------------------------------------------
def test_training_config_is_this_one_on_the_training_corpus() -> None:
    config = _config()
    assert VectorSiameseExtractor(config).training_config() is config


def test_training_config_swaps_only_the_dataset_on_another_corpus() -> None:
    config = _other_config()
    training = VectorSiameseExtractor(config).training_config().to_dict()

    assert training["dataset"] == {"directory": DATASET}
    expected = config.to_dict()
    expected["dataset"] = {"directory": DATASET}
    assert training == expected


def test_the_checkpoint_is_the_same_on_every_corpus() -> None:
    assert (
        VectorSiameseExtractor(_other_config()).checkpoint_dir()
        == VectorSiameseExtractor(_config()).checkpoint_dir()
    )


def test_another_corpus_reuses_the_head_trained_on_the_training_corpus(index, monkeypatch) -> None:
    _extract(_config(), index)
    other = _other_index()

    def no_training(*args, **kwargs):
        raise AssertionError("the head was trained again")

    monkeypatch.setattr(train_module, "train", no_training)
    config = _other_config()
    X, ids, failures = _extract(config, other)

    assert ids.tolist() == OTHER_IDS
    assert failures == []
    np.testing.assert_allclose(
        X, _expected_other(config, np.ones(len(OTHER_IDS), dtype=bool)), rtol=0, atol=1e-6
    )


def test_another_corpus_trains_the_head_on_the_training_corpus(index, monkeypatch) -> None:
    config = _other_config()
    _extract(config, _other_index())
    extractor = VectorSiameseExtractor(config)

    summary = _summary(extractor)
    assert summary["n_training_items"] == N_TRAINING
    assert summary["n_validation_items"] == N_VALIDATION
    resolved = (extractor.checkpoint_dir() / RESOLVED_CONFIG).read_text(encoding="utf-8")
    assert resolved == extractor.training_config().to_yaml()

    def no_training(*args, **kwargs):
        raise AssertionError("the head was trained again")

    monkeypatch.setattr(train_module, "train", no_training)
    X, _, _ = _extract(_config(), index)
    assert X.shape == (N_ITEMS, DIM)


def test_another_corpus_is_standardised_on_its_fit_on_split(index) -> None:
    other = _other_index()
    by_all, _, _ = _extract(_other_config(fit_on="all"), other)
    by_training, _, _ = _extract(_other_config(fit_on=None), other)

    training_rows = np.arange(len(OTHER_IDS)) < OTHER_TRAINING
    np.testing.assert_allclose(
        by_training, _expected_other(_other_config(fit_on=None), training_rows), atol=1e-6
    )
    # The two standardisations differ, so the test above tells them apart.
    assert np.abs(by_training - by_all).max() > 1e-3


def test_another_corpus_needs_a_split_column_unless_fitted_on_all(index) -> None:
    other = _other_index(with_split=False)

    with pytest.raises(ValueError, match="no 'split' column"):
        _extract(_other_config(fit_on="training"), other)
    X, _, _ = _extract(_other_config(fit_on="all"), other)
    assert X.shape == (len(OTHER_IDS), DIM)


# --- input and index errors -----------------------------------------------------------
def test_items_without_input_features_are_failures(index) -> None:
    config = _config()
    cache = extract_stage(VectorSiameseExtractor(config).input_config()).cache_dir
    missing_id = int(index["item_id"].iloc[-1])
    cached_ids = np.load(cache / "ids.npy")
    keep = cached_ids != missing_id
    np.save(cache / "features.npy", np.load(cache / "features.npy")[keep])
    np.save(cache / "ids.npy", cached_ids[keep])

    X, ids, failures = _extract(config, index)

    assert failures == [(missing_id, "no 'random' features for this item")]
    assert missing_id not in ids
    assert X.shape == (N_ITEMS - 1, DIM)


def test_needs_validation_items(isolated_roots) -> None:
    toy = _toy_index()
    toy.loc[toy["split"] == "validation", "split"] = "test"
    index = _write_index(toy)

    with pytest.raises(ValueError, match="no 'validation' item"):
        _extract(_config(), index)


def test_needs_training_items(isolated_roots) -> None:
    toy = _toy_index()
    toy.loc[toy["split"] == "training", "split"] = "test"
    index = _write_index(toy)

    with pytest.raises(ValueError, match="no training row"):
        _extract(_config(), index)


def test_needs_the_artist_column(isolated_roots) -> None:
    index = _write_index(_toy_index().drop(columns="artist_id"))

    with pytest.raises(ValueError, match=r"missing \['artist_id'\]"):
        _extract(_config(), index)


# --- standardisation -----------------------------------------------------------------
def test_standardisation_uses_only_the_training_rows() -> None:
    features = np.array([[1.0, 5.0], [3.0, 5.0], [10.0, 7.0]])
    # Training rows: mean (2, 5), population std (1, 0); the constant column is
    # only centred.
    expected = np.array([[-1.0, 0.0], [1.0, 0.0], [8.0, 2.0]], dtype=np.float32)

    result = standardize_on_training(features, np.array([True, True, False]))

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, expected)


def test_standardisation_needs_a_training_row() -> None:
    with pytest.raises(ValueError, match="no training row"):
        standardize_on_training(np.ones((3, 2)), np.zeros(3, dtype=bool))


def test_standardisation_rejects_misaligned_rows() -> None:
    with pytest.raises(ValueError, match=r"\(3, 2\) and \(2,\)"):
        standardize_on_training(np.ones((3, 2)), np.ones(2, dtype=bool))


# --- collapse diagnostics ----------------------------------------------------------------
def test_identical_embeddings_are_reported_as_collapsed(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=vector_siamese.__name__):
        diagnostics = collapse_diagnostics(np.ones((5, 4)))

    assert diagnostics["effective_rank"] == pytest.approx(1.0)
    assert diagnostics["mean_cosine"] == pytest.approx(1.0)
    assert diagnostics["collapsed"] is True
    assert "looks collapsed" in caplog.text


def test_orthonormal_embeddings_are_not_collapsed(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=vector_siamese.__name__):
        diagnostics = collapse_diagnostics(np.eye(12))

    assert diagnostics["effective_rank"] == pytest.approx(12.0)
    assert diagnostics["mean_cosine"] == pytest.approx(0.0)
    assert diagnostics["collapsed"] is False
    assert caplog.text == ""


def test_an_effective_rank_of_exactly_ten_counts_as_collapsed() -> None:
    # The criterion is a rank *above* 10; ten orthonormal rows have rank 10.
    diagnostics = collapse_diagnostics(np.eye(10))

    assert diagnostics["effective_rank"] == pytest.approx(10.0)
    assert diagnostics["collapsed"] is True


# --- the training task -----------------------------------------------------------------
def _task(n_labels: int = N_TRAINING) -> VectorContrastiveTask:
    labels = np.repeat(np.arange(N_GENRES), TRAINING_PER_GENRE)[:n_labels]
    artists = np.arange(N_TRAINING) // TRACKS_PER_ARTIST
    rng = np.random.default_rng(0)
    return VectorContrastiveTask(
        rng.normal(size=(N_TRAINING, D_IN)),
        labels,
        artists[:n_labels],
        rng.normal(size=(N_VALIDATION, D_IN)),
        margin=0.5,
        p=N_GENRES,
        k=TRAINING_PER_GENRE,
    )


def test_task_draws_one_batch_holding_every_training_item() -> None:
    batches = list(_task().batches(np.random.default_rng(1)))

    assert len(batches) == 1
    assert sorted(batches[0].tolist()) == list(range(N_TRAINING))


def test_task_embeds_the_validation_items_in_order() -> None:
    task = _task()
    head = torch.nn.Linear(D_IN, DIM)

    embeddings = task.embed_validation(head)

    assert embeddings.shape == (N_VALIDATION, DIM)
    assert embeddings.dtype == np.float32
    np.testing.assert_allclose(
        embeddings,
        head(torch.from_numpy(task._validation[np.arange(N_VALIDATION)])).detach().numpy(),
        rtol=0,
        atol=1e-6,
    )


def test_task_rejects_labels_not_aligned_with_the_features() -> None:
    with pytest.raises(ValueError, match="aligned with the 64 training items"):
        _task(n_labels=N_TRAINING - 1)
