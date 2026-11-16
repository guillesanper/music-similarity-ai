"""Tests for the networks of the learned models: MLP head, CNN backbone and classifier.

Parameter counts are worked out by hand in each test. The initialisation is
checked against PyTorch's own default one, built from the global random state
inside :func:`torch.random.fork_rng` so that state is left as it was. Skipped as
a whole without the optional deep-learning extra.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402 (after the torch skip)

from musicsim.seeding import derive_seed  # noqa: E402
from musicsim.training.models import (  # noqa: E402
    EMBEDDING_DIM,
    CnnBackbone,
    CnnClassifier,
    MlpHead,
    count_parameters,
)

SEED = 7

#: Width of a MERT layer and of the acoustic descriptors, the two MLP inputs.
MERT_DIM = 768
DESC_DIM = 141

#: Bands of the cached log-mel spectrograms and frames of a 3 s crop.
N_MELS = 128
CROP_FRAMES = 130

#: FMA small's genres.
N_GENRES = 8


def _seed(stream: str) -> int:
    return derive_seed(SEED, "test-models", stream)


def _gen(stream: str) -> torch.Generator:
    return torch.Generator().manual_seed(_seed(stream))


def _randn(*shape: int, stream: str) -> torch.Tensor:
    return torch.randn(*shape, generator=_gen(stream))


def _assert_same_weights(first: nn.Module, second: nn.Module) -> None:
    first_state, second_state = first.state_dict(), second.state_dict()
    assert list(first_state) == list(second_state)
    for name, tensor in first_state.items():
        assert torch.equal(tensor, second_state[name]), name


# --- MlpHead -----------------------------------------------------------------
@pytest.mark.parametrize(("d_in", "hidden"), [(MERT_DIM, (512,)), (DESC_DIM, (256,))])
def test_mlp_head_output_is_unit_norm_embedding(d_in, hidden):
    head = MlpHead(d_in, hidden, generator=_gen("mlp")).eval()
    out = head(_randn(5, d_in, stream="mlp-input"))
    assert out.shape == (5, EMBEDDING_DIM)
    torch.testing.assert_close(out.norm(dim=1), torch.ones(5), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    ("d_in", "hidden", "expected"),
    [
        # Linear without bias + BatchNorm (scale and shift) + final Linear with bias.
        (MERT_DIM, (512,), 768 * 512 + 2 * 512 + 512 * 128 + 128),  # 459 904
        (DESC_DIM, (256,), 141 * 256 + 2 * 256 + 256 * 128 + 128),  # 69 504
        # Two hidden layers: 10 -> 6 -> 5 -> 128.
        (10, (6, 5), 10 * 6 + 2 * 6 + 6 * 5 + 2 * 5 + 5 * 128 + 128),  # 880
        # No hidden layer: a single Linear with bias.
        (10, (), 10 * 128 + 128),  # 1408
    ],
)
def test_mlp_head_parameter_count_by_hand(d_in, hidden, expected):
    assert count_parameters(MlpHead(d_in, hidden, generator=_gen("count"))) == expected


def test_mlp_head_layer_sequence():
    head = MlpHead(10, (6,), 4, dropout=0.3, generator=_gen("layers"))
    kinds = [type(layer) for layer in head.layers]
    assert kinds == [nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout, nn.Linear]
    assert head.layers[0].bias is None
    assert head.layers[3].p == 0.3
    assert head.layers[4].bias is not None


def test_mlp_head_without_hidden_layers_is_a_normalised_linear_map():
    head = MlpHead(3, (), 2, generator=_gen("linear"))
    x = _randn(4, 3, stream="linear-input")
    linear = head.layers[0]
    expected = nn.functional.normalize(x @ linear.weight.T + linear.bias, dim=1)
    torch.testing.assert_close(head(x), expected)


def test_count_parameters_skips_frozen_ones():
    head = MlpHead(MERT_DIM, (512,), generator=_gen("frozen"))
    head.layers[-1].requires_grad_(False)
    assert count_parameters(head) == 768 * 512 + 2 * 512  # 394 240


# --- CNN ---------------------------------------------------------------------
def test_cnn_backbone_and_classifier_shapes():
    crops = _randn(2, N_MELS, CROP_FRAMES, stream="cnn-input")
    backbone = CnnBackbone(generator=_gen("cnn")).eval()
    classifier = CnnClassifier(N_GENRES, generator=_gen("cnn")).eval()
    assert backbone(crops).shape == (2, EMBEDDING_DIM)
    assert classifier(crops).shape == (2, N_GENRES)


def test_cnn_backbone_takes_any_input_length():
    """The global average pooling makes a whole clip embeddable as well as a crop."""
    backbone = CnnBackbone(generator=_gen("length")).eval()
    clip = _randn(1, N_MELS, 1292, stream="length-input")
    assert backbone(clip).shape == (1, EMBEDDING_DIM)
    assert backbone(clip[:, :, :32]).shape == (1, EMBEDDING_DIM)


def test_cnn_classifier_logits_are_the_head_over_the_backbone():
    classifier = CnnClassifier(N_GENRES, generator=_gen("logits")).eval()
    crops = _randn(3, N_MELS, CROP_FRAMES, stream="logits-input")
    expected = classifier.backbone(crops) @ classifier.classifier.weight.T
    torch.testing.assert_close(classifier(crops), expected + classifier.classifier.bias)


def test_cnn_parameter_counts_by_hand():
    channels = (1, 32, 64, 128, 256, 256)
    convs = sum(9 * c_in * c_out for c_in, c_out in pairwise(channels))  # 977 184
    norms = sum(2 * c for c in channels[1:])  # 1472
    projection = 256 * 128 + 128  # 32 896
    backbone = convs + norms + projection
    assert backbone == 1_011_552
    assert count_parameters(CnnBackbone(generator=_gen("count"))) == backbone
    classifier = backbone + 128 * N_GENRES + N_GENRES  # 1 012 584
    assert count_parameters(CnnClassifier(N_GENRES, generator=_gen("count"))) == classifier


# --- seeded initialisation ---------------------------------------------------
@pytest.mark.parametrize(
    "build",
    [
        lambda gen: MlpHead(MERT_DIM, (512,), generator=gen),
        lambda gen: CnnBackbone(generator=gen),
        lambda gen: CnnClassifier(N_GENRES, generator=gen),
    ],
    ids=["mlp", "backbone", "classifier"],
)
def test_initialisation_is_determined_by_the_generator(build):
    _assert_same_weights(build(_gen("init")), build(_gen("init")))
    first_weight = next(build(_gen("init")).parameters())
    other_weight = next(build(_gen("init-other")).parameters())
    assert not torch.equal(first_weight, other_weight)


def test_mlp_initialisation_equals_pytorch_default():
    seed = _seed("default-mlp")
    # Reference: PyTorch's own default init from a seeded global state, isolated
    # by fork_rng so the global stream is restored afterwards.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = nn.Sequential(
            nn.Linear(MERT_DIM, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, EMBEDDING_DIM),
        )
    head = MlpHead(MERT_DIM, (512,), generator=torch.Generator().manual_seed(seed))
    _assert_same_weights(head.layers, reference)


def test_cnn_classifier_initialisation_equals_pytorch_default():
    seed = _seed("default-cnn")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layers: list[nn.Module] = []
        for c_in, c_out in pairwise((1, 32, 64, 128, 256, 256)):
            layers += [
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
        blocks = nn.Sequential(*layers)
        projection = nn.Linear(256, EMBEDDING_DIM)
        head = nn.Linear(EMBEDDING_DIM, N_GENRES)
    model = CnnClassifier(N_GENRES, generator=torch.Generator().manual_seed(seed))
    _assert_same_weights(model.backbone.blocks, blocks)
    _assert_same_weights(model.backbone.projection, projection)
    _assert_same_weights(model.classifier, head)


def test_building_a_model_leaves_the_global_random_state_alone():
    before = torch.random.get_rng_state()
    MlpHead(MERT_DIM, (512,), generator=_gen("global"))
    CnnClassifier(N_GENRES, generator=_gen("global"))
    assert torch.equal(torch.random.get_rng_state(), before)


# --- errors ------------------------------------------------------------------
@pytest.mark.parametrize(
    ("d_in", "hidden", "d_out"),
    [(0, (4,), 2), (3, (0,), 2), (3, (4,), -1), (3, (True,), 2), (3.0, (4,), 2)],
)
def test_mlp_head_rejects_bad_widths(d_in, hidden, d_out):
    with pytest.raises(ValueError, match="positive integers"):
        MlpHead(d_in, hidden, d_out, generator=_gen("bad"))


@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_mlp_head_rejects_bad_dropout(dropout):
    with pytest.raises(ValueError, match=r"dropout must lie in \[0, 1\)"):
        MlpHead(3, (4,), 2, dropout=dropout, generator=_gen("bad"))


@pytest.mark.parametrize(
    "build",
    [
        lambda gen: MlpHead(3, (4,), 2, generator=gen),
        lambda gen: CnnBackbone(generator=gen),
        lambda gen: CnnClassifier(N_GENRES, generator=gen),
    ],
    ids=["mlp", "backbone", "classifier"],
)
@pytest.mark.parametrize("generator", [None, SEED], ids=["none", "int"])
def test_models_reject_a_missing_generator(build, generator):
    with pytest.raises(ValueError, match=r"CPU torch\.Generator"):
        build(generator)


@pytest.mark.parametrize("shape", [(4, 5), (4,), (4, 3, 1)])
def test_mlp_head_rejects_bad_input_shape(shape):
    head = MlpHead(3, (4,), 2, generator=_gen("bad")).eval()
    with pytest.raises(ValueError, match=r"expected a \(batch, 3\) tensor"):
        head(torch.zeros(shape))


@pytest.mark.parametrize("shape", [(1, 1, N_MELS, CROP_FRAMES), (1, N_MELS, 31), (1, 31, 64)])
def test_cnn_backbone_rejects_bad_input_shape(shape):
    backbone = CnnBackbone(generator=_gen("bad")).eval()
    with pytest.raises(ValueError, match=r"n_frames >= 32"):
        backbone(torch.zeros(shape))


@pytest.mark.parametrize("n_classes", [1, True, 8.0])
def test_cnn_classifier_rejects_bad_class_count(n_classes):
    with pytest.raises(ValueError, match="n_classes must be an integer >= 2"):
        CnnClassifier(n_classes, generator=_gen("bad"))
