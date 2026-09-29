"""Regression checks for the retrieval galleries against real FMA data: the
archived pipeline's figures on ``fma_all`` (B.3) and the new ``fma_test``
headline references this closes the baseline with. Needs the real FMA index
and downloaded audio (``musicsim download``/``index --dataset fma_small``);
skipped otherwise, and marked ``slow`` like the other real-data checks
(``pytest -m "not slow"`` deselects it).
"""

from __future__ import annotations

import pandas as pd
import pytest

from musicsim import paths
from musicsim.config import load_config
from musicsim.experiments import run_experiment

pytestmark = pytest.mark.slow

_E01 = paths.REPO_ROOT / "configs" / "experiments" / "e01_mfcc_baseline.yaml"
_E02 = paths.REPO_ROOT / "configs" / "experiments" / "e02_pca_ablation.yaml"


def _skip_unless_fma_ready() -> None:
    if not paths.dataset_index("fma").is_file():
        pytest.skip("fma index not built; run `musicsim index --dataset fma_small` first")


def _p_at_10_dedup(retrieval: pd.DataFrame, *, gallery: str, queries: str) -> float:
    row = retrieval[
        (retrieval["representation"] == "mfcc")
        & (retrieval["gallery"] == gallery)
        & (retrieval["queries"] == queries)
        & (retrieval["variant"] == "dedup")
        & (retrieval["metric"] == "P")
        & (retrieval["k"] == 10)
    ]
    assert len(row) == 1, f"expected exactly one P@10 dedup row for gallery={gallery!r}"
    return float(row["value"].iloc[0])


@pytest.fixture(scope="module")
def e01_retrieval() -> pd.DataFrame:
    """``retrieval.csv`` of a real, unmodified run of e01 (both galleries)."""
    _skip_unless_fma_ready()
    run = run_experiment(load_config(_E01))
    return pd.read_csv(run.metrics_dir / "retrieval.csv")


@pytest.fixture(scope="module")
def baseline_p10_fma_all(e01_retrieval: pd.DataFrame) -> float:
    """e01's un-reduced (no PCA) P@10 dedup on ``fma_all``, the ablation's reference point."""
    return _p_at_10_dedup(e01_retrieval, gallery="fma_all", queries="all")


def test_fma_all_p10_dedup_matches_the_archived_pipeline(baseline_p10_fma_all: float) -> None:
    # B.3: P@10 dedup 0.327 [0.321; 0.333] on the archived pipeline's mfcc.npy.
    assert baseline_p10_fma_all == pytest.approx(0.327, abs=0.002)


def test_fma_test_gallery_reports_the_new_headline_references(
    e01_retrieval: pd.DataFrame,
) -> None:
    """R@10, MAP and nDCG@10 of mfcc and random on fma_test, this phase's new references."""
    mask = (
        (e01_retrieval["gallery"] == "fma_test")
        & (e01_retrieval["queries"] == "all")
        & (e01_retrieval["variant"] == "dedup")
        & (e01_retrieval["k"].isin([0, 10]))
    )
    value = e01_retrieval.loc[mask].set_index(["representation", "metric"])["value"]

    assert value[("mfcc", "R")] == pytest.approx(0.0350, abs=0.0015)
    assert value[("mfcc", "MAP")] == pytest.approx(0.2398, abs=0.005)
    assert value[("mfcc", "nDCG")] == pytest.approx(0.3630, abs=0.005)
    assert value[("random", "R")] == pytest.approx(0.0130, abs=0.0015)
    assert value[("random", "MAP")] == pytest.approx(0.1316, abs=0.005)
    assert value[("random", "nDCG")] == pytest.approx(0.1288, abs=0.005)

    # mfcc must beat the random control on every headline metric.
    for metric in ("R", "MAP", "nDCG"):
        assert value[("mfcc", metric)] > value[("random", metric)]


def test_queries_test_over_fma_all_reproduces_the_archived_regression() -> None:
    """`queries=test` over the full gallery: a check, not a headline report row."""
    _skip_unless_fma_ready()
    config = load_config(
        _E01,
        overrides=[
            "evaluation.retrieval.galleries=[fma_all]",
            "evaluation.retrieval.queries=[test]",
        ],
    )
    run = run_experiment(config)
    retrieval = pd.read_csv(run.metrics_dir / "retrieval.csv")
    value = _p_at_10_dedup(retrieval, gallery="fma_all", queries="test")
    # B.3: queries=test over the full gallery reproduces P@10 dedup 0.306.
    assert value == pytest.approx(0.306, abs=0.003)


@pytest.mark.parametrize(
    ("n_components", "expected_delta"),
    [(16, -0.026), (32, -0.012), (48, -0.003)],
)
def test_pca_ablation_matches_the_archived_delta(
    n_components: int, expected_delta: float, baseline_p10_fma_all: float
) -> None:
    """e02: ΔP@10 dedup on fma_all against the un-reduced mfcc embedding (B.3)."""
    config = load_config(_E02, overrides=[f"embedding.pca.n_components={n_components}"])
    run = run_experiment(config)
    retrieval = pd.read_csv(run.metrics_dir / "retrieval.csv")
    value = _p_at_10_dedup(retrieval, gallery="fma_all", queries="all")
    assert value - baseline_p10_fma_all == pytest.approx(expected_delta, abs=0.003)
