"""Tests for the structure figures: the plotted numbers, not just that something draws."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

from musicsim.plotting import (
    plot_community_genre_matrix,
    plot_community_sizes,
    plot_k_occurrence,
    save_figure,
)


def test_k_occurrence_plots_the_complementary_distribution() -> None:
    # N_k = [0, 1, 1, 2, 4]: P(N_k >= 1) = 4/5, P(N_k >= 2) = 2/5, P(N_k >= 4) = 1/5.
    figure = plot_k_occurrence({"a": np.array([0, 1, 1, 2, 4]), "b": np.array([1, 1, 1, 1, 1])}, 5)
    (ax,) = figure.axes
    *lines, mean_marker = ax.lines  # one line per label, then the vertical line at k
    assert len(lines) == 2
    np.testing.assert_array_equal(lines[0].get_xdata(), [1, 2, 4])
    np.testing.assert_allclose(lines[0].get_ydata(), [0.8, 0.4, 0.2])
    np.testing.assert_allclose(lines[1].get_ydata(), [1.0])
    assert list(mean_marker.get_xdata()) == [5, 5]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]
    assert ax.get_xscale() == ax.get_yscale() == "log"
    assert ax.get_xlabel() and ax.get_ylabel()


def test_community_sizes_are_sorted_decreasing() -> None:
    figure = plot_community_sizes({"a": np.array([2, 7, 1]), "b": np.array([5])})
    (ax,) = figure.axes
    assert len(ax.lines) == 2
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 3])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [7, 2, 1])
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), [5])
    assert ax.get_xlabel() and ax.get_ylabel()


def test_genre_matrix_rows_are_shares_ordered_by_community_size() -> None:
    # Community 5 has 3 nodes (genres 0, 0, 1); community 2 has 1 node (genre 1).
    figure = plot_community_genre_matrix(
        np.array([2, 5, 5, 5]), np.array([1, 0, 0, 1]), ["rock", "jazz", "pop"]
    )
    image, colorbar = figure.axes[0].images[0], figure.axes[1]
    np.testing.assert_allclose(image.get_array(), [[2 / 3, 1 / 3, 0.0], [0.0, 1.0, 0.0]])
    assert len(figure.axes) == 2  # the heat map and its colour bar
    assert colorbar.get_ylabel()
    assert [t.get_text() for t in figure.axes[0].get_xticklabels()] == ["rock", "jazz", "pop"]
    assert figure.axes[0].get_xlabel() and figure.axes[0].get_ylabel()


def test_genre_matrix_rejects_inconsistent_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        plot_community_genre_matrix(np.array([0, 1]), np.array([0]), ["a"])
    with pytest.raises(ValueError, match="genre name"):
        plot_community_genre_matrix(np.array([0, 1]), np.array([0, 2]), ["a", "b"])


def test_empty_mappings_are_rejected() -> None:
    with pytest.raises(ValueError, match="n_k_by_label is empty"):
        plot_k_occurrence({}, 10)
    with pytest.raises(ValueError, match="sizes_by_label is empty"):
        plot_community_sizes({})


def test_save_figure_writes_png_and_pdf(tmp_path: Path) -> None:
    figure = plot_community_sizes({"a": np.array([3, 1])})
    written = save_figure(figure, tmp_path / "figures" / "sizes_fma.test")
    assert [p.name for p in written] == ["sizes_fma.test.png", "sizes_fma.test.pdf"]
    assert all(p.is_file() and p.stat().st_size > 0 for p in written)
    assert written[0].read_bytes()[:4] == b"\x89PNG" and written[1].read_bytes()[:4] == b"%PDF"


def test_plotting_leaves_global_state_alone() -> None:
    before = dict(matplotlib.rcParams)
    plot_k_occurrence({"a": np.array([1, 2])}, 1)
    assert dict(matplotlib.rcParams) == before
