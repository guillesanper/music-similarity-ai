"""Figures for the graph-structure evaluation.

Every function builds a :class:`matplotlib.figure.Figure` through the object-oriented
API (``Figure`` plus an Agg canvas) and never touches ``pyplot``: there is no global
figure registry to leak from, no current-figure state to depend on, and no display
backend to choose, so the same code runs in a test, in a batch job on a server and
inside a notebook. The price is that nothing is shown interactively; the figures are
meant to be written to disk with :func:`save_figure`.

The look is defined once, in :data:`STYLE` and :data:`PALETTE`, and applied to each
figure with :func:`matplotlib.rc_context`, so the process-wide matplotlib settings
are left as found. The palette is the Okabe-Ito one, chosen because its colours
stay distinguishable under the common forms of colour blindness; lines also differ
in marker so a greyscale print still tells them apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

__all__ = [
    "PALETTE",
    "STYLE",
    "plot_community_genre_matrix",
    "plot_community_sizes",
    "plot_k_occurrence",
    "save_figure",
]

#: Okabe-Ito colours, in the order lines are assigned to representations.
PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)

_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")

#: The matplotlib settings every figure of this module is drawn with.
STYLE: dict[str, object] = {
    "figure.figsize": (6.4, 4.4),
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


def _new_figure() -> tuple[Figure, matplotlib.axes.Axes]:
    """A figure with one axes and an Agg canvas attached (call inside ``rc_context``)."""
    figure = Figure()
    FigureCanvasAgg(figure)
    return figure, figure.add_subplot(1, 1, 1)


def _style_of(position: int) -> dict[str, object]:
    """Colour and marker of the ``position``-th line; both wrap around when exhausted."""
    return {
        "color": PALETTE[position % len(PALETTE)],
        "marker": _MARKERS[position % len(_MARKERS)],
    }


def _require_series(series_by_label: Mapping[str, np.ndarray], name: str) -> None:
    if not series_by_label:
        raise ValueError(f"{name} is empty; expected at least one representation to draw")


def plot_k_occurrence(n_k_by_label: Mapping[str, np.ndarray], k: int) -> Figure:
    """Complementary distribution of the k-occurrence, log-log, one line per label.

    For each value ``x >= 1`` the line gives the fraction of nodes whose k-occurrence
    is at least ``x``. A heavy right tail is hubness. Nodes with k-occurrence 0
    (anti-hubs) cannot sit on a log axis and are left out of the plot, though they
    still count in the denominator, so the line starts at one minus their fraction.
    The dotted vertical line is the mean k-occurrence, which is ``k`` in any k-NN
    graph without padding.
    """
    _require_series(n_k_by_label, "n_k_by_label")
    with matplotlib.rc_context(STYLE):
        figure, ax = _new_figure()
        for position, (label, n_k) in enumerate(n_k_by_label.items()):
            n_k = np.asarray(n_k)
            values, counts = np.unique(n_k[n_k >= 1], return_counts=True)
            # counts of nodes with N_k >= value: reverse cumulative sum over sorted values.
            survival = counts[::-1].cumsum()[::-1] / len(n_k)
            ax.plot(
                values,
                survival,
                label=label,
                linestyle="-",
                linewidth=1.2,
                markersize=3,
                **_style_of(position),
            )
        ax.axvline(k, color="grey", linestyle=":", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("k-occurrence $N_k$")
        ax.set_ylabel("Fraction of nodes with k-occurrence $\\geq N_k$")
        ax.set_title(f"k-occurrence distribution (k = {k})")
        ax.legend()
        figure.tight_layout()
    return figure


def plot_community_sizes(sizes_by_label: Mapping[str, np.ndarray]) -> Figure:
    """Community sizes in decreasing order (rank-size plot), one line per label.

    The x axis is the rank of the community and the y axis its number of nodes, both
    logarithmic: a graph that fragments into many small communities falls steeply, a
    graph dominated by a giant community starts high and drops at once.
    """
    _require_series(sizes_by_label, "sizes_by_label")
    with matplotlib.rc_context(STYLE):
        figure, ax = _new_figure()
        for position, (label, sizes) in enumerate(sizes_by_label.items()):
            ordered = np.sort(np.asarray(sizes))[::-1]
            ax.plot(
                np.arange(1, len(ordered) + 1),
                ordered,
                label=label,
                linestyle="-",
                linewidth=1.2,
                markersize=3,
                **_style_of(position),
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Community rank (largest first)")
        ax.set_ylabel("Community size (nodes)")
        ax.set_title("Community sizes")
        ax.legend()
        figure.tight_layout()
    return figure


def plot_community_genre_matrix(
    communities: np.ndarray, genres: np.ndarray, genre_names: Sequence[str]
) -> Figure:
    """Heat map of the genre make-up of every community.

    ``communities`` and ``genres`` hold one integer per node (community id, genre
    code); ``genre_names[c]`` is the name of genre code ``c``. Each row is a
    community, ordered from the largest down (ties by id), each column a genre, and
    each cell the share of that community's nodes with that genre, so rows sum to 1.
    A graph whose communities follow the genres shows one dark cell per row.
    """
    communities = np.asarray(communities)
    genres = np.asarray(genres)
    if communities.shape != genres.shape or communities.ndim != 1:
        raise ValueError(
            f"communities {communities.shape} and genres {genres.shape} must be "
            "one-dimensional arrays of the same length"
        )
    if len(genres) and (genres.min() < 0 or genres.max() >= len(genre_names)):
        raise ValueError(
            f"genre codes span [{genres.min()}, {genres.max()}] but only "
            f"{len(genre_names)} genre name(s) were given"
        )
    community_ids, community_rows = np.unique(communities, return_inverse=True)
    counts = np.zeros((len(community_ids), len(genre_names)))
    np.add.at(counts, (community_rows, genres), 1.0)
    sizes = counts.sum(axis=1)
    order = np.lexsort((community_ids, -sizes))  # largest first, ties by community id
    shares = counts[order] / sizes[order, None]
    with matplotlib.rc_context(STYLE):
        figure, ax = _new_figure()
        ax.grid(False)
        image = ax.imshow(shares, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(
            np.arange(len(genre_names)), labels=list(genre_names), rotation=45, ha="right"
        )
        ax.set_xlabel("Genre")
        ax.set_ylabel("Community (largest first)")
        ax.set_title("Genre composition of the communities")
        colorbar = figure.colorbar(image, ax=ax)
        colorbar.set_label("Share of the community's nodes")
        figure.tight_layout()
    return figure


def save_figure(figure: Figure, path_without_suffix: Path) -> list[Path]:
    """Write ``figure`` as ``.png`` and ``.pdf`` next to each other; return both paths.

    The parent directory is created if missing. The suffixes are appended to the
    name as given, so a name that already contains a dot keeps it.
    """
    path_without_suffix = Path(path_without_suffix)
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".png", ".pdf"):
        target = path_without_suffix.parent / (path_without_suffix.name + suffix)
        figure.savefig(target)
        written.append(target)
    return written
