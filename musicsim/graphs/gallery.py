"""Which rows of a gallery are the nodes of a graph that gets analysed, and where it is written.

A graph that is analysed is always built from the rows of its own gallery. It is
never built over every item and cropped afterwards: cropping leaves nodes with
fewer than ``k`` neighbours and changes the k-occurrence of those that remain,
so every structural figure would describe a graph nobody built.

Near-duplicates are detected inside the gallery and on the ``reference``
representation, the same rule the retrieval evaluator applies with its main
representation. By default one item per duplicate group is kept, the one with
the smallest ``item_id``: left in, each duplicate pair is a trivial reciprocal
edge that inflates reciprocity and clustering. The similarity is a plain dot
product, as in :func:`musicsim.graphs.duplicates.near_duplicate_pairs`, so it is
a cosine only if ``reference`` is L2-normalised.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from musicsim.galleries import gallery_mask
from musicsim.graphs.duplicates import duplicate_groups, near_duplicate_pairs
from musicsim.representations import AlignedRepresentations

__all__ = ["GalleryNodes", "gallery_nodes", "graph_dir"]


@dataclasses.dataclass(frozen=True, slots=True)
class GalleryNodes:
    """The rows that become nodes, and how many items were collapsed to get them."""

    rows: np.ndarray  # positions into AlignedRepresentations.ids, ascending
    ids: np.ndarray  # the item ids of those rows
    n_collapsed: int  # items dropped because a near-duplicate with a smaller id was kept


def gallery_nodes(
    aligned: AlignedRepresentations,
    gallery: str,
    *,
    reference: str,
    threshold: float,
    collapse_duplicates: bool = True,
) -> GalleryNodes:
    """Rows of ``aligned`` that are the nodes of the graph of ``gallery``.

    ``aligned.ids`` is sorted, so ascending rows are also ascending ids and the
    first row of a duplicate group is the one with the smallest ``item_id``.
    """
    if reference not in aligned.X:
        raise ValueError(f"reference must be one of {sorted(aligned.X)}, got {reference!r}")
    rows = np.flatnonzero(gallery_mask(aligned.index, gallery))
    if len(rows) == 0:
        raise ValueError(f"gallery {gallery!r} selects no item out of {len(aligned.ids)}")

    ids = aligned.ids[rows]
    n_collapsed = 0
    if collapse_duplicates:
        pairs = near_duplicate_pairs(aligned.X[reference][rows], ids, threshold=threshold)
        groups = duplicate_groups(ids, pairs)
        _, first = np.unique(groups, return_index=True)
        keep = np.sort(first)
        n_collapsed = len(rows) - len(keep)
        rows, ids = rows[keep], ids[keep]
    return GalleryNodes(rows=rows, ids=ids, n_collapsed=n_collapsed)


def graph_dir(run_directory: Path, label: str, gallery: str, graph: str) -> Path:
    """``<run>/graphs/<label>/<gallery>/<graph>``. Only builds the path; the writer creates it."""
    return run_directory / "graphs" / label / gallery / graph
