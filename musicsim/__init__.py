"""musicsim: a framework for music similarity experiments.

The pipeline is one arrow::

    audio -> features -> embedding -> cosine similarity -> kNN graph -> evaluation

and the evaluation has two levels:

N1  Does the embedding contain musical information? Probing classifiers on
    genre, sub-genre and tag tasks, in the spirit of the MARBLE benchmark.
N2  Does the embedding induce a good similarity graph? Retrieval with
    multi-faceted relevance, agreement with human triplet judgements, graph
    structure, robustness and generalisation.

Design principles, in one line each:

* one experiment is one YAML file under ``configs/experiments/``;
* datasets, extractors, graph builders, relevance functions and evaluators are
  registered by name, so adding a model means adding a class and a YAML file;
* :mod:`musicsim.paths` is the only place that builds a path;
* expensive artefacts are cached under the hash of the configuration that
  produced them;
* every execution writes a run directory with its resolved configuration and
  its provenance.

The top level imports nothing heavy on purpose, so ``python -m musicsim --help``
stays fast and works before the optional dependencies are installed.
"""

from __future__ import annotations

__all__ = ["__version__"]

#: Keep in sync with ``project.version`` in pyproject.toml.
__version__ = "0.1.0"
