"""musicsim: a framework for music similarity experiments.

The pipeline is one arrow::

    audio -> features -> embedding -> similarity measure -> graph (crisp or fuzzy) -> evaluation

Siamese networks over three kinds of input (spectrogram, audio embedding,
acoustic descriptors), a triplet network, a CNN classifier and MERT are
compared against classical techniques: MFCC, acoustic descriptors, a Gaussian
timbre model, classical classifiers used as graph generators, and several
similarity measures. A fuzzy block models graded similarity and genre
membership on top of any representation.

The evaluation has two levels, plus a diagnostic that is not a level:

N1  Do different representations induce significantly different graphs, in
    quality and structure? Comparative questions: which siamese input works
    best, how siamese networks compare with classical techniques, and whether
    the size of the representation gap predicts the size of the graph gap.
N2  Does a representation induce a good similarity graph on its own, against
    chance and against published references, and do the conclusions drawn
    with the genre proxy hold up against the human triplet judgements of
    MagnaTagATune?

Probing classifiers over frozen embeddings are a diagnostic used inside N1,
not a third level: they measure what a representation makes linearly
decodable, which is a different question from graph quality.

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
