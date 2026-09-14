# Music similarity graphs from audio embeddings

Bachelor's Thesis (*Trabajo de Fin de Grado*), Facultad de Informática,
Universidad Complutense de Madrid.

A framework for music similarity experiments. The pipeline is one arrow:

```
audio -> features -> embedding -> similarity measure -> graph (crisp or fuzzy) -> evaluation
```

**Objective.** Study similarity-learning techniques for music analysis and for
building graphs of relationships between songs, with **siamese neural networks**
as the principal focus (O1): three siamese inputs — a log-mel spectrogram
(`siamese_mel`), a frozen audio embedding (`siamese_emb`), and classical
acoustic descriptors (`siamese_desc`) — plus a triplet network, a purpose-built
CNN and MERT. These are compared experimentally (O2) against **classical
classification techniques** (kNN, logistic regression, SVM, random forest, used
both as classifiers and as graph generators) and **classical similarity
measures** (Euclidean, Manhattan, Pearson correlation, Mahalanobis, symmetric
KL divergence). The resulting similarity graphs are built and analysed (O3),
including a **fuzzy block** that models graded, ambiguous musical relationships
on top of any representation (O4). Everything is evaluated (O5) on FMA small
and MagnaTagATune with **Recall@K, MAP and nDCG** as the headline quality
metrics, plus agreement with MagnaTagATune's human similarity judgements.

The main research question: do differences between representations (siamese
networks with different inputs, the triplet network, the CNN, MERT, classical
techniques) translate into significant differences in the **structure** and the
**quality** of the resulting similarity graphs? The evaluation answers it at two
levels:

- **N1 — comparative.** Do graphs from different representations differ
  significantly in quality and structure? This is the level of the
  siamese-input comparison, siamese versus classical techniques, and of
  relating the size of the representation gap to the size of the graph gap.
- **N2 — absolute and of validity.** Does a representation induce a good
  similarity graph on its own, against chance and published references, and do
  conclusions drawn with the genre proxy hold up against MagnaTagATune's human
  judgements?

Probing classifiers over frozen embeddings are a **diagnostic** used inside N1,
not a third level: they measure what a representation makes linearly
decodable, not whether its graph is good. A fuzzy sub-question (does grading
similarity and genre membership help, and does the gain depend on the
representation?) cuts across both levels.

Two students work along two parallel lines after the shared baseline: one on
representation learning and model evaluation, the other on graph construction,
fuzzy techniques and results analysis. Section 10 has the full phase table and
the diagram of the two lines.

**What is shipped today is the MFCC baseline**: MFCC features, cosine similarity,
a kNN graph and its retrieval evaluation, over FMA small, with a random embedding
as the chance floor. A configuration file appears when the phase that implements
it does, so `configs/` never lists an experiment that cannot be run. Section 10
has the phase table.

**Everything an experiment needs is code plus one configuration file.** There are
no manual steps and no constants to edit: a command in section 5 reproduces a
table or a figure of the report.

---

## Contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Getting the data](#3-getting-the-data)
4. [Running on another machine or a server](#4-running-on-another-machine-or-a-server)
5. [Experiments](#5-experiments)
6. [What a run leaves behind](#6-what-a-run-leaves-behind)
7. [Repository layout](#7-repository-layout)
8. [Tests and linter](#8-tests-and-linter)
9. [Building the report](#9-building-the-report)
10. [Status](#10-status)
11. [Licences](#11-licences)

---

## 1. Requirements

- **Python 3.12.** Tested with 3.12.10. Do not use 3.13 or newer: `numba`, a
  dependency of `librosa`, has no stable wheels there and the install fails.
- About **25 GB of free disk** for FMA (7.5 GB of archives, about 16 GB
  unpacked). MagnaTagATune, which arrives in phase R5, needs about 3 GB more.
- **No GPU is needed** for the MFCC baseline, the graph, the retrieval
  evaluation, the structure analysis or the triplet agreement: all of it runs on
  CPU. Looking ahead to later phases: the classical techniques (acoustic
  descriptors, the Gaussian timbre model, the classical classifiers) and the
  fuzzy block also run on CPU only, in minutes; a GPU, or a slow CPU fallback,
  is needed only for the trained networks (the siamese, triplet and CNN
  models) and for MERT (section 4).
- **No FFmpeg is needed**: `soundfile >= 0.12` bundles `libsndfile` with MP3
  support, which is enough for the MP3 files of this corpus. Install it only as
  a safety net if feature extraction reports files it cannot decode:
  `winget install --id Gyan.FFmpeg -e`, then reopen the terminal.

Check the interpreter first:

```powershell
py -3.12 --version    # -> Python 3.12.x
```

## 2. Installation

From the repository root.

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

**Linux or macOS:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

If PowerShell refuses to run the activation script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

The prompt now shows `(.venv)` and `python --version` says 3.12.x.

`pip install -e .` installs the pinned runtime dependencies of
`requirements.txt` and puts the `musicsim` package on the path in editable mode.
The two extra profiles are installed only when they are needed:

```powershell
pip install -r requirements-dev.txt     # pytest, ruff
pip install -r requirements-dl.txt      # torch, torchaudio, transformers (phases R8-R9)
```

Every version is pinned with `==`. The numeric stack in particular is pinned to
the exact versions the reference figures were produced with, so that the
regression checks between phases mean something.

### Verifying the installation

```powershell
python -c "import numpy, scipy, pandas, sklearn, librosa, soundfile, numba, tqdm, joblib, matplotlib, networkx, yaml; import soundfile as sf; print('OK - MP3 supported:', 'MP3' in sf.available_formats())"
python -m musicsim --version
python -m musicsim --help
```

The first command must print `OK - MP3 supported: True`.

## 3. Getting the data

No corpus is redistributed here. Each has its own directory with full
instructions, checksums, licence and the exact layout expected:

| Corpus | Instructions | Size | Used for | State |
|---|---|---|---|---|
| FMA small | [`datasets/fma/README.md`](datasets/fma/README.md) | ~7.5 GB | Everything: training, retrieval, structure, probing | phase R1 |
| MagnaTagATune | [`datasets/mtt/README.md`](datasets/mtt/README.md) | ~3 GB | Evaluation only: human triplets, tag facets | phase R5 |

Only FMA has a configuration file so far; the MagnaTagATune README is there
because the instructions and the licence situation are already worked out, and
`docs/ground-truth-study.md` explains why that corpus is worth the trouble.

The short version, which downloads, verifies the checksums and unpacks:

```powershell
python -m musicsim download --dataset fma_small
python -m musicsim index --dataset fma_small
```

After that the layout is exactly:

```
datasets/
└── fma/
    ├── raw/fma_small/000/000002.mp3     ... 156 directories, 8000 MP3 files
    ├── raw/fma_metadata/tracks.csv      ... genres.csv, features.csv, echonest.csv
    └── index.csv                        item_id, path, split, artist_id, group_id, genre_top, genres_all
```

`index` asserts the counts that every later number depends on: **7994** usable
FMA tracks across 8 genres with an artist-disjoint split. If an assertion fails,
stop: something in the download is wrong.

## 4. Running on another machine or a server

Two environment variables move the two roots. No code and no configuration file
has to change.

| Variable | Meaning | Default |
|---|---|---|
| `MUSICSIM_DATA_DIR` | Directory that holds the corpus directories, `fma/` and later `mtt/` | `<repo>/datasets` |
| `MUSICSIM_OUTPUT_DIR` | Everything the code writes | `<repo>/outputs` |

```powershell
$env:MUSICSIM_DATA_DIR  = "D:\corpora"
$env:MUSICSIM_OUTPUT_DIR = "D:\musicsim-outputs"
```

```bash
export MUSICSIM_DATA_DIR=/scratch/corpora
export MUSICSIM_OUTPUT_DIR=/scratch/musicsim-outputs
```

Every command also takes `--data-dir` and `--output-dir`, which win over the
environment.

### GPU server

The trained models (phase R8) and MERT (phase R9) are the only stages that need
a GPU. Neither exists yet; this is what the setup will look like, and it is here
so that the server can be prepared in advance. On the server:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-dl.txt --no-deps
export MUSICSIM_DATA_DIR=/scratch/corpora MUSICSIM_OUTPUT_DIR=/scratch/musicsim-outputs
python -m musicsim run configs/experiments/e01_mfcc_baseline.yaml
```

Pick the PyTorch index URL that matches the installed CUDA driver; `cu124` above
is an example. From phase R8 on, every extractor will take
`-o extractor.device={auto,cpu,cuda}` and `-o extractor.batch_size=N`.

> The server specification (CPU, RAM, GPU model, disk, operating system) is still
> to be filled in, here and in `report/appendices/a_reproducibility.tex`, once it
> is known.

## 5. Experiments

One experiment is one YAML file, and one command runs it:

```powershell
python -m musicsim run configs/experiments/e01_mfcc_baseline.yaml
```

Anything can be overridden without editing the file, which keeps a quick check
from turning into a committed change:

```powershell
python -m musicsim run configs/experiments/e01_mfcc_baseline.yaml -o graph.k=20 -o seed=0
python -m musicsim show configs/experiments/e01_mfcc_baseline.yaml       # resolve it, run nothing
python -m musicsim registries                                            # what names exist
```

### Experiment table

| Experiment | Command | Configuration | Outputs | Report figure or table |
|---|---|---|---|---|
| **e01** Retrieval baseline | `python -m musicsim run configs/experiments/e01_mfcc_baseline.yaml` | `configs/experiments/e01_mfcc_baseline.yaml` | `metrics/retrieval.csv`, `metrics/retrieval_per_genre.csv` | Section 6.1, main retrieval table |

Later phases add one row each, and the configuration file is written by the
phase that can run it:

- **e02** PCA ablation (R3)
- **e03** graph structure (R4)
- **e04** human triplets (R5)
- **e05** classical classifiers and probing over every representation (R6)
- **e06** graph variants (R7)
- **e07** robustness (R7)
- **e08** model comparison: every representation, Holm-corrected (R10)
- **e09** representation-to-graph link, the noise floor (R10)
- **e10** similarity measures (R12)
- **e11** fuzzy block (R11)

Section 10 has the full phase table.

To run the experiment with a different model, change the extractor; nothing else
moves:

```powershell
python -m musicsim run configs/experiments/e01_mfcc_baseline.yaml -o extractor.name=random
```

Once the figures and tables are final, copy them into the report:

```powershell
python -m musicsim export --run outputs/runs/e01_mfcc_baseline/<timestamp>_<hash>
```

That writes into `report/figures/generated/` and `report/tables/generated/`,
which **are** committed, so that the report compiles without re-running anything.

### Individual stages

The stage subcommands exist for development; they read the same configuration
files, never a different code path.

```powershell
python -m musicsim extract  --config configs/experiments/e01_mfcc_baseline.yaml
python -m musicsim embed    --config configs/experiments/e01_mfcc_baseline.yaml
python -m musicsim graph    --config configs/experiments/e01_mfcc_baseline.yaml
python -m musicsim evaluate --config configs/experiments/e01_mfcc_baseline.yaml
```

Expensive artefacts are cached under the hash of the configuration that produced
them, so re-running a stage with unchanged parameters reuses the result instead
of recomputing it. Add `-o runtime.cache=false` to force recomputation.

## 6. What a run leaves behind

```
outputs/runs/e01_mfcc_baseline/20260912-143512_a694603e/
├── config.resolved.yaml    the configuration after composition and overrides
├── run.json                git commit, whether the tree was dirty, every
│                           dependency version, seed, platform, stage timings
├── log.txt                 everything the run logged
├── metrics/*.csv           the numbers
└── figures/*.pdf           the plots
```

Re-running `config.resolved.yaml` reproduces the run exactly. `run.json` is what
lets a number in the report be traced back to the code that produced it; in
particular `git.dirty` records whether the working tree had uncommitted changes,
because a result produced from a dirty tree is not reproducible from the commit
alone.

## 7. Repository layout

```
.
├── README.md               this file
├── LICENSE                 MIT, for the code only
├── pyproject.toml          package metadata, ruff and pytest configuration
├── requirements*.txt       pinned dependencies: CPU, deep learning, development
├── configs/
│   ├── base.yaml           global defaults: seed, audio, graph, bootstrap
│   ├── datasets/           one file per corpus, added when the corpus is wired up
│   ├── extractors/         one file per model, added when the model exists
│   └── experiments/        one file per runnable experiment
├── datasets/               one directory per corpus, with its own README
├── docs/
│   ├── sources.md              annotated sources; keys match report/bibliography.bib
│   └── ground-truth-study.md   why MagnaTagATune triplets, and what MARBLE is not
├── musicsim/               the package
│   ├── cli.py              subcommands
│   ├── config.py           YAML composition, overrides, stable hashing
│   ├── paths.py            the only place that builds a path
│   ├── registry.py         name-based registries
│   ├── runlog.py           run directories and provenance
│   └── seeding.py          seeds and determinism
├── tests/                  pytest suite, synthetic data only
├── outputs/                gitignored: cache and runs
└── report/                 the LaTeX report; see report/README.md
```

The package sits at the repository root rather than under `src/`, because the
modules share imports and registries and a package is what makes that work.

## 8. Tests and linter

```powershell
pip install -r requirements-dev.txt
pytest
ruff check .
```

The suite runs on synthetic data and needs no corpus, so it finishes in seconds
on a clean clone. Tests that do need real audio are marked `slow` and are
deselected with `pytest -m "not slow"`.

To format rather than only check:

```powershell
ruff format .
```

## 9. Building the report

The report lives in [`report/`](report/), written in LaTeX with the official
TFGTeXiS template. There is no LaTeX installation on the development laptop, so
the default route is Overleaf:

1. Zip `report/` and create an Overleaf project from it.
2. Set the main document to `main.tex` and the compiler to pdfLaTeX.
3. Compile.

Locally, with MiKTeX or TeX Live:

```powershell
cd report
latexmk -pdf main.tex
```

[`report/README.md`](report/README.md) has the details: the directory layout,
every change made to the template and why, the requirements the regulations
impose, and the checklist to run before handing anything in.

## 10. Status

The framework is built phase by phase. Each phase ends green, with its own
acceptance criterion, and adds its draft of the corresponding report section.
After the shared baseline (R1-R3), the work splits into two parallel lines,
one per student, and converges again for the final comparison:

```
Joint:      R1 -> R2 -> R3   (baseline; freezes the interface both lines build on)
                       |
            +----------+-----------+
Student 1:  R5 -> R6 -> R8 -> R9        Student 2:  R4 -> R12 -> R11 -> R7
(representation learning                (graph construction, fuzzy
 and model evaluation)                   techniques, results analysis)
            +----------+-----------+
                       |
Joint:               R10   (e08, e09, model comparison, memoria)
```

| Phase | Student | Deliverable | State |
|---|---|---|---|
| **R0** | — | Scaffold: package core, configuration, registries, run log, tests, report skeleton, docs | **done** |
| R1 | both | Datasets: downloads with checksums, FMA and MTT indices, triplet constraints | pending |
| R2 | both | Features: audio, `mfcc` and `random` extractors, cache, embeddings | pending |
| R3 | both | Graph and retrieval: kNN, duplicates, relevance, R@K/MAP/nDCG/P@K, bootstrap, multi-representation runner, the two-line interface contract; e01, e02 | pending |
| R4 | Student 2 | Structure: communities, hubness, graph comparison and export, plots; e03 | pending |
| R5 | Student 1 | Ground truth: triplets, MTT facets, graded genres; e04 | pending |
| R6 | Student 1 | Classical techniques: `acoustic`, `gauss_mfcc`, classical classifiers, `posterior_*`, probing; e05 | pending |
| R7 | Student 2 | Graph variants and robustness; e06, e07 | pending |
| R8 | Student 1 | Siamese and triplet networks, the CNN classifier, on the server | pending |
| R9 | Student 1 | MERT and `siamese_emb`, on the server | pending |
| R12 | Student 2 | Similarity measures; e10 | pending |
| R11 | Student 2 | Fuzzy block; e11 | pending |
| R10 | both | Model comparison, representation-to-graph link, export and final writing; e08, e09 | pending |

Each phase adds its own configuration files, so `configs/` is a list of what can
be run rather than a wish list. A stage that is not implemented yet says so, and
names the phase that brings it, instead of failing obscurely:

```
$ python -m musicsim graph --config configs/experiments/e01_mfcc_baseline.yaml
musicsim graph: error: stage 'graph' is not implemented yet; it arrives in phase R3.
```

## 11. Licences

The code in this repository is under the **MIT** licence; see
[`LICENSE`](LICENSE).

The corpora are **not** redistributed here and keep their own terms. FMA is used
from phase R1; MagnaTagATune from phase R5.

- **FMA**: code MIT, metadata CC BY 4.0, audio under the licence each artist
  chose.
- **MagnaTagATune**: no explicit licence is published; academic use only.

Third-party libraries keep their own licences, and the ones the results depend on
are cited in the report, as the regulations require. The LaTeX template under
`report/texis/` is TFGTeXiS, distributed by the Facultad de Informática under the
LaTeX Project Public License.

[`docs/sources.md`](docs/sources.md) records every external source, what it
contributes, where it is used and whether it has been verified.
