# Music similarity graphs from audio representations

Music similarity has no operational definition: no public corpus annotates
"sounds like", so the field substitutes proxies — genre labels chosen by the
artist, tags written by listeners, and a few small sets of human judgements.
This project studies **similarity-learning techniques** for music and the
**graphs of relations between songs** they induce.

The pipeline is one arrow:

```
audio -> features -> embedding -> similarity -> kNN graph -> evaluation
```

**The question:** do differences between music representations translate into
*significant* differences in the structure and the quality of the resulting
similarity graphs?

Around it:

- **Siamese networks** are the main focus, learning latent representations from
  spectrograms, from pretrained audio embeddings and from acoustic descriptors.
- They are compared against **classical classification and similarity
  techniques** (MFCC and acoustic descriptors, kNN, logistic regression, SVM,
  random forest, Mahalanobis and Gaussian-KL distances), a **CNN classifier** of
  our own and the pretrained **MERT** model.
- **Fuzzy** membership degrees model relations that are gradual or ambiguous,
  both on graph edges and on genre membership.
- Evaluation uses **FMA** and **MagnaTagATune** with **Recall@K, MAP and NDCG**,
  plus the human similarity judgements shipped with MagnaTagATune.

Everything an experiment needs is code plus one configuration file: no manual
steps, no constants to edit. Methodological detail lives in [`docs/`](docs/) and
in the report under [`report/`](report/), not in this file.

> **Status.** The MFCC baseline is the current milestone: FMA download and
> indexing, feature extraction, the kNN graph and the first retrieval metrics.
> See [Status](#status) for what runs today and what each stage still waits for.

## Contents

1. [Repository layout](#repository-layout)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Datasets](#datasets)
5. [Running experiments](#running-experiments)
6. [Tests](#tests)
7. [Status](#status)
8. [Licence](#licence)

## Repository layout

```
.
├── pyproject.toml          package metadata, dependencies, ruff and pytest config
├── requirements*.txt       thin wrappers over the pyproject extras
├── configs/
│   ├── base.yaml           global defaults: seed, audio, graph, bootstrap
│   ├── datasets/           one file per corpus
│   ├── extractors/         one file per representation
│   └── experiments/        one file per runnable experiment
├── datasets/               one directory per corpus, each with its own README
├── docs/                   annotated sources and the ground-truth study
├── musicsim/               the package (config, paths, registries, run log, CLI)
├── tests/                  pytest suite, synthetic data only
├── outputs/                gitignored: cache and one directory per run
└── report/                 the LaTeX report; see report/README.md
```

A configuration file appears when the phase that can run it does, so `configs/`
never lists an experiment that does not work.

## Installation

Requires **Python 3.12** (`numba`, a dependency of `librosa`, has no stable
wheels on 3.13+). Check with `python --version`.

```bash
python -m venv .venv
```

Activate it — this is the one command that differs per platform:

```bash
source .venv/bin/activate        # Linux, macOS
.venv\Scripts\activate           # Windows (cmd)
.venv\Scripts\Activate.ps1       # Windows (PowerShell)
```

Then install the package and its pinned dependencies:

```bash
python -m pip install --upgrade pip
pip install -e .
```

**`pyproject.toml` is the single source of dependencies.** Two extras are
installed only when needed:

```bash
pip install -e ".[dev]"    # pytest, ruff
pip install -e ".[dl]"     # torch, torchaudio, transformers (trained models, MERT)
```

The `requirements*.txt` files are kept for backwards compatibility and only
forward to those extras. Every version is pinned with `==`, and the numeric
stack is pinned to the exact versions the reference figures were produced with,
so the regression checks between phases mean something.

Verify the install:

```bash
musicsim --version
musicsim registries
```

`musicsim` and `python -m musicsim` are equivalent; the rest of this file uses
the shorter form.

<details>
<summary>If audio files fail to decode</summary>

`soundfile >= 0.12` bundles `libsndfile` with MP3 support, which is enough for
these corpora. Install FFmpeg only as a fallback if feature extraction reports
files it cannot decode (`winget install --id Gyan.FFmpeg -e` on Windows,
`apt install ffmpeg` or `brew install ffmpeg` elsewhere), then reopen the shell.
</details>

## Quick start

Reproducing the MFCC baseline of
[`configs/experiments/e01_mfcc_baseline.yaml`](configs/experiments/e01_mfcc_baseline.yaml)
takes three commands:

```bash
musicsim download --dataset fma_small     # ~7.5 GB, checksum-verified
musicsim index    --dataset fma_small     # builds datasets/fma/index.csv
musicsim run configs/experiments/e01_mfcc_baseline.yaml
```

The run writes its metrics, figures and provenance under
`outputs/runs/e01_mfcc_baseline/<timestamp>_<hash>/`.

Any stage that is not implemented yet stops with a message naming the phase that
brings it, instead of failing obscurely:

```
$ musicsim graph --config configs/experiments/e01_mfcc_baseline.yaml
musicsim graph: error: stage 'graph' is not implemented yet; it arrives in phase P3.
```

## Datasets

No corpus is redistributed here. Each directory holds the download commands,
the checksums, the licence and the expected layout.

| Corpus | Instructions | Size | Used for | Arrives in |
|---|---|---|---|---|
| FMA small | [`datasets/fma/README.md`](datasets/fma/README.md) | ~7.5 GB | Training and evaluation | P1 |
| MagnaTagATune | [`datasets/mtt/README.md`](datasets/mtt/README.md) | ~3 GB | External evaluation: human triplets, tags | P8 |

`musicsim index` asserts the counts every later number depends on: **7994**
usable FMA tracks across 8 genres, with no artist shared between splits. If an
assertion fails, stop — something in the download is wrong.

Two environment variables move the roots, so a server can read the corpora from
another disk without any change to code or configuration:
`MUSICSIM_DATA_DIR` (default `datasets/`) and `MUSICSIM_OUTPUT_DIR` (default
`outputs/`). Every command also accepts `--data-dir` and `--output-dir`, which
win over the environment.

## Running experiments

One experiment is one YAML file, and one command runs it:

```bash
musicsim run configs/experiments/e01_mfcc_baseline.yaml
```

Values can be overridden without editing the file, which keeps a quick check
from turning into a committed change:

```bash
musicsim run configs/experiments/e01_mfcc_baseline.yaml -o graph.k=20 -o seed=0
musicsim show configs/experiments/e01_mfcc_baseline.yaml     # resolve it, run nothing
```

| Experiment | Question | Outputs |
|---|---|---|
| **e01** MFCC baseline | Does the MFCC embedding retrieve same-genre neighbours better than chance? | `metrics/retrieval.csv`, `metrics/retrieval_per_genre.csv` |

Later phases add one row each: PCA ablation, graph structure, human triplets,
classical classification, graph variants, robustness, model comparison, fuzzy
modelling and graph neural link prediction.

Expensive artefacts are cached under the hash of the configuration that produced
them, so re-running with unchanged parameters reuses the result. Every run also
records the git commit, whether the tree was dirty, dependency versions, the
seed, the platform and stage timings in `run.json`, which is what lets a number
in the report be traced back to the code that produced it.

Individual stages (`extract`, `embed`, `graph`, `evaluate`) exist for
development and read the same configuration files, never a different code path.

## Tests

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

The suite runs on synthetic data and needs no corpus, so it finishes in seconds
on a clean clone. Tests that need real audio are marked `slow`
(`pytest -m "not slow"` skips them).

## Status

The framework is built in phases; each ends green with its own acceptance
criterion and adds its draft of the corresponding report section.

| Stage | Phase | Deliverable | State |
|---|---|---|---|
| — | **P0** | Scaffold: configuration, paths, registries, run log, CLI, tests, report skeleton | **done** |
| **S1 Baselines** | P1 | FMA download with checksums and item index | **done** |
| | P2 | Audio, log-mel cache, `mfcc` and `random` extractors, embeddings | pending |
| | P3 | kNN graph, retrieval metrics (Recall@K, MAP, NDCG), bootstrap; e01, e02 | pending |
| **S2 Graphs** | P4-P7 | Structure and hubness, graph variants and robustness, similarity measures, fuzzy graphs and memberships | pending |
| | P8-P9 | MagnaTagATune ground truth; classical classification and probing | pending |
| **S3 Siamese** | P10 | Siamese (spectrogram, embedding, descriptors), triplet and CNN | pending |
| **S4 Comparison** | P11 | Paired model comparison, export, final writing | pending |
| **S5 Extension** | P12 | Graph neural network for link prediction | deferred until S4 is reproducible |

## Licence

The code is under the **MIT** licence: see [`LICENSE`](LICENSE). The corpora are
not redistributed and keep their own terms, documented in each
`datasets/*/README.md`; third-party libraries keep theirs. The LaTeX template
under `report/texis/` is TFGTeXiS, distributed by the Facultad de Informática of
the Universidad Complutense de Madrid.

[`docs/sources.md`](docs/sources.md) records every external source, what it
contributes and whether it has been verified.

---

Bachelor's thesis (*Trabajo de Fin de Grado*), Facultad de Informática,
Universidad Complutense de Madrid.
