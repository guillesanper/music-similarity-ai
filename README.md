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
musicsim download --dataset fma_small     # ~7.5 GB, checksum-verified: minutes to hours, connection-bound
musicsim index    --dataset fma_small     # builds datasets/fma/index.csv: a few seconds
musicsim run configs/experiments/e01_mfcc_baseline.yaml
```

`run` extracts and embeds the 7994 tracks (a few minutes the first time, on
CPU; seconds afterwards, from cache), builds the graph (seconds), then
evaluates retrieval for `mfcc` and the `random` control (roughly 1-3 minutes,
depending on the cache). The run writes its metrics, figures and provenance
under `outputs/runs/e01_mfcc_baseline/<timestamp>_<hash>/`.

A stage that is not implemented yet stops with a clear message instead of
failing obscurely:

```
$ musicsim export --run outputs/runs/e01_mfcc_baseline/<dir>
musicsim export: error: stage 'export' is not implemented yet. See README.md for what runs today.
```

## Datasets

No corpus is redistributed here. Each directory holds the download commands,
the checksums, the licence and the expected layout.

| Corpus | Instructions | Size | Used for | State |
|---|---|---|---|---|
| FMA small | [`datasets/fma/README.md`](datasets/fma/README.md) | ~7.5 GB | Training and evaluation | **done** |
| MagnaTagATune | [`datasets/mtt/README.md`](datasets/mtt/README.md) | ~3 GB | External evaluation: human triplets, tags | pending |

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
| **e01** MFCC baseline | Does the MFCC embedding retrieve same-genre neighbours better than chance? | `metrics/per_query.csv`, `metrics/retrieval.csv`, `metrics/retrieval_per_genre.csv`, `graphs/mfcc/{neighbors.npz, edges.csv, duplicates.csv}` |
| **e01 (MERT)** MERT baseline | Does MERT's titular layer retrieve same-genre neighbours better than MFCC and chance? | same `metrics/` files, with `mert` as the main representation and `mfcc`/`random` as baselines |
| **e02** PCA ablation | Does reducing MFCC's 80 dimensions cost retrieval quality? | same `metrics/` files, one run per `-o embedding.pca.n_components=` override (16, 32, 48) |

Later phases add one row each: graph structure, human triplets, classical
classification, graph variants, robustness, model comparison, fuzzy modelling
and graph neural link prediction.

Retrieval evaluates two galleries, both restricted to the items every
representation being compared has in common (`retrieval.gallery` in
[`configs/base.yaml`](configs/base.yaml), a `gallery` column in every
`metrics/*.csv`): **`fma_test`** (default) restricts both queries and
candidates to the ~800 test-split items — artists no representation has been
fitted on — and is the main evaluation everywhere a model was actually
trained on `training`. **`fma_all`** lifts the restriction to all 7994 items;
an experiment opts into it explicitly, as a secondary analysis and to stay
comparable with the archived pipeline (whose regression figures, `P@10` dedup
0.327 and 0.306 for `queries=all`/`test`, are reproduced on `fma_all`;
`configs/experiments/e02_pca_ablation.yaml` reproduces its PCA ablation,
ΔP@10 −0.026/−0.012/−0.003 at 16/32/48 components).

`fma_test` new reference figures for e01 (`mfcc` vs. `random`, dedup, headline
metrics): R@10 0.035 vs. 0.013, MAP 0.240 vs. 0.132, nDCG@10 0.363 vs. 0.129 —
checked in [`tests/test_retrieval_regression.py`](tests/test_retrieval_regression.py)
(`slow`, needs the real FMA download).

### MERT

[`configs/extractors/mert.yaml`](configs/extractors/mert.yaml) runs
[MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) (Li et al. 2023, a
frozen 12-layer transformer pre-trained on 24 kHz music audio), needs
`pip install -e ".[dl]"`, and downloads its weights from Hugging Face into
the user's own cache the first time it runs — never into this repository —
under **CC-BY-NC-4.0** (non-commercial research use, which this thesis is).
Because the model was pre-trained on 5 s crops, every 30 s clip is processed
as six non-overlapping 5 s windows rather than one 30 s forward pass, and all
13 hidden states (the convolutional feature extractor's output plus the 12
transformer layers) are cached per track; the "titular" layer the rest of the
pipeline actually uses is chosen once, by Recall@10 on FMA **validation**,
never test (`musicsim.extractors.mert.select_titular_layer`), and is baked
into `configs/extractors/mert.yaml`'s `extractor.mert_layer`. Measured on the
800-item FMA validation gallery: layer **6** (of 0–12) gives the highest
Recall@10, **0.0548**, over a roughly unimodal curve across the 13 layers
(0.0411–0.0548, lowest at the edges of the stack, highest around the middle).

Extraction is CPU-viable but slow: **~4.0 s/track** measured on an 8-core CPU
(six 5 s forward passes per track, single process — see
`MertExtractor.extract_all`'s docstring for why), so a full `fma_small` run
(`configs/experiments/e01_mert_baseline.yaml`, 7994 tracks) costs on the order
of **9 hours of CPU time** — plan accordingly, or run it once and let the
cache absorb the cost for every later experiment that reuses the same titular
layer.

With a CUDA GPU the extraction is much faster, and nothing needs to be
switched on: `extractor.device: auto` (the default in
`configs/extractors/mert.yaml`) uses the GPU when torch sees one and the CPU
otherwise, `cpu` / `cuda` / `cuda:N` force a choice, and an explicit `cuda`
without a GPU is an error rather than a silent fallback. The CPU path is
unchanged. On a GPU, still one process, the windows of several consecutive
tracks are grouped into one forward pass (`extractor.batch_windows`, 24 by
default, about 5 GB of VRAM; it halves itself if a forward runs out of
memory), and the audio of the next tracks is decoded and resampled on
`extractor.decode_workers` threads while the GPU works, because decoding 30 s
of mp3 costs about as much as the forward pass itself. The 13-layer cache is
written per track as soon as its batch is done, so an interrupted run resumes
from where it stopped, and none of these settings is part of the cache key: a
cache filled on the CPU is valid on a GPU and the other way round (the values
agree to float tolerance, not bit for bit), and the cache directory
(`outputs/cache/fma/mert-layers-*`, about 160 MB for `fma_small`) can simply be
copied to another machine. `extractor.precision: float16` (autocast, off by
default) is available but should only be switched on after checking that it
agrees with `float32`. Timings on a GPU have not been measured yet; the
`gpu`-marked tests (`pytest -m gpu`) check the device path on a machine that
has one and skip themselves otherwise.

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

The framework is built incrementally; each milestone ends green with its own
acceptance criterion and adds its draft of the corresponding report section.

| Stage | Deliverable | State |
|---|---|---|
| Scaffold | Configuration, paths, registries, run log, CLI, tests, report skeleton | **done** |
| **S1 Baselines** | FMA download with checksums and item index | **done** |
| **S1 Baselines** | Audio, log-mel cache, `mfcc` and `random` extractors, embeddings | **done** |
| **S1 Baselines** | kNN graph, near-duplicate detection, retrieval metrics (Recall@K, MAP, nDCG@10, P@10) over the `fma_test`/`fma_all` galleries, clustered bootstrap with Holm correction; e01, e02 | **done** |
| **S1 Baselines** | `mert` extractor (13 layers) and titular layer selection on real FMA validation data | **done** |
| **S1 Baselines** | e01 (MERT): full `fma_test`/`fma_all` comparison against `mfcc`/`random` | **pending** — extractor and config are ready (`configs/experiments/e01_mert_baseline.yaml`); the full `fma_small` extraction costs ~9h of CPU time, so it awaits either that time budget or GPU acceleration |
| **S2 Graphs** | Structure and hubness, graph variants and robustness, similarity measures, fuzzy graphs and memberships | pending |
| **S2 Graphs** | MagnaTagATune ground truth; classical classification and probing | pending |
| **S3 Siamese** | Siamese (spectrogram, embedding, descriptors), triplet and CNN | pending |
| **S4 Comparison** | Paired model comparison, export, final writing | pending |
| **S5 Extension** | Graph neural network for link prediction | deferred until S4 is reproducible |

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
