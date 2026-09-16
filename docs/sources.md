# Sources

Annotated list of every external source the project relies on. The BibTeX keys
are the ones in [`report/bibliography.bib`](../report/bibliography.bib); keep the
two files in step, so that a citation in the report can always be traced to a
line here.

**Status**
- `[verified]` — read, or checked against the primary source, and the claim
  attributed to it here is the claim it actually makes.
- `[to read]` — located and cited from a secondary source, not yet read in full.
  Anything the report states on its authority has to be promoted to `[verified]`
  before submission (V.11).

---

## Datasets

### `Defferrard2017` — FMA: A Dataset for Music Analysis `[verified]`
<https://github.com/mdeff/fma> · <https://arxiv.org/abs/1612.01840>

The primary corpus. `fma_small` is 8000 excerpts of 30 s, 8 genres, 1000 each.

*What it gives.* The audio, the metadata (`tracks.csv`, `genres.csv`), the
official training/validation/test split, and the genre hierarchy.

*Where it is used.* Every FMA experiment: `musicsim/datasets/fma.py`,
`configs/datasets/fma_small.yaml`, report section 3.1, and the download
instructions in `datasets/fma/README.md`.

*Licence.* Code MIT, metadata CC BY 4.0, audio under the licence each artist
chose. The audio is not redistributed here.

*Caveats that the code encodes.* `tracks.csv` has a two-row header. Six excerpts
are unusable and are excluded: `99134`, `108925`, `133297` are about 1 KB with no
decodable audio, and `98565`, `98567`, `98569` decode but last far less than 30 s.

---

### `Law2009` — Evaluation of Algorithms Using Games: The Case of Music Tagging `[verified]`
<https://archives.ismir.net/ismir2009/paper/000019.pdf> ·
dataset: <https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset>

MagnaTagATune: 25 863 clips of about 29 s from 5405 Magnatune songs, 188 tags,
270 artists, collected through the TagATune game. The second corpus, used for
evaluation only.

*What it gives.* `clip_info_final.csv`, `annotations_final.csv` and, decisively,
`comparisons_final.csv`, the odd-one-out votes that are the only human
similarity judgements in the project.

*Where it is used.* `musicsim/datasets/mtt.py`, `configs/datasets/mtt.yaml`,
experiment `e04`, report sections 3.2 and 3.3.

*Licence.* **None is published.** MARBLE lists MTT as not commercially
available. Academic use is well established; the report says exactly this rather
than claiming a licence that does not exist.

*Caveats.* The audio is 16 kHz mono MP3, so it carries nothing above 8 kHz, and
the catalogue leans towards classical, new age and electronic music.

---

### `Bogdanov2019` — The MTG-Jamendo Dataset `[verified]`
<https://github.com/MTG/mtg-jamendo-dataset>

*What it gives.* 55 525 full tracks with genre, instrument and mood/theme tags.

*Where it is used.* **Rejected**, and the report says why: covering the standard
split-0 test set means downloading about 166 GB, the items are whole tracks
rather than excerpts, and the tags are written by whoever uploaded the music, so
the result would be another metadata proxy at a disproportionate cost.

---

## Similarity ground truth and evaluation

### `Wolff2012` — A Systematic Comparison of Music Similarity Adaptation Approaches `[verified]`
<https://openaccess.city.ac.uk/id/eprint/2963/1/ismir2012.pdf>

The source of the protocol this project reproduces.

*What it gives.* The derivation of similarity constraints from the odd-one-out
votes: a vote for the outlier *k* of a triplet *(i, j, k)* yields d(i,j) < d(i,k)
and d(j,i) < d(j,k), and two-cycles are removed by subtracting opposing votes.
It also gives the reference numbers: 15 300 edges, 1598 unique, **860 consistent
constraints of weight 6898, over 337 triplets and 993 clips**, and the accuracy
of learned metrics, 69.5–75.6 %.

*Where it is used.* `musicsim/datasets/mtt.py` asserts every one of those counts;
`musicsim/evaluation/triplets.py`; experiment `e04`; report sections 3.3 and 5.4.

---

### `Karamanolakis2016` — Audio-Based Distributional Semantic Models `[verified]`
<https://arxiv.org/abs/1612.08391>

*What it gives.* The reference figure for an untrained audio descriptor on those
same constraints: a bag of audio words over MFCC + Δ + ΔΔ satisfies
**63.6–64.6 %** of them, and fusing audio with tags reaches 73.1 %. This is the
number the MFCC baseline of this project is read against.

*Where it is used.* Report section 5.4, results table of experiment `e04`.

---

### `Lee2020` — Disentangled Multidimensional Metric Learning for Music Similarity `[verified]`
<https://arxiv.org/abs/2008.03720>

*What it gives.* Similarity is multidimensional, and a set of 879 high-agreement
human triplets over the Million Song Dataset.

*Where it is used.* Related work, and the future-work section. Not used as data:
the MSD audio is not freely available.

---

### `Cleveland2020` — Content-Based Music Similarity with Triplet Networks `[to read]`
<https://arxiv.org/abs/2008.04938>

*What it gives.* Artist identity as a relevance signal on FMA, with retrieval
AUC. Reported there: Euclidean distance over the 518 standardised dimensions of
`features.csv` reaches 0.825.

*Where it is used.* Justifies the `same_artist` relevance function and the
`artist_filter` variant. Read it in full before the report leans on the 0.825.

---

### `Ellis2002` — The Quest for Ground Truth in Musical Artist Similarity `[to read]`
<https://archives.ismir.net/ismir2002/paper/000060.pdf>

*What it gives.* The canonical statement of the problem: there is no ground truth
for musical similarity, and every substitute distorts it.

*Where it is used.* Report section 1.1. Located through `Karamanolakis2016`.

---

### `Logan2003` — Toward Evaluation Techniques for Music Similarity `[to read]`

*What it gives.* Early evaluation methodology for similarity measures.

*Where it is used.* Report section 1.1, alongside `Ellis2002`. Located through
`Karamanolakis2016`.

---

## Representations and benchmarks

### `Castellon2021` — Codified Audio Language Modeling (JukeMIR) `[verified]`
<https://arxiv.org/abs/2107.05677> · <https://github.com/p-lambda/jukemir>

*What it gives.* The probing protocol, and the **MFCC reference figures** that
MARBLE itself does not provide: on MagnaTagATune, ROC-AUC 85.8 and AP 30.2 for a
120-dimensional MFCC descriptor (mean and standard deviation of MFCCs, Δ and ΔΔ),
probed with a linear model and a 512-unit MLP.

*Where it is used.* `musicsim/evaluation/probes.py`, experiment `e05`, and the
acceptance criterion of phase P9.

*Caveat.* The descriptor here is 80-dimensional, not 120, and the
hyper-parameter grids differ. The comparison is indicative, and the report says
so.

---

### `Yuan2023` — MARBLE: Music Audio Representation Benchmark `[verified]`
<https://arxiv.org/abs/2306.10548> ·
<https://github.com/a43992899/MARBLE>

*What it gives.* The two-level framing this project borrows, and the probing
protocol with a frozen backbone and a 512-unit MLP head.

*Where it is used.* Report section 2.2, experiment `e05`.

*Three corrections the report must make, because they are commonly assumed
otherwise.* MARBLE contains **no similarity or retrieval task at all** (cover
song detection is explicitly excluded in v1 and is `[Planning]` in v2); it ships
**no MFCC baseline**, which is why the MFCC numbers come from `Castellon2021`;
and the MTT **triplets are not part of MARBLE**, which uses only the top-50 tags.
The v2 code has no LICENSE file, so there is no explicit permission to reuse it;
none is reused here. Full argument in
[`ground-truth-study.md`](ground-truth-study.md).

---

### `Li2023` — MERT `[to read]`
<https://arxiv.org/abs/2306.00107> ·
<https://huggingface.co/m-a-p/MERT-v1-95M>

*What it gives.* A self-supervised music representation model, and the strongest
published probing figures among the models considered here.

*Where it is used.* `musicsim/extractors/mert.py`, phase P2b, on the GPU server
(or on CPU, at a measured cost).

*Open item.* The licence of the published weights has to be checked on the model
card and recorded here before the report cites the model as used rather than
merely referenced.

---

## Metric learning

### `Hadsell2006` — Dimensionality Reduction by Learning an Invariant Mapping `[to read]`

The contrastive loss used by the siamese extractor. Report section 4.6,
`musicsim/training/losses.py`.

### `Schroff2015` — FaceNet `[to read]`
<https://arxiv.org/abs/1503.03832>

The triplet loss and semi-hard negative mining used by the triplet extractor.
Report section 4.6, `musicsim/training/losses.py`.

---

## Classical techniques and similarity measures

### `Tzanetakis2002` — Musical Genre Classification of Audio Signals `[to read]`

*What it gives.* The classical acoustic descriptor set beyond MFCC: chroma,
spectral contrast, centroid, bandwidth, rolloff, zero-crossing rate, RMS
energy and tempo, pooled as mean and standard deviation. This is the basis of
the `acoustic` extractor.

*Where it is used.* `musicsim/extractors/acoustic.py` (phase P9), report
sections 2 (classical techniques) and 4 (methods).

---

### `Mandel2005` — Song-Level Features and Support Vector Machines for Music Classification `[to read]`

*What it gives.* The single-Gaussian-per-song model over MFCC frames (mean
plus full covariance), the basis of `gauss_mfcc`, and its comparison via
symmetric KL divergence.

*Where it is used.* `musicsim/extractors/gauss_mfcc.py`,
`musicsim/similarity.py` (phase P6), report sections 2 and 4.

---

### `LedoitWolf2004` — A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices `[to read]`

*What it gives.* The shrinkage estimator used to regularise the covariance
matrix for the Mahalanobis similarity measure on `mfcc` and `acoustic`.

*Where it is used.* `musicsim/similarity.py` (phase P6), report section 4
(similarity measures).

---

## Fuzzy modelling

All entries in this section are `[to read]`: they must be read in full and
verified before the report cites them (V.11).

### `Zadeh1965` — Fuzzy Sets `[to read]`

*What it gives.* The foundational definition of graded set membership that
the whole fuzzy block (F1–F4) builds on.

*Where it is used.* `musicsim/fuzzy/*` (phase P7), report section 2 (fuzzy
modelling) and section 4.

---

### `Rosenfeld1975` — Fuzzy Graphs `[to read]`

*What it gives.* Fuzzy graphs, i.e. graphs with graded edge membership instead
of a crisp 0/1 edge, the formal object F1 constructs.

*Where it is used.* `musicsim/fuzzy/graph.py` (F1, phase P7), report
sections 2 and 4.

---

### `McInnes2018` — UMAP: Uniform Manifold Approximation and Projection `[to read]`

*What it gives.* Fuzzy simplicial sets as a related construction for local,
calibrated membership from a distance, cited as related work for the F1
membership function ($\rho_i$, $\sigma_i$ calibration).

*Where it is used.* Report section 2 (fuzzy modelling), as related work only;
not a dependency of `musicsim/fuzzy/graph.py`.

---

### `Mamdani1975` — An Experiment in Linguistic Synthesis with a Fuzzy Logic Controller `[to read]`

*What it gives.* Mamdani-style fuzzy inference (min/max composition,
centroid defuzzification), one of the two inference schemes F2 may use.

*Where it is used.* `musicsim/fuzzy/inference.py` (F2, optional extension after
stage S4: the supervisor did not endorse the rule-based route), report sections
2 and 4.

---

### `Takagi1985` — Fuzzy Identification of Systems and Its Applications to Modeling and Control `[to read]`

*What it gives.* Takagi-Sugeno (zero-order) fuzzy inference, the second
inference scheme F2 may use.

*Where it is used.* `musicsim/fuzzy/inference.py` (F2, optional extension after
stage S4), report sections 2 and 4.

---

### `Keller1985` — A Fuzzy K-Nearest Neighbor Algorithm `[to read]`

*What it gives.* Fuzzy kNN classification, the basis of F3a: graded genre
membership from the similarity graph.

*Where it is used.* `musicsim/fuzzy/knn.py` (F3a, phase P7), report sections
2 and 4.

---

### `Bezdek1981` — Pattern Recognition with Fuzzy Objective Function Algorithms `[to read]`

*What it gives.* Fuzzy c-means clustering, the basis of F3b: overlapping
genre communities over the embeddings.

*Where it is used.* `musicsim/fuzzy/cmeans.py` (F3b, phase P7), report
sections 2 and 4.

---

### `Hullermeier2012` — The Rand Index and Beyond `[to read]`

*What it gives.* The fuzzy Rand index used to score F3 against multi-label
ground truth, and as the fuzzy analogue of the crisp ARI used for Louvain
communities.

*Where it is used.* `musicsim/evaluation/fuzzy_metrics.py` (phase P7), report
sections 4 and 6 (fuzzy modelling).

---

## Representation comparison

### `Kornblith2019` — Similarity of Neural Network Representations Revisited `[to read]`

*What it gives.* Linear CKA (and, as a check, CKA with an RBF kernel), used to
relate the distance between representations to the distance between the
graphs they induce (SQ3, the noise-floor and dose-response analysis).

*Where it is used.* `musicsim/evaluation/representation.py`,
`musicsim/evaluation/link.py` (phase P11), report sections 2 and 6
(representation gap and graph gap).

---

## Tools

Cited because V.11 requires third-party software to be credited and its licence
respected. All four are permissively licensed (ISC, BSD-3-Clause, BSD-3-Clause
and BSD-3-Clause respectively) and none of them is redistributed here.

| Key | Tool | Used for |
|---|---|---|
| `McFee2015` | librosa | decoding, resampling, log-mel spectrograms, MFCCs |
| `Pedregosa2011` | scikit-learn | scaling, PCA, probing classifiers, clustering metrics |
| `Harris2020` | NumPy | every array operation, and the kNN search itself |
| — | pandas, SciPy, matplotlib, NetworkX | indices, statistics, figures, graph metrics |

---

## Regulations and templates

Not cited in the bibliography; listed here because the repository depends on
them.

| Source | URL | Used for |
|---|---|---|
| TFG regulations, Junta de Facultad 18/03/2024 | <https://informatica.ucm.es/file/normativa-tfg-actual?ver> | Every requirement in `report/README.md` section 3 |
| TFGTeXiS template | <https://informatica.ucm.es/file/plantilla_tfg_latex?ver> | `report/texis/`, LPPL |
| TFG page, 2026–2027 | <https://informatica.ucm.es/tfgs-2026-2027> | Calendar and submission procedure |

---

## Open items

1. Promote every `[to read]` entry to `[verified]`, or drop the claim that rests
   on it.
2. Record the licence of the MERT weights (`Li2023`).
3. Confirm the exact venue and pages of `Logan2003`, which is currently cited
   second-hand.
4. Settle whether "academic use, no explicit licence" is the wording the
   supervisor wants for MagnaTagATune in the report.
5. Read and verify the twelve classical-technique and fuzzy-modelling entries
   (`Tzanetakis2002` through `Kornblith2019`) before the sections that cite them
   (report sections 2, 4 and 6) are written past `\todo` stage.
6. Confirm exact venue, year and author list for `Mandel2005`,
   `LedoitWolf2004`, `Mamdani1975`, `Takagi1985`, `Keller1985` and
   `Hullermeier2012`: they are cited here from memory of the standard
   reference and need a primary-source check before entering
   `report/bibliography.bib`.
