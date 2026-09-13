# Ground truth for musical similarity: what MARBLE can and cannot provide

*Study carried out on 2026-09-11 in the archived repository, ported to English
here. No code was written and no data produced; the figures marked `[V]` were
computed read-only from the original distribution files.*

**Convention.** `[V]` means verified against the primary source cited, or
computed here from its original files (Appendix A). `[I]` means an inference or
an estimate. Every `[V]` carries its URL; full references are at the end, and the
BibTeX keys match [`sources.md`](sources.md) and
[`../report/bibliography.bib`](../report/bibliography.bib).

---

## 0. Summary

1. **MARBLE has no similarity task and no retrieval task.** It is a benchmark of
   *classification and regression* over frozen representations. Version 1
   explicitly excludes cover song detection for lack of a common dataset, and
   version 2 lists it as `[Planning]`. It also ships no MFCC baseline; the MFCC
   reference numbers under that protocol come from Castellon et al. (2021), the
   work MARBLE takes its tasks from.
2. As a **source of ground truth**, the only part of MARBLE that contains
   **human similarity judgements** is the set of **TagATune triplets** shipped
   with MagnaTagATune — which MARBLE itself does not use: 533 rows, 7650 votes,
   **860 consistent constraints over 337 triplets and 993 clips**. It is a small
   set, but it is the standard of the literature, it weighs about 115 MB if only
   the clips that are needed are extracted, and extracting MFCCs from them costs
   roughly three minutes of CPU time.
3. The remaining annotations (MTT and MTG-Jamendo tags, emotion, key) are
   **proxies, exactly as genre is**: multi-faceted, but not similarity
   judgements. MTG-Jamendo does not fit on this hardware either: the items are
   whole tracks and the split-0 test set is spread over 100 tar files of about
   1.66 GB each, roughly 166 GB of download.
4. **No MARBLE dataset is FMA.** Taking a ground truth from MARBLE means
   **adding a second corpus**. With MTT the second corpus would be
   evaluation-only: no training, no MERT.
5. **Recommendation:** as the primary option, the **MTT human triplets**, as a
   perceptual evaluation and for evaluation only. As a secondary option,
   **graded relevance over the FMA `genres_all` hierarchy**, which costs no
   download at all. Out of scope: MTG-Jamendo, MARBLE as a probing benchmark
   until there is a CNN of our own, and Echonest.
6. One preliminary finding, computed here without any audio: on MTT, when the
   genre tags separate the two candidates of a constraint — which happens in only
   **12 %** of them — they agree with the human vote **90 %** of the time. Genre
   is a **precise proxy with poor coverage** for perceived similarity. That is the
   core of the research question this study proposes.

---

## 1. What MARBLE actually is

### 1.1 Publication and versions

- `[V]` *MARBLE: Music Audio Representation Benchmark for Universal Evaluation*,
  Yuan, Ma, Li, Zhang *et al.*, 25 authors, published at **NeurIPS 2023, Datasets
  and Benchmarks** (volume 36, pp. 39626–39647), arXiv 2306.10548v4.
- `[V]` Same group as MERT: MARBLE evaluates `MAP-MERT-v0/v1` and links to
  `huggingface.co/m-a-p/MERT-v1-95M` and `-330M`.
- `[V]` The current version is **MARBLE v2**, published on the `main` branch on
  2025-06-04; v1 is kept on `main-v1-archived`. v2 has no paper of its own and
  cites the NeurIPS 2023 one.
- `[V]` The leaderboard at `https://marble-bm.shef.ac.uk`, cited in the paper,
  **does not respond** (connection refused or timeout, over both HTTP and HTTPS,
  as of 2026-09-11). Every figure quoted here comes from the paper tables.

### 1.2 Taxonomy, tasks and metrics (v1)

`[V]` Four levels — high-level description, score, performance and acoustic —
with 18 tasks over 12 datasets (Table 1 of the paper):

| Level | Task | Dataset | Metric |
|---|---|---|---|
| High-level | Key | GiantSteps key | weighted score |
| | Tagging | **MagnaTagATune**, MTG Top50 | ROC-AUC, PR-AUC/AP |
| | Genre | GTZAN, MTG Genre | accuracy; ROC-AUC, AP |
| | Emotion | **EmoMusic** (V/A), MTG MoodTheme | R² valence/arousal; ROC-AUC, AP |
| Score | Pitch | NSynth | accuracy |
| | Beat | GTZAN Rhythm | F1 (20 ms) |
| | Melody | "MelodyDB" | accuracy |
| | Chords | GuitarSet | 8 mir_eval metrics |
| | Lyrics | MulJam2.0, Jamendo | CER, WER |
| Performance | Vocal technique | VocalSet | accuracy |
| Acoustic | Singer | VocalSet | accuracy |
| | Instrument | NSynth, MTG Instrument | accuracy; ROC-AUC, AP |
| | Separation | MUSDB18 | SDR |

### 1.3 Evaluation protocol

- `[V]` **v1** has three tracks: *unconstrained* (fine-tuning allowed),
  *semi-constrained* (frozen backbone) and ***constrained*** (frozen backbone
  plus a **single-layer 512-unit MLP**, or an LSTM/transformer for sequential
  tasks), with a bounded grid: each layer or a weighted sum of layers,
  lr ∈ {5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2}, batch 64, dropout 0.2. The stated
  compute budget is one week per task on a single RTX 3090.
- `[V]` **v2** uses LightningCLI and YAML; the encoder returns
  `hidden_states[B, L, T, H]`, a *LayerSelector* and a *TimeAvgPool* are applied,
  and a decoder (the probe) follows. It needs `torch==2.6.0`,
  `lightning==2.5.1`, conda, **ffmpeg** and a Weights & Biases account. Data
  comes from Hugging Face (`m-a-p/{dataset}`) and **some of it is gated**.
- `[V]` **Licences.** The v2 code has **no LICENSE file** in its root and
  `pyproject.toml` declares none (checked through the GitHub API on 2026-09-11),
  so there is no explicit permission to reuse it. Table 1 of the paper marks only
  GiantSteps key, NSynth and VocalSet as commercially available (plus MIT for
  GuitarSet and Jamendo-lyrics). **MTT, MTG-Jamendo, GTZAN and EmoMusic are
  marked "–".**

### 1.4 Similarity? Retrieval? An MFCC baseline?

- `[V]` **There is no similarity task and no retrieval task.** Quoting the v1
  limitations: *"we do not include some MIR tasks that lack a common dataset
  currently, such as cover song detection and query-by-humming"*. In v2, SHS is
  `[Planning]`. Cover song detection measures identity of work anyway, not
  "sounds alike".
- `[V]` **Models evaluated in v1:** nine variants of seven models — MusiCNN,
  CLMR, Jukebox-5B, MULE, MAP-Music2Vec, MAP-MERT-v0 (two) and MAP-MERT-v1 (95M,
  330M). **No classical descriptor.** v2 adds CLaMP3, DaSheng, MERT, MuQ,
  MuQ-MuLan, MusicFM, Qwen2-Audio, Qwen2.5-Omni, Xcodec and an identity encoder;
  still no MFCC.
- `[V]` **The MFCC reference exists outside MARBLE**, in Castellon, Donahue and
  Liang (ISMIR 2021, "JukeMIR"), the work MARBLE takes MTT, GTZAN, GS and EMO
  from. Their MFCC is librosa with default settings, mean and standard deviation
  of MFCCs, Δ and ΔΔ, 120 dimensions, probed with a linear model and a 512-unit
  MLP (their Table 3):

| Representation | MTT AUC | MTT AP | GTZAN acc | GS key | Emo A (R²) | Emo V (R²) |
|---|---|---|---|---|---|---|
| MFCC (Castellon) | 85.8 | 30.2 | 44.8 | 14.6 | 47.9 | 26.5 |
| CHROMA (Castellon) | 77.6 | 18.5 | 32.8 | 56.5 | 29.3 | 5.9 |
| MusiCNN (MARBLE, constrained) | 90.3 | 37.8 | 73.5 | 14.4 | 68.8 | 44.4 |
| MERT-v1-95M (MARBLE) | 91.0 | 39.3 | 74.8 | 63.5 | 76.3 | 55.5 |

`[I]` The Castellon rows and the MARBLE rows are not strictly comparable — the
hyper-parameter grids and the probe details differ — but they share tasks and
splits.

---

## 2. Two distinct uses of MARBLE

### 2.a As a probing benchmark

*What it gives.* It places our MFCCs, and tomorrow our CNN, next to MusiCNN or
MERT on standard tasks. **It is not a similarity ground truth**: it measures what
is linearly decodable, not whether the cosine geometry orders neighbours well.

*Cost.* `[V/I]` The v2 code needs a GPU in practice for the large encoders, plus
ffmpeg, Lightning and W&B, and has no explicit licence. `[I]` For MFCCs the
MARBLE code is unnecessary: reproducing the protocol with scikit-learn (a
512-unit MLP or a logistic model over our 80-dimensional vector) runs on CPU in
minutes. The data is the expensive part: the full MTT is 2.97 GB and about
25 900 clips, roughly 65 minutes of MFCC extraction, scaling linearly from the
20 minutes our 7994 clips take.

*Marginal value today: low.* Castellon already publishes the MFCC row. Repeating
it with an 80-dimensional descriptor would only confirm the figure. It becomes
worthwhile **once there is a CNN of our own**.

*Risk.* Mixing probing and similarity in the report. They answer different
questions and belong in different sections.

### 2.b As a source of similarity ground truth

The subject of section 3. In short: of everything MARBLE touches, **only MTT
carries human similarity judgements**, and MARBLE does not use them. The rest are
per-item class labels — proxies like genre, from other facets.

---

## 3. Candidate ground truths

Cost reference: MFCCs for 7994 clips of 30 s (66.6 hours of audio) take about
20 minutes on 8 cores, roughly 200× real time. The CPU costs below scale linearly
from that figure `[I]`.

### 3.1 MagnaTagATune

**Basics** `[V]` (the [official page](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset),
hosted by City University "with kind permission" since tagatune.org went down):

- 25 863 clips of about 29 s from 5405 Magnatune songs; 188 tags; 270 artists.
- Audio in a three-part zip (1.10 + 1.10 + 0.77 GB = **2.97 GB**, from the HTTP
  headers). The first MP3 in the archive is **MPEG-2 Layer III, 16 kHz, 32 kbps,
  mono, about 29.3 s, 117 KB** (header decompressed through an HTTP Range
  request; the server supports ranges, response 206).
- **Licence:** the page **does not state one**. MARBLE marks MTT as
  non-commercial. `[I]` Academic use is well established (MARBLE, JukeMIR and
  hundreds of papers), but the report should say "academic use, no explicit
  licence".

**Top-50 tags** `[V]` (`annotations_final.csv`, computed here): 21 111 of the
25 863 clips carry at least one of the 50 most frequent tags (guitar, classical,
slow, techno, strings, drums, …). Castellon notes that MTT is biased towards
instrumentation. Used as pairwise relevance (Jaccard or cosine over tag vectors)
it would be another proxy, annotated by players, and very sparse — see the tie
rate below.

**Human similarity judgements: they exist** `[V]`. `comparisons_final.csv`
(132 KB) records the bonus mode of TagATune: two players hear three clips and
vote independently for the most different one. Computed here from the original
CSV (Appendix A):

| Quantity | Value |
|---|---|
| Rows (triplet presentations) | **533** |
| Votes | **7650** (mean 14.4 per row, median 4, range 1–153) |
| Distinct triplets (as sets) | 346 |
| Distinct clips | **1019**, spread over all 16 directories of the archive: all three parts are needed |
| Rows where the top choice ties | 87 |
| Constraints (2 per vote) | 15 300 edges, 1598 unique |
| **Consistent constraints** (after subtracting opposing votes) | **860** (weight 6898), over **337 triplets** and **993 clips** |
| Vote margin per constraint | median 3; **218 (25 %) with margin 1**; 642 with margin ≥ 2 |
| Triplets with 2+ clips by one artist / from one song | 10 / 1 |

These figures **reproduce Wolff et al. (2012) exactly**: 15 300 → 1598 → 860
constraints, 6898 edges, 337 components.

- **Inter-annotator agreement** `[V, computed here]`: the most-voted clip takes
  **70.7 %** of the votes of its row on average (median 66.7 %; chance 33.3 %).
  The probability that two random voters agree (Simpson index) is **0.48**
  against 0.33 by chance. Agreement is clear but far from unanimous; `[I]` the
  ceiling for any model sits well below 100 %.
- **How the literature uses it** `[V]`: each vote for the outlier *k* of a
  triplet *(i, j, k)* gives d(i,j) < d(i,k) and d(j,i) < d(j,k). Two-cycles are
  removed by subtracting opposing votes (Wolff et al. 2012, §3.2). The metric is
  the **percentage of satisfied constraints**. Trained methods use 10-fold cross
  validation, either by constraint (*sampling A*, 774/86) or **by triplet**
  (*sampling B*, no clip shared between train and test).
- **Reference figures** `[V]`:

| Method | Constraints satisfied | Source |
|---|---|---|
| Chance | 50 % | by construction |
| Euclidean over EchoNest | 59.8 % | Wolff et al. 2015, via Karamanolakis Table 2 |
| Random facet weights (EchoNest + tags) | 63 % | Wolff et al. 2012, Fig. 2 |
| Bag of audio words over **MFCC+Δ+ΔΔ** (not trained on the triplets) | 63.6–64.6 % | Karamanolakis Table 3 |
| Metric learned on the triplets (MLR, SVM, …) | 69.5–75.6 % | Wolff et al. 2012, Fig. 2 |
| Audio plus tags (FUSION) | 73.1 % | Karamanolakis Table 3 |

- **Semantic ceiling, computed here without audio** `[V]`, with 95 % bootstrap
  intervals clustered **by triplet**:

| Similarity | Accuracy | CI | n | Ties |
|---|---|---|---|---|
| Jaccard over the 188 tags | 0.570 | [0.546; 0.593] | 812 | **565 (70 %)** |
| Jaccard over the top 50 | 0.571 | [0.548; 0.594] | 799 | 558 |
| Cosine over the top 50 | 0.586 | [0.550; 0.618] | 490 | 302 |
| **"Same genre"** (30 genre tags), when it decides | **0.901** | [0.822; 0.961] | **101 of 860 (12 %)** | — |

  Ties count as 0.5. `[I]` The tags tie on 70 % of the constraints out of sheer
  sparsity; where they do not tie they are right about 73 % of the time. The
  genre proxy is nearly always right when it takes a side, but **it takes a side
  in only one constraint out of eight**. The list of genre tags was chosen by
  hand here; the implementation has to fix it and document it.

- **Does it work as the main perceptual ground truth? Yes, with two caveats.**
  `[I]` In favour: these are direct human judgements of similarity, the published
  protocol reproduces figure for figure, the metric depends on neither genres nor
  the gallery, and the cost is minimal — 993 clips, about 115 MB if only those
  are extracted, roughly 2.5 minutes of MFCC extraction. Against: (1) **low
  statistical power**, since 860 constraints over 337 triplets give an interval of
  ±3 to 5 points, enough to separate MFCC (~0.63) from chance and from a strong
  model (~0.70) but not two close variants such as PCA 32 against PCA 48; and (2)
  a **different domain** (Magnatune leans heavily towards classical, new age and
  electronic music, and the triplets "vary widely in genre") with an **8 kHz
  bandwidth** (16 kHz MP3) against FMA at 22 050 Hz.

### 3.2 MTG-Jamendo

`[V]` (the [repository](https://github.com/MTG/mtg-jamendo-dataset); statistics
computed from its split-0 TSV files):

- 55 525 **whole tracks**, not clips: 3764 hours, mean 244 s, median 224 s.
  Split-0 is 32 859 / 11 101 / 11 565 with 2139 / 713 / 713 artists and **no
  artist shared between train and test**.
- Tags are **written by whoever uploaded the music**: genre on 55 094 tracks (87
  tags, 2.44 per track), instrument on 24 976 (40, 2.57), mood/theme on 17 982
  (56, 1.77). Top-50 covers 54 380. Only **2325 test tracks carry all three
  facets**.
- Download: 508 GB (320 kbps), **156 GB** (mono VBR), 229 GB (mel-spectrograms),
  46 GB (mono audio of the emotion subset only). The audio ships as **100 tar
  files** per variant (the `00` one is 1.66 GB) and **each tar holds about 116
  test tracks** (92–138), so the test set cannot be downloaded on its own.
- Licences: metadata CC BY-NC-SA 4.0, audio CC per track, code Apache-2.0.
- **Verdict:** expensive for what it adds. It would allow the question "which
  facets does each representation capture?", but with uploader tags rather than
  perception.

### 3.3 Other MARBLE tasks as similarity facets

| Task | Does it give a useful notion of similarity? `[I]` |
|---|---|
| **EmoMusic (V/A)** | **Yes, as a human facet**: 744 songs, 45 s excerpts, valence and arousal on a 9-point scale. Relevance would be distance in the V/A plane. Small, access by form, licence unstated. It also comes from the Free Music Archive (3.4). |
| GiantSteps key / tempo | Objective facets, EDM only; "same key" is not "sounds alike", and an algorithm computes it just as well. No. |
| GTZAN genre | The same proxy already in hand, with known label errors and no commercial licence. No. |
| Beat, chords, melody, pitch, vocal technique, singer, separation | Frame- or note-level, not relations between songs. No. |
| Instrument (NSynth / MTG) | NSynth is isolated notes; MTG is uploader tags (3.2). No. |

### 3.4 Does any MARBLE dataset overlap with FMA?

- `[V]` **No.** MTT comes from Magnatune, MTG-Jamendo from Jamendo, GTZAN from CD
  and radio, and so on.
- `[V]` **One caveat: EmoMusic** states that "1000 songs has been selected from
  Free Music Archive (FMA)" — the same *website* the FMA dataset of Defferrard et
  al. is drawn from. After removing redundancies, 744 songs remain. `[I]` It does
  not carry FMA `track_id`s, so crossing it with `fma_small` would need the access
  form and a match on artist and title. By chance few matches would be expected
  (`fma_small` is 8000 of some 106 000 tracks), and V/A is a single facet. **Not
  verified; not recommended as a main route.**
- **Consequence:** a ground truth from MARBLE **adds a second corpus**. With MTT
  it would be **evaluation only**: nothing trained, 993 clips and one new index.

### 3.5 Contrast inside FMA, without leaving it

Computed from `tracks.csv`, `genres.csv` and `echonest.csv` over the 7994 tracks
of the index (Appendix A):

- **`genres_all` and the hierarchy** `[V]`: 114 distinct genres in `small`, depth
  at most 4. **5857 tracks** have at least one sub-genre, 2137 have only the root,
  **none** has more than one root, and the root of `genres_all` agrees with
  `genre_top` for **7994 of 7994**. Among tracks sharing `genre_top`, only
  **21.6 %** share any sub-genre (mean sub-genre Jaccard 0.142), with wide
  variation: Instrumental 52.9 %, Rock 29.0 %, Electronic 24.4 %, Pop 18.8 %,
  Hip-Hop 9.3 %.
  - Proposed graded gain `[I]`: `g = 1[same root] · (1 + Jaccard(sub-genres))`
    ∈ [0, 2], with linear-gain nDCG. P@k, R@k and MAP stay binary, since the root
    is `genre_top`.
  - Cost: no download and no extraction; evaluation only.
  - **Limit:** it is still artist-chosen genre metadata, that is, **the same
    proxy, at a finer grain**. It does not answer "does it sound alike?".
- **Echonest** `[V]`: only **1294 of 7994** tracks (16 %), badly unbalanced (Folk
  290, Hip-Hop 244, Rock 234, Pop 202, Electronic 147, International 108,
  Instrumental 52, **Experimental 17**), 496 artists, splits 987/118/189. Its
  eight `audio_features` (acousticness, danceability, energy, instrumentalness,
  liveness, speechiness, tempo, valence) were **computed by an algorithm from the
  audio**. `[I]` Using them as ground truth compares one model against another
  (circular) over a biased subset. **No.**
- **Same artist** `[V/I]`: already in the index. It is the relevance used by
  Cleveland et al. (2020) on FMA. It confounds timbre with album production, and
  it is already reported as the `artist_filter` variant.
- **FMA user tags** `[V]`: only 1358 tracks carry any. No.

---

## 4. How it fits the code (design only)

### 4.1 The dataset as a second dimension of the paths

The corpus becomes a dimension of the naming convention, with `fma` as the
default. Every path function takes the dataset, so a model stays "just a name":
`mfcc` on MTT is the same extractor applied to different clips. In the new
repository this is `musicsim/paths.py` and the `--dataset` option.

**Minimum common index:** `item_id, path, artist_id, split`, plus the columns a
corpus brings of its own (`genre_top` for FMA). An MTT builder produces:

- the index of the 993 clips that carry a constraint, with `item_id` = clip id,
  `artist_id` factorised from `clip_info_final.csv`, and `split` = test;
- `triplet_constraints.csv` with `anchor, pos, neg, margin, triplet_id`, derived
  as in Wolff et al. and asserting 15 300 / 1598 / 860 / 6898 / 337 / 993.

Extracting only the needed MP3s from the three-part archive: the simple option is
to download all three parts (2.97 GB) and read the concatenation with `zipfile`.
The alternative is to read the central directory over HTTP Range and fetch only
about 115 MB, which is more code and more fragile.

**Feature extraction does not change**: 22 050 Hz (MTT is resampled up from
16 kHz, with nothing above 8 kHz) and a fixed length of 30 s (MTT clips, at about
29.1 s, gain roughly 0.9 s of zero padding). This has to be measured and
reported; it is a documented limitation, not a change of extractor.

**Scaling:** the z-score is fitted on `split == "training"`, and MTT has none. The
choice is explicit: for MTT, fit on all 993 clips — **no labels are involved, so
nothing leaks**. `[I]` This is preferable to the FMA scaler because the missing
high band would shift every MTT vector uniformly, and cosine similarity is not
invariant to that shift. The FMA-scaler variant stays as a check.

### 4.2 Retrieval: from "same genre" to a relevance function

Abstract relevance as `gains(rows, ranked) → (Q, K)` plus `ideal_gains(rows)`,
the best K gains each query could obtain in the filtered gallery.

- **Binary "genre"**: exactly what it is today. The existing retrieval CSV must
  come out byte for byte identical; that is the acceptance test of the refactor.
- **Graded "genres_all"**: the gain of 3.5. Graded nDCG is
  `Σ gᵢ / log₂(i+1) / IDCG`, with the IDCG taken over the K largest gains the
  query could reach in the whole filtered gallery. Computed in query blocks (512
  × 7994 float32 ≈ 16 MB) against a sparse multi-hot sub-genre matrix. P, R and
  MAP stay binary on the root.

Everything else in the protocol is kept: the full gallery, `queries` in
{all, test}, the dedup / raw / artist_filter variants, intervals by query and by
artist, paired bootstrap, and the random control.

### 4.3 Triplet agreement (MTT)

For each constraint (s, p, n), a hit is `cos(s,p) > cos(s,n)`, ties at 0.5. The
headline metric is the **unweighted mean over the 860 constraints**, as in the
literature, plus a **vote-weighted** variant and one restricted to **margin ≥ 2**
(642 constraints). 95 % bootstrap intervals cluster **by triplet** (337
clusters), and model differences use the paired version with the same clusters.

Reference rows for the table: chance 0.5 and a random embedding as the negative
control; tag Jaccard about 0.57; "same genre when it decides" about 0.90 over
101 constraints; and the literature figures of 3.1.

---

## 5. Value for the report

| Option | Research question it enables |
|---|---|
| **MTT triplets** | **Q1. Does the genre proxy measure the same thing as perceived similarity?** (a) Without models: how far the genre proxy agrees with humans — 0.90 when it decides, but it decides in 12 % of cases. (b) With models: do the conclusions drawn from genre survive when the same models are ranked by human judgements? With five or six models a rank correlation is only descriptive; what is solid is the paired differences with intervals, in both evaluations. |
| **Graded `genres_all`** | **Q2. Does the diagnosis change when relevance is refined inside the genre?** For instance: do the neighbours of a Pop track, which look random under binary genre, at least share a sub-genre? Does MFCC order well inside a genre, or only between genres? |
| MTG-Jamendo facets | Q3. Which facets does each representation capture? Interesting, but uploader tags at a disproportionate cost. |
| MARBLE probing | Q4. What does the representation contain linearly? Complements, but does not answer, the similarity question. Useful once there is a CNN. |

---

## 6. Comparison

| Option | Annotation | Useful size | Licence | Download / CPU | Value | Risks |
|---|---|---|---|---|---|---|
| **A. MTT human triplets** | Odd-one-out votes by players (human, perceptual) | 860 constraints, 337 triplets, 993 clips | None explicit; academic use; MARBLE: non-commercial | 132 KB CSV + audio 2.97 GB (or ~115 MB by Range) / ~2.5 min | **High**: the only perceptual ground truth; answers Q1 | Low power (±3–5 pt); Magnatune domain; 16 kHz audio; a second corpus |
| **B. Graded `genres_all` (FMA)** | Artist-chosen genre hierarchy | 7994 tracks, 114 genres | CC BY 4.0 (metadata) | 0 / 0 | Medium: Q2, same corpus | Still a genre proxy |
| C. MTT tags (top 50) | Game tags | 21 111 tagged clips | As A | 2.97 GB / ~65 min | Low–medium | Very sparse (70 % ties); a proxy |
| D. MTG-Jamendo facets | Uploader tags | 11 565 test (2325 with all three facets) | Metadata CC BY-NC-SA 4.0; audio CC | ~166 GB / 0.5–19 h | Medium (Q3) | Disproportionate cost; a proxy; whole tracks |
| E. EmoMusic V/A | Annotator valence/arousal | 744 songs | Unstated (access form) | Small / minutes | Low–medium (one facet) | Access form; emotion only |
| F. MARBLE probing | Per-task class labels | MTT 25 863, EMO 744, … | Mixed; v2 code unlicensed | 2.97 GB+ / ~65 min + probes | Low today; medium with a CNN | Not similarity; MERT out of scope |
| G. Echonest (FMA) | Algorithmic descriptors | 1294 biased tracks | CC BY 4.0 | 0 / 0 | Low | Circular (model against model) |

---

## 7. Recommendation

- **Primary: A, the MTT human triplets, as a perceptual evaluation and for
  evaluation only.** It is the only option that turns an acknowledged limitation
  of the project — "genre is a poor proxy" — into a **measurement**: how often
  model and humans agree, and whether the conclusions drawn from genre hold up. It
  is cheap (minutes of CPU, 3 GB or 115 MB of disk), it reproduces a published
  protocol figure by figure, and it needs neither MERT nor a GPU nor any training.
  - **Condition:** present it honestly as a **small** set, with intervals
    clustered by triplet, good for large differences and not for fine-tuning
    between close variants.
- **Secondary: B, graded relevance over `genres_all`.** Almost free, and it
  **forces the generalisation of the retrieval code that is needed anyway**
  (relevance as a function, graded nDCG). It gives a second reading, inside FMA,
  of where the MFCC baseline fails.
- **Out of scope:** MTG-Jamendo (about 166 GB and whole tracks for another
  metadata proxy); MARBLE as a probing benchmark (does not measure similarity;
  Castellon already publishes the MFCC figure; revisit once there is a CNN); MTT
  tags as relevance (too sparse); EmoMusic (access form, single facet); Echonest
  (circular and biased).

---

## 8. Open questions for the supervisor

1. **Is a second corpus (MTT), for evaluation only, acceptable?** The alternative
   without one is option B alone.
2. **Full MTT download (2.97 GB) or HTTP Range extraction of only the ~1000
   clips (~115 MB, more code and more fragile)?** The full download leaves the
   door open to use 2.a once there is a CNN.
3. **Headline triplet metric:** unweighted over the 860 constraints (comparable
   with the literature) or vote-weighted? The proposal is the former as headline
   and the others as variants.
4. **Scaling on MTT:** z-score fitted on the MTT clips themselves (no labels
   involved) or the FMA scaler? The proposal is MTT as primary and FMA as a check.
5. **Graded `genres_all` gain:** is `1[same root]·(1 + Jaccard of sub-genres)`
   acceptable, or is a gain based on the depth of the common ancestor preferred?
6. **Does Q1 work as the research question of the report?** It decides how much
   weight the experiments give it.
7. **MTT licence:** is "academic use, no explicit licence" acceptable wording?

---

## 9. Corrections to what was assumed before this study

- MARBLE: the authors, the venue and the m-a-p group are **correct**. But the
  current version is **v2 (2025-06-04)**, which is a framework rather than a new
  paper, and the web leaderboard does not respond.
- **"Linear / MLP probe protocol"**: in v1 *constrained*, the probe is a
  **single-layer 512-unit MLP**; the linear probe is Castellon's, not MARBLE's.
- **MARBLE has no similarity task and no MFCC baseline** (1.4).
- **The MTT triplets exist**, but they are **not part of MARBLE**, which uses only
  the MTT top-50 tags.
- **MTG-Jamendo is not clips:** they are whole tracks (mean 244 s), and its tags
  are written by uploaders, not annotators.
- **FMA Echonest** is not a human annotation and covers only 16 % of `fma_small`.

---

## References

See [`sources.md`](sources.md) for the annotated list and the BibTeX keys. Direct
links used in this study:

- MARBLE: <https://arxiv.org/abs/2306.10548> ·
  <https://github.com/a43992899/MARBLE> · leaderboard (unreachable on
  2026-09-11): <https://marble-bm.shef.ac.uk>
- Castellon et al. (2021): <https://arxiv.org/abs/2107.05677> ·
  <https://github.com/p-lambda/jukemir>
- MERT: <https://arxiv.org/abs/2306.00107>
- Law et al. (2009): <https://archives.ismir.net/ismir2009/paper/000019.pdf>
- MagnaTagATune: <https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset> ·
  <https://mirg.city.ac.uk/datasets/magnatagatune/comparisons_final.csv>
- Wolff et al. (2012):
  <https://openaccess.city.ac.uk/id/eprint/2963/1/ismir2012.pdf> · code and
  folds:
  <https://mirg.city.ac.uk/codeapps/music-similarity-adaptation-wolffstober-ismir-2012>
- Karamanolakis et al. (2016): <https://arxiv.org/abs/1612.08391>
- Lee et al. (2020): <https://arxiv.org/abs/2008.03720> ·
  <https://jongpillee.github.io/multi-dim-music-sim/>
- Cleveland et al. (2020): <https://arxiv.org/abs/2008.04938>
- Bogdanov et al. (2019): <https://github.com/MTG/mtg-jamendo-dataset>
- EmoMusic: <https://cvml.unige.ch/databases/emoMusic/>
- FMA: <https://github.com/mdeff/fma>
- Cited second-hand, through Wolff 2012 or Karamanolakis 2016, and not read here:
  Law and von Ahn (2009), CHI; Stober and Nürnberger (2011), AMR; McFee and
  Lanckriet (2010), ICML; Ellis et al. (2002), ISMIR; Logan et al. (2003); Wolff
  et al. (2015), ISMIR.

---

## Appendix A. How the computed figures were obtained

Read-only scripts run against copies in the session temporary directory, outside
the repository. Nothing was written to any data or results directory.

- **FMA:** the track index plus `fma_metadata/{tracks,genres,echonest}.csv`:
  Echonest coverage, `genres_all` statistics, root against `genre_top`, and
  sub-genre overlap over 500 random queries (seed 42).
- **MTT:** `comparisons_final.csv`, `annotations_final.csv` and
  `clip_info_final.csv` downloaded from the official page (30 MB in total); the
  header of the first MP3 obtained with a 64 KB Range request over `mp3.zip.001`
  and inflated.
- **MTG-Jamendo:** the split-0 TSV files and
  `raw_30s_audio-low_sha256_tars.txt` from the repository; the size of tar `00`
  from its HTTP header.

Constraint derivation, reproducing Wolff et al. (2012):

```python
edges = Counter()
for i, j, k, vi, vj, vk in rows:               # clip ids and votes of each row
    for (a, b, o), v in (((j, k, i), vi), ((i, k, j), vj), ((i, j, k), vk)):
        if v:                                  # v votes for the outlier o
            edges[(a, b, o)] += v              # anchor a: d(a,b) < d(a,o)
            edges[(b, a, o)] += v              # anchor b: d(b,a) < d(b,o)
cons = {(s, p, n): w - edges.get((s, n, p), 0)  # opposing votes are subtracted
        for (s, p, n), w in edges.items() if w > edges.get((s, n, p), 0)}
# -> 15300 edges, 1598 unique, 860 constraints, weight 6898, 337 triplets, 993 clips
```
