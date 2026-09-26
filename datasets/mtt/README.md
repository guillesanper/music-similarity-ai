# MagnaTagATune (MTT)

The second corpus, used for **evaluation only**: nothing is trained on it. It is
here for one reason no other corpus in this project provides — the odd-one-out
votes of the TagATune game are **direct human judgements of musical similarity**.

**Nothing in this directory is committed.** `datasets/mtt/raw/` is ignored by git.

| Item | Value |
|---|---|
| Source | <https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset> |
| Clips | 25 863, about 29 s each, from 5405 Magnatune songs |
| Tags | 188, by 270 artists |
| Audio | Three-part zip: 1.10 + 1.10 + 0.77 GB = about 2.97 GB |
| Annotation CSVs | About 30 MB in total |
| Audio format | MPEG-2 Layer III, **16 kHz**, 32 kbps, mono |
| Licence | **None is published.** Academic use only; see section 6. |

---

## 1. Download

```powershell
python -m musicsim download --dataset mtt
```

By hand:

```powershell
New-Item -ItemType Directory -Force datasets\mtt\raw
$base = "https://mirg.city.ac.uk/datasets/magnatagatune"
foreach ($f in "clip_info_final.csv","annotations_final.csv","comparisons_final.csv") {
  curl.exe -L -C - --retry 5 -o "datasets\mtt\raw\$f" "$base/$f"
}
foreach ($p in "001","002","003") {
  curl.exe -L -C - --retry 5 -o "datasets\mtt\raw\mp3.zip.$p" "$base/mp3.zip.$p"
}
```

The three `mp3.zip.00N` files are **parts of a single zip**, not three separate
archives. They have to be concatenated before they can be opened:

```powershell
cmd /c copy /b datasets\mtt\raw\mp3.zip.001+datasets\mtt\raw\mp3.zip.002+datasets\mtt\raw\mp3.zip.003 datasets\mtt\raw\mp3.zip
python -c "import zipfile; zipfile.ZipFile('datasets/mtt/raw/mp3.zip').extractall('datasets/mtt/raw/audio')"
```

On Linux or macOS, `cat mp3.zip.001 mp3.zip.002 mp3.zip.003 > mp3.zip`.

### Only the clips that are needed

The evaluation uses **993 clips**, roughly 115 MB, not the full 2.97 GB. The
clips that carry a similarity constraint are spread over all 16 directories of
the archive, so all three parts are still required. Downloading everything is
what this project does, because it keeps the door open to the tagging
experiments of the classical-classification milestone (stage S2) and avoids a
fragile partial-extraction path.

## 2. Expected layout

```
datasets/mtt/
└── raw/
    ├── clip_info_final.csv     clip id, song, artist, album, url
    ├── annotations_final.csv   188 binary tag columns per clip
    ├── comparisons_final.csv   the odd-one-out votes: 533 rows, 7650 votes
    └── audio/
        ├── 0/...mp3
        └── f/...mp3            16 directories, 0 to f
```

## 3. Verify

```powershell
(Get-ChildItem datasets\mtt\raw\audio -Recurse -Filter *.mp3).Count   # 25863
python -m musicsim index --dataset mtt
```

The index builder asserts the whole derivation chain, and fails loudly if any
count is off:

| Quantity | Expected |
|---|---|
| Rows in `comparisons_final.csv` | 533 |
| Votes | 7650 |
| Constraint edges (2 per vote) | 15 300 |
| Unique edges | 1598 |
| **Consistent constraints** | **860** (total weight 6898) |
| Triplets | 337 |
| Clips involved | 993 |

These reproduce the published counts of Wolff et al. (2012) exactly. An
assertion failure here means the CSV changed, and every triplet result in the
report would be invalid, so the failure is deliberate rather than a warning.

## 3b. The full index

`python -m musicsim index --dataset mtt` (stage S2) builds `index.csv` over
**all 25 863 clips**, not only the 993 involved in a triplet constraint, with
the standard split and a boolean `in_triplets` column:

```
datasets/mtt/index.csv   item_id, path, artist_id, split, in_triplets
```

The three uses of this index, each covering a different part of the corpus:

- **Human-judgement evaluation** (`mtt_triplets`, 993 clips, `in_triplets`):
  agreement with the constraints derived below (SQ4, stage S2, experiment e04)
  and the model-versus-human ambiguity analysis of the fuzzy block (F4, stage
  S2, experiment e11).
- **Multi-label genre membership** (all 25 863 clips, the tag columns of
  `annotations_final.csv`): ground truth for the fuzzy genre-membership
  evaluation (F3, stage S2).
- **Out-of-distribution gallery** (`mtt_all`, all 25 863 clips): no model is
  trained on MTT, so the whole index is the titular out-of-distribution
  gallery for structure, retrieval and fuzzy evaluation, distinct from
  `mtt_triplets`.

## 4. How the constraints are derived

Each row of `comparisons_final.csv` presents a triplet and records the votes of
the players. A vote for the outlier *k* of a triplet *(i, j, k)* means the other
two are more alike, and yields two constraints:

    d(i, j) < d(i, k)     and     d(j, i) < d(j, k)

Votes pointing in opposite directions cancel: for every ordered pair the opposing
weight is subtracted, and only the pairs with a positive remainder survive. That
removes the two-cycles and leaves the 860 consistent constraints.

`musicsim/datasets/mtt.py` implements this and writes
`triplet_constraints.csv` with `anchor, pos, neg, margin, triplet_id`. The margin
is the vote surplus, which the weighted variants of the metric use.

## 5. Limitations this corpus imposes

Both go in the report; neither is a reason not to use it.

- **Bandwidth.** The audio is 16 kHz mono MP3, so it carries nothing above
  8 kHz, against 22 050 Hz for FMA. Clips are resampled to 22 050 Hz so that a
  single extractor serves both corpora. This is a documented limitation, not a
  change of method.
- **Length.** The clips last about 29.1 s and are zero-padded to 30 s, about
  0.9 s of silence.
- **Domain.** The Magnatune catalogue leans towards classical, new age and
  electronic music.
- **Statistical power.** 860 constraints over 337 triplets give a confidence
  interval of roughly ±3 to 5 points. That separates a baseline from a strong
  model; it does not separate two close variants.
- **Scaling.** The standardiser is fitted on the MTT clips themselves rather than
  transferred from FMA. No labels are involved, so nothing leaks, and the missing
  high band would otherwise shift every MTT vector under the FMA scaler — a shift
  cosine similarity is not invariant to.

## 6. Licence

The official page **states no licence**, and MARBLE lists MTT as not
commercially available. Academic use is well established, and this project does
not redistribute any of it: only the download instructions and the code are here.
The report says "academic use, no explicit licence" rather than claiming a
licence that does not exist (V.11).

## 7. Citation

Law, E., West, K., Mandel, M., Bay, M., Downie, J. S. (2009). *Evaluation of
Algorithms Using Games: The Case of Music Tagging.* ISMIR. Key `Law2009`.

Wolff, D., Stober, S., Nürnberger, A., Weyde, T. (2012). *A Systematic Comparison
of Music Similarity Adaptation Approaches.* ISMIR. Key `Wolff2012`, the source of
the constraint derivation above.
