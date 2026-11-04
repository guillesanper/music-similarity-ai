# Redistributing the MERT layer cache

Extracting MERT for all of `fma_small` takes about 27 minutes on a GPU and
about 9 hours on a CPU (see the MERT section of the [README](../README.md)).
Publishing the cache as a download, rather than committing it, would spare
anyone reproducing the results that cost. This document records what would be
published, under which licences the inputs are, and the question that has to
be answered before anything is published. **Nothing has been published yet.**

It is a record of the facts and of the open question, not a legal opinion.

---

## 1. What would be redistributed

One file per track, `<track_id>.npy` (six digits, zero-padded), in the cache
directory `mert-layers-22171214/`:

| Item | Value |
|---|---|
| Content per track | the 13 hidden states of MERT-v1-95M (the convolutional feature extractor's output and the 12 transformer layers), each mean-pooled over time and over the six 5 s windows of the clip |
| Array | `13 × 768`, `float16` |
| Tracks | 7994, the working index of `fma_small` (8000 minus the six unusable excerpts listed in [`datasets/fma/README.md`](../datasets/fma/README.md)) |
| Size | about 158 MB on disk, 148 681 433 bytes as `mert-layers-22171214-fma_small.tgz` |

**What is not in it:** no audio, no spectrograms, no model weights and no
metadata beyond the track id in the file name. Audio cannot be reconstructed
from a 768-dimensional average over 30 s. The titular layer is not part of the
cache; it is chosen in `configs/extractors/mert.yaml`.

`22171214` is the hash of the configuration that produced the arrays (the
`audio` block, and the model name, revision, window length, number of layers
and hidden size). A different configuration looks in a different directory, so
it never reads these arrays by mistake (`_mert_layers_cache_dir` in
`musicsim/extractors/mert.py`).

## 2. Licences of the audio (FMA small)

The FMA code is MIT and its metadata CC BY 4.0, but every track keeps the
licence its artist chose (`Defferrard2017`). The column `('track', 'license')`
of `tracks.csv` holds 63 distinct strings across the 7994 indexed tracks (the
same Creative Commons licence appears under several versions and ports). This
table groups them by licence family. The count was made directly from
`tracks.csv` and `datasets/fma/index.csv`.

| Family | Tracks | % | Most frequent strings |
|---|---:|---:|---|
| BY-NC-SA | 2883 | 36.1 | `Attribution-Noncommercial-Share Alike 3.0 United States` (1233), `…ShareAlike 3.0 International` (912), `Attribution-NonCommercial-ShareAlike` (607) |
| BY-NC-ND | 2738 | 34.3 | `…NoDerivatives (aka Music Sharing) 3.0 International` (998), `Creative Commons Attribution-NonCommercial-NoDerivatives 4.0` (967), `…No Derivative Works 3.0 United States` (694); includes one track tagged only `Music Sharing` |
| BY-NC | 927 | 11.6 | `Attribution-NonCommercial` (473), `…3.0 International` (399) |
| BY | 893 | 11.2 | `Attribution` (547), `Creative Commons Attribution` (221), `Attribution 3.0 United States` (101) |
| BY-SA | 369 | 4.6 | `Attribution-ShareAlike` (182), `…3.0 International` (155) |
| BY-ND | 50 | 0.6 | `Attribution-NoDerivatives 3.0 International` (22), `…4.0 International` (18) |
| Other: CC0 / public domain | 112 | 1.4 | `CC0 1.0 Universal` (54), `Public Domain` (40), `Public Domain Mark 1.0` (18) |
| Other: Sampling Plus | 11 | 0.1 | `Noncommercial Sampling Plus` (8), `Sampling Plus` (3) |
| Other: Free Art Licence | 6 | 0.1 | `Art Libre` (6) |
| Other: none recorded | 5 | 0.1 | empty field |
| **Total** | **7994** | 100 | |

**2788 of the 7994 tracks** (BY-NC-ND plus BY-ND, 34.9 %) carry a
no-derivatives clause, of which 2237 are in training, 221 in validation and
**330 in the 800-track `fma_test` gallery**.

What each family means for **data derived from the audio**, such as these
arrays. The deciding question is whether a vector of averaged activations
counts as an *adaptation* of the track, which the CC licences define as
material that modifies the work in a way that needs the copyright holder's
permission.

- **BY.** Adaptations may be shared with attribution. No obstacle.
- **BY-SA.** Adaptations may be shared, but under the same or a compatible
  licence. Whether that applies to a feature vector depends on the adaptation
  question; if it does, the arrays of these tracks would carry a BY-SA notice.
- **BY-NC.** Like BY, for non-commercial use only. A research thesis is
  non-commercial.
- **BY-NC-SA.** Both conditions: non-commercial, and share-alike if the arrays
  are adaptations.
- **BY-ND and BY-NC-ND.** Adapted material may not be shared. If the arrays are
  adaptations, these tracks cannot be published; if they are not, they can, like
  BY or BY-NC. **This is the doubtful case**, and this document does not answer
  it. Numeric descriptors of FMA tracks are commonly shared: FMA itself ships
  `features.csv` and `echonest.csv` for every track, ND included, under the
  metadata's CC BY 4.0. But that is a choice of the dataset's authors, not a
  ruling on the licences.
- **Other.** CC0 and public-domain tracks have no conditions. Sampling Plus (a
  retired CC licence) allows transformation, and the Noncommercial variant
  restricts it to non-commercial use. The Free Art Licence is a copyleft licence
  similar to BY-SA. The five tracks with no licence recorded have unknown terms
  and are as doubtful as the ND ones.

## 3. Licence of the model (MERT-v1-95M)

The model card of [`m-a-p/MERT-v1-95M`](https://huggingface.co/m-a-p/MERT-v1-95M)
declares **CC-BY-NC-4.0** (`Li2023`); this was checked on the model card. The
licence covers the weights. The card says nothing about the model's outputs or
about the licence of its training data. Using an NC model in non-commercial
research is within its terms. Sharing its outputs for the same non-commercial
research purpose, with attribution to MERT, is the reading closest to the
licence, so any release would state the non-commercial restriction and cite
both MERT and FMA.

## 4. Provisional conclusion and the open question

Three options:

1. **Publish everything** (7994 tracks). The download fully replaces the
   extraction. This relies on the arrays not being adaptations of the ND
   tracks.
2. **Publish everything except the ND tracks** (and the five with no licence
   recorded): 5201 tracks. This does not depend on how the adaptation question
   is answered. The download is still useful: the extractor fills an incomplete
   cache directory track by track, so a user would extract only the 2793
   missing tracks (about 35 % of the cost, including 330 of the 800 `fma_test`
   tracks).
3. **Publish nothing.** Anyone reproducing extracts MERT themselves. The
   cost is the time given in the README; the results do not change.

**Provisional conclusion:** option 2 is the one that does not depend on the
open question, and option 1 is the one that is useful. Either way the release
is non-commercial and cites FMA and MERT. The choice is not ours to make alone.

**Question for the tutor:** *the cache holds, per FMA track, 13 × 768 averaged
MERT activations, with no audio and no weights. 2788 of the 7994 tracks are
under no-derivatives licences (BY-ND or BY-NC-ND) and five have no licence
recorded. Should we publish the cache for every track, for every track except
those 2793, or not at all?*

The archive is ready but **will not be published until that answer arrives**.
The README describes how to install it and gives a placeholder instead of a
link.
