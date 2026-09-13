# FMA small

The primary corpus: 8000 excerpts of 30 seconds, 8 genres, 1000 excerpts each,
from the Free Music Archive dataset of Defferrard et al. (2017).

**Nothing in this directory is committed.** The repository holds the
instructions and the download code, not the bytes. `datasets/fma/raw/` is
ignored by git.

| Item | Value |
|---|---|
| Source | <https://github.com/mdeff/fma> |
| Audio archive | `fma_small.zip`, about 7.15 GB |
| Metadata archive | `fma_metadata.zip`, about 0.34 GB |
| Disk after unpacking | about 16 GB (the archives can be deleted afterwards) |
| Licence, code | MIT |
| Licence, metadata | CC BY 4.0 |
| Licence, audio | Per track, chosen by each artist |
| Checksums | Published in the `mdeff/fma` README and in `fma_metadata/checksums` |

---

## 1. Download

With the code:

```powershell
python -m musicsim download --dataset fma_small
```

That resumes interrupted transfers and verifies the checksums. To do it by hand
instead:

```powershell
New-Item -ItemType Directory -Force datasets\fma\raw
curl.exe -L -C - --retry 5 --retry-delay 10 -o datasets\fma\raw\fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl.exe -L -C - --retry 5 --retry-delay 10 -o datasets\fma\raw\fma_small.zip    https://os.unil.cloud.switch.ch/fma/fma_small.zip
```

`-C -` resumes a partial download; the Unil mirror is slow at times. If it keeps
failing, the `mdeff/fma` README links alternative mirrors (archive.org, Zenodo, a
torrent).

## 2. Unpack

**`Expand-Archive` does not work on these archives.** Their entries use BZip2
compression, which the .NET implementation does not support, and it fails with
`The archive entry was compressed using BZip2 and is not supported`.

Use the Python `zipfile` module, which does support it and needs nothing
installed:

```powershell
python -c "import zipfile; [zipfile.ZipFile(f'datasets/fma/raw/{n}.zip').extractall('datasets/fma/raw') for n in ('fma_metadata','fma_small')]"
```

7-Zip is faster for the large archive, if it is installed:

```powershell
& "C:\Program Files\7-Zip\7z.exe" x datasets\fma\raw\fma_metadata.zip -odatasets\fma\raw -y
& "C:\Program Files\7-Zip\7z.exe" x datasets\fma\raw\fma_small.zip    -odatasets\fma\raw -y
```

## 3. Expected layout

Exactly this, because `configs/datasets/fma_small.yaml` points at these paths:

```
datasets/fma/
└── raw/
    ├── fma_small/              156 sub-directories, 000/ to 155/
    │   ├── 000/000002.mp3
    │   └── ...
    └── fma_metadata/
        ├── tracks.csv          two-row header
        ├── genres.csv          the genre tree
        ├── features.csv
        ├── echonest.csv
        └── checksums
```

If the data lives on another disk, do not move it: point `MUSICSIM_DATA_DIR` at
the directory that holds `fma/`.

## 4. Verify

```powershell
Test-Path datasets\fma\raw\fma_small\000\000002.mp3                        # True
(Get-ChildItem datasets\fma\raw\fma_small -Recurse -Filter *.mp3).Count    # 8000
Test-Path datasets\fma\raw\fma_metadata\tracks.csv                         # True
python -c "import librosa; y,sr=librosa.load(r'datasets/fma/raw/fma_small/000/000002.mp3', sr=22050); print(y.shape, sr)"
```

The last command prints something like `(661504,) 22050`.

Once verified, the archives can be deleted to recover about 7.5 GB:

```powershell
Remove-Item datasets\fma\raw\*.zip
```

## 5. Things the code knows about this corpus

These are encoded in `musicsim/datasets/fma.py` and in
`configs/datasets/fma_small.yaml`; they are listed here so that a surprising
number has an explanation.

- **`tracks.csv` has a two-row header** and must be read with `header=[0, 1]`.
- **Six excerpts are excluded**, so the working index has **7994** items, not
  8000. Documented in the `mdeff/fma` wiki:
  - `99134`, `108925`, `133297`: roughly 1 KB, no decodable audio;
  - `98565`, `98567`, `98569`: they decode, but last far less than 30 seconds, so
    their pooled vectors would not be comparable with the rest.
- **The excerpts are not exactly 30 seconds** (track 2 decodes to 660 984
  samples). Every clip is trimmed or zero-padded to 661 500 samples.
- **The official split keeps artists apart.** The code asserts this rather than
  trusting it: a shared artist between training and test would inflate every
  retrieval figure.
- **The genre hierarchy is in `genres.csv`.** In the small subset there are 114
  distinct genres, depth at most 4, and the root of `genres_all` agrees with
  `genre_top` for all 7994 tracks.

## 6. Citation

Defferrard, M., Benzi, K., Vandergheynst, P., Bresson, X. (2017). *FMA: A Dataset
for Music Analysis.* ISMIR. BibTeX key `Defferrard2017` in
`report/bibliography.bib`.
