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

Every command below runs on Linux, macOS and Windows. Where a platform needs a
different spelling, both are given.

---

## 1. Download

With the code, which resumes interrupted transfers and verifies the checksums:

```bash
musicsim download --dataset fma_small
```

By hand, if the code is not available yet. `curl` ships with macOS, most Linux
distributions and Windows 10+; on Windows PowerShell write `curl.exe`, because
`curl` there is an alias of `Invoke-WebRequest`:

```bash
mkdir -p datasets/fma/raw
curl -L -C - --retry 5 --retry-delay 10 -o datasets/fma/raw/fma_metadata.zip https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -L -C - --retry 5 --retry-delay 10 -o datasets/fma/raw/fma_small.zip    https://os.unil.cloud.switch.ch/fma/fma_small.zip
```

`-C -` resumes a partial download; the Unil mirror is slow at times. If it keeps
failing, the `mdeff/fma` README links alternative mirrors (archive.org, Zenodo, a
torrent).

## 2. Unpack

Python's `zipfile` works everywhere and needs nothing installed:

```bash
python -c "import zipfile; [zipfile.ZipFile(f'datasets/fma/raw/{n}.zip').extractall('datasets/fma/raw') for n in ('fma_metadata','fma_small')]"
```

Two platform notes:

- **Windows: `Expand-Archive` does not work on these archives.** Their entries
  use BZip2 compression, which the .NET implementation does not support; it
  fails with `The archive entry was compressed using BZip2 and is not
  supported`. Use the Python command above, or 7-Zip
  (`7z x datasets/fma/raw/fma_small.zip -odatasets/fma/raw -y`), which is faster
  for the large archive.
- **Linux, macOS:** `unzip datasets/fma/raw/fma_small.zip -d datasets/fma/raw`
  also works if `unzip` is installed with BZip2 support.

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

One portable command counts the excerpts, checks the metadata and decodes a
clip:

```bash
python -c "import pathlib,librosa; r=pathlib.Path('datasets/fma/raw'); print('mp3 files:', sum(1 for _ in r.glob('fma_small/*/*.mp3'))); print('tracks.csv:', (r/'fma_metadata/tracks.csv').is_file()); y,sr=librosa.load(r/'fma_small/000/000002.mp3', sr=22050); print('decoded:', y.shape, sr)"
```

It must print 8000 mp3 files, `tracks.csv: True` and a decoded shape of about
`(661504,) 22050`.

Once verified, the archives can be deleted to recover about 7.5 GB:

```bash
rm datasets/fma/raw/*.zip                  # Linux, macOS
Remove-Item datasets\fma\raw\*.zip         # Windows (PowerShell)
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
