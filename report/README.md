# The report

The Bachelor's Thesis report, written in LaTeX with the official **TFGTeXiS**
template of the Facultad de Informática (UCM), reorganised into the directory
layout of this repository and switched to English.

The master file is `main.tex`.

---

## 1. Building it

There is no LaTeX installation on the development laptop, so **Overleaf is the
default route**, and a local build is the alternative.

### Overleaf

1. Zip the `report/` directory, or push this repository and use Overleaf's
   GitHub import.
2. Create a new project from the zip.
3. In *Menu → Settings*, set **Main document** to `main.tex` and **Compiler** to
   **pdfLaTeX**.
4. Compile. The first run needs three passes (pdfLaTeX, BibTeX, pdfLaTeX twice);
   Overleaf does this automatically.

Everything the project needs is inside `report/`: the template style files, the
crest images, the bibliography and the generated figures and tables. Nothing
outside that directory has to be uploaded.

### Locally, with MiKTeX or TeX Live

```powershell
winget install --id MiKTeX.MiKTeX -e     # once, on Windows
cd report
latexmk -pdf main.tex
```

`latexmk` resolves the BibTeX passes and the cross-references on its own; MiKTeX
downloads any missing package the first time. To clean up:

```powershell
latexmk -C
```

To build a single chapter while writing, uncomment the matching
`\compilaCapitulo{...}` line in `config.tex` — a full build of the whole report
is slow enough to break concentration.

---

## 2. Layout

| Path | What it holds |
|---|---|
| `main.tex` | Master file. Includes everything else, in order. |
| `constants.tex` | Title (Spanish and English), authors, supervisor, degree, academic year, repository URL. |
| `config.tex` | Release or draft mode, single-chapter builds. |
| `hyphenation.tex` | Hyphenation exceptions for the English domain terms. |
| `texis/` | Template style files. See section 4 before touching them. |
| `frontmatter/` | `cover.tex`, `resumen.tex` (Spanish), `abstract.tex` (English), `acknowledgements.tex`, `dedication.tex`, `bibliography.tex`, `end.tex`. |
| `sections/` | The nine chapters of the body, `01_` to `09_`. |
| `appendices/` | `a_reproducibility.tex`, `b_full_tables.tex`. |
| `figures/static/` | Hand-made diagrams and the university crest. |
| `figures/generated/` | Figures exported by the code. **Versioned on purpose.** |
| `tables/generated/` | LaTeX tables exported by the code. **Versioned on purpose.** |
| `bibliography.bib` | The references. Keys match `docs/sources.md`. |

Inside the document, a figure is included with the template macro
`\imagen{generated/<name>}` or `\imagen{static/<name>}`: the macro prefixes
`figures/`, so the path in the source is relative to that directory.

### Why `figures/generated/` is versioned

The general rule of this repository is that anything the code produces is not
committed. These two directories are the deliberate exception, and the reason is
practical: the report has to compile for a reader who has not downloaded 10 GB of
audio and has not re-run the experiments. `musicsim export` refreshes them from a
run directory:

```powershell
python -m musicsim export --run outputs/runs/e01_mfcc_baseline/<timestamp>_<hash>
```

Everything else LaTeX writes — `*.aux`, `*.bbl`, `*.log`, `*.toc`, `main.pdf` —
is ignored by git.

---

## 3. What the regulations require

The rules below come from the TFG regulations of the Facultad de Informática
(Junta de Facultad, 18/03/2024). They are what the board checks, so they are
repeated here rather than left in a separate document.

- **V.1** Normalised cover: title in Spanish **and** English, **both** authors,
  supervisor, academic year, and the full identification of the subject. The
  title and the supervisor must match what is published on the faculty TFG
  website.
- **V.2** Contents: table of contents; summary and at most **10 keywords**, in
  Spanish **and** English; introduction with background, objectives and work
  plan; results with a critical discussion and conclusions; bibliography.
- **V.3** At least **30 pages** counting **only** the introduction and the
  results, discussion and conclusions. That is 25 pages plus 5 for the second
  author. The cover, the contents, the summaries, the related work, the personal
  contributions, the appendices and the bibliography do **not** count.
- **V.4** The report may be written entirely in English; the title, the summary
  and the keywords still appear in both languages. That is why
  `frontmatter/resumen.tex` is the only Spanish text in the document.
- **V.5** This is a **two-author** project, so
  `sections/09_personal_contributions.tex` is **mandatory** and needs **at least
  two pages per author**, with clearly different content. Do not delete it: that
  instruction in the template applies to single-author projects only.
- **V.10** The repository URL goes in the report. It is defined once, as
  `\repositorio` in `constants.tex`.
- **V.11** Every figure, table, text fragment, dataset and third-party library
  that is not original must be cited, with its licence respected. A report that
  fails this is not allowed to be defended.
- **V.12** The grade goes on the cover of the **final version only**. The line
  that prints it is commented out in `frontmatter/cover.tex`.

---

## 4. Changes made to the template

The template is TFGTeXiS, downloaded from
<https://informatica.ucm.es/file/plantilla_tfg_latex?ver>. The style files under
`texis/` are the template's own and are distributed under the LaTeX Project
Public License; they are kept as they came, except for the six edits below,
which were needed to fit the template into this layout and into an
English-language report.

| File | Change | Why |
|---|---|---|
| `texis/TeXiS.sty` | The five `\include{TeXiS/...}` paths and the `\ProvidesPackage` name now read `texis/...`. | The directory is lowercase here. LaTeX paths are case-sensitive on Linux, which is what Overleaf runs. |
| `texis/TeXiS.sty` | `\imagen` now prefixes `figures/` instead of `Imagenes/`. | Figures live in `figures/` in this layout. |
| `texis/TeXiS_bib.tex` | Same path rewrite, and the bibliography style is now `texis/TeXiS_en` instead of `texis/TeXiS`. | The report is in English, and the template ships a separate English style. |
| `texis/TeXiS_pream.tex` | Same path rewrite, and babel is loaded as `[spanish,activeacute,english]` instead of `[english,activeacute,spanish]`. | In babel the last option is the main language. English becomes the main language, and Spanish stays available for the cover title and the resumen. |
| `texis/TeXiS_cover.tex` | The default cover image points at `figures/static/escudoUCM`. | Same reason as the `\imagen` change; `frontmatter/cover.tex` overrides it anyway. |

Structural changes made outside `texis/`, all of them renames or deletions of
template content rather than edits to the template machinery:

- `TFGTeXiS.tex` became `main.tex`, with `Cascaras/` → `frontmatter/`,
  `Capitulos/` → `sections/`, `Apendices/` → `appendices/`, `Imagenes/` →
  `figures/`, `constantes.tex` → `constants.tex` and `guionado.tex` →
  `hyphenation.tex`.
- The Spanish duplicates of the introduction and the conclusions
  (`Capitulos/Introduccion.tex`, `Capitulos/ConclusionesTrabajoFuturo.tex`) are
  gone: the report has a single version of each chapter, in English.
- The `otherlanguage{english}` wrapper around the abstract is gone, and an
  `otherlanguage{spanish}` wrapper around the resumen took its place, since the
  languages are now the other way round.
- `Capitulos/ContribucionesPersonales.tex` is **kept**, as
  `sections/09_personal_contributions.tex` (V.5).
- The Don Quijote quotations of `Cascaras/fin.tex` and the sample bibliography of
  `biblio.bib` are gone.
- `TeXiS-Manual-1.0.pdf` and `TFGTeXiS.pdf`, the template documentation, are not
  committed. They are in the downloadable zip if they are needed.

---

## 5. Checklist before handing anything in

1. Cover: Spanish and English titles, both authors, supervisor, academic year,
   full subject identification. Title and supervisor match the TFG website.
2. Resumen and abstract, each with at most 10 keywords.
3. Table of contents generated and correct.
4. Introduction covers background, objectives and work plan; the results are
   critically discussed and lead to conclusions; the bibliography is complete.
5. At least 30 pages counting only chapters 1 and 6 to 8.
6. Chapter 9 present, at least two pages per author, with different content.
7. `\repositorio` filled in and the repository reachable by the board.
8. Every third-party figure, dataset and library cited, licences respected.
9. No `\todo` notes left, and `\def\release{1}` uncommented in `config.tex`.
10. No red placeholder text left over from the template or from `constants.tex`.
11. It compiles with no errors and no `??` cross-references.
12. Draft sent to the supervisor before the date in the TFG calendar.
