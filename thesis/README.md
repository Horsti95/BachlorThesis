# Bachelor Thesis (LaTeX source)

LaTeX source for the submitted Bachelor thesis. The PDF was submitted in May 2026; this directory exists for the audit trail.

## Files

- `main.tex` — main document, includes all chapters
- `imc-inf.cls` — IMC informatics thesis class
- `references.bib` — bibliography (BibLaTeX/biber)
- `imc.png`, `imclogo.png`, `imc_logo_print.jpg` — IMC logos used by the title page
- `chapters/`
  - `01_introduction.tex`
  - `02_background.tex`
  - `03_related_work.tex`
  - `04_methodology.tex`
  - `05_summary.tex`
  - `A_appendix.tex`
- `figures/` — final figures (PDFs); older variants in `figures/archive/`
- `tables/` — final result tables; older variants in `tables/archive/`
- `template/`, `template.tex`, `template.pdf` — original IMC template, kept for reference

## Compile

```bash
latexmk -pdf main.tex
```

or manually:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Provenance

Figures and tables in `figures/` and `tables/` are produced by the scripts archived under `../archive/scripts/` (`generate_thesis_figures.py`, `update_figures_pc1.py`, the various `generate_*_figure.py` files). They run against the thesis-run results in `../results/`.

Development metadata (status notes, change-plan reviews, todo lists) lives in `../archive/docs/` (files prefixed with `thesis_`).
