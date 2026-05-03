# Bachelor Thesis: Sleep Stage Classification

LaTeX source files for the Bachelor thesis.

## Files

- `main.tex` - Main document (includes all chapters)
- `imc-inf.cls` - IMC thesis template class file
- `references.bib` - Bibliography database
- `chapters/` - Individual chapter files
  - `01_introduction.tex`
  - `02_related_work.tex`
  - `03_methodology.tex`
  - `04_implementation.tex`
  - `05_results.tex` (fill in after experiments)
  - `06_discussion.tex`
  - `07_conclusion.tex`
  - `appendix_a.tex`
- `figures/` - Figures and images

## Required: IMC Logo

**IMPORTANT:** The template requires `imc.png` in the `thesis/` directory for the title page.

You need to obtain the official IMC logo and place it here as `imc.png`.

## Compilation

Compile with:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

Or use latexmk for automatic compilation:

```bash
latexmk -pdf main.tex
```

## Metadata to Update

Before compiling, update the following in `main.tex`:

1. `\author{Your Name}` - Your name
2. `\supervisor{Supervisor Name}` - Your supervisor's name
3. `\submissiondate{Month Year}` - Submission date (e.g., "February 2026")
4. `\copyrightyear{2026}` - Year
5. Abstract content (replace placeholder text)
6. Acknowledgements (replace placeholder text)

## TODO Items

Throughout the `.tex` files, search for `TODO` comments indicating sections that need to be filled in:

- **Results chapter (05_results.tex)**: All results must be filled in after running experiments
- **Discussion chapter (06_discussion.tex)**: Analysis of results
- **References (references.bib)**: Replace `TODO` entries with actual citations

### After Running Experiments

1. Fill in results tables in Chapter 5
2. Add figures (confusion matrices, performance plots) to `figures/`
3. Update discussion based on actual results
4. Complete the abstract with actual findings
5. Add actual references to `references.bib`

## Figures

Add your figures (PDF, PNG, JPG) to the `figures/` directory and reference them with:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/your_figure.pdf}
    \caption{Figure caption}
    \label{fig:your_label}
\end{figure}
```

## Word Count

Estimated word count per chapter (current state):

- Introduction: ~1,200 words
- Related Work: ~1,500 words
- Methodology: ~2,000 words
- Implementation: ~2,500 words
- Results: ~1,500 words (to be filled)
- Discussion: ~2,000 words
- Conclusion: ~800 words
- **Total: ~12,000-15,000 words** (typical Bachelor thesis length)

## Tips

1. Compile frequently to catch LaTeX errors early
2. Use `\label{}` and `\ref{}` for cross-references (not hardcoded numbers)
3. All figures and tables should be referenced in the text
4. Keep line breaks in source for better git diffs
5. Use `%` for comments in LaTeX
6. Check for missing citations with `grep -r "TODO_ADD" chapters/`
