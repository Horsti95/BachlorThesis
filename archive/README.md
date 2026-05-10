# archive/

Everything in this directory was used during thesis development but is **not** part of the active pipeline. Kept here for the audit trail (defense, supervisor questions, future reference). Nothing in `archive/` is imported by the runtime code in the repo root.

## Contents

### `scripts/`
One-shot scripts: figure generators, benchmark suites, machine-specific PowerShell drivers, and one-off maintenance utilities.

| File | Purpose |
|---|---|
| `generate_thesis_figures.py` | Master script: regenerates most thesis figures + LaTeX tables from the cached results in `../results/`. |
| `generate_distribution_figure.py` | Sleep-stage distribution figure (hardcoded BOAS counts, no data needed). |
| `generate_globalvsfold_figure.py` | Global-vs-per-fold ANOVA comparison figure. |
| `generate_missing_figures.py` | Confusion-matrix figure (uses model cache). |
| `generate_pipeline_figures.py` | Pipeline-overview + fingerprint-zoom architecture figures. |
| `generate_subject_variation_figure.py` | Per-subject sleep-stage variation figure (needs raw BOAS annotations). |
| `update_figures_pc1.py` | Re-renders the speedup-bar and crossover figures with PC1 rerun data. |
| `benchmark_cache_strategies.py` | Cache-strategy benchmark experiment. |
| `benchmark_global_vs_perfold.py` | ANOVA scope (global vs. per-fold) benchmark. |
| `run_thesis_benchmark.py` | Three thesis claim benchmarks (scaling / fingerprint / reproducibility). |
| `run_combo_cold_warm_suite.py` | Comprehensive cold/warm benchmark sweep across all extras. |
| `run_pagecache_comparison.py` | OS page-cache vs. application-cache comparison. |
| `run_5090_extras.ps1` | RTX 5090 machine driver: extra benchmarks + figure refresh. |
| `run_pc1_rerun.ps1` | PC1 (Intel desktop) rerun driver. |
| `fnn_size_probe.py` | One-shot FNN sizing utility. |
| `rebuild_feature_cache.py` | One-shot feature-cache rebuild helper. |
| `check_benchmark_status.py` | Status checker for in-progress benchmark sweeps. |

### `tests/`
Development verification scripts (not pytest fixtures, just standalone runners).

| File | Purpose |
|---|---|
| `test_cache_comprehensive.py` | End-to-end LOSO cache hit/miss check across 12 configs × 3 folds. |
| `test_imports.py` | Regression test guarding against the `cli_menu` → `interactive_menu` rename. |
| `test_loso_cache_fixes.py` | Verifies specific bug fixes in the LOSO cache (feature-mismatch, pickle, hit/miss). |
| `test_ram_cache_comparison.py` | RAM-resident vs. SSD model-cache prototype comparison. |

### `experiments/`
Standalone experiment directories.

| Directory | Purpose |
|---|---|
| `model_tryouts/` | **Not part of the thesis.** Compares classical / neural / DL models on the cached features. Has its own `TRYOUTS_README.txt`. |
| `testing/` | Mini cache-viability and RF-tree-sweep benchmarks. |
| `benchmarks_and_tests/` | Older PowerShell-driven benchmarks (data scaling, eco mode, feature size, n_jobs, RAM vs. SSD, SVM scaling, XGB/RF scaling) plus the original ANOVA-vs-MI feature-selection benchmark. |

### `docs/`
Old design notes, verification reports, presentation outlines, and thesis development metadata.

Notable files:
- `MASTER_IMPLEMENTATION_GUIDE.md` — original implementation guide for the caching layer.
- `BUG_FIX_VERIFICATION.md`, `CACHE_INVALIDATION_VERIFICATION.md`, `METHODOLOGY_VERIFICATION_REPORT.md`, `RAM_ANALYSIS.md`, `TEST_SUMMARY.md`, `VERIFICATION_REPORT.md` — one-shot verification reports written during development (audit trail).
- `Future_work_and_to_dos.md`, `CODE_REVIEW_FULL.md` — internal notes.
- `PRESENTATION_OUTLINE.md`, `SCQA_PRESENTATION.md` — defense / presentation prep.
- `PIPELINE_VISUALIZATION.txt`, `PROJECT_STRUCTURE.md`, `TWO_TIER_CACHE_EXPLAINED.md` — older internal docs.
- `thesis_CHANGE_PLAN_REVIEW.md`, `thesis_THESIS_STATUS.md`, `thesis_ToDoforClaude` — thesis-development metadata.
- Files prefixed `papierkorb_` were duplicates resolved from the old `papierkorb/` ("trash") directory.

### Stale supplementary docs
Files prefixed `markdowns_` were moved out of the repo-root `markdowns/` directory because they described the project at an earlier (pre-thesis-submission) state and would mislead a reader (e.g., "Stage 4 NOT YET IMPLEMENTED", outdated chapter list, planning notes). The currently-accurate equivalents are the rewritten root `README.md` and `thesis/README.md`.

## What is **not** here

- `markdowns/` (project documentation that's still useful) stays in the repo root.
- `thesis/` (LaTeX source) stays in the repo root.
- `results/` (thesis output data) stays in the repo root.
- All active pipeline modules and the three CLI entry points (`run_experiment.py`, `run_training.py`, `run_full_pipeline.py`) stay at the repo root.
