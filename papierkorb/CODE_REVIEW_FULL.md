# Full Code Review - BachlorThesis_V2

Date: 2026-04-06
Scope: All Python modules in workspace (31 files)
Reviewer mode: Bug/risk-focused code review with maintainability and experiment-validity checks

## Review Method
- Read-through and targeted inspection of core modules (`pipeline.py`, `training.py`, `feature_selection.py`, `benchmark_cache_strategies.py`, `loso_cache.py`, `models.py`, `config.py`)
- Pattern scan across all modules (bare except, broad except, hardcoded paths, TODO/deprecated markers)
- Static lint pass (`ruff check .`) for concrete lint/code-quality issues
- Cross-check against benchmark behavior and logging history from this session

## Executive Summary
The project architecture is strong (clear pipeline layering, LOSO design, two-level caching), and the benchmarking workflow has improved significantly during this session. The main remaining risks are:
1. Experiment-validity ambiguity around global feature selection default (intentional leakage/speed tradeoff, but dangerous as default)
2. Portability risk from many hardcoded Windows paths
3. Error-handling blind spots (one bare `except`, multiple broad `Exception` handlers)
4. Benchmark comparability confusion (RAM timing excludes preload by design; must be stated clearly everywhere)

## Confirmed High-Priority Findings

### 1) Data leakage risk by default configuration
- File: `feature_selection.py` (around `FeatureSelectionConfig.scope`)
- Issue: Default `scope='global'` fits feature selection on all data before LOSO folds.
- Risk: Can leak information from held-out subjects into feature ranking; may inflate metrics.
- Recommendation:
  - Option A: Change default to `scope='per_fold'`.
  - Option B: Keep `global`, but hard-fail unless user explicitly enables leakage mode for speed benchmarks.

### 2) Bare exception suppresses cleanup failures
- File: `benchmark_cache_strategies.py` (around line 486)
- Issue: `except: pass` in cache cleanup loop.
- Risk: Silent failures and hidden file-lock issues on Windows.
- Recommendation: Replace with `except OSError as e:` + warning/exception log containing filename.

### 3) Hardcoded absolute Windows paths across multiple modules
- Files: `config.py`, `run_training.py`, `run_experiment.py`, `pipeline.py`, `preprocessing.py`, `interactive_menu.py`, `data_loader_boas.py`
- Issue: Repeated `C:\Users\DerHo\Desktop\Data` defaults.
- Risk: Non-portable code; brittle on other machines.
- Recommendation: Centralize path resolution via env var (`DATA_PATH`) + config default + CLI override.

### 4) Abstract methods declared but no explicit NotImplementedError
- File: `models.py` (BaseModel `fit/predict/predict_proba`)
- Issue: Abstract methods use `pass`.
- Risk: Minor API clarity issue (ABC already enforces, but diagnostics less clear).
- Recommendation: Raise `NotImplementedError` for clearer failure messages in subclasses/tests.

### 5) Benchmark timing semantics need explicit labeling
- File: `benchmark_cache_strategies.py`
- Issue: Warm RAM reports time excluding preload (intended), while Warm SSD includes all run time.
- Risk: Misinterpretation if not clearly presented in all outputs/docs.
- Recommendation: Rename column to `Warm RAM (excl preload)` and optionally add `Warm RAM raw` in quick summary.

## Medium-Priority Findings

### 6) Inconsistent exception handling policy
- Files: `data_loader_boas.py`, `feature_extractor.py`, `pipeline.py`, `run_training.py`, `loso_cache.py`
- Issue: Mixed fail-fast and broad catches.
- Risk: Uneven observability and hard debugging.
- Recommendation: Define policy by layer:
  - data ingestion: fail-fast + rich context
  - batch operations: continue-with-errors + structured failure list

### 7) Deprecated/legacy flags still visible in active configs
- Files: `feature_selection.py`, `training.py`, `data_loader_boas.py`
- Issue: `use_hybrid` deprecated but still present.
- Risk: User confusion and config drift.
- Recommendation: Remove from dataclass in next minor version, keep migration shim in loader only.

### 8) Feature-count magic numbers repeated
- File: `config.py`
- Issue: Per-channel/global feature counts hardcoded in multiple places.
- Risk: silent mismatch if extractor changes.
- Recommendation: derive expected feature count from extractor metadata or single source constant.

### 9) Diagnostic printing mixed with logger in many modules
- Files: `pipeline.py`, `run_training.py`, `output_formatter.py`, tests
- Issue: Heavy `print()` usage alongside logging.
- Risk: ordering/interleaving issues in piped sessions.
- Recommendation: route through one structured output interface per mode (CLI formatter vs logger).

### 10) Bench/test scripts are large and noisy
- Files: `test_loso_cache.py`, benchmark scripts
- Issue: very verbose output, many inline prints.
- Risk: difficult CI diagnostics and slower local debugging.
- Recommendation: add verbosity flag and concise summary mode.

## Low-Priority / Hygiene Findings
- `feature_cache.py` module docstrings contain outdated TODO references that are already implemented.
- `check_benchmark_status.py` still uses non-ASCII status emojis; can break in cp1252 terminals.
- Some module-level imports in tests violate style (`ruff E402` in `test_loso_cache.py`).
- Several unnecessary f-strings in tests (`ruff F541`).

## Lint Snapshot (Ruff)
Observed primarily in tests:
- `E402` imports not at top of file (`test_loso_cache.py`)
- `F401` unused imports (`test_loso_cache.py`)
- `F541` f-strings without placeholders (`test_loso_cache.py`)

Recommendation: separate test diagnostics script behavior from test module imports, or move bootstrapping logic into a helper script.

## Per-Module Coverage Matrix (All Python Files)

### Root modules
- `__init__.py`: No major issues.
- `benchmark_cache_strategies.py`: Good structure; fix bare except; clarify RAM timing labels; add aggregate output for raw/preload-adjusted timing.
- `cache_visualization.py`: Large plotting utility; recommend splitting into smaller render/data prep units.
- `check_benchmark_status.py`: Functional; replace emoji output with ASCII for Windows compatibility.
- `config.py`: Strong dataclass design; remove hardcoded path defaults; reduce duplicated feature-count constants.
- `cross_validation.py`: Generally clean; consider explicit split-order docs (lexicographic subject behavior).
- `data_loader_boas.py`: Strong functionality; improve exception context and remove hardcoded fallback path in script sections.
- `evaluation.py`: Good metric handling; verify calibration/threshold assumptions are documented.
- `feature_cache.py`: Works, but docs include stale TODO references; consider stricter cache validation default.
- `feature_extractor.py`: Rich implementation; add stricter error context and optional profiling hooks.
- `feature_selection.py`: Key validity concern on global default scope; deprecated field cleanup needed.
- `fingerprint.py`: Core cache keying logic is valuable; add more explicit collision/debug utilities in API.
- `interactive_menu.py`: Useful UX, but many prints and hardcoded path defaults; extract config I/O helpers.
- `leaderboard.py`: Fine overall; verify tie-breaking and metric weighting are documented.
- `loso_cache.py`: Strong cache abstraction; review broad exceptions and registry failure visibility.
- `models.py`: Good wrappers; improve abstract base diagnostics and note deterministic settings per backend.
- `output_formatter.py`: Rich formatting; consider strict ASCII fallback mode toggle in config/CLI.
- `pipeline.py`: Central orchestrator is solid; reduce print/log mix and isolate long methods.
- `preprocessing.py`: Generally robust; remove hardcoded path in runnable section.
- `run_experiment.py`: Good CLI; path defaults should be centralized/env-driven.
- `run_training.py`: Good orchestration; path default and print/log consistency improvements.
- `training.py`: Core logic is strong; revisit global FS default and reduce deprecated config visibility.
- `utils.py`: No major critical issues found.
- `visualization.py`: Large plotting module; candidate for decomposition and shared style config.

### Benchmarks/Test subfolder modules
- `benchmarks_and_tests/benchmark_anova_vs_mi.py`: Useful benchmark script; add reproducibility seed logging and machine info dump.
- `benchmarks_and_tests/benchmark_feature_selection.py`: Good experiment utility; standardize output schema with main benchmark JSON.

### Test modules
- `test_cache_comprehensive.py`: Good coverage intent; reduce verbosity and add assertions for failure modes.
- `test_imports.py`: Useful smoke test.
- `test_loso_cache.py`: Largest quality debt among tests (lint/style/noise).
- `test_loso_cache_fixes.py`: Focused; verify deterministic setup and cleanup robustness.
- `test_ram_cache_comparison.py`: Useful benchmark-style test; ensure clear separation between performance and correctness assertions.

## Open Questions / Clarifications Needed
1. Should `global` feature selection remain default for thesis runtime reasons, or is methodological purity (`per_fold`) now preferred?
2. Should reported RAM benchmark numbers be thesis-facing as `excl preload`, or do you also need an end-to-end (incl preload) table?
3. Do you want full portability now (env/config only), or keep local absolute paths for faster experimentation?
4. Is `test_loso_cache.py` intended as a benchmark/debug harness rather than a CI-style unit test?

## Recommended Action Plan (Prioritized)
1. Fix bare `except` in benchmark cleanup.
2. Decide and enforce feature selection leakage policy (`global` vs `per_fold`).
3. Centralize data path configuration (remove hardcoded paths).
4. Update benchmark summary labels to include RAM preload semantics explicitly.
5. Clean up test lint debt in `test_loso_cache.py`.
6. Standardize print/logging policy across CLI modules.

## Confidence Statement
Findings are based on direct source inspection, pattern scans, and lint output. High-priority findings above are confirmed in code and should be addressed before treating timing comparisons as final thesis-grade artifacts.
