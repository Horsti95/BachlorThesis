# ML Experiment Caching Pipeline

Companion code for the Bachelor thesis **"Caching strategies for reproducible ML experiments on EEG sleep-stage classification"**.

- **Author:** Lennart Gorzel
- **Institution:** IMC FH Krems
- **Supervisor:** Prof. Himanshu Buckchash
- **Status:** Thesis submitted (May 2026)

---

## What this does

End-to-end pipeline for sleep-stage classification on the BOAS dataset (128 subjects, 6 EEG channels):

1. Load raw EEG (EDF) and human-consensus annotations.
2. Preprocess (bandpass 0.5–40 Hz, notch 50 Hz, downsample 256→128 Hz, 30-s epochs).
3. Extract 149 hand-crafted features (time / frequency / complexity / global).
4. Run feature selection (correlation filter + ANOVA top-k) under LOSO cross-validation.
5. Train XGBoost and Random Forest models on each fold.
6. Cache features (Layer 1) and per-fold trained models (Layer 2) using SHA-256 fingerprints. Re-runs hit the cache when nothing relevant changed.

The thesis evaluates this caching against a non-cached baseline across 18 configurations (2 models × 3 correlation thresholds × 3 feature counts) on 128 LOSO folds.

---

## Repository layout

```
.
├── README.md                  # this file
├── requirements.txt           # loose pins (development)
├── requirements.lock          # exact pins (thesis-run reproducibility)
│
├── example_config.yaml        # 6-channel EEG config (149 features) — default
├── config_8channels.yaml      # 8-channel EEG+EOG+EMG config (195 features)
│
├── run_experiment.py          # CLI: feature extraction + interactive menu
├── run_training.py            # CLI: training grid on cached features
├── run_full_pipeline.py       # CLI: end-to-end (extraction + training + eval)
│
├── pipeline.py                # feature-extraction pipeline orchestration
├── training.py                # training pipeline (Layer 2 cache integration)
├── interactive_menu.py        # interactive config menu used by run_experiment
│
├── config.py                  # YAML-backed config dataclasses
├── data_loader_boas.py        # BOAS EDF + annotation loading
├── preprocessing.py           # filtering, notch, resample, epoching
├── feature_extractor.py       # 149-/195-feature extractor
├── feature_cache.py           # Layer 1 cache (per-subject NPZ)
├── feature_selection.py       # correlation filter + ANOVA SelectKBest
├── fingerprint.py             # SHA-256 fingerprinting (LOSOFingerprint)
├── loso_cache.py              # Layer 2 cache (per-fold trained models)
├── models.py                  # XGBoost + Random Forest wrappers
├── cross_validation.py        # LOSO splitter + summarisation helpers
├── evaluation.py              # per-fold metrics + aggregation
├── visualization.py           # plotting helpers used by run_training
├── cache_visualization.py     # cache-stats plotting
├── leaderboard.py             # ranking + clinical-target reporting
├── output_formatter.py        # console/log output formatting
└── utils.py                   # logging, timestamp, small helpers
│
├── markdowns/                 # supplementary project docs (kept for reference)
├── thesis/                    # LaTeX source of the submitted thesis
├── results/                   # experiment outputs (gitignored, partially tracked)
└── archive/                   # one-shot scripts, dev tests, old docs (see archive/README.md)
```

---

## Installation

```bash
pip install -r requirements.txt    # development install
pip install -r requirements.lock   # exact thesis-run versions (reproducibility)
```

Key dependencies: `mne`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `xgboost`, `joblib`, `pyyaml`, `tqdm`, `matplotlib`.

The BOAS dataset path is configured per-machine in `example_config.yaml` (`data.base_path`) or via `--data-path` on the CLI. The original thesis run used `C:/Users/DerHo/Desktop/Data`.

---

## Usage

### End-to-end (recommended)

```bash
python run_full_pipeline.py --quick --data-path /path/to/BOAS    # 3 subjects
python run_full_pipeline.py --full  --data-path /path/to/BOAS    # 128 subjects
python run_full_pipeline.py --full  --data-path /path/to/BOAS --benchmark   # also collect cache timings
```

### Step-by-step

```bash
# 1. extract + cache features
python run_experiment.py --quick-test                   # 3 subjects
python run_experiment.py --pilot                        # 10 subjects
python run_experiment.py --full                         # 128 subjects
python run_experiment.py --interactive                  # interactive menu

# 2. train on cached features (requires features_cache_global/ to be populated)
python run_training.py --quick                          # quick check
python run_training.py --pilot                          # pilot grid
python run_training.py --full                           # full 18-config grid × 128 folds
```

Common flags: `--config FILE`, `--data-path PATH`, `--output-dir DIR`, `--experiment-name NAME`, `--log-level DEBUG`, `--log-file FILE`. See `python <script>.py --help` for the full list.

---

## Configuration

`example_config.yaml` is the default 6-channel preset (149 features). Switch to 8 channels (149→195 features, adds EOG + EMG):

```bash
python run_experiment.py --pilot --config config_8channels.yaml
```

Channel presets supported by `config.py`:

| Preset | Channels | Features |
|---|---|---|
| `eeg_only` *(default)* | F3, F4, C3, C4, O1, O2 | 149 |
| `eeg_plus_physiological` | + EOG, EMG | 195 |
| `custom` | any subset (specify under `data.channels`) | varies |

---

## Output

```
results/
├── features_cache_global/                    # Layer 1: per-subject NPZ feature caches
├── loso_model_cache/                         # Layer 2: per-fold trained-model caches
└── <experiment_name>_<timestamp>/
    ├── per_subject/                          # raw + preprocessed per subject
    ├── features/                             # aggregated feature matrix
    ├── training_results/                     # per-config JSON results
    ├── latex/                                # LaTeX tables (if --benchmark)
    └── pipeline_stats.json                   # timings
```

Thesis numbers reported in the submitted PDF live in `results/training_*_full/` and `results/benchmark_results_pc1_5090/`.

---

## Reproducing the thesis numbers

1. `pip install -r requirements.lock`
2. Place BOAS data at the path set in `example_config.yaml` (or pass `--data-path`).
3. `python run_full_pipeline.py --full --data-path /path/to/BOAS --benchmark`
4. Cold run ≈ 9 h on the thesis hardware; warm re-run hits the cache.

The exact CLI invocations used for the figures and tables in the thesis are preserved in `archive/scripts/` (e.g., `run_thesis_benchmark.py`, `run_combo_cold_warm_suite.py`, `update_figures_pc1.py`, `generate_thesis_figures.py`).

---

## archive/

Everything that was needed to write the thesis but is not part of the active pipeline lives under `archive/`:

- `archive/scripts/` — figure generators, benchmark suites, machine-specific PowerShell drivers, one-shot helpers
- `archive/tests/` — development verification scripts (`test_cache_comprehensive.py`, `test_imports.py`, `test_loso_cache_fixes.py`, `test_ram_cache_comparison.py`)
- `archive/experiments/` — `model_tryouts/` (explicitly *not* part of thesis), `testing/`, `benchmarks_and_tests/`
- `archive/docs/` — verification reports, design notes, presentation outlines
- `archive/old_versions/` — superseded copies of files still present in root

See `archive/README.md` for an inventory.

---

## Smoke test

`verify_pipeline.ps1` (Windows PowerShell) compiles every root `*.py`, runs `--help` on the three entry points, runs the data-free thesis tests, and — if the BOAS dataset is present — runs the 3-subject pipeline plus the data-dependent tests. It continues past failures and writes a single timestamped log:

```powershell
.\verify_pipeline.ps1
.\verify_pipeline.ps1 -DataPath "D:\BOAS"
```

Output goes to `verify_<yyyyMMdd_HHmmss>.txt` in the repo root (gitignored).

---

## Future work

Items below are open extensions to this thesis. The first group is taken directly from Chapter 5 of the submitted PDF; the rest were proposed during development but deferred for scope or time.

### From the thesis (Chapter 5)

- **Lazy per-fold feature selection cache.** The thesis used global ANOVA feature selection, which leaks ~0.8% of test data per fold. A per-fold ANOVA fixes the leakage but, naively, kills the warm-run speedup (22.9× → 4.4×). A second-layer cache keyed on `(dataset_version, code_version, corr, top_k, held_out_subject)` would let warm runs skip ANOVA entirely (~1 ms registry lookup) and recover the speedup without the leakage. Designed but not implemented.
- **Cross-domain validation.** The fingerprinting approach is domain-agnostic. Applying it to other subject-cross-validated tasks (BCIs, speaker-independent ASR, patient-level medical imaging) would test how well the viability metrics (η, ρ) generalise.
- **MLOps integration.** Package the framework as plugins for DVC, ZenML, or MLflow so that fold-aware caching is available to researchers already on those platforms. The η-based viability verdicts would inform the default caching policy.
- **Viability knowledge base (master / PhD trajectory).** The thesis produced a single-domain viability map. A natural extension is a cross-domain viability database, and on top of it an "intelligent caching orchestrator" that reads an experimental plan and configures the caching strategy automatically.

### Models, validation, and data scaling (deferred from thesis scope)

- **FNN model.** A placeholder exists (`models.py`, `fnn_size_probe.py` in `archive/scripts/`) but the FNN was not run at full scale due to environment constraints. Re-running the full grid with FNN added would extend the thesis from 18 to 27 configurations.
- **More cache-viable model families.** The 15-model viability study includes Logistic Regression, SVM-RBF/Linear, LightGBM, CatBoost, AdaBoost, etc. Several were evaluated under `archive/experiments/model_tryouts/` ("not part of the thesis") but a clean integration into the main `training.py` grid is open.
- **Alternative cross-validation strategies.** Currently only LOSO. Subject-grouped K-fold (faster development iteration), stratified K-fold (alternative baseline), and nested CV (hyperparameter tuning without leakage) are all natural extensions and benefit identically from the feature cache.
- **Extended scaling grids.** The thesis only has subject scaling for SVM-Linear/RBF and feature scaling for XGBoost. A 2-D grid (subjects × features) for 3–4 key models would produce a viability heatmap, turning the binary VIABLE / NOT VIABLE verdict into a viability curve. See `archive/docs/papierkorb_*` and `archive/scripts/benchmark_*` for the partial groundwork.
- **RAM-resident model cache.** Prototype in `archive/tests/test_ram_cache_comparison.py` showed a 1.8–3.7× *marginal* gain over the SSD cache at 50 subjects, dropping to negligible at 128 (GC pressure). Worth revisiting with a smarter eviction policy if datasets grow much larger.

### Tooling and UX

- **User interface.** Today the framework is CLI-only (`run_experiment.py`, `run_training.py`, `run_full_pipeline.py`) plus the YAML configs. A nicer UX would help non-CLI users:
  - A Streamlit / Gradio web dashboard for picking a config, launching a run, and watching cache hit/miss + per-fold timing live.
  - A desktop GUI (Tauri / PyQt) wrapping the same flows, with a results browser that loads `results/training_*/training_results/*.json` and renders metrics.
  - A "configurator" step that asks for hardware specs and dataset size, then suggests cache strategy + estimated cold/warm time using the η/ρ viability metrics from the thesis.
- **Configurable data path.** `interactive_menu.py` and other scripts hardcode `C:\Users\DerHo\Desktop\Data`. Move to an env var (`BOAS_DATA_PATH`) or a single `paths.yaml` so the repo runs on any machine without source edits.
- **Real pytest suite.** The four scripts under `archive/tests/` are standalone runners, not pytest fixtures. Promoting the most important ones into a real `tests/` dir with `pytest`-style assertions would catch regressions automatically (currently nothing does).
- **Public reproducibility check.** A small CI workflow that runs `--quick-test` on a tiny synthetic dataset (3 fake subjects) on every push would catch breakage without needing the BOAS data on the runner.

### Pipeline polish

- **Re-enable amplitude validation for BOAS** with thresholds calibrated to the dataset (the current validator was disabled because the original limits rejected 100% of BOAS epochs).
- **Streaming feature extraction.** Currently each subject is loaded, preprocessed, and feature-extracted before the next one. For very large datasets, a streaming pass that releases memory between subjects would help.
- **Cache eviction.** Layer 2 grows unbounded (~18.5 GB for the full RF run). An LRU policy, or a per-fingerprint TTL, would let the cache live indefinitely without manual pruning.

---

## Citation

```bibtex
@mastersthesis{gorzel2026mlcaching,
  author  = {Lennart Gorzel},
  title   = {Caching strategies for reproducible ML experiments on EEG sleep-stage classification},
  school  = {IMC Fachhochschule Krems},
  year    = {2026},
  type    = {Bachelor's Thesis},
  address = {Krems, Austria}
}

@misc{boas2023,
  title        = {Bitbrain Open Access Sleep (BOAS) Database},
  author       = {Bitbrain Technologies},
  year         = {2023},
  howpublished = {PhysioNet}
}
```

---

## Contact

- **Author:** Lennart Gorzel
- **Supervisor:** Prof. Himanshu Buckchash (IMC FH Krems)
- **License:** Academic / educational use.
