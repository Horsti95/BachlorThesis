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
