# ML Experiment Caching Pipeline
## Intelligent Result Caching for Machine Learning Experiments

**Bachelor Thesis Project**
**Author:** Lennart Gorzel
**Institution:** IMC FH Krems
**Supervisor:** Prof. Himanshu Buckchash

---

## Overview

This project implements and evaluates **fingerprint-based result caching** for iterative ML experiments on EEG sleep stage classification. Using the BOAS dataset (128 subjects, 5-class sleep staging), we demonstrate that a two-tier caching system (feature-level + model-level) can reduce experiment iteration time by up to **498x** while guaranteeing identical results.

**Research Question:** *Can fingerprint-based result caching reduce computational costs in iterative ML experiments by >80% while ensuring reproducibility through intelligent cache invalidation?*

**Answer:** Yes. XGBoost achieves 498x speedup, SVM-Linear 3,093x. 11 of 15 tested models are cache-viable. All cached results are bit-identical to fresh training.

---

## Development Status - ALL STAGES COMPLETE

| Stage | Description | Status |
|-------|-------------|--------|
| **Stage 1** | Data Loading (BOAS, 128 subjects, 6 EEG channels) | COMPLETE |
| **Stage 2** | Preprocessing + Feature Extraction (149 features/epoch) | COMPLETE |
| **Stage 3** | Feature Selection (ANOVA) + Models (XGBoost, RF) + LOSO splitter | COMPLETE |
| **Stage 4** | LOSO Model Caching (fingerprint-based, two-tier) | COMPLETE |
| **Experiments** | 9 configs x 128 folds x 2 models + 7 bonus experiments | COMPLETE |
| **Results** | All thesis claims validated with data | COMPLETE |

### Thesis Claims - Validation Summary

| # | Claim | Result |
|---|-------|--------|
| 1 | Cache provides significant LOSO speedup | XGBoost: 122-498x, SVM-Linear: 1,550-3,093x |
| 2 | Cached models = identical accuracy | 100% match across all 18 configs |
| 3 | ANOVA is fast and effective for feature selection | 174x faster AND +3.68% accuracy vs MI |
| 4 | Cache benefit scales with dataset size | SVM-RBF: 9x at 10 subjects to 201x at 128 subjects |
| 5 | Clear model viability spectrum exists | 11/15 viable (boosting=viable, tree ensembles=not viable) |
| 6 | Results are hardware-portable | Same verdicts on old and new machine |

---

## Project Structure

```
BachlorThesis/
|
|-- Core Pipeline (22 Python files)
|   |-- config.py                  # Central configuration management (YAML-based)
|   |-- data_loader_boas.py        # BOAS dataset loader (128 subjects, 6 EEG channels)
|   |-- preprocessing.py           # EEG signal preprocessing (bandpass, notch, resample)
|   |-- feature_extractor.py       # 149 features per epoch (time, freq, complexity, global)
|   |-- feature_cache.py           # Layer 1: Feature-level caching with SHA-256 keys
|   |-- feature_selection.py       # ANOVA-based selection with correlation filtering
|   |-- fingerprint.py             # SHA-256 fingerprint generation for LOSO cache keys
|   |-- loso_cache.py              # Layer 2: Model-level caching per LOSO fold
|   |-- models.py                  # XGBoost, Random Forest, FNN implementations
|   |-- cross_validation.py        # LOSO and K-Fold cross-validation
|   |-- training.py                # Training orchestrator with cache integration
|   |-- evaluation.py              # Accuracy, Cohen's Kappa, F1-Macro metrics
|   |-- visualization.py           # Confusion matrices, heatmaps, feature importance
|   |-- cache_visualization.py     # Cache performance visualizations
|   |-- output_formatter.py        # Structured console output
|   |-- leaderboard.py             # Cache performance tracking across experiments
|   |-- interactive_menu.py        # Interactive configuration interface
|   |-- utils.py                   # Common helper functions
|   |-- pipeline.py                # ML pipeline orchestrator
|   |-- run_full_pipeline.py       # Single entry point for complete pipeline
|   |-- run_training.py            # Training experiment runner
|   |-- run_experiment.py          # CLI for ML caching experiments
|   +-- run_thesis_benchmark.py    # Three core thesis experiments (scaling, fingerprint, reproducibility)
|
|-- benchmarks_and_tests/          # Benchmark scripts and test suites
|   |-- benchmark_anova_vs_mi.py   # ANOVA vs Mutual Information comparison
|   |-- benchmark_feature_selection.py  # Feature selection combination testing
|   |-- benchmark_njobs.py         # Parallelism (n_jobs) impact analysis
|   |-- test_loso_cache.py         # Comprehensive LOSO cache test suite
|   +-- *.ps1                      # PowerShell automation scripts (Windows)
|
|-- model_tryouts/                 # Experimental model comparison (15 models)
|   |-- all_models.py              # Standalone 30+ model comparison
|   |-- benchmark_cache_all_models.py  # Cache viability per model type
|   |-- evaluate_rf_cache.py       # RF 9-config cache evaluation
|   +-- evaluate_xgb_cache.py      # XGBoost 9-config cache evaluation
|
|-- results/                       # All experiment outputs
|   |-- thesis_results_human_readable.txt   # Consolidated summary
|   |-- thesis_results_machine_readable.json  # Machine-readable results
|   |-- xgb_cache_evaluation_*.csv           # XGBoost 9-config results
|   |-- rf_cache_evaluation_*.csv            # RF 9-config results
|   |-- benchmark/                 # ANOVA vs MI benchmark results
|   |-- eco_mode/                  # Eco vs power mode analysis
|   |-- feature_size_*/            # Feature count scaling study
|   |-- svm_scaling_*/             # Dataset size scaling study
|   +-- old_machine_30subj/        # Hardware portability results
|
|-- thesis/                        # LaTeX thesis document
|   |-- main.tex                   # Master document
|   |-- chapters/                  # 7 chapters + 2 appendices
|   +-- references.bib             # Bibliography
|
|-- markdowns/                     # Documentation
|   |-- THESIS_DESIGN_DECISIONS.md # Key design rationale
|   |-- BOAS_DATASET_APPROACH.md   # Dataset loading strategy
|   |-- PIPELINE_PROCESSING_DETAILS.md  # Step-by-step pipeline
|   |-- PIPELINE_README.md         # Pipeline architecture
|   |-- CHANNEL_CONFIGURATION_GUIDE.md  # EEG channel configs
|   |-- EXAMPLE_OUTPUT.md          # Example pipeline outputs
|   +-- QUICK_START.md             # Getting started guide
|
|-- test_*.py                      # Root-level test files
|-- example_config.yaml            # Default 6-channel config
|-- config_8channels.yaml          # 8-channel config
|-- requirements.txt               # Python dependencies
+-- papierkorb/                    # Archived outdated documentation
```

---

## Module Responsibilities

| Module | Role |
|--------|------|
| `config.py` | Experiment and channel configuration (YAML-based presets, paths, feature counts) |
| `data_loader_boas.py` | BOAS dataset loader (EDF/annotation files, channel selection, MNE integration) |
| `preprocessing.py` | Signal preprocessing (bandpass 0.5-40Hz, notch 50Hz, resample 256->128Hz, 30s epochs) |
| `feature_extractor.py` | 149 features per epoch: 10 time-domain + 9 frequency-domain + 4 complexity per channel + 11 global |
| `feature_cache.py` | **Layer 1 cache:** Compressed .npz feature caches with SHA-256 keys (224x speedup) |
| `fingerprint.py` | SHA-256 fingerprint generation for LOSO model caching (includes seed, version, model, features, subject) |
| `loso_cache.py` | **Layer 2 cache:** Per-fold trained model caching with fingerprint-based invalidation |
| `feature_selection.py` | ANOVA (f_classif) + correlation filtering, 9 config variations (3 thresholds x 3 top-k) |
| `models.py` | XGBoost, Random Forest, FNN (PyTorch) model implementations |
| `cross_validation.py` | LOSO (128 folds) and K-Fold cross-validation |
| `training.py` | Training orchestrator: loads cached features, applies selection, runs LOSO CV, tracks cache metrics |
| `evaluation.py` | Metrics: Accuracy, Cohen's Kappa, F1-Macro; per-subject and aggregate reporting |
| `pipeline.py` | Pipeline orchestrator (load -> preprocess -> extract features) |
| `run_full_pipeline.py` | Single entry point: feature extraction -> training -> evaluation -> visualization |
| `run_thesis_benchmark.py` | Three thesis experiments: scaling, fingerprint invalidation, reproducibility |

---

## Installation & Usage

### Install
```bash
pip install -r requirements.txt
```

**Key Packages:**
- `mne >= 1.5.0` - EEG processing
- `numpy >= 1.23.0` - Numerical computing
- `scipy >= 1.9.0` - Signal processing
- `pandas >= 1.5.0` - Data manipulation
- `scikit-learn >= 1.2.0` - Machine learning (for future stages)
- `xgboost >= 1.7.0` - Gradient boosting (for future stages)
- `pyyaml >= 6.0` - Configuration files

### Set Up Data Path

The BOAS dataset (~20 GB) is **not included in the repository** and must be provided separately (e.g., via USB stick).

Place the dataset on your machine and note the path. The expected directory structure is:

```
<your-data-path>/
├── sub-1/eeg/sub-1_task-Sleep_acq-psg_eeg.edf
├── sub-1/eeg/sub-1_task-Sleep_acq-psg_events.txt
├── sub-2/eeg/...
└── ...
```

**Configure the data path** using one of these methods:

```bash
# Option 1: Pass via command line (recommended)
python run_experiment.py --quick-test --data-path "/path/to/your/data"
python run_training.py --quick --data-path "/path/to/your/data"

# Option 2: Edit config.py default (line ~79) to your local path
# base_path: str = "/path/to/your/data"
```

Verify access:
```bash
# Windows
dir "C:\Users\YourName\Desktop\Data\sub-1"

# Linux/Mac
ls /path/to/your/data/sub-1/eeg/
```

### Run Experiments
```bash
# Quick test (3 subjects)
python run_experiment.py --quick-test

# Pilot (10 subjects)
python run_experiment.py --pilot

# Full dataset (128 subjects)
python run_experiment.py --full

# Complete thesis pipeline
python run_full_pipeline.py

# Thesis benchmarks (scaling, fingerprint, reproducibility)
python run_thesis_benchmark.py

# Custom config
python run_experiment.py --config my_config.yaml --data-path /path/to/BOAS
```

### Re-Running the Full 9-Config Evaluation
To reproduce the thesis results from scratch:
```bash
# XGBoost: 9 configs x 128 folds (~3.5 hours cold, ~1 min warm)
python model_tryouts/evaluate_xgb_cache.py

# Random Forest: 9 configs x 128 folds (~24 hours cold, ~50 min warm)
python model_tryouts/evaluate_rf_cache.py

# Multi-model viability: 15 models x 5 folds (~2 hours)
python model_tryouts/benchmark_cache_all_models.py

# SVM scaling study
# Run via PowerShell: benchmarks_and_tests/benchmark_svm_scaling.ps1
# Or manually adjust subject count in benchmark_cache_all_models.py

# ANOVA vs MI benchmark
python benchmarks_and_tests/benchmark_anova_vs_mi.py
```

---

## Technical Specifications

### Dataset: BOAS (Bitbrain Open Access Sleep)
- **Source:** PhysioNet
- **Subjects:** 128 (108 unique individuals)
- **Channels:** 6 EEG (F3, F4, C3, C4, O1, O2)
- **Sampling Rate:** 256 Hz (downsampled to 128 Hz)
- **Sleep Stages:** 5 (Wake, N1, N2, N3, REM)
- **Labeling:** Human consensus from 3 expert scorers

### Preprocessing Pipeline
```
Raw EDF (256 Hz) -> Bandpass 0.5-40 Hz (FIR) -> Notch 50 Hz -> Downsample to 128 Hz -> 30s epochs -> (n_epochs, 6, 3840)
```

### Feature Extraction (149 Features)
- **Per-channel (23 x 6 = 138):** Time-domain (10), Frequency-domain (9), Complexity (4)
- **Global (11):** Coherence (6 pairs), Phase-Locking Value (3 pairs), entropy, complexity

### Experiment Grid (18 Configurations)
- **Models:** XGBoost, Random Forest (2)
- **Correlation thresholds:** 0.75, 0.90, None (3)
- **Top-K features:** 30, 50, None/149 (3)
- **Validation:** LOSO (128 folds per config)

### Fingerprint Components
The cache fingerprint (SHA-256, 32 hex chars) includes:
- `random_seed` - Reproducibility guarantee
- `code_version` - Tracks code changes
- `model_config` - Algorithm name + all hyperparameters
- `feature_config` - Base features, correlation threshold, top-k, selected feature hash
- `held_out_subject` - Prevents data leakage across LOSO folds

**Note:** Dataset version and preprocessing config are NOT in the fingerprint (they affect feature extraction, which is cached separately in Layer 1).

---

## Key Results

### XGBoost (Best Overall)
- Best config: corr=None, k=None (all 149 features) -> **85.5% accuracy, 498x speedup, 188 MB cache**
- Average across 9 configs: 208x speedup, 185 MB cache, 7.1s warm time

### Random Forest (Borderline Cache-Viable)
- Best config: corr=None, k=None -> **81.9% accuracy, 109.8x speedup, 16.3 GB cache**
- Average: 28x speedup, 18.5 GB cache, 330s warm time

### Multi-Model Viability (15 models tested)
- **11 VIABLE:** Gradient Boosting (18,030x), AdaBoost (2,616x), SVM-Linear (1,950x), XGBoost (119x), LightGBM (90x), and 6 more
- **4 NOT VIABLE:** Random Forest (15x), Extra Trees (4x), KNN (5x) - large serialized models

### Scaling: More Subjects = More Speedup
SVM-RBF: 9x (10 subjects) -> 201x (128 subjects). Training cost grows quadratically, cache loading is constant.

---

## Configuration

### Default (example_config.yaml)
```yaml
experiment_name: "pilot_xgboost_baseline"
data:
  base_path: "C:/Users/DerHo/Desktop/Data"
  channel_preset: "eeg_only"    # 6 channels -> 149 features
  use_human_labels: true
preprocessing:
  bandpass_low: 0.5
  bandpass_high: 40.0
  notch_frequency: 50.0
  target_sfreq: 128.0
  epoch_duration: 30.0
output_dir: "./results"
```

---

## Documentation

| File | Purpose |
|------|---------|
| `markdowns/THESIS_DESIGN_DECISIONS.md` | Why BOAS, why LOSO, why ANOVA, why 149 features |
| `markdowns/PIPELINE_PROCESSING_DETAILS.md` | Step-by-step pipeline walkthrough |
| `markdowns/BOAS_DATASET_APPROACH.md` | Dataset structure and loading strategy |
| `markdowns/CHANNEL_CONFIGURATION_GUIDE.md` | 6 vs 8 channel comparison |
| `markdowns/QUICK_START.md` | Getting started guide |
| `METHODOLOGY_VERIFICATION_REPORT.md` | Known discrepancies between thesis text and code (9 items) |
| `CACHE_INVALIDATION_VERIFICATION.md` | Proof that cache invalidation works correctly |
| `TWO_TIER_CACHE_EXPLAINED.md` | Two-tier caching architecture explanation |
| `results/thesis_results_human_readable.txt` | Complete experimental results summary |

---

## Citation

```bibtex
@mastersthesis{gorzel2026mlcaching,
  author  = {Lennart Gorzel},
  title   = {ML Experiment Optimization with Intelligent Caching},
  school  = {IMC Fachhochschule Krems},
  year    = {2026},
  type    = {Bachelor's Thesis},
  address = {Krems, Austria}
}
```

---

**Author:** Lennart Gorzel | **Supervisor:** Prof. Himanshu Buckchash | **Institution:** IMC FH Krems
**README Version:** 3.0 | **Last Updated:** March 16, 2026 | **Status:** All stages complete, experiments done
