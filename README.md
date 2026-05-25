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

# ML Experiment Caching Pipeline

Minimal, self-contained pipeline for extracting features from BOAS EEG data and running LOSO experiments with a two-layer cache (features + per-fold models).

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a fast smoke test (3 subjects):

```bash
python run_experiment.py --quick-test
```

Run full pipeline (128 subjects):

```bash
python run_full_pipeline.py --full --data-path "C:/path/to/BOAS"
```

## Core files

- `run_experiment.py` — extract and cache features
- `run_training.py` — train models on cached features
- `feature_extractor.py`, `preprocessing.py`, `feature_cache.py` — main pipeline pieces
- `example_config.yaml`, `config_8channels.yaml` — presets

## Configuration

Edit `example_config.yaml` or pass `--config` / `--data-path` on the CLI. Default preset `eeg_only` produces 149 features.

## Output

Results and caches are written under `results/`:

- `features_cache_global/` — per-subject feature npz caches
- `loso_model_cache/` — per-fold trained-model caches

## License & Contact

Academic / educational use. Author: Lennart Gorzel — see project root for thesis sources under `thesis/`.

----

If you'd like a slightly longer README (examples, benchmarks, or citation), I can expand any section.
The exact CLI invocations used for the figures and tables in the thesis are preserved in `archive/scripts/` (e.g., `run_thesis_benchmark.py`, `run_combo_cold_warm_suite.py`, `update_figures_pc1.py`, `generate_thesis_figures.py`).
