# ML Experiment Caching Pipeline

Companion code for the Bachelor thesis **"Caching strategies for reproducible ML experiments on EEG sleep-stage classification"**.

- **Author:** Lennart Gorzel
- **Institution:** IMC FH Krems
- **Supervisor:** Prof. Himanshu Buckchash
- **Status:** Thesis submitted (May 2026)

---

## Quick start

```bash
git clone https://github.com/Horsti95/BachlorThesis
cd BachlorThesis
pip install -r requirements.txt
python download_dataset.py                                       # downloads BOAS dataset (~10-20 GB)
python run_full_pipeline.py --quick --data-path ./data/BOAS      # quick test (3 subjects, ~2 min)
```

The `--quick` flag runs only 3 subjects so you can verify the pipeline works without processing all 128 subjects. For the full thesis run use `--full` instead.

---

## Dataset

This pipeline uses the **Bitbrain Open Access Sleep (BOAS)** dataset — a publicly available, BIDS-formatted PSG/EEG sleep recording dataset with 128 subjects.

- **Source:** https://openneuro.org/datasets/ds005555
- **Size:** ~10-20 GB total
- **Contents per subject:** PSG EEG recording (.edf) + sleep stage annotations (.txt)

### Download

```bash
python download_dataset.py                         # downloads + validates automatically
python download_dataset.py --target-dir /my/path   # custom download location
```

The script downloads from OpenNeuro, normalizes file extensions for pipeline compatibility, and validates the result.

### Expected directory structure

```
data/BOAS/
  sub-1/eeg/
    sub-1_task-Sleep_acq-psg_eeg.edf
    sub-1_task-Sleep_events.txt
  sub-2/eeg/
    sub-2_task-Sleep_acq-psg_eeg.edf
    sub-2_task-Sleep_events.txt
  ...
  sub-128/eeg/...
```

### Validate an existing download

```bash
python download_dataset.py --validate-only ./data/BOAS --test-load
```

This checks structure, file presence, and loads one subject through the pipeline to confirm everything works.

---

## Installation

```bash
pip install -r requirements.txt    # development install
pip install -r requirements.lock   # exact thesis-run versions (for reproducibility)
```

Requires Python 3.8+ (3.10+ recommended).

Key dependencies: `mne`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `xgboost`, `joblib`, `pyyaml`, `tqdm`, `matplotlib`.

---

## Usage

### End-to-end pipeline (recommended)

```bash
python run_full_pipeline.py --quick --data-path ./data/BOAS              # 3 subjects (~2 min)
python run_full_pipeline.py --pilot --data-path ./data/BOAS              # 10 subjects
python run_full_pipeline.py --full  --data-path ./data/BOAS              # 128 subjects (full thesis run)
python run_full_pipeline.py --full  --data-path ./data/BOAS --benchmark  # cold vs warm cache comparison
```

### Step-by-step

```bash
# 1. Extract + cache features
python run_experiment.py --quick-test       # 3 subjects
python run_experiment.py --pilot            # 10 subjects
python run_experiment.py --full             # all 128 subjects

# 2. Train on cached features (requires step 1 to have run first)
python run_training.py --quick
python run_training.py --pilot
```

### Configuration

Edit `example_config.yaml` or pass CLI flags:
- `--data-path` — location of the BOAS dataset
- `--config` — path to a custom YAML config file

Default preset (`eeg_only`) uses 6 EEG channels and produces 149 features per epoch.

---

## What this does

End-to-end pipeline for sleep-stage classification on the BOAS dataset (128 subjects, 6 EEG channels):

1. Load raw EEG (EDF) and human-consensus annotations.
2. Preprocess (bandpass 0.5-40 Hz, notch 50 Hz, downsample 256->128 Hz, 30-s epochs).
3. Extract 149 hand-crafted features (time / frequency / complexity / global).
4. Run feature selection (correlation filter + ANOVA top-k) under LOSO cross-validation.
5. Train XGBoost and Random Forest models on each fold.
6. Cache features (Layer 1) and per-fold trained models (Layer 2) using SHA-256 fingerprints.

The thesis evaluates this caching against a non-cached baseline across 18 configurations (2 models x 3 correlation thresholds x 3 feature counts) on 128 LOSO folds.

---

## Output

Results and caches are written under `results/`:

- `features_cache_global/` — per-subject feature caches (Layer 1)
- `loso_model_cache/` — per-fold trained-model caches (Layer 2)
- `run_report.json` — timing, metrics, and configuration for each run

---

## Repository layout

```
.
├── download_dataset.py        # dataset download + validation helper
├── run_experiment.py          # CLI: feature extraction + interactive menu
├── run_training.py            # CLI: training grid on cached features
├── run_full_pipeline.py       # CLI: end-to-end (extraction + training + eval)
│
├── pipeline.py                # feature-extraction pipeline orchestration
├── training.py                # training pipeline (Layer 2 cache integration)
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
│
├── example_config.yaml        # 6-channel EEG config (149 features) — default
├── config_8channels.yaml      # 8-channel EEG+EOG+EMG config (195 features)
├── requirements.txt           # loose dependency pins
├── requirements.lock          # exact pins (thesis reproducibility)
│
├── thesis/                    # LaTeX source of the submitted thesis
├── results/                   # experiment outputs
└── archive/                   # development scripts and old docs
```

---

## License

Academic / educational use. Author: Lennart Gorzel — see `thesis/` for the full thesis document.
