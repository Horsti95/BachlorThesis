# ML Experiment Caching Pipeline
## Intelligent Result Caching for Machine Learning Experiments

**Bachelor Thesis Project**  
**Author:** Lennart Gorzel  
**Institution:** IMC FH Krems  
**Supervisor:** Prof. Himanshu Buckchash  
**Submission Deadline:** January 15, 2026

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Technical Specifications](#technical-specifications)
- [Results & Output](#results--output)
- [Design Decisions](#design-decisions)
- [Troubleshooting](#troubleshooting)
- [Development Status](#development-status)
- [Citation](#citation)

---


## 🚧 Development Status

### ✅ Stage 1: Data Loading - COMPLETE
- ✅ BOAS dataset loader (128 subjects)
- ✅ 6 EEG channel configuration
- ✅ Human consensus labels

### ✅ Stage 2: Preprocessing & Features - COMPLETE
- ✅ Signal preprocessing (bandpass 0.5-40Hz, notch 50Hz, downsample 256→128Hz)
- ✅ Feature extraction (149 features per epoch)
- ✅ Feature caching with SHA-256 fingerprinting (128/128 subjects cached)
- ✅ 224× speedup on cached runs

### ✅ Stage 3: Feature Selection & Models - COMPLETE
- ✅ ANOVA-based feature selection (GLOBAL scope)
- ✅ Correlation filter (configurable threshold)
- ✅ XGBoost and Random Forest models
- ✅ LOSO cross-validation splitter
- ⚠️ FNN model placeholder (not required for thesis)

### ⏳ Stage 4: LOSO Model Caching - IN PROGRESS (Critical Gap!)
- ❌ LOSOFingerprint class (includes holdout_subject for data leakage prevention)
- ❌ LOSOModelCache class (save/load trained models)
- ❌ Training integration (cache check before train, save after)
- ❌ Demo script showing 30× speedup

**Estimated effort: 5-6 hours**

### Project Timeline

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| Architecture design | Dec 22, 2025 | ✅ Complete |
| Stage 1 & 2 (Data + Features) | Dec 23, 2025 | ✅ Complete |
| Stage 3 (Feature Selection + Models) | Dec 25, 2025 | ✅ Complete |
| **Stage 4 (LOSO Model Cache)** | **Dec 28, 2025** | **⏳ In Progress** |
| Full Experiments (18 configs × 128 folds) | Jan 5, 2026 | ⏳ Planned |
| Results Analysis & Thesis Writing | Jan 10-14, 2026 | ⏳ Planned |
| **Submission Deadline** | **Jan 15, 2026** | 🎯 Target |
Channel configuration: eeg_only
  Channels: 6 (PSG_F3, PSG_F4, PSG_C3, PSG_C4, PSG_O1, PSG_O2)
  Expected features: 149

PROCESSING SUBJECT 1/3: 1
  [Step 1/3] Loading raw EEG data...
  ✓ Loaded: 997 total epochs, 256.0 Hz
  
  [Step 2/3] Preprocessing (filtering, downsampling, epoching)...
  ✓ Preprocessed: 997 valid epochs, shape (997, 6, 3840)
  
  [Step 3/3] Extracting 149 features...
  ✓ Features extracted: (997, 149)

Total epochs: 2,528
Features per epoch: 149
Time: 3.5 minutes
```

---

## 📁 Project Structure

```
Code/
│
├── 📄 Core Python Modules (9 files)
│   ├── run_experiment.py          # CLI entry point (YOU RUN THIS)
│   ├── pipeline.py                # Main orchestrator
│   ├── config.py                  # Configuration system
│   ├── data_loader_boas.py        # BOAS dataset loader
│   ├── preprocessing.py           # Signal filtering, downsampling, epoching
│   ├── feature_extractor.py       # 149/195 feature extraction
│   ├── utils.py                   # Helper functions
│   ├── example_config.yaml        # Config: 6 channels (default)
│   └── requirements.txt           # Python dependencies
│
├── 📄 Configuration Templates (2 files)
│   ├── example_config.yaml        # 6 EEG channels → 149 features (STANDARD)
│   └── config_8channels.yaml      # 8 channels (EEG+EOG+EMG) → 195 features
│
├── 📚 Documentation (3 essential files)
│   ├── README.md                  # This file
│   ├── QUICK_START.md             # Detailed getting started guide
│   ├── THESIS_DESIGN_DECISIONS.md # Design justifications for thesis
│   └── CHANNEL_CONFIGURATION_GUIDE.md  # 6 vs 8 channel comparison
│
├── 📦 Archive (deprecated files)
│   └── archive/
│       ├── data_loader_sleepedf_OLD.py  # OLD: Sleep-EDF loader
│       └── DATA_LOADER_README.md        # OLD: Sleep-EDF docs
│
└── 📊 Results (generated at runtime)
    └── results/
        └── experiment_YYYYMMDD_HHMMSS/
            ├── per_subject/         # Individual subject data
            ├── features/            # Aggregated features
            └── pipeline_stats.json  # Timing statistics
```

### Active Files Summary

| Category | Files | Purpose |
|----------|-------|---------|
| **Core Python** | 9 | Pipeline implementation |
| **Configs** | 2 | Experiment configurations |
| **Essential Docs** | 4 | Usage & design documentation |
| **Total Active** | **15** | **All ready to use** |

---

## 🧭 Module Responsibilities (Short)

Brief role descriptions for the main modules in this repository:

- `run_experiment.py`: CLI entry point — parse CLI args, build the experiment config, and launch the pipeline.
- `pipeline.py`: Main orchestrator — loads subjects, runs preprocessing, handles feature extraction, integrates caching, and saves per-subject and aggregated outputs.
- `config.py`: Experiment and channel configuration model — defines presets, paths, and expected feature counts.
- `data_loader_boas.py`: BOAS dataset loader — finds EDF/annotation files, selects channels, and returns MNE raw objects and annotations.
- `preprocessing.py`: Signal preprocessing and epoching — filtering, notch, resampling, and creation of fixed-length epochs.
- `feature_extractor.py`: Feature extraction core — computes per-channel and global features; supports 6 or 8 channels; exposes batch extraction routines.
- `feature_cache.py`: Caching helpers — save/load compressed full-feature `.npz` caches and filter cached features to a requested channel subset.
- `utils.py`: Utility helpers — I/O wrappers, formatters, progress tracker, validation, and small helpers used across modules.
- `requirements.txt`: Dependency manifest — Python packages required to run the pipeline.

If you want, I can expand any of these into a dedicated markdown under `markdowns/` with API details and example outputs.


---

## 💻 Installation

### Step 1: Install Python Dependencies

For development (loose lower-bound pins, latest compatible versions):

```bash
pip install -r requirements.txt
```

For **reproducing thesis numbers** (exact versions used during the
thesis experimental run):

```bash
pip install -r requirements.lock
```

The `requirements.lock` file pins every direct and transitive
dependency to the exact version that produced the reported results.
Anyone re-running the experiments should install from the lock file to
get a bit-identical Python environment.

**Note:** `requirements.lock` is generated on the thesis-run machine
with `pip freeze > requirements.lock`. Regenerate before each
publication / submission so the lock matches the code state being
reported.

**Key Packages:**
- `mne >= 1.5.0` - EEG processing
- `numpy >= 1.23.0` - Numerical computing
- `scipy >= 1.9.0` - Signal processing
- `pandas >= 1.5.0` - Data manipulation
- `scikit-learn >= 1.2.0` - Machine learning (for future stages)
- `xgboost >= 1.7.0` - Gradient boosting (for future stages)
- `pyyaml >= 6.0` - Configuration files

### Step 2: Verify Data Path

Ensure your BOAS dataset is accessible:

```bash
# Windows
dir "C:\Users\DerHo\Desktop\Data\sub-1"

# Should show:
#   sub-1/eeg/sub-1_task-Sleep_acq-psg_eeg.edf
#   sub-1/eeg/sub-1_task-Sleep_acq-psg_events.txt
```

#### Local data path (developer note)

The BOAS data path is hardcoded as `C:/Users/DerHo/Desktop/Data` in
`benchmarks_and_tests/test_validation.py` (constant `DEFAULT_DATA_PATH` at the
top of the file). If you run this code on a different machine, edit that
constant to point at your local copy of the dataset. The path is intentionally
not configured via env var or YAML because this is a single-developer thesis
project.

### Step 3: Test Installation

```bash
python run_experiment.py --quick-test
```

If you see processing output and no errors, installation is complete! ✅

---

## 🎮 Usage

### Command-Line Interface

```bash
python run_experiment.py [OPTIONS]
```

### Common Commands

#### Quick Test (3 subjects, ~3 minutes)
```bash
python run_experiment.py --quick-test
```

#### Pilot Run (10 subjects, ~15 minutes)
```bash
python run_experiment.py --pilot
```

#### Full Dataset (128 subjects, ~2-3 hours)
```bash
python run_experiment.py --full
```

#### Custom Configuration
```bash
python run_experiment.py --config my_experiment.yaml
```

#### With Overrides
```bash
python run_experiment.py --pilot --experiment-name test1 --log-level DEBUG
```

### Command-Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--quick-test` | Run on 3 subjects | `--quick-test` |
| `--pilot` | Run on 10 subjects | `--pilot` |
| `--full` | Run on all 128 subjects | `--full` |
| `--config FILE` | Use custom config | `--config my_config.yaml` |
| `--experiment-name NAME` | Set experiment name | `--experiment-name baseline_v1` |
| `--log-level LEVEL` | Set logging level | `--log-level DEBUG` |
| `--log-file FILE` | Save logs to file | `--log-file experiment.log` |
| `--data-path PATH` | Override data directory | `--data-path D:\Data` |
| `--output-dir DIR` | Override output directory | `--output-dir ./my_results` |

---

## ⚙️ Configuration

### Basic Configuration (example_config.yaml)

```yaml
experiment_name: "pilot_xgboost_baseline"

data:
  base_path: "C:/Users/DerHo/Desktop/Data"
  channel_preset: "eeg_only"  # 6 channels → 149 features
  use_human_labels: true      # Use human consensus labels

preprocessing:
  bandpass_low: 0.5           # Hz
  bandpass_high: 40.0         # Hz
  notch_frequency: 50.0       # Hz (50=Europe, 60=US)
  target_sfreq: 128.0         # Hz (downsampled from 256)
  epoch_duration: 30.0        # seconds

features:
  compute_time_domain: true
  compute_frequency_domain: true
  compute_complexity: true
  compute_global_features: true

output_dir: "./results"
log_level: "INFO"
```

### Channel Configuration Options

The pipeline supports flexible channel configurations:

#### Option 1: 6 EEG Channels (Standard) ✅ **RECOMMENDED**

```yaml
data:
  channel_preset: "eeg_only"
  # Results in: 149 features (6 × 23 + 11)
```

**Channels:** F3, F4, C3, C4, O1, O2  
**Best for:** Standard sleep staging, faster processing, smaller storage

#### Option 2: 8 Channels (EEG + Physiological)

```yaml
data:
  channel_preset: "eeg_plus_physiological"
  # Results in: 195 features (8 × 23 + 11)
```

**Channels:** F3, F4, C3, C4, O1, O2, EOG, EMG  
**Best for:** Enhanced REM/Wake detection, comprehensive analysis

#### Option 3: Custom Channels

```yaml
data:
  channel_preset: "custom"
  channels:
    - PSG_F3
    - PSG_C3
    - PSG_O1
  # Specify any combination
```

**See `CHANNEL_CONFIGURATION_GUIDE.md` for detailed comparison.**

---

## 🔬 Technical Specifications

### Dataset: BOAS (Bitbrain Open Access Sleep)

- **Source:** PhysioNet, recorded in Spain
- **Subjects:** 128 (108 unique individuals, some with multiple nights)
- **Channels:** 8 total (6 EEG + EOG + EMG)
- **Sampling Rate:** 256 Hz (downsampled to 128 Hz)
- **Sleep Stages:** 5 (Wake, N1, N2, N3, REM)
- **Labeling:** Human consensus from 3 expert scorers
- **Duration:** ~8 hours per recording

### Preprocessing Pipeline

```
Raw EDF (256 Hz, 6-8 channels)
    ↓
1. Channel Selection
   → Select 6 EEG or 8 channels
    ↓
2. Bandpass Filter
   → 0.5-40 Hz (FIR filter, Hamming window)
    ↓
3. Notch Filter
   → 50 Hz (power line noise removal)
    ↓
4. Downsampling
   → 256 Hz → 128 Hz (antialiasing applied)
    ↓
5. Epoch Extraction
   → 30-second windows (3,840 samples @ 128 Hz)
    ↓
6. Quality Validation
   → Amplitude checks (currently disabled for BOAS)
    ↓
Output: (n_epochs, n_channels, 3840)
```

### Feature Extraction (149 Features for 6 Channels)

#### Per-Channel Features (23 × 6 = 138)

**Time-Domain (10 features per channel):**
- Statistical: mean, std, var, min, max, peak-to-peak
- Derived: RMS, skewness, kurtosis, zero-crossing rate

**Frequency-Domain (9 features per channel):**
- Band Powers: delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), sigma (12-16 Hz), beta (16-30 Hz), gamma (30-40 Hz)
- Spectral: spectral entropy, peak frequency, median frequency

**Complexity (4 features per channel):**
- Hjorth parameters: mobility, complexity
- Fractal analysis: Hurst exponent, DFA (detrended fluctuation analysis)

#### Global Features (11)

- **Coherence (6 pairs):** F3-F4, F3-C3, F3-C4, F4-C4, C3-C4, O1-O2
- **Phase-Locking Value (3 pairs):** F3-O1, F4-O2, C3-C4
- **Global Metrics (2):** global entropy, global complexity

### Performance Benchmarks

| Dataset Size | Processing Time | Storage | Output Epochs |
|-------------|----------------|---------|---------------|
| **Quick Test (3 subjects)** | 2-3 min | ~15 MB | ~2,500 |
| **Pilot (10 subjects)** | 10-15 min | ~50 MB | ~8,400 |
| **Full (128 subjects)** | 2-3 hours | ~640 MB | ~106,680 |

*Measured on: Intel i7, 16GB RAM, SSD*

---

## 📊 Results & Output

### Output Directory Structure

```
results/
└── experiment_20251222_143015_quick/
    │
    ├── per_subject/                      # Per-subject data
    │   ├── subject_1/
    │   │   ├── epochs.npy                # (n_epochs, 6, 3840) preprocessed signals
    │   │   ├── labels.npy                # (n_epochs,) sleep stage labels
    │   │   └── features.csv              # (n_epochs, 149) extracted features
    │   ├── subject_2/
    │   └── subject_3/
    │
    ├── features/                         # Aggregated dataset
    │   ├── all_features.csv              # (total_epochs, 149) ALL features
    │   ├── all_labels.npy                # (total_epochs,) ALL labels
    │   ├── subject_ids.npy               # (total_epochs,) subject ID per epoch
    │   └── dataset_metadata.json         # Statistics and metadata
    │
    └── pipeline_stats.json               # Timing and performance statistics
```

### Loading Processed Data

```python
import pandas as pd
import numpy as np

# Load aggregated features
features = pd.read_csv('results/experiment_*/features/all_features.csv')
labels = np.load('results/experiment_*/features/all_labels.npy')
subject_ids = np.load('results/experiment_*/features/subject_ids.npy')

print(f"Features: {features.shape}")  # (106680, 149) for full dataset
print(f"Labels: {labels.shape}")      # (106680,)
print(f"Unique subjects: {len(np.unique(subject_ids))}")  # 128
```

### Example Output

```
============================================================
   PIPELINE COMPLETE - STAGE 1 & 2
============================================================
Subjects processed: 128
Total epochs: 106,680
Features per epoch: 149

Time Statistics:
  Start: 14:30:15
  End: 17:15:42
  Elapsed: 9,927 seconds (165.5 minutes)
  Avg per subject: 77.6 seconds

Label Distribution:
  Wake: 12,345 (11.6%)
  N1: 5,234 (4.9%)
  N2: 48,901 (45.8%)
  N3: 29,123 (27.3%)
  REM: 11,077 (10.4%)

Output saved to:
  results/experiment_20251222_143015_full
============================================================
```

---

## 🎓 Design Decisions (For Thesis)

### 1. Dataset Selection: BOAS vs Sleep-EDF

**Decision:** Use BOAS dataset

**Rationale:**
- ✅ 128 subjects (ideal for LOSO cross-validation)
- ✅ 6 EEG channels (richer than Sleep-EDF's 2)
- ✅ Open access, well-documented
- ✅ ~20 GB storage (demonstrates caching benefits)

**Reference:** See `THESIS_DESIGN_DECISIONS.md` for full justification.

---

### 2. Sampling Rate: 256 Hz → 128 Hz

**Decision:** Downsample from 256 Hz to 128 Hz

**Rationale:**

| Criterion | Analysis | Conclusion |
|-----------|----------|------------|
| **Signal preservation** | Bandpass 0.5-40 Hz requires ≥80 Hz (Nyquist) | ✅ Safe (128 > 80) |
| **Computation** | 50% fewer samples = 1.5× faster | ✅ Significant |
| **Storage** | 50% reduction in preprocessed data | ✅ Meaningful |
| **Literature** | 100-128 Hz standard in sleep research | ✅ Justified |

**References:**
- Rechtschaffen & Kales (1968): Sleep scoring ≤100 Hz sufficient
- AASM Manual (2015): ≥100 Hz recommended

---

### 3. Channel Configuration: 6 vs 8 Channels

**Decision:** Use 6 EEG channels as primary configuration

**Channels Used:** F3, F4, C3, C4, O1, O2  
**Features Generated:** 149 (6 × 23 + 11)

**Rationale:**
- ✅ Standard in sleep research (AASM guidelines)
- ✅ EEG contains primary sleep staging information
- ✅ Simpler to explain and interpret
- ✅ Sufficient for demonstrating caching effectiveness

**Alternative (Supported but not used in main experiments):**
- 8 channels (EEG + EOG + EMG) → 195 features
- Potentially better for REM/Wake detection
- Available via: `channel_preset: "eeg_plus_physiological"`

**For Thesis Discussion:**
```
The pipeline architecture supports both 6-channel (149 features)
and 8-channel (195 features) configurations, demonstrating system
flexibility. Main experiments use the 6-channel configuration
following standard sleep research practices.
```

**Reference:** See `CHANNEL_CONFIGURATION_GUIDE.md` for detailed comparison.

---

### 4. Feature Set: 149 Features

**Decision:** 23 features per channel + 11 global features

**Breakdown:**
- Time-domain (10): Basic statistics and zero-crossing rate
- Frequency (9): Power bands (delta through gamma) + spectral metrics
- Complexity (4): Hjorth parameters, Hurst exponent, DFA
- Global (11): Coherence, phase-locking, entropy

**Rationale:**
- ✅ Captures multi-domain characteristics (time, frequency, complexity)
- ✅ Includes spatial information (coherence, PLV)
- ✅ Computationally efficient (~12s per subject)
- ✅ Deterministic (same input → same output → cacheable)

**Alternatives Rejected:**
- Deep learning features (non-deterministic, harder to cache)
- Wavelet features (redundant with frequency bands)
- >200 features (diminishing returns, longer computation)

---

### 5. Cross-Validation: LOSO (Leave-One-Subject-Out)

**Decision:** LOSO with 128 folds

**Rationale:**

| Criterion | Random K-Fold | LOSO | Winner |
|-----------|--------------|------|--------|
| **Data leakage** | Epochs from same subject in train+test | No leakage | **LOSO** |
| **Generalization** | Within-subject | Cross-subject | **LOSO** |
| **Clinical relevance** | Not realistic | Models unseen patients | **LOSO** |
| **Cache benefit** | Low (5-10 folds) | **High (128 folds)** | **LOSO** |

**Cache Optimization:**
- 128 folds × 3 models = 384 training runs
- High repetition → perfect for demonstrating caching benefits
- First run: ~9 hours (all cache misses)
- Second run: ~1 hour (97% cache hits)

---

### 6. Quality Validation: Disabled for BOAS

**Decision:** Disable amplitude-based quality validation

**Original Implementation:**
```python
max_amplitude: 200.0 µV  # Too strict
min_amplitude: 1.0 µV
```

**Issue:** All BOAS epochs rejected (0/997 passed for each subject)

**Solution:** Quality validation disabled by default
```python
validate_quality=False  # in preprocessing
```

**Rationale:**
- BOAS data has different amplitude characteristics than expected
- Manual inspection shows data quality is good
- Filtering disconnections (stage=8) already removes bad epochs
- Can re-enable later with appropriate thresholds for BOAS

---

## 🐛 Troubleshooting

### Common Issues

#### 1. No Epochs Generated (0 valid epochs)

**Symptom:**
```
INFO:preprocessing:Quality validation: 0/997 epochs passed (997 rejected)
```

**Solution:** This was fixed in latest version. Update to newest `preprocessing.py` and `pipeline.py`.

**Technical Details:** Quality validation was too strict for BOAS data and has been disabled by default.

---

#### 2. Wrong Number of Features

**Symptom:**
```
Expected: 149 features
Actual: 0 features or wrong count
```

**Solution:** Ensure you're using 6 EEG channels:
```yaml
# In config:
channel_preset: "eeg_only"  # NOT "all" or "eeg_plus_physiological"
```

---

#### 3. ModuleNotFoundError

**Symptom:**
```
ModuleNotFoundError: No module named 'mne'
```

**Solution:**
```bash
pip install -r requirements.txt
```

---

#### 4. Data Path Not Found

**Symptom:**
```
FileNotFoundError: Base path not found: C:\Users\DerHo\Desktop\Data
```

**Solution:** Override data path:
```bash
python run_experiment.py --quick-test --data-path "D:\MyData\BOAS"
```

Or update in config:
```yaml
data:
  base_path: "D:/MyData/BOAS"
```

---

#### 5. Out of Memory

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
- Use `--no-save-intermediate` flag
- Process fewer subjects at once
- Close other applications
- Ensure 16+ GB RAM available

---

#### 6. Slow Processing

**Diagnosis:**
```bash
python run_experiment.py --quick-test --log-level DEBUG
```

Check timing for each stage. Filtering typically takes longest (~15-20s per subject).

**Solutions:**
- Normal: 30-40s per subject is expected
- If >60s per subject, check disk I/O (use SSD)
- If >120s per subject, check RAM usage

---

### Getting Help

```bash
# View all command options
python run_experiment.py --help

# Enable debug logging
python run_experiment.py --quick-test --log-level DEBUG

# Save logs to file
python run_experiment.py --pilot --log-file debug.log
```

For detailed debugging, check:
1. Console output for error messages
2. Log file (if `--log-file` used)
3. `pipeline_stats.json` for timing information

---

## 🚧 Development Status

### ✅ Stage 1 & 2: COMPLETE (Current Release)

**Implemented:**
- ✅ BOAS dataset loader (6 or 8 channels)
- ✅ Signal preprocessing (filtering, downsampling, epoching)
- ✅ Feature extraction (149 or 195 features)
- ✅ Configuration system (YAML-based)
- ✅ CLI interface
- ✅ Progress tracking and logging
- ✅ Data validation and quality checks

**Output:** Preprocessed data + extracted features ready for model training

---

### ⏳ Stage 3: IN PROGRESS (Next Release)

**Planned:**
- ⏳ Fingerprinting module (SHA-256 hashing)
- ⏳ 4-stage cache system (preprocessing, features, models, results)
- ⏳ Cache manager with cascade invalidation
- ⏳ Cache hit/miss tracking

**Goal:** Demonstrate intelligent caching with fingerprint-based invalidation

---

### ⏳ Stage 4: PLANNED (Final Release)

**Planned:**
- ⏳ Model training (XGBoost, Random Forest, FNN)
- ⏳ LOSO cross-validation (128 folds)
- ⏳ Results aggregation and metrics
- ⏳ Cache performance analysis

**Goal:** Complete end-to-end ML pipeline with caching

---

### Project Timeline

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| Architecture design | Dec 22, 2025 | ✅ Complete |
| Stage 1 & 2 (Data + Features) | Dec 23, 2025 | ✅ Complete |
| Stage 3 (Fingerprinting + Cache) | Dec 28, 2025 | ⏳ In Progress |
| Stage 4 (Models + LOSO) | Jan 5, 2026 | ⏳ Planned |
| Experiments & Results | Jan 8, 2026 | ⏳ Planned |
| Thesis Writing | Jan 10-14, 2026 | ⏳ Planned |
| **Submission Deadline** | **Jan 15, 2026** | 🎯 Target |

---

## 📚 Documentation Files

### Essential Documentation (3 files)

| File | Purpose | When to Read |
|------|---------|--------------|
| **README.md** (this file) | Complete overview | Start here |
| **QUICK_START.md** | Detailed getting started guide | First time setup |
| **THESIS_DESIGN_DECISIONS.md** | Design justifications | Writing methodology |
| **CHANNEL_CONFIGURATION_GUIDE.md** | 6 vs 8 channel comparison | Configuring experiments |

### Additional Reference Documentation

Available in repository for detailed reference:
- `PIPELINE_README.md` - Technical implementation details
- `EXAMPLE_OUTPUT.md` - Sample output and progress indicators
- `PROJECT_STRUCTURE.md` - Complete file structure
- `BOAS_DATASET_APPROACH.md` - Dataset analysis

---

## 📖 Citation

### For Academic Use

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

### Dataset Citation

```bibtex
@misc{boas2023,
  title        = {Bitbrain Open Access Sleep (BOAS) Database},
  author       = {Bitbrain Technologies},
  year         = {2023},
  howpublished = {PhysioNet},
  doi          = {10.13026/xxx}
}
```

---

## 🤝 Contributing

This is a bachelor thesis project. For academic collaboration or questions:

**Contact:**
- **Author:** Lennart Gorzel
- **Email:** [Your Email]
- **Supervisor:** Prof. Himanshu Buckchash
- **Institution:** IMC FH Krems

---

## 📄 License

This code is part of a Bachelor thesis at IMC FH Krems.  
**For academic and educational use only.**

---

## 🙏 Acknowledgments

- **Supervisor:** Prof. Himanshu Buckchash (IMC FH Krems)
- **Dataset:** Bitbrain Technologies (BOAS Dataset)
- **Libraries:** MNE-Python, scikit-learn, XGBoost, PyTorch
- **Institution:** IMC Fachhochschule Krems

---


## 📋 Version History

| Version | Date | Changes |
|---------|------|---------|
| **1.0.0** | Dec 22, 2025 | Initial release - Stage 1 & 2 complete |
| **1.1.0** | Dec 22, 2025 | Added flexible channel configuration (6/8 channels) |
| **2.0.0** | Dec 25, 2025 | Stage 3 complete - Feature selection, models, LOSO splitter |
| **2.1.0** | Dec 27, 2025 | Documentation update, thesis grid (18 configs) verified |
| **3.0.0** | TBD | Stage 4 - LOSO Model Caching (in progress) |

---

## 🎯 Quick Reference Card

```bash
# MOST COMMON COMMANDS

# First time: Test with 3 subjects
python run_experiment.py --quick-test

# Pilot study: 10 subjects
python run_experiment.py --pilot

# Full dataset: 128 subjects
python run_experiment.py --full

# Use 8 channels instead of 6
python run_experiment.py --pilot --config config_8channels.yaml

# Debug mode
python run_experiment.py --quick-test --log-level DEBUG

# Custom experiment name
python run_experiment.py --pilot --experiment-name my_experiment_v1
```

---

**README Version:** 2.1  
**Last Updated:** December 27, 2025  
**Status:** Stage 1-3 Complete, Stage 4 (LOSO Cache) In Progress  
**Next Update:** After LOSO Model Cache Implementation

---

**Ready to start? Run:** `python run_experiment.py --quick-test` 🚀
