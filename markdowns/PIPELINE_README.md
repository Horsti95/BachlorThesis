# ML Experiment Caching Pipeline - Stage 1
## Data Loading, Preprocessing, and Feature Extraction

**Bachelor Thesis:** ML Experiment Optimization with Intelligent Caching  
**Author:** Lennart Gorzel  
**Institution:** IMC FH Krems  
**Date:** December 2025

---

## Overview

This is **Stage 1** of the complete ML caching pipeline. It handles:

1. ✅ Loading BOAS dataset (128 subjects, 6 EEG channels)
2. ✅ Preprocessing (bandpass filter, notch filter, downsampling)
3. ✅ Epoch extraction (30-second windows)
4. ✅ Feature extraction (149 features per epoch)
5. ✅ Data validation and quality checks

**NOT YET IMPLEMENTED:**
- ❌ Fingerprinting (SHA-256 hashing)
- ❌ 4-stage caching system
- ❌ Model training (XGBoost, RF, FNN)
- ❌ LOSO cross-validation

These will be added in the next iteration.

---

## File Structure

```
├── config.py                    # Configuration management
├── data_loader_boas.py          # BOAS dataset loader
├── preprocessing.py             # Signal filtering, downsampling, epoching
├── feature_extractor.py         # 149 feature extraction
├── utils.py                     # Helper functions
├── pipeline.py                  # Pipeline orchestrator
├── run_experiment.py            # CLI interface
├── example_config.yaml          # Example configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- **mne** (≥1.5.0): EEG processing
- **numpy** (≥1.23.0): Numerical computing
- **scipy** (≥1.9.0): Signal processing
- **pandas** (≥1.5.0): Data manipulation
- **pyyaml** (≥6.0): Configuration files

### 2. Verify Data Path

Ensure your BOAS dataset is at:
```
C:\Users\DerHo\Desktop\Data\
├── sub-1/eeg/*.edf
├── sub-2/eeg/*.edf
└── ...
```

---

## Quick Start

### Test on 3 Subjects (Quick Test)

```bash
python run_experiment.py --quick-test
```

**Expected output:**
```
Processing subjects: [████████████████████] 100% (3/3)
✓ 3 subjects processed
✓ ~2,500 epochs total
✓ 149 features per epoch
✓ Time: ~2-3 minutes
```

### Pilot Run (10 Subjects)

```bash
python run_experiment.py --pilot
```

**Expected output:**
```
Processing subjects: [████████████████████] 100% (10/10)
✓ 10 subjects processed
✓ ~8,400 epochs total
✓ 149 features per epoch
✓ Time: ~10-15 minutes
```

### Full Dataset (128 Subjects)

```bash
python run_experiment.py --full
```

**Expected output:**
```
Processing subjects: [████████████████████] 100% (128/128)
✓ 128 subjects processed
✓ ~106,680 epochs total
✓ 149 features per epoch
✓ Time: ~2-3 hours
```

---

## Using Configuration Files

### Create Custom Configuration

1. Copy example:
   ```bash
   cp example_config.yaml my_experiment.yaml
   ```

2. Edit `my_experiment.yaml`:
   ```yaml
   experiment_name: "my_custom_experiment"
   
   data:
     subjects: ['1', '2', '3', '10', '20']  # Specific subjects
   
   preprocessing:
     bandpass_low: 0.5
     bandpass_high: 35.0  # Narrower bandwidth
     target_sfreq: 100.0  # Different sampling rate
   
   features:
     correlation_threshold: 0.90  # Remove correlated features
   ```

3. Run with config:
   ```bash
   python run_experiment.py --config my_experiment.yaml
   ```

---

## Command-Line Options

### Basic Usage
```bash
python run_experiment.py [OPTIONS]
```

### Configuration Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config FILE` | Use YAML config file | `--config configs/exp1.yaml` |
| `--quick-test` | Run on 3 subjects | `--quick-test` |
| `--pilot` | Run on 10 subjects | `--pilot` |
| `--full` | Run on all 128 subjects | `--full` |

### Override Options

| Option | Description | Example |
|--------|-------------|---------|
| `--data-path PATH` | Override data directory | `--data-path /data/boas` |
| `--output-dir DIR` | Override output directory | `--output-dir ./my_results` |
| `--experiment-name NAME` | Set experiment name | `--experiment-name test1` |
| `--model TYPE` | Set model type | `--model xgboost` |

### Logging Options

| Option | Description | Example |
|--------|-------------|---------|
| `--log-level LEVEL` | Set log level | `--log-level DEBUG` |
| `--log-file FILE` | Log to file | `--log-file experiment.log` |

### Processing Options

| Option | Description |
|--------|-------------|
| `--no-save-intermediate` | Don't save per-subject data (saves disk space) |

---

## Output Structure

After running, the output directory contains:

```
results/
└── experiment_name/
    ├── per_subject/              # Per-subject data
    │   ├── subject_1/
    │   │   ├── epochs.npy        # Preprocessed epochs (n_epochs, 6, 3840)
    │   │   ├── labels.npy        # Sleep stage labels (n_epochs,)
    │   │   └── features.csv      # Extracted features (n_epochs, 149)
    │   ├── subject_2/
    │   └── ...
    │
    ├── features/                 # Aggregated dataset
    │   ├── all_features.csv      # All features (total_epochs, 149)
    │   ├── all_labels.npy        # All labels (total_epochs,)
    │   ├── subject_ids.npy       # Subject ID per epoch (total_epochs,)
    │   └── dataset_metadata.json # Dataset statistics
    │
    └── logs/                     # Log files (if --log-file used)
```

---

## Technical Specifications

### Preprocessing Pipeline

1. **Bandpass Filter:** 0.5-40 Hz (FIR filter, Hamming window)
2. **Notch Filter:** 50 Hz (Europe) or 60 Hz (US) - removes power line noise
3. **Downsampling:** 256 Hz → 128 Hz (antialiasing applied)
4. **Epoch Extraction:** 30-second windows
5. **Quality Validation:** Amplitude checks, flat signal detection

### Feature Extraction (149 Features)

**Per-Channel Features (23 × 6 channels = 138):**
- **Time-domain (10):** mean, std, var, min, max, ptp, rms, skew, kurtosis, zcr
- **Frequency (9):** delta, theta, alpha, sigma, beta, gamma power, spectral entropy, peak/median frequency
- **Complexity (4):** Hjorth mobility/complexity, Hurst exponent, DFA

**Global Features (11):**
- **Coherence (6):** F3-F4, F3-C3, F3-C4, F4-C4, C3-C4, O1-O2
- **Phase-Locking Value (3):** F3-O1, F4-O2, C3-C4
- **Global metrics (2):** entropy, complexity

---

## Performance Benchmarks

### Processing Time (per subject)

| Stage | Time | Notes |
|-------|------|-------|
| Load EDF | ~2s | Depends on disk I/O |
| Preprocessing | ~20s | Filtering + downsampling |
| Feature extraction | ~12s | 149 features × ~800 epochs |
| **Total** | **~34s** | Per subject average |

### Expected Total Times

| Dataset Size | Time | Storage |
|--------------|------|---------|
| **3 subjects** (quick test) | ~2 min | ~15 MB |
| **10 subjects** (pilot) | ~6 min | ~50 MB |
| **128 subjects** (full) | ~70 min | ~640 MB |

*Note: Times measured on standard laptop (16GB RAM, SSD). Your times may vary.*

---

## Data Format Details

### Input (BOAS Dataset)

**EDF Files:**
- Format: EDF+ (European Data Format)
- Channels: 6 EEG (PSG_F3, PSG_F4, PSG_C3, PSG_C4, PSG_O1, PSG_O2)
- Sampling rate: 256 Hz
- Duration: ~8 hours per recording

**Annotation Files (events.txt):**
```
onset  duration  begsample  endsample  offset  stage_hum  stage_ai
0      30        1          7680       0       3          3
30     30        7681       15360      0       3          3
...
```
- `stage_hum`: Human consensus (0=W, 1=N1, 2=N2, 3=N3, 4=REM, 8=Disconnection)
- `stage_ai`: AI predictions

### Output

**Preprocessed Epochs (epochs.npy):**
```python
shape: (n_epochs, 6, 3840)
dtype: float64
# 6 channels, 3840 samples (30s × 128 Hz)
```

**Labels (labels.npy):**
```python
shape: (n_epochs,)
dtype: int64
values: 0, 1, 2, 3, 4  # W, N1, N2, N3, REM
```

**Features (features.csv):**
```python
shape: (n_epochs, 149)
columns: ['F3_mean', 'F3_std', ..., 'global_entropy', 'global_complexity']
```

---

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'mne'**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**2. FileNotFoundError: Data path not found**
```bash
# Solution: Check data path
python -c "from pathlib import Path; print(Path(r'C:\Users\DerHo\Desktop\Data').exists())"

# Or override:
python run_experiment.py --data-path /your/path/to/data
```

**3. MemoryError during processing**
```bash
# Solution: Use --no-save-intermediate to save RAM
python run_experiment.py --pilot --no-save-intermediate
```

**4. Processing is slow**
```bash
# Start with quick test
python run_experiment.py --quick-test

# Check if filtering is the bottleneck:
python run_experiment.py --quick-test --log-level DEBUG
```

### Getting Help

```bash
# Show all available options
python run_experiment.py --help

# Enable debug logging for detailed output
python run_experiment.py --quick-test --log-level DEBUG
```

---

## Next Steps

### For Development:

1. ✅ **Test current stage:**
   ```bash
   python run_experiment.py --quick-test
   ```

2. ⏳ **Add fingerprinting module** (next iteration)

3. ⏳ **Add caching system** (4-stage hierarchy)

4. ⏳ **Add model training** (XGBoost, RF, FNN)

5. ⏳ **Add LOSO cross-validation**

### For Thesis:

- Use this stage to generate baseline results
- Measure processing time without caching
- Validate 149 features are computed correctly
- Document preprocessing decisions

---

## Code Quality

### Testing

```bash
# Test individual modules
python data_loader_boas.py  # Test data loading
python preprocessing.py     # Test preprocessing
python feature_extractor.py # Test feature extraction
python config.py           # Test configuration
```

### Code Style

- Python 3.10+ compatible
- Type hints throughout
- Comprehensive docstrings
- Logging at appropriate levels
- Error handling with informative messages

---

## Citation

If you use this code, please cite:

```
Gorzel, L. (2026). ML Experiment Optimization with Intelligent Caching.
Bachelor Thesis, IMC FH Krems.
```

---

## License

This code is part of a Bachelor thesis at IMC FH Krems.
For academic use only.

---

## Contact

**Author:** Lennart Gorzel  
**Supervisor:** Prof. Himanshu Buckchash  
**Institution:** IMC FH Krems  
**Date:** December 2025

---

**Document Version:** 1.0  
**Last Updated:** December 22, 2025
