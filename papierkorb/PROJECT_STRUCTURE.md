# ML Experiment Caching Pipeline - File Structure
## Clean, Organized Project Structure

**Date:** December 22, 2025  
**Status:** Stage 1 & 2 Complete (Data Loading + Feature Extraction)

---

## **📁 Current Project Structure**

```
Code/
│
├── 📄 Core Python Modules (Production Code)
│   ├── config.py                    # Configuration management (YAML-based)
│   ├── data_loader_boas.py          # BOAS dataset loader (6 channels, 256 Hz)
│   ├── preprocessing.py             # Signal filtering, downsampling, epoching
│   ├── feature_extractor.py         # 149 feature extraction
│   ├── utils.py                     # Helper functions (I/O, logging, progress)
│   ├── pipeline.py                  # Pipeline orchestrator
│   └── run_experiment.py            # CLI interface (main entry point)
│
├── 📄 Configuration Files
│   ├── example_config.yaml          # Example experiment configuration
│   └── requirements.txt             # Python dependencies
│
├── 📚 Documentation
│   ├── QUICK_START.md               # 5-minute getting started guide
│   ├── PIPELINE_README.md           # Complete pipeline documentation
│   ├── EXAMPLE_OUTPUT.md            # What output looks like
│   ├── THESIS_DESIGN_DECISIONS.md   # All design decisions for thesis
│   └── BOAS_DATASET_APPROACH.md     # Dataset analysis and approach
│
├── 📦 Archive (Old/Deprecated Files)
│   ├── data_loader_sleepedf_OLD.py  # OLD: Sleep-EDF loader (WRONG dataset)
│   └── DATA_LOADER_README.md        # OLD: Sleep-EDF documentation
│
└── 📊 Results (Generated at Runtime)
    └── results/
        └── experiment_YYYYMMDD_HHMMSS/
            ├── per_subject/         # Per-subject preprocessed data
            ├── features/            # Aggregated features dataset
            └── logs/                # Execution logs

```

---

## **✅ ACTIVE FILES (Use These)**

### **Core Modules (9 files)**

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| **run_experiment.py** | ~250 | CLI entry point | ✅ Ready |
| **pipeline.py** | ~450 | Orchestrates entire pipeline | ✅ Ready |
| **config.py** | ~350 | Configuration management | ✅ Ready |
| **data_loader_boas.py** | ~550 | BOAS dataset loader | ✅ Ready |
| **preprocessing.py** | ~350 | Signal preprocessing | ✅ Ready |
| **feature_extractor.py** | ~650 | 149 feature extraction | ✅ Ready |
| **utils.py** | ~400 | Helper functions | ✅ Ready |
| **example_config.yaml** | ~100 | Config template | ✅ Ready |
| **requirements.txt** | ~20 | Dependencies | ✅ Ready |
| **Total** | **~3,120** | **Fully functional** | ✅ |

### **Documentation (5 files)**

| File | Purpose | Audience |
|------|---------|----------|
| **QUICK_START.md** | Get started in 5 minutes | New users |
| **PIPELINE_README.md** | Complete documentation | All users |
| **EXAMPLE_OUTPUT.md** | Show expected output | Users running experiments |
| **THESIS_DESIGN_DECISIONS.md** | Design justifications | Thesis writing |
| **BOAS_DATASET_APPROACH.md** | Dataset analysis | Technical reference |

---

## **❌ ARCHIVED FILES (Don't Use)**

These files are in `archive/` directory and should NOT be used:

| File | Why Archived | Date Archived |
|------|--------------|---------------|
| **data_loader_sleepedf_OLD.py** | Wrong dataset (Sleep-EDF instead of BOAS) | Dec 22, 2025 |
| **DATA_LOADER_README.md** | Documentation for wrong dataset | Dec 22, 2025 |

**Note:** These were created based on incorrect assumption about dataset. Your dataset is BOAS, not Sleep-EDF.

---

## **🚀 Quick Start**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Run Quick Test (3 subjects, 2-3 minutes)**
```bash
python run_experiment.py --quick-test
```

### **3. Check Results**
```bash
dir results\
```

**See `QUICK_START.md` for detailed instructions.**

---

## **📋 File Dependencies**

```
run_experiment.py
    ├── imports: config.py
    ├── imports: pipeline.py
    └── imports: utils.py

pipeline.py
    ├── imports: data_loader_boas.py
    ├── imports: preprocessing.py
    ├── imports: feature_extractor.py
    ├── imports: config.py
    └── imports: utils.py

preprocessing.py
    ├── imports: config.py
    └── uses: mne (external library)

feature_extractor.py
    └── uses: numpy, scipy, pandas (external libraries)

data_loader_boas.py
    └── uses: mne, pandas (external libraries)
```

**Key Point:** All modules import from `data_loader_boas.py` (CORRECT), NOT from archived `data_loader_sleepedf_OLD.py`.

---

## **🎯 What Each File Does**

### **User-Facing Files:**

1. **run_experiment.py** - YOU RUN THIS
   - Command-line interface
   - Parses arguments (--quick-test, --pilot, --full)
   - Orchestrates the experiment
   - Prints progress and results

2. **example_config.yaml** - YOU EDIT THIS
   - Experiment configuration
   - Data paths, preprocessing params
   - Feature settings, model params
   - Copy and customize for your experiments

### **Core Pipeline Files:**

3. **pipeline.py** - Main orchestrator
   - Coordinates all modules
   - Processes all subjects
   - Aggregates results
   - Saves outputs

4. **data_loader_boas.py** - Loads BOAS data
   - Reads EDF files (6 channels, 256 Hz)
   - Parses annotations (stage_hum)
   - Filters disconnections (stage=8)
   - Validates data quality

5. **preprocessing.py** - Signal processing
   - Bandpass filter (0.5-40 Hz)
   - Notch filter (50 Hz)
   - Downsample (256→128 Hz)
   - Extract 30-second epochs

6. **feature_extractor.py** - Extracts 149 features
   - Time-domain: 60 features (10 per channel)
   - Frequency: 54 features (9 per channel)
   - Complexity: 24 features (4 per channel)
   - Global: 11 features (coherence, PLV, etc.)

7. **config.py** - Configuration system
   - YAML parsing
   - Validation
   - Default settings
   - Parameter management

8. **utils.py** - Helper functions
   - File I/O (save/load)
   - Logging setup
   - Progress tracking
   - Time/byte formatting

---

## **📊 Output Files Generated**

When you run an experiment, these files are created in `results/`:

```
results/experiment_YYYYMMDD_HHMMSS/
│
├── per_subject/                      # Individual subject data
│   ├── subject_1/
│   │   ├── epochs.npy                # (n_epochs, 6, 3840) - preprocessed signals
│   │   ├── labels.npy                # (n_epochs,) - sleep stages
│   │   └── features.csv              # (n_epochs, 149) - extracted features
│   ├── subject_2/
│   └── ...
│
├── features/                         # Aggregated dataset
│   ├── all_features.csv              # (total_epochs, 149) - ALL features
│   ├── all_labels.npy                # (total_epochs,) - ALL labels
│   ├── subject_ids.npy               # (total_epochs,) - subject per epoch
│   └── dataset_metadata.json         # Dataset statistics
│
└── pipeline_stats.json               # Timing and performance stats
```

---

## **🔄 Workflow**

```
1. USER EDITS CONFIG
   ├── Copy example_config.yaml
   ├── Set data path, subjects, parameters
   └── Save as my_experiment.yaml

2. USER RUNS EXPERIMENT
   └── python run_experiment.py --config my_experiment.yaml

3. PIPELINE EXECUTES
   ├── Load Data (data_loader_boas.py)
   ├── Preprocess (preprocessing.py)
   ├── Extract Features (feature_extractor.py)
   ├── Save Results (pipeline.py)
   └── Print Summary

4. USER CHECKS RESULTS
   └── Open results/experiment_*/features/all_features.csv
```

---

## **⏳ What's Next (Future Stages)**

### **Stage 3: Fingerprinting & Caching**
- `fingerprint.py` - SHA-256 hashing of configurations
- `cache_manager.py` - 4-stage cache system
- Cache hit/miss tracking
- Cascade invalidation

### **Stage 4: Model Training**
- `model_xgboost.py` - XGBoost wrapper
- `model_rf.py` - Random Forest wrapper
- `model_fnn.py` - Neural Network wrapper

### **Stage 5: Cross-Validation**
- `cross_validation.py` - LOSO implementation (128 folds)
- `aggregation.py` - Results aggregation
- Performance metrics computation
- Cache performance analysis

---

## **📝 Code Statistics**

### **Total Project Size:**
- **Python code:** ~3,120 lines
- **Documentation:** ~15,000 words
- **Configuration:** ~100 lines YAML
- **Tests included:** In each module's `if __name__ == "__main__"`

### **Code Quality:**
- ✅ Type hints throughout
- ✅ Comprehensive docstrings
- ✅ Error handling
- ✅ Logging at all levels
- ✅ Progress tracking
- ✅ Validation checks

---

## **🎓 For Your Thesis**

### **Files to Reference:**

1. **THESIS_DESIGN_DECISIONS.md**
   - All design decisions with justifications
   - Use for Methodology chapter
   - Cite specific decisions

2. **BOAS_DATASET_APPROACH.md**
   - Dataset specifications
   - Why BOAS was chosen
   - Technical details

3. **PIPELINE_README.md**
   - Implementation details
   - Technical specifications
   - Performance benchmarks

### **Code to Include in Thesis:**

**Minimal code snippets recommended:**
- Configuration example (5-10 lines from `example_config.yaml`)
- Feature extraction outline (10-15 lines showing structure)
- Pipeline orchestration (10-15 lines showing flow)
- **Total code in thesis:** ~30-40 lines maximum

**Put full code in appendix or supplementary materials.**

---

## **✅ Verification Checklist**

Before running experiments, verify:

- [ ] `archive/` folder contains old Sleep-EDF files
- [ ] No imports of `data_loader.py` (old name) anywhere
- [ ] All imports use `data_loader_boas.py`
- [ ] `requirements.txt` dependencies installed
- [ ] Data path exists and is accessible
- [ ] ~1 GB free disk space for full dataset

---

## **🔧 Maintenance**

### **Adding New Features:**
1. Add to `feature_extractor.py`
2. Update `FeatureConfig` in `config.py`
3. Update documentation
4. Update expected feature count (149 → new count)

### **Adding New Models:**
1. Create `model_yourmodel.py` in new `models/` directory
2. Add to `ModelConfig` choices in `config.py`
3. Update pipeline to support new model
4. Document in README

### **Updating Documentation:**
- Update `PIPELINE_README.md` for technical changes
- Update `QUICK_START.md` if workflow changes
- Update `EXAMPLE_OUTPUT.md` if output format changes
- Update `THESIS_DESIGN_DECISIONS.md` if design changes

---

**Document Version:** 1.0  
**Last Updated:** December 22, 2025  
**Status:** Stage 1 & 2 Complete - Ready for Use  
**Next Stage:** Fingerprinting & Caching
