# BOAS Dataset: Complete Approach Update

## 🎯 **Critical Discovery: You're Using BOAS, Not Sleep-EDF!**

Your data is the **Bitbrain Open Access Sleep (BOAS)** dataset, which has:
- ✅ **6 EEG channels** (PSG_F3, PSG_F4, PSG_C3, PSG_C4, PSG_O1, PSG_O2)
- ✅ **256 Hz original sampling rate**
- ✅ **149 features** (6 × 23 + 11 global) - YOUR V5.7 ARCHITECTURE WAS CORRECT!
- ✅ **Human consensus + AI labels**
- ✅ **128 subjects**

**Bottom line:** Your old code and v5.7 architecture were CORRECT. I was wrong to assume Sleep-EDF.

---

## **Answering Your 3 Questions**

### **1. Recalculating Features (149 per epoch with downsampling)**

**Answer: YES - This is exactly what we'll build!**

**Specifications:**
- **Input:** 6 channels (PSG_F3, PSG_F4, PSG_C3, PSG_C4, PSG_O1, PSG_O2)
- **Original sampling rate:** 256 Hz
- **Target sampling rate:** 128 Hz (your choice - keep old approach)
- **Epoch duration:** 30 seconds
- **Epoch samples:** 3840 samples (128 Hz × 30s)
- **Features per epoch:** 149 total
  - Time-domain: 10 per channel × 6 = 60
  - Frequency: 9 per channel × 6 = 54
  - Complexity: 4 per channel × 6 = 24
  - Global: 11 features
  - **Total: 60 + 54 + 24 + 11 = 149 features**

**What we're building:**
```
Raw EDF (256 Hz, 6 channels)
    ↓ [Preprocessing]
Filtered & Downsampled (128 Hz, 6 channels)
    ↓ [Epoching]
30-second epochs (3840 samples each)
    ↓ [Feature Extraction]
149 features per epoch
    ↓ [Caching with Fingerprinting]
Cached features (Stage 2 cache)
```

---

### **2. Fix 6-Channel Assumption**

**Answer: NOTHING TO FIX - It Was Already Correct!**

Your v5.7 architecture and old code were RIGHT all along. The BOAS dataset has exactly 6 EEG channels.

**What we're keeping:**
- ✅ 6 channels (PSG_F3, PSG_F4, PSG_C3, PSG_C4, PSG_O1, PSG_O2)
- ✅ 149 features
- ✅ All feature calculations based on 6 channels

**What we're updating:**
- ✅ Data loader now correctly loads BOAS format (not Sleep-EDF)
- ✅ Filter stage=8 (disconnections/bathroom breaks)
- ✅ Use stage_hum column (human consensus labels)
- ✅ Handle BOAS-specific event file format

---

### **3. Orchestrating Pipeline + User Interface**

**Answer: YES - Absolutely Essential!**

We need a **3-tier system**:

```
┌─────────────────────────────────────────────────────────┐
│  TIER 1: User Interface                                 │
│  ├── run_experiment.py (CLI interface)                  │
│  ├── experiment_config.yaml (configuration)             │
│  └── Progress reporting, error handling                 │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  TIER 2: Pipeline Orchestrator                          │
│  ├── pipeline.py (main orchestrator)                    │
│  ├── Connects all modules                               │
│  ├── Manages data flow through 4 cache stages           │
│  ├── Handles fingerprinting at each stage               │
│  └── Tracks dependencies and invalidations              │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  TIER 3: Core Modules                                   │
│  ├── data_loader.py (BOAS EDF loading)                  │
│  ├── preprocessing.py (filter, downsample, epoch)       │
│  ├── feature_extractor.py (149 features)                │
│  ├── fingerprint.py (SHA-256 hashing)                   │
│  ├── cache_manager.py (4-stage cache)                   │
│  ├── model_xgboost.py, model_rf.py, model_fnn.py        │
│  └── cross_validation.py (LOSO)                         │
└─────────────────────────────────────────────────────────┘
```

**User workflow:**
```bash
# 1. Edit config
vim configs/experiment.yaml

# 2. Run experiment
python run_experiment.py --config configs/experiment.yaml

# 3. See progress
[Stage 1/4] Preprocessing: 100% ████████████ 128/128 subjects
[Stage 2/4] Feature Extraction: 100% ████████████ 128/128 subjects
[Stage 3/4] Model Training: 75% █████████░░░ 96/128 folds
Cache hits: 85% | Time saved: 4.2 hours

# 4. Results saved to
results/experiment_20251222_143052/
```

---

## **What to Keep from Your Old Code**

### **KEEP (100% Correct):** ✅

1. **Downsampling approach:**
   ```python
   TARGET_SAMPLING_RATE = 128  # Hz
   EPOCH_SAMPLES = TARGET_SAMPLING_RATE * EPOCH_DURATION  # 3840 samples
   ```

2. **Label mapping:**
   ```python
   STAGE_MAPPING = {
       0: 'Wake',
       1: 'N1',
       2: 'N2',
       3: 'N3',
       4: 'REM',
       8: -1,  # Disconnection → filter
       -2: -1  # Artifact → filter
   }
   ```

3. **Quality checks:**
   - Amplitude thresholds
   - Artifact detection
   - Missing label handling
   - Annotation coverage validation

4. **Feature extraction:**
   - Time-domain: mean, std, var, min, max, ptp, rms, skew, kurtosis, zcr (10)
   - Frequency: delta, theta, alpha, sigma, beta, gamma, spectral_entropy, peak_freq, median_freq (9)
   - Complexity: hjorth_mobility, hjorth_complexity, hurst, dfa (4)
   - Global: coherence, PLV, entropy, complexity (11)

5. **Cross-validation:**
   - LOSO (Leave-One-Subject-Out)
   - 128 folds (one per subject)

---

### **ADD (New Requirements):** 🆕

1. **Fingerprinting system:**
   ```python
   # Compute SHA-256 hash of:
   # - preprocessing params (filter cutoffs, downsample rate)
   # - feature extraction params (which features, window sizes)
   # - model hyperparameters
   # - held-out subject ID (for LOSO)
   fingerprint = compute_fingerprint(config, held_out_subject)
   ```

2. **4-stage cache hierarchy:**
   ```
   cache/
   ├── preprocessing/     # Stage 1: Filtered signals
   │   └── subject_001_<fingerprint>.pkl
   ├── features/          # Stage 2: 149 features
   │   └── subject_001_<fingerprint>.pkl
   ├── models/            # Stage 3: Trained models
   │   └── xgb_fold_001_<fingerprint>.pkl
   └── aggregation/       # Stage 4: Results
       └── results_<fingerprint>.pkl
   ```

3. **Cascade invalidation:**
   - If preprocessing params change → invalidate all downstream caches
   - If feature params change → invalidate feature + model caches
   - If model params change → invalidate only model cache

4. **Cache metrics:**
   - Track hit rate per stage
   - Measure time saved
   - Report storage usage

---

## **Updated Architecture Specifications**

### **Data Specifications**

| Property | Value |
|----------|-------|
| **Dataset** | Bitbrain Open Access Sleep (BOAS) |
| **Subjects** | 128 recordings (108 unique individuals) |
| **Channels** | 6 EEG (PSG_F3, PSG_F4, PSG_C3, PSG_C4, PSG_O1, PSG_O2) |
| **Original Sampling Rate** | 256 Hz |
| **Target Sampling Rate** | 128 Hz (downsampled) |
| **Epoch Duration** | 30 seconds |
| **Epoch Samples** | 3840 (at 128 Hz) |
| **Sleep Stages** | 5 (Wake, N1, N2, N3, REM) |
| **Labels** | Human consensus (stage_hum) |
| **Invalid Labels** | 8 (disconnection), -2 (artifact) → filter |

---

### **Feature Specifications**

| Feature Group | Per Channel | Total |
|---------------|-------------|-------|
| **Time-domain** | 10 | 60 (10 × 6 channels) |
| **Frequency** | 9 | 54 (9 × 6 channels) |
| **Complexity** | 4 | 24 (4 × 6 channels) |
| **Global** | — | 11 |
| **TOTAL** | 23 | **149 features** |

---

### **Preprocessing Pipeline**

```python
# Stage 1: Load raw data
raw = load_raw_edf(subject_id)  # 256 Hz, 6 channels

# Stage 2: Bandpass filter
raw_filtered = apply_bandpass(raw, low=0.5, high=30.0)

# Stage 3: Notch filter (line noise)
raw_filtered = apply_notch(raw_filtered, freq=50.0)  # or 60 Hz for US

# Stage 4: Downsample
raw_downsampled = resample(raw_filtered, target_sfreq=128.0)

# Stage 5: Epoch extraction
epochs = extract_epochs(raw_downsampled, duration=30.0)

# Stage 6: Load annotations
annotations = load_annotations(subject_id)

# Stage 7: Filter invalid epochs
valid_epochs, valid_labels = filter_valid_epochs(epochs, annotations)

# Stage 8: Cache with fingerprint
cache_preprocessed_data(subject_id, valid_epochs, valid_labels, fingerprint)
```

---

### **Cache Storage Estimates**

**Per Subject:**
- Raw signal (6 channels, 256 Hz, 8 hours): ~360 MB
- Preprocessed (6 channels, 128 Hz, filtered): ~180 MB
- Features (149 × ~800 epochs): ~1 MB
- Model (XGBoost): ~5-10 MB

**Total for 128 subjects:**
- Stage 1 (preprocessing): ~23 GB (or skip if memory-constrained)
- Stage 2 (features): ~128 MB ✅ **Recommended to cache**
- Stage 3 (models): ~1.3 GB
- Stage 4 (results): ~10 MB

**Recommendation:** Cache features (Stage 2) and models (Stage 3). Skip Stage 1 if disk space is limited.

---

## **Next Steps: What I'll Create**

### **Immediate (Core Modules)**

1. ✅ **data_loader_boas.py** (already created)
2. ⏳ **preprocessing.py** (filter, downsample, epoch)
3. ⏳ **feature_extractor.py** (149 features)
4. ⏳ **fingerprint.py** (SHA-256 hashing)
5. ⏳ **cache_manager.py** (4-stage cache)
6. ⏳ **config.py** (YAML configuration)

### **Integration (Pipeline)**

7. ⏳ **pipeline.py** (orchestrator)
8. ⏳ **run_experiment.py** (CLI interface)
9. ⏳ **experiment_config.yaml** (example config)

### **Models**

10. ⏳ **model_xgboost.py** (XGBoost wrapper)
11. ⏳ **model_rf.py** (Random Forest wrapper)
12. ⏳ **model_fnn.py** (Feedforward Neural Network)

### **Cross-Validation**

13. ⏳ **cross_validation.py** (LOSO implementation)
14. ⏳ **aggregation.py** (results aggregation)

---

## **Validation Against Your Old Code**

| Aspect | Your Old Code | New Implementation | Status |
|--------|---------------|-------------------|---------|
| **Channels** | 6 (F3,F4,C3,C4,O1,O2) | 6 PSG channels | ✅ Same |
| **Sampling rate** | 256→128 Hz | 256→128 Hz | ✅ Same |
| **Epoch duration** | 30s | 30s | ✅ Same |
| **Epoch samples** | 3840 | 3840 | ✅ Same |
| **Features** | 105? | 149 | ⚠️ Different |
| **Label filtering** | -1 for invalid | Filter stage=8,-2 | ✅ Improved |
| **Quality checks** | Yes | Yes | ✅ Kept |
| **Caching** | SmartCachingSystem | 4-stage fingerprinted | 🆕 Enhanced |
| **Fingerprinting** | No | Yes | 🆕 New |
| **Orchestration** | Scattered | Unified pipeline | 🆕 New |

---

## **Key Decisions to Confirm**

Before I create all modules, please confirm:

### **1. Downsampling Rate**
- ✅ **128 Hz** (your old approach)
- ❌ 100 Hz
- ❌ Keep 256 Hz

### **2. Filtering**
- **Bandpass:** 0.5-30 Hz (standard for sleep)
- **Notch:** 50 Hz (Europe) or 60 Hz (US data?)

### **3. Features**
You mentioned 105 features in old code, but with 6 channels we get 149. Did you:
- Use fewer feature types? (which ones to skip?)
- Use a subset of channels?
- Want to add more features?

### **4. Label Source**
- ✅ **stage_hum** (human consensus) - recommended
- ❌ stage_ai (AI predictions)
- ❌ Both (train on human, test on AI agreement)

### **5. Cache Strategy**
Which stages to cache?
- ✅ **Stage 2 (features):** ~128 MB ← Recommended
- ✅ **Stage 3 (models):** ~1.3 GB ← Recommended
- ❌ Stage 1 (preprocessed signals): ~23 GB (skip if space limited)
- ✅ Stage 4 (results): ~10 MB ← Always

---

## **Timeline Estimate**

**Development:**
- Core modules (1-6): 2-3 days
- Pipeline integration (7-9): 1 day
- Model wrappers (10-12): 1 day
- Cross-validation (13-14): 1 day
- **Total: 5-6 days**

**Thesis write-up:**
- Results chapter: After running experiments
- You have until January 15, 2026
- **Plenty of time!**

---

## **What You Can Do Now**

1. **Confirm the decisions above**
2. **Test the new data_loader_boas.py:**
   ```bash
   cd C:\Users\DerHo\Desktop\Data
   python data_loader_boas.py
   ```
3. **Tell me which modules to create first** (all at once? step by step?)
4. **Share any constraints** (disk space limits, RAM limits, GPU availability)

---

## **Summary**

✅ **Your v5.7 architecture was CORRECT** - 6 channels, 149 features
✅ **Your old code approach was SOUND** - keep downsampling, filtering, quality checks
🆕 **Adding fingerprinting + 4-stage cache** - your main thesis contribution
🆕 **Creating orchestrating pipeline** - user-friendly interface
📊 **BOAS dataset is perfect** for demonstrating caching benefits (128 subjects, LOSO)

**Ready to build the complete system?** Tell me your decisions and I'll create all the modules!
