# Example Pipeline Output
## What You'll See When Running the Pipeline

---

## **Quick Test Mode (3 Subjects)**

### Command:
```bash
python run_experiment.py --quick-test
```

### Expected Output:
```
============================================================
   ML EXPERIMENT PIPELINE - STAGE 1 & 2
   Data Preparation: Load → Preprocess → Extract Features
============================================================
Start time: 2025-12-22 14:30:15
Experiment: experiment_20251222_143015_quick
Output dir: ./results/experiment_20251222_143015_quick
============================================================


============================================================
PIPELINE: DATA PREPARATION
============================================================
Total subjects to process: 3
Expected features per epoch: 149
Save intermediate results: True
============================================================


============================================================
PROCESSING SUBJECT 1/3: 1
============================================================
[Step 1/3] Loading raw EEG data...
  ✓ Loaded: 916 total epochs, 256.0 Hz

[Step 2/3] Preprocessing (filtering, downsampling, epoching)...
  → Bandpass filter: 0.5-40.0 Hz
  → Notch filter: 50.0 Hz
  → Downsampling: 256.0 Hz → 128.0 Hz
  ✓ Preprocessed: 908 valid epochs, shape (908, 6, 3840)

[Step 3/3] Extracting 149 features...
  → Processing 908 epochs × 149 features...
     Progress: 200/908 epochs (22%)
     Progress: 400/908 epochs (44%)
     Progress: 600/908 epochs (66%)
     Progress: 800/908 epochs (88%)
     Progress: 908/908 epochs (100%)
  ✓ Features extracted: (908, 149)

  → Saving intermediate results...
  ✓ Saved to ./results/experiment_20251222_143015_quick/per_subject/subject_1

  SUMMARY: Subject 1
    Epochs: 908
    Features: (908, 149)
    Labels: (array([0, 1, 2, 3, 4]), array([120, 45, 380, 280, 83]))
  ✓ Subject 1/3 complete


============================================================
PROCESSING SUBJECT 2/3: 2
============================================================
[Step 1/3] Loading raw EEG data...
  ✓ Loaded: 852 total epochs, 256.0 Hz

[Step 2/3] Preprocessing (filtering, downsampling, epoching)...
  → Bandpass filter: 0.5-40.0 Hz
  → Notch filter: 50.0 Hz
  → Downsampling: 256.0 Hz → 128.0 Hz
  ✓ Preprocessed: 845 valid epochs, shape (845, 6, 3840)

[Step 3/3] Extracting 149 features...
  → Processing 845 epochs × 149 features...
     Progress: 200/845 epochs (24%)
     Progress: 400/845 epochs (47%)
     Progress: 600/845 epochs (71%)
     Progress: 800/845 epochs (95%)
     Progress: 845/845 epochs (100%)
  ✓ Features extracted: (845, 149)

  → Saving intermediate results...
  ✓ Saved to ./results/experiment_20251222_143015_quick/per_subject/subject_2

  SUMMARY: Subject 2
    Epochs: 845
    Features: (845, 149)
    Labels: (array([0, 1, 2, 3, 4]), array([95, 52, 420, 195, 83]))
  ✓ Subject 2/3 complete


============================================================
PROCESSING SUBJECT 3/3: 3
============================================================
[Step 1/3] Loading raw EEG data...
  ✓ Loaded: 781 total epochs, 256.0 Hz

[Step 2/3] Preprocessing (filtering, downsampling, epoching)...
  → Bandpass filter: 0.5-40.0 Hz
  → Notch filter: 50.0 Hz
  → Downsampling: 256.0 Hz → 128.0 Hz
  ✓ Preprocessed: 775 valid epochs, shape (775, 6, 3840)

[Step 3/3] Extracting 149 features...
  → Processing 775 epochs × 149 features...
     Progress: 200/775 epochs (26%)
     Progress: 400/775 epochs (52%)
     Progress: 600/775 epochs (77%)
     Progress: 775/775 epochs (100%)
  ✓ Features extracted: (775, 149)

  → Saving intermediate results...
  ✓ Saved to ./results/experiment_20251222_143015_quick/per_subject/subject_3

  SUMMARY: Subject 3
    Epochs: 775
    Features: (775, 149)
    Labels: (array([0, 1, 2, 3, 4]), array([110, 38, 350, 210, 67]))
  ✓ Subject 3/3 complete


============================================================
PROCESSING COMPLETE
============================================================
  Successful: 3/3 subjects
============================================================


============================================================
AGGREGATING DATA FROM ALL SUBJECTS
============================================================
  Subject 1: 908 epochs
  Subject 2: 845 epochs
  Subject 3: 775 epochs

  → Concatenating all data...

  AGGREGATED DATASET:
    Total epochs: 2,528
    Total features: 149
    Total subjects: 3

  LABEL DISTRIBUTION:
    Wake: 325 (12.9%)
    N1: 135 (5.3%)
    N2: 1,150 (45.5%)
    N3: 685 (27.1%)
    REM: 233 (9.2%)
============================================================


============================================================
SAVING AGGREGATED DATASET
============================================================
  → Saving features to CSV...
    ✓ ./results/experiment_20251222_143015_quick/features/all_features.csv
  → Saving labels...
    ✓ ./results/experiment_20251222_143015_quick/features/all_labels.npy
  → Saving subject IDs...
    ✓ ./results/experiment_20251222_143015_quick/features/subject_ids.npy
  → Saving metadata...
    ✓ ./results/experiment_20251222_143015_quick/features/dataset_metadata.json

  ALL FILES SAVED TO: ./results/experiment_20251222_143015_quick/features
============================================================


============================================================
   PIPELINE COMPLETE - STAGE 1 & 2
============================================================
Subjects processed: 3
Total epochs: 2,528
Features per epoch: 149

Time Statistics:
  Start: 14:30:15
  End: 14:33:42
  Elapsed: 207.3 seconds (3.5 minutes)
  Avg per subject: 69.1 seconds

Output saved to:
  ./results/experiment_20251222_143015_quick
============================================================

Pipeline statistics saved to: ./results/experiment_20251222_143015_quick/pipeline_stats.json
```

---

## **When Models + LOSO Are Added (Future)**

This is what you'll see once we add model training and LOSO cross-validation:

```
============================================================
   ML EXPERIMENT PIPELINE - COMPLETE
   Stage 1: Load → Stage 2: Features → Stage 3: Models → Stage 4: Aggregate
============================================================


[STAGE 1 & 2: DATA PREPARATION]
... (same as above) ...


[STAGE 3: LOSO CROSS-VALIDATION - 128 FOLDS]
============================================================

============================================================
LOSO FOLD 1/128: Testing on Subject SC4001
============================================================
[Training Data]
  Train subjects: 127 (SC4002, SC4003, ..., SC4128)
  Train epochs: 105,772
  Features: 149

[Model Training: XGBoost]
  → Training XGBoost with 127 subjects...
  → Hyperparameters: max_depth=6, n_estimators=200, lr=0.1
     Training progress: [████████████████████] 100% (200/200 trees)
  ✓ Model trained in 23.4 seconds

[Testing]
  → Testing on Subject SC4001 (840 epochs)...
  ✓ Predictions complete

[Fold Results]
  Accuracy: 78.5%
  F1-score (macro): 0.723
  Per-class F1:
    Wake: 0.85, N1: 0.42, N2: 0.81, N3: 0.89, REM: 0.65

[Cache Status]
  ✓ Preprocessed data: CACHE HIT (loaded in 0.3s)
  ✓ Features: CACHE HIT (loaded in 0.2s)
  ✗ Model: CACHE MISS (trained in 23.4s)
  → Saved model to cache for future use

  ✓ Fold 1/128 complete (Total time: 24.1s)


============================================================
LOSO FOLD 2/128: Testing on Subject SC4002
============================================================
[Training Data]
  Train subjects: 127 (SC4001, SC4003, ..., SC4128)
  Train epochs: 105,685
  Features: 149

[Model Training: XGBoost]
  → Training XGBoost with 127 subjects...
     Training progress: [████████████████████] 100% (200/200 trees)
  ✓ Model trained in 23.1 seconds

[Testing]
  → Testing on Subject SC4002 (927 epochs)...
  ✓ Predictions complete

[Fold Results]
  Accuracy: 80.2%
  F1-score (macro): 0.748
  Per-class F1:
    Wake: 0.88, N1: 0.45, N2: 0.83, N3: 0.91, REM: 0.67

[Cache Status]
  ✓ Preprocessed data: CACHE HIT (loaded in 0.3s)
  ✓ Features: CACHE HIT (loaded in 0.2s)
  ✗ Model: CACHE MISS (trained in 23.1s)
  → Saved model to cache

  ✓ Fold 2/128 complete (Total time: 23.8s)


... (Folds 3-127 continue) ...


============================================================
LOSO FOLD 128/128: Testing on Subject SC4128
============================================================
... (same structure) ...


============================================================
AGGREGATING RESULTS FROM ALL 128 FOLDS
============================================================
  → Computing average metrics...
  → Generating confusion matrix...
  → Computing per-subject statistics...

[OVERALL RESULTS]
  Mean Accuracy: 79.3% (±2.4%)
  Mean F1-score: 0.735 (±0.032)

  Per-class Performance:
    Wake:  F1=0.86 (Precision=0.88, Recall=0.84)
    N1:    F1=0.43 (Precision=0.52, Recall=0.37) ← Expected low
    N2:    F1=0.82 (Precision=0.84, Recall=0.80)
    N3:    F1=0.90 (Precision=0.91, Recall=0.89)
    REM:   F1=0.66 (Precision=0.70, Recall=0.63)

[CACHE STATISTICS]
  First Run:
    Cache hits: 0/384 (0%)
    Time: 8.7 hours (31,320 seconds)

  Second Run (same config):
    Cache hits: 384/384 (100%)
    Time: 0.9 hours (3,240 seconds)
    Time saved: 7.8 hours (89.7% reduction)


============================================================
   EXPERIMENT COMPLETE - ALL STAGES
============================================================
Total time: 8.7 hours
Output: ./results/experiment_full_xgboost/
============================================================
```

---

## **Command Options Summary**

### **Quick Test (3 subjects)**
```bash
python run_experiment.py --quick-test
# Time: ~3 minutes
# Epochs: ~2,500
```

### **Pilot (10 subjects)**
```bash
python run_experiment.py --pilot
# Time: ~10-15 minutes
# Epochs: ~8,400
```

### **Full Dataset (128 subjects)**
```bash
python run_experiment.py --full
# Time: ~2-3 hours
# Epochs: ~106,680
```

### **With Custom Config**
```bash
python run_experiment.py --config my_experiment.yaml
```

### **With Overrides**
```bash
python run_experiment.py --pilot --experiment-name my_test --log-level DEBUG
```

---

## **Output Directory Structure**

After running, you'll have:

```
results/
└── experiment_20251222_143015_quick/
    ├── per_subject/
    │   ├── subject_1/
    │   │   ├── epochs.npy          # (908, 6, 3840)
    │   │   ├── labels.npy          # (908,)
    │   │   └── features.csv        # (908, 149)
    │   ├── subject_2/
    │   └── subject_3/
    │
    ├── features/
    │   ├── all_features.csv        # (2528, 149) - ALL subjects combined
    │   ├── all_labels.npy          # (2528,)
    │   ├── subject_ids.npy         # (2528,) - which subject each epoch belongs to
    │   └── dataset_metadata.json   # Statistics
    │
    └── pipeline_stats.json         # Timing and performance stats
```

---

## **Progress Indicators Explained**

### **Per-Subject Progress:**
```
============================================================
PROCESSING SUBJECT 2/10: SC4002
============================================================
```
- Shows which subject is being processed
- Shows progress (2 out of 10)

### **Step Progress:**
```
[Step 1/3] Loading raw EEG data...
[Step 2/3] Preprocessing...
[Step 3/3] Extracting features...
```
- Shows which stage of processing
- 3 steps per subject: Load → Preprocess → Extract Features

### **Feature Extraction Progress:**
```
Progress: 200/908 epochs (22%)
Progress: 400/908 epochs (44%)
```
- Updates every 200 epochs
- Shows percentage complete

### **Fold Progress (when models added):**
```
LOSO FOLD 5/128: Testing on Subject SC4005
```
- Shows which cross-validation fold
- Which subject is being held out for testing

---

## **Color Coding (in Terminal)**

✓ = Success (green)
✗ = Failed (red)
→ = In progress (blue)
⚠ = Warning (yellow)

---

**Document Version:** 1.0  
**Date:** December 22, 2025
