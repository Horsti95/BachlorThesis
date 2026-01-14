# Quick Start Guide
## Running Your First Experiment in 5 Minutes

---

## **Prerequisites**

1. **Python installed** (3.10+ recommended)
2. **BOAS dataset** at: `C:\Users\DerHo\Desktop\Data`
3. **Dependencies installed:**
   ```bash
   pip install -r requirements.txt
   ```

---

## **Step 1: Test with 3 Subjects (2-3 minutes)**

Open your terminal and run:

```bash
cd "C:\Users\DerHo\Desktop\Lego\Studium\FH\8. Semester Krems\BachlorThesis\Code"

python run_experiment.py --quick-test
```

**What you'll see:**
```
============================================================
   ML EXPERIMENT PIPELINE - STAGE 1 & 2
============================================================
...
PROCESSING SUBJECT 1/3: 1
  [Step 1/3] Loading raw EEG data...
  [Step 2/3] Preprocessing...
  [Step 3/3] Extracting 149 features...
  ✓ Subject 1/3 complete
...
============================================================
   PIPELINE COMPLETE
============================================================
Subjects processed: 3
Total epochs: 2,528
Time: 3.5 minutes
```

---

## **Step 2: Check the Results**

Results are saved in: `./results/experiment_YYYYMMDD_HHMMSS_quick/`

**Directory structure:**
```
results/
└── experiment_20251222_143015_quick/
    ├── per_subject/          # Individual subject data
    │   ├── subject_1/
    │   │   ├── epochs.npy    # Preprocessed EEG epochs
    │   │   ├── labels.npy    # Sleep stage labels
    │   │   └── features.csv  # 149 extracted features
    │   ├── subject_2/
    │   └── subject_3/
    │
    └── features/             # Combined dataset
        ├── all_features.csv  # ALL features (2528 epochs × 149)
        ├── all_labels.npy    # ALL labels (2528,)
        └── dataset_metadata.json
```

---

## **Step 3: Inspect the Features**

### **Open the features file:**

**Option A: Using Python**
```python
import pandas as pd

# Load features
features = pd.read_csv('results/experiment_20251222_143015_quick/features/all_features.csv')

print(f"Shape: {features.shape}")  # Should be (2528, 149)
print(f"Columns: {features.columns.tolist()[:5]}")  # First 5 feature names
print(features.head())
```

**Option B: Using Excel**
Open `all_features.csv` in Excel and inspect the 149 columns.

### **Expected feature names:**
```
F3_mean, F3_std, F3_var, F3_min, F3_max, ...
F4_mean, F4_std, ...
C3_mean, C3_std, ...
C4_mean, C4_std, ...
O1_mean, O1_std, ...
O2_mean, O2_std, ...
coherence_F3_F4, coherence_F3_C3, ...
plv_F3_O1, ...
global_entropy, global_complexity
```

---

## **Step 4: Pilot Run (10 subjects, ~15 minutes)**

Once satisfied with the quick test, run on 10 subjects:

```bash
python run_experiment.py --pilot
```

**Expected output:**
```
Total subjects to process: 10
...
PROCESSING SUBJECT 1/10: 1
  ✓ Subject 1/10 complete
...
PROCESSING SUBJECT 10/10: 128
  ✓ Subject 10/10 complete

Total epochs: ~8,400
Time: ~15 minutes
```

---

## **Step 5: Full Dataset (128 subjects, ~2-3 hours)**

When ready for the complete dataset:

```bash
python run_experiment.py --full
```

**Expected:**
- **Time:** 2-3 hours
- **Epochs:** ~106,680 total
- **Storage:** ~640 MB

**Pro tip:** Run this overnight or during a long break!

---

## **Common Issues & Solutions**

### **Issue 1: ModuleNotFoundError**
```
ModuleNotFoundError: No module named 'mne'
```

**Solution:**
```bash
pip install -r requirements.txt
```

---

### **Issue 2: Data Path Not Found**
```
FileNotFoundError: Base path not found: C:\Users\DerHo\Desktop\Data
```

**Solution:** Override the data path:
```bash
python run_experiment.py --quick-test --data-path "D:\MyData\BOAS"
```

---

### **Issue 3: Out of Memory**
```
MemoryError: Unable to allocate array
```

**Solution:** Don't save intermediate files:
```bash
python run_experiment.py --pilot --no-save-intermediate
```

---

### **Issue 4: Slow Processing**

**Check if filtering is the bottleneck:**
```bash
python run_experiment.py --quick-test --log-level DEBUG
```

Look for timing information in the debug logs.

---

## **Understanding the Output**

### **Per-Subject Summary:**
```
SUMMARY: Subject 1
  Epochs: 908
  Features: (908, 149)
  Labels: (array([0, 1, 2, 3, 4]), array([120, 45, 380, 280, 83]))
```

**Interpretation:**
- **908 epochs:** Subject 1 has 908 × 30-second epochs
- **Features (908, 149):** 908 epochs, each with 149 features
- **Labels:** Stage distribution:
  - Stage 0 (Wake): 120 epochs
  - Stage 1 (N1): 45 epochs
  - Stage 2 (N2): 380 epochs
  - Stage 3 (N3): 280 epochs
  - Stage 4 (REM): 83 epochs

### **Aggregated Dataset:**
```
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
```

**Interpretation:**
- Most epochs are **N2 (45.5%)** - this is normal for sleep
- **N1 is rare (5.3%)** - this is also expected (short transition stage)
- Distribution looks healthy for sleep data

---

## **Next Steps**

### **After Quick Test Works:**
1. ✅ Run pilot (10 subjects)
2. ✅ Inspect feature distributions
3. ✅ Validate preprocessing quality
4. ✅ Run full dataset (128 subjects)

### **After Full Dataset Completes:**
1. ⏳ Add fingerprinting module (next iteration)
2. ⏳ Add 4-stage caching system
3. ⏳ Add model training (XGBoost, RF, FNN)
4. ⏳ Add LOSO cross-validation (128 folds)
5. ⏳ Measure cache performance

---

## **Help & Debugging**

### **Enable Debug Output:**
```bash
python run_experiment.py --quick-test --log-level DEBUG
```

### **Save Logs to File:**
```bash
python run_experiment.py --pilot --log-file experiment.log
```

### **Check Command Options:**
```bash
python run_experiment.py --help
```

---

## **Pro Tips**

### **Tip 1: Start Small**
Always test with `--quick-test` first before running larger datasets.

### **Tip 2: Check Output Directory**
Results go to `./results/` by default. Override with:
```bash
python run_experiment.py --pilot --output-dir "D:\MyResults"
```

### **Tip 3: Named Experiments**
Give your experiments meaningful names:
```bash
python run_experiment.py --pilot --experiment-name "baseline_test_v1"
```

### **Tip 4: Monitor Progress**
The pipeline prints progress every few subjects. Watch for:
- ✓ symbols (success)
- ✗ symbols (failure)
- Epoch counts (should be 700-1000 per subject)

---

## **Expected Timings**

| Dataset | Subjects | Epochs | Time | Storage |
|---------|----------|--------|------|---------|
| **Quick Test** | 3 | ~2,500 | 2-3 min | ~15 MB |
| **Pilot** | 10 | ~8,400 | 10-15 min | ~50 MB |
| **Full** | 128 | ~106,680 | 2-3 hours | ~640 MB |

*Times measured on standard laptop (Intel i7, 16GB RAM, SSD)*

---

## **Success Checklist**

After running, verify:

- [ ] No error messages in output
- [ ] Output directory created
- [ ] `all_features.csv` has 149 columns
- [ ] `all_labels.npy` has same row count as features
- [ ] Label distribution looks reasonable (N2 should be ~40-50%)
- [ ] Pipeline statistics saved to `pipeline_stats.json`

---

## **Getting Help**

1. **Check logs:** `--log-level DEBUG`
2. **Read error messages carefully** - they're detailed
3. **Verify data path exists:** `dir C:\Users\DerHo\Desktop\Data`
4. **Check disk space:** Need ~1 GB free for full dataset

---

**You're ready to go! Start with `--quick-test` now! 🚀**

---

**Document Version:** 1.0  
**Date:** December 22, 2025
