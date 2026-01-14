# Channel Configuration Guide
## 6 Channels vs 8 Channels - When to Use Which?

---

## **Quick Comparison**

| Aspect | 6 Channels (EEG Only) | 8 Channels (EEG + Physiological) |
|--------|----------------------|----------------------------------|
| **Channels** | F3, F4, C3, C4, O1, O2 | + EOG, EMG |
| **Features** | 149 | 195 |
| **Best for** | Standard sleep staging | Enhanced REM/Wake detection |
| **Processing time** | Faster (~30s/subject) | Slightly slower (~35s/subject) |
| **Storage** | Smaller (~5 MB/subject) | Larger (~7 MB/subject) |
| **Scientific standard** | ✅ Standard | ✅ Comprehensive |
| **Complexity** | Simple | Moderate |

---

## **Option 1: 6 Channels (EEG Only)** ✅ **RECOMMENDED FOR START**

### **Configuration:**
```yaml
data:
  channel_preset: "eeg_only"
```

### **Channels:**
- **PSG_F3, PSG_F4** - Frontal EEG (attention, executive function)
- **PSG_C3, PSG_C4** - Central EEG (motor, sensory)
- **PSG_O1, PSG_O2** - Occipital EEG (visual, alpha waves)

### **Features: 149 total**
- Time-domain: 60 (10 per channel × 6)
- Frequency: 54 (9 per channel × 6)
- Complexity: 24 (4 per channel × 6)
- Global: 11 (coherence, PLV, entropy, complexity)

### **Use when:**
- ✅ Starting your experiments
- ✅ Following standard sleep research practices
- ✅ Your documentation is based on 149 features
- ✅ You want faster processing
- ✅ Storage space is limited

### **Expected Performance:**
- Good accuracy for all sleep stages
- N1 may be challenging (expected - it's a transition stage)
- N2, N3, REM, Wake should be well-classified

---

## **Option 2: 8 Channels (EEG + Physiological)** 🚀 **FOR COMPARISON**

### **Configuration:**
```yaml
data:
  channel_preset: "eeg_plus_physiological"
```

### **Channels:**
- **PSG_F3, PSG_F4, PSG_C3, PSG_C4, PSG_O1, PSG_O2** - EEG (same as 6-channel)
- **PSG_EOG** - Electrooculography (eye movements)
  - Critical for REM detection (rapid eye movements)
  - Helps distinguish Wake from drowsiness
- **PSG_EMG** - Electromyography (muscle tone)
  - Helps distinguish Wake from Sleep (muscle tone drops during sleep)
  - Critical for REM detection (muscle atonia during REM)

### **Features: 195 total**
- Time-domain: 80 (10 per channel × 8)
- Frequency: 72 (9 per channel × 8)
- Complexity: 32 (4 per channel × 8)
- Global: 11 (same as 6-channel)

### **Use when:**
- ✅ You want to compare against 6-channel results
- ✅ REM detection is particularly important
- ✅ You want to maximize classification performance
- ✅ You have extra time and storage
- ✅ Your thesis can discuss the impact of physiological signals

### **Expected Performance:**
- Better REM detection (~5-10% improvement expected)
- Better Wake vs Sleep discrimination
- Slightly better overall accuracy
- N1 may still be challenging (inherently difficult stage)

---

## **Option 3: Custom Channels** 🔧 **FOR SPECIFIC EXPERIMENTS**

### **Configuration:**
```yaml
data:
  channel_preset: "custom"
  channels:
    - PSG_F3
    - PSG_C3
    - PSG_O1
    # Add any channels you want
```

### **Use when:**
- You want to test specific channel combinations
- You're exploring minimal channel sets
- You want to study specific brain regions

---

## **Recommended Workflow**

### **Phase 1: Start with 6 Channels**
```bash
# Use default configuration (6 channels)
python run_experiment.py --quick-test

# Run pilot
python run_experiment.py --pilot

# Run full dataset
python run_experiment.py --full --config example_config.yaml
```

**Why?**
- Establish baseline performance
- Complete documentation already written
- Faster experimentation

---

### **Phase 2: Compare with 8 Channels**
```bash
# Run with 8 channels
python run_experiment.py --full --config config_8channels.yaml
```

**Compare:**
- Accuracy: 6-channel vs 8-channel
- REM detection: Improvement with EOG
- Wake detection: Improvement with EMG
- Processing time difference
- Storage difference

**For thesis:**
- Table comparing 6 vs 8 channel results
- Discussion of physiological signal contribution
- Analysis of which stages benefited most

---

### **Phase 3: Feature Analysis** (Optional)
```bash
# Create configs with different combinations
python run_experiment.py --config config_eeg_plus_eog.yaml  # 7 channels
python run_experiment.py --config config_eeg_plus_emg.yaml  # 7 channels
```

**Analyze:**
- EOG contribution (for REM)
- EMG contribution (for Wake/Sleep)
- Which signal is more valuable?

---

## **Storage & Time Estimates**

### **6 Channels:**
```
Quick test (3 subjects):  ~15 MB,  2-3 minutes
Pilot (10 subjects):      ~50 MB,  10-15 minutes
Full (128 subjects):      ~640 MB, 2-3 hours
```

### **8 Channels:**
```
Quick test (3 subjects):  ~20 MB,  3-4 minutes
Pilot (10 subjects):      ~70 MB,  12-18 minutes
Full (128 subjects):      ~900 MB, 2.5-3.5 hours
```

**Difference:** ~30% more storage, ~15% more time

---

## **Scientific Justification**

### **6 Channels (EEG Only):**

**Pros:**
- ✅ Standard in sleep research (AASM guidelines focus on EEG)
- ✅ EEG contains most relevant information for sleep staging
- ✅ Simpler to interpret and explain
- ✅ Sufficient for good classification performance

**Cons:**
- ⚠️ May miss subtle REM indicators (eye movements)
- ⚠️ May confuse drowsy Wake with light sleep

### **8 Channels (EEG + Physiological):**

**Pros:**
- ✅ Follows full polysomnography standard
- ✅ Better REM detection (EOG captures eye movements)
- ✅ Better Wake/Sleep discrimination (EMG captures muscle tone)
- ✅ More comprehensive feature set
- ✅ Shows you considered all available signals

**Cons:**
- ⚠️ Slightly more complex to explain
- ⚠️ Requires updating documentation (149 → 195 features)
- ⚠️ Longer processing time

---

## **For Your Thesis**

### **Option A: Start with 6, mention 8 as future work**

**Pros:**
- Documentation already complete
- Simpler to explain
- Focus on caching (main contribution)

**Cons:**
- Miss opportunity to show comprehensive analysis

### **Option B: Do both, compare in results**

**Pros:**
- ✅ Shows thoroughness
- ✅ Demonstrates impact of physiological signals
- ✅ Interesting comparison table for thesis
- ✅ Shows caching works for different feature counts

**Cons:**
- ~1 extra day of experimentation
- Slightly more to write up

**Recommendation:** **Option B** - Do both! It's minimal extra work (just one config change) and makes your thesis more comprehensive.

---

## **Example Thesis Section**

```markdown
### 4.2 Channel Selection

We evaluated two channel configurations:

**Configuration 1: EEG-Only (6 channels)**
- Channels: F3, F4, C3, C4, O1, O2
- Features: 149 (6 × 23 + 11 global)
- Rationale: Standard approach in sleep research

**Configuration 2: Full Polysomnography (8 channels)**
- Channels: EEG (6) + EOG + EMG
- Features: 195 (8 × 23 + 11 global)
- Rationale: Comprehensive physiological monitoring

Results showed that adding EOG and EMG improved:
- REM detection: +8.3% F1-score
- Wake detection: +5.1% F1-score
- Overall accuracy: +3.2%

However, processing time increased by 15% and storage by 30%.
The caching system maintained >95% hit rate for both configurations.
```

---

## **Decision Matrix**

| If you want to... | Use |
|------------------|-----|
| **Get started quickly** | 6 channels |
| **Follow existing docs** | 6 channels |
| **Maximize performance** | 8 channels |
| **Compare approaches** | Both |
| **Impress thesis committee** | Both |
| **Minimize time/storage** | 6 channels |
| **Show comprehensive analysis** | Both |

---

## **My Recommendation:** 🎯

**Start with 6 channels NOW, then add 8 channels later**

### **Why this approach:**
1. ✅ Get pipeline working with 6 channels first
2. ✅ Validate everything works
3. ✅ Then simply change config to `"eeg_plus_physiological"`
4. ✅ Run experiments again
5. ✅ Compare results
6. ✅ Write up comparison in thesis

**Effort:** 5 minutes to change config, 3 hours to re-run
**Benefit:** Much stronger thesis with comparison

---

**Document Version:** 1.0  
**Date:** December 22, 2025
