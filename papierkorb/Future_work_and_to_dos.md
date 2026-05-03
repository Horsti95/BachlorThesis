# Future Work & Development Roadmap


## Intelligent ML Caching Framework — Bachelor Thesis

**Document Version**: 1.3  
**Created**: December 23, 2025  
**Last Updated**: December 27, 2025  
**Author**: Lennart  
**Thesis Deadline**: January 15, 2026  

---

## 📋 Document Purpose

This document outlines potential extensions, improvements, and future development directions for the Intelligent ML Caching Framework. Items are categorized by priority, effort, and strategic value for both thesis completion and long-term framework development.


**Version History**:
| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 23, 2025 | Initial creation |
| 1.1 | Dec 23, 2025 | Added Framework Focus section, Scope Comparison |
| 1.2 | Dec 23, 2025 | Added Dual-Mode Pipeline Architecture (deferred) |
| 1.3 | Dec 27, 2025 | Updated implementation status to match current state |

---

## 🔑 Framework Focus: Fingerprint-Based Caching

### Core Research Contribution

This implementation focuses on **intelligent caching via cryptographic fingerprinting** to achieve two primary goals:

1. **Computational Time Reduction**
   - Avoid redundant feature extraction across experiments
   - Enable rapid iteration through hyperparameter spaces
   - Reduce full pipeline runtime from ~53 minutes to ~30 seconds (100× speedup)

2. **Scientific Reproducibility**
   - Deterministic fingerprints ensure identical inputs produce identical outputs
   - SHA-256 hashing of configuration parameters guarantees cache validity
   - Version tracking enables result verification across sessions

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      FINGERPRINT-BASED CACHING                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Config Parameters           SHA-256 Hash              Cache Decision   │
│  ┌─────────────────┐        ┌──────────┐              ┌─────────────┐  │
│  │ bandpass: 0.5-40│   ──►  │ a3f2b1c8 │  ──► Match?  │ ✅ HIT: Load│  │
│  │ notch: 50 Hz    │        │ 9d4e7f2a │      ▼       │ ❌ MISS:    │  │
│  │ epoch: 30s      │        │ ...      │    Yes/No    │    Compute  │  │
│  │ channels: 6     │        └──────────┘              └─────────────┘  │
│  └─────────────────┘                                                    │
│                                                                         │
│  Cache Invalidation Triggers:                                           │
│  • Preprocessing parameters change  →  Fingerprint changes  →  MISS    │
│  • Same parameters                  →  Same fingerprint     →  HIT     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Metrics for Thesis

| Metric | Purpose | Target |
|--------|---------|--------|
| **Cache Hit Rate** | Measure caching effectiveness | > 90% |
| **Time Saved** | Quantify efficiency gains | > 50 minutes |
| **Speedup Factor** | Cold vs warm runtime ratio | > 100× |
| **Fingerprint Determinism** | Reproducibility validation | 100% |

### Benchmark Results (50 Subjects, Validated)

| Subjects | Model | Cold -> SSD Speedup | SSD -> RAM Speedup |
|---:|---|---:|---:|
| 50 | XGBoost | **14.6x** | 1.8x |
| 50 | Random Forest | **5.4x** | 3.7x |

**Implications:**

- **Cold -> SSD is the dominant speedup.** Fingerprint-based disk caching eliminates redundant model training entirely. XGBoost benefits more (14.6x) because its training is compute-heavy relative to serialization cost, so cache hits save proportionally more time.
- **SSD -> RAM adds marginal benefit.** At 50 subjects the RAM advantage is 1.8-3.7x over SSD, but at 128 subjects it degraded due to memory pressure (GC overhead exceeding disk I/O savings). Modern NVMe SSDs already deliver near-RAM read speeds for the small model files (~1 MB each), so preloading to RAM provides diminishing returns.
- **Thesis focus: Cold vs Warm SSD.** This is where fingerprint-based invalidation provides the clearest, most reproducible benefit. RAM caching is noted as a potential optimization but not recommended as a default strategy for this workload scale.

### Why This Matters

> *"The primary contribution of this thesis is not achieving state-of-the-art sleep classification accuracy, but demonstrating that intelligent caching with fingerprint-based invalidation can dramatically reduce ML experimentation costs while ensuring reproducibility. The sleep stage classification task serves as a validation case study."*

---

## 🎯 Scope Comparison: Planned vs Current vs Minimum

### Original Thesis Outline (Planned)

From the thesis table of contents and pipeline architecture documents:

```
PLANNED EXPERIMENTAL GRID (18 combinations):
────────────────────────────────────────────
Feature Counts:     [30, 50, 105]      → 3 options
Corr Thresholds:    [0.75, 0.90, None] → 3 options
Models:             [XGBoost, RF]      → 2 options
────────────────────────────────────────────
Total: 18 combinations × 128 LOSO folds = 2,304 training runs

Additional Planned Options:
• Feature Selection Methods: mutual_info, f_classif
• Class Balancing: None, class_weight
• Models from Capstone: GBM, FNN (78% accuracy)
```

### Current Implementation Status

| Component | Planned | Current Status | Gap |
|-----------|---------|----------------|-----|
| **Data Loading** | BOAS 128 subjects | ✅ Complete | None |
| **Preprocessing** | Bandpass, Notch, Resample | ✅ Complete | None |
| **Feature Extraction** | 105 features | ✅ 149 features | Exceeded ✓ |
| **Feature Caching** | Fingerprint + NPZ | ✅ Complete (224× speedup) | None |
| **Feature Selection** | Correlation + Top-K | ✅ Complete (ANOVA + GLOBAL) | None |
| **Models** | XGBoost, RF, FNN | ✅ XGB, RF complete | FNN deferred |
| **LOSO CV Splitter** | 128 folds | ✅ Complete | None |
| **LOSO Model Cache** | Fingerprint-based | ❌ Not implemented | **~5-6 hours (estimated)** |
| **Evaluation** | Accuracy, F1, Kappa | ✅ Complete | None |
| **Visualization** | Confusion matrix, plots | ✅ Complete | None |

### Implementation Status Summary

| Module | Status | Details |
|--------|--------|---------|
| Data Loading | ✅ Complete | BOAS dataset, 128 subjects |
| Preprocessing | ✅ Complete | Bandpass, Notch, Downsampling |
| Feature Extraction | ✅ Complete | 149 features (6 channels), cached |
| Feature Caching | ✅ Complete | SHA-256 fingerprinting, 224× speedup |
| Feature Selection | ✅ Complete | ANOVA + GLOBAL scope |
| Cache Leaderboard | ✅ Complete | Performance tracking, JSON persistence |
| CLI Interface | ✅ Complete | Interactive menu + command-line flags |
| Models | ✅ Complete | XGBoost, RF ready; FNN deferred |
| LOSO CV Splitter | ✅ Complete | 128-fold cross-validation |
| **LOSO Model Cache** | ❌ **TODO** | **Critical gap for thesis! (~5-6 hours)** |
| Evaluation | ✅ Complete | Metrics aggregation |
| Visualization | ✅ Complete | Publication-quality figures |

### Experimental Grids Comparison

#### Originally Planned (18 combinations)
```
Feature Counts:     [30, 50, 105]      → 3 options
Corr Thresholds:    [0.75, 0.90, None] → 3 options
Models:             [XGBoost, RF]      → 2 options
Channels:           [6]                → 1 option
CV Strategy:        [LOSO]             → 1 option
────────────────────────────────────────────────────
Total: 18 combinations × 128 LOSO folds = 2,304 training runs
```

#### Extended Option (27 combinations) — If Including FNN
```
Feature Counts:     [30, 50, 149]      → 3 options
Corr Thresholds:    [0.90, 0.95, None] → 3 options
Models:             [XGBoost, RF, FNN] → 3 options
────────────────────────────────────────────────────
Total: 27 combinations × 128 LOSO folds = 3,456 training runs
```

#### Minimum Viable (8 combinations) — Thesis Safety Net
```
Feature Counts:     [50, 149]          → 2 options
Corr Thresholds:    [0.95, None]       → 2 options
Models:             [XGBoost, RF]      → 2 options
────────────────────────────────────────────────────
Total: 8 combinations × 128 LOSO folds = 1,024 training runs
```

### Decision Matrix

| Grid | Combinations | LOSO Runs | Est. Time | Risk Level |
|------|--------------|-----------|-----------|------------|
| **Minimum (8)** | 2×2×2 | 1,024 | ~4h | 🟢 Safe |
| **Planned (18)** | 3×3×2 | 2,304 | ~8h | 🟢 Safe |
| **Extended (27)** | 3×3×3 | 3,456 | ~12h | 🟡 Moderate |

**Recommendation**: Run the **18-combination grid** as originally planned. Fall back to 8 if time constrained.

---

## 🚀 Future Work Categories

### Legend

| Symbol | Meaning |
|--------|---------|
| 🎓 | Thesis-relevant (could strengthen submission) |
| 🔬 | Research extension (post-thesis publication) |
| 🛠️ | Engineering improvement (code quality) |
| 📊 | Visualization/reporting enhancement |

---

## 1️⃣ Validation Strategies

### Currently Implemented
- ✅ LOSO (Leave-One-Subject-Out) — 128 folds

### Future Extensions

| Extension | Priority | Effort | Value | Type |
|-----------|----------|--------|-------|------|
| **Subject-Grouped K-Fold** | Medium | 2h | Faster iteration during development | 🛠️ |
| **Stratified K-Fold** | Low | 1h | Alternative validation baseline | 🔬 |
| **70/30 Hold-Out Split** | Low | 30min | Quick sanity checks | 🛠️ |
| **Nested Cross-Validation** | Low | 4h | Hyperparameter tuning without leakage | 🔬 |

#### Implementation Notes

**Subject-Grouped K-Fold** (Recommended if time permits):
```python
# Concept: Group subjects into K folds
# Fold 1: Subjects 1-26 (Test) vs 27-128 (Train)
# Fold 2: Subjects 27-52 (Test) vs Rest (Train)
# etc.

class SubjectGroupedKFold:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
    
    def split(self, X, y, subject_ids):
        unique_subjects = np.unique(subject_ids)
        fold_size = len(unique_subjects) // self.n_splits
        # ... implementation
```

**Caching Benefit**: All strategies benefit from cached features — demonstrates framework flexibility.

---

## 2️⃣ Additional Models

### Currently Implemented
- ✅ XGBoost (Gradient Boosting)
- ✅ Random Forest (Bagging Ensemble)
- ⚠️ FNN (Feed-Forward Neural Network) — placeholder

### Future Extensions

| Model | Priority | Effort | Expected Benefit | Type |
|-------|----------|--------|------------------|------|
| **Logistic Regression** | High | 30min | Simplest baseline, interpretable | 🎓 |
| **SVM (RBF Kernel)** | High | 1h | Classic ML baseline, literature comparison | 🎓 |
| **LightGBM** | Medium | 1h | Faster training than XGBoost | 🛠️ |
| **CatBoost** | Low | 1h | Alternative boosting method | 🔬 |
| **Naive Bayes** | Low | 30min | Probabilistic baseline | 🔬 |
| **k-NN** | Low | 30min | Instance-based baseline | 🔬 |

#### Implementation Template

```python
# Add to models.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class LogisticRegressionModel(BaseModel):
    """Logistic Regression baseline classifier."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1
        )
    
    @property
    def name(self) -> str:
        return "LogisticRegression"


class SVMModel(BaseModel):
    """Support Vector Machine with RBF kernel."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self.model = SVC(
            kernel='rbf',
            random_state=42,
            class_weight='balanced',
            probability=True  # For predict_proba
        )
    
    @property
    def name(self) -> str:
        return "SVM_RBF"
```

#### Strategic Value

| Model Combination | Thesis Narrative |
|-------------------|------------------|
| LR + SVM + RF + XGB | "Framework supports diverse classifier families" |
| LR as baseline | "Features are discriminative even with simple models" |
| XGB vs LightGBM | "Framework enables rapid model comparison" |

---

## 3️⃣ Deep Learning Extensions

### Current State
- ✅ FNN on extracted features (implemented)
- ❌ CNN on raw signals (not implemented)
- ❌ LSTM on sequences (not implemented)

### Future Extensions

| Architecture | Priority | Effort | Data Requirement | Type |
|--------------|----------|--------|------------------|------|
| **FNN Completion** | High | 2h | Existing 149 features | 🎓 |
| **1D-CNN on Raw Epochs** | Low | 8h | Raw epochs (3840 × 6) | 🔬 |
| **LSTM on Epoch Sequences** | Low | 10h | Sequential epochs | 🔬 |
| **Transformer** | Very Low | 15h+ | Sequential + positional | 🔬 |
| **Hybrid CNN-LSTM** | Very Low | 12h | Raw sequential data | 🔬 |

#### Important Distinction

```
CURRENT (Feature-Based):
┌─────────────────────────────────────────────────────┐
│  Raw Signal → Feature Extraction → [149] → Model   │
│              (cached)              (table)          │
└─────────────────────────────────────────────────────┘

FUTURE (End-to-End Deep Learning):
┌─────────────────────────────────────────────────────┐
│  Raw Signal → [3840 × 6] → CNN/LSTM → Prediction   │
│              (cached epochs)  (learned features)    │
└─────────────────────────────────────────────────────┘
```

#### New Pipeline Required for Deep Learning

```python
# epoch_cache.py (NEW - for deep learning)

def cache_raw_epochs(subject_id: int, epochs: np.ndarray, labels: np.ndarray):
    """
    Cache raw epochs for deep learning models.
    
    Shape: (n_epochs, n_samples, n_channels) = (~950, 3840, 6)
    Size: ~88 MB per subject (vs ~2 MB for features)
    """
    cache_path = CACHE_DIR / f"subject_{subject_id}_epochs.npz"
    np.savez_compressed(
        cache_path,
        epochs=epochs,      # (950, 3840, 6) float32
        labels=labels,      # (950,) int
        sampling_rate=128,
        epoch_duration=30
    )
```

#### Thesis Framing (if not implemented)

> *"The current framework focuses on feature-based classification to demonstrate caching benefits with traditional ML algorithms. The modular architecture supports extension to end-to-end deep learning approaches, where raw epoch tensors could be cached instead of extracted features. This would require approximately 44× more storage (88 MB vs 2 MB per subject) but would enable automatic feature learning through convolutional or recurrent architectures."*

---

## 4️⃣ Channel Configuration Extensions

### Current State
- ✅ 6 Channels (EEG only): 149 features — **Primary**
- ✅ 8 Channels (EEG + EOG + EMG): 195 features — **Cached but unused**

### Future Extensions

| Extension | Priority | Effort | Value | Type |
|-----------|----------|--------|-------|------|
| **8-Channel Experiments** | Medium | 1h | Compare EEG-only vs multimodal | 🎓 |
| **Channel Ablation Study** | Low | 3h | Which channels matter most? | 🔬 |
| **Single-Channel Models** | Low | 2h | Minimal sensor requirements | 🔬 |
| **Channel Selection Search** | Low | 4h | Optimal channel subset | 🔬 |

#### 8-Channel Experiment Design

```python
# Experiment configuration
CHANNEL_EXPERIMENTS = {
    'eeg_only': {
        'channels': ['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2'],
        'n_features': 149,
        'hypothesis': 'Baseline EEG performance'
    },
    'eeg_eog_emg': {
        'channels': ['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2', 'PSG_EOG', 'PSG_EMG'],
        'n_features': 195,
        'hypothesis': 'EOG improves REM detection, EMG improves Wake detection'
    }
}
```

#### Expected Results

| Stage | EEG Only | + EOG | + EMG | Rationale |
|-------|----------|-------|-------|-----------|
| Wake | Good | Same | **Better** | EMG detects muscle activity |
| N1 | Poor | Same | Same | N1 is inherently difficult |
| N2 | Good | Same | Same | K-complexes in EEG |
| N3 | Good | Same | Same | Delta waves in EEG |
| REM | Good | **Better** | Same | EOG detects eye movements |

#### Caching Benefit

```
Cache contains 195 features (8 channels)
├── Load 195 features
├── Runtime filter to 149 (6 channels)
└── NO re-extraction needed!

This demonstrates: "Compute once, select at runtime"
```

---

## 5️⃣ Feature Engineering Extensions

### Current State
- ✅ 149 features (6 channels) / 195 features (8 channels)
- ✅ Time domain (60), Frequency domain (54), Complexity (24), Global (11)

### Future Extensions

| Extension | Priority | Effort | Features Added | Type |
|-----------|----------|--------|----------------|------|
| **Band Power Ratios** | Medium | 1h | +6 (theta/alpha, delta/beta, etc.) | 🎓 |
| **Sample Entropy** | Low | 2h | +6 (per channel) | 🔬 |
| **Permutation Entropy** | Low | 2h | +6 (per channel) | 🔬 |
| **Spectral Edge Frequency** | Low | 1h | +6 (per channel) | 🔬 |
| **Spindle Detection** | Very Low | 8h | +6 (spindle density per channel) | 🔬 |
| **Slow Oscillation Power** | Low | 1h | +6 (<1 Hz per channel) | 🔬 |

#### Implementation Note

```
⚠️ WARNING: Adding new features INVALIDATES existing cache!

New features → New fingerprint → Cache miss → Full re-extraction

For thesis: Keep current 149 features
For future: Extend and re-cache
```

#### Band Power Ratios (Quick Win)

```python
# Can compute POST-HOC from existing features without re-caching!

def compute_band_ratios(features_df: pd.DataFrame) -> pd.DataFrame:
    """Compute band power ratios from existing band powers."""
    ratios = pd.DataFrame()
    
    for ch in ['F3', 'F4', 'C3', 'C4', 'O1', 'O2']:
        ratios[f'{ch}_theta_alpha_ratio'] = (
            features_df[f'{ch}_theta_power'] / 
            features_df[f'{ch}_alpha_power']
        )
        ratios[f'{ch}_delta_beta_ratio'] = (
            features_df[f'{ch}_delta_power'] / 
            features_df[f'{ch}_beta_power']
        )
    
    return pd.concat([features_df, ratios], axis=1)
```

---

## 6️⃣ Caching System Enhancements

### Current State
- ✅ SHA-256 fingerprinting
- ✅ NPZ file storage
- ✅ Version tracking
- ✅ Leaderboard metrics

### Future Extensions

| Extension | Priority | Effort | Value | Type |
|-----------|----------|--------|-------|------|
| **Cache Compression Levels** | Low | 2h | Reduce storage 30-50% | 🛠️ |
| **Partial Cache Loading** | Medium | 3h | Load only needed features | 🛠️ |
| **Cache Warming Script** | Low | 1h | Pre-populate cache | 🛠️ |
| **Distributed Caching** | Very Low | 8h | Multi-machine support | 🔬 |
| **Cache Analytics Dashboard** | Low | 4h | Visual cache performance | 📊 |
| **Automatic Cache Cleanup** | Low | 2h | Remove stale entries | 🛠️ |
| **ML-Based Cache Prediction** | Very Low | 15h+ | Predict next cache access | 🔬 |

#### Partial Cache Loading (High Value)

```python
# Current: Load all 149 features, then filter
features = load_from_cache(subject_id)  # All 149
selected = features[top_50_columns]      # Filter to 50

# Future: Load only needed features
features = load_from_cache(
    subject_id, 
    columns=['F3_mean', 'F3_std', ...]  # Only 50
)
```

#### Cache Analytics Dashboard Concept

```
┌─────────────────────────────────────────────────────────────┐
│                 CACHE PERFORMANCE DASHBOARD                  │
├─────────────────────────────────────────────────────────────┤
│  Total Runs: 47          Cache Size: 256 MB                 │
│  Hit Rate: 94.2%         Time Saved: 3h 42min               │
├─────────────────────────────────────────────────────────────┤
│  Hit Rate Over Time                                         │
│  100%|████████████████████████████████████▓░░| 94.2%        │
│                                                             │
│  Cumulative Time Saved                                      │
│  [========================================] 3:42:15         │
│                                                             │
│  Cache Size by Subject                                      │
│  S001: ██ 2.1MB    S065: ██ 2.3MB    S128: ██ 1.9MB        │
└─────────────────────────────────────────────────────────────┘
```

---

## 7️⃣ Visualization & Reporting

### Current State
- ❌ No automated visualizations
- ❌ No thesis-ready figures

### Future Extensions

| Visualization | Priority | Effort | Thesis Section | Type |
|---------------|----------|--------|----------------|------|
| **Confusion Matrix (Aggregated)** | High | 1h | Results | 🎓📊 |
| **Confusion Matrix (Per-Fold)** | Low | 2h | Appendix | 📊 |
| **Feature Importance Plot** | High | 1h | Results | 🎓📊 |
| **ROC Curves (Per-Class)** | Medium | 2h | Results | 🎓📊 |
| **Learning Curves** | Low | 2h | Discussion | 📊 |
| **Cache Performance Timeline** | High | 2h | Results | 🎓📊 |
| **Time Savings Bar Chart** | High | 1h | Results | 🎓📊 |
| **Hyperparameter Sensitivity** | Low | 3h | Appendix | 📊 |
| **t-SNE Feature Visualization** | Low | 2h | Background | 📊 |

#### Thesis-Critical Figures

```python
# visualization.py (Priority implementations)

def plot_cache_performance_comparison():
    """
    Bar chart: Cold Start vs Cached Runtime
    
    Essential for thesis Chapter 5 (Results)
    Shows: "Caching reduces pipeline time by X%"
    """
    pass

def plot_aggregated_confusion_matrix():
    """
    5×5 confusion matrix aggregated across all LOSO folds
    
    Essential for thesis Chapter 5 (Results)
    Shows: "N1 is difficult, N2/N3 are reliable"
    """
    pass

def plot_feature_importance_top_k():
    """
    Horizontal bar chart of top-K most important features
    
    Essential for thesis Chapter 5 (Results)
    Shows: "Delta power and complexity features dominate"
    """
    pass

def plot_time_saved_cumulative():
    """
    Line chart: Cumulative time saved over experiments
    
    Essential for thesis Chapter 5 (Results)
    Shows: "Caching benefit grows with repeated experiments"
    """
    pass
```

---

## 8️⃣ Code Quality & Engineering

### Current State
- ✅ Modular architecture
- ✅ Dataclass configurations
- ⚠️ Limited test coverage
- ⚠️ No CI/CD pipeline

### Future Extensions

| Extension | Priority | Effort | Value | Type |
|-----------|----------|--------|-------|------|
| **Unit Tests** | Medium | 4h | Reliability, refactoring safety | 🛠️ |
| **Integration Tests** | Medium | 3h | End-to-end validation | 🛠️ |
| **Type Hints Completion** | Low | 2h | IDE support, documentation | 🛠️ |
| **Docstring Completion** | Medium | 3h | API documentation | 🛠️ |
| **GitHub Actions CI** | Low | 2h | Automated testing | 🛠️ |
| **Pre-commit Hooks** | Low | 1h | Code quality enforcement | 🛠️ |
| **Sphinx Documentation** | Low | 4h | Professional docs | 🛠️ |
| **Docker Container** | Low | 3h | Reproducible environment | 🛠️ |

#### Minimal Test Suite

```python
# tests/test_feature_extraction.py

def test_feature_count_6ch():
    """Verify 149 features for 6 channels."""
    extractor = FeatureExtractor(n_channels=6)
    features = extractor.extract_single_epoch(dummy_epoch)
    assert len(features) == 149

def test_cache_fingerprint_determinism():
    """Same config must produce same fingerprint."""
    config1 = PreprocessingConfig(bandpass_low=0.5, bandpass_high=40)
    config2 = PreprocessingConfig(bandpass_low=0.5, bandpass_high=40)
    assert compute_fingerprint(config1) == compute_fingerprint(config2)

def test_cache_invalidation_on_config_change():
    """Different config must produce different fingerprint."""
    config1 = PreprocessingConfig(bandpass_low=0.5, bandpass_high=40)
    config2 = PreprocessingConfig(bandpass_low=0.5, bandpass_high=35)
    assert compute_fingerprint(config1) != compute_fingerprint(config2)
```

---

## 9️⃣ Publication & Dissemination

### Post-Thesis Opportunities

| Opportunity | Timeline | Effort | Venue |
|-------------|----------|--------|-------|
| **Conference Paper** | +3 months | 40h | IEEE EMBC, ICASSP |
| **Journal Article** | +6 months | 80h | IEEE JBHI, Frontiers |
| **Open Source Release** | +1 month | 20h | GitHub, PyPI |
| **Blog Post / Tutorial** | +2 weeks | 8h | Medium, Personal Blog |

#### Paper Abstract Draft

> *"We present an intelligent caching framework for machine learning experiment optimization, demonstrated through EEG-based sleep stage classification. Our approach combines SHA-256 fingerprint-based cache invalidation with a hierarchical storage strategy that reduces repeated experimental runs from hours to seconds. Evaluated on the BOAS dataset (128 subjects, ~120,000 epochs), we achieve 94% cache hit rates and 100× speedup on subsequent experiments while maintaining classification accuracy of XX% (Cohen's κ = 0.XX). The framework's modular architecture supports diverse ML algorithms and validation strategies, enabling rapid hyperparameter exploration without redundant computation."*

---

## 📅 Implementation Timeline

### Phase 1: Thesis Completion (Now → Jan 15, 2026)

| Week | Focus | Deliverables |
|------|-------|--------------|
| **Dec 23-29** | Training pipeline | LOSO CV working, first results |
| **Dec 30-Jan 5** | Full experiments | All 8 configurations complete |
| **Jan 6-10** | Visualization | Thesis figures generated |
| **Jan 11-14** | Writing | Results chapter complete |
| **Jan 15** | **SUBMISSION** | 🎓 |

### Phase 2: Post-Thesis Polish (Jan-Feb 2026)

| Task | Effort | Priority |
|------|--------|----------|
| Additional models (LR, SVM) | 2h | Medium |
| 8-channel experiments | 3h | Medium |
| Extended visualizations | 4h | Low |
| Code cleanup & documentation | 6h | Medium |

### Phase 3: Publication (Mar-Jun 2026)

| Task | Effort | Priority |
|------|--------|----------|
| Conference paper draft | 40h | Optional |
| Open source preparation | 20h | Optional |
| Extended experiments | 20h | Optional |

---

## 🔮 Post-Thesis: Dual-Mode Pipeline Architecture

### Concept Overview

After thesis submission, the pipeline can be extended to support two distinct operational modes:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DUAL-MODE PIPELINE ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐          ┌─────────────────────────────────┐  │
│  │    THESIS MODE      │          │      EXPLORATION MODE           │  │
│  │    (Academic)       │          │      (Personal Project)         │  │
│  ├─────────────────────┤          ├─────────────────────────────────┤  │
│  │ • 18 configurations │          │ • Unlimited configurations      │  │
│  │ • XGBoost + RF      │          │ • All models + hyperparameter   │  │
│  │ • LOSO only         │          │   grid search                   │  │
│  │ • 6 EEG channels    │          │ • Multiple CV strategies        │  │
│  │ • Thesis figures    │          │ • 6 or 8 channels               │  │
│  │ • LaTeX outputs     │          │ • Advanced analytics            │  │
│  │                     │          │ • Model persistence             │  │
│  │ Focus: Caching      │          │ • Web dashboard integration     │  │
│  │ demonstration       │          │                                 │  │
│  │                     │          │ Focus: Best model discovery     │  │
│  └──────────┬──────────┘          └───────────────┬─────────────────┘  │
│             │                                      │                    │
│             └──────────────┬───────────────────────┘                    │
│                            ▼                                            │
│             ┌─────────────────────────────────────┐                     │
│             │        SHARED FOUNDATION            │                     │
│             ├─────────────────────────────────────┤                     │
│             │ • Caching system (unchanged)        │                     │
│             │ • Feature extraction (unchanged)    │                     │
│             │ • Config dataclasses (extensible)   │                     │
│             │ • Fingerprint validation            │                     │
│             │ • Leaderboard tracking              │                     │
│             └─────────────────────────────────────┘                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Not Implemented Now

| Reason | Explanation |
|--------|-------------|
| **Deadline Risk** | 23 days until thesis submission — no time for architectural expansion |
| **YAGNI Principle** | "You Aren't Gonna Need It" — speculative features before real requirements |
| **Premature Abstraction** | Building flexibility before understanding actual needs leads to wrong abstractions |
| **Focus Dilution** | Thesis contribution is caching, not pipeline flexibility |

### Current Architecture Already Supports This

The existing modular design requires minimal changes for dual-mode support:

```python
# Future: config.py extension (NOT IMPLEMENTED)

@dataclass
class ExperimentConfig:
    mode: str = "thesis"  # "thesis" or "exploration"
    
    def get_grid(self) -> dict:
        if self.mode == "thesis":
            return {
                'feature_counts': [30, 50, 105],
                'corr_thresholds': [0.75, 0.90, None],
                'models': ['xgboost', 'random_forest'],
                'cv_strategy': 'loso',
                'channels': 6
            }
        else:  # exploration mode
            return {
                'feature_counts': [30, 50, 75, 105, 149],
                'corr_thresholds': [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, None],
                'models': ['logistic_regression', 'svm', 'random_forest', 
                          'xgboost', 'lightgbm', 'catboost', 'fnn'],
                'cv_strategy': ['loso', 'grouped_kfold', 'stratified_kfold'],
                'channels': [6, 8],
                'hyperparameter_search': True
            }
```

### Exploration Mode Feature Roadmap

#### Step 1: Extended Model Grid
| Model | Purpose | Effort |
|-------|---------|--------|
| Logistic Regression | Baseline interpretability | 30min |
| SVM (RBF) | Classic ML comparison | 1h |
| LightGBM | Faster boosting | 1h |
| CatBoost | Alternative boosting | 1h |
| Complete FNN | Neural network baseline | 2h |

#### Step 2: Hyperparameter Search
```python
# Future: hyperparameter_search.py (NOT IMPLEMENTED)

XGBOOST_SEARCH_SPACE = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# Use: RandomizedSearchCV or Optuna for efficient search
```

#### Step 3: Advanced Visualization & Analytics
| Feature | Description | Effort |
|---------|-------------|--------|
| Interactive dashboard | Streamlit/Gradio web UI | 8h |
| Model comparison heatmaps | Performance across configs | 2h |
| Learning curves | Training dynamics | 2h |
| SHAP explanations | Feature importance deep-dive | 4h |
| ROC/PR curves per class | Detailed classification analysis | 2h |

#### Step 4: Model Persistence & Deployment
```python
# Future: model_registry.py (NOT IMPLEMENTED)

class ModelRegistry:
    """Store, version, and retrieve trained models."""
    
    def save_model(self, model, config, metrics):
        """Save model with full provenance tracking."""
        pass
    
    def get_best_model(self, metric='kappa'):
        """Retrieve best performing model."""
        pass
    
    def export_for_deployment(self, model_id, format='onnx'):
        """Export model for production use."""
        pass
```

### Implementation Timeline (Post-Thesis)

| Phase | Timeframe | Tasks | Effort |
|-------|-----------|-------|--------|
| **Phase 1** | Feb 2026 | Add mode flag, grid configs, LR/SVM models | 4h |
| **Phase 2** | Feb 2026 | Hyperparameter search integration | 6h |
| **Phase 3** | Mar 2026 | Extended visualizations, SHAP | 8h |
| **Phase 4** | Mar-Apr 2026 | Web dashboard, model registry | 15h |

**Total Estimated Effort**: 33 hours (spread over 2-3 months post-thesis)

### Decision Record

**Date**: December 23, 2025  
**Decision**: Defer dual-mode architecture to post-thesis  
**Rationale**: 
- Current architecture already supports future extension
- No blocking refactoring needed
- Thesis deadline takes absolute priority
- "Working thesis" > "Perfect architecture"

**Revisit Date**: January 16, 2026 (day after submission)

---

## ✅ Quick Reference: What to Implement Now vs Later

### 🔴 IMPLEMENT NOW (Thesis-Critical) — ~5-6 hours total

- [ ] **LOSOFingerprint class** (30 min) - includes holdout_subject!
- [ ] **LOSOModelCache class** (1 hour) - save/load with fingerprint validation
- [ ] **Training integration** (2 hours) - cache check before train, save after
- [ ] **Demo script** (30 min) - show 100% hit rate on second run
- [ ] **Run 18-config thesis grid** (4-8 hours runtime)
- [ ] **Generate results tables** for thesis Chapter 5

### ✅ ALREADY COMPLETE

- [x] Feature extraction and caching (224× speedup)
- [x] LOSO cross-validation splitter
- [x] 18-configuration thesis grid defined
- [x] XGBoost and Random Forest models
- [x] Evaluation metrics (Accuracy, F1, Kappa)
- [x] Visualization (confusion matrix, cache performance)
- [x] Feature selection (ANOVA + GLOBAL scope)

### 🟡 IMPLEMENT IF TIME PERMITS

- [ ] Logistic Regression baseline
- [ ] SVM baseline  
- [ ] 8-channel experiments
- [ ] Extended visualizations

### 🔴 DEFER TO FUTURE WORK

- [ ] **Dual-mode pipeline architecture** (Thesis vs Exploration modes)
- [ ] Deep learning (CNN/LSTM)
- [ ] Distributed caching
- [ ] ML-based cache prediction
- [ ] Hyperparameter search (Optuna)
- [ ] Web dashboard (Streamlit)

---

## 📝 Thesis "Future Work" Section Draft

> ### 6.5 Future Work
>
> The presented framework establishes a foundation for intelligent ML experiment caching with several promising extension directions:
>
> **Validation Strategies**: While this work employs Leave-One-Subject-Out cross-validation for rigorous generalization assessment, the framework architecture supports alternative strategies including subject-grouped k-fold validation and nested cross-validation for hyperparameter tuning.
>
> **Model Extensions**: The current implementation demonstrates caching benefits with tree-based ensembles (Random Forest, XGBoost) and feed-forward neural networks. Future work could extend evaluation to support vector machines, gradient boosting variants (LightGBM, CatBoost), and end-to-end deep learning architectures operating on raw signal tensors.
>
> **Deep Learning Integration**: The modular caching architecture could be adapted for deep learning pipelines by caching preprocessed epoch tensors rather than extracted features. This would enable automatic feature learning through convolutional or recurrent networks while maintaining the computational efficiency benefits of intelligent caching.
>
> **8-Channel Feature Extraction (195 features)**: The current thesis uses 6 EEG channels (149 features). The codebase already supports 8-channel extraction (adding EOG and EMG, producing 195 features), which could improve REM detection (EOG) and wake detection (EMG). This was not included in the thesis experiments to maintain a consistent feature set across all configurations, but the implementation is ready for future evaluation, particularly with deep learning models that may benefit from the additional physiological signals.
>
> **Production-Grade Error Handling**: The current error handling is sufficient for research use (per-subject try/catch in the main pipeline loop, graceful degradation on cache misses). A production deployment would benefit from a consistent error handling strategy across all modules, structured logging with error codes, and automatic retry logic for transient failures (e.g., disk I/O errors during cache operations).
>
> **Advanced Caching Strategies**: The current SHA-256 fingerprinting approach provides deterministic cache invalidation. Future work could explore ML-based cache access prediction, partial feature loading, and distributed caching for multi-machine environments.
>
> **Cross-Domain Validation**: While demonstrated on sleep stage classification, the framework's design is domain-agnostic. Validation on other time-series classification tasks (e.g., seizure detection, emotion recognition, motor imagery) would establish broader applicability.

---

*Document End*

**Last Updated**: December 27, 2025  
**Next Review**: After LOSO Model Cache implementation