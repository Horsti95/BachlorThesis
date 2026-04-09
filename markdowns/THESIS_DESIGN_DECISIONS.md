# Thesis Design Decisions & Justifications
**Bachelor Thesis: ML Experiment Optimization with Intelligent Caching**  
**Author:** Lennart Gorzel  
**Institution:** IMC FH Krems  
**Date:** December 2025  

---

## Executive Summary

This document records all major design decisions made during the development of the intelligent caching system for machine learning experiments. Each decision is justified with technical, practical, and scientific reasoning to support the thesis argumentation.

---

## 1. Dataset Selection

### Decision: Use BOAS (Bitbrain Open Access Sleep) Dataset

**Rationale:**
- **Size:** 128 subjects → ideal for demonstrating LOSO cross-validation benefits
- **Complexity:** 6 EEG channels, 256 Hz sampling → computationally intensive preprocessing
- **Real-world:** Clinical sleep staging is a genuine ML application
- **Availability:** Open access, ethically approved, well-documented
- **Challenge:** ~20 GB total storage → demonstrates need for efficient caching

**Alternatives Considered:**
- Sleep-EDF (rejected: only 2 channels, different structure)
- PhysioNet databases (rejected: smaller sample sizes)
- Custom dataset (rejected: time constraints, ethics approval)

**Impact on Thesis:**
- Enables realistic demonstration of caching benefits
- Provides sufficient data for statistically significant results
- Industry-relevant problem (medical AI)

---

## 2. Sampling Rate: 256 Hz vs 128 Hz

### Decision: Downsample from 256 Hz to 128 Hz

**Technical Justification:**

| Criterion | Analysis | Conclusion |
|-----------|----------|------------|
| **Signal preservation** | Bandpass filter: 0.5-40 Hz<br>Nyquist requirement: 2 × 40 = 80 Hz<br>128 Hz > 80 Hz → no information loss | ✅ Safe |
| **Computational efficiency** | 128 Hz = 50% fewer samples<br>→ 1.5× faster preprocessing<br>→ 1.2× faster feature extraction | ✅ Significant |
| **Storage** | Preprocessed epochs: 1.2 GB → 640 MB (50% reduction) | ✅ Meaningful |
| **Literature standard** | Sleep research: 100-128 Hz typical<br>Higher rates (256 Hz) used for EEG research (>40 Hz signals) | ✅ Justified |

**Scientific References:**
- Rechtschaffen & Kales (1968): Sleep scoring doesn't require >100 Hz
- AASM Manual (2015): Recommends ≥100 Hz for clinical sleep scoring
- Fiorillo et al. (2019): 128 Hz sufficient for sleep stage classification

**Impact on Results:**
- Faster cold-run times (improves cache hit rate impact measurement)
- Smaller storage footprint (makes caching more practical)
- No accuracy loss (maintains scientific validity)

**Thesis Argumentation:**
> "Downsampling to 128 Hz reduces preprocessing time by 33% and storage requirements by 50% without information loss, as the Nyquist criterion (128 Hz > 2×40 Hz) is satisfied for our bandpass filter range (0.5-40 Hz)."

---

## 3. Signal Processing Parameters

### Decision: Bandpass 0.5-40 Hz, Notch 50 Hz (European Standard)

**Bandpass Filter (0.5-40 Hz):**

| Component | Frequency Range | Justification |
|-----------|----------------|---------------|
| **High-pass (0.5 Hz)** | Remove DC drift | Standard for EEG (Widmann et al., 2015) |
| **Low-pass (40 Hz)** | Remove muscle artifacts | Sleep signals: delta (0.5-4), theta (4-8), alpha (8-13), sigma (12-16), beta (16-30) |

**Notch Filter (50 Hz):**
- **Europe:** 50 Hz AC power line interference (Austria, Germany, BOAS dataset collected in Spain)
- **Configurable:** Added 60 Hz option for US/Canada datasets (thesis generalizability)

**Alternative Approaches Rejected:**
- 0.3-35 Hz bandpass (too narrow, loses beta band information)
- 0.1-100 Hz bandpass (includes artifacts, unnecessary for sleep staging)
- No notch filter (power line noise degrades feature quality)

**Impact on Features:**
- Cleaner frequency-domain features (spectral entropy, power bands)
- Reduced artifact interference in complexity metrics (Hurst, DFA)

**Thesis Presentation:**
> "Preprocessing parameters (0.5-40 Hz bandpass, 50 Hz notch) follow AASM guidelines and are included in the fingerprint, ensuring cache invalidation if modified."

---

## 4. Feature Set: 149 Features

### Decision: 6 Channels × 23 Per-Channel + 11 Global = 149 Features

**Feature Categories:**

```
Per-Channel Features (23):
├── Time-Domain (10):
│   mean, std, var, min, max, ptp, rms, skew, kurtosis, zero_crossing_rate
│   → Capture amplitude and distribution characteristics
│
├── Frequency-Domain (9):
│   delta_power, theta_power, alpha_power, sigma_power, beta_power, gamma_power,
│   spectral_entropy, peak_frequency, median_frequency
│   → Capture sleep stage-specific frequency patterns
│
└── Complexity (4):
    hjorth_mobility, hjorth_complexity, hurst_exponent, detrended_fluctuation_analysis
    → Capture signal regularity and predictability

Global Features (11):
├── Coherence (6): F3-F4, F3-C3, F3-C4, F4-C4, C3-C4, O1-O2
│   → Measure synchronization between brain regions
│
├── Phase-Locking Value (3): F3-O1, F4-O2, C3-C4
│   → Measure phase synchronization
│
└── Global Metrics (2): global_entropy, global_complexity
    → Capture overall brain state
```

**Justification:**

1. **Time-domain features:**
   - Wake: higher variance, more zero crossings
   - Deep sleep (N3): lower variance, high amplitude (delta waves)
   - Literature: Berry et al. (2015) - AASM scoring manual

2. **Frequency-domain features:**
   - Wake: high beta/gamma power
   - N1: reduced alpha, increased theta
   - N2: sleep spindles (sigma band 12-16 Hz)
   - N3: dominant delta power (0.5-4 Hz)
   - REM: theta dominance, low delta
   - Literature: Iber et al. (2007), Wolpert (1969)

3. **Complexity features:**
   - Sleep depth correlates with reduced complexity (more regular)
   - Hjorth parameters: standard in EEG analysis (Hjorth, 1970)
   - Hurst/DFA: measure long-range correlations (Penzel et al., 2003)

4. **Global features:**
   - Coherence increases in deep sleep (synchronized brain activity)
   - Phase-locking captures functional connectivity
   - Literature: Massimini et al. (2005)

**Alternatives Considered:**
- 57 features (2 channels only) - rejected: BOAS has 6 channels, wasteful not to use
- 200+ features (wavelet decomposition) - rejected: redundant, longer computation
- Deep learning features (CNN) - rejected: thesis focuses on caching, not feature learning

**Thesis Contribution:**
> "The 149-feature set balances discriminative power with computational efficiency. Feature extraction is deterministic and cacheable, taking ~12 seconds per subject. Features are computed once and reused across all model configurations."

---

## 5. Cross-Validation Strategy

### Decision: LOSO (Leave-One-Subject-Out) with 128 Folds

**LOSO Justification:**

```
Train Set: 127 subjects (~106,680 epochs)
Test Set: 1 subject (~840 epochs)
Repeat: 128 times (one per subject)
```

**Why LOSO over Random K-Fold?**

| Criterion | Random K-Fold | LOSO | Winner |
|-----------|--------------|------|--------|
| **Data leakage prevention** | Epochs from same subject in train+test | No leakage | **LOSO** |
| **Generalization test** | Within-subject patterns | Cross-subject generalization | **LOSO** |
| **Clinical relevance** | Not realistic | Models unseen patients | **LOSO** |
| **Computational cost** | Lower (5-10 folds) | Higher (128 folds) | K-Fold |
| **Cache demonstration** | Less benefit | **High repetition → perfect for caching** | **LOSO** |

**Scientific Justification:**
- Sleep patterns are subject-specific (individual variability in EEG)
- Random split → model memorizes individual brain patterns → overfitting
- LOSO → true test of generalization to new patients
- Literature: Biswal et al. (2010), Sors et al. (2018)

**Cache Optimization:**
- 128 folds × 3 models = **384 training runs**
- First run: ~9 hours (all cold misses)
- Second run: ~1 hour (97% cache hits)
- **Cache benefit maximized by high repetition**

**Thesis Argumentation:**
> "LOSO cross-validation ensures realistic evaluation of model generalization to unseen subjects while providing an ideal scenario for demonstrating caching benefits: 384 training runs with identical preprocessing and features across folds."

---

## 6. Model Selection

### Decision: XGBoost, Random Forest, Feedforward Neural Network (PyTorch)

**Model Diversity Rationale:**

| Model | Type | Training Time | Cache Size | Why Include? |
|-------|------|--------------|------------|--------------|
| **XGBoost** | Gradient boosting | ~20-30s | ~15 MB/fold | Industry standard, fast, interpretable |
| **Random Forest** | Ensemble | ~40-60s | ~50 MB/fold | Robust baseline, popular in medical AI |
| **FNN** | Deep learning | ~60-90s | ~5 MB/fold | Neural network representation, PyTorch non-determinism challenge |

**Strategic Choices:**

1. **XGBoost:** 
   - Dominates Kaggle competitions
   - Deterministic (same input → same output → easy to cache)
   - Fast training → demonstrates caching overhead is minimal

2. **Random Forest:**
   - Scikit-learn standard
   - Larger model size → tests cache storage efficiency
   - Slightly slower → shows caching benefit more clearly

3. **FNN (PyTorch):**
   - **Key challenge:** Non-deterministic training (even with seed!)
   - Tests cache validity: "Is cached inference reproducible?"
   - Thesis contribution: "Cached predictions are deterministic even when training isn't"

**Alternatives Rejected:**
- SVM (too slow for 128 folds)
- CNN/LSTM (requires sequence input, different architecture)
- Ensemble of all 3 (too complex for thesis scope)

**Thesis Impact:**
> "Three models with different computational profiles demonstrate caching generalizability: fast deterministic (XGBoost), large ensemble (Random Forest), and non-deterministic neural network (PyTorch). Cache hit rates remain >95% across all model types."

---

## 7. Sleep Stage Labeling

### Decision: Use Human Consensus Labels (stage_hum)

**Label Source:**

```
BOAS Dataset provides:
- stage_hum: Consensus of 3 expert scorers (2/3 agreement)
- stage_ai: Deep learning predictions (87% agreement with human)
```

**Why Human Labels?**

| Criterion | Human (stage_hum) | AI (stage_ai) | Decision |
|-----------|------------------|---------------|----------|
| **Ground truth** | Gold standard | Derived | **Human** |
| **Inter-rater reliability** | 85% (Danker-Hopfe 2009) | 87% with human | Human |
| **Scientific validity** | AASM standard | Experimental | **Human** |
| **Thesis credibility** | Clinical standard | Circular reasoning | **Human** |

**Label Filtering:**
- `stage == 8` → Disconnection (bathroom breaks) → **FILTER OUT**
- `stage == -2` → Artifact (AI only) → **FILTER OUT**
- Valid: `stage ∈ {0, 1, 2, 3, 4}` (W, N1, N2, N3, REM)

**Impact:**
- ~2-5% of epochs filtered per subject
- Ensures training on clean, expert-validated labels
- Matches clinical practice

**Thesis Presentation:**
> "We use human consensus labels (stage_hum) from three expert scorers following AASM criteria, filtering disconnections and artifacts to ensure data quality."

---

## 8. Cache Architecture

### Decision: 4-Stage Hierarchical Cache with Fingerprint-Based Invalidation

**Cache Stages:**

```
Stage 1: Preprocessed Signals
├── Input: Raw EDF (256 Hz, 6 channels)
├── Output: Filtered, downsampled epochs (128 Hz)
├── Fingerprint includes: bandpass, notch, target_sfreq
├── Storage: ~640 MB (5 MB per subject)
└── Benefit: Saves ~20s preprocessing per subject

Stage 2: Features
├── Input: Preprocessed epochs
├── Output: 149 features × n_epochs
├── Fingerprint includes: Stage 1 fingerprint + feature config
├── Storage: ~640 MB (5 MB per subject)
└── Benefit: Saves ~12s feature extraction per subject

Stage 3: Trained Models
├── Input: Features + model config + held-out subject
├── Output: Trained model weights + predictions
├── Fingerprint includes: Stage 2 fingerprint + model params + subject_id
├── Storage: ~9 GB (XGBoost: 1.9 GB, RF: 6.4 GB, FNN: 0.6 GB)
└── Benefit: Saves ~15-90s training per fold

Stage 4: Aggregated Results
├── Input: All fold predictions
├── Output: Accuracy, F1, confusion matrices, metrics
├── Fingerprint includes: All Stage 3 fingerprints
├── Storage: ~16 MB
└── Benefit: Saves ~5s aggregation
```

**Why 4 Stages?**
- **Granularity:** Changing model params doesn't invalidate features
- **Flexibility:** Can cache only some stages (e.g., skip Stage 1 if RAM limited)
- **Transparency:** Each stage is independently verifiable

**Fingerprinting (SHA-256):**
```python
fingerprint = SHA256(
    canonical_json({
        'preprocessing': {'bandpass': [0.5, 40], 'notch': 50, 'sfreq': 128},
        'features': {'type': 'comprehensive', 'version': '1.0'},
        'model': {'name': 'xgboost', 'params': {...}},
        'held_out_subject': 'SC4001',  # CRITICAL for LOSO
        'random_seed': 42,
        'code_version': 'v1.0.0'
    })
)[:32]  # First 32 hex chars (128 bits)
```

**Why Include Held-Out Subject in Fingerprint?**
- **Data leakage prevention:** Each fold uses different train/test split
- **Cache safety:** Cannot accidentally use Fold 1's model for Fold 2
- **Thesis contribution:** Novel approach to prevent cross-contamination

**Alternatives Rejected:**
- Single-stage cache (too coarse, invalidates everything)
- No fingerprinting (unsafe, can't detect stale cache)
- Git commit hash only (doesn't capture config changes)

**Thesis Core Contribution:**
> "The 4-stage hierarchical cache with fingerprint-based invalidation ensures reproducibility while minimizing recomputation. Including the held-out subject ID in the fingerprint guarantees cache safety in LOSO cross-validation, preventing data leakage."

---

## 9. Implementation Choices

### Programming Language: Python 3.10+

**Rationale:**
- MNE-Python: Gold standard for EEG analysis
- Scikit-learn, XGBoost, PyTorch: ML ecosystem
- Pandas, NumPy: Data manipulation
- Ubiquitous in ML research

### Key Libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| mne | ≥1.5.0 | EEG/EDF reading, filtering |
| numpy | ≥1.23 | Numerical operations |
| scipy | ≥1.9 | Signal processing (spectral analysis) |
| pandas | ≥1.5 | Data manipulation |
| scikit-learn | ≥1.2 | Random Forest, metrics |
| xgboost | ≥1.7 | Gradient boosting |
| torch | ≥2.0 | Neural networks |
| pyyaml | ≥6.0 | Configuration files |

**Configuration: YAML**
- Human-readable
- Supports nested structures
- Standard for ML pipelines (Kubeflow, MLflow)

---

## 10. Experimental Design

### Configuration Space:

```
18 Total Configurations = 2 × 3 × 3:

Data Variants (2):
├── Pilot: 10 subjects (fast prototyping)
└── Full: 128 subjects (final results)

Feature Variants (3):
├── Baseline: 149 features, no correlation filter
├── Corr 0.85: Remove features with >0.85 correlation
└── Corr 0.90: Remove features with >0.90 correlation

Model Variants (3):
├── XGBoost
├── Random Forest
└── FNN (PyTorch)
```

**Experimental Protocol:**

1. **Run 1:** All 18 configs, cold start
   - Measures: Total time, cache misses
   - Expected: ~9 hours (pilot: ~45 min)

2. **Run 2:** Repeat same 18 configs
   - Measures: Cache hit rate, time saved
   - Expected: ~1 hour (pilot: ~5 min), >95% hits

3. **Run 3:** Change preprocessing (e.g., 0.5-35 Hz bandpass)
   - Measures: Cascade invalidation
   - Expected: Features + models invalidated, preprocessing cached

**Success Metrics:**

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Cache hit rate** | >80% | (hits / total requests) × 100 |
| **Time saved** | >80% | (t_cold - t_hot) / t_cold × 100 |
| **Storage overhead** | <25 GB | Total disk usage |
| **Cache load time** | <2s | Time to load cached model |

---

## 11. Limitations & Future Work

### Acknowledged Limitations:

1. **Dataset-specific:**
   - Results specific to BOAS (sleep staging)
   - Generalization to other domains unknown

2. **PyTorch non-determinism:**
   - Training varies ±0.5% even with fixed seed
   - Cached inference is deterministic, but retraining produces different models

3. **RAM constraints:**
   - Full cache (9 GB models) requires 16GB+ RAM
   - Solution: Memory-mapped Random Forest models

4. **Manual cache invalidation:**
   - Code changes require manual cache wipe
   - Solution: Include code version in fingerprint (future work)

### Future Extensions:

1. **Interstate checkpoint caching:**
   - Cache training checkpoints (e.g., every 10 epochs)
   - Currently: Only cache final model (post-training)

2. **Distributed caching:**
   - Share cache across machines/users
   - Redis or cloud storage backend

3. **Automatic hyperparameter tuning:**
   - Grid search with intelligent cache reuse
   - Bayesian optimization with warm starts

4. **Other domains:**
   - Time series forecasting
   - Image classification
   - NLP tasks

---

## 12. Timeline & Milestones

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| Architecture design | Dec 22, 2025 | ✅ Complete |
| Data loader + preprocessing | Dec 23, 2025 | ⏳ In progress |
| Feature extraction | Dec 24, 2025 | ⏳ Planned |
| Model training (no cache) | Dec 26, 2025 | ⏳ Planned |
| Fingerprinting system | Dec 28, 2025 | ⏳ Planned |
| Cache implementation | Dec 30, 2025 | ⏳ Planned |
| Experiments (pilot) | Jan 2, 2026 | ⏳ Planned |
| Experiments (full) | Jan 5, 2026 | ⏳ Planned |
| Results analysis | Jan 8, 2026 | ⏳ Planned |
| Thesis writing | Jan 10-14, 2026 | ⏳ Planned |
| **Submission deadline** | **Jan 15, 2026** | 🎯 Target |

---

## 13. Feature Selection Method: ANOVA F-classif

### Decision: Use ANOVA F-test as sole feature selection method

**Rationale:**
Feature selection serves the caching pipeline, not classification optimization. Wrapper-based methods (SFS, RFE) and multivariate approaches were benchmarked but deliberately excluded: at 2–323s per fit, they would negate the caching speedup they are meant to support.

**Benchmark Results (11 methods, 195 features, 914 samples, top-K=30):**

| Method | Runtime | Category |
|--------|---------|----------|
| Variance (unsupervised) | 0.003s | Filter |
| Chi-squared | 0.008s | Filter |
| **ANOVA F-classif** | **0.016s** | **Filter (chosen)** |
| Kruskal-H | 0.24s | Filter |
| Hybrid (ANOVA+MI) | 0.63s | Hybrid |
| Random Forest importance | 1.33s | Embedded |
| RFE (LogReg) | 1.96s | Wrapper |
| Mutual Information | 2.00s | Filter |
| XGBoost importance | 9.05s | Embedded |
| L1 LogReg (OVR) | 18.5s | Embedded |
| SFS Forward (LogReg) | 323.1s | Wrapper |

**Key finding:** All supervised methods select similar O1/O2 occipital features in top-10. The choice of method barely affects which features get selected — but runtime differs by 20,000x.

**Data:** `results/viability_benchmarks/feature_selection_methods/`

---

## 14. Thesis Focus: Caching, Not Classification

### Decision: Sleep staging pipeline is the evaluation testbed, not the contribution

**Framing:**
The classification pipeline follows established methodology (AASM standards, LOSO cross-validation, standard EEG features). This work contributes a two-tier fingerprint-based caching layer that wraps around any such pipeline.

**Recommended thesis balance:**

| Section | Sleep Lab / EEG | Caching (contribution) |
|---------|:-:|:-:|
| Background/Related Work | 30% | 70% |
| Methodology | 20% | 80% |
| Results | 20% | 80% |
| Discussion | 10% | 90% |

**Caching background should cover:**
- Why ML experiments are iterative (hyperparameter tuning, feature exploration)
- Existing approaches (MLflow, DVC, joblib.Memory) and their limitations
- Why fingerprint-based invalidation beats timestamp/manual versioning
- LOSO as particularly cache-friendly (128 independent folds, same config)

**Sleep lab background should cover (briefly):**
- AASM sleep staging standard (5 classes)
- BOAS dataset (128 subjects, 6 EEG channels)
- LOSO cross-validation rationale
- Feature extraction overview (149 features)

---

## 15. Cache Viability Across Models

### Decision: Demonstrate caching is model-dependent, propose MB/s-saved metric

**Finding:** Cache viability depends on model serialization size relative to training time saved. The MB/s-saved metric (cache_size / time_saved) provides a universal threshold:

| Verdict | MB/s-saved | Models |
|---------|-----------|--------|
| VIABLE | < 0.5 | SVM, AdaBoost, Gradient Boosting, XGBoost, LogReg, Decision Tree, CatBoost, Naive Bayes, Ridge, SVM-RBF |
| BORDERLINE | 0.5–2.0 | LightGBM |
| NOT VIABLE | > 2.0 | Random Forest (30.5 MB/fold), Extra Trees (71.6 MB), KNN (26.0 MB) |

**Validation:**
- Consistent across hardware (old laptop vs 5090 desktop)
- Consistent across scale (30 subjects vs 128 subjects)
- Consistent across feature counts (top-K 10/30/50/149)

**Data:** `results/viability_benchmarks/`

---

## 16. Experimental Results: Cold vs Warm Speedup

### Core thesis result: 14–21x speedup with fingerprint-based caching

**5090 Desktop — 18 configs, 128 LOSO folds, 2304 total models:**

| Metric | Cold | Warm (SSD cached) |
|--------|------|-------------------|
| Total time | 15.9 hours | 45.7 minutes |
| Overall speedup | — | 21x |
| Time saved | — | 15.1 hours |
| Cache hit rate | 0% | 100% |

**By model type:**

| Model | Cold total | Avg cold s/fold | Warm s/fold |
|-------|-----------|----------------|-------------|
| XGBoost (9 configs) | 4.1h | 12.9s | ~0.15s |
| Random Forest (9 configs) | 11.7h | 36.7s | ~2.1s |

**Note:** 15/18 cold configs from 5090, 3 RF corrNone from old laptop. Clean same-machine comparison for 15 configs: 10.6h cold → 45.7m warm = 14x speedup.

**Data:** `results/training_*/training_results/result_*.json`

---

## Summary

These design decisions collectively support the thesis's core argument:

> **"Fingerprint-based result caching can reduce computational costs in iterative ML experiments by >80% while ensuring reproducibility through intelligent cache invalidation."**

Every decision—from downsampling to 128 Hz to including held-out subjects in fingerprints—serves this goal while maintaining scientific rigor and practical feasibility.

---

**Document Version:** 2.0  
**Last Updated:** April 9, 2026  
**Author:** Lennart Gorzel  
**Supervisor:** Prof. Himanshu Buckchash
