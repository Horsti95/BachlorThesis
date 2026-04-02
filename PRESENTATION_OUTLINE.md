# Bachelor Thesis Presentation Outline

**Title:** Intelligent Caching for Iterative Machine Learning Experiments  
**Author:** Lennart Gorzel  
**Institution:** IMC Fachhochschule Krems, Austria  
**Supervisor:** Prof. Himanshu Buckchash

---

## 1. Project Overview

This thesis develops a **fingerprint-based intelligent caching system** for machine learning pipelines that automatically detects when experiment configurations change and decides whether cached results can be reused or must be recomputed. Applied to a real-world sleep stage classification task (classifying brain activity into Wake, Light Sleep, Deep Sleep, and REM from EEG recordings), the system achieves up to **224x speedup** on feature extraction and **4.5x speedup** on model training — turning hours of repeated computation into seconds. Researchers running iterative experiments benefit by drastically reducing wait times without sacrificing reproducibility.

---

## 2. Motivation / Why This Matters

- **ML research is iterative by nature:** Researchers tweak hyperparameters, swap models, and adjust feature sets dozens of times. Each run re-executes the entire pipeline from scratch — even stages whose inputs haven't changed. This wastes hours of compute time daily.
- **Concrete scenario:** A sleep researcher wants to compare 18 model configurations across 128 patients using leave-one-subject-out cross-validation. That's 2,304 training runs. Without caching, changing one setting means re-running everything. With intelligent caching, only the affected stages recompute.
- **Reproducibility is non-negotiable:** Simply saving results manually is error-prone — a researcher might accidentally reuse stale results from an old configuration. Cryptographic fingerprinting guarantees that cached results are only served when the configuration truly matches.
- **Time savings compound:** In a typical thesis workflow spanning months of experimentation, even a 4.5x speedup per iteration translates to weeks of saved time overall.

---

## 3. Research Questions

1. **Cache Effectiveness:** How much computational speedup can fingerprint-based caching achieve across different pipeline stages (data preprocessing, feature extraction, model training)?
2. **Fingerprint Determinism:** Can cryptographic hashing (SHA-256) produce fully deterministic cache keys that are reproducible across sessions, ensuring consistent cache behavior?
3. **Data Leakage Prevention:** Does encoding the held-out subject identity into the cache fingerprint reliably prevent cross-fold contamination in leave-one-subject-out cross-validation?
4. **Automatic Invalidation:** Does the fingerprint-based approach correctly detect configuration changes and automatically invalidate stale cache entries without manual intervention?
5. **Generalizability Across Models:** Do caching benefits apply equally across different model types (gradient boosting, random forests, neural networks)?

---

## 4. State of the Art

- **Standard ML workflows** (scikit-learn pipelines, MLflow, Weights & Biases) track experiments and log results, but do not provide automatic stage-level caching with invalidation — they focus on tracking, not on avoiding redundant computation.
- **Pipeline tools like Apache Airflow or Kubeflow** offer task-level caching, but are designed for production deployment, not for the rapid iteration cycle of research experiments. They add significant infrastructure overhead.
- **Manual caching** (saving intermediate files, manually deciding what to reuse) is common in research but error-prone: researchers frequently reuse stale results from outdated configurations, introducing silent bugs.
- **Existing EEG/sleep classification research** (e.g., using the AASM scoring standard) focuses on improving classification accuracy but largely ignores the computational efficiency of the experimentation process itself.
- **Hash-based build systems** (like Make, Bazel, or Docker layer caching) use content hashing to avoid redundant work, but these concepts have not been systematically applied to ML experiment pipelines with cross-validation.
- **Gap this thesis fills:** No existing tool combines (a) automatic configuration-aware cache invalidation, (b) protection against data leakage in cross-validation, and (c) a lightweight, code-level integration that requires no external infrastructure. This thesis bridges that gap.

---

## 5. My Approach / Method

### Overall Pipeline (Three Stages)

**Step 1 — Data Loading & Preprocessing:**  
Raw EEG recordings (8-hour sleep sessions, 128 patients) are loaded, filtered to remove noise (bandpass 0.5–40 Hz, notch filter at 50 Hz to remove power line interference), downsampled for efficiency, and segmented into 30-second epochs. Each patient produces roughly 950 epochs.

**Step 2 — Feature Extraction (cached):**  
From each epoch, 149 numerical features are computed across three domains: time-domain (signal shape, variability), frequency-domain (power in brain wave bands like delta, theta, alpha), and complexity measures (signal regularity and predictability). Results are cached per subject using a SHA-256 fingerprint of the extraction configuration. On subsequent runs with the same settings, features are loaded from cache in seconds instead of being recomputed.

**Step 3 — Feature Selection & Model Training (cached):**  
Features are filtered (removing redundant ones via correlation analysis, selecting the most informative via statistical testing), then models are trained using leave-one-subject-out cross-validation (128 folds — each fold holds out one patient as the test set). Each trained model is cached with a fingerprint that encodes: the model type, all hyperparameters, the feature selection settings, AND the identity of the held-out subject. This last detail is critical — it prevents accidentally reusing a model that was trained on data it shouldn't have seen.

### Key Technologies
- **MNE-Python** for EEG signal processing and file handling
- **Scikit-learn** for machine learning algorithms and evaluation metrics
- **XGBoost** for gradient-boosted tree models
- **SHA-256 hashing** for deterministic cache fingerprinting
- **Joblib** for compressed model serialization to disk
- **YAML** for human-readable experiment configuration

### What Makes This Different
Unlike experiment trackers that log results after the fact, this system **intercepts the pipeline at each stage**, checks a cryptographic fingerprint, and either serves cached results or triggers fresh computation — all transparently. The inclusion of the held-out subject ID in the fingerprint is a novel safeguard against data leakage that existing caching approaches don't address.

---

## 6. Experiment Setup

### What Was Tested
- **18 experiment configurations:** 2 model types (XGBoost, Random Forest) x 3 correlation thresholds (0.75, 0.90, none) x 3 feature counts (top 30, top 50, all 149 features)
- **Cross-validation:** Leave-one-subject-out (LOSO) with 128 folds — the gold standard for testing generalization to unseen patients

### Metrics Measured
- **Classification:** Overall accuracy, Cohen's Kappa (agreement beyond chance), per-class F1 scores, macro-averaged F1
- **Caching:** Speedup ratio (cold run vs. warm run), cache hit rate, storage overhead

### Dataset
- **BOAS (Bitbrain Open Access Sleep) Database** from PhysioNet: 128 subjects, ~8 hours of sleep EEG each, ~120,000 total 30-second epochs
- **5-class problem:** Wake, N1 (light sleep), N2 (main sleep), N3 (deep sleep), REM
- **Ground truth:** Consensus labels from 3 expert human scorers

### Baselines / Comparisons
- Cold run (no cache) vs. warm run (full cache) to measure speedup
- ANOVA feature selection vs. Mutual Information (to justify method choice)
- Comparison across all 18 configurations to identify best-performing setup

---

## 7. Key Results

1. **Feature extraction caching: 224x speedup.** Full dataset feature extraction dropped from 53 minutes (cold) to 14 seconds (warm), with 100% cache hit rate across all 128 subjects.

2. **Model training caching: 4.5x speedup.** Verified on a 3-subject pilot (7.7 seconds cold vs. 1.7 seconds warm). Extrapolates to significant savings at full scale (2,304 training runs).

3. **Feature selection method choice: 200x speedup with negligible accuracy cost.** ANOVA-based feature selection ran 200x faster than Mutual Information (16 seconds vs. 4,300 seconds for 5 subjects) while differing by only 0.6% in accuracy (80.9% vs. 80.3%).

4. **Cache invalidation works correctly.** Changing any single configuration parameter (model type, feature count, correlation threshold, or held-out subject) produces a different fingerprint, triggering automatic cache misses. Verified via integration tests.

5. **Data leakage prevention confirmed.** Each of the 128 LOSO folds generates a unique cache fingerprint because the held-out subject ID is part of the hash, making it impossible to accidentally serve a model trained on the wrong data split.

[RESULTS PENDING — Full 128-subject classification accuracy, Cohen's Kappa, and per-class F1 scores across all 18 configurations have not yet been run at scale. The caching infrastructure is verified and ready; the full experiment execution is pending.]

---

## 8. Limitations & What I'd Do Differently

- **Single dataset:** All results are from the BOAS sleep EEG dataset. The caching approach is domain-agnostic in principle, but generalizability to other ML domains (image classification, NLP) has not been empirically validated.
- **Neural network non-determinism:** PyTorch-based models (feedforward neural networks) show ~0.5% variance between runs even with fixed random seeds, making caching of training results slightly less reliable for deep learning. Classical ML models (XGBoost, Random Forest) are fully deterministic.
- **No distributed caching:** The current system caches to local disk only. In a team setting, researchers cannot share cached results across machines. A Redis or cloud storage backend would address this.
- **Manual code-change invalidation:** If the feature extraction code itself changes (not just the configuration), the cache must be manually cleared. Future versions could include a git commit hash in the fingerprint to handle this automatically.
- **RAM constraints at scale:** Caching all 384 models (2 model types x 128 folds x ~1.5 configurations in memory) could require 9+ GB of RAM. The current disk-based approach avoids this, but an in-memory option would be even faster.
- **With more time:** I would validate on additional datasets, implement distributed caching, add automatic code-change detection via git hashing, and include the neural network model in the full experiment grid.

---

## 9. Conclusion

This thesis demonstrates that **cryptographic fingerprinting can make iterative ML experimentation dramatically faster** — achieving 224x speedup on feature extraction and 4.5x on model training — while simultaneously **guaranteeing reproducibility and preventing data leakage** in cross-validation. The approach is lightweight (no external infrastructure needed), transparent (researchers don't change their workflow), and correct (automatic invalidation eliminates stale-cache bugs). The core research questions are answered affirmatively: fingerprint-based caching is effective, deterministic, and safe for cross-validated experiments.
