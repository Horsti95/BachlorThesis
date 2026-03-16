# Spannungsfelder (Tension Fields) in ML Caching for Sleep Stage Classification

**Based on experimental results from the Bachelor Thesis**
**Date:** March 2026
**Author:** Lennart Gorzel

---

## Overview

These Spannungsfelder represent inherent trade-offs and tensions discovered through our experiments. Each tension field is grounded in actual experimental data and highlights a non-trivial design decision or finding.

---

## 1. Speedup vs. Storage Cost

**The core trade-off of model caching.**

| Model | Speedup | Cache Size | Verdict |
|-------|---------|-----------|---------|
| XGBoost | 208x avg | ~185 MB | VIABLE |
| Gradient Boosting | 18,030x | 2.01 MB | VIABLE |
| Random Forest | 28x avg | ~18.5 GB | BORDERLINE |
| Extra Trees | 4x | 1,544 MB | NOT VIABLE |

**Tension:** Models with the highest speedup potential (Gradient Boosting: 18,030x) also have the smallest caches (2 MB). Meanwhile, ensemble tree models (RF, ET) that store individual trees require orders of magnitude more disk space for modest speedups.

**Insight:** This is NOT a true trade-off — it's a **separation**. Boosting models are inherently cache-friendly (compact serialized form), while bagging ensembles are inherently cache-hostile (many independent trees). The architect doesn't choose between speedup and storage; the model family determines both.

**Implication for practitioners:** When caching is a requirement, prefer boosting over bagging. The accuracy penalty is minimal (XGBoost 85.5% vs RF 81.9% on LOSO), and the storage reduction is 100x.

---

## 2. Accuracy vs. Cacheability — The Surprising Non-Trade-Off

**Expected:** Higher-performing models should be harder to cache.
**Found:** The opposite is true.

| Rank | Model | Accuracy (train/test) | Cache Viable? | Speedup |
|------|-------|-----------------------|---------------|---------|
| 1 | LightGBM | 91.3% | YES | 90x |
| 2 | Gradient Boosting | 90.2% | YES | 18,030x |
| 3 | XGBoost | 90.2% | YES | 119x |
| 4 | Extra Trees | 88.9% | NO | 4x |
| 5 | Random Forest | 88.3% | NO | 15x |

**Tension (resolved):** There IS no accuracy-cacheability trade-off for this dataset. The top-3 models are all cache-viable with excellent speedups. The first non-cacheable model (Extra Trees) ranks 4th.

**Why this matters for the thesis:** This finding is stronger than expected. We can argue that caching not only doesn't sacrifice accuracy — it naturally favors the best-performing model families. This is a non-trivial finding that strengthens the caching narrative.

---

## 3. Feature Dimensionality: Accuracy Saturation vs. Speedup Growth

**More features = higher accuracy AND higher cache speedup, but at different rates.**

| Top-K | Features | Accuracy | Speedup | Accuracy Gain | Speedup Gain |
|-------|----------|----------|---------|---------------|--------------|
| 10 | 10 | 72.3% | 39.8x | baseline | baseline |
| 30 | 30 | 78.2% | 48.7x | +5.9% | +22% |
| 50 | 50 | 81.3% | 66.3x | +3.1% | +36% |
| 149 | 149 | 81.9% | 143.3x | +0.6% | +116% |

**Tension:** Accuracy saturates around k=50 (diminishing returns), but speedup continues to grow near-linearly with feature count. The "sweet spot" depends on what you optimize for:
- **If optimizing accuracy:** k=50 captures 99.3% of max performance
- **If optimizing speedup ratio:** k=149 provides 2.2x higher speedup than k=50
- **If optimizing wall-clock:** k=50 has lower cold-start cost (12.5s vs 28.1s)

**Implication:** For iterative experimentation, use all features (max speedup on re-runs). For deployment, k=50 offers the best accuracy/complexity ratio.

---

## 4. Subject Count Scaling: Quadratic Training vs. Constant Loading

**Cache loading time is O(1) while training time grows super-linearly.**

| Subjects | SVM-RBF Cold (s) | SVM-RBF Warm (s) | Speedup |
|----------|-------------------|-------------------|---------|
| 10 | ~few seconds | ~constant | 9.2x |
| 30 | ~tens of seconds | ~constant | 33.4x |
| 64 | ~minutes | ~constant | 82.3x |
| 128 | ~many minutes | ~constant | 200.9x |

**Tension:** The bigger the dataset, the MORE valuable caching becomes. This is counterintuitive — you might expect cache overhead (storage, I/O) to scale with data size, negating benefits. But loading a serialized model from disk is effectively O(1) regardless of how long training took.

**Thesis significance:** This means caching becomes MORE justified as research scales up. A dataset with 1000 subjects would see even larger speedup ratios. The cache investment pays increasing dividends.

---

## 5. Parallelization vs. Caching: Competing Speedup Strategies

**Both strategies reduce wall-clock time. When does each win?**

| Model | Parallel Speedup (1→24 cores) | Cache Speedup | Which Wins? |
|-------|-------------------------------|---------------|-------------|
| Random Forest | 16.8x | 15x | **TIE** — both similar |
| Extra Trees | 10.3x | 4x | **Parallel wins** |
| XGBoost | 5.9x | 119x | **Caching wins by 20x** |
| LightGBM | 4.1x | 90x | **Caching wins by 22x** |

**Tension:** For models that parallelize well (RF, ET), adding cores can match or beat caching. For models that parallelize poorly (XGB, LGB — sequential boosting), caching provides 20x more speedup than maxing out cores.

**Critical insight:** These strategies are COMPLEMENTARY, not competing:
- **Cold run:** Use parallelization (all cores, maximize training speed)
- **Warm run:** Use caching (skip training entirely)
- **Optimal:** Cache + parallel cold runs = fastest possible pipeline

**Implication:** The thesis should frame caching as complementary to, not replacing, hardware scaling.

---

## 6. Determinism vs. Model Expressiveness (FNN Caching Problem)

**Fingerprint-based caching assumes: same input → same output.**

| Property | sklearn/XGBoost | PyTorch FNN |
|----------|-----------------|-------------|
| Deterministic training | Yes (with seed) | No (even with seed*) |
| Cache fingerprint valid | Always | Only for inference |
| Retraining matches cache | 100% | ~0% (different weights) |

*GPU non-determinism, floating-point ordering, cuDNN algorithm selection

**Tension:** Neural networks (FNN, CNN, LSTM) are non-deterministic during training. Caching the trained model is still valid for INFERENCE (same model → same predictions), but:
- Retraining will produce a DIFFERENT model
- The fingerprint can't distinguish "same config, different training run"
- You cache a specific model instance, not a reproducible result

**Resolution:** Cache the trained model + its predictions. The cache contract changes from "deterministic reproduction" to "checkpoint preservation." This is valid but conceptually different from the sklearn case.

**Future work direction:** Explore whether FNN caching is viable given that (a) training time is significant, (b) model size is small (~5 MB), and (c) inference caching would still provide speedup for LOSO folds.

---

## 7. Validation Rigor vs. Computational Cost (The LOSO Paradox)

**LOSO is the gold standard for subject-independent evaluation. It's also absurdly expensive.**

| Validation | Folds | Training Runs | Cold Time | Hot Time |
|------------|-------|---------------|-----------|----------|
| 5-fold CV | 5 | 5 | ~3 min | ~0.3s |
| 10-fold CV | 10 | 10 | ~6 min | ~0.5s |
| LOSO (128 subj) | 128 | 128 | ~61 min | ~7.4s |

**Tension:** Without caching, LOSO is 25x more expensive than 5-fold CV. Most researchers avoid LOSO because it's too expensive, defaulting to random K-fold that leaks subject-specific patterns into the test set.

**Resolution (our thesis contribution):** Caching makes LOSO tractable. After the first cold run (61 min for XGBoost), subsequent runs take 7.4 seconds. This is FASTER than a single cold 5-fold CV run. Caching doesn't just save time — it enables methodologically superior validation.

**Thesis framing:** "Intelligent caching democratizes LOSO cross-validation by reducing its computational barrier from hours to seconds, removing the need to compromise on validation rigor."

---

## 8. Hardware Independence vs. Absolute Performance

**Cache viability verdicts are consistent across hardware, but absolute speedups differ.**

| Model | New Machine Speedup | Old Machine Speedup | Same Verdict? |
|-------|---------------------|---------------------|---------------|
| Gradient Boosting | 10,020x | 5,026x | YES (VIABLE) |
| XGBoost | 99x | 208x | YES (VIABLE) |
| Random Forest | 7x | 13x | YES (NOT VIABLE) |
| Extra Trees | 4x | 4x | YES (NOT VIABLE) |

**Tension:** Speedup numbers vary 2-3x between machines, but the VIABLE/NOT VIABLE classification is stable. This means:
- You can't cite exact speedup numbers as universal
- But you CAN cite viability classifications as generalizable

**Implication for thesis:** Report speedup ranges, not point estimates. The contribution is the viability framework, not specific speedup factors.

---

## 9. Energy Efficiency vs. Raw Performance

**CPU governor affects wall time but NOT cache viability.**

| Mode | RF Training (30 subj) | XGBoost Training | Cache Benefit |
|------|----------------------|------------------|---------------|
| Eco (powersave) | 3.03s | 5.72s | Same |
| Power (performance) | 1.68s | 2.98s | Same |
| Ratio | 1.8x slower | 1.9x slower | Unchanged |

**Tension:** Running in eco mode nearly doubles training time, but caching provides the same absolute benefit. The relative speedup of caching is actually HIGHER in eco mode because the baseline (cold run) is slower.

**Insight:** Caching is especially valuable in energy-constrained environments (laptops, shared servers, cloud with spot pricing). The less powerful the hardware, the more valuable the cache.

---

## 10. Feature Selection Method: Speed vs. Information-Theoretic Optimality

**Expected MI to win on accuracy. It didn't.**

| Method | Time | Accuracy | Speedup vs MI |
|--------|------|----------|---------------|
| ANOVA (f_classif) | 0.022s | 84.26% | 174x |
| Mutual Information | 3.74s | 80.58% | baseline |

**Tension (resolved):** There IS no trade-off. ANOVA is 174x faster AND 3.68% more accurate. This is a strict dominance result, not a Pareto front.

**Why?** Likely because:
1. MI estimation is noisy with limited samples (binning artifacts)
2. ANOVA's F-statistic is well-suited for the Gaussian-like EEG feature distributions
3. MI's non-parametric advantage doesn't help when the class-feature relationship is approximately linear

**Thesis significance:** Justifies ANOVA as the default without needing to argue a trade-off.

---

## Summary: Spannungsfeld Map

```
                    TRUE TRADE-OFFS                  RESOLVED (Non-Trade-Offs)
                    ───────────────                  ─────────────────────────
                    3. Features: accuracy            2. Accuracy vs cacheability
                       saturates, speedup grows         (cacheable = best models)

                    4. Subjects: quadratic cost      10. ANOVA vs MI
                       vs constant loading               (ANOVA strictly dominates)

                    5. Parallel vs cache             7. LOSO cost
                       (complementary strategies)        (caching makes it free)

                    6. Determinism vs expressiveness  8. Hardware independence
                       (FNN caching problem)             (verdicts are portable)

                    9. Energy vs performance
                       (caching helps more in eco)

                    1. Speedup vs storage
                       (model family determines both)
```

---

## Actionable Recommendations from Spannungsfelder

1. **Default to boosting models** (XGBoost, LightGBM, GBM) — best accuracy AND best cacheability
2. **Use all features for cached experiments** — speedup ratio grows with dimensionality
3. **Use k=50 for deployment** — 99% of peak accuracy at lower complexity
4. **Always enable caching for LOSO** — makes gold-standard validation practically free
5. **Combine caching with parallelization** — parallel for cold runs, cache for warm runs
6. **Report speedup ranges, not point values** — hardware-independent viability is the real finding
7. **Defer FNN caching to future work** — non-determinism requires a different caching contract
8. **ANOVA is the correct feature selection default** — no reason to use MI

---

*End of Spannungsfelder Analysis*
