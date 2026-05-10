# Proposal: Extended Scaling Experiments

**Why we should run subject scaling, feature scaling, and combined scaling for more models**
**Date:** March 2026

---

## Current State

We currently have scaling data for only a few models:
- **Subject scaling (10→128):** SVM-Linear, SVM-RBF only
- **Feature scaling (k=10→149):** XGBoost only (30 subjects, 3 folds)
- **Combined scaling:** None

This is a gap. We have 15 models evaluated for cache viability, but scaling behavior for only 2-3 of them.

---

## Why This Is Worth Doing

### Argument 1: The Scaling Curves May Not Be Monotonic

Our SVM-RBF results show a clean quadratic scaling pattern (9x → 33x → 82x → 201x). But this might be specific to SVMs with their O(n^2) kernel computation. Other models could show:

- **Sublinear scaling (RF, ET):** These models train individual trees independently. Adding subjects increases data per tree but not tree count. The speedup curve might flatten earlier.
- **Threshold effects (boosting):** XGBoost/LightGBM use early stopping. More data might trigger more rounds, causing a jump in training time at certain thresholds.
- **Regime changes:** Some models might flip from VIABLE to NOT VIABLE (or vice versa) at certain dataset sizes.

**Non-trivial finding potential:** If a model that's VIABLE at 128 subjects becomes NOT VIABLE at 256+ subjects (cache size grows faster than speedup), that's a genuinely useful insight for practitioners planning to scale.

### Argument 2: Feature Scaling Interacts With Model Type Differently

Our XGBoost feature scaling shows speedup grows near-linearly with features (39.8x → 143.3x). But:

- **Linear models (Logistic Regression, SVM-Linear):** Feature count directly affects the weight matrix size. Scaling should be roughly linear in both training and cache size.
- **Tree models (RF, Decision Tree):** More features means more split candidates but doesn't change tree structure. Feature scaling might barely affect training time.
- **Kernel models (SVM-RBF, KNN):** Feature count affects distance computation (O(n*d)). Doubling features doubles kernel matrix computation.

**Non-trivial finding potential:** If tree-based models show flat speedup curves with feature count (because tree training time is dominated by sample count, not feature count), that contradicts the intuition from XGBoost. This would show that the optimal feature selection strategy depends on the model family.

### Argument 3: Combined Scaling Reveals the Dominant Factor

The real question practitioners face: "I have X subjects and Y features. How much will caching help?"

With individual scaling curves, you can't answer this because interactions exist:
- More subjects × more features → larger training set → longer training → higher speedup
- But also: more subjects × more features → larger model → larger cache → slower loading

**The interaction might be non-additive.** For example:
- SVM-RBF with 10 subjects and 149 features: training time dominated by features
- SVM-RBF with 128 subjects and 10 features: training time dominated by subjects
- SVM-RBF with 128 subjects and 149 features: could be multiplicative (128x × 149 dimensions)

A 2D grid (subjects × features) for 3-4 key models would produce a **heatmap of cache viability** — showing exactly where caching crosses from "marginal" to "essential."

### Argument 4: It Strengthens the Viability Framework

Our current viability classification (VIABLE / NOT VIABLE) is static — measured at one point (128 subjects, 149 features, 5 folds). With scaling data, we can make it dynamic:

```
STATIC:   "XGBoost caching is VIABLE" (one data point)
DYNAMIC:  "XGBoost caching is VIABLE for datasets with >10 subjects,
           becomes increasingly beneficial beyond 30 subjects,
           and achieves >100x speedup beyond 50 subjects"
```

This transforms a binary verdict into a **viability curve** — much more useful for practitioners.

### Argument 5: Low Marginal Cost (We Already Have the Infrastructure)

The `run_cache_viability_evaluation.py` script already supports:
- Configurable subject counts
- Configurable feature counts (top-K)
- Multiple models
- Automated cold/warm timing

Running extended scaling is ~2-4 hours of compute time on the existing infrastructure. The analysis code already exists. The marginal cost of insight is low.

---

## Proposed Experiment Design

### Phase 1: Subject Scaling (6 models × 4 subject counts)

| Model | 10 subj | 30 subj | 64 subj | 128 subj |
|-------|---------|---------|---------|----------|
| XGBoost | run | run | run | have |
| LightGBM | run | run | run | have |
| Gradient Boosting | run | run | run | have |
| Random Forest | run | run | run | have |
| SVM-Linear | have | have | have | have |
| SVM-RBF | have | have | have | have |

**New runs needed:** 4 × 4 = 16 runs (we already have SVM data + 128-subj data)

### Phase 2: Feature Scaling (4 models × 4 feature counts)

| Model | k=10 | k=30 | k=50 | k=149 |
|-------|------|------|------|-------|
| XGBoost | have | have | have | have |
| LightGBM | run | run | run | run |
| Random Forest | run | run | run | run |
| SVM-RBF | run | run | run | run |

**New runs needed:** 3 × 4 = 12 runs (we already have XGBoost data)

### Phase 3: Combined Scaling (2 models × 3×3 grid)

For XGBoost and SVM-RBF (most interesting contrast):

| | k=30 | k=50 | k=149 |
|---------|------|------|-------|
| 10 subj | run | run | run |
| 30 subj | partial | partial | partial |
| 128 subj | have | have | have |

**New runs needed:** ~12 runs (partially covered by Phase 1+2)

### Total Estimate

- **New runs:** ~30-35 individual evaluations
- **Time per run:** 5-15 minutes (3 folds, reduced subjects)
- **Total compute:** ~4-8 hours
- **Analysis:** ~2 hours (scripts already exist)

---

## Expected Outputs

1. **Subject scaling curves** for 6 models — showing how speedup grows with dataset size
2. **Feature scaling curves** for 4 models — showing how speedup responds to dimensionality
3. **2D heatmap** for XGBoost and SVM-RBF — showing combined subject × feature scaling
4. **Dynamic viability thresholds** — "caching becomes viable at X subjects for model Y"
5. **Scaling regime classification** — which models scale linearly, quadratically, or show threshold effects

---

## What Could Go Wrong (Risk Assessment)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| All models scale identically | Low | Boring result, still publishable | Focus on the few that differ |
| Compute takes too long | Medium | Delay thesis | Use 3 folds, not 128 |
| Results contradict existing claims | Low | Must revise discussion | Our existing claims are model-specific, not universal |
| Data is too noisy (small subject counts) | Medium | Unreliable curves | Use 3-fold average, report error bars |

---

## Recommendation

**Do Phase 1 (subject scaling) as minimum.** This directly extends our existing SVM scaling results to the models people actually care about (XGBoost, LightGBM, RF). It's ~2 hours of compute and would produce the most impactful figures.

Phase 2 and 3 are bonus — nice for a very thorough thesis discussion section or a follow-up paper, but not strictly necessary.

**Even if we don't run the experiments now, the proposal itself belongs in the Future Work section** as a concrete, well-motivated research direction.

---

*End of Proposal*
