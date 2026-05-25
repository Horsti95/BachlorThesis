# Minto Pyramid Presentation – 2-3 Minutes
## "ML Experiment Optimization with Intelligent Caching"
### Lennart Gorzel | IMC FH Krems

---

## Full Pyramid Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        LEVEL 0: SCQA                            │
│                    (Introduction / Hook)                         │
│                                                                 │
│  S: Sleep EEG classification, 128 subjects, 2304 training runs  │
│  C: Takes 9-12h per run, 80% is redundant recomputation         │
│  Q: How to cache without risking wrong results?                 │
│  A: Fingerprint-based 4-stage caching → 80%+ speedup            │
└────────────────────────────┬────────────────────────────────────┘
                             │
          ┌──────────────────┼───────────────────┐
          │                  │                   │
          ▼                  ▼                   ▼
┌───────────────────┐ ┌────────────────┐ ┌───────────────────┐
│ LEVEL 1: ARGUMENT │ │LEVEL 1: ARGUM. │ │ LEVEL 1: ARGUMENT │
│                   │ │                │ │                   │
│  1. THE PROBLEM   │ │ 2. THE SOLUTION│ │  3. THE PROOF     │
│  Why is there     │ │ How does the   │ │  Does it actually │
│  so much waste?   │ │ caching work?  │ │  work?            │
└─────────┬─────────┘ └───────┬────────┘ └─────────┬─────────┘
          │                   │                     │
    ┌─────┴──────┐      ┌────┴─────┐         ┌─────┴──────┐
    ▼            ▼      ▼          ▼         ▼            ▼
┌────────┐┌─────────┐┌────────┐┌────────┐┌─────────┐┌─────────┐
│LEVEL 2 ││LEVEL 2  ││LEVEL 2 ││LEVEL 2 ││LEVEL 2  ││LEVEL 2  │
│Evidence││Evidence ││Evidence││Evidence││Evidence ││Evidence │
│        ││         ││        ││        ││         ││         │
│128×18= ││Same feat││4 stages││Subject ││Cold: 9h ││Feature  │
│2304    ││recalc'd ││Preproc ││ID in   ││Warm: 1h ││cache    │
│runs    ││every    ││→ Model ││finger- ││= 4.5×   ││= 224×   │
│        ││time     ││        ││print   ││speedup  ││speedup  │
└────────┘└─────────┘└────────┘└────────┘└─────────┘└─────────┘
```

---

## LEVEL 0 — SCQA: The Story (90 sec)

### S — Situation (30 sec)

In sleep research, we classify brain signals (EEG) into sleep stages — Wake, N1, N2, N3, REM.

- **BOAS dataset**: 128 subjects, ~107,000 epochs of EEG data
- **149 features** per epoch (spectral power, complexity, coherence…)
- ML models (XGBoost, Random Forest) with **Leave-One-Subject-Out** cross-validation
- **128 folds × 18 configurations = 2,304 training runs**

Standard practice in biomedical ML research.

### C — Complication (30 sec)

**This takes forever.**

- One full experiment run = **9–12 hours**
- Change one parameter? Run it all again. Another 9–12 hours.
- Most computation is **redundant** — same data, same features, same preprocessing

> "80%+ of compute time is spent recalculating things that haven't changed."

### Q — Question (15 sec)

**How can we avoid redundant computation without risking wrong results?**

How do we cache intelligently — knowing *exactly* when a cached result is still valid and when it must be recomputed?

### A — Answer (15 sec)

**A 4-stage fingerprint-based caching system** that hashes all inputs with SHA-256. Change anything → fingerprint changes → auto-recompute. Change nothing → load from cache instantly.

---

## LEVEL 1 — Three Key Arguments (45 sec)

### Argument 1: The Problem is Real (15 sec)

ML experiments are inherently iterative. Researchers tweak one parameter at a time, but the system recomputes *everything* from scratch:

- 128 subjects × 18 configs = **2,304 training runs** per experiment
- Same 149 features extracted over and over for unchanged data
- Days of GPU/CPU time wasted on identical calculations

### Argument 2: The Solution is Smart (15 sec)

Every computation gets a **SHA-256 fingerprint** built from all its inputs:

| Cache Stage | What's Cached | Fingerprint Includes |
|-------------|--------------|---------------------|
| 1. Preprocessing | Filtered EEG signals | Raw data + filter settings |
| 2. Features | 149 extracted features | Preprocessed data + extraction config |
| 3. Feature Selection | Selected subsets | Correlation threshold + top-K value |
| 4. Models | Trained ML models | Features + model params + **held-out subject ID** |

**Key insight:** The held-out subject ID is part of the fingerprint → **no data leakage possible**, even with caching.

### Argument 3: The Proof is Clear (15 sec)

It works. The numbers speak for themselves:

| Metric | Value |
|--------|-------|
| Cache hit rate | **100%** |
| Cold run (no cache) | **~9 hours** |
| Warm run (cached) | **~1 hour** |
| Feature cache speedup | **224×** |
| Overall speedup | **>4.5×** (expected 8–10× at scale) |

---

## LEVEL 2 — Supporting Evidence (reference only, not presented)

### For Argument 1 (The Problem):
- **2,304 runs per experiment** — 128 LOSO folds × 18 config combinations (2 models × 3 corr thresholds × 3 top-K)
- **Feature extraction is identical** across configs that share the same preprocessing — yet recomputed every time without caching
- **Real researcher workflow**: change corr threshold from 0.75 → 0.90, wait 9h again for 99% identical results

### For Argument 2 (The Solution):
- **4 hierarchical stages** — each stage only invalidates downstream; changing feature selection does NOT recompute preprocessing
- **SHA-256 deterministic hashing** — same inputs always produce same fingerprint, different inputs always produce different fingerprint
- **Subject ID in model fingerprint** — LOSO fold identity is part of the cache key, mathematically preventing train/test leakage

### For Argument 3 (The Proof):
- **Cold start: 9h → Warm start: 1h** — measured on full 128-subject BOAS dataset
- **224× feature cache speedup** — 53 min cold vs 0.2 min warm for 128 subjects
- **100% cache hit rate** — 24/24 models loaded correctly in validation test, 0 serialization errors, 0 feature mismatches
- **Zero data leakage** — verified by checking that no held-out subject's data appears in any cached training set

---

## Speaker Notes (combined flow)

**SCQA opening (90 sec):**
"Imagine you record brain waves from 128 people sleeping. You want to automatically tell which sleep stage they're in. Standard ML — extract features, train models, cross-validate. Here's the catch: one full run takes 9 to 12 hours. Tweak one parameter? Start over. Most of that time is wasted recalculating identical results. So the question is: can we cache safely? The answer is yes — by fingerprinting every computation with SHA-256 hashes of all inputs."

**Three arguments (45 sec):**
"Three points. First, the problem is real — 2,304 training runs per experiment, most of them redundant. Second, the solution is smart — a 4-stage cache where each stage has a fingerprint including ALL its inputs, even the held-out subject ID, so data leakage is impossible. Third, the proof is clear — 100% cache hit rate, 9 hours down to 1, and 224× speedup on feature extraction alone."

**Closing (15 sec):**
"Bottom line: fingerprint everything, cache everything, recompute only what changed. 80% less compute, zero risk."
