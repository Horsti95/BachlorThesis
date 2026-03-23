# SCQA Presentation – 2-3 Minutes
## "ML Experiment Optimization with Intelligent Caching"
### Lennart Gorzel | IMC FH Krems

---

## SITUATION (30 sec)

In sleep research, we classify brain signals (EEG) into sleep stages — Wake, N1, N2, N3, REM.

- We use the **BOAS dataset**: 128 subjects, ~107,000 epochs of EEG data
- We extract **149 features** per epoch (spectral power, complexity measures, coherence…)
- We train ML models (XGBoost, Random Forest) using **Leave-One-Subject-Out** cross-validation
- That means: **128 folds × 18 configurations = 2,304 training runs**

This is standard practice in biomedical ML research.

---

## COMPLICATION (30 sec)

**The problem: This takes forever.**

- A single full experiment run takes **9–12 hours**
- Change one parameter? Run it all again. Another 9–12 hours.
- Most of that computation is **redundant** — same data, same features, same preprocessing
- Researchers waste days waiting for results they already computed before

> "80%+ of compute time is spent recalculating things that haven't changed."

---

## QUESTION (15 sec)

**How can we avoid redundant computation without risking wrong results?**

In other words: How do we cache intelligently — knowing *exactly* when a cached result is still valid and when it must be recomputed?

---

## ANSWER (60 sec)

### A 4-stage fingerprint-based caching system.

Every computation gets a **SHA-256 fingerprint** built from its inputs:

| Stage | What's Cached | Fingerprint Includes |
|-------|--------------|---------------------|
| 1. Preprocessing | Filtered EEG signals | Raw data + filter settings |
| 2. Features | 149 extracted features | Preprocessed data + extraction config |
| 3. Feature Selection | Selected feature subsets | Correlation threshold + top-K value |
| 4. Models | Trained ML models | Features + model params + **held-out subject ID** |

**Key insight:** The held-out subject ID is part of the model fingerprint → no data leakage possible, even with caching.

### Results

| Metric | Value |
|--------|-------|
| Cache hit rate | **100%** |
| Cold run (no cache) | **~9 hours** |
| Warm run (cached) | **~1 hour** |
| Feature cache speedup | **224×** |
| Overall speedup | **>4.5×** (expected 8–10× at scale) |

### One sentence takeaway

> **By fingerprinting every input to every computation, we cut experiment time by 80%+ while guaranteeing reproducibility — change one parameter, and only what needs to recompute, recomputes.**

---

## Speaker Notes

**Slide 1 – Situation (keep it relatable):**
"Imagine you record brain waves from 128 people sleeping. You want to automatically tell which sleep stage they're in. Standard ML problem — extract features, train models, cross-validate. Lots of computation."

**Slide 2 – Complication (show the pain):**
"Here's the catch: one full run takes 9 to 12 hours. And in research, you tweak things constantly. Changed a threshold? Start over. Tried a new model? Start over. Most of that time is wasted recalculating identical results."

**Slide 3 – Question (brief, punchy):**
"So the question becomes: can we cache results safely? The tricky part isn't caching — it's knowing when a cache is *wrong*."

**Slide 4 – Answer (confident, concrete):**
"Yes. We fingerprint every computation with SHA-256 hashes of all its inputs. If anything changes — data, parameters, even which subject is held out — the fingerprint changes and we recompute. If nothing changed, we load from cache. Result: 80% less compute, zero risk of stale results. The system is config-aware and cross-validation-safe by design."
