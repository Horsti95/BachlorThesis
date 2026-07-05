# Fingerprint-Based Caching for Reproducible ML Experiments

**A detailed look at my bachelor thesis**

*Bachelor Thesis · IMC FH Krems · Supervisor: Prof. Himanshu Buckchash · 2026*
*Author: Lennart Gorzel*

> **Full title:** *Caching strategies for reproducible ML experiments on EEG
> sleep-stage classification.*

---

## 1. The one-sentence version

I built a caching framework that gives every machine-learning experiment a
deterministic cryptographic "fingerprint," so that features and trained models
are computed **once** and safely reused forever after — turning multi-hour
cross-validation sweeps into runs that finish in seconds, without ever
sacrificing reproducibility.

The framework was validated on a real, hard problem — classifying 30-second EEG
epochs into sleep stages (Wake, N1, N2, N3, REM) across 128 patients — but the
contribution is the caching methodology, not the classifier.

---

## 2. Where the problem came from

This thesis didn't start as a caching project. It started as a sleep-staging
project, and the caching idea was forced on me by a very concrete pain.

The gold-standard way to evaluate a model that has to work on *new, unseen
patients* is **Leave-One-Subject-Out (LOSO) cross-validation**: you hold out one
subject, train on the other 127, test on the held-out one, then repeat — once for
every subject. With 128 subjects, that's **128 separate models trained per
configuration**.

Now multiply that by an experiment grid. I wanted to compare models, correlation
thresholds, and feature counts. Every time I changed a single knob — a
correlation threshold, the number of features, XGBoost vs. Random Forest — the
pipeline retrained all 128 models from scratch. A 9-configuration XGBoost sweep
took ~3.8 hours cold; the same sweep with Random Forest exceeded **21 hours**. A
full 18-configuration experiment ran to roughly a day of wall-clock time — and
most of that compute was *redundant*, because the majority of those 128 models
were still perfectly valid after a small change.

That reframed the whole project. The interesting question was no longer *"how do
I make the classifier better?"* but *"why am I retraining models whose inputs
never changed?"* This is a textbook example of what Sculley et al. call the
"hidden technical debt" of ML systems — the real cost isn't the algorithm, it's
the infrastructure around it. So I made the infrastructure the thesis.

---

## 3. The core idea: configuration fingerprinting

The central mechanism is simple and, I'd argue, elegant.

Every experiment is fully described by its configuration: the preprocessing
parameters, the feature set, the model type and its hyperparameters, the random
seed, the code version, and — crucially — **which subject is held out**. I
serialize all of that into a canonical (order-independent) JSON representation
and hash it with **SHA-256**. The first 32 hex characters of that hash become the
experiment's fingerprint.

That single fingerprint does three jobs at once:

1. **Cache key** — "have I computed this exact thing before?"
2. **Reproducibility identifier** — identical config ⇒ identical hash ⇒
   guaranteed identical result.
3. **Experiment label** — a stable, human-auditable ID for every run.

Because the key is derived from the *complete* configuration, the reproducibility
guarantee is precise: a cached result is guaranteed to be bitwise-identical to a
freshly computed one. Change any parameter — even one bit of one hyperparameter —
and you get a different hash, so the stale cache entry is never returned. I
verified this: changing the subject, the correlation threshold, the top-K, or the
model each produced a completely different fingerprint, with zero collisions
across all configurations and folds.

### The one design decision that matters most

The single most important — and least obvious — decision was **putting the
held-out subject ID into the cache key.**

In LOSO, a model trained with Subject A held out is *only* valid for testing
Subject A. If you accidentally reuse it to score Subject B, you get a
scientifically invalid result — not because of data leakage (the train/test split
is fine), but because you've applied a cached artifact to the wrong experimental
condition. This is exactly the kind of bug that *no unit test catches*, because
the code runs fine and produces plausible-looking numbers.

By baking the held-out subject into the fingerprint, that entire class of bug
becomes **structurally impossible**: the fingerprints simply don't match, so the
wrong model can never be retrieved. Correctness is enforced by construction, not
by convention. That's the piece I'm proudest of, and it's the thing that
distinguishes this from generic pipeline caching (like DVC or ZenML), which
don't handle fold identity out of the box.

---

## 4. The two-layer cache architecture

Rather than one monolithic cache, I split it into two layers so that a change in
one place doesn't needlessly invalidate everything downstream.

**Layer 1 — Feature store (subject-keyed).**
Raw EEG is expensive to load and process, and feature extraction is
deterministic, so I compute all 149 features for each subject **once** and
persist them. Cold extraction of all 128 subjects is a meaningful cost; after
that, any downstream experiment — top-30 features, top-50, all 149, any model —
reads features with near-zero latency instead of touching the raw signal again.
Features are keyed only by the data + preprocessing + feature config, *not* by the
model or the fold, because they don't depend on those.

**Layer 2 — Model cache (fold-keyed).**
Each trained LOSO model is stored under the full fingerprint, *including* the
held-out subject. This is what turns a 128-fold retrain into 128 cache hits when
nothing relevant changed.

This "key / no-key" split — Layer 1 doesn't know about folds, Layer 2 does — is
reflected directly in the directory layout, and it's what makes the invalidation
behave sensibly: tweak a model hyperparameter and only Layer 2 rebuilds; the
expensive feature extraction in Layer 1 stays warm.

---

## 5. The validation domain: EEG sleep staging

I needed a workload that was genuinely expensive to compute (so caching mattered)
and scientifically legitimate (so the results meant something). Sleep staging on
the **BOAS (Bitbrain Open Access Sleep) dataset** fit both.

- **128 subjects**, full-night polysomnography, 6 EEG channels, ~119,759 labelled
  30-second epochs (~120k total).
- **Labels** are a consensus of three independent human scorers arbitrated by a
  senior expert — a real clinical gold standard, not machine-generated labels.
  (The dataset's ~87% human-AI agreement acts as a natural performance ceiling.)
- **Open access**, so anyone can reproduce every benchmark without institutional
  approval.
- **Continuity** — it was the dataset from the prior course project where I first
  hit the caching bottleneck, so the conditions were preserved.

**The pipeline:**

1. **Load** raw EEG (EDF) + human consensus annotations.
2. **Preprocess** — 0.5–40 Hz bandpass, 50 Hz notch (European power line),
   downsample 256 → 128 Hz, segment into 30-second epochs. (128 Hz is safe by the
   Nyquist criterion for a 40 Hz upper band, halves the data, and follows AASM
   sleep-scoring guidance — no information loss.)
3. **Extract 149 hand-crafted features per epoch** — time-domain (amplitude,
   variance, zero-crossings…), frequency-domain (delta/theta/alpha/sigma/beta
   band powers, spectral entropy, peak/median frequency), signal-complexity
   metrics (Hjorth parameters, Hurst exponent, DFA), and global cross-channel
   features (coherence, phase-locking). Each family maps to real sleep
   physiology — e.g. deep sleep is dominated by delta power, N2 by sigma-band
   spindles, REM by theta.
4. **Feature selection** — a correlation filter plus ANOVA top-K
   (30 / 50 / all 149).
5. **Train** XGBoost and Random Forest under 128-fold LOSO.
6. **Cache** features (Layer 1) and per-fold models (Layer 2) by fingerprint.

---

## 6. Key design decisions (and why)

A few choices beyond the fingerprint were load-bearing for the thesis:

- **LOSO over random k-fold.** Random splits let epochs from the same patient
  land in both train and test, so the model memorizes individuals and
  over-reports accuracy. LOSO tests true generalization to unseen patients — and,
  conveniently, its high repetition (128 folds) is the *ideal* stress test for
  caching.
- **149 features, deliberately more than strictly necessary.** A leaner set would
  classify fine, but a larger, more expensive feature set makes a *stronger* test
  of the feature cache and lets feature-count become a tunable configuration axis.
- **Human consensus labels, not AI labels.** Using the dataset's AI-generated
  labels to train and evaluate would be circular reasoning; human labels are the
  clinical standard.
- **Global vs. per-fold feature selection — an honest trade-off.** I fit the
  ANOVA selection once, globally, on all subjects. This is fast and keeps the
  cache key stable, but it introduces a tiny amount of label leakage
  (~0.8% per fold — each held-out subject contributes 1/128 of the ranking data).
  I measured the actual accuracy impact and it was ≤1.1%, within run-to-run
  noise. The fully-clean per-fold alternative would have crippled the warm-run
  speedup (dropping XGBoost from ~23× to ~4×), so I chose the global approach and
  documented the bound explicitly rather than hiding it. (My proposed fix — a
  "lazy per-fold selection cache" — is in the future-work section.)

---

## 7. Results

The experimental core was 18 configurations (2 models × 3 correlation thresholds
× 3 feature counts) × 128 LOSO folds = **2,304 model training operations**, run
both fully cold (uncached) and fully warm (cached) to measure the difference.
Plus a broader 15-model viability study.

### Efficiency (RQ1)

- **XGBoost: 37.5–202× speedup** (median ~54×), with a compact cache of ~185 MB.
  The best configuration (all 149 features, no correlation filter) went from a
  **3,816-second cold run to 18.9 seconds warm — a 202× speedup.**
- **Random Forest: 11–23.5× speedup**, but needed ~18.5 GB of cache. It works,
  but it's *borderline* — the storage cost is high enough that whether it's
  "worth it" depends on how many times you'll re-run.

### Model viability spectrum (RQ4)

I benchmarked 15 model families and found a clean split: **11 are cache-viable, 4
are not.** The non-viable four — Random Forest, Extra Trees, and kNN (k=5, k=10) —
fail for the same reason: their serialized models are huge (130–1,544 MB per
fold), so *loading* them from cache costs almost as much as *retraining* them. The
viable models (all boosting methods, SVMs, logistic/ridge, decision trees)
serialize tiny.

To make this rigorous I used two complementary metrics:
- **η (eta):** seconds of training saved per MB stored — an *absolute* efficiency.
- **ρ (rho):** speedup factor per MB — a *relative* efficiency.

Both metrics agree on all 15 models, with a **near-decade gap** between the viable
cluster (η ≈ 4–5,000 s/MB) and the non-viable cluster (η ≈ 0–0.4 s/MB). The
verdicts held on two completely different machines (an HP ProBook laptop and an
RTX 5090 desktop), which makes the viability conclusion hardware-robust even
though the absolute speedup numbers are not.

### Scalability — and a counterintuitive twist (RQ3)

Caching gets **more** valuable as datasets grow: SVM-RBF's speedup rose from 9× at
10 subjects to **201× at 128 subjects**, because training cost grows
super-linearly while cache-load cost stays flat.

The surprising corollary: caching also becomes more valuable on **slower**
hardware. Cache loading is I/O-bound (roughly constant across machines), while
cold training is CPU-bound (faster on better hardware). So a researcher on a
modest laptop sees *larger* speedup ratios than one on a workstation — the
framework delivers its biggest wins exactly where compute is most scarce.

### Reproducibility (RQ2)

**100%.** Every cached model produced bitwise-identical predictions to a cold
retrain across all 18 configurations and 128 folds, verified with
`numpy.array_equal()`. Identical configs always produced identical fingerprints.

### And the classifier itself?

The sleep-staging model was competitive as a side effect: the best XGBoost
configuration reached ~**85% LOSO accuracy** (Cohen's κ ≈ 0.73, macro-F1 ≈ 0.63)
on five-class staging — solidly in the expected range for hand-crafted-feature
approaches, and close to the human-AI agreement ceiling. Most errors were the
clinically expected confusions (e.g. N1, the ambiguous transition stage, is the
hardest to score even for humans).

---

## 8. Honest limitations

I was deliberate about scoping the claims:

- Only **tree-ensemble models** (XGBoost, Random Forest) were run at full scale;
  neural networks were excluded due to environment constraints.
- Results are specific to this **moderately imbalanced, five-class** sleep-staging
  task and the 149-feature pipeline — other domains may behave differently.
- The **global feature selection leakage** (~0.8%/fold, ≤1.1% accuracy impact) is
  a real methodological bound, disclosed rather than buried.
- Speedup magnitudes are **hardware-dependent**; the *viability verdicts* are
  robust, the *headline ratios* are single-hardware data points.

The framing throughout the thesis is that these findings are *conditional*: the
**mechanism** (fingerprinting, the two-layer cache, the η/ρ viability method) is
transferable; the **specific numbers** are not. The contribution is a methodology
for finding the right answer in a given context, not a universal answer.

---

## 9. What I actually built (tech stack)

- **Language:** Python 3.12
- **EEG / signal processing:** MNE-Python, SciPy, NumPy
- **ML:** scikit-learn (Random Forest, SVMs, metrics), XGBoost, plus 15 model
  families for the viability study
- **Caching core:** SHA-256 fingerprinting over canonical JSON, a two-layer
  file-based cache (per-subject NPZ features + per-fold serialized models)
- **Config & reproducibility:** YAML-driven configuration, pinned dependency
  lockfile, an end-to-end CLI pipeline with cold-vs-warm benchmarking modes
- **Scale of the empirical evaluation:** 128 subjects, ~120k epochs, 2,304
  cached-vs-cold training operations, 15-model viability sweep across two
  machines

The full pipeline is reproducible from a single command, and the dataset
downloads automatically from OpenNeuro.

---

## 10. What I took away from it

Three things I'd carry into any future ML-systems work:

1. **The bottleneck is usually infrastructure, not the model.** The most
   valuable thing I did wasn't tuning a classifier — it was making the experiment
   loop 50–200× faster and reproducible.
2. **Correctness can be structural.** Encoding the fold identity into the cache
   key turned a silent, untestable class of bug into an impossibility. Designing
   so that the wrong thing *can't* happen beats testing for it after the fact.
3. **Measure, then claim.** Two independent viability metrics, cross-hardware
   validation, and an explicitly-quantified leakage bound made the conclusions
   defensible — and taught me to state exactly which conditions a result depends
   on.

---

### Figures available to reuse on the site

The thesis includes ready-made figures you can drop into a project page (in
`thesis/figures/`): the speedup bar chart (`fig1_speedup_bar`), the η/ρ dual-metric
and viability scatter plots (`fig4c_dual_metrics`, `fig4b_viability_scatter`), the
SVM scaling curve (`fig_bonus_svm_scaling`), the pipeline overview
(`fig_pipeline_overview`), the confusion matrix and per-class F1
(`fig_confusion_matrix`, `fig6_per_class_f1`), and the sleep-stage distribution
(`fig_sleep_stage_distribution`).
