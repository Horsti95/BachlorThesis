# Blog-post blurb (short version)

> **Fingerprint-Based Caching for Reproducible ML Experiments**
> Bachelor thesis · IMC FH Krems · 2026

For my bachelor thesis I built a caching framework that eliminates redundant
computation in iterative machine-learning experiments. The problem surfaced in a
real sleep-staging project: evaluating models with Leave-One-Subject-Out (LOSO)
cross-validation on 128 EEG recordings meant training 128 separate models per
configuration, so a single 18-configuration sweep took roughly 21 hours — and
every small parameter tweak retrained everything from scratch, even the models
that hadn't changed. My solution gives each experiment a deterministic SHA-256
"fingerprint" derived from its full configuration (preprocessing, features, model
hyperparameters, random seed, code version, *and* the held-out subject), then
caches features and trained models against that key. Including the held-out
subject in the key structurally prevents a subtle cross-validation bug where the
wrong fold's model gets silently reused. On the sleep-staging benchmark the
framework cut the best XGBoost configuration from 3,816 seconds to 18.9 seconds
(a 202× speedup) while producing bitwise-identical predictions across all 2,304
folds, and a study of 15 model families mapped out exactly which ones are worth
caching and which ones aren't. The classifier itself reached ~85% accuracy
(Cohen's κ ≈ 0.73) on five-class sleep staging — but the real contribution is a
validated, reproducible way to stop paying for computation you've already done.

---

## Even shorter (one-liner for a project card)

A fingerprint-based caching framework that makes cross-validated ML experiments
reproducible and up to 200× faster by never recomputing a result it has already
seen — validated on EEG sleep-stage classification across 128 subjects.
