# Eco vs Power Mode Analysis

## What was measured
- Training time per model at n_jobs=1, n_jobs=12, n_jobs=all(24)
- Two Windows power plans: "eco" (balanced) vs "power" (high performance)

## Key findings (defensible)
1. Power mode is 1.5-2x faster than eco mode at same n_jobs setting
2. Parallelization speedup ranges from 5.7x to 17.7x (model-dependent)
3. Caching eliminates retraining entirely (100-18000x speedup), making power mode choice irrelevant for cached runs

## What was NOT measured
- Actual CPU power consumption (watts) was not measured
- No Intel RAPL, no wall-plug meter, no energy profiling
- Therefore: no claim can be made about total energy consumption (Wh)

## Thesis recommendation
Frame as: "Parallelization reduces wall-clock time by 5.7-17.7x. Whether single-core
execution consumes less total energy requires hardware power monitoring, which is outside
the scope of this study. However, caching renders this question moot by eliminating
retraining entirely."
