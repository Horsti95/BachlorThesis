"""
Global vs Per-Fold Feature Selection Benchmark
================================================

Compares global ANOVA (current thesis approach) vs per-fold ANOVA
(scientifically correct, no leakage) on small subject samples.

Measures:
  - Accuracy difference between scopes (expected: ~negligible)
  - Cold run time (per-fold adds ~22ms × N_folds overhead)
  - Warm run time (model cache still works for both scopes)

Usage:
    python benchmark_global_vs_perfold.py
    python benchmark_global_vs_perfold.py --cache "C:/path/to/features_cache_global"

Output:
    results/global_vs_perfold_benchmark.csv
    results/global_vs_perfold_benchmark.json
"""

import sys
import time
import json
import shutil
import argparse
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

DEFAULT_CACHE = REPO / "results" / "features_cache_global"
OUT_DIR       = REPO / "results"

# ── Benchmark matrix ────────────────────────────────────────────────────────
SUBJECT_COUNTS     = [3, 5, 10, 20]
MODELS             = ["xgboost", "random_forest"]
CORR_THRESHOLDS    = [None, 0.75, 0.90]
SCOPES             = ["global", "per_fold"]
TOP_K              = 50   # fixed; makes global vs per_fold actually differ


def get_subject_ids(cache_dir: Path, n: int) -> list[int]:
    files = sorted(cache_dir.glob("subject_*_full.npz"))
    ids = []
    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            ids.append(int(parts[1]))
    return sorted(ids)[:n]


def run_config(features_df, labels, subject_ids_arr, config,
               cache_dir: Path, run_label: str) -> dict:
    from training import TrainingPipeline

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = TrainingPipeline(
            features_df, labels, subject_ids_arr,
            output_dir=Path(tmp),
            model_cache_dir=str(cache_dir),
            enable_model_cache=True,
        )
        result = pipeline.run_single_config(config, show_progress=False)
    elapsed = time.perf_counter() - t0

    cache_hits = sum(1 for fr in result.fold_results if getattr(fr, "cache_hit", False))
    n_folds    = len(result.fold_results)

    return {
        "run_type":           run_label,
        "total_time_s":       round(elapsed, 2),
        "time_per_fold_s":    round(elapsed / n_folds, 3) if n_folds else 0,
        "accuracy":           round(result.accuracy_mean, 4),
        "kappa":              round(result.kappa_mean, 4),
        "f1_macro":           round(result.f1_macro_mean, 4),
        "cache_hits":         cache_hits,
        "n_folds":            n_folds,
    }


def main(cache_dir: Path) -> None:
    from run_training import load_cached_features
    from training import TrainingConfig
    from feature_selection import FeatureSelectionConfig

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    total_combos = (len(SUBJECT_COUNTS) * len(MODELS) *
                    len(CORR_THRESHOLDS) * len(SCOPES))
    done = 0

    for n_subj in SUBJECT_COUNTS:
        subject_ids = get_subject_ids(cache_dir, n_subj)
        print(f"\n{'='*60}")
        print(f"  N={n_subj} subjects: {subject_ids}")
        print(f"{'='*60}")

        features_df, labels, subject_ids_arr = load_cached_features(
            subject_ids, cache_dir)

        for model_type in MODELS:
            for corr in CORR_THRESHOLDS:
                for scope in SCOPES:
                    done += 1
                    corr_str = str(corr) if corr else "None"
                    tag = f"[{done}/{total_combos}] N={n_subj} {model_type} corr={corr_str} scope={scope}"
                    print(f"\n  {tag}")

                    fs_config = FeatureSelectionConfig(
                        correlation_threshold=corr,
                        top_k_features=TOP_K,
                        selection_method="anova",
                        scope=scope,
                        random_state=42,
                    )
                    config = TrainingConfig(
                        model_type=model_type,
                        feature_selection=fs_config,
                        random_state=42,
                    )

                    base_row = {
                        "n_subjects":        n_subj,
                        "model":             model_type,
                        "corr_threshold":    corr_str,
                        "scope":             scope,
                        "top_k":             TOP_K,
                    }

                    with tempfile.TemporaryDirectory() as model_cache_tmp:
                        cache_path = Path(model_cache_tmp)

                        # Cold run (empty model cache)
                        print(f"    cold...", end=" ", flush=True)
                        cold = run_config(features_df, labels, subject_ids_arr,
                                          config, cache_path, "cold")
                        print(f"{cold['total_time_s']:.1f}s  acc={cold['accuracy']:.3f}")

                        # Warm run (reuse model cache from cold run)
                        print(f"    warm...", end=" ", flush=True)
                        warm = run_config(features_df, labels, subject_ids_arr,
                                          config, cache_path, "warm")
                        print(f"{warm['total_time_s']:.1f}s  acc={warm['accuracy']:.3f}  "
                              f"hits={warm['cache_hits']}/{warm['n_folds']}")

                    rows.append({**base_row, **cold})
                    rows.append({**base_row, **warm})

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "global_vs_perfold_benchmark.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: accuracy difference global vs per_fold (cold runs)")
    print("="*80)
    cold_df = df[df["run_type"] == "cold"]
    for n_subj in SUBJECT_COUNTS:
        for model_type in MODELS:
            sub = cold_df[(cold_df["n_subjects"] == n_subj) &
                          (cold_df["model"] == model_type)]
            g = sub[sub["scope"] == "global"]["accuracy"].mean()
            p = sub[sub["scope"] == "per_fold"]["accuracy"].mean()
            print(f"  N={n_subj:3d}  {model_type:15s}  "
                  f"global={g:.4f}  per_fold={p:.4f}  "
                  f"diff={abs(g-p):.4f}")

    print("\nSUMMARY: warm speedup over cold")
    print("="*80)
    for scope in SCOPES:
        print(f"\n  scope={scope}")
        for n_subj in SUBJECT_COUNTS:
            for model_type in MODELS:
                c = df[(df["run_type"] == "cold") &
                       (df["n_subjects"] == n_subj) &
                       (df["model"] == model_type) &
                       (df["scope"] == scope)]["total_time_s"].mean()
                w = df[(df["run_type"] == "warm") &
                       (df["n_subjects"] == n_subj) &
                       (df["model"] == model_type) &
                       (df["scope"] == scope)]["total_time_s"].mean()
                speedup = c / w if w > 0 else 0
                print(f"    N={n_subj:3d}  {model_type:15s}  "
                      f"cold={c:.1f}s  warm={w:.1f}s  speedup={speedup:.1f}x")

    json_path = OUT_DIR / "global_vs_perfold_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(DEFAULT_CACHE),
                        help="Path to features_cache_global directory")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    if not cache_dir.exists():
        print(f"ERROR: Cache not found: {cache_dir}")
        sys.exit(1)

    n_available = len(list(cache_dir.glob("subject_*_full.npz")))
    if n_available < max(SUBJECT_COUNTS):
        print(f"ERROR: Need {max(SUBJECT_COUNTS)} subjects, found {n_available}")
        sys.exit(1)

    print(f"Cache: {cache_dir}  ({n_available} subjects available)")
    print(f"Matrix: {SUBJECT_COUNTS} subjects × {MODELS} × "
          f"{len(CORR_THRESHOLDS)} thresholds × {SCOPES} scopes × cold+warm")
    print(f"Total configurations: {len(SUBJECT_COUNTS)*len(MODELS)*len(CORR_THRESHOLDS)*len(SCOPES)*2}")

    main(cache_dir)
