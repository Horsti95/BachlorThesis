"""
Quick n_jobs Impact Benchmark
=============================
Tests how parallelism (n_jobs) affects training time for models that support it.
Thesis question: "When does throwing more cores beat caching?"

Only tests models that support n_jobs: Random Forest, Extra Trees, XGBoost, LightGBM, k-NN.
SVM is single-threaded by design, so caching is its ONLY speedup path.

Usage:
  python benchmarks_and_tests/benchmark_njobs.py --subjects 30 --max-folds 2
  python benchmarks_and_tests/benchmark_njobs.py --subjects 30 --max-folds 2 --filter xgboost
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, 'model_tryouts'))


def get_njobs_models():
    """Return only models that support n_jobs, keyed by name."""
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.neighbors import KNeighborsClassifier

    models = {
        'random_forest': {
            'base_params': dict(n_estimators=200, random_state=42, class_weight='balanced'),
            'class': RandomForestClassifier,
            'needs_scaling': False,
        },
        'extra_trees': {
            'base_params': dict(n_estimators=200, random_state=42, class_weight='balanced'),
            'class': ExtraTreesClassifier,
            'needs_scaling': False,
        },
        'knn_5': {
            'base_params': dict(n_neighbors=5),
            'class': KNeighborsClassifier,
            'needs_scaling': True,
        },
    }

    try:
        import xgboost as xgb
        models['xgboost'] = {
            'base_params': dict(
                max_depth=6, n_estimators=200, learning_rate=0.1,
                objective='multi:softmax', num_class=5,
                random_state=42, verbosity=0, eval_metric='mlogloss'
            ),
            'class': xgb.XGBClassifier,
            'needs_scaling': False,
        }
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        models['lightgbm'] = {
            'base_params': dict(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, verbose=-1, class_weight='balanced'
            ),
            'class': lgb.LGBMClassifier,
            'needs_scaling': False,
        }
    except ImportError:
        pass

    return models


def run_njobs_benchmark(
    max_subjects: int = 30,
    max_folds: int = 2,
    model_filter: str = None,
    output_dir: str = None,
    cache_dir: str = None,
):
    """Benchmark training time across different n_jobs values."""
    from all_models import load_features_from_thesis_cache

    # Detect CPU count
    cpu_count = os.cpu_count() or 4
    # n_jobs values to test: 1 core, half cores, all cores
    njobs_values = [1, max(1, cpu_count // 2), -1]
    # Remove duplicates (e.g. if cpu_count=2, half=1=already tested)
    njobs_values = sorted(set(njobs_values), key=lambda x: x if x > 0 else 9999)

    print(f"\nDetected {cpu_count} CPU cores")
    print(f"Testing n_jobs = {njobs_values}  (where -1 = all {cpu_count} cores)\n")

    # Load data
    print("Loading features...")
    X, y, subject_ids = load_features_from_thesis_cache(
        cache_dir=cache_dir, max_subjects=max_subjects
    )
    n_subjects = len(np.unique(subject_ids))
    print(f"Loaded {len(y)} epochs from {n_subjects} subjects\n")

    models = get_njobs_models()
    if model_filter:
        models = {k: v for k, v in models.items() if model_filter.lower() in k.lower()}

    if not models:
        print("No models match filter!")
        return

    logo = LeaveOneGroupOut()
    n_total_folds = len(np.unique(subject_ids))
    n_folds = min(max_folds, n_total_folds) if max_folds > 0 else n_total_folds

    results: List[Dict[str, Any]] = []

    for model_name, model_info in models.items():
        print(f"{'=' * 60}")
        print(f"  {model_name}")
        print(f"{'=' * 60}")

        for nj in njobs_values:
            nj_label = f"{nj}" if nj > 0 else f"all({cpu_count})"
            print(f"  n_jobs={nj_label}: ", end="", flush=True)

            # Create model with this n_jobs
            params = {**model_info['base_params'], 'n_jobs': nj}
            model_cls = model_info['class']
            model = model_cls(**params)

            fold_times = []
            fold_iter = logo.split(
                X.values if hasattr(X, 'values') else X, y, subject_ids
            )

            for fold_idx, (train_idx, test_idx) in enumerate(fold_iter):
                if fold_idx >= n_folds:
                    break

                X_arr = X.values if hasattr(X, 'values') else X
                X_train, X_test = X_arr[train_idx], X_arr[test_idx]
                y_train = y[train_idx]

                if model_info['needs_scaling']:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)

                m = clone(model)
                t0 = time.time()
                m.fit(X_train, y_train)
                fold_times.append(time.time() - t0)

            avg_time = np.mean(fold_times)
            std_time = np.std(fold_times)
            print(f"{avg_time:.2f}s/fold (±{std_time:.2f}s)")

            results.append({
                'model': model_name,
                'n_jobs': nj,
                'n_jobs_label': nj_label,
                'avg_fold_time_s': round(avg_time, 3),
                'std_fold_time_s': round(std_time, 3),
                'n_folds': len(fold_times),
                'n_subjects': n_subjects,
                'cpu_count': cpu_count,
            })

        print()

    # Summary table
    print("\n" + "=" * 80)
    print("  n_jobs IMPACT SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<20}", end="")
    for nj in njobs_values:
        label = f"n_jobs={nj}" if nj > 0 else f"n_jobs=all({cpu_count})"
        print(f" {label:>18}", end="")
    print(f" {'Speedup 1→all':>14}")
    print("-" * 80)

    for model_name in models:
        model_results = [r for r in results if r['model'] == model_name]
        print(f"{model_name:<20}", end="")
        times = {}
        for r in model_results:
            print(f" {r['avg_fold_time_s']:>17.2f}s", end="")
            times[r['n_jobs']] = r['avg_fold_time_s']
        # Speedup: single-core / all-core
        t1 = times.get(1, 0)
        tall = times.get(-1, times.get(max(njobs_values), 0))
        speedup = t1 / tall if tall > 0 else 0
        print(f" {speedup:>13.1f}x")

    print("-" * 80)

    # Thesis insight
    print("\nTHESIS INSIGHT:")
    print("  Models where n_jobs speedup is SMALL -> caching is the better strategy")
    print("  Models where n_jobs speedup is LARGE -> parallelism competes with caching")
    print("  SVM (not tested here) has NO n_jobs -> caching is the ONLY speedup path")

    # Save
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"njobs_benchmark_{n_subjects}subj_{timestamp}.csv"

    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark n_jobs impact on training time")
    parser.add_argument('--subjects', type=int, default=30)
    parser.add_argument('--max-folds', type=int, default=2)
    parser.add_argument('--filter', type=str, default=None)
    parser.add_argument('--cache-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    run_njobs_benchmark(
        max_subjects=args.subjects,
        max_folds=args.max_folds,
        model_filter=args.filter,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )
