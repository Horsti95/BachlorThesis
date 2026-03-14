"""
XGBoost Cache Evaluation - Cold vs Warm LOSO Analysis
=====================================================

Reads cold training times from the cache registry, then runs a warm
pass (loading all 9 XGB configs from cache) to produce a thesis-ready
comparison table.

Usage (PowerShell, from project root):
  python model_tryouts/evaluate_xgb_cache.py
  python model_tryouts/evaluate_xgb_cache.py --cache-dir results/loso_model_cache
  python model_tryouts/evaluate_xgb_cache.py --feature-cache results/features_cache_global

Author: Lennart Gorzel
Date: March 2026
"""

import sys
import os
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fingerprint import LOSOFingerprint
from loso_cache import LOSOModelCache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_registry(cache_dir: str) -> Dict[str, Any]:
    """
    Analyze the cache registry to extract cold training time stats per config.
    """
    registry_path = Path(cache_dir) / "cache_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"No registry at {registry_path}")

    with open(registry_path, 'r') as f:
        registry = json.load(f)

    # Group by (model_type, fingerprint_base)
    # Since fingerprint includes held_out_subject, each entry has unique fingerprint
    # Group by model_type to get XGB entries
    xgb_entries = {}
    for key, info in registry.items():
        if info.get('model_type') == 'xgboost':
            fp = info['fingerprint']
            if fp not in xgb_entries:
                xgb_entries[fp] = {
                    'fingerprint': fp,
                    'subject': info['held_out_subject'],
                    'training_time_s': info['training_time_seconds'],
                    'file_size_bytes': info.get('file_size_bytes', 0),
                    'created_at': info.get('created_at', ''),
                }
            # Each fingerprint is unique (includes subject), so just store directly
            xgb_entries[key] = info

    # Group by config: entries with same training_time pattern belong together
    # Better approach: regenerate fingerprints for each config to match
    # But we don't know the configs... Instead, group by file_size_bytes (approximately)
    # Actually simplest: just report aggregate stats

    total_entries = sum(1 for _, v in registry.items() if v.get('model_type') == 'xgboost')
    total_training_time = sum(
        v['training_time_seconds']
        for v in registry.values()
        if v.get('model_type') == 'xgboost'
    )
    total_size_bytes = sum(
        v.get('file_size_bytes', 0)
        for v in registry.values()
        if v.get('model_type') == 'xgboost'
    )

    n_configs = total_entries // 128 if total_entries >= 128 else 1
    per_fold_avg = total_training_time / total_entries if total_entries > 0 else 0

    return {
        'total_entries': total_entries,
        'n_configs': n_configs,
        'total_cold_training_time_s': total_training_time,
        'avg_cold_per_fold_s': per_fold_avg,
        'total_cache_size_mb': total_size_bytes / (1024 * 1024),
        'avg_file_size_mb': (total_size_bytes / total_entries / (1024 * 1024)) if total_entries > 0 else 0,
    }


def run_warm_evaluation(
    cache_dir: str,
    feature_cache_dir: str = None,
    max_subjects: int = None,
) -> pd.DataFrame:
    """
    Run warm evaluation: load all cached XGB models and evaluate accuracy.
    Compares cold training times (from registry) vs warm load times.
    """
    from all_models import load_features_from_thesis_cache

    # ── Analyze registry for cold stats ──
    print("=" * 70)
    print("XGBoost LOSO Cache Evaluation: Cold vs Warm")
    print("=" * 70)

    cold_stats = analyze_registry(cache_dir)
    print(f"\nCache Registry Analysis:")
    print(f"  XGB cached folds:  {cold_stats['total_entries']}")
    print(f"  XGB configs:       {cold_stats['n_configs']} (of 9)")
    print(f"  Total cold time:   {cold_stats['total_cold_training_time_s']:.1f}s "
          f"({cold_stats['total_cold_training_time_s']/3600:.2f} hrs)")
    print(f"  Avg cold/fold:     {cold_stats['avg_cold_per_fold_s']:.2f}s")
    print(f"  Total cache size:  {cold_stats['total_cache_size_mb']:.1f} MB")
    print(f"  Avg file size:     {cold_stats['avg_file_size_mb']:.2f} MB")

    # ── Load features ──
    print(f"\nLoading features...")
    X, y, subject_ids = load_features_from_thesis_cache(
        cache_dir=feature_cache_dir,
        max_subjects=max_subjects
    )
    n_subjects = len(np.unique(subject_ids))
    unique_subjects = np.unique(subject_ids)
    print(f"  Loaded {len(y)} epochs from {n_subjects} subjects")

    # ── Define the 9 XGB configs (matching thesis grid) ──
    # 3 corr × 3 top_k = 9 configs
    import xgboost as xgb

    xgb_params = {
        'max_depth': 6, 'n_estimators': 200, 'learning_rate': 0.1,
        'objective': 'multi:softmax', 'num_class': 5,
        'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
        'eval_metric': 'mlogloss'
    }

    corr_thresholds = [0.75, 0.90, None]
    top_k_values = [30, 50, None]

    configs = []
    for corr in corr_thresholds:
        for top_k in top_k_values:
            configs.append({
                'corr': corr,
                'top_k': top_k,
                'label': f"corr={corr}, top_k={top_k}",
            })

    # ── Run warm evaluation per config ──
    results = []
    logo = LeaveOneGroupOut()

    for cfg_idx, cfg in enumerate(configs):
        label = cfg['label']
        print(f"\n[{cfg_idx+1}/9] Config: {label}")
        print("-" * 50)

        feature_config = {
            'base': X.shape[1],
            'corr': cfg['corr'],
            'top_k': cfg['top_k'],
        }

        # Apply feature selection to match what was used during cold training
        # For now, we use all features (the fingerprint will match or not based on config)
        X_run = X.values if hasattr(X, 'values') else X

        cache = LOSOModelCache(
            cache_dir=cache_dir,
            enable_registry=True,
            estimated_training_time=cold_stats['avg_cold_per_fold_s'],
        )

        warm_preds = []
        warm_true = []
        warm_subjects = []
        cache_hits = 0
        cache_misses = 0

        warm_start = time.time()

        for fold_idx, (train_idx, test_idx) in enumerate(
            tqdm(logo.split(X_run, y, subject_ids),
                 total=n_subjects, desc=f"  Warm {label}", unit="fold", leave=False)
        ):
            held_out = str(subject_ids[test_idx[0]])
            X_test = X_run[test_idx]
            y_test = y[test_idx]

            # Generate fingerprint (must match cold run)
            fingerprint = LOSOFingerprint.generate(
                random_seed=42,
                model_config={'name': 'xgboost', 'params': xgb_params},
                feature_config=feature_config,
                held_out_subject=held_out,
            )

            # Try cache
            cached_model = cache.get(
                fingerprint=fingerprint,
                held_out_subject=held_out,
                model_type='xgboost',
                record_metrics=True,
            )

            if cached_model is not None:
                cache_hits += 1
                # Need to scale if the cold run scaled
                # XGBoost doesn't need scaling, so predict directly
                y_pred = cached_model.predict(X_test)
                warm_preds.extend(y_pred)
                warm_true.extend(y_test)
                warm_subjects.append(held_out)
            else:
                cache_misses += 1

        warm_total = time.time() - warm_start

        if warm_preds:
            acc = accuracy_score(warm_true, warm_preds)
            kappa = cohen_kappa_score(warm_true, warm_preds)
            f1 = f1_score(warm_true, warm_preds, average='macro')
        else:
            acc = kappa = f1 = 0.0

        # Estimate cold time for this config from registry average
        cold_est = cold_stats['avg_cold_per_fold_s'] * n_subjects
        time_saved = cold_est - warm_total
        speedup = cold_est / warm_total if warm_total > 0 else float('inf')
        cache_size_est = cold_stats['avg_file_size_mb'] * cache_hits

        result = {
            'config': label,
            'corr': cfg['corr'],
            'top_k': cfg['top_k'],
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'hit_rate': f"{cache_hits/(cache_hits+cache_misses):.0%}" if (cache_hits+cache_misses) > 0 else "N/A",
            'warm_time_s': round(warm_total, 2),
            'warm_per_fold_s': round(warm_total / max(cache_hits, 1), 3),
            'cold_est_s': round(cold_est, 1),
            'time_saved_s': round(time_saved, 1),
            'speedup': round(speedup, 1),
            'accuracy': round(acc, 4),
            'kappa': round(kappa, 4),
            'f1_macro': round(f1, 4),
        }
        results.append(result)

        print(f"  Hits: {cache_hits}/{cache_hits+cache_misses} | "
              f"Warm: {warm_total:.1f}s | Cold est: {cold_est:.1f}s | "
              f"Speedup: {speedup:.1f}x")
        if cache_hits > 0:
            print(f"  Acc: {acc:.4f} | Kappa: {kappa:.4f} | F1: {f1:.4f}")

    # ── Summary Table ──
    df = pd.DataFrame(results)

    print("\n\n" + "=" * 100)
    print("XGBoost LOSO Cache: Cold vs Warm Summary")
    print("=" * 100)
    print(f"\n{'Config':<25} {'Hits':>5} {'Warm(s)':>8} {'Cold est(s)':>11} "
          f"{'Speedup':>8} {'Saved(s)':>9} {'Acc':>7} {'Kappa':>7}")
    print("-" * 100)

    for _, row in df.iterrows():
        print(f"{row['config']:<25} {row['cache_hits']:>5} {row['warm_time_s']:>8.1f} "
              f"{row['cold_est_s']:>11.1f} {row['speedup']:>7.1f}x "
              f"{row['time_saved_s']:>8.1f} {row['accuracy']:>7.4f} {row['kappa']:>7.4f}")

    # Totals
    total_warm = df['warm_time_s'].sum()
    total_cold = df['cold_est_s'].sum()
    total_saved = df['time_saved_s'].sum()
    avg_speedup = total_cold / total_warm if total_warm > 0 else 0

    print("-" * 100)
    print(f"{'TOTAL':<25} {df['cache_hits'].sum():>5} {total_warm:>8.1f} "
          f"{total_cold:>11.1f} {avg_speedup:>7.1f}x {total_saved:>8.1f}")

    print(f"\n  Cache size (XGB total): {cold_stats['total_cache_size_mb']:.0f} MB")
    print(f"  Time saved (total):     {total_saved:.0f}s ({total_saved/60:.1f} min)")
    print(f"  MB per second saved:    {cold_stats['total_cache_size_mb']/total_saved:.4f}" if total_saved > 0 else "")

    # Save
    output_dir = Path(cache_dir).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"xgb_cache_evaluation_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XGBoost LOSO cache: cold vs warm")
    parser.add_argument('--cache-dir', type=str, default='results/loso_model_cache',
                        help='Path to LOSO model cache directory')
    parser.add_argument('--feature-cache', type=str, default=None,
                        help='Path to thesis feature cache directory')
    parser.add_argument('--subjects', type=int, default=None,
                        help='Limit number of subjects')
    args = parser.parse_args()

    run_warm_evaluation(
        cache_dir=args.cache_dir,
        feature_cache_dir=args.feature_cache,
        max_subjects=args.subjects,
    )
