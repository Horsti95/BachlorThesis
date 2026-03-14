"""
XGBoost Cache Evaluation - Cold vs Warm LOSO Analysis
=====================================================

Reads cold training times from the cache registry, then runs a warm
pass (loading cached XGB models by their stored fingerprints) to
produce a thesis-ready comparison table.

Key insight: We read fingerprints FROM the registry (not regenerate them)
so we're guaranteed to match the exact fingerprints used during training.

Usage (PowerShell, from project root):
  python model_tryouts/evaluate_xgb_cache.py
  python model_tryouts/evaluate_xgb_cache.py --cache-dir results/loso_model_cache

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
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from loso_cache import LOSOModelCache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_xgb_registry(cache_dir: str) -> Dict[str, Dict]:
    """
    Load registry and extract all XGB entries grouped by config.

    Returns dict of config_group -> list of {fingerprint, subject, training_time, file_size}
    Since each fingerprint is unique (includes held_out_subject), we group configs
    by looking at which fingerprints share the same set of subjects (= same config, different folds).
    """
    registry_path = Path(cache_dir) / "cache_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"No registry at {registry_path}")

    with open(registry_path, 'r') as f:
        registry = json.load(f)

    # Collect all XGB entries
    xgb_entries = []
    for key, info in registry.items():
        if info.get('model_type') == 'xgboost':
            xgb_entries.append({
                'cache_key': key,
                'fingerprint': info['fingerprint'],
                'subject': info['held_out_subject'],
                'training_time_s': info['training_time_seconds'],
                'file_size_bytes': info.get('file_size_bytes', 0),
                'created_at': info.get('created_at', ''),
            })

    # Group by subject to find config groups
    # All entries for the same subject but different fingerprints = different configs
    # All entries with the same fingerprint prefix pattern = same config
    # Since fingerprint = hash(config + subject), same config + different subject = different fingerprint
    # Best grouping: by creation time batches (configs were run sequentially)

    # Sort by creation time
    xgb_entries.sort(key=lambda x: x['created_at'])

    # Group into batches of 128 (each config = 128 folds)
    configs = {}
    n_subjects = 128  # Expected folds per config
    batch_size = min(n_subjects, len(xgb_entries))

    if batch_size == 0:
        return {}

    # Calculate how many complete configs we have
    n_complete = len(xgb_entries) // n_subjects
    remainder = len(xgb_entries) % n_subjects

    for cfg_idx in range(n_complete):
        start = cfg_idx * n_subjects
        end = start + n_subjects
        batch = xgb_entries[start:end]
        configs[f"config_{cfg_idx+1}"] = batch

    # Partial config at the end
    if remainder > 0:
        start = n_complete * n_subjects
        batch = xgb_entries[start:]
        configs[f"config_{n_complete+1}_partial"] = batch

    return configs


def run_warm_evaluation(
    cache_dir: str,
    feature_cache_dir: str = None,
    max_subjects: int = None,
) -> pd.DataFrame:
    """
    Run warm evaluation using fingerprints from the registry.
    """
    from all_models import load_features_from_thesis_cache

    print("=" * 70)
    print("XGBoost LOSO Cache Evaluation: Cold vs Warm")
    print("=" * 70)

    # ── Load registry and group by config ──
    configs = load_xgb_registry(cache_dir)
    if not configs:
        print("No XGB entries found in registry!")
        return pd.DataFrame()

    # Print registry summary
    total_entries = sum(len(v) for v in configs.values())
    total_cold_time = sum(e['training_time_s'] for batch in configs.values() for e in batch)
    total_size_bytes = sum(e['file_size_bytes'] for batch in configs.values() for e in batch)

    print(f"\nCache Registry Analysis:")
    print(f"  XGB cached folds:  {total_entries}")
    print(f"  XGB configs:       {len(configs)}")
    print(f"  Total cold time:   {total_cold_time:.1f}s ({total_cold_time/3600:.2f} hrs)")
    print(f"  Avg cold/fold:     {total_cold_time/total_entries:.2f}s")
    print(f"  Total cache size:  {total_size_bytes/(1024*1024):.1f} MB")
    print(f"  Avg file size:     {total_size_bytes/total_entries/(1024*1024):.2f} MB")

    # ── Load features for evaluation ──
    print(f"\nLoading features...")
    X, y, subject_ids = load_features_from_thesis_cache(
        cache_dir=feature_cache_dir,
        max_subjects=max_subjects,
    )
    X_arr = X.values if hasattr(X, 'values') else X
    n_subjects = len(np.unique(subject_ids))
    print(f"  Loaded {len(y)} epochs from {n_subjects} subjects")

    # Build subject -> test indices mapping
    subject_test_map = {}
    for idx, sid in enumerate(subject_ids):
        sid_str = str(sid)
        if sid_str not in subject_test_map:
            subject_test_map[sid_str] = []
        subject_test_map[sid_str].append(idx)

    # ── Initialize cache ──
    cache = LOSOModelCache(
        cache_dir=cache_dir,
        enable_registry=True,
        estimated_training_time=total_cold_time / total_entries,
    )

    # ── Warm evaluation per config ──
    results = []

    for cfg_name, entries in configs.items():
        print(f"\n[{cfg_name}] {len(entries)} cached folds")
        print("-" * 50)

        # Cold stats for this config
        cfg_cold_time = sum(e['training_time_s'] for e in entries)
        cfg_cache_size = sum(e['file_size_bytes'] for e in entries) / (1024 * 1024)

        warm_preds = []
        warm_true = []
        cache_hits = 0
        cache_misses = 0
        load_times = []

        warm_start = time.time()

        for entry in tqdm(entries, desc=f"  Warm {cfg_name}", unit="fold", leave=False):
            fp = entry['fingerprint']
            subject = entry['subject']

            # Load model from cache using the EXACT fingerprint from registry
            t0 = time.time()
            cached_model = cache.get(
                fingerprint=fp,
                held_out_subject=subject,
                model_type='xgboost',
                record_metrics=True,
            )
            load_time = time.time() - t0

            if cached_model is not None:
                cache_hits += 1
                load_times.append(load_time)

                # Get test data for this subject
                if subject in subject_test_map:
                    test_idx = subject_test_map[subject]
                    X_test = X_arr[test_idx]
                    y_test = y[test_idx]

                    y_pred = cached_model.predict(X_test)
                    warm_preds.extend(y_pred)
                    warm_true.extend(y_test)
            else:
                cache_misses += 1

        warm_total = time.time() - warm_start

        # Metrics
        if warm_preds:
            acc = accuracy_score(warm_true, warm_preds)
            kappa = cohen_kappa_score(warm_true, warm_preds)
            f1 = f1_score(warm_true, warm_preds, average='macro')
        else:
            acc = kappa = f1 = 0.0

        time_saved = cfg_cold_time - warm_total
        speedup = cfg_cold_time / warm_total if warm_total > 0 else float('inf')
        avg_load = np.mean(load_times) if load_times else 0
        mb_per_s_saved = cfg_cache_size / time_saved if time_saved > 0 else float('inf')

        result = {
            'config': cfg_name,
            'n_folds': len(entries),
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'hit_rate': f"{cache_hits/(cache_hits+cache_misses):.0%}" if (cache_hits+cache_misses) > 0 else "N/A",
            'cold_total_s': round(cfg_cold_time, 1),
            'cold_per_fold_s': round(cfg_cold_time / len(entries), 2),
            'warm_total_s': round(warm_total, 2),
            'warm_per_fold_s': round(avg_load, 4),
            'speedup': round(speedup, 1),
            'time_saved_s': round(time_saved, 1),
            'cache_size_mb': round(cfg_cache_size, 1),
            'mb_per_s_saved': round(mb_per_s_saved, 4),
            'accuracy': round(acc, 4),
            'kappa': round(kappa, 4),
            'f1_macro': round(f1, 4),
        }
        results.append(result)

        print(f"  Hits: {cache_hits}/{cache_hits+cache_misses} | "
              f"Warm: {warm_total:.1f}s ({avg_load:.3f}s/fold) | "
              f"Cold: {cfg_cold_time:.1f}s ({cfg_cold_time/len(entries):.2f}s/fold)")
        print(f"  Speedup: {speedup:.1f}x | Cache: {cfg_cache_size:.1f} MB | "
              f"MB/s-saved: {mb_per_s_saved:.4f}")
        if cache_hits > 0:
            print(f"  Acc: {acc:.4f} | Kappa: {kappa:.4f} | F1: {f1:.4f}")

    # ── Summary Table ──
    df = pd.DataFrame(results)

    print("\n\n" + "=" * 120)
    print("XGBoost LOSO Cache: Cold vs Warm Summary")
    print("=" * 120)
    print(f"\n{'Config':<22} {'Folds':>5} {'Hits':>5} {'Cold(s)':>9} {'Cold/fold':>10} "
          f"{'Warm(s)':>8} {'Warm/fold':>10} {'Speedup':>8} {'MB':>7} {'MB/s-sav':>9} {'Acc':>7}")
    print("-" * 120)

    for _, row in df.iterrows():
        print(f"{row['config']:<22} {row['n_folds']:>5} {row['cache_hits']:>5} "
              f"{row['cold_total_s']:>9.1f} {row['cold_per_fold_s']:>9.2f}s "
              f"{row['warm_total_s']:>8.1f} {row['warm_per_fold_s']:>9.4f}s "
              f"{row['speedup']:>7.1f}x {row['cache_size_mb']:>6.1f} "
              f"{row['mb_per_s_saved']:>8.4f} {row['accuracy']:>7.4f}")

    # Totals
    total_warm = df['warm_total_s'].sum()
    total_cold = df['cold_total_s'].sum()
    total_saved = df['time_saved_s'].sum()
    total_cache_mb = df['cache_size_mb'].sum()
    avg_speedup = total_cold / total_warm if total_warm > 0 else 0

    print("-" * 120)
    print(f"{'TOTAL':<22} {df['n_folds'].sum():>5} {df['cache_hits'].sum():>5} "
          f"{total_cold:>9.1f} {'':>10} "
          f"{total_warm:>8.1f} {'':>10} "
          f"{avg_speedup:>7.1f}x {total_cache_mb:>6.1f}")

    print(f"\n  Total cold training:   {total_cold:.0f}s ({total_cold/60:.1f} min = {total_cold/3600:.2f} hrs)")
    print(f"  Total warm loading:    {total_warm:.1f}s ({total_warm/60:.1f} min)")
    print(f"  Time saved:            {total_saved:.0f}s ({total_saved/60:.1f} min)")
    print(f"  Overall speedup:       {avg_speedup:.1f}x")
    print(f"  Cache size (XGB):      {total_cache_mb:.0f} MB ({total_cache_mb/1024:.2f} GB)")
    if total_saved > 0:
        print(f"  MB per second saved:   {total_cache_mb/total_saved:.4f}")
        print(f"  Cache efficiency:      {total_saved/total_cache_mb:.1f} seconds saved per MB")

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
