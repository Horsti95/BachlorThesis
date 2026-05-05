"""
XGBoost Cache Evaluation - Cold vs Warm LOSO Analysis
=====================================================

Reads cold training times from the cache registry, loads cached models
by their stored fingerprints, runs feature selection to match the exact
feature subsets used during training, and evaluates warm predictions.

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
from typing import Dict, Any, List
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
from feature_selection import FeatureSelectionConfig, FeatureSelectionPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_xgb_registry_grouped(cache_dir: str) -> Dict[str, List[Dict]]:
    """
    Load registry and group XGB entries into configs (batches of 128).
    Returns dict of config_name -> list of entries.
    """
    registry_path = Path(cache_dir) / "cache_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"No registry at {registry_path}")

    with open(registry_path, 'r') as f:
        registry = json.load(f)

    # Collect all XGB entries sorted by creation time
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

    xgb_entries.sort(key=lambda x: x['created_at'])

    # Group into batches of 128 (each config = 128 LOSO folds)
    configs = {}
    n_per_config = 128
    n_complete = len(xgb_entries) // n_per_config
    remainder = len(xgb_entries) % n_per_config

    for i in range(n_complete):
        start = i * n_per_config
        configs[f"config_{i+1}"] = xgb_entries[start:start + n_per_config]

    if remainder > 0:
        start = n_complete * n_per_config
        configs[f"config_{n_complete+1}_partial"] = xgb_entries[start:]

    return configs


def run_warm_evaluation(
    cache_dir: str,
    feature_cache_dir: str = None,
    max_subjects: int = None,
) -> pd.DataFrame:
    """
    Run warm evaluation using fingerprints from the registry.
    Applies feature selection to match the training feature subsets.
    """
    from all_models import load_features_from_thesis_cache

    print("=" * 70)
    print("XGBoost LOSO Cache Evaluation: Cold vs Warm")
    print("=" * 70)

    # ── Load registry ──
    configs = load_xgb_registry_grouped(cache_dir)
    if not configs:
        print("No XGB entries found in registry!")
        return pd.DataFrame()

    total_entries = sum(len(v) for v in configs.values())
    total_cold_time = sum(e['training_time_s'] for b in configs.values() for e in b)
    total_size_bytes = sum(e['file_size_bytes'] for b in configs.values() for e in b)

    print(f"\nCache Registry Analysis:")
    print(f"  XGB cached folds:  {total_entries}")
    print(f"  XGB configs:       {len(configs)}")
    print(f"  Total cold time:   {total_cold_time:.1f}s ({total_cold_time/3600:.2f} hrs)")
    print(f"  Avg cold/fold:     {total_cold_time/total_entries:.2f}s")
    print(f"  Total cache size:  {total_size_bytes/(1024*1024):.1f} MB")
    print(f"  Avg file size:     {total_size_bytes/total_entries/(1024*1024):.2f} MB")

    # ── Load features ──
    print(f"\nLoading features...")
    X, y, subject_ids = load_features_from_thesis_cache(
        cache_dir=feature_cache_dir,
        max_subjects=max_subjects,
    )
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    n_subjects = len(np.unique(subject_ids))
    print(f"  Loaded {len(y)} epochs from {n_subjects} subjects")

    # ── Define the 9 thesis configs (3 corr × 3 top_k) ──
    # These must match what run_training.py used
    thesis_configs = [
        {'corr': 0.75, 'top_k': 30},
        {'corr': 0.75, 'top_k': 50},
        {'corr': 0.75, 'top_k': None},
        {'corr': 0.90, 'top_k': 30},
        {'corr': 0.90, 'top_k': 50},
        {'corr': 0.90, 'top_k': None},
        {'corr': None,  'top_k': 30},
        {'corr': None,  'top_k': 50},
        {'corr': None,  'top_k': None},
    ]

    # Pre-compute global feature selection for each config
    # (matches the training pipeline's 'global' scope)
    print(f"\nPre-computing feature selection for {len(thesis_configs)} configs...")
    config_features = {}
    for i, tc in enumerate(thesis_configs):
        fs_config = FeatureSelectionConfig(
            correlation_threshold=tc['corr'],
            top_k_features=tc['top_k'],
            selection_method='anova',
            scope='global',
            random_state=42,
        )
        fs_pipeline = FeatureSelectionPipeline(fs_config)
        fs_pipeline.fit(X_df, y)
        selected = fs_pipeline.get_selected_features()
        n_feat = len(selected)
        config_features[i] = selected
        label = f"corr={tc['corr']}, top_k={tc['top_k']}"
        print(f"  Config {i+1}: {label} -> {n_feat} features")

    # Build subject -> indices mapping
    subject_idx_map = defaultdict(list)
    for idx, sid in enumerate(subject_ids):
        subject_idx_map[str(sid)].append(idx)

    # ── Initialize cache ──
    cache = LOSOModelCache(
        cache_dir=cache_dir,
        enable_registry=True,
        estimated_training_time=total_cold_time / total_entries,
    )

    # ── Warm evaluation per config ──
    results = []
    config_names = sorted(configs.keys())

    for cfg_idx, cfg_name in enumerate(config_names):
        entries = configs[cfg_name]
        n_folds = len(entries)

        # Match to thesis config by index
        tc_idx = cfg_idx if cfg_idx < len(thesis_configs) else len(thesis_configs) - 1
        tc = thesis_configs[tc_idx]
        selected_features = config_features[tc_idx]
        label = f"corr={tc['corr']}, k={tc['top_k']} ({len(selected_features)}f)"

        print(f"\n[{cfg_idx+1}/{len(configs)}] {cfg_name}: {label} | {n_folds} folds")
        print("-" * 60)

        cfg_cold_time = sum(e['training_time_s'] for e in entries)
        cfg_cache_size = sum(e['file_size_bytes'] for e in entries) / (1024 * 1024)

        warm_preds = []
        warm_true = []
        cache_hits = 0
        cache_misses = 0
        load_times = []

        warm_start = time.time()

        for entry in tqdm(entries, desc=f"  Warm", unit="fold", leave=False):
            fp = entry['fingerprint']
            subject = entry['subject']

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

                if subject in subject_idx_map:
                    test_idx = subject_idx_map[subject]
                    # Select only the features this config used
                    X_test = X_df.iloc[test_idx][selected_features].values
                    y_test = y[test_idx]

                    y_pred = cached_model.predict(X_test)
                    warm_preds.extend(y_pred)
                    warm_true.extend(y_test)
            else:
                cache_misses += 1

        warm_total = time.time() - warm_start

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
            'config': label,
            'n_folds': n_folds,
            'n_features': len(selected_features),
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cold_total_s': round(cfg_cold_time, 1),
            'cold_per_fold_s': round(cfg_cold_time / n_folds, 2),
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

        hit_pct = f"{cache_hits}/{cache_hits+cache_misses}"
        print(f"  Hits: {hit_pct} | Warm: {warm_total:.1f}s ({avg_load:.3f}s/fold) | "
              f"Cold: {cfg_cold_time:.1f}s ({cfg_cold_time/n_folds:.2f}s/fold)")
        print(f"  Speedup: {speedup:.1f}x | Cache: {cfg_cache_size:.1f} MB | "
              f"MB/s-saved: {mb_per_s_saved:.4f}")
        if cache_hits > 0:
            print(f"  Acc: {acc:.4f} | Kappa: {kappa:.4f} | F1: {f1:.4f}")

    # ── Summary Table ──
    df = pd.DataFrame(results)

    print("\n\n" + "=" * 130)
    print("XGBoost LOSO Cache: Cold vs Warm Summary")
    print("=" * 130)
    print(f"\n{'Config':<30} {'#F':>3} {'Hits':>5} {'Cold(s)':>9} {'Cold/f':>8} "
          f"{'Warm(s)':>8} {'Warm/f':>8} {'Speed':>6} {'MB':>7} {'MB/s-sv':>8} {'Acc':>7} {'Kappa':>7}")
    print("-" * 130)

    for _, row in df.iterrows():
        print(f"{row['config']:<30} {row['n_features']:>3} {row['cache_hits']:>5} "
              f"{row['cold_total_s']:>9.1f} {row['cold_per_fold_s']:>7.2f}s "
              f"{row['warm_total_s']:>8.1f} {row['warm_per_fold_s']:>7.4f}s "
              f"{row['speedup']:>5.1f}x {row['cache_size_mb']:>6.1f} "
              f"{row['mb_per_s_saved']:>7.4f} {row['accuracy']:>7.4f} {row['kappa']:>7.4f}")

    # Totals
    total_warm = df['warm_total_s'].sum()
    total_cold = df['cold_total_s'].sum()
    total_saved = df['time_saved_s'].sum()
    total_cache_mb = df['cache_size_mb'].sum()
    avg_speedup = total_cold / total_warm if total_warm > 0 else 0

    print("-" * 130)
    print(f"{'TOTAL':<30} {'':>3} {df['cache_hits'].sum():>5} "
          f"{total_cold:>9.1f} {'':>8} "
          f"{total_warm:>8.1f} {'':>8} "
          f"{avg_speedup:>5.1f}x {total_cache_mb:>6.1f}")

    print(f"\n  Total cold training:   {total_cold:.0f}s ({total_cold/60:.1f} min = {total_cold/3600:.2f} hrs)")
    print(f"  Total warm loading:    {total_warm:.1f}s ({total_warm/60:.1f} min)")
    print(f"  Time saved:            {total_saved:.0f}s ({total_saved/60:.1f} min = {total_saved/3600:.2f} hrs)")
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
