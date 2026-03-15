"""
Random Forest Cache Evaluation - Cold+Warm per Config with Cleanup
==================================================================

Runs each RF config through the full LOSO pipeline:
  1. Cold run: Train 128 folds, cache each model
  2. Warm run: Load all 128 models from cache
  3. Measure cold time, warm time, cache size, accuracy
  4. DELETE cache before next config (prevents filling disk)

This gives the same quality data as the XGB evaluation but is safe
for RF's ~20 GB/config cache size.

Usage (PowerShell, from project root):
  # Full 9 configs (each ~20-40 min cold + seconds warm)
  python model_tryouts/evaluate_rf_cache.py

  # Single config for quick test
  python model_tryouts/evaluate_rf_cache.py --config 1

  # Custom cache location
  python model_tryouts/evaluate_rf_cache.py --cache-dir D:/temp/rf_cache

Author: Lennart Gorzel
Date: March 2026
"""

import sys
import os
import time
import json
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from fingerprint import LOSOFingerprint
from loso_cache import LOSOModelCache
from feature_selection import FeatureSelectionConfig, FeatureSelectionPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thesis RF params (must match models.py RandomForestModel defaults)
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced',
}

# Thesis grid: 3 corr × 3 top_k = 9 configs
THESIS_CONFIGS = [
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


def run_single_config_cold_warm(
    cfg_idx: int,
    cfg: Dict,
    X_df: pd.DataFrame,
    y: np.ndarray,
    subject_ids: np.ndarray,
    selected_features: List[str],
    cache_dir: Path,
) -> Dict[str, Any]:
    """
    Run cold + warm for ONE RF config, then delete cache.

    Returns dict with all metrics.
    """
    label = f"corr={cfg['corr']}, k={cfg['top_k']} ({len(selected_features)}f)"
    n_features = len(selected_features)

    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    logo = LeaveOneGroupOut()

    # Build subject -> test indices
    subject_idx_map = defaultdict(list)
    for idx, sid in enumerate(subject_ids):
        subject_idx_map[str(sid)].append(idx)

    # Clean cache dir for this config
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Feature-selected data
    X_sel = X_df[selected_features]

    # ── COLD RUN: Train + Cache ──
    cache = LOSOModelCache(
        cache_dir=str(cache_dir),
        enable_registry=True,
        estimated_training_time=0.0,
        min_free_space_gb=2.0,
    )

    cold_preds = []
    cold_true = []
    fold_train_times = []

    print(f"\n  COLD RUN ({label})...")
    cold_start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(logo.split(X_sel.values, y, subject_ids),
             total=n_subjects, desc=f"  Cold RF", unit="fold", leave=False)
    ):
        held_out = str(subject_ids[test_idx[0]])
        X_train = X_sel.values[train_idx]
        y_train = y[train_idx]
        X_test = X_sel.values[test_idx]
        y_test = y[test_idx]

        # Generate fingerprint matching thesis pipeline
        fingerprint = LOSOFingerprint.generate(
            random_seed=42,
            code_version='1.0.0',
            model_name='random_forest',
            model_params=RF_PARAMS,
            feature_config={
                'base': X_df.shape[1],
                'corr': cfg['corr'],
                'top_k': cfg['top_k'],
                'n_selected': n_features,
                'selected_features': selected_features,
            },
            held_out_subject=held_out,
        )

        # Train
        fold_start = time.time()
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train, y_train)
        fold_time = time.time() - fold_start
        fold_train_times.append(fold_time)

        # Cache
        cache.put(
            fingerprint=fingerprint,
            held_out_subject=held_out,
            model=model,
            model_type='random_forest',
            training_time=fold_time,
            record_metrics=False,
        )

        # Predict
        y_pred = model.predict(X_test)
        cold_preds.extend(y_pred)
        cold_true.extend(y_test)

    cold_total = time.time() - cold_start
    cold_acc = accuracy_score(cold_true, cold_preds)
    cold_kappa = cohen_kappa_score(cold_true, cold_preds)
    cold_f1 = f1_score(cold_true, cold_preds, average='macro')

    # Measure cache size
    cache_size_bytes = sum(
        f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
    )
    cache_size_mb = cache_size_bytes / (1024 * 1024)
    cache_size_gb = cache_size_mb / 1024

    print(f"  Cold done: {cold_total:.1f}s ({cold_total/n_subjects:.2f}s/fold) | "
          f"Cache: {cache_size_gb:.1f} GB | Acc: {cold_acc:.4f}")

    # ── WARM RUN: Load from cache ──
    cache_warm = LOSOModelCache(
        cache_dir=str(cache_dir),
        enable_registry=True,
        estimated_training_time=np.mean(fold_train_times),
    )

    warm_preds = []
    warm_true = []
    cache_hits = 0
    cache_misses = 0
    load_times = []

    print(f"  WARM RUN ({label})...")
    warm_start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(logo.split(X_sel.values, y, subject_ids),
             total=n_subjects, desc=f"  Warm RF", unit="fold", leave=False)
    ):
        held_out = str(subject_ids[test_idx[0]])
        X_test = X_sel.values[test_idx]
        y_test = y[test_idx]

        fingerprint = LOSOFingerprint.generate(
            random_seed=42,
            code_version='1.0.0',
            model_name='random_forest',
            model_params=RF_PARAMS,
            feature_config={
                'base': X_df.shape[1],
                'corr': cfg['corr'],
                'top_k': cfg['top_k'],
                'n_selected': n_features,
                'selected_features': selected_features,
            },
            held_out_subject=held_out,
        )

        t0 = time.time()
        cached_model = cache_warm.get(
            fingerprint=fingerprint,
            held_out_subject=held_out,
            model_type='random_forest',
            record_metrics=True,
        )
        load_time = time.time() - t0

        if cached_model is not None:
            cache_hits += 1
            load_times.append(load_time)
            y_pred = cached_model.predict(X_test)
            warm_preds.extend(y_pred)
            warm_true.extend(y_test)
        else:
            cache_misses += 1
            logger.warning(f"  Cache miss for fold {fold_idx} (subject {held_out})")

    warm_total = time.time() - warm_start
    warm_acc = accuracy_score(warm_true, warm_preds) if warm_preds else 0
    warm_kappa = cohen_kappa_score(warm_true, warm_preds) if warm_preds else 0
    warm_f1 = f1_score(warm_true, warm_preds, average='macro') if warm_preds else 0

    avg_load = np.mean(load_times) if load_times else 0
    time_saved = cold_total - warm_total
    speedup = cold_total / warm_total if warm_total > 0 else float('inf')
    mb_per_s_saved = cache_size_mb / time_saved if time_saved > 0 else float('inf')
    accuracy_match = abs(cold_acc - warm_acc) < 1e-6

    print(f"  Warm done: {warm_total:.1f}s ({avg_load:.3f}s/fold) | "
          f"Hits: {cache_hits}/{cache_hits+cache_misses} | "
          f"Speedup: {speedup:.1f}x | Acc: {warm_acc:.4f}")
    print(f"  Accuracy match: {accuracy_match} | MB/s-saved: {mb_per_s_saved:.4f}")

    # ── DELETE CACHE ──
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"  Cache deleted ({cache_size_gb:.1f} GB freed)")

    return {
        'config': label,
        'corr': cfg['corr'],
        'top_k': cfg['top_k'],
        'n_features': n_features,
        'n_folds': n_subjects,
        'cache_hits': cache_hits,
        'cache_misses': cache_misses,
        'cold_total_s': round(cold_total, 1),
        'cold_per_fold_s': round(cold_total / n_subjects, 2),
        'warm_total_s': round(warm_total, 2),
        'warm_per_fold_s': round(avg_load, 4),
        'speedup': round(speedup, 1),
        'time_saved_s': round(time_saved, 1),
        'cache_size_mb': round(cache_size_mb, 1),
        'cache_size_gb': round(cache_size_gb, 2),
        'mb_per_s_saved': round(mb_per_s_saved, 4),
        'cold_accuracy': round(cold_acc, 4),
        'warm_accuracy': round(warm_acc, 4),
        'accuracy_match': accuracy_match,
        'cold_kappa': round(cold_kappa, 4),
        'warm_kappa': round(warm_kappa, 4),
        'cold_f1': round(cold_f1, 4),
        'warm_f1': round(warm_f1, 4),
    }


def run_rf_evaluation(
    feature_cache_dir: str = None,
    cache_dir: str = None,
    config_indices: List[int] = None,
    max_subjects: int = None,
    output_dir: str = None,
) -> pd.DataFrame:
    """
    Run cold+warm evaluation for RF, one config at a time with cleanup.
    """
    from all_models import load_features_from_thesis_cache

    print("=" * 70)
    print("Random Forest LOSO Cache Evaluation: Cold+Warm per Config")
    print("=" * 70)

    # ── Load features ──
    print("\nLoading features...")
    X, y, subject_ids = load_features_from_thesis_cache(
        cache_dir=feature_cache_dir,
        max_subjects=max_subjects,
    )
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    n_subjects = len(np.unique(subject_ids))
    print(f"  Loaded {len(y)} epochs from {n_subjects} subjects")

    # ── Pre-compute feature selection for each config ──
    print(f"\nPre-computing feature selection...")
    config_features = {}
    for i, tc in enumerate(THESIS_CONFIGS):
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
        config_features[i] = selected
        label = f"corr={tc['corr']}, top_k={tc['top_k']}"
        print(f"  Config {i+1}: {label} -> {len(selected)} features")

    # Which configs to run?
    if config_indices:
        indices = [i - 1 for i in config_indices]  # 1-based to 0-based
    else:
        indices = list(range(len(THESIS_CONFIGS)))

    # Cache dir
    if cache_dir is None:
        cache_dir = Path(_PROJECT_DIR) / "results" / "rf_cache_benchmark"
    cache_path = Path(cache_dir)

    # Output dir
    if output_dir is None:
        output_dir = Path(_PROJECT_DIR) / "results"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Run each config ──
    results = []
    total_start = time.time()

    for run_idx, cfg_idx in enumerate(indices):
        cfg = THESIS_CONFIGS[cfg_idx]
        selected = config_features[cfg_idx]

        print(f"\n{'='*70}")
        print(f"[{run_idx+1}/{len(indices)}] Config {cfg_idx+1}: "
              f"corr={cfg['corr']}, top_k={cfg['top_k']} ({len(selected)} features)")
        print(f"{'='*70}")

        # Check free space
        try:
            usage = shutil.disk_usage(cache_path.parent)
            free_gb = usage.free / (1024**3)
            print(f"  Free disk space: {free_gb:.1f} GB")
            if free_gb < 25:
                print(f"  WARNING: Low disk space! RF config needs ~20 GB. Skipping.")
                continue
        except OSError:
            pass

        result = run_single_config_cold_warm(
            cfg_idx=cfg_idx,
            cfg=cfg,
            X_df=X_df,
            y=y,
            subject_ids=subject_ids,
            selected_features=selected,
            cache_dir=cache_path,
        )
        results.append(result)

        # Save intermediate results after each config
        df_partial = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        partial_path = output_path / f"rf_cache_evaluation_partial_{timestamp}.csv"
        df_partial.to_csv(partial_path, index=False)

    total_time = time.time() - total_start

    # ── Final Summary Table ──
    df = pd.DataFrame(results)

    print("\n\n" + "=" * 140)
    print("Random Forest LOSO Cache: Cold vs Warm Summary")
    print("=" * 140)
    print(f"\n{'Config':<30} {'#F':>3} {'Hits':>5} {'Cold(s)':>9} {'Cold/f':>8} "
          f"{'Warm(s)':>9} {'Warm/f':>8} {'Speed':>6} {'Cache GB':>9} {'MB/s-sv':>8} "
          f"{'Acc':>7} {'Match':>5}")
    print("-" * 140)

    for _, row in df.iterrows():
        print(f"{row['config']:<30} {row['n_features']:>3} {row['cache_hits']:>5} "
              f"{row['cold_total_s']:>9.1f} {row['cold_per_fold_s']:>7.2f}s "
              f"{row['warm_total_s']:>9.1f} {row['warm_per_fold_s']:>7.4f}s "
              f"{row['speedup']:>5.1f}x {row['cache_size_gb']:>8.2f} "
              f"{row['mb_per_s_saved']:>7.4f} {row['cold_accuracy']:>7.4f} "
              f"{'Y' if row['accuracy_match'] else 'N':>5}")

    # Totals
    if not df.empty:
        total_warm = df['warm_total_s'].sum()
        total_cold = df['cold_total_s'].sum()
        total_saved = df['time_saved_s'].sum()
        avg_cache_gb = df['cache_size_gb'].mean()
        avg_speedup = total_cold / total_warm if total_warm > 0 else 0

        print("-" * 140)
        print(f"\n  Configs evaluated:     {len(df)}")
        print(f"  Total cold training:   {total_cold:.0f}s ({total_cold/60:.1f} min = {total_cold/3600:.2f} hrs)")
        print(f"  Total warm loading:    {total_warm:.1f}s ({total_warm/60:.1f} min)")
        print(f"  Time saved:            {total_saved:.0f}s ({total_saved/60:.1f} min = {total_saved/3600:.2f} hrs)")
        print(f"  Overall speedup:       {avg_speedup:.1f}x")
        print(f"  Avg cache size/config: {avg_cache_gb:.2f} GB")
        if total_saved > 0:
            total_cache_mb = df['cache_size_mb'].sum()
            print(f"  MB per second saved:   {total_cache_mb/total_saved:.4f}")
            print(f"  Cache efficiency:      {total_saved/total_cache_mb:.1f} seconds saved per MB")

        # Compare with XGB
        print(f"\n  COMPARISON WITH XGBoost (from previous evaluation):")
        print(f"  {'Metric':<25} {'XGBoost':>12} {'Random Forest':>15}")
        print(f"  {'-'*55}")
        rf_cache_str = f"~{avg_cache_gb:.1f} GB"
        rf_speedup_str = f"{avg_speedup:.0f}x"
        rf_mbs_str = f"{df['mb_per_s_saved'].mean():.2f}"
        rf_mbs_val = df['mb_per_s_saved'].mean()
        rf_verdict = 'NOT VIABLE' if rf_mbs_val > 2.0 else 'BORDERLINE' if rf_mbs_val > 0.5 else 'VIABLE'
        print(f"  {'Cache size/config':<25} {'~184 MB':>12} {rf_cache_str:>15}")
        print(f"  {'Avg speedup':<25} {'210x':>12} {rf_speedup_str:>15}")
        print(f"  {'MB/s-saved (lower=better)':<25} {'0.12':>12} {rf_mbs_str:>15}")
        print(f"  {'Cache verdict':<25} {'VIABLE':>12} {rf_verdict:>15}")

    # Save final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"rf_cache_evaluation_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")
    print(f"  Total wall time: {total_time:.0f}s ({total_time/60:.1f} min)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RF cache evaluation: cold+warm per config with cleanup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All 9 configs (slow but complete, ~3-6 hours)
  python evaluate_rf_cache.py

  # Single config for quick test (~20-40 min)
  python evaluate_rf_cache.py --config 1

  # Configs 1, 5, 9 only (representative subset)
  python evaluate_rf_cache.py --config 1 5 9

  # With custom cache location (e.g. faster SSD)
  python evaluate_rf_cache.py --cache-dir D:/temp/rf_cache
        """
    )
    parser.add_argument('--feature-cache', type=str, default=None,
                        help='Path to thesis feature cache directory')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Temp cache directory for RF models (deleted after each config)')
    parser.add_argument('--config', type=int, nargs='+', default=None,
                        help='Config indices to run (1-9). Default: all')
    parser.add_argument('--subjects', type=int, default=None,
                        help='Limit number of subjects')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    args = parser.parse_args()

    run_rf_evaluation(
        feature_cache_dir=args.feature_cache,
        cache_dir=args.cache_dir,
        config_indices=args.config,
        max_subjects=args.subjects,
        output_dir=args.output_dir,
    )
