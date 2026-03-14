"""
Cache Viability Benchmark - All Models Cold+Warm with LOSO
==========================================================

Runs EACH model through a minimal LOSO cold+warm cycle and measures:
  - Cold training time (no cache)
  - Warm loading time (from cache)
  - Cache size on disk per model
  - Speedup ratio
  - MB/s-saved metric (cache viability indicator)

Output: A "future work" evaluation matrix as CSV + console table.

Usage (PowerShell, from project root):
  # Minimal: 30 subjects, first 3 folds only (fast, ~10min total)
  python model_tryouts/benchmark_cache_all_models.py --subjects 30 --max-folds 3

  # Full 128-subject LOSO (slow, but gold standard)
  python model_tryouts/benchmark_cache_all_models.py --subjects 128

  # Only boosting models
  python model_tryouts/benchmark_cache_all_models.py --subjects 30 --max-folds 3 --filter boosting

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
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Add parent directory so we can import thesis modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Add model_tryouts directory so we can import all_models
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fingerprint import generate_cache_key, LOSOFingerprint
from loso_cache import LOSOModelCache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Benchmark Result Container
# =============================================================================

@dataclass
class ModelCacheBenchmark:
    """Results for one model's cold+warm LOSO benchmark."""
    model_name: str
    category: str
    n_folds_run: int
    n_subjects: int
    # Cold run
    cold_total_time_s: float
    cold_per_fold_time_s: float
    # Warm run
    warm_total_time_s: float
    warm_per_fold_time_s: float
    # Cache metrics
    cache_size_mb: float
    cache_size_per_fold_mb: float
    # Derived
    speedup_ratio: float
    time_saved_s: float
    mb_per_second_saved: float  # KEY metric: lower = cache more viable
    # Accuracy (sanity check)
    cold_accuracy: float
    warm_accuracy: float
    accuracy_match: bool
    # Cache viable?
    cache_verdict: str
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


# =============================================================================
# Core Benchmark Logic
# =============================================================================

def benchmark_single_model(
    model_name: str,
    model_info: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    cache_base_dir: Path,
    max_folds: int = 0,
) -> ModelCacheBenchmark:
    """
    Run cold + warm LOSO for a single model and measure cache viability.

    Args:
        model_name: Name of the model
        model_info: Dict with 'model', 'needs_scaling', 'category'
        X, y, subject_ids: Features, labels, subject IDs
        cache_base_dir: Base directory for model caches
        max_folds: Max number of LOSO folds to run (0 = all)

    Returns:
        ModelCacheBenchmark with all metrics
    """
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score

    category = model_info.get('category', 'unknown')
    cache_dir = cache_base_dir / f"bench_{model_name}"

    # Clean any previous cache for this model
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    logo = LeaveOneGroupOut()
    unique_subjects = np.unique(subject_ids)
    n_total_folds = len(unique_subjects)
    n_folds = min(max_folds, n_total_folds) if max_folds > 0 else n_total_folds

    # Get model params for fingerprinting
    model_obj = model_info['model']
    if hasattr(model_obj, 'get_params'):
        model_params = model_obj.get_params()
    else:
        model_params = {}

    # ── COLD RUN: Train + Cache ──
    cache = LOSOModelCache(
        cache_dir=str(cache_dir),
        enable_registry=True,
        estimated_training_time=0.0,
        min_free_space_gb=1.0,
    )

    cold_preds_all = []
    cold_true_all = []
    cold_start = time.time()

    fold_iter = logo.split(X, y, subject_ids)
    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(fold_iter, total=n_folds, desc=f"  COLD {model_name}", unit="fold", leave=False)
    ):
        if fold_idx >= n_folds:
            break

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        held_out = str(subject_ids[test_idx[0]])

        # Scale if needed
        scaler = None
        if model_info.get('needs_scaling', False):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Generate fingerprint
        fingerprint = LOSOFingerprint.generate(
            random_seed=42,
            model_config={'name': model_name, 'params': model_params},
            feature_config={'base': X.shape[1], 'corr': None, 'top_k': None},
            held_out_subject=held_out,
        )

        # Train fresh (cold)
        fold_start = time.time()
        model = clone(model_obj) if hasattr(model_obj, 'get_params') else model_obj
        model.fit(X_train, y_train)
        fold_train_time = time.time() - fold_start

        # Cache the model
        cache.put(
            fingerprint=fingerprint,
            held_out_subject=held_out,
            model=model,
            model_type=model_name,
            training_time=fold_train_time,
            record_metrics=False,
        )

        y_pred = model.predict(X_test)
        cold_preds_all.extend(y_pred)
        cold_true_all.extend(y_test)

    cold_total = time.time() - cold_start
    cold_accuracy = accuracy_score(cold_true_all, cold_preds_all)

    # Measure cache size
    cache_size_bytes = sum(
        f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
    )
    cache_size_mb = cache_size_bytes / (1024 * 1024)

    # ── WARM RUN: Load from cache ──
    cache_warm = LOSOModelCache(
        cache_dir=str(cache_dir),
        enable_registry=True,
        estimated_training_time=0.0,
        min_free_space_gb=1.0,
    )

    warm_preds_all = []
    warm_true_all = []
    warm_start = time.time()

    fold_iter = logo.split(X, y, subject_ids)
    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(fold_iter, total=n_folds, desc=f"  WARM {model_name}", unit="fold", leave=False)
    ):
        if fold_idx >= n_folds:
            break

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        held_out = str(subject_ids[test_idx[0]])

        # Scale if needed (must match cold run)
        if model_info.get('needs_scaling', False):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Generate same fingerprint
        fingerprint = LOSOFingerprint.generate(
            random_seed=42,
            model_config={'name': model_name, 'params': model_params},
            feature_config={'base': X.shape[1], 'corr': None, 'top_k': None},
            held_out_subject=held_out,
        )

        # Try cache (should hit)
        cached_model = cache_warm.get(
            fingerprint=fingerprint,
            held_out_subject=held_out,
            model_type=model_name,
            record_metrics=True,
        )

        if cached_model is not None:
            model = cached_model
        else:
            # Cache miss (shouldn't happen) - train fresh
            logger.warning(f"Unexpected cache miss for {model_name} fold {fold_idx}")
            model = clone(model_obj) if hasattr(model_obj, 'get_params') else model_obj
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        warm_preds_all.extend(y_pred)
        warm_true_all.extend(y_test)

    warm_total = time.time() - warm_start
    warm_accuracy = accuracy_score(warm_true_all, warm_preds_all)

    # ── Compute Metrics ──
    time_saved = cold_total - warm_total
    speedup = cold_total / warm_total if warm_total > 0 else float('inf')
    mb_per_s_saved = cache_size_mb / time_saved if time_saved > 0 else float('inf')

    # Verdict
    if mb_per_s_saved < 0.5:
        verdict = "VIABLE"
    elif mb_per_s_saved < 2.0:
        verdict = "BORDERLINE"
    elif cold_total / n_folds < 1.0:
        verdict = "TOO_FAST"
    else:
        verdict = "NOT_VIABLE"

    # ── Cleanup cache ──
    shutil.rmtree(cache_dir, ignore_errors=True)

    return ModelCacheBenchmark(
        model_name=model_name,
        category=category,
        n_folds_run=n_folds,
        n_subjects=len(unique_subjects),
        cold_total_time_s=round(cold_total, 2),
        cold_per_fold_time_s=round(cold_total / n_folds, 3),
        warm_total_time_s=round(warm_total, 2),
        warm_per_fold_time_s=round(warm_total / n_folds, 3),
        cache_size_mb=round(cache_size_mb, 2),
        cache_size_per_fold_mb=round(cache_size_mb / n_folds, 3),
        speedup_ratio=round(speedup, 2),
        time_saved_s=round(time_saved, 2),
        mb_per_second_saved=round(mb_per_s_saved, 4),
        cold_accuracy=round(cold_accuracy, 4),
        warm_accuracy=round(warm_accuracy, 4),
        accuracy_match=abs(cold_accuracy - warm_accuracy) < 1e-6,
        cache_verdict=verdict,
    )


# =============================================================================
# Main Runner
# =============================================================================

def run_cache_benchmark(
    cache_dir_path: str = None,
    max_subjects: int = None,
    max_folds: int = 3,
    model_filter: str = None,
    output_dir: str = None,
) -> pd.DataFrame:
    """
    Run the full cache viability benchmark across all models.

    Args:
        cache_dir_path: Path to thesis feature cache
        max_subjects: Limit subjects to load
        max_folds: Max LOSO folds per model (default 3 for speed)
        model_filter: Only run models matching this category filter
        output_dir: Where to save results
    """
    # Import model definitions from all_models
    from all_models import load_features_from_thesis_cache, get_classical_models

    # Load features
    logger.info("Loading features from thesis cache...")
    X, y, subject_ids = load_features_from_thesis_cache(
        cache_dir=cache_dir_path,
        max_subjects=max_subjects
    )
    n_subjects = len(np.unique(subject_ids))
    logger.info(f"Loaded {len(y)} epochs from {n_subjects} subjects")

    # Get all models
    models = get_classical_models()

    # Filter if requested
    if model_filter:
        models = {
            name: info for name, info in models.items()
            if model_filter.lower() in info.get('category', '').lower()
            or model_filter.lower() in name.lower()
        }
        logger.info(f"Filtered to {len(models)} models matching '{model_filter}'")

    if not models:
        logger.error("No models to benchmark!")
        return pd.DataFrame()

    # Setup output
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_base = output_dir / "cache_benchmark_temp"
    cache_base.mkdir(parents=True, exist_ok=True)

    # Print plan
    print("\n" + "=" * 70)
    print("CACHE VIABILITY BENCHMARK")
    print("=" * 70)
    print(f"  Subjects: {n_subjects}")
    print(f"  Max folds per model: {max_folds if max_folds > 0 else 'ALL'}")
    print(f"  Models to benchmark: {len(models)}")
    print(f"  Models: {', '.join(models.keys())}")
    print("=" * 70 + "\n")

    # Run benchmarks
    results: List[ModelCacheBenchmark] = []

    for i, (name, info) in enumerate(models.items(), 1):
        print(f"\n[{i}/{len(models)}] Benchmarking: {name} ({info.get('category', '?')})")
        print("-" * 50)

        try:
            result = benchmark_single_model(
                model_name=name,
                model_info=info,
                X=X.values if hasattr(X, 'values') else X,
                y=y,
                subject_ids=subject_ids,
                cache_base_dir=cache_base,
                max_folds=max_folds,
            )
            results.append(result)

            # Print immediate result
            print(f"  Cold: {result.cold_total_time_s:.1f}s "
                  f"({result.cold_per_fold_time_s:.2f}s/fold)")
            print(f"  Warm: {result.warm_total_time_s:.1f}s "
                  f"({result.warm_per_fold_time_s:.2f}s/fold)")
            print(f"  Cache: {result.cache_size_mb:.1f} MB "
                  f"({result.cache_size_per_fold_mb:.1f} MB/fold)")
            print(f"  Speedup: {result.speedup_ratio:.1f}x | "
                  f"MB/s-saved: {result.mb_per_second_saved:.4f}")
            print(f"  Accuracy match: {result.accuracy_match} "
                  f"(cold={result.cold_accuracy:.4f}, warm={result.warm_accuracy:.4f})")
            print(f"  Verdict: {result.cache_verdict}")

        except Exception as e:
            logger.error(f"Failed to benchmark {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append(ModelCacheBenchmark(
                model_name=name,
                category=info.get('category', '?'),
                n_folds_run=0, n_subjects=n_subjects,
                cold_total_time_s=0, cold_per_fold_time_s=0,
                warm_total_time_s=0, warm_per_fold_time_s=0,
                cache_size_mb=0, cache_size_per_fold_mb=0,
                speedup_ratio=0, time_saved_s=0,
                mb_per_second_saved=float('inf'),
                cold_accuracy=0, warm_accuracy=0,
                accuracy_match=False,
                cache_verdict="ERROR",
                error=str(e),
            ))

    # Cleanup temp cache dir
    shutil.rmtree(cache_base, ignore_errors=True)

    # ── Build Results Table ──
    df = pd.DataFrame([r.to_dict() for r in results])
    df = df.sort_values('mb_per_second_saved', ascending=True)

    # Save CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"cache_viability_{n_subjects}subj_{max_folds}folds_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    # Save JSON (detailed)
    json_path = output_dir / f"cache_viability_{n_subjects}subj_{max_folds}folds_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump([r.to_dict() for r in results], f, indent=2, default=str)

    # ── Print Summary Table ──
    print("\n\n" + "=" * 110)
    print("CACHE VIABILITY MATRIX - Future Work Evaluation")
    print("=" * 110)
    print(f"\n{'Model':<22} {'Category':<12} {'Cold/fold':>10} {'Warm/fold':>10} "
          f"{'Speedup':>8} {'Cache/fold':>10} {'MB/s-saved':>11} {'Verdict':<12} {'Acc':>7}")
    print("-" * 110)

    for _, row in df.iterrows():
        if row.get('error'):
            print(f"{row['model_name']:<22} {'ERROR':<12} {row.get('error', '')}")
            continue
        print(
            f"{row['model_name']:<22} {row['category']:<12} "
            f"{row['cold_per_fold_time_s']:>9.3f}s {row['warm_per_fold_time_s']:>9.3f}s "
            f"{row['speedup_ratio']:>7.1f}x {row['cache_size_per_fold_mb']:>9.2f}MB "
            f"{row['mb_per_second_saved']:>10.4f} {row['cache_verdict']:<12} "
            f"{row['cold_accuracy']:>6.4f}"
        )

    print("-" * 110)
    print("\nVerdict legend:")
    print("  VIABLE     = MB/s-saved < 0.5  (cache saves significant time per MB)")
    print("  BORDERLINE = MB/s-saved 0.5-2.0 (marginal benefit)")
    print("  NOT_VIABLE = MB/s-saved > 2.0  (cache too large for time saved)")
    print("  TOO_FAST   = Training < 1s/fold (caching adds complexity for no benefit)")

    # ── Category Summary ──
    print("\n\nCATEGORY SUMMARY:")
    print("-" * 60)

    viable = df[df['cache_verdict'] == 'VIABLE']
    borderline = df[df['cache_verdict'] == 'BORDERLINE']
    not_viable = df[df['cache_verdict'].isin(['NOT_VIABLE', 'TOO_FAST'])]

    if not viable.empty:
        print(f"\n  CACHE VIABLE ({len(viable)} models):")
        for _, r in viable.iterrows():
            print(f"    - {r['model_name']}: {r['speedup_ratio']:.1f}x speedup, "
                  f"{r['cache_size_per_fold_mb']:.1f} MB/fold")

    if not borderline.empty:
        print(f"\n  BORDERLINE ({len(borderline)} models):")
        for _, r in borderline.iterrows():
            print(f"    - {r['model_name']}: {r['speedup_ratio']:.1f}x speedup, "
                  f"{r['cache_size_per_fold_mb']:.1f} MB/fold")

    if not not_viable.empty:
        print(f"\n  NOT VIABLE / TOO FAST ({len(not_viable)} models):")
        for _, r in not_viable.iterrows():
            print(f"    - {r['model_name']}: {r['cache_verdict']}, "
                  f"{r['cache_size_per_fold_mb']:.1f} MB/fold")

    # ── 128-subject LOSO projection ──
    if n_subjects < 128:
        print("\n\n128-SUBJECT LOSO PROJECTION (estimated):")
        print("-" * 80)
        print(f"{'Model':<22} {'Est. cold time':>14} {'Est. cache size':>15} {'Est. time saved':>15}")
        print("-" * 80)
        for _, row in df.iterrows():
            if row.get('error'):
                continue
            # Scale: cold time scales roughly with n_subjects, cache size scales linearly
            scale = 128 / n_subjects
            # Training time scales with (n_subjects)^exponent per fold, and folds = n_subjects
            # For most models, exponent ~1.0-1.2, so total ≈ scale^2
            est_cold_total = row['cold_per_fold_time_s'] * 128 * (scale ** 0.5)
            est_cache_mb = row['cache_size_per_fold_mb'] * 128
            est_time_saved = est_cold_total - (row['warm_per_fold_time_s'] * 128)

            def fmt_time(s):
                if s < 60: return f"{s:.0f}s"
                if s < 3600: return f"{s/60:.1f}min"
                return f"{s/3600:.1f}hrs"

            print(f"{row['model_name']:<22} {fmt_time(est_cold_total):>14} "
                  f"{est_cache_mb:>12.0f} MB {fmt_time(est_time_saved):>14}")

    print(f"\n\nResults saved to:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")

    return df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark cache viability for ALL models with LOSO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark: 30 subjects, 3 folds per model (~10 min)
  python benchmark_cache_all_models.py --subjects 30 --max-folds 3

  # Only boosting models
  python benchmark_cache_all_models.py --subjects 30 --max-folds 3 --filter boosting

  # Full 128-subject LOSO (slow but complete)
  python benchmark_cache_all_models.py --subjects 128 --max-folds 0

  # Quick sanity check with 5 subjects, 2 folds
  python benchmark_cache_all_models.py --subjects 5 --max-folds 2
        """
    )

    parser.add_argument('--subjects', type=int, default=30,
                        help='Number of subjects to load (default: 30)')
    parser.add_argument('--max-folds', type=int, default=3,
                        help='Max LOSO folds per model (0=all, default: 3)')
    parser.add_argument('--filter', type=str, default=None,
                        help='Filter models by name or category (e.g. "boosting", "xgboost")')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Path to thesis feature cache directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')

    args = parser.parse_args()

    run_cache_benchmark(
        cache_dir_path=args.cache_dir,
        max_subjects=args.subjects,
        max_folds=args.max_folds,
        model_filter=args.filter,
        output_dir=args.output_dir,
    )
