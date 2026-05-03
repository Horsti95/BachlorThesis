#!/usr/bin/env python3
"""
Comprehensive LOSO Cache Test - 3 Subjects, All Models, Multiple Correlations

Tests caching behavior for:
- 3 subjects (LOSO = 3 folds)
- 3 models: XGBoost, Random Forest, FNN
- 2 correlation cutoffs: 0.75, 0.90
- 2 top-k values: 30, all features

Total configurations: 3 models × 2 corr × 2 top_k = 12 configs
Total runs: 12 configs × 3 folds = 36 training runs

Run twice to verify:
- First run: All cache MISSES (train fresh)
- Second run: All cache HITS (load from cache)
"""

import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path
import tempfile

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from training import TrainingPipeline, TrainingConfig
from cross_validation import LOSOCrossValidator
from feature_selection import FeatureSelectionConfig
from output_formatter import TrainingOutputFormatter, Verbosity
from loso_cache import LOSOModelCache

# Check model availability
try:
    from models import FNNModel
    FNN_AVAILABLE = True
except ImportError:
    FNN_AVAILABLE = False
    logger.warning("⚠ PyTorch not available - FNN tests will be skipped")


def generate_synthetic_data(n_subjects=3, n_epochs_per_subject=200, n_features=149):
    """Generate synthetic EEG-like data for testing."""
    np.random.seed(42)

    all_features = []
    all_labels = []
    all_subject_ids = []

    for subject_id in range(1, n_subjects + 1):
        # Generate random features
        features = np.random.randn(n_epochs_per_subject, n_features)

        # Generate labels (5 sleep stages: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
        # Imbalanced like real sleep data
        labels = np.random.choice([0, 1, 2, 3, 4], size=n_epochs_per_subject,
                                  p=[0.15, 0.05, 0.45, 0.20, 0.15])

        subject_ids = [f"Subject_{subject_id}"] * n_epochs_per_subject

        all_features.append(features)
        all_labels.append(labels)
        all_subject_ids.extend(subject_ids)

    # Combine all data
    features_array = np.vstack(all_features)
    labels_array = np.concatenate(all_labels)

    # Convert to DataFrame with feature names
    feature_names = [f"feat_{i}" for i in range(n_features)]
    features_df = pd.DataFrame(features_array, columns=feature_names)

    subject_ids_array = np.array(all_subject_ids)

    return features_df, labels_array, subject_ids_array


def create_test_configs(include_fnn=True):
    """Create test configuration grid."""

    # Models to test
    models = ['xgboost', 'random_forest']
    if include_fnn and FNN_AVAILABLE:
        models.append('fnn')

    # Configuration grid
    correlation_thresholds = [0.75, 0.90]
    top_k_features = [30, None]  # 30 features or all

    configs = []

    for model in models:
        for corr in correlation_thresholds:
            for top_k in top_k_features:

                # Model-specific params
                if model == 'xgboost':
                    model_params = {'n_estimators': 50, 'max_depth': 4}
                elif model == 'random_forest':
                    model_params = {'n_estimators': 50, 'max_depth': 10}
                else:  # fnn
                    model_params = {'epochs': 5, 'batch_size': 64}

                # Feature selection config
                fs_config = FeatureSelectionConfig(
                    correlation_threshold=corr,
                    top_k_features=top_k,
                    selection_method='anova',
                    scope='global'  # Global scope for speed
                )

                config = TrainingConfig(
                    model_type=model,
                    model_params=model_params,
                    feature_selection=fs_config,
                    random_state=42
                )

                configs.append(config)

    return configs


def run_test(run_number, features_df, labels, subject_ids, configs, cache_dir):
    """Run training pipeline and return cache statistics."""

    print(f"\n{'='*70}")
    print(f"RUN #{run_number} - {'COLD START (train fresh)' if run_number == 1 else 'WARM START (load cache)'}")
    print(f"{'='*70}")

    # Create formatter (quiet mode for cleaner output)
    formatter = TrainingOutputFormatter(verbosity=Verbosity.NORMAL)

    # Create pipeline with model cache
    pipeline = TrainingPipeline(
        features_df=features_df,
        labels=labels,
        subject_ids=subject_ids,
        output_dir=Path(cache_dir) / f"results_run{run_number}",
        experiment_name=f"test_run_{run_number}",
        formatter=formatter,
        enable_model_cache=True,
        model_cache_dir=cache_dir / "loso_model_cache"
    )

    # Run training
    start_time = time.time()
    results = pipeline.run_grid(configs, save_intermediate=False)
    elapsed = time.time() - start_time

    # Get cache statistics
    cache_stats = pipeline.model_cache.get_stats() if pipeline.model_cache else None

    return {
        'run': run_number,
        'n_configs': len(configs),
        'n_results': len(results),
        'elapsed_seconds': elapsed,
        'cache_stats': cache_stats,
        'results': results
    }


def print_cache_report(run1_stats, run2_stats):
    """Print comprehensive cache performance report."""

    print("\n" + "="*70)
    print("CACHE PERFORMANCE REPORT")
    print("="*70)

    # Run 1 (Cold start)
    cache1 = run1_stats['cache_stats']
    print(f"\n📊 RUN 1 (Cold Start - Train Fresh)")
    print(f"   Configurations: {run1_stats['n_configs']}")
    print(f"   Time: {run1_stats['elapsed_seconds']:.1f}s")
    if cache1:
        print(f"   Cache Misses: {cache1['metrics']['misses']} (expected)")
        print(f"   Cache Hits: {cache1['metrics']['hits']}")
        print(f"   Models Cached: {cache1['storage']['cached_models']}")

    # Run 2 (Warm start)
    cache2 = run2_stats['cache_stats']
    print(f"\n📊 RUN 2 (Warm Start - Load Cache)")
    print(f"   Configurations: {run2_stats['n_configs']}")
    print(f"   Time: {run2_stats['elapsed_seconds']:.1f}s")
    if cache2:
        print(f"   Cache Hits: {cache2['metrics']['hits']} (expected)")
        print(f"   Cache Misses: {cache2['metrics']['misses']}")
        print(f"   Hit Rate: {cache2['metrics']['hit_rate']*100:.1f}%")

    # Performance comparison
    print(f"\n⚡ PERFORMANCE")
    speedup = run1_stats['elapsed_seconds'] / run2_stats['elapsed_seconds']
    time_saved = run1_stats['elapsed_seconds'] - run2_stats['elapsed_seconds']
    print(f"   Cold Start: {run1_stats['elapsed_seconds']:.1f}s")
    print(f"   Warm Start: {run2_stats['elapsed_seconds']:.1f}s")
    print(f"   Speedup: {speedup:.1f}x")
    print(f"   Time Saved: {time_saved:.1f}s")

    # Validation
    print(f"\n✓ VALIDATION")

    expected_cache_hits = run1_stats['n_configs'] * 3  # 3 subjects (LOSO folds)
    actual_hits = cache2['metrics']['hits'] if cache2 else 0

    if actual_hits >= expected_cache_hits:
        print(f"   ✅ All models cached: {actual_hits}/{expected_cache_hits} hits")
    else:
        print(f"   ⚠️  Some misses: {actual_hits}/{expected_cache_hits} hits")

    if cache2 and cache2['metrics']['hit_rate'] >= 0.95:
        print(f"   ✅ High cache hit rate: {cache2['metrics']['hit_rate']*100:.1f}%")
    else:
        print(f"   ⚠️  Low hit rate: {cache2['metrics']['hit_rate']*100:.1f}%")

    if speedup >= 1.2:
        print(f"   ✅ Significant speedup: {speedup:.1f}x")
    else:
        print(f"   ⚠️  Limited speedup: {speedup:.1f}x")

    print("="*70)


def main():
    """Run comprehensive cache test."""

    print("\n" + "="*70)
    print("COMPREHENSIVE LOSO CACHE TEST")
    print("="*70)
    print("\nTest Configuration:")
    print("  Subjects: 3")
    print("  Models: XGBoost, Random Forest" + (", FNN" if FNN_AVAILABLE else " (FNN skipped)"))
    print("  Correlation cutoffs: 0.75, 0.90")
    print("  Top-K values: 30, all features")
    print("  Total configs: 3 models × 2 corr × 2 top_k = 12")
    print("  Total runs: 12 configs × 3 folds = 36 training runs")
    print("\nTest Plan:")
    print("  1. Run 1: Train all models (cache MISS expected)")
    print("  2. Run 2: Load from cache (cache HIT expected)")
    print("  3. Verify cache statistics")
    print("="*70)

    # Generate synthetic data
    print("\n[1/5] Generating synthetic data...")
    features_df, labels, subject_ids = generate_synthetic_data(
        n_subjects=3,
        n_epochs_per_subject=200,
        n_features=149
    )
    print(f"   ✓ Generated {len(features_df)} epochs")
    print(f"   ✓ {len(np.unique(subject_ids))} subjects")
    print(f"   ✓ {features_df.shape[1]} features")
    print(f"   ✓ Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Create configs
    print("\n[2/5] Creating configuration grid...")
    configs = create_test_configs(include_fnn=True)
    print(f"   ✓ Created {len(configs)} configurations")
    for i, config in enumerate(configs, 1):
        print(f"      {i}. {config.get_config_id()}")

    # Use temporary directory for cache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # Run 1: Cold start (cache misses)
        print("\n[3/5] Running first pass (cold start)...")
        run1_stats = run_test(1, features_df, labels, subject_ids, configs, cache_dir)

        # Run 2: Warm start (cache hits)
        print("\n[4/5] Running second pass (warm start)...")
        run2_stats = run_test(2, features_df, labels, subject_ids, configs, cache_dir)

        # Report
        print("\n[5/5] Generating cache report...")
        print_cache_report(run1_stats, run2_stats)

    print("\n✅ Test complete!")

    # Return success if cache hit rate is good
    cache2 = run2_stats['cache_stats']
    if cache2 and cache2['metrics']['hit_rate'] >= 0.95:
        return 0
    else:
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
