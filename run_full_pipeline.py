"""
Full Thesis Pipeline Runner
============================

Single entry point to run the complete thesis pipeline:
  Stage 1: Feature extraction (with Layer 1 caching)
  Stage 2: Training with LOSO CV (with Layer 2 model caching)
  Stage 3: Evaluation and visualization

Designed for reproducible runs across different machines.

Usage:
    python run_full_pipeline.py --quick --data-path /path/to/data
    python run_full_pipeline.py --full --data-path /path/to/data
    python run_full_pipeline.py --full --data-path /path/to/data --benchmark

Author: Lennart Gorzel
Date: March 2026
"""

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

GLOBAL_CACHE_DIR = Path("./results/features_cache_global")
LOSO_CACHE_DIR = Path("./results/loso_model_cache")
RESULTS_DIR = Path("./results")

# Thesis-locked grid: 2 models x 3 correlation x 3 top-k = 18 configs
THESIS_MODELS = ['xgboost', 'random_forest']
THESIS_CORRELATION = [0.75, 0.90, None]
THESIS_TOP_K = [30, 50, None]


# =============================================================================
# System Info
# =============================================================================

def collect_system_info() -> Dict:
    """Collect machine info for cross-system comparison."""
    import multiprocessing

    info = {
        'timestamp': datetime.now().isoformat(),
        'hostname': platform.node(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu': platform.processor() or 'unknown',
        'cpu_count': multiprocessing.cpu_count(),
        'architecture': platform.machine(),
    }

    # Try to get memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['ram_total_gb'] = round(mem.total / (1024**3), 1)
        info['ram_available_gb'] = round(mem.available / (1024**3), 1)
    except ImportError:
        info['ram_total_gb'] = 'unknown (install psutil)'

    return info


# =============================================================================
# Environment Check
# =============================================================================

def check_environment() -> List[str]:
    """Check that all required packages are installed. Returns list of issues."""
    issues = []
    required = [
        ('mne', 'mne'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('antropy', 'antropy'),
        ('pandas', 'pandas'),
        ('sklearn', 'scikit-learn'),
        ('xgboost', 'xgboost'),
        ('joblib', 'joblib'),
        ('yaml', 'pyyaml'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
    ]

    for import_name, pip_name in required:
        try:
            __import__(import_name)
        except ImportError:
            issues.append(f"Missing: {pip_name} (pip install {pip_name})")

    return issues


def check_data(data_path: str, expected_subjects: int) -> Dict:
    """Verify BOAS dataset exists and count subjects."""
    data_dir = Path(data_path)

    if not data_dir.exists():
        return {'valid': False, 'error': f'Data directory not found: {data_dir}'}

    # Count subject directories
    subject_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith('sub-')
    ])

    if not subject_dirs:
        return {'valid': False, 'error': f'No sub-* directories in {data_dir}'}

    return {
        'valid': True,
        'n_subjects': len(subject_dirs),
        'subjects': [d.name.replace('sub-', '') for d in subject_dirs],
        'data_path': str(data_dir)
    }


# =============================================================================
# Pipeline Stages
# =============================================================================

def run_stage1_feature_extraction(
    data_path: str,
    subjects: List[str],
    experiment_name: str
) -> Dict:
    """
    Stage 1: Feature extraction with Layer 1 caching.

    Returns timing and cache metrics.
    """
    from config import ConfigManager
    from pipeline import DataPipeline

    config = ConfigManager.create_default_config(
        experiment_name=experiment_name,
        data_path=data_path,
        model_type='xgboost'
    )
    config.data.subjects = subjects

    pipeline = DataPipeline(config)
    stats = pipeline.run()

    return {
        'n_subjects': stats.get('n_subjects_processed', 0),
        'n_epochs': stats.get('n_total_epochs', 0),
        'n_features': stats.get('n_features', 0),
        'elapsed_seconds': stats.get('elapsed_time_seconds', 0),
        'cache_hits': stats.get('cache_hits', 0),
        'cache_misses': stats.get('cache_misses', 0),
        'output_dir': stats.get('output_directory', ''),
    }


def run_stage2_training(
    subjects: List[str],
    output_dir: Path,
    experiment_name: str,
    enable_model_cache: bool = True,
    n_jobs: int = 1,
) -> Dict:
    """
    Stage 2: Training with LOSO CV and Layer 2 model caching.

    Returns timing, cache metrics, and model results.
    """
    import numpy as np
    import pandas as pd
    from training import TrainingPipeline, create_training_grid
    from output_formatter import get_formatter, set_verbosity_from_args
    from evaluation import AggregatedEvaluation

    # Load cached features
    from run_training import load_cached_features

    load_start = time.time()
    features_df, labels, subject_ids = load_cached_features(subjects)
    load_time = time.time() - load_start

    # Create thesis grid
    configs = create_training_grid(
        models=THESIS_MODELS,
        correlation_thresholds=THESIS_CORRELATION,
        top_k_features=THESIS_TOP_K,
        random_state=42,
    )

    # Setup formatter
    set_verbosity_from_args(verbose=False, quiet=True)
    formatter = get_formatter()

    # Create training pipeline
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    pipeline = TrainingPipeline(
        features_df=features_df,
        labels=labels,
        subject_ids=subject_ids,
        output_dir=exp_dir,
        experiment_name=experiment_name,
        formatter=formatter,
        n_jobs=n_jobs,
        enable_model_cache=enable_model_cache,
    )

    # Run grid
    train_start = time.time()
    results = pipeline.run_grid(configs, save_intermediate=True)
    train_time = time.time() - train_start

    # Collect model cache metrics
    model_cache_metrics = {}
    if pipeline.model_cache is not None:
        stats = pipeline.model_cache.get_stats()
        model_cache_metrics = stats.get('metrics', {})
        model_cache_metrics['storage'] = stats.get('storage', {})

    # Build results summary
    results_summary = []
    best_result = None
    best_acc = 0

    for r in results:
        entry = {
            'config_id': r.config_id,
            'accuracy_mean': round(r.accuracy_mean, 4),
            'accuracy_std': round(r.accuracy_std, 4),
            'kappa_mean': round(r.kappa_mean, 4),
            'f1_macro_mean': round(r.f1_macro_mean, 4),
            'meets_targets': bool(
                getattr(r, 'meets_accuracy_target', False)
                and getattr(r, 'meets_kappa_target', False)
            ),
        }
        results_summary.append(entry)
        if r.accuracy_mean > best_acc:
            best_acc = r.accuracy_mean
            best_result = entry

    return {
        'feature_load_time': round(load_time, 2),
        'training_time': round(train_time, 2),
        'n_configs': len(configs),
        'n_folds': len(np.unique(subject_ids)),
        'total_runs': len(configs) * len(np.unique(subject_ids)),
        'model_cache_metrics': model_cache_metrics,
        'best_result': best_result,
        'all_results': results_summary,
        'output_dir': str(exp_dir),
    }


def run_stage3_evaluation(
    stage2_output_dir: Path,
    stage2_results: Dict,
) -> Dict:
    """
    Stage 3: Generate visualizations and LaTeX tables.
    """
    viz_start = time.time()

    try:
        from cache_visualization import (
            generate_all_cache_figures,
            generate_results_latex_table,
        )

        viz_dir = stage2_output_dir / "figures"
        viz_dir.mkdir(exist_ok=True)
        latex_dir = stage2_output_dir / "latex"
        latex_dir.mkdir(exist_ok=True)

        # Generate results LaTeX table
        if stage2_results.get('all_results'):
            generate_results_latex_table(
                stage2_results['all_results'],
                latex_dir / "results_table.tex"
            )

        viz_time = time.time() - viz_start
        return {'elapsed_seconds': round(viz_time, 2), 'output_dir': str(viz_dir)}

    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}")
        return {'elapsed_seconds': 0, 'error': str(e)}


# =============================================================================
# Benchmark Mode
# =============================================================================

def run_benchmark(
    data_path: str,
    subjects: List[str],
    n_jobs: int = 1,
) -> Dict:
    """
    Benchmark mode: Run cold -> warm to measure REAL caching speedup.

    This is the core measurement for the thesis:
    - Run 1 (COLD): Clear LOSO model cache, train everything from scratch
    - Run 2 (WARM): LOSO cache populated, all folds should be cache hits
    - Comparison: Real wall-clock time difference = actual speedup

    Feature cache (Layer 1) is NOT cleared -- it's always populated.
    The benchmark focuses on Layer 2 (LOSO model caching).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    system_info = collect_system_info()

    print("\n" + "=" * 70)
    print("BENCHMARK MODE: Measuring LOSO Model Cache Performance")
    print("=" * 70)
    print(f"Machine: {system_info.get('hostname', 'unknown')}")
    print(f"CPU: {system_info.get('cpu', 'unknown')} ({system_info.get('cpu_count', '?')} cores)")
    print(f"RAM: {system_info.get('ram_total_gb', '?')} GB")
    print(f"Subjects: {len(subjects)}")
    print(f"Grid: {len(THESIS_MODELS)} models x {len(THESIS_CORRELATION)} corr x {len(THESIS_TOP_K)} top-k"
          f" = {len(THESIS_MODELS) * len(THESIS_CORRELATION) * len(THESIS_TOP_K)} configs")
    print(f"Total folds: {len(subjects)} per config (LOSO)")
    print("=" * 70)

    # Ensure feature cache is populated (Stage 1)
    print("\n[Stage 1] Ensuring feature cache is populated...")
    stage1_start = time.time()
    stage1 = run_stage1_feature_extraction(
        data_path, subjects, f"benchmark_features_{timestamp}"
    )
    stage1_time = time.time() - stage1_start
    print(f"  Feature cache: {stage1['cache_hits']} hits, {stage1['cache_misses']} misses "
          f"({stage1_time:.1f}s)")

    # RUN 1: COLD (clear LOSO model cache)
    print("\n" + "-" * 70)
    print("[Run 1 - COLD] Clearing LOSO model cache and training from scratch...")
    print("-" * 70)

    # Clear the LOSO model cache
    if LOSO_CACHE_DIR.exists():
        n_cleared = len(list(LOSO_CACHE_DIR.glob("*.joblib")))
        shutil.rmtree(LOSO_CACHE_DIR)
        print(f"  Cleared {n_cleared} cached models")

    cold_start = time.time()
    cold_results = run_stage2_training(
        subjects, output_dir, f"cold_run_{timestamp}",
        enable_model_cache=True,  # ON so it populates cache for warm run
        n_jobs=n_jobs,
    )
    cold_time = time.time() - cold_start

    cold_cache = cold_results.get('model_cache_metrics', {})
    print(f"\n  COLD run complete: {cold_time:.1f}s")
    print(f"  Cache: {cold_cache.get('hits', 0)} hits, {cold_cache.get('misses', 0)} misses")

    # RUN 2: WARM (LOSO cache should now be populated)
    print("\n" + "-" * 70)
    print("[Run 2 - WARM] Running with populated LOSO model cache...")
    print("-" * 70)

    warm_start = time.time()
    warm_results = run_stage2_training(
        subjects, output_dir, f"warm_run_{timestamp}",
        enable_model_cache=True,
        n_jobs=n_jobs,
    )
    warm_time = time.time() - warm_start

    warm_cache = warm_results.get('model_cache_metrics', {})
    print(f"\n  WARM run complete: {warm_time:.1f}s")
    print(f"  Cache: {warm_cache.get('hits', 0)} hits, {warm_cache.get('misses', 0)} misses")

    # COMPARISON
    speedup = cold_time / warm_time if warm_time > 0 else float('inf')
    time_saved = cold_time - warm_time

    # Per-fold timing from cache registry
    per_fold_cold_time = cold_time / cold_results.get('total_runs', 1)
    per_fold_warm_time = warm_time / warm_results.get('total_runs', 1)

    benchmark_report = {
        'system_info': system_info,
        'dataset': {
            'n_subjects': len(subjects),
            'subject_ids': subjects,
        },
        'grid': {
            'models': THESIS_MODELS,
            'correlation_thresholds': [str(c) for c in THESIS_CORRELATION],
            'top_k_features': [str(k) for k in THESIS_TOP_K],
            'n_configs': cold_results.get('n_configs', 0),
            'n_folds': cold_results.get('n_folds', 0),
            'total_runs': cold_results.get('total_runs', 0),
        },
        'stage1_feature_extraction': {
            'elapsed_seconds': round(stage1_time, 2),
            'cache_hits': stage1['cache_hits'],
            'cache_misses': stage1['cache_misses'],
        },
        'cold_run': {
            'total_seconds': round(cold_time, 2),
            'per_fold_seconds': round(per_fold_cold_time, 2),
            'cache_hits': cold_cache.get('hits', 0),
            'cache_misses': cold_cache.get('misses', 0),
            'hit_rate': cold_cache.get('hit_rate', 0),
        },
        'warm_run': {
            'total_seconds': round(warm_time, 2),
            'per_fold_seconds': round(per_fold_warm_time, 2),
            'cache_hits': warm_cache.get('hits', 0),
            'cache_misses': warm_cache.get('misses', 0),
            'hit_rate': warm_cache.get('hit_rate', 0),
        },
        'comparison': {
            'speedup_factor': round(speedup, 2),
            'time_saved_seconds': round(time_saved, 2),
            'time_saved_minutes': round(time_saved / 60, 2),
            'time_saved_percent': round((time_saved / cold_time) * 100, 1) if cold_time > 0 else 0,
        },
        'model_results': {
            'cold_best': cold_results.get('best_result'),
            'warm_best': warm_results.get('best_result'),
        },
    }

    # Save benchmark report
    report_path = output_dir / "benchmark_report.json"
    with open(report_path, 'w') as f:
        json.dump(benchmark_report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Machine:          {system_info.get('hostname', 'unknown')}")
    print(f"Subjects:         {len(subjects)}")
    print(f"Configs:          {cold_results.get('n_configs', 0)}")
    print(f"Total folds:      {cold_results.get('total_runs', 0)}")
    print()
    print(f"  COLD run (no cache):   {cold_time:>8.1f}s  "
          f"({cold_cache.get('hits', 0)} hits / {cold_cache.get('misses', 0)} misses)")
    print(f"  WARM run (cached):     {warm_time:>8.1f}s  "
          f"({warm_cache.get('hits', 0)} hits / {warm_cache.get('misses', 0)} misses)")
    print()
    print(f"  Speedup:               {speedup:>8.1f}x")
    print(f"  Time saved:            {time_saved:>8.1f}s ({time_saved/60:.1f} min)")
    print(f"  Time saved:            {(time_saved/cold_time)*100:>7.1f}%" if cold_time > 0 else "")
    print()
    print(f"  Per-fold (cold):       {per_fold_cold_time:>8.2f}s")
    print(f"  Per-fold (warm):       {per_fold_warm_time:>8.2f}s")
    print()

    if cold_results.get('best_result'):
        b = cold_results['best_result']
        print(f"  Best model:            {b['config_id']}")
        print(f"  Accuracy:              {b['accuracy_mean']:.4f}")
        print(f"  Kappa:                 {b['kappa_mean']:.4f}")
        print(f"  F1-macro:              {b['f1_macro_mean']:.4f}")

    print()
    print(f"  Report saved to:       {report_path}")
    print("=" * 70)

    return benchmark_report


# =============================================================================
# Main
# =============================================================================

def parse_arguments():
    """Parse command-line arguments. Kept minimal for thesis reproducibility."""
    parser = argparse.ArgumentParser(
        description="Run the complete thesis pipeline (feature extraction + training + evaluation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_pipeline.py --quick --data-path /path/to/data
  python run_full_pipeline.py --full --data-path /path/to/data
  python run_full_pipeline.py --full --data-path /path/to/data --benchmark
        """
    )

    # Dataset size (required)
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument('--quick', action='store_true', help='3 subjects (test run)')
    size_group.add_argument('--pilot', action='store_true', help='10 subjects (pilot)')
    size_group.add_argument('--full', action='store_true', help='128 subjects (thesis)')

    # Data path (required on first run)
    parser.add_argument(
        '--data-path', type=str, required=True,
        help='Path to BOAS dataset directory (containing sub-1/, sub-2/, ...)'
    )

    # Benchmark mode
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Run cold+warm to measure real LOSO cache speedup'
    )

    # Optional
    parser.add_argument('--n-jobs', type=int, default=1, help='Parallel jobs for training')

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Setup logging
    from utils import setup_logging
    setup_logging('INFO')

    total_start = time.time()

    # Determine subjects
    if args.quick:
        subjects = [str(i) for i in range(1, 4)]
        mode = "quick"
    elif args.pilot:
        subjects = [str(i) for i in range(1, 11)]
        mode = "pilot"
    else:
        subjects = [str(i) for i in range(1, 129)]
        mode = "full"

    # Step 0: Check environment
    print("\n[0/3] Checking environment...")
    issues = check_environment()
    if issues:
        print("  Missing dependencies:")
        for issue in issues:
            print(f"    - {issue}")
        print("\n  Run: pip install -r requirements.txt")
        sys.exit(1)
    print("  All dependencies OK")

    # Step 0b: Check data
    print(f"\n  Checking data at: {args.data_path}")
    data_status = check_data(args.data_path, len(subjects))
    if not data_status['valid']:
        print(f"  ERROR: {data_status['error']}")
        sys.exit(1)
    print(f"  Found {data_status['n_subjects']} subjects")

    # Step 0c: System info
    system_info = collect_system_info()
    print(f"  Machine: {system_info.get('hostname', 'unknown')}, "
          f"{system_info.get('cpu_count', '?')} cores, "
          f"{system_info.get('ram_total_gb', '?')} GB RAM")

    # Benchmark mode
    if args.benchmark:
        report = run_benchmark(args.data_path, subjects, n_jobs=args.n_jobs)
        total_time = time.time() - total_start
        print(f"\nTotal benchmark time: {total_time:.1f}s ({total_time/60:.1f} min)")
        return

    # Normal mode: run all 3 stages
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"thesis_{mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Feature extraction
    print(f"\n[1/3] Feature extraction ({len(subjects)} subjects)...")
    stage1_start = time.time()
    stage1 = run_stage1_feature_extraction(
        args.data_path, subjects, f"thesis_{mode}_{timestamp}"
    )
    stage1_time = time.time() - stage1_start
    print(f"  Done: {stage1['n_epochs']} epochs, {stage1['n_features']} features "
          f"({stage1_time:.1f}s)")
    print(f"  Cache: {stage1['cache_hits']} hits, {stage1['cache_misses']} misses")

    # Stage 2: Training
    print(f"\n[2/3] Training with LOSO CV...")
    stage2_start = time.time()
    stage2 = run_stage2_training(
        subjects, output_dir, f"training_{mode}_{timestamp}",
        enable_model_cache=True,
        n_jobs=args.n_jobs,
    )
    stage2_time = time.time() - stage2_start
    print(f"  Done: {stage2['n_configs']} configs x {stage2['n_folds']} folds "
          f"({stage2_time:.1f}s)")

    cache_m = stage2.get('model_cache_metrics', {})
    print(f"  LOSO cache: {cache_m.get('hits', 0)} hits, {cache_m.get('misses', 0)} misses")

    # Stage 3: Evaluation
    print(f"\n[3/3] Generating evaluation outputs...")
    stage2_dir = Path(stage2['output_dir'])
    stage3 = run_stage3_evaluation(stage2_dir, stage2)
    print(f"  Done ({stage3.get('elapsed_seconds', 0):.1f}s)")

    # Save run report
    total_time = time.time() - total_start
    run_report = {
        'system_info': system_info,
        'mode': mode,
        'n_subjects': len(subjects),
        'timing': {
            'stage1_feature_extraction_s': round(stage1_time, 2),
            'stage2_training_s': round(stage2_time, 2),
            'stage3_evaluation_s': round(stage3.get('elapsed_seconds', 0), 2),
            'total_s': round(total_time, 2),
        },
        'stage1_cache': {
            'hits': stage1['cache_hits'],
            'misses': stage1['cache_misses'],
        },
        'stage2_model_cache': cache_m,
        'best_result': stage2.get('best_result'),
        'all_results': stage2.get('all_results', []),
    }

    report_path = output_dir / "run_report.json"
    with open(report_path, 'w') as f:
        json.dump(run_report, f, indent=2, default=str)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Mode:              {mode} ({len(subjects)} subjects)")
    print(f"  Feature extraction: {stage1_time:.1f}s")
    print(f"  Training:          {stage2_time:.1f}s")
    print(f"  Evaluation:        {stage3.get('elapsed_seconds', 0):.1f}s")
    print(f"  Total:             {total_time:.1f}s ({total_time/60:.1f} min)")

    if stage2.get('best_result'):
        b = stage2['best_result']
        print(f"\n  Best: {b['config_id']}")
        print(f"    Accuracy: {b['accuracy_mean']:.4f}, Kappa: {b['kappa_mean']:.4f}, "
              f"F1: {b['f1_macro_mean']:.4f}")

    print(f"\n  Results: {output_dir}")
    print(f"  Report:  {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
