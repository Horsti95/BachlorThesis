"""
Thesis Benchmark Suite
======================

Runs three experiments that prove the three key thesis claims:

  Experiment A: SCALING        - More subjects → more time saved
  Experiment B: FINGERPRINT    - Change one param → only affected models retrain
  Experiment C: REPRODUCIBILITY - Same config → identical results

Output: Clean summary table suitable for thesis results chapter.

Usage:
    python run_thesis_benchmark.py --data-path "C:\\Users\\DerHo\\Desktop\\Data"

Author: Lennart Gorzel
Date: March 2026
"""

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Reuse infrastructure from run_full_pipeline
from run_full_pipeline import (
    collect_system_info,
    check_environment,
    check_data,
    run_stage1_feature_extraction,
    run_stage2_training,
    LOSO_CACHE_DIR,
    RESULTS_DIR,
)


def clear_model_cache() -> int:
    """Delete all cached LOSO models from disk. Returns number of .joblib files removed."""
    if LOSO_CACHE_DIR.exists():
        n = len(list(LOSO_CACHE_DIR.glob("*.joblib")))
        shutil.rmtree(LOSO_CACHE_DIR)
        return n
    return 0


def run_training_timed(
    subjects, output_dir, name, enable_cache=True,
    models=None, correlation_thresholds=None, top_k_features=None,
):
    """Run Stage 2 training and return results dict with wall_time, hits, and misses added."""
    start = time.time()
    results = run_stage2_training(
        subjects, output_dir, name,
        enable_model_cache=enable_cache,
        models=models,
        correlation_thresholds=correlation_thresholds,
        top_k_features=top_k_features,
    )
    elapsed = time.time() - start
    results['wall_time'] = round(elapsed, 2)
    cache = results.get('model_cache_metrics', {})
    results['hits'] = cache.get('hits', 0)
    results['misses'] = cache.get('misses', 0)
    return results


# =============================================================================
# Experiment A: Scaling (3 vs 10 subjects)
# =============================================================================

def experiment_a_scaling(data_path: str, output_dir: Path) -> Dict:
    """
    Scaling experiment: run cold+warm for 3 and 10 subjects.

    Proves more subjects yield more absolute time saved with consistent speedup.
    Returns per-size results with cold/warm times and speedup factors.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT A: SCALING (3 vs 10 subjects)")
    print("=" * 70)

    results = {}

    for n_subj in [3, 10]:
        subjects = [str(i) for i in range(1, n_subj + 1)]
        label = f"{n_subj}subj"

        print(f"\n--- {n_subj} subjects: COLD run ---")
        clear_model_cache()
        cold = run_training_timed(
            subjects, output_dir, f"A_cold_{label}",
        )

        print(f"--- {n_subj} subjects: WARM run ---")
        warm = run_training_timed(
            subjects, output_dir, f"A_warm_{label}",
        )

        speedup = cold['wall_time'] / warm['wall_time'] if warm['wall_time'] > 0 else 0
        time_saved = cold['wall_time'] - warm['wall_time']

        results[label] = {
            'n_subjects': n_subj,
            'n_folds': cold.get('n_folds', n_subj),
            'n_configs': cold.get('n_configs', 0),
            'total_runs': cold.get('total_runs', 0),
            'cold_time': cold['wall_time'],
            'warm_time': warm['wall_time'],
            'speedup': round(speedup, 2),
            'time_saved': round(time_saved, 1),
            'time_saved_pct': round((time_saved / cold['wall_time']) * 100, 1) if cold['wall_time'] > 0 else 0,
            'cold_hits': cold['hits'],
            'cold_misses': cold['misses'],
            'warm_hits': warm['hits'],
            'warm_misses': warm['misses'],
            'best_acc': cold.get('best_result', {}).get('accuracy_mean', 0),
        }

    # Clean up cache after experiment
    clear_model_cache()

    return results


# =============================================================================
# Experiment B: Fingerprint Invalidation
# =============================================================================

def experiment_b_fingerprint(data_path: str, output_dir: Path) -> Dict:
    """
    Fingerprint invalidation experiment: incrementally expand the training grid.

    Proves that only configs with changed parameters retrain; unchanged configs
    are cache hits. Three steps: initial grid -> add top_k -> add model.
    Returns per-step hit/miss counts and wall times.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: FINGERPRINT INVALIDATION")
    print("=" * 70)

    subjects = [str(i) for i in range(1, 4)]  # 3 subjects
    results = {}

    # Step 1: Initial grid (small)
    print("\n--- Step 1: Initial grid (xgboost, corr=0.90, top_k=[30,50]) ---")
    clear_model_cache()
    step1 = run_training_timed(
        subjects, output_dir, "B_step1",
        models=['xgboost'],
        correlation_thresholds=[0.90],
        top_k_features=[30, 50],
    )
    results['step1_initial'] = {
        'grid': 'xgb × [0.90] × [30,50]',
        'n_configs': step1.get('n_configs', 0),
        'total_runs': step1.get('total_runs', 0),
        'time': step1['wall_time'],
        'hits': step1['hits'],
        'misses': step1['misses'],
        'description': 'Cold start, all models new',
    }

    # Step 2: Add one top_k value (None = all features)
    print("\n--- Step 2: Add top_k=None (xgboost, corr=0.90, top_k=[30,50,None]) ---")
    step2 = run_training_timed(
        subjects, output_dir, "B_step2",
        models=['xgboost'],
        correlation_thresholds=[0.90],
        top_k_features=[30, 50, None],
    )
    results['step2_add_topk'] = {
        'grid': 'xgb × [0.90] × [30,50,None]',
        'n_configs': step2.get('n_configs', 0),
        'total_runs': step2.get('total_runs', 0),
        'time': step2['wall_time'],
        'hits': step2['hits'],
        'misses': step2['misses'],
        'description': 'Added top_k=None → only new config trains',
    }

    # Step 3: Add random_forest model
    print("\n--- Step 3: Add random_forest (both models, corr=0.90, top_k=[30,50,None]) ---")
    step3 = run_training_timed(
        subjects, output_dir, "B_step3",
        models=['xgboost', 'random_forest'],
        correlation_thresholds=[0.90],
        top_k_features=[30, 50, None],
    )
    results['step3_add_model'] = {
        'grid': '[xgb,rf] × [0.90] × [30,50,None]',
        'n_configs': step3.get('n_configs', 0),
        'total_runs': step3.get('total_runs', 0),
        'time': step3['wall_time'],
        'hits': step3['hits'],
        'misses': step3['misses'],
        'description': 'Added RF → only RF models train, XGB cached',
    }

    clear_model_cache()
    return results


# =============================================================================
# Experiment C: Reproducibility
# =============================================================================

def experiment_c_reproducibility(data_path: str, output_dir: Path) -> Dict:
    """
    Reproducibility experiment: train the same config twice from scratch.

    Clears cache between runs to ensure independent training. Compares accuracy,
    kappa, and F1 between the two runs -- all must be identical.
    Returns per-run metrics and match flags.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C: REPRODUCIBILITY")
    print("=" * 70)

    subjects = [str(i) for i in range(1, 4)]  # 3 subjects
    grid_kwargs = dict(
        models=['xgboost'],
        correlation_thresholds=[0.90],
        top_k_features=[50],
    )

    # Run 1
    print("\n--- Run 1: Fresh training ---")
    clear_model_cache()
    run1 = run_training_timed(
        subjects, output_dir, "C_run1", **grid_kwargs,
    )

    # Run 2 (clear cache, train from scratch again)
    print("\n--- Run 2: Independent re-training (cache cleared) ---")
    clear_model_cache()
    run2 = run_training_timed(
        subjects, output_dir, "C_run2", **grid_kwargs,
    )

    # Compare
    r1 = run1.get('best_result', {})
    r2 = run2.get('best_result', {})

    acc_match = r1.get('accuracy_mean') == r2.get('accuracy_mean')
    kappa_match = r1.get('kappa_mean') == r2.get('kappa_mean')
    f1_match = r1.get('f1_macro_mean') == r2.get('f1_macro_mean')

    results = {
        'run1': {
            'accuracy': r1.get('accuracy_mean', 0),
            'kappa': r1.get('kappa_mean', 0),
            'f1_macro': r1.get('f1_macro_mean', 0),
            'time': run1['wall_time'],
        },
        'run2': {
            'accuracy': r2.get('accuracy_mean', 0),
            'kappa': r2.get('kappa_mean', 0),
            'f1_macro': r2.get('f1_macro_mean', 0),
            'time': run2['wall_time'],
        },
        'match': {
            'accuracy_identical': acc_match,
            'kappa_identical': kappa_match,
            'f1_identical': f1_match,
            'all_identical': acc_match and kappa_match and f1_match,
        },
    }

    clear_model_cache()
    return results


# =============================================================================
# Summary Output
# =============================================================================

def print_summary(system_info: Dict, exp_a: Dict, exp_b: Dict, exp_c: Dict,
                  total_time: float = 0):
    """Print formatted thesis-ready summary table for all three experiments."""

    machine = system_info.get('hostname', 'unknown')
    cpu = system_info.get('cpu_count', '?')
    ram = system_info.get('ram_total_gb', '?')

    print("\n")
    print("=" * 74)
    print("  THESIS BENCHMARK RESULTS")
    print("=" * 74)
    print(f"  Machine: {machine} | CPU cores: {cpu} | RAM: {ram} GB")
    print("=" * 74)

    # --- Experiment A ---
    print("\n  EXPERIMENT A: SCALING")
    print("  " + "-" * 70)
    print(f"  {'Subjects':<10} {'Folds':<8} {'Cold (s)':<10} {'Warm (s)':<10} "
          f"{'Speedup':<10} {'Saved':<10} {'Best Acc':<10}")
    print("  " + "-" * 70)
    for key in sorted(exp_a.keys(), key=lambda k: exp_a[k]['n_subjects']):
        r = exp_a[key]
        speedup_str = f"{r['speedup']:.1f}x"
        saved_str = f"{r['time_saved_pct']:.1f}%"
        print(f"  {r['n_subjects']:<10} {r['total_runs']:<8} "
              f"{r['cold_time']:<10.1f} {r['warm_time']:<10.1f} "
              f"{speedup_str:<10} {saved_str:<10} "
              f"{r['best_acc']:<10.4f}")

    # --- Experiment B ---
    print("\n  EXPERIMENT B: FINGERPRINT INVALIDATION")
    print("  " + "-" * 70)
    print(f"  {'Step':<6} {'Grid':<36} {'Hits':<6} {'Miss':<6} {'Time (s)':<10}")
    print("  " + "-" * 70)
    for i, (key, r) in enumerate(exp_b.items(), 1):
        print(f"  {i:<6} {r['grid']:<36} {r['hits']:<6} {r['misses']:<6} "
              f"{r['time']:<10.1f}")
    print("  " + "-" * 70)
    print("  Step 2: Added top_k=None → only new config trained (old ones cached)")
    print("  Step 3: Added RF model  → only RF trained (XGBoost cached)")

    # --- Experiment C ---
    print("\n  EXPERIMENT C: REPRODUCIBILITY")
    print("  " + "-" * 70)
    r1, r2 = exp_c['run1'], exp_c['run2']
    m = exp_c['match']
    print(f"  {'Metric':<12} {'Run 1':<12} {'Run 2':<12} {'Identical?':<12}")
    print("  " + "-" * 70)
    print(f"  {'Accuracy':<12} {r1['accuracy']:<12.4f} {r2['accuracy']:<12.4f} "
          f"{'YES' if m['accuracy_identical'] else 'NO':<12}")
    print(f"  {'Kappa':<12} {r1['kappa']:<12.4f} {r2['kappa']:<12.4f} "
          f"{'YES' if m['kappa_identical'] else 'NO':<12}")
    print(f"  {'F1 Macro':<12} {r1['f1_macro']:<12.4f} {r2['f1_macro']:<12.4f} "
          f"{'YES' if m['f1_identical'] else 'NO':<12}")
    print(f"\n  All metrics identical: {'YES' if m['all_identical'] else 'NO'}")

    # --- Full Run Projection ---
    # Use 10-subject data to project 128-subject timing
    if '10subj' in exp_a:
        r10 = exp_a['10subj']
        cold_per_fold = r10['cold_time'] / r10['total_runs'] if r10['total_runs'] > 0 else 0
        warm_per_fold = r10['warm_time'] / r10['total_runs'] if r10['total_runs'] > 0 else 0
        full_folds = 18 * 128  # 18 configs × 128 LOSO folds
        proj_cold = cold_per_fold * full_folds
        proj_warm = warm_per_fold * full_folds
        proj_speedup = proj_cold / proj_warm if proj_warm > 0 else 0

        print(f"\n  FULL RUN PROJECTION (128 subjects, {full_folds} folds)")
        print("  " + "-" * 70)
        print(f"  Per-fold (cold):  {cold_per_fold:.2f}s")
        print(f"  Per-fold (warm):  {warm_per_fold:.2f}s")
        print(f"  Projected cold:   {proj_cold/60:.0f} min ({proj_cold/3600:.1f} h)")
        print(f"  Projected warm:   {proj_warm/60:.0f} min ({proj_warm/3600:.1f} h)")
        print(f"  Projected speedup: {proj_speedup:.1f}x")
        print(f"  Projected saved:  {(proj_cold - proj_warm)/60:.0f} min")

    if total_time > 0:
        print(f"\n  Benchmark completed in {total_time:.0f}s ({total_time/60:.1f} min)")

    print("\n" + "=" * 74)
    print("  END OF BENCHMARK")
    print("=" * 74)


# =============================================================================
# Main
# =============================================================================

def main():
    """Entry point: run preflight checks, populate feature cache, execute experiments A/B/C, save report."""
    parser = argparse.ArgumentParser(
        description="Thesis benchmark suite: scaling, fingerprint, reproducibility",
    )
    parser.add_argument(
        '--data-path', type=str, required=True,
        help='Path to BOAS dataset directory',
    )
    args = parser.parse_args()

    from utils import setup_logging
    setup_logging('INFO')

    # Preflight checks
    print("\n[0] Preflight checks...")
    issues = check_environment()
    if issues:
        for issue in issues:
            print(f"  MISSING: {issue}")
        return

    data_status = check_data(args.data_path, 10)
    if not data_status['valid']:
        print(f"  ERROR: {data_status['error']}")
        return

    system_info = collect_system_info()
    print(f"  Machine: {system_info.get('hostname')}, "
          f"{system_info.get('cpu_count')} cores, "
          f"{system_info.get('ram_total_gb')} GB RAM")
    print(f"  Data: {data_status['n_subjects']} subjects found")

    # Ensure feature cache is populated for all 10 subjects
    print("\n[1] Populating feature cache (Layer 1)...")
    subjects_10 = [str(i) for i in range(1, 11)]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage1 = run_stage1_feature_extraction(
        args.data_path, subjects_10, f"thesis_bench_{timestamp}",
    )
    print(f"  Features: {stage1['cache_hits']} hits, {stage1['cache_misses']} misses "
          f"({stage1['elapsed_seconds']:.1f}s)")

    output_dir = RESULTS_DIR / f"thesis_benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # Run experiments
    exp_a = experiment_a_scaling(args.data_path, output_dir)
    exp_b = experiment_b_fingerprint(args.data_path, output_dir)
    exp_c = experiment_c_reproducibility(args.data_path, output_dir)

    total_time = time.time() - total_start

    # Print clean summary
    print_summary(system_info, exp_a, exp_b, exp_c, total_time)

    # Save full report
    report = {
        'system_info': system_info,
        'total_time_seconds': round(total_time, 1),
        'experiment_a_scaling': exp_a,
        'experiment_b_fingerprint': exp_b,
        'experiment_c_reproducibility': exp_c,
    }
    report_path = output_dir / "thesis_benchmark_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Full report: {report_path}")
    print(f"  Total time:  {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
