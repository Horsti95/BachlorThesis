"""
Training Experiment Runner
===========================

Main entry point for running training experiments with LOSO cross-validation.

This script:
1. Loads pre-cached features (from global cache)
2. Runs the training grid (models × feature selection configs)
3. Saves evaluation results and visualizations

Prerequisites:
    - Feature cache must be populated (run `python run_experiment.py --full` first)
    - All 128 subjects should be cached in results/features_cache_global/

Usage:
    python run_training.py --quick      # Quick test (3 subjects, 2 configs)
    python run_training.py --pilot      # Pilot (10 subjects, full grid)
    python run_training.py --full       # Full experiment (128 subjects, full grid)

Author: Lennart Gorzel
Date: December 2025
"""

import argparse
import logging
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from config import ConfigManager, ExperimentConfig
from feature_cache import load_features_from_cache, get_cache_info
from feature_selection import FeatureSelectionConfig, create_feature_selection_grid
from cross_validation import LOSOCrossValidator, summarize_cv_splits
from training import (
    TrainingPipeline,
    TrainingConfig,
    create_training_grid,
    create_quick_grid,
    create_thesis_grid,
    create_pilot_grid
)
from evaluation import (
    evaluate_predictions,
    aggregate_evaluations,
    format_evaluation_report,
    format_aggregated_report,
    format_comparison_table,
    AggregatedEvaluation,
    save_all_evaluations
)
from visualization import (
    plot_confusion_matrix,
    plot_performance_comparison,
    plot_multi_metric_comparison,
    plot_per_class_f1,
    plot_class_distribution,
    generate_all_figures
)
from cache_visualization import (
    generate_all_cache_figures,
    calculate_cache_metrics,
    generate_cache_metrics_latex_table,
    generate_results_latex_table
)
from output_formatter import (
    TrainingOutputFormatter,
    Verbosity,
    get_formatter,
    set_verbosity_from_args
)
from utils import setup_logging, get_timestamp
from leaderboard import get_leaderboard

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_DATA_PATH = r"C:\Users\DerHo\Desktop\Data"
DEFAULT_OUTPUT_DIR = "./results"
GLOBAL_CACHE_DIR = Path("./results/features_cache_global")


# =============================================================================
# Data Loading
# =============================================================================

def load_cached_features(
    subjects: List[str],
    cache_dir: Path = GLOBAL_CACHE_DIR,
    n_channels: int = 6
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load features from global cache for specified subjects.
    
    Args:
        subjects: List of subject IDs to load
        cache_dir: Path to cache directory
        n_channels: Number of channels to select (6 or 8)
        
    Returns:
        Tuple of (features_df, labels, subject_ids)
    """
    all_features = []
    all_labels = []
    all_subject_ids = []
    
    start_time = time.time()
    loaded = 0
    failed = []
    
    for subject_id in subjects:
        cache_path = cache_dir / f"subject_{subject_id}_full.npz"
        
        cached = load_features_from_cache(cache_path)
        if cached is None:
            failed.append(subject_id)
            logger.warning(f"Cache miss for subject {subject_id}")
            continue
        
        features_df, labels, cached_channels = cached
        
        # ALWAYS select the requested number of channels to ensure consistent features
        # This handles both standard (F3_, F4_...) and generic (CH1_, CH2_...) naming
        from feature_cache import select_channel_features
        features_df = select_channel_features(features_df, channels_to_keep=n_channels)
        
        all_features.append(features_df)
        all_labels.append(labels)
        all_subject_ids.extend([subject_id] * len(labels))
        loaded += 1
    
    load_time = time.time() - start_time
    
    if not all_features:
        raise RuntimeError(f"No cached features found in {cache_dir}")
    
    # Concatenate all data
    features = pd.concat(all_features, ignore_index=True)
    labels = np.concatenate(all_labels)
    subject_ids = np.array(all_subject_ids)
    
    logger.debug(f"Loaded {loaded}/{len(subjects)} subjects from cache in {load_time:.2f}s")
    if failed:
        logger.warning(f"Failed to load: {failed[:5]}...")
    
    return features, labels, subject_ids


def verify_cache(cache_dir: Path, expected_subjects: int = 128) -> Dict:
    """
    Verify cache integrity before training.
    
    Args:
        cache_dir: Path to cache directory
        expected_subjects: Expected number of cached subjects
        
    Returns:
        Dictionary with cache status
    """
    cache_dir = Path(cache_dir)
    
    if not cache_dir.exists():
        return {
            'valid': False,
            'error': f"Cache directory not found: {cache_dir}",
            'cached_subjects': 0
        }
    
    cache_files = list(cache_dir.glob("subject_*_full.npz"))
    n_cached = len(cache_files)
    
    # Sample one file for info
    sample_info = None
    if cache_files:
        sample_info = get_cache_info(cache_files[0])
    
    return {
        'valid': n_cached > 0,
        'cached_subjects': n_cached,
        'expected_subjects': expected_subjects,
        'complete': n_cached >= expected_subjects,
        'sample_info': sample_info,
        'cache_dir': str(cache_dir)
    }


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run training experiments with LOSO cross-validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset size
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick test: 3 subjects, minimal config grid'
    )
    size_group.add_argument(
        '--pilot', '-p',
        action='store_true',
        help='Pilot test: 10 subjects, full config grid'
    )
    size_group.add_argument(
        '--full', '-f',
        action='store_true',
        help='Full experiment: 128 subjects, full config grid'
    )
    size_group.add_argument(
        '--subjects', '-n',
        type=int,
        help='Custom number of subjects (1-128)'
    )
    
    # Model selection
    parser.add_argument(
        '--models', '-m',
        type=str,
        nargs='+',
        choices=['xgboost', 'random_forest', 'fnn', 'all'],
        default=['xgboost', 'random_forest'],
        help='Models to train (XGBoost + RF for thesis; FNN available but excluded)'
    )
    
    # Grid selection (thesis preset)
    parser.add_argument(
        '--grid', '-g',
        type=str,
        choices=['quick', 'pilot', 'thesis', 'custom'],
        default='custom',
        help='Predefined configuration grid (thesis = 18 configs)'
    )
    
    # Feature selection (for custom grid)
    parser.add_argument(
        '--correlation',
        type=float,
        nargs='+',
        default=[0.75, 0.90, 0],
        help='Correlation thresholds for feature selection (0 = None, thesis: 0.75, 0.90, None)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        nargs='+',
        default=[30, 50, 0],
        help='Top-K features to select (0 = all 149, thesis: 30, 50, None)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Custom experiment name'
    )
    
    # Feature selection method
    parser.add_argument(
        '--pure-mi',
        action='store_true',
        help='Use pure Mutual Information (slower, default is hybrid f_classif->MI)'
    )
    
    # Output options
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
    )
    
    # Verbosity control (mutually exclusive)
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output: detailed per-fold information with tables'
    )
    verbosity_group.add_argument(
        '--quiet',
        action='store_true',
        help='Quiet output: only final summaries, use progress bars'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='Skip confirmation prompt'
    )

    # Cache space management
    parser.add_argument(
        '--cache-min-free-gb',
        type=float,
        default=5.0,
        help='Minimum free disk space (GB) to allow model caching. Caching is skipped below this.'
    )
    parser.add_argument(
        '--cache-max-size-gb',
        type=float,
        default=None,
        help='Maximum model cache size (GB). Oldest models evicted when exceeded. Default: unlimited.'
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs for model training (default: 1)'
    )

    return parser.parse_args()


# =============================================================================
# Main Training Runner
# =============================================================================

def run_training_experiment(
    subjects: List[str],
    models: List[str],
    correlation_thresholds: List[Optional[float]],
    top_k_features: List[Optional[int]],
    output_dir: Path,
    experiment_name: str,
    generate_viz: bool = True,
    use_hybrid: bool = True,
    n_jobs: int = 1,
    cache_min_free_gb: float = 5.0,
    cache_max_size_gb: Optional[float] = None
) -> Dict:
    """
    Run the complete training experiment.
    
    Args:
        subjects: List of subject IDs to include
        models: List of model types
        correlation_thresholds: List of correlation threshold options
        top_k_features: List of top-K options (None = all)
        output_dir: Directory for results
        experiment_name: Experiment identifier
        generate_viz: Whether to generate visualizations
        use_hybrid: Use hybrid feature selection
    Returns:
        Dictionary with experiment results
    """
    start_time = time.time()
    
    # Create output directory
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load cached features
    print("\n[1/4] Loading cached features...")
    features_df, labels, subject_ids = load_cached_features(subjects)

    n_epochs = len(features_df)
    n_features = features_df.shape[1]
    n_subj = len(np.unique(subject_ids))
    class_dist = dict(zip(*np.unique(labels, return_counts=True)))
    print(f"  [v] {n_epochs:,} epochs, {n_features} features, {n_subj} subjects")
    print(f"  [v] Classes: {class_dist}")

    # Step 2: Create training grid
    print("\n[2/4] Creating configuration grid...")
    
    # Process top_k (convert 0 to None for "all features")
    top_k_processed = [None if k == 0 else k for k in top_k_features]
    
    # Process correlation (convert 0 to None for "no filtering")
    corr_processed = [None if c == 0 else c for c in correlation_thresholds]
    
    configs = create_training_grid(
        models=models,
        correlation_thresholds=corr_processed,
        top_k_features=top_k_processed,
        random_state=42,
        use_hybrid=use_hybrid
    )
    
    n_configs = len(configs)
    n_folds = len(np.unique(subject_ids))
    total_runs = n_configs * n_folds
    print(f"  [v] {n_configs} configs x {n_folds} folds = {total_runs:,} runs")

    # Step 3: Run training
    print("\n[3/4] Running training pipeline...")
    
    # Get the formatter for output
    formatter = get_formatter()
    
    pipeline = TrainingPipeline(
        features_df=features_df,
        labels=labels,
        subject_ids=subject_ids,
        output_dir=exp_dir,
        experiment_name=experiment_name,
        formatter=formatter,
        cache_min_free_space_gb=cache_min_free_gb,
        cache_max_size_gb=cache_max_size_gb
    )
    
    results = pipeline.run_grid(configs, save_intermediate=True)
    
    # Step 4: Generate visualizations
    if generate_viz and results:
        print("\n[4/4] Generating visualizations...")
        
        viz_dir = exp_dir / "figures"
        viz_dir.mkdir(exist_ok=True)
        
        # Convert results to format expected by visualization
        agg_results = {r.config_id: r for r in results}
        
        # Create aggregated evaluation objects for visualization
        agg_evals = []
        for r in results:
            agg_eval = AggregatedEvaluation(
                accuracy_mean=r.accuracy_mean,
                accuracy_std=r.accuracy_std,
                kappa_mean=r.kappa_mean,
                kappa_std=r.kappa_std,
                f1_macro_mean=r.f1_macro_mean,
                f1_macro_std=r.f1_macro_std,
                class_f1_means=r.f1_per_class_mean,
                config_id=r.config_id
            )
            agg_evals.append(agg_eval)
        
        try:
            # Performance comparison
            plot_performance_comparison(
                agg_evals,
                metric='accuracy_mean',
                title='Accuracy Comparison',
                save_path=viz_dir / 'accuracy_comparison.png',
                show=False
            )
            
            # Multi-metric comparison
            plot_multi_metric_comparison(
                agg_evals,
                title='Multi-Metric Comparison',
                save_path=viz_dir / 'multi_metric_comparison.png',
                show=False
            )
            
            # Per-class F1 for best model
            best_result = max(agg_evals, key=lambda r: r.accuracy_mean)
            plot_per_class_f1(
                best_result,
                title=f'Per-Class F1 ({best_result.config_id})',
                save_path=viz_dir / 'per_class_f1_best.png',
                show=False
            )
            
            # Class distribution
            plot_class_distribution(
                labels,
                title='Sleep Stage Distribution',
                save_path=viz_dir / 'class_distribution.png',
                show=False
            )
            
            print(f"  [v] Figures saved to {viz_dir}")
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
            print(f"  [!] Visualization failed: {e}")
    
    # Print final results
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Results saved to: {exp_dir}")
    
    if results:
        pipeline.print_results_table()
        
        # Best result summary
        best = max(results, key=lambda r: r.accuracy_mean)
        print(f"\n[#1] Best Configuration: {best.config_id}")
        print(f"   Accuracy: {best.accuracy_mean:.4f} ± {best.accuracy_std:.4f}")
        print(f"   Kappa: {best.kappa_mean:.4f} ± {best.kappa_std:.4f}")
        print(f"   F1-Macro: {best.f1_macro_mean:.4f} ± {best.f1_macro_std:.4f}")
        
        # Clinical targets check
        if best.meets_accuracy_target and best.meets_kappa_target:
            print("\n[OK] MEETS CLINICAL TARGETS")
        else:
            print("\n[X] Does not meet all clinical targets")
    
    print("=" * 70)
    
    return {
        'experiment_name': experiment_name,
        'n_subjects': len(subjects),
        'n_configs': len(configs),
        'n_folds': n_folds,
        'total_runs': total_runs,
        'elapsed_seconds': elapsed,
        'results': [r.to_dict() for r in results] if results else [],
        'output_dir': str(exp_dir)
    }


def main():
    """Parse CLI args, verify cache, build training grid, run experiment, and generate outputs."""
    # Ensure UTF-8 output on Windows (avoids UnicodeEncodeError when piping)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    args = parse_arguments()

    # Suppress harmless sklearn parallel warning (fires thousands of times with RF)
    import warnings
    warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

    # Setup logging
    setup_logging(args.log_level)
    
    # Setup output formatter verbosity
    set_verbosity_from_args(
        verbose=getattr(args, 'verbose', False),
        quiet=getattr(args, 'quiet', False)
    )
    formatter = get_formatter()
    
    # Determine subjects
    if args.quick:
        subjects = [str(i) for i in range(1, 4)]  # 3 subjects
        mode = "quick"
    elif args.pilot:
        subjects = [str(i) for i in range(1, 11)]  # 10 subjects
        mode = "pilot"
    elif args.full:
        subjects = [str(i) for i in range(1, 129)]  # 128 subjects
        mode = "full"
    elif args.subjects:
        n = min(max(1, args.subjects), 128)
        subjects = [str(i) for i in range(1, n + 1)]
        mode = f"{n}subj"
    else:
        # Default to quick
        subjects = [str(i) for i in range(1, 4)]
        mode = "quick"
    
    # =========================================================================
    # VERIFY PREREQUISITES
    # =========================================================================
    cache_status = verify_cache(GLOBAL_CACHE_DIR, len(subjects))

    if not cache_status['valid']:
        print(f"\n[X] Feature cache not found: {cache_status.get('error')}")
        print("    Run feature extraction first: python run_experiment.py --full")
        sys.exit(1)

    if cache_status['cached_subjects'] < len(subjects):
        print(f"\n[!] Warning: Only {cache_status['cached_subjects']} cached, requested {len(subjects)}")
        subjects = [str(i) for i in range(1, cache_status['cached_subjects'] + 1)]

    # Process model selection
    models = args.models
    if 'all' in models:
        models = ['xgboost', 'random_forest', 'fnn']

    # Determine grid configuration
    if args.grid == 'thesis':
        correlation_thresholds = [0.75, 0.90, None]
        top_k_features = [30, 50, None]
        grid_name = "thesis"
    elif args.grid == 'pilot':
        correlation_thresholds = [0.75, 0.90, None]
        top_k_features = [30, 50, None]
        grid_name = "thesis"
    elif args.grid == 'quick':
        correlation_thresholds = [0.90]
        top_k_features = [50]
        grid_name = "quick"
    else:
        correlation_thresholds = [None if c == 0 else c for c in args.correlation]
        top_k_features = args.top_k
        grid_name = "custom"

    # Experiment name
    timestamp = get_timestamp()
    exp_name = args.experiment_name or f"training_{timestamp}_{mode}"

    # Calculate totals
    n_configs = len(models) * len(correlation_thresholds) * len(top_k_features)
    n_folds = len(subjects)
    total_runs = n_configs * n_folds

    # Cache metrics
    cache_metrics = calculate_cache_metrics(GLOBAL_CACHE_DIR, n_subjects_in_run=len(subjects))

    # Time estimate
    time_per_fold_cold = 8  # seconds
    estimated_cold_min = (total_runs * time_per_fold_cold) / 60

    # =========================================================================
    # PRE-RUN SUMMARY
    # =========================================================================
    w = 70
    print()
    print("=" * w)
    print("  THESIS EXPERIMENT: Fingerprint-Based Caching for ML")
    print("=" * w)

    print()
    print("  OBJECTIVE")
    print(f"    Evaluate caching speedup across {n_configs} feature selection configs")
    print(f"    using LOSO cross-validation on sleep stage classification.")

    print()
    print("  DATA")
    print(f"    Dataset:      BOAS ({len(subjects)} subjects)")
    print(f"    Features:     149 (6-channel EEG: time/freq/complexity)")
    print(f"    Classes:      5 (Wake, N1, N2, N3, REM)")
    print(f"    Feature cache: {cache_status['cached_subjects']}/{len(subjects)} subjects ({cache_metrics['storage_mb']:.1f} MB)")

    # Model cache info
    model_cache_dir = Path("results/loso_model_cache")
    if model_cache_dir.exists():
        n_cached_models = len(list(model_cache_dir.glob('*.joblib')))
        print(f"    Model cache:   {n_cached_models} cached models")
    else:
        print(f"    Model cache:   empty (cold start)")

    print()
    print(f"  EXPERIMENT GRID ({grid_name}: {n_configs} configurations)")
    print(f"    Models:       {', '.join(m.replace('_', ' ').title() for m in models)}")

    corr_strs = [str(c) if c else 'None' for c in correlation_thresholds]
    topk_strs = [str(k) if k else 'All' for k in top_k_features]
    print(f"    Correlation:  [{', '.join(corr_strs)}]")
    print(f"    Top-K:        [{', '.join(topk_strs)}]")
    print(f"    Selection:    ANOVA (global scope)")

    print()
    print("  EXECUTION")
    print(f"    CV Method:    Leave-One-Subject-Out ({n_folds} folds/config)")
    print(f"    Total runs:   {total_runs:,}")
    print(f"    Est. time:    ~{estimated_cold_min:.0f} min (cold) | ~2 min (cached)")
    print(f"    Output:       {args.output_dir}/{exp_name}")

    print()
    print("=" * w)

    # Confirmation
    if not args.yes:
        response = input("\nProceed? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
    
    # Run training
    try:
        output_dir = Path(args.output_dir)
        
        results = run_training_experiment(
            subjects=subjects,
            models=models,
            correlation_thresholds=correlation_thresholds,
            top_k_features=top_k_features,
            output_dir=output_dir,
            experiment_name=exp_name,
            generate_viz=not args.no_viz,
            use_hybrid=not args.pure_mi,
            n_jobs=args.n_jobs,
            cache_min_free_gb=args.cache_min_free_gb,
            cache_max_size_gb=args.cache_max_size_gb
        )
        
        # Generate cache-focused outputs (THESIS FOCUS)
        exp_dir = output_dir / exp_name
        print("\n[5/5] Generating cache-focused thesis outputs...")
        
        # Update cache metrics with actual run data
        cache_metrics['total_runs'] = 1  # This run
        cache_metrics['warm_time_seconds'] = results['elapsed_seconds']
        
        # Generate cache visualizations
        experiment_runs = [
            {'run_id': 'This Run', 'hit_rate': cache_metrics['hit_rate'], 
             'duration_seconds': results['elapsed_seconds']}
        ]
        
        cache_files = generate_all_cache_figures(
            cache_metrics, 
            experiment_runs, 
            exp_dir
        )
        print(f"  [v] Generated {len(cache_files)} cache visualization files")
        
        # Generate LaTeX tables if we have results
        if results.get('results'):
            latex_dir = exp_dir / "latex"
            latex_dir.mkdir(exist_ok=True)
            
            generate_results_latex_table(
                results['results'],
                latex_dir / "results_table.tex"
            )
            print(f"  [v] Generated LaTeX results table")
        
        # Save cache metrics JSON
        cache_metrics_path = exp_dir / "cache_metrics.json"
        import json
        with open(cache_metrics_path, 'w') as f:
            json.dump(cache_metrics, f, indent=2, default=str)
        print(f"  [v] Saved cache metrics to {cache_metrics_path}")
        
        # Final summary with cache focus
        print("\n" + "=" * 70)
        print("CACHE PERFORMANCE SUMMARY (THESIS FOCUS)")
        print("=" * 70)
        print(f"Cache Hit Rate: {cache_metrics['hit_rate']*100:.1f}% ({cache_metrics['n_hits']}/{cache_metrics['n_total']} subjects)")
        print(f"Cold Start Time (estimated): ~{cache_metrics['cold_time_seconds']/60:.1f} minutes  (based on ~25s/subject)")
        print(f"Warm Start Time (measured):   {results['elapsed_seconds']/60:.1f} minutes")
        print(f"Speedup Factor: {cache_metrics['cold_time_seconds']/results['elapsed_seconds']:.0f}×  (estimated)")
        print(f"Time Saved: ~{(cache_metrics['cold_time_seconds']-results['elapsed_seconds'])/60:.1f} minutes  (estimated)")
        print("=" * 70)
        
        logger.info("Training experiment completed successfully")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n[X] Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
