"""
Comprehensive Cache Strategy Benchmark
========================================

Runs 4 model configurations across 3 caching strategies:
- Cold: Cache cleared before run
- Warm SSD: Using existing disk cache
- Warm RAM: Preload cache to RAM

Configuration combinations:
- XGBoost strict   (corr 0.75, k=30)
- XGBoost all      (corr None, k=all)
- RF strict        (corr 0.75, k=30)
- RF all           (corr None, k=all)

Each configuration runs on all 128 LOSO folds with timing tracking.

Results saved to JSON with timestamps and performance statistics.
"""

import json
import logging
import shutil
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from feature_cache import load_features_from_cache
from training import TrainingPipeline, create_training_grid, TrainingConfig

# Constants
GLOBAL_CACHE_DIR = Path("./results/features_cache_global")
MODEL_CACHE_DIR = Path("./results/loso_model_cache")
RESULTS_DIR = Path("./results")


def _subject_sort_key(cache_file: Path) -> int:
    """Sort subject cache files numerically by subject ID."""
    subject_id = cache_file.stem.replace("subject_", "").replace("_full", "")
    try:
        return int(subject_id)
    except ValueError:
        return 10**9


def load_features_safe() -> tuple:
    """Load cached features from all subjects (all 128)."""
    try:
        logger.info("[CACHE] Loading cached features from all 128 subjects...")
        
        all_features = []
        all_labels = []
        all_subject_ids = []
        
        # Find all cached subjects
        subject_files = sorted(
            GLOBAL_CACHE_DIR.glob("subject_*_full.npz"),
            key=_subject_sort_key
        )
        n_subjects = len(subject_files)
        
        if n_subjects == 0:
            logger.error("[ERROR] No cached subjects found!")
            raise FileNotFoundError(f"No .npz files in {GLOBAL_CACHE_DIR}")
        
        for cache_file in subject_files:
            # Extract subject ID from filename (e.g., "subject_1_full.npz" -> "1")
            subject_id = cache_file.stem.replace("subject_", "").replace("_full", "")
            
            cached = load_features_from_cache(cache_file)
            if cached is None:
                logger.warning(f"  [WARN] Failed to load {cache_file.name}")
                continue
            
            features_df, labels, n_channels = cached
            all_features.append(features_df)
            all_labels.append(labels)
            all_subject_ids.extend([f"Subject_{subject_id}"] * len(labels))
        
        # Combine all
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_labels = np.concatenate(all_labels)
        combined_subject_ids = np.array(all_subject_ids)
        
        logger.info(f"[OK] Loaded {combined_features.shape[0]} epochs from {n_subjects} subjects")
        logger.info(f"  Features shape: {combined_features.shape}")
        return combined_features, combined_labels, combined_subject_ids
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to load features: {e}")
        raise


def load_features_subset(max_subjects: int = 10) -> tuple:
    """Load cached features from a subset of subjects for quick benchmarking."""
    try:
        if max_subjects <= 0:
            raise ValueError("max_subjects must be > 0")

        logger.info(f"[CACHE] Loading cached features from first {max_subjects} subjects...")

        all_features = []
        all_labels = []
        all_subject_ids = []

        subject_files = sorted(
            GLOBAL_CACHE_DIR.glob("subject_*_full.npz"),
            key=_subject_sort_key
        )[:max_subjects]

        n_subjects = len(subject_files)
        if n_subjects == 0:
            logger.error("[ERROR] No cached subjects found!")
            raise FileNotFoundError(f"No .npz files in {GLOBAL_CACHE_DIR}")

        for cache_file in subject_files:
            subject_id = cache_file.stem.replace("subject_", "").replace("_full", "")

            cached = load_features_from_cache(cache_file)
            if cached is None:
                logger.warning(f"  [WARN] Failed to load {cache_file.name}")
                continue

            features_df, labels, n_channels = cached
            all_features.append(features_df)
            all_labels.append(labels)
            all_subject_ids.extend([f"Subject_{subject_id}"] * len(labels))

        combined_features = pd.concat(all_features, ignore_index=True)
        combined_labels = np.concatenate(all_labels)
        combined_subject_ids = np.array(all_subject_ids)

        logger.info(
            f"[OK] Loaded {combined_features.shape[0]} epochs from {n_subjects} subjects (subset mode)"
        )
        logger.info(f"  Features shape: {combined_features.shape}")
        return combined_features, combined_labels, combined_subject_ids

    except Exception as e:
        logger.error(f"[ERROR] Failed to load subset features: {e}")
        raise


def load_single_subject_features(subject_id: int = 1) -> tuple:
    """Quick load: cached features from a single subject (default: subject_1)."""
    try:
        logger.info(f"[CACHE] Loading cached features from subject_{subject_id}...")
        
        cache_file = GLOBAL_CACHE_DIR / f"subject_{subject_id}_full.npz"
        
        if not cache_file.exists():
            logger.error(f"[ERROR] Cache file not found: {cache_file}")
            raise FileNotFoundError(f"No cache for subject_{subject_id}")
        
        cached = load_features_from_cache(cache_file)
        if cached is None:
            raise ValueError(f"Failed to load {cache_file.name}")
        
        features_df, labels, n_channels = cached
        subject_ids = np.array([f"Subject_{subject_id}"] * len(labels))
        
        logger.info(f"[OK] Loaded {len(labels)} epochs from subject_{subject_id}")
        logger.info(f"  Features shape: {features_df.shape}")
        return features_df, labels, subject_ids
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to load features: {e}")
        raise


class RAMModelCache:
    """In-memory model cache for quick access during training."""
    
    def __init__(self, cache_dir: str = "results/loso_model_cache"):
        """Preload all cached models into RAM."""
        self.cache_dir = Path(cache_dir)
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.load_time = 0.0
        
        start = time.time()
        self._preload_models()
        self.load_time = time.time() - start
        
    def _preload_models(self):
        """Load all .joblib files into RAM."""
        import joblib
        # Exclude scaler sidecar files (only preload model files)
        joblib_files = sorted(
            p for p in self.cache_dir.glob("*.joblib")
            if not p.name.endswith("_scaler.joblib")
        )
        
        for cache_file in joblib_files:
            try:
                model = joblib.load(cache_file)
                self.models[str(cache_file)] = model
                
                scaler_file = Path(str(cache_file) + "_scaler.joblib")
                if scaler_file.exists():
                    scaler = joblib.load(scaler_file)
                    self.scalers[str(cache_file)] = scaler
            except Exception as e:
                logger.warning(f"Failed to load {cache_file.name}: {e}")
    
    def get(self, cache_path_str: str):
        """Retrieve model from RAM."""
        return self.models.get(cache_path_str)
    
    def get_scaler(self, cache_path_str: str):
        """Retrieve scaler from RAM."""
        return self.scalers.get(cache_path_str)


def get_test_configurations(max_configs: int = 4) -> List[TrainingConfig]:
    """
    Get test configurations: 2 XGB + 2 RF (or just 1 of each if max_configs=2, etc).
    
    Args:
        max_configs: Maximum number of configs to return (default 4 = 2 XGB + 2 RF)
    """
    
    # Create grid for all combinations
    full_grid = create_training_grid(
        models=['xgboost', 'random_forest'],
        correlation_thresholds=[0.75, None],  # strict and all
        top_k_features=[30, None],            # strict and all
        random_state=42,
        use_hybrid=True,
        selection_method='anova',
        scope='global'
    )
    
    # Filter to get desired number of configs
    test_configs = []
    xgb_count = 0
    rf_count = 0
    max_per_model = max_configs // 2
    
    for config in full_grid:
        corr = config.feature_selection.correlation_threshold
        k = config.feature_selection.top_k_features
        
        if config.model_type == 'xgboost' and xgb_count < max_per_model:
            # Pick first XGB (strict: corr0.75, k30) and another (all)
            if (corr == 0.75 and k == 30) or (corr is None and k is None):
                test_configs.append(config)
                xgb_count += 1
        
        elif config.model_type == 'random_forest' and rf_count < max_per_model:
            # Pick first RF (strict) and another (all)
            if (corr == 0.75 and k == 30) or (corr is None and k is None):
                test_configs.append(config)
                rf_count += 1
        
        if xgb_count == max_per_model and rf_count == max_per_model:
            break
    
    return sorted(test_configs, key=lambda c: (c.model_type, c.get_config_id()))


def run_configuration(
    config: TrainingConfig,
    features_df,
    labels,
    subject_ids,
    use_ram_cache: bool = False,
    enable_disk_cache: bool = True
) -> Dict[str, Any]:
    """
    Run a single configuration and track timing.
    
    Returns:
        {
            'config_name': str,
            'model': str,
            'features_selected': int,
            'total_time': float,
            'avg_fold_time': float,
            'num_folds': int,
            'cache_type': str  # 'none', 'disk', 'ram'
        }
    """
    cache_type = 'ram' if use_ram_cache else ('disk' if enable_disk_cache else 'none')
    
    config_name = config.get_config_id()
    logger.info(f"\n  Running {config_name}...")
    logger.info(f"  Cache mode: {cache_type}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"benchmark_{config_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training pipeline
    start_total = time.time()
    try:
        ram_preload_time = 0.0

        pipeline = TrainingPipeline(
            features_df=features_df,
            labels=labels,
            subject_ids=subject_ids,
            output_dir=output_dir,
            experiment_name=f"bench_{config_name}",
            enable_model_cache=enable_disk_cache,
            model_cache_dir=str(MODEL_CACHE_DIR)
        )
        
        # TEMPORARY: If using RAM, replace the cache with RAM version
        if use_ram_cache and pipeline.model_cache:
            logger.info(f"  Preloading cache to RAM...")
            ram_start = time.time()
            ram_cache = RAMModelCache(str(MODEL_CACHE_DIR))
            ram_preload_time = time.time() - ram_start
            
            # Monkey-patch cache get: prefer RAM, fallback to disk cache
            original_get = pipeline.model_cache.get

            def ram_first_get(fp, subject, **kwargs):
                cache_path = str(pipeline.model_cache._get_cache_path(fp, subject))
                cached_model = ram_cache.get(cache_path)
                if cached_model is not None:
                    return cached_model
                return original_get(fp, subject, **kwargs)

            pipeline.model_cache.get = ram_first_get
            logger.info(f"    RAM preload took {ram_preload_time:.2f}s")
        
        # Run the configuration
        result = pipeline.run_single_config(
            config=config,
            show_progress=True,
            config_idx=1,
            total_configs=1
        )
        
        fold_times = []
        if hasattr(result, 'fold_times') and result.fold_times:
            fold_times = result.fold_times
        
        raw_total_time = time.time() - start_total
        # For RAM mode, report runtime excluding preload so SSD vs RAM compares pure cache retrieval.
        measured_total_time = raw_total_time - ram_preload_time if use_ram_cache else raw_total_time
        if measured_total_time < 0:
            measured_total_time = 0.0

        avg_fold_time = measured_total_time / pipeline.n_folds if pipeline.n_folds > 0 else 0
        
        return {
            'config_name': config_name,
            'model': config.model_type,
            'correlation_threshold': config.feature_selection.correlation_threshold,
            'top_k_features': config.feature_selection.top_k_features,
            'features_selected': config.feature_selection.top_k_features or 149,
            'total_time': round(measured_total_time, 2),
            'raw_total_time': round(raw_total_time, 2),
            'ram_preload_time': round(ram_preload_time, 2),
            'avg_fold_time': round(avg_fold_time, 3),
            'num_folds': pipeline.n_folds,
            'cache_type': cache_type,
            'n_cached_models': len(list(MODEL_CACHE_DIR.glob('*.joblib'))),
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {
            'config_name': config_name,
            'model': config.model_type,
            'cache_type': cache_type,
            'status': 'error',
            'error': str(e),
            'total_time': time.time() - start_total
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark cold vs warm SSD vs warm RAM model caching")
    parser.add_argument(
        "--mini-subjects",
        type=int,
        default=0,
        help="Use only first N subjects for a fast mini benchmark (e.g., 5 or 10)."
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=2,
        help="Number of configs to benchmark (default 2 = RF strict + XGB strict)."
    )
    args = parser.parse_args()

    logger.info("\n" + "=" * 70)
    logger.info("  128-Subject LOSO Cache Strategy Benchmark")
    logger.info("=" * 70)
    logger.info(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not GLOBAL_CACHE_DIR.exists():
        logger.error(f"[ERROR] Feature cache not found at {GLOBAL_CACHE_DIR}")
        logger.error("   Run: python run_experiment.py --full")
        return
    
    # Load features once
    # For quick testing with just subject_1, use instead:
    # features_df, labels, subject_ids = load_single_subject_features(subject_id=1)
    if args.mini_subjects > 0:
        features_df, labels, subject_ids = load_features_subset(max_subjects=args.mini_subjects)
    else:
        features_df, labels, subject_ids = load_features_safe()

    n_subjects = len(np.unique(subject_ids))
    logger.info(f"  Active subjects for this run: {n_subjects}")
    
    # Get test configurations (max_configs=2 means just 1 XGB for quick test)
    logger.info(f"\n[CONFIG] Loading configurations...")
    test_configs = get_test_configurations(max_configs=args.max_configs)
    logger.info(f"[OK] Will test {len(test_configs)} configuration(s)\n")
    
    for config in test_configs:
        logger.info(f"  - {config.get_config_id()}")
    
    # Results container
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'total_subjects': int(n_subjects),
        'n_folds': int(n_subjects),
        'configurations': test_configs.__len__(),
        'mini_subjects': int(args.mini_subjects) if args.mini_subjects > 0 else None,
        'cache_strategies': ['cold', 'warm_ssd', 'warm_ram'],
        'results': []
    }
    
    # Run each configuration with all 3 cache strategies
    for idx, config in enumerate(test_configs, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"  [{idx}/{len(test_configs)}] Configuration: {config.get_config_id()}")
        logger.info(f"{'='*70}")
        
        config_results = {
            'config_name': config.get_config_id(),
            'model': config.model_type,
            'scenarios': {}
        }
        
        # 1. COLD RUN
        logger.info(f"\n  [COLD] COLD RUN (clearing cache)...")
        if MODEL_CACHE_DIR.exists():
            logger.info(f"    Clearing {MODEL_CACHE_DIR}...")
            # Try to delete, with retry for Windows file locking
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(MODEL_CACHE_DIR)
                    break
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"    Files locked, retrying... (attempt {attempt+1}/{max_retries})")
                        time.sleep(0.5)
                    else:
                        logger.warning(f"    Could not delete cache (files in use), proceeding anyway...")
                        # Remove files individually if directory delete fails
                        for f in MODEL_CACHE_DIR.glob("*"):
                            try:
                                f.unlink(missing_ok=True)
                            except:
                                pass
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        cold_result = run_configuration(
            config, features_df, labels, subject_ids,
            use_ram_cache=False,
            enable_disk_cache=True
        )
        config_results['scenarios']['cold'] = cold_result
        
        # 2. WARM SSD RUN
        logger.info(f"\n  [SSD] WARM SSD RUN (reusing disk cache)...")
        warm_ssd_result = run_configuration(
            config, features_df, labels, subject_ids,
            use_ram_cache=False,
            enable_disk_cache=True
        )
        config_results['scenarios']['warm_ssd'] = warm_ssd_result
        
        # 3. WARM RAM RUN
        logger.info(f"\n  [RAM] WARM RAM RUN (preload to RAM)...")
        warm_ram_result = run_configuration(
            config, features_df, labels, subject_ids,
            use_ram_cache=True,
            enable_disk_cache=True
        )
        config_results['scenarios']['warm_ram'] = warm_ram_result
        
        all_results['results'].append(config_results)
        
        # Print summary for this config
        logger.info(f"\n  [RESULTS] {config.get_config_id()}:")
        logger.info(f"    Cold:     {cold_result.get('total_time', 'ERROR')}s")
        logger.info(f"    Warm SSD: {warm_ssd_result.get('total_time', 'ERROR')}s")
        logger.info(f"    Warm RAM: {warm_ram_result.get('total_time', 'ERROR')}s")
        
        if all([
            cold_result.get('status') == 'success',
            warm_ssd_result.get('status') == 'success',
            warm_ram_result.get('status') == 'success'
        ]):
            cold_time = cold_result['total_time']
            warm_ssd_time = warm_ssd_result['total_time']
            warm_ram_time = warm_ram_result['total_time']
            
            logger.info(f"    SSD = {cold_time}/1.0x (baseline)")
            if warm_ssd_time > 0:
                logger.info(f"    SSD warm = {cold_time/warm_ssd_time:.2f}x faster")
            if warm_ram_time > 0:
                logger.info(f"    RAM = {cold_time/warm_ram_time:.2f}x faster")

            if warm_ram_result.get('ram_preload_time', 0) > 0:
                logger.info(
                    f"    RAM timing excludes preload ({warm_ram_result['ram_preload_time']:.2f}s)"
                )
    
    # Save results
    results_file = RESULTS_DIR / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    logger.info(f"\n[SAVE] Saving results to {results_file}...")
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"[OK] Results saved")
    
    # Print final summary
    logger.info(f"\n" + "=" * 70)
    logger.info(f"  BENCHMARK COMPLETE")
    logger.info(f"=" * 70)
    logger.info(f"  Results file: {results_file}")
    logger.info(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print quick summary table
    logger.info(f"\n  Quick Summary:")
    logger.info(f"  {'Config':<40} {'Cold':<10} {'Warm SSD':<10} {'Warm RAM':<10}")
    logger.info(f"  {'-'*70}")
    
    for result in all_results['results']:
        config_name = result['config_name'][:35]
        scenarios = result['scenarios']
        
        cold = scenarios['cold'].get('total_time', 'ERROR')
        warm_ssd = scenarios['warm_ssd'].get('total_time', 'ERROR')
        warm_ram = scenarios['warm_ram'].get('total_time', 'ERROR')
        
        logger.info(f"  {config_name:<40} {str(cold):<10} {str(warm_ssd):<10} {str(warm_ram):<10}")
    
    logger.info(f"\n[OK] Done!")


if __name__ == "__main__":
    main()
