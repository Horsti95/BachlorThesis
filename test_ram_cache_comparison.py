"""
RAM vs SSD Cache Performance Comparison
========================================

Quick comparison of:
1. SSD Cache (current): Load each fold model from disk via joblib
2. RAM Cache (proposed): Preload all models into RAM, access from memory

Tests on first N folds to approximate full 128-fold speedup.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path("results/loso_model_cache")
N_TEST_FOLDS = 10  # Quick test on first 10 folds


class RAMModelCache:
    """
    Variant of LOSOModelCache that preloads all models into RAM.
    """
    
    def __init__(self, cache_dir: str = "results/loso_model_cache"):
        """Load all cached models into RAM."""
        self.cache_dir = Path(cache_dir)
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.load_time = 0.0
        
        # Preload all models
        start = time.time()
        self._preload_models()
        self.load_time = time.time() - start
        
    def _preload_models(self):
        """Load all .joblib files from cache directory into RAM."""
        joblib_files = sorted(self.cache_dir.glob("*.joblib"))
        logger.info(f"🔄 Preloading {len(joblib_files)} models into RAM...")
        
        for i, cache_file in enumerate(joblib_files):
            try:
                model = joblib.load(cache_file)
                self.models[str(cache_file)] = model
                
                # Check for associated scaler
                scaler_file = Path(str(cache_file) + "_scaler.joblib")
                if scaler_file.exists():
                    scaler = joblib.load(scaler_file)
                    self.scalers[str(cache_file)] = scaler
                    
            except Exception as e:
                logger.warning(f"Failed to load {cache_file}: {e}")
        
        logger.info(f"✓ Loaded {len(self.models)} models into RAM in {self.load_time:.2f}s")
    
    def get(self, cache_path_str: str) -> Optional[Any]:
        """Retrieve model from RAM."""
        return self.models.get(cache_path_str)
    
    def get_scaler(self, cache_path_str: str) -> Optional[Any]:
        """Retrieve scaler from RAM."""
        return self.scalers.get(cache_path_str)


def benchmark_ssd_cache(n_folds: int = 10):
    """Benchmark reading models from SSD cache."""
    joblib_files = sorted(CACHE_DIR.glob("*.joblib"))[:n_folds]
    
    logger.info(f"\n📊 SSD Cache Benchmark ({len(joblib_files)} folds)")
    logger.info("=" * 50)
    
    times = []
    for i, cache_file in enumerate(joblib_files):
        start = time.time()
        model = joblib.load(cache_file)
        elapsed = time.time() - start
        times.append(elapsed)
        logger.info(f"  Fold {i+1:2d}: {elapsed*1000:.1f}ms")
    
    avg_time = np.mean(times)
    total_time = np.sum(times)
    logger.info(f"\nAverage per fold: {avg_time*1000:.1f}ms")
    logger.info(f"Total: {total_time:.2f}s")
    
    return avg_time, total_time


def benchmark_ram_cache(n_folds: int = 10):
    """Benchmark reading models from RAM cache."""
    logger.info(f"\n🚀 RAM Cache Benchmark ({n_folds} folds)")
    logger.info("=" * 50)
    
    # Preload phase
    logger.info("Phase 1: Preloading into RAM...")
    ram_cache = RAMModelCache(str(CACHE_DIR))
    preload_time = ram_cache.load_time
    
    # Access phase
    logger.info(f"\nPhase 2: Accessing from RAM ({n_folds} folds)...")
    joblib_files = sorted(CACHE_DIR.glob("*.joblib"))[:n_folds]
    
    times = []
    for i, cache_file in enumerate(joblib_files):
        start = time.time()
        model = ram_cache.get(str(cache_file))
        elapsed = time.time() - start
        times.append(elapsed)
        logger.info(f"  Fold {i+1:2d}: {elapsed*1000:.3f}ms")
    
    avg_time = np.mean(times)
    total_access_time = np.sum(times)
    total_with_preload = preload_time + total_access_time
    
    logger.info(f"\nPreload one-time cost: {preload_time:.2f}s")
    logger.info(f"Average access per fold: {avg_time*1000:.3f}ms")
    logger.info(f"Total access time: {total_access_time:.2f}s")
    logger.info(f"Total with preload: {total_with_preload:.2f}s")
    
    return avg_time, total_with_preload, preload_time, total_access_time


def extrapolate_to_full_training(ssd_avg: float, ram_avg: float, preload_time: float, n_test: int = 10, total_folds: int = 128):
    """Extrapolate from test sample to full 128-fold training."""
    logger.info(f"\n📈 Extrapolation to Full 128-Fold Training")
    logger.info("=" * 60)
    
    ssd_total = ssd_avg * total_folds
    ram_total = (ram_avg * total_folds) + preload_time
    
    speedup = ssd_total / ram_total if ram_total > 0 else 0
    time_saved = ssd_total - ram_total
    
    logger.info(f"Configuration: xgboost_corr0.75_k30_anova_glo (similar setup)")
    logger.info(f"\nSSD Cache (current):")
    logger.info(f"  Per fold:      {ssd_avg*1000:.1f}ms")
    logger.info(f"  {total_folds} folds:    {ssd_total:.2f}s")
    
    logger.info(f"\nRAM Cache (proposed):")
    logger.info(f"  Preload:       {preload_time:.2f}s (one-time)")
    logger.info(f"  Per fold:      {ram_avg*1000:.3f}ms")
    logger.info(f"  {total_folds} folds:    {ram_total:.2f}s (including preload)")
    
    logger.info(f"\nSpeedup: {speedup:.2f}x faster")
    logger.info(f"Time saved: {time_saved:.2f}s on full training")
    logger.info(f"Percentage improvement: {(1 - ram_total/ssd_total) * 100:.1f}%")
    
    return speedup, time_saved


def main():
    logger.info("=" * 65)
    logger.info("   RAM vs SSD Cache Performance Comparison")
    logger.info("   Testing on first 10 folds of 128-fold LOSO CV")
    logger.info("=" * 65)
    
    # Verify cache exists
    if not CACHE_DIR.exists():
        logger.error(f"❌ Cache directory not found: {CACHE_DIR}")
        return
    
    joblib_files = sorted(CACHE_DIR.glob("*.joblib"))
    if not joblib_files:
        logger.error(f"❌ No cached models found in {CACHE_DIR}")
        return
    
    logger.info(f"✓ Found {len(joblib_files)} cached models")
    
    # Run benchmarks
    ssd_avg, ssd_total = benchmark_ssd_cache(N_TEST_FOLDS)
    ram_avg, ram_total_with_preload, preload_time, ram_access_total = benchmark_ram_cache(N_TEST_FOLDS)
    
    # Extrapolate
    speedup, time_saved = extrapolate_to_full_training(ssd_avg, ram_avg, preload_time, N_TEST_FOLDS, 128)
    
    logger.info("\n" + "=" * 65)
    logger.info("✅ Benchmark Complete")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
