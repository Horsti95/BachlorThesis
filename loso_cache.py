"""
LOSO Model Cache - Layer 2 Intelligent Caching
==============================================

Implements fingerprint-based caching for trained LOSO fold models.
This is the core thesis contribution: intelligent caching with 
configuration-aware invalidation.

Cache Architecture:
    - Each trained model is stored with its fingerprint as key
    - Fingerprint includes held_out_subject → prevents data leakage
    - Changing ANY config parameter → automatic cache miss (invalidation)

Storage Format:
    - Models stored as .joblib files (efficient for sklearn/xgboost)
    - Filename: {fingerprint}_{held_out_subject}.joblib
    - Optional metadata in cache_registry.json

Performance Tracking:
    - Cache hits/misses tracked for thesis metrics
    - Time saved calculated from training time estimates

Author: Lennart Gorzel
Date: January 2026
Status: IMPLEMENTED
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import joblib

from fingerprint import LOSOFingerprint, generate_cache_key, __version__ as FINGERPRINT_VERSION

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """
    Tracks cache performance metrics for thesis evaluation.
    
    Attributes:
        hits: Number of cache hits (model loaded from cache)
        misses: Number of cache misses (model trained fresh)
        time_saved_seconds: Estimated time saved from cache hits
        time_spent_training: Actual time spent training on misses
    """
    hits: int = 0
    misses: int = 0
    time_saved_seconds: float = 0.0
    time_spent_training: float = 0.0
    
    @property
    def total(self) -> int:
        """Total cache lookups."""
        return self.hits + self.misses
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        if self.total == 0:
            return 0.0
        return self.hits / self.total
    
    @property
    def miss_rate(self) -> float:
        """Cache miss rate (0.0 to 1.0)."""
        return 1.0 - self.hit_rate
    
    def record_hit(self, estimated_training_time: float = 0.0):
        """
        Record a cache hit.

        Args:
            estimated_training_time: Actual or estimated time that would have been
                                    spent training. Use real time from cache registry
                                    when available (default: 0 = not tracked).
        """
        self.hits += 1
        self.time_saved_seconds += estimated_training_time
    
    def record_miss(self, actual_training_time: float = 0.0):
        """
        Record a cache miss.
        
        Args:
            actual_training_time: Actual time spent training the model
        """
        self.misses += 1
        self.time_spent_training += actual_training_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total': self.total,
            'hit_rate': round(self.hit_rate, 4),
            'miss_rate': round(self.miss_rate, 4),
            'time_saved_seconds': round(self.time_saved_seconds, 2),
            'time_spent_training': round(self.time_spent_training, 2),
            'time_saved_formatted': self._format_time(self.time_saved_seconds)
        }
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def __str__(self) -> str:
        return (
            f"CacheMetrics(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.1%}, time_saved={self._format_time(self.time_saved_seconds)})"
        )


@dataclass
class CachedModelInfo:
    """
    Metadata stored alongside cached models.
    
    Used for debugging, auditing, and cache management.
    """
    fingerprint: str
    held_out_subject: str
    model_type: str
    created_at: str
    code_version: str
    training_time_seconds: float
    file_size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LOSOModelCache:
    """
    Intelligent cache for trained LOSO fold models.
    
    Key Features:
    - Fingerprint-based cache keys (automatic invalidation on config change)
    - Held-out subject in key (prevents data leakage between folds)
    - Performance tracking for thesis metrics
    - Optional registry for cache inspection
    
    Usage:
        cache = LOSOModelCache(cache_dir="results/loso_model_cache")
        
        # In LOSO loop
        for fold in folds:
            fingerprint = LOSOFingerprint.generate(...)
            
            cached_model = cache.get(fingerprint, fold.test_subject)
            if cached_model is not None:
                model = cached_model  # CACHE HIT
            else:
                model = train_model(...)  # CACHE MISS
                cache.put(fingerprint, fold.test_subject, model, ...)
        
        # Get stats for thesis
        print(cache.metrics)
    
    Cache File Structure:
        cache_dir/
        ├── a3f2b8c1d4e5f6a7b8c9d0e1f2a3b4c5_sub-001.joblib
        ├── a3f2b8c1d4e5f6a7b8c9d0e1f2a3b4c5_sub-002.joblib
        └── cache_registry.json (optional metadata)
    """
    
    # File extension for cached models
    MODEL_EXTENSION = ".joblib"
    REGISTRY_FILENAME = "cache_registry.json"
    
    def __init__(
        self,
        cache_dir: str = "results/loso_model_cache",
        enable_registry: bool = True,
        estimated_training_time: float = 120.0,
        min_free_space_gb: float = 5.0,
        max_cache_size_gb: Optional[float] = None
    ):
        """
        Initialize LOSO model cache.

        Args:
            cache_dir: Directory to store cached models
            enable_registry: Whether to maintain a JSON registry of cached models
            estimated_training_time: Default estimated training time per fold (seconds)
                                    Used for time_saved calculations
            min_free_space_gb: Minimum free disk space (GB) required to cache new models.
                              If free space drops below this, caching is skipped. Default: 5 GB.
            max_cache_size_gb: Maximum total cache size (GB). When exceeded, oldest models
                              are evicted (LRU) until under limit. None = unlimited.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.enable_registry = enable_registry
        self.estimated_training_time = estimated_training_time
        self.min_free_space_gb = min_free_space_gb
        self.max_cache_size_gb = max_cache_size_gb

        # Track how many cache writes were skipped due to space constraints
        self._space_skips = 0

        # Performance metrics
        self.metrics = CacheMetrics()

        # Load existing registry if available
        self._registry: Dict[str, CachedModelInfo] = {}
        if enable_registry:
            self._load_registry()

        logger.info(f"LOSOModelCache initialized at {self.cache_dir}")
        logger.info(f"  Existing cached models: {len(list(self.cache_dir.glob('*.joblib')))}")
        logger.info(f"  Min free space: {min_free_space_gb:.1f} GB, Max cache size: {max_cache_size_gb or 'unlimited'}")
    
    def _get_cache_path(self, fingerprint: str, held_out_subject: str) -> Path:
        """
        Get the file path for a cached model.
        
        Format: {fingerprint}_{held_out_subject}.joblib
        """
        # Sanitize subject ID for filename
        safe_subject = str(held_out_subject).replace("/", "_").replace("\\", "_")
        filename = f"{fingerprint}_{safe_subject}{self.MODEL_EXTENSION}"
        return self.cache_dir / filename
    
    def _get_registry_path(self) -> Path:
        """Get path to the cache registry JSON file."""
        return self.cache_dir / self.REGISTRY_FILENAME
    
    def _load_registry(self):
        """Load cache registry from disk."""
        registry_path = self._get_registry_path()
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    data = json.load(f)
                self._registry = {
                    k: CachedModelInfo(**v) for k, v in data.items()
                }
                logger.debug(f"Loaded registry with {len(self._registry)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache registry: {e}")
                self._registry = {}
    
    def _save_registry(self):
        """Save cache registry to disk."""
        if not self.enable_registry:
            return
        
        registry_path = self._get_registry_path()
        try:
            data = {k: v.to_dict() for k, v in self._registry.items()}
            with open(registry_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache registry: {e}")
    
    def exists(self, fingerprint: str, held_out_subject: str) -> bool:
        """
        Check if a cached model exists.
        
        Args:
            fingerprint: Configuration fingerprint
            held_out_subject: Subject ID held out for testing
            
        Returns:
            True if cached model exists, False otherwise
        """
        cache_path = self._get_cache_path(fingerprint, held_out_subject)
        return cache_path.exists()
    
    def get(
        self,
        fingerprint: str,
        held_out_subject: str,
        model_type: str = "unknown",
        model_class: Any = None,
        model_params: dict = None,
        record_metrics: bool = True
    ) -> Optional[Any]:
        """
        Get a cached model if it exists.
        Args:
            fingerprint: Configuration fingerprint
            held_out_subject: Subject ID held out for testing
            model_type: Model type string (e.g., 'fnn')
            model_class: Class to instantiate for FNN (required for FNN)
            model_params: Parameters for model_class (required for FNN)
            record_metrics: Whether to record hit/miss in metrics
        Returns:
            Cached model object, or None if not found (cache miss)
        """
        cache_path = self._get_cache_path(fingerprint, held_out_subject)
        if not cache_path.exists():
            logger.debug(f"Cache MISS: {fingerprint[:16]}..._{held_out_subject}")
            return None
        try:
            # Look up actual training time from registry (real data, not estimate)
            cache_key = f"{fingerprint}_{held_out_subject}"
            actual_time = self.estimated_training_time
            if cache_key in self._registry:
                actual_time = self._registry[cache_key].training_time_seconds

            if model_type == "fnn":
                import torch
                # For FNN, load state_dict and scaler
                state = torch.load(str(cache_path) + ".pt", map_location="cpu")
                scaler_path = str(cache_path) + "_scaler.joblib"
                if model_class is None or model_params is None:
                    logger.error("FNN cache get: model_class and model_params required.")
                    return None
                model = model_class(model_params)
                model.model.load_state_dict(state["model_state"])
                model.scaler = joblib.load(scaler_path)
                model.is_fitted = True
                if record_metrics:
                    self.metrics.record_hit(actual_time)
                logger.debug(f"Cache HIT (FNN): {fingerprint[:16]}..._{held_out_subject}")
                return model
            else:
                model = joblib.load(cache_path)
                if record_metrics:
                    self.metrics.record_hit(actual_time)
                logger.debug(f"Cache HIT: {fingerprint[:16]}..._{held_out_subject}")
                return model
        except Exception as e:
            logger.warning(f"Failed to load cached model: {e}")
            return None
    
    def _get_free_space_gb(self) -> float:
        """Get free disk space on the cache directory's filesystem in GB."""
        try:
            usage = shutil.disk_usage(self.cache_dir)
            return usage.free / (1024 ** 3)
        except OSError:
            return float('inf')  # If we can't check, don't block caching

    def _get_cache_size_bytes(self) -> int:
        """Get total size of all cached model files in bytes."""
        total = 0
        for ext in [self.MODEL_EXTENSION, ".pt", "_scaler.joblib"]:
            for f in self.cache_dir.glob(f"*{ext}"):
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total

    def _evict_oldest(self, target_bytes: int) -> int:
        """
        Evict oldest cached models (LRU) until target_bytes of space is freed.

        Uses file modification time as the age indicator.

        Returns:
            Number of models evicted.
        """
        # Collect all joblib cache files with their mtime
        cache_files = []
        for f in self.cache_dir.glob(f"*{self.MODEL_EXTENSION}"):
            if f.name == self.REGISTRY_FILENAME:
                continue
            try:
                cache_files.append((f, f.stat().st_mtime, f.stat().st_size))
            except OSError:
                continue

        # Also collect .pt files for FNN models
        for f in self.cache_dir.glob("*.pt"):
            try:
                cache_files.append((f, f.stat().st_mtime, f.stat().st_size))
            except OSError:
                continue
        for f in self.cache_dir.glob("*_scaler.joblib"):
            try:
                cache_files.append((f, f.stat().st_mtime, f.stat().st_size))
            except OSError:
                continue

        # Sort by mtime ascending (oldest first)
        cache_files.sort(key=lambda x: x[1])

        freed = 0
        evicted = 0
        for filepath, _, size in cache_files:
            if freed >= target_bytes:
                break
            try:
                filepath.unlink()
                freed += size
                evicted += 1

                # Remove from registry
                stem = filepath.stem
                # Try to find matching registry entry
                keys_to_remove = [k for k in self._registry if stem.startswith(k.rsplit("_", 1)[0])]
                for k in keys_to_remove:
                    if k in self._registry:
                        del self._registry[k]
            except OSError as e:
                logger.warning(f"Failed to evict {filepath}: {e}")

        if evicted > 0:
            logger.info(f"Cache eviction: removed {evicted} files, freed {freed / (1024**2):.1f} MB")
            if self.enable_registry:
                self._save_registry()

        return evicted

    def _check_space_and_evict(self) -> bool:
        """
        Check disk space constraints and evict if needed.

        Returns:
            True if caching is allowed, False if it should be skipped.
        """
        # Check minimum free disk space
        free_gb = self._get_free_space_gb()
        if free_gb < self.min_free_space_gb:
            if self._space_skips == 0:
                logger.warning(
                    f"Disk space low: {free_gb:.1f} GB free (minimum: {self.min_free_space_gb:.1f} GB). "
                    f"Skipping model caching to prevent disk-full errors."
                )
            self._space_skips += 1
            return False

        # Check max cache size and evict if needed
        if self.max_cache_size_gb is not None:
            cache_size_bytes = self._get_cache_size_bytes()
            max_bytes = self.max_cache_size_gb * (1024 ** 3)
            if cache_size_bytes > max_bytes:
                overshoot = cache_size_bytes - max_bytes
                # Evict at least 10% of max to avoid evicting every single put()
                evict_target = max(overshoot, int(max_bytes * 0.1))
                self._evict_oldest(evict_target)

        return True

    def put(
        self,
        fingerprint: str,
        held_out_subject: str,
        model: Any,
        model_type: str = "unknown",
        training_time: float = 0.0,
        record_metrics: bool = True
    ) -> bool:
        """
        Store a trained model in the cache.
        Args:
            fingerprint: Configuration fingerprint
            held_out_subject: Subject ID held out for testing
            model: Trained model object
            model_type: Model type name for registry
            training_time: Time spent training (for metrics)
            record_metrics: Whether to record miss in metrics
        Returns:
            True if successfully cached, False otherwise
        """
        # Check disk space constraints before writing
        if not self._check_space_and_evict():
            if record_metrics:
                self.metrics.record_miss(training_time)
            return False

        cache_path = self._get_cache_path(fingerprint, held_out_subject)
        try:
            if model_type == "fnn":
                import torch
                # Save FNN model state_dict and scaler separately
                torch.save({"model_state": model.model.state_dict()}, str(cache_path) + ".pt")
                joblib.dump(model.scaler, str(cache_path) + "_scaler.joblib", compress=3)
            else:
                joblib.dump(model, cache_path, compress=3)
            # Record metrics
            if record_metrics:
                self.metrics.record_miss(training_time)
            # Update registry
            if self.enable_registry:
                cache_key = f"{fingerprint}_{held_out_subject}"
                file_size = 0
                if model_type == "fnn":
                    pt_path = str(cache_path) + ".pt"
                    scaler_path = str(cache_path) + "_scaler.joblib"
                    file_size = sum([Path(pt_path).stat().st_size, Path(scaler_path).stat().st_size])
                else:
                    file_size = cache_path.stat().st_size
                self._registry[cache_key] = CachedModelInfo(
                    fingerprint=fingerprint,
                    held_out_subject=held_out_subject,
                    model_type=model_type,
                    created_at=datetime.now().isoformat(),
                    code_version=FINGERPRINT_VERSION,
                    training_time_seconds=training_time,
                    file_size_bytes=file_size
                )
                self._save_registry()
            logger.debug(f"Cached model: {fingerprint[:16]}..._{held_out_subject}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache model: {e}")
            return False
    
    def invalidate(
        self,
        fingerprint: Optional[str] = None,
        held_out_subject: Optional[str] = None
    ) -> int:
        """
        Invalidate (delete) cached models.
        
        Args:
            fingerprint: If provided, invalidate only models with this fingerprint
            held_out_subject: If provided, invalidate only models for this subject
            If both None, invalidates ALL cached models
            
        Returns:
            Number of models invalidated
        """
        count = 0
        
        for cache_file in self.cache_dir.glob(f"*{self.MODEL_EXTENSION}"):
            filename = cache_file.stem  # Without extension
            
            # Parse filename: {fingerprint}_{subject}
            parts = filename.rsplit("_", 1)
            if len(parts) != 2:
                continue
            
            file_fp, file_subject = parts
            
            # Check if should invalidate
            should_invalidate = True
            if fingerprint is not None and file_fp != fingerprint:
                should_invalidate = False
            if held_out_subject is not None and file_subject != str(held_out_subject):
                should_invalidate = False
            
            if should_invalidate:
                try:
                    cache_file.unlink()
                    count += 1
                    
                    # Remove from registry
                    cache_key = f"{file_fp}_{file_subject}"
                    if cache_key in self._registry:
                        del self._registry[cache_key]
                except Exception as e:
                    logger.warning(f"Failed to delete {cache_file}: {e}")
        
        if self.enable_registry:
            self._save_registry()
        
        logger.info(f"Invalidated {count} cached models")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache metrics and storage info
        """
        # Count cached files
        cached_files = list(self.cache_dir.glob(f"*{self.MODEL_EXTENSION}"))
        total_size = sum(f.stat().st_size for f in cached_files)
        
        free_gb = self._get_free_space_gb()
        return {
            'metrics': self.metrics.to_dict(),
            'storage': {
                'cached_models': len(cached_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'avg_size_mb': round(total_size / (1024 * 1024 * max(len(cached_files), 1)), 2),
                'cache_dir': str(self.cache_dir),
                'free_space_gb': round(free_gb, 2),
                'max_cache_size_gb': self.max_cache_size_gb,
                'space_skips': self._space_skips
            },
            'session': {
                'hits': self.metrics.hits,
                'misses': self.metrics.misses,
                'hit_rate': f"{self.metrics.hit_rate:.1%}",
                'time_saved': self.metrics._format_time(self.metrics.time_saved_seconds)
            }
        }
    
    def reset_metrics(self):
        """Reset session metrics (does not delete cached files)."""
        self.metrics = CacheMetrics()
    
    def __repr__(self) -> str:
        cached_count = len(list(self.cache_dir.glob(f"*{self.MODEL_EXTENSION}")))
        return f"LOSOModelCache(dir='{self.cache_dir}', cached={cached_count}, {self.metrics})"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_loso_cache(
    cache_dir: str = "results/loso_model_cache",
    estimated_training_time: float = 120.0
) -> LOSOModelCache:
    """
    Factory function to create a LOSO model cache.
    
    Args:
        cache_dir: Directory for cached models
        estimated_training_time: Estimated seconds per fold training
        
    Returns:
        Configured LOSOModelCache instance
    """
    return LOSOModelCache(
        cache_dir=cache_dir,
        enable_registry=True,
        estimated_training_time=estimated_training_time
    )


# ============================================================================
# QUICK TEST / DEMO
# ============================================================================

if __name__ == "__main__":
    import tempfile
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    print("=" * 60)
    print("LOSOModelCache Demo")
    print("=" * 60)
    
    # Create temporary cache directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = LOSOModelCache(cache_dir=tmpdir, estimated_training_time=60.0)
        
        # Simulate training and caching
        print("\n--- First Run (All Cache Misses) ---")
        
        for subject_id in ["sub-001", "sub-002", "sub-003"]:
            fingerprint = LOSOFingerprint.generate(
                random_seed=42,
                code_version="v1.0",
                model_name="rf",
                model_params={"n_estimators": 100},
                feature_config={"base": 149, "corr": 0.85},
                held_out_subject=subject_id
            )
            
            # Check cache
            cached_model = cache.get(fingerprint, subject_id)
            
            if cached_model is not None:
                print(f"  {subject_id}: CACHE HIT")
                model = cached_model
            else:
                print(f"  {subject_id}: CACHE MISS - training...")
                # Simulate training
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                model.fit(np.random.randn(100, 10), np.random.randint(0, 5, 100))
                
                # Cache the model
                cache.put(fingerprint, subject_id, model, 
                         model_type="rf", training_time=60.0)
        
        print(f"\nAfter first run: {cache.metrics}")
        
        # Second run - should be all cache hits
        print("\n--- Second Run (All Cache Hits) ---")
        cache.reset_metrics()
        
        for subject_id in ["sub-001", "sub-002", "sub-003"]:
            fingerprint = LOSOFingerprint.generate(
                random_seed=42,
                code_version="v1.0",
                model_name="rf",
                model_params={"n_estimators": 100},
                feature_config={"base": 149, "corr": 0.85},
                held_out_subject=subject_id
            )
            
            cached_model = cache.get(fingerprint, subject_id)
            
            if cached_model is not None:
                print(f"  {subject_id}: CACHE HIT ✓")
            else:
                print(f"  {subject_id}: CACHE MISS ✗")
        
        print(f"\nAfter second run: {cache.metrics}")
        
        # Different config - should be cache miss
        print("\n--- Third Run (Different Config - Cache Misses) ---")
        cache.reset_metrics()
        
        fingerprint_new = LOSOFingerprint.generate(
            random_seed=42,
            code_version="v1.0",
            model_name="xgboost",  # Different model!
            model_params={"max_depth": 6},
            feature_config={"base": 149, "corr": 0.85},
            held_out_subject="sub-001"
        )
        
        cached_model = cache.get(fingerprint_new, "sub-001")
        if cached_model is None:
            print("  sub-001 (xgboost): CACHE MISS ✓ (correct - different config)")
        
        # Print final stats
        print("\n--- Cache Statistics ---")
        stats = cache.get_stats()
        print(f"  Cached models: {stats['storage']['cached_models']}")
        print(f"  Total size: {stats['storage']['total_size_mb']} MB")
        
    print("\n" + "=" * 60)
    print("LOSOModelCache demo complete!")
    print("=" * 60)
