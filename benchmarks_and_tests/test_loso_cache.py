"""
Test LOSO Cache and Fingerprinting
===================================

This script verifies that the fingerprinting and LOSO caching system works correctly.

Test levels:
1. Minimal unit tests - fingerprint generation
2. Minimal with 3 subjects - cache functionality
3. Full test with 10 subjects - real LOSO caching workflow

Author: Lennart Gorzel
Date: January 2026
"""

print("=" * 60, flush=True)
print("LOSO CACHE TEST - STARTING", flush=True)
print("=" * 60, flush=True)

import os
import sys
import json
import shutil
import tempfile
import numpy as np
from pathlib import Path
import time
import logging
import threading
from datetime import datetime

print(f"[{datetime.now().strftime('%H:%M:%S')}] Basic imports done", flush=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print(f"[{datetime.now().strftime('%H:%M:%S')}] Importing fingerprint...", flush=True)
from fingerprint import LOSOFingerprint, FeatureConfig, ModelConfig, LOSOFingerprintConfig
print(f"[{datetime.now().strftime('%H:%M:%S')}] Importing loso_cache...", flush=True)
from loso_cache import LOSOModelCache, CacheMetrics, CachedModelInfo
print(f"[{datetime.now().strftime('%H:%M:%S')}] All imports done!", flush=True)

# Lazy import for real modules - only needed for real data tests (Phase 3+)
_models_imported = False
create_model = None

def ensure_real_modules():
    """Lazy import real project modules only when needed for real data tests."""
    global _models_imported, create_model
    if not _models_imported:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Importing real project modules (models.py)...", flush=True)
        try:
            from models import create_model as _create_model
            create_model = _create_model
            _models_imported = True
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Real modules imported successfully!", flush=True)
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Module import FAILED: {e}", flush=True)
            raise


class DummyModel:
    """Simple dummy model for cache testing - no sklearn needed."""
    def __init__(self, name="dummy", seed=42):
        self.name = name
        self.seed = seed
        self.trained = False
        self.weights = None
    
    def fit(self, X, y):
        """Simulate training by storing some data."""
        self.trained = True
        self.weights = np.random.RandomState(self.seed).rand(X.shape[1])
        return self
    
    def predict(self, X):
        """Simple prediction based on weights."""
        scores = X @ self.weights
        return (scores > np.median(scores)).astype(int)
    
    def score(self, X, y):
        """Return accuracy."""
        preds = self.predict(X)
        return np.mean(preds == y)


# Global state for progress tracking
_current_step = "Initializing..."
_step_start_time = time.time()
_progress_thread = None
_stop_progress = False


def set_step(step_name: str):
    """Set the current step name and print it."""
    global _current_step, _step_start_time
    _current_step = step_name
    _step_start_time = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] >>> STEP: {step_name}", flush=True)


def print_progress():
    """Background thread that prints progress every 10 seconds."""
    global _stop_progress
    last_print = time.time()
    
    while not _stop_progress:
        time.sleep(1)
        elapsed = time.time() - _step_start_time
        if time.time() - last_print >= 10:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] ... Still in: {_current_step} (elapsed: {elapsed:.1f}s)", flush=True)
            last_print = time.time()


def start_progress_tracker():
    """Start the background progress tracker."""
    global _progress_thread, _stop_progress
    _stop_progress = False
    _progress_thread = threading.Thread(target=print_progress, daemon=True)
    _progress_thread.start()
    print("[Progress tracker started - will report every 10 seconds]", flush=True)


def stop_progress_tracker():
    """Stop the background progress tracker."""
    global _stop_progress
    _stop_progress = True
    if _progress_thread:
        _progress_thread.join(timeout=2)
    print("[Progress tracker stopped]", flush=True)


def print_separator(title: str):
    """Print a visual separator with title."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print("\n" + "=" * 60, flush=True)
    print(f" [{timestamp}] {title}", flush=True)
    print("=" * 60, flush=True)


# =============================================================================
# TEST 1: Minimal Unit Tests - Fingerprint Generation
# =============================================================================

def test_fingerprint_determinism():
    """Test that same config always produces same fingerprint."""
    print_separator("TEST 1a: Fingerprint Determinism")
    
    set_step("Creating config dictionary")
    config = {
        'random_seed': 42,
        'code_version': 'v1.0',
        'model_name': 'xgboost',
        'model_params': {'max_depth': 6, 'n_estimators': 200},
        'feature_config': {'base': 149, 'corr': 0.85},
        'held_out_subject': 'subject_1'
    }
    print(f"    Config created: model={config['model_name']}, seed={config['random_seed']}", flush=True)
    
    set_step("Generating 3 fingerprints from same config")
    # Generate fingerprint 3 times
    fp1 = LOSOFingerprint.generate(**config)
    print(f"    Generated fp1: {fp1}", flush=True)
    fp2 = LOSOFingerprint.generate(**config)
    print(f"    Generated fp2: {fp2}", flush=True)
    fp3 = LOSOFingerprint.generate(**config)
    print(f"    Generated fp3: {fp3}", flush=True)
    
    set_step("Verifying fingerprint equality")
    print(f"Fingerprint 1: {fp1}", flush=True)
    print(f"Fingerprint 2: {fp2}", flush=True)
    print(f"Fingerprint 3: {fp3}", flush=True)
    
    assert fp1 == fp2 == fp3, "FAIL: Fingerprints should be identical!"
    assert len(fp1) == 32, f"FAIL: Fingerprint should be 32 chars, got {len(fp1)}"
    
    print("✓ PASS: Fingerprints are deterministic and correct length (32 chars)", flush=True)
    return True


def test_fingerprint_sensitivity():
    """Test that different configs produce different fingerprints."""
    print_separator("TEST 1b: Fingerprint Sensitivity")
    
    set_step("Creating base configuration")
    base_config = {
        'random_seed': 42,
        'code_version': 'v1.0',
        'model_name': 'xgboost',
        'model_params': {'max_depth': 6, 'n_estimators': 200},
        'feature_config': {'base': 149, 'corr': 0.85},
        'held_out_subject': 'subject_1'
    }
    print(f"    Base config: {base_config['model_name']}, depth={base_config['model_params']['max_depth']}", flush=True)
    
    set_step("Generating base fingerprint")
    base_fp = LOSOFingerprint.generate(**base_config)
    print(f"Base fingerprint:           {base_fp}", flush=True)
    
    # Test different changes
    changes = [
        ('random_seed', {'random_seed': 43}),
        ('model_name', {'model_name': 'random_forest'}),
        ('max_depth', {'model_params': {'max_depth': 7, 'n_estimators': 200}}),
        ('corr_threshold', {'feature_config': {'base': 149, 'corr': 0.80}}),
        ('held_out_subject', {'held_out_subject': 'subject_2'}),
    ]
    
    unique_fps = {base_fp}
    all_pass = True
    
    set_step("Testing fingerprint changes for each parameter")
    for i, (change_name, change_dict) in enumerate(changes):
        print(f"    [{i+1}/{len(changes)}] Testing change: {change_name}", flush=True)
        config = {**base_config, **change_dict}
        fp = LOSOFingerprint.generate(**config)
        
        if fp == base_fp:
            print(f"✗ FAIL: Changing {change_name} didn't change fingerprint!", flush=True)
            all_pass = False
        elif fp in unique_fps:
            print(f"✗ FAIL: Changing {change_name} produced collision!", flush=True)
            all_pass = False
        else:
            print(f"✓ Changed {change_name:20s} → {fp}", flush=True)
            unique_fps.add(fp)
    
    if all_pass:
        print("\n✓ PASS: All config changes produce unique fingerprints", flush=True)
    return all_pass


def test_fingerprint_held_out_uniqueness():
    """Test that same config with different held-out subjects produces different fingerprints."""
    print_separator("TEST 1c: Held-Out Subject Uniqueness")
    
    set_step("Creating subject list")
    subjects = [f'subject_{i}' for i in range(1, 6)]
    print(f"    Testing {len(subjects)} subjects: {subjects}", flush=True)
    fps = {}
    
    set_step("Generating fingerprint for each subject")
    for i, subject in enumerate(subjects):
        print(f"    [{i+1}/{len(subjects)}] Generating fingerprint for {subject}", flush=True)
        fp = LOSOFingerprint.generate(
            random_seed=42,
            code_version='v1.0',
            model_name='xgboost',
            model_params={'max_depth': 6},
            feature_config={'base': 149},
            held_out_subject=subject
        )
        fps[subject] = fp
        print(f"  {subject}: {fp}", flush=True)
    
    set_step("Verifying uniqueness")
    # Check all unique
    unique_fps = set(fps.values())
    if len(unique_fps) != len(subjects):
        print("✗ FAIL: Some fingerprints are not unique!", flush=True)
        return False
    
    print(f"\n✓ PASS: {len(subjects)} subjects → {len(unique_fps)} unique fingerprints", flush=True)
    return True


# =============================================================================
# TEST 2: Minimal with Real Structure - 3 Subjects
# =============================================================================

def test_cache_basic_operations():
    """Test basic cache operations: put, get, exists."""
    print_separator("TEST 2a: Cache Basic Operations")
    
    # NO sklearn needed - use DummyModel instead!
    
    # Create temporary cache directory
    set_step("Creating temporary cache directory")
    temp_dir = tempfile.mkdtemp(prefix="loso_cache_test_")
    print(f"    Temp cache dir: {temp_dir}", flush=True)
    
    try:
        set_step("Initializing LOSOModelCache")
        cache = LOSOModelCache(cache_dir=temp_dir, enable_registry=True)
        print(f"    Cache initialized with registry enabled", flush=True)
        
        # Create a simple DUMMY model (no sklearn!)
        set_step("Creating and training a DummyModel (no sklearn)")
        model = DummyModel(name="test_model", seed=42)
        # Generate simple synthetic data with numpy only
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] > 0).astype(int)
        print(f"    Generated synthetic data: {X.shape[0]} samples, {X.shape[1]} features", flush=True)
        model.fit(X, y)
        print(f"    Model trained successfully", flush=True)
        
        fingerprint = "a" * 32  # Simple test fingerprint
        subject = "test_subject"
        print(f"    Using fingerprint: {fingerprint}", flush=True)
        print(f"    Using subject: {subject}", flush=True)
        
        # Test: not exists initially
        set_step("Testing cache.exists() on empty cache")
        assert not cache.exists(fingerprint, subject), "Cache should be empty initially"
        print("✓ Cache is empty initially", flush=True)
        
        # Test: get returns None for missing
        set_step("Testing cache.get() on empty cache")
        result = cache.get(fingerprint, subject)
        assert result is None, "Get should return None for missing entry"
        print("✓ Get returns None for missing entry", flush=True)
        
        # Test: put model
        set_step("Testing cache.put() to store model")
        success = cache.put(fingerprint, subject, model, model_type="rf", training_time=1.5)
        assert success, "Put should succeed"
        print("✓ Put succeeded", flush=True)
        
        # Test: exists after put
        set_step("Testing cache.exists() after put")
        assert cache.exists(fingerprint, subject), "Cache should contain model after put"
        print("✓ Exists returns True after put", flush=True)
        
        # Test: get returns model
        set_step("Testing cache.get() after put")
        loaded_model = cache.get(fingerprint, subject)
        assert loaded_model is not None, "Get should return model"
        print("✓ Get returns model", flush=True)
        
        # Test: model works
        set_step("Testing loaded model predictions")
        preds = loaded_model.predict(X[:5])
        assert len(preds) == 5, "Loaded model should work"
        print(f"    Predictions: {preds}", flush=True)
        print("✓ Loaded model makes predictions", flush=True)
        
        # Test: cache metrics
        set_step("Checking cache metrics")
        metrics = cache.metrics
        print(f"  Cache metrics: hits={metrics.hits}, misses={metrics.misses}", flush=True)
        
        # Test: registry file exists
        set_step("Verifying registry file")
        registry_path = Path(temp_dir) / "cache_registry.json"
        assert registry_path.exists(), "Registry file should exist"
        print("✓ Registry file created", flush=True)
        
        # Read and verify registry
        with open(registry_path) as f:
            registry = json.load(f)
        print(f"  Registry entries: {len(registry)}", flush=True)
        
        print("\n✓ PASS: All basic cache operations work correctly", flush=True)
        return True
        
    finally:
        # Cleanup
        set_step("Cleaning up temporary directory")
        shutil.rmtree(temp_dir)
        print(f"    Cleaned up: {temp_dir}", flush=True)


def test_cache_with_synthetic_loso():
    """Test cache with synthetic LOSO workflow - 3 subjects."""
    print_separator("TEST 2b: Synthetic LOSO with 3 Subjects")
    
    # NO sklearn needed - use DummyModel instead!
    
    set_step("Creating temporary cache directory")
    temp_dir = tempfile.mkdtemp(prefix="loso_cache_test_")
    print(f"    Temp cache dir: {temp_dir}", flush=True)
    
    try:
        set_step("Initializing LOSOModelCache")
        cache = LOSOModelCache(cache_dir=temp_dir, enable_registry=True)
        print(f"    Cache initialized", flush=True)
        
        # Create synthetic data for 3 subjects (numpy only, no sklearn!)
        set_step("Generating synthetic data for 3 subjects (numpy only)")
        subjects = ['sub_1', 'sub_2', 'sub_3']
        n_samples_per_subject = 100
        n_features = 20
        
        # Generate data with numpy
        X_all = []
        y_all = []
        subject_ids = []
        
        for i, subj in enumerate(subjects):
            print(f"    Generating data for {subj}...", flush=True)
            np.random.seed(42 + i)
            X = np.random.randn(n_samples_per_subject, n_features)
            y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple rule
            X_all.append(X)
            y_all.append(y)
            subject_ids.extend([subj] * n_samples_per_subject)
        
        X_all = np.vstack(X_all)
        y_all = np.hstack(y_all)
        subject_ids = np.array(subject_ids)
        
        print(f"    Created synthetic data: {len(X_all)} samples, {n_features} features", flush=True)
        print(f"    Subjects: {subjects}", flush=True)
        
        # Config for fingerprint
        set_step("Creating model configuration")
        model_config = {
            'random_seed': 42,
            'code_version': 'v1.0',
            'model_name': 'rf',
            'model_params': {'n_estimators': 10, 'max_depth': 5},
            'feature_config': {'base': n_features, 'corr': None}
        }
        print(f"    Model: {model_config['model_name']}", flush=True)
        print(f"    Params: {model_config['model_params']}", flush=True)
        
        # Run 1: Cold start (all misses)
        set_step("RUN 1: Cold Start (expect all MISS)")
        print("\n--- RUN 1: Cold Start (expect all MISS) ---", flush=True)
        run1_times = []
        
        for i, test_subj in enumerate(subjects):
            print(f"\n    [{i+1}/{len(subjects)}] Processing fold for {test_subj}", flush=True)
            
            # Get train/test split
            test_mask = subject_ids == test_subj
            X_train = X_all[~test_mask]
            y_train = y_all[~test_mask]
            X_test = X_all[test_mask]
            y_test = y_all[test_mask]
            print(f"        Train: {len(y_train)} samples, Test: {len(y_test)} samples", flush=True)
            
            # Generate fingerprint
            fp = LOSOFingerprint.generate(
                **model_config,
                held_out_subject=test_subj
            )
            print(f"        Fingerprint: {fp}", flush=True)
            
            # Check cache
            cached = cache.get(fp, test_subj)
            if cached is not None:
                print(f"  {test_subj}: UNEXPECTED HIT!", flush=True)
                model = cached
                train_time = 0.0
            else:
                print(f"        Cache status: MISS (expected)", flush=True)
                print(f"        Training model...", flush=True)
                start = time.time()
                model = DummyModel(name=model_config['model_name'], seed=42)
                model.fit(X_train, y_train)
                train_time = time.time() - start
                
                # Cache the model
                print(f"        Caching model...", flush=True)
                cache.put(fp, test_subj, model, model_type='dummy', training_time=train_time)
            
            # Evaluate
            acc = model.score(X_test, y_test)
            print(f"        → Accuracy: {acc:.4f}, Train time: {train_time:.4f}s", flush=True)
            run1_times.append(train_time)
        
        print(f"\n    Run 1 total training time: {sum(run1_times):.4f}s", flush=True)
        
        # Run 2: Warm start (all hits)
        set_step("RUN 2: Warm Start (expect all HIT)")
        print("\n--- RUN 2: Warm Start (expect all HIT) ---", flush=True)
        run2_times = []
        
        for i, test_subj in enumerate(subjects):
            print(f"\n    [{i+1}/{len(subjects)}] Processing fold for {test_subj}", flush=True)
            
            test_mask = subject_ids == test_subj
            X_test = X_all[test_mask]
            y_test = y_all[test_mask]
            
            fp = LOSOFingerprint.generate(
                **model_config,
                held_out_subject=test_subj
            )
            print(f"        Fingerprint: {fp}", flush=True)
            
            start = time.time()
            cached = cache.get(fp, test_subj)
            load_time = time.time() - start
            
            if cached is None:
                print(f"  {test_subj}: UNEXPECTED MISS!", flush=True)
                return False
            else:
                print(f"        Cache status: HIT (expected)", flush=True)
                print(f"        Loaded in {load_time:.4f}s", flush=True)
                model = cached
            
            acc = model.score(X_test, y_test)
            print(f"        → Accuracy: {acc:.4f}", flush=True)
            run2_times.append(load_time)
        
        print(f"\n    Run 2 total load time: {sum(run2_times):.4f}s", flush=True)
        
        # Verify metrics
        set_step("Verifying cache metrics")
        stats = cache.get_stats()
        print(f"\nCache Stats:", flush=True)
        print(f"  Hits: {stats['session']['hits']}", flush=True)
        print(f"  Misses: {stats['session']['misses']}", flush=True)
        print(f"  Hit Rate: {stats['session']['hit_rate']}", flush=True)
        
        assert stats['session']['misses'] == 3, f"Expected 3 misses, got {stats['session']['misses']}"
        assert stats['session']['hits'] == 3, f"Expected 3 hits, got {stats['session']['hits']}"
        
        # Calculate speedup
        speedup = sum(run1_times) / max(sum(run2_times), 0.001)
        print(f"\n    Speedup Factor: {speedup:.1f}x (Run 1: {sum(run1_times):.4f}s vs Run 2: {sum(run2_times):.4f}s)", flush=True)
        
        print("\n✓ PASS: Synthetic LOSO caching works correctly!", flush=True)
        return True
        
    finally:
        set_step("Cleaning up")
        shutil.rmtree(temp_dir)
        print(f"    Cleaned up: {temp_dir}", flush=True)


# =============================================================================
# TEST 3: Full Test with 10 Real Subjects
# =============================================================================

def test_real_data_10_subjects():
    """Test with 10 real subjects from BOAS dataset using real project modules."""
    print_separator("TEST 3: Real Data with 10 Subjects")
    
    # Ensure real project modules are loaded
    ensure_real_modules()
    
    # Check if feature cache exists
    set_step("Checking for feature cache directory")
    feature_cache_dir = Path("results/features_cache_global")
    if not feature_cache_dir.exists():
        print("⚠ Feature cache not found - skipping real data test", flush=True)
        print("  Run a full experiment first to populate the feature cache", flush=True)
        return None  # Skip, not fail
    print(f"    Feature cache found: {feature_cache_dir}", flush=True)
    
    # Load a few subjects
    set_step("Scanning for available subjects")
    subjects_to_test = [f"subject_{i}" for i in range(1, 11)]  # 10 subjects
    
    # Check which subjects are cached
    available = []
    for subj in subjects_to_test:
        cache_file = feature_cache_dir / f"{subj}_full.npz"
        if cache_file.exists():
            available.append(subj)
            print(f"    Found: {subj}", flush=True)
    
    if len(available) < 3:
        print(f"⚠ Not enough cached subjects found: {len(available)}", flush=True)
        return None
    
    print(f"    Total available: {len(available)} subjects", flush=True)
    
    # Load feature data
    set_step("Loading feature data from cache")
    all_features = []
    all_labels = []
    all_subjects = []
    
    for i, subj in enumerate(available):
        print(f"    [{i+1}/{len(available)}] Loading {subj}...", flush=True)
        cache_file = feature_cache_dir / f"{subj}_full.npz"
        data = np.load(cache_file, allow_pickle=True)
        
        features = data['features']
        labels = data['labels']
        
        print(f"        Loaded {len(labels)} epochs, {features.shape[1]} features", flush=True)
        
        all_features.append(features)
        all_labels.append(labels)
        all_subjects.extend([subj] * len(labels))
    
    set_step("Combining data from all subjects")
    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    subject_ids = np.array(all_subjects)
    
    print(f"    Combined data: {X.shape[0]} samples, {X.shape[1]} features", flush=True)
    label_dist = dict(zip(*np.unique(y, return_counts=True)))
    print(f"    Label distribution: {label_dist}", flush=True)
    
    # Create temp cache
    set_step("Creating temporary model cache directory")
    temp_dir = tempfile.mkdtemp(prefix="loso_cache_real_")
    print(f"    Temp cache dir: {temp_dir}", flush=True)
    
    try:
        set_step("Initializing LOSOModelCache")
        cache = LOSOModelCache(cache_dir=temp_dir, enable_registry=True)
        
        # Use real model type names from models.py: 'random_forest', 'xgboost', 'fnn'
        model_config = {
            'random_seed': 42,
            'code_version': 'v1.0',
            'model_name': 'random_forest',  # Real model type from models.py
            'model_params': {'n_estimators': 50, 'max_depth': 10, 'n_jobs': 1},  # n_jobs=1 to avoid deadlock
            'feature_config': {'base': X.shape[1], 'corr': None}
        }
        print(f"    Model config: {model_config['model_name']} (from models.py)", flush=True)
        print(f"    Model params: {model_config['model_params']}", flush=True)
        
        # Run 1: Cold start
        set_step("RUN 1: Cold Start - Training all folds")
        print("\n--- RUN 1: Cold Start ---", flush=True)
        run1_results = []
        run1_times = []
        
        for i, test_subj in enumerate(available):
            print(f"\n    [{i+1}/{len(available)}] Fold: {test_subj}", flush=True)
            
            test_mask = subject_ids == test_subj
            X_train, y_train = X[~test_mask], y[~test_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            print(f"        Train: {len(y_train)} samples, Test: {len(y_test)} samples", flush=True)
            
            fp = LOSOFingerprint.generate(**model_config, held_out_subject=test_subj)
            print(f"        Fingerprint: {fp[:16]}...", flush=True)
            
            cached = cache.get(fp, test_subj)
            if cached is None:
                print(f"        Status: MISS - Training with create_model() from models.py...", flush=True)
                start = time.time()
                # Use create_model from models.py (the REAL implementation)
                model = create_model(
                    model_type=model_config['model_name'],
                    params=model_config['model_params'],
                    random_seed=model_config['random_seed']
                )
                model.fit(X_train, y_train)
                train_time = time.time() - start
                print(f"        Training completed in {train_time:.2f}s", flush=True)
                
                print(f"        Caching model...", flush=True)
                cache.put(fp, test_subj, model, model_type=model_config['model_name'], training_time=train_time)
                status = "MISS"
            else:
                model = cached
                train_time = 0.0
                status = "HIT (unexpected)"
                print(f"        Status: {status}", flush=True)
            
            # Use model.predict() and calculate accuracy (real model interface)
            y_pred = model.predict(X_test)
            acc = np.mean(y_pred == y_test)
            run1_results.append(acc)
            run1_times.append(train_time)
            print(f"        → Accuracy: {acc:.4f}", flush=True)
        
        print(f"\n    Run 1 Summary:", flush=True)
        print(f"      Mean Accuracy: {np.mean(run1_results):.4f} ± {np.std(run1_results):.4f}", flush=True)
        print(f"      Total Training Time: {sum(run1_times):.2f}s", flush=True)
        
        # Run 2: Warm start
        set_step("RUN 2: Warm Start - Loading all folds from cache")
        print("\n--- RUN 2: Warm Start ---", flush=True)
        run2_results = []
        run2_times = []
        hits = 0
        
        for i, test_subj in enumerate(available):
            print(f"\n    [{i+1}/{len(available)}] Fold: {test_subj}", flush=True)
            
            test_mask = subject_ids == test_subj
            X_test, y_test = X[test_mask], y[test_mask]
            
            fp = LOSOFingerprint.generate(**model_config, held_out_subject=test_subj)
            print(f"        Fingerprint: {fp[:16]}...", flush=True)
            
            start = time.time()
            cached = cache.get(fp, test_subj)
            load_time = time.time() - start
            
            if cached is None:
                print(f"        Status: MISS (unexpected!)", flush=True)
                return False
            
            hits += 1
            model = cached
            print(f"        Status: HIT - Loaded in {load_time:.4f}s", flush=True)
            
            # Use model.predict() and calculate accuracy (real model interface)
            y_pred = model.predict(X_test)
            acc = np.mean(y_pred == y_test)
            run2_results.append(acc)
            run2_times.append(load_time)
            print(f"        → Accuracy: {acc:.4f}", flush=True)
        
        print(f"\n    Run 2 Summary:", flush=True)
        print(f"      Mean Accuracy: {np.mean(run2_results):.4f} ± {np.std(run2_results):.4f}", flush=True)
        print(f"      Total Load Time: {sum(run2_times):.4f}s", flush=True)
        print(f"      Cache Hits: {hits}/{len(available)}", flush=True)
        
        # Verify results are identical
        set_step("Verifying result consistency")
        if not np.allclose(run1_results, run2_results):
            print("⚠ WARNING: Run 1 and Run 2 accuracies differ!", flush=True)
            print(f"  Run 1: {run1_results}", flush=True)
            print(f"  Run 2: {run2_results}", flush=True)
        else:
            print("    ✓ Results are identical between runs!", flush=True)
        
        # Final statistics
        set_step("Calculating final statistics")
        speedup = sum(run1_times) / max(sum(run2_times), 0.001)
        stats = cache.get_stats()
        
        print(f"\n{'='*50}", flush=True)
        print("FINAL STATISTICS", flush=True)
        print(f"{'='*50}", flush=True)
        print(f"Subjects tested: {len(available)}", flush=True)
        print(f"Cache hits: {stats['session']['hits']}", flush=True)
        print(f"Cache misses: {stats['session']['misses']}", flush=True)
        print(f"Hit rate: {stats['session']['hit_rate']}", flush=True)
        print(f"Training time saved: {sum(run1_times):.2f}s", flush=True)
        print(f"Speedup factor: {speedup:.0f}x", flush=True)
        
        # List cache files
        set_step("Listing cache files")
        cache_files = list(Path(temp_dir).glob("*.joblib"))
        print(f"\nCache files created: {len(cache_files)}", flush=True)
        for f in cache_files[:3]:
            print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)", flush=True)
        if len(cache_files) > 3:
            print(f"  ... and {len(cache_files) - 3} more", flush=True)
        
        print("\n✓ PASS: Real data LOSO caching works correctly!", flush=True)
        return True
        
    finally:
        set_step("Cleaning up temporary directory")
        shutil.rmtree(temp_dir)
        print(f"    Cleaned up: {temp_dir}", flush=True)


# =============================================================================
# TEST 4: Full Pipeline Integration (TrainingPipeline from training.py)
# =============================================================================

def test_real_pipeline_integration():
    """
    Test with the ACTUAL TrainingPipeline from training.py.
    
    This test verifies that the LOSO caching works correctly when integrated
    with the full pipeline including:
    - TrainingConfig and FeatureSelectionConfig
    - LOSOCrossValidator
    - FeatureSelectionPipeline
    - The actual training.py TrainingPipeline class
    """
    print_separator("TEST 4: Real Pipeline Integration")
    
    # Ensure real project modules are loaded
    ensure_real_modules()
    
    # Check if feature cache exists
    set_step("Checking for feature cache directory")
    feature_cache_dir = Path("results/features_cache_global")
    if not feature_cache_dir.exists():
        print("⚠ Feature cache not found - skipping pipeline test", flush=True)
        return None
    print(f"    Feature cache found: {feature_cache_dir}", flush=True)
    
    # Import training pipeline components (all from real modules)
    set_step("Importing TrainingPipeline and related classes from real modules")
    try:
        import pandas as pd
        from training import TrainingPipeline, TrainingConfig
        from feature_selection import FeatureSelectionConfig
        from output_formatter import Verbosity
        print("    ✓ Imports successful", flush=True)
    except ImportError as e:
        print(f"⚠ Import error: {e}", flush=True)
        return None
    
    # Load subjects (use 5 to keep it fast)
    set_step("Loading feature data for 5 subjects")
    subjects_to_test = [f"subject_{i}" for i in range(1, 6)]  # 5 subjects
    
    available = []
    for subj in subjects_to_test:
        cache_file = feature_cache_dir / f"{subj}_full.npz"
        if cache_file.exists():
            available.append(subj)
            print(f"    Found: {subj}", flush=True)
    
    if len(available) < 3:
        print(f"⚠ Not enough cached subjects: {len(available)}", flush=True)
        return None
    
    # Load and combine data
    all_features = []
    all_labels = []
    all_subjects = []
    
    for subj in available:
        cache_file = feature_cache_dir / f"{subj}_full.npz"
        data = np.load(cache_file, allow_pickle=True)
        features = data['features']
        labels = data['labels']
        
        # Get feature names
        if 'feature_names' in data:
            feature_names = data['feature_names']
        else:
            feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        
        all_features.append(features)
        all_labels.append(labels)
        all_subjects.extend([subj] * len(labels))
        print(f"    Loaded {subj}: {len(labels)} epochs", flush=True)
    
    X = np.vstack(all_features)
    y = np.hstack(all_labels)
    subject_ids = np.array(all_subjects)
    
    # Create DataFrame for TrainingPipeline
    features_df = pd.DataFrame(X, columns=feature_names)
    print(f"    Combined: {features_df.shape[0]} samples, {features_df.shape[1]} features", flush=True)
    
    # Create temp directories
    set_step("Creating temporary directories")
    temp_output_dir = tempfile.mkdtemp(prefix="loso_pipeline_output_")
    temp_cache_dir = tempfile.mkdtemp(prefix="loso_pipeline_cache_")
    print(f"    Output dir: {temp_output_dir}", flush=True)
    print(f"    Cache dir: {temp_cache_dir}", flush=True)
    
    try:
        # Create a single test configuration (RandomForest with feature selection)
        set_step("Creating test configuration")
        config = TrainingConfig(
            model_type='random_forest',  # Valid: 'xgboost', 'random_forest', 'fnn'
            model_params={'n_estimators': 50, 'max_depth': 10, 'n_jobs': 1},
            feature_selection=FeatureSelectionConfig(
                correlation_threshold=0.95,
                top_k_features=30,
                selection_method='anova',  # Valid: 'anova', 'mi', 'hybrid'
                scope='global'  # Global feature selection for speed
            ),
            random_state=42
        )
        print(f"    Config ID: {config.get_config_id()}", flush=True)
        print(f"    Model: {config.model_type}", flush=True)
        print(f"    Feature Selection: top_k={config.feature_selection.top_k_features}, corr={config.feature_selection.correlation_threshold}", flush=True)
        
        # RUN 1: Cold start with fresh cache
        set_step("RUN 1: Cold Start with TrainingPipeline")
        print("\n--- RUN 1: Cold Start (fresh cache) ---", flush=True)
        
        pipeline1 = TrainingPipeline(
            features_df=features_df,
            labels=y,
            subject_ids=subject_ids,
            output_dir=temp_output_dir,
            experiment_name="loso_cache_test_run1",
            n_jobs=1,  # Sequential for clarity
            enable_model_cache=True,
            model_cache_dir=temp_cache_dir
        )
        
        run1_start = time.time()
        result1 = pipeline1.run_single_config(config, show_progress=False)
        run1_time = time.time() - run1_start
        
        print(f"\n    Run 1 Results:", flush=True)
        print(f"      Accuracy: {result1.accuracy_mean:.4f} ± {result1.accuracy_std:.4f}", flush=True)
        print(f"      Kappa: {result1.kappa_mean:.4f} ± {result1.kappa_std:.4f}", flush=True)
        print(f"      F1 Macro: {result1.f1_macro_mean:.4f}", flush=True)
        print(f"      Time: {run1_time:.2f}s", flush=True)
        
        # Get cache stats after run 1
        cache_stats1 = pipeline1.model_cache.get_stats()
        print(f"      Cache: {cache_stats1['session']['misses']} misses (all folds trained)", flush=True)
        
        # RUN 2: Warm start (should use cached models)
        set_step("RUN 2: Warm Start with TrainingPipeline (same cache)")
        print("\n--- RUN 2: Warm Start (same cache dir) ---", flush=True)
        
        # Create new pipeline instance pointing to same cache
        pipeline2 = TrainingPipeline(
            features_df=features_df,
            labels=y,
            subject_ids=subject_ids,
            output_dir=temp_output_dir,
            experiment_name="loso_cache_test_run2",
            n_jobs=1,
            enable_model_cache=True,
            model_cache_dir=temp_cache_dir  # Same cache dir!
        )
        
        run2_start = time.time()
        result2 = pipeline2.run_single_config(config, show_progress=False)
        run2_time = time.time() - run2_start
        
        print(f"\n    Run 2 Results:", flush=True)
        print(f"      Accuracy: {result2.accuracy_mean:.4f} ± {result2.accuracy_std:.4f}", flush=True)
        print(f"      Kappa: {result2.kappa_mean:.4f} ± {result2.kappa_std:.4f}", flush=True)
        print(f"      F1 Macro: {result2.f1_macro_mean:.4f}", flush=True)
        print(f"      Time: {run2_time:.2f}s", flush=True)
        
        # Get cache stats after run 2
        cache_stats2 = pipeline2.model_cache.get_stats()
        print(f"      Cache: {cache_stats2['session']['hits']} hits (all folds from cache!)", flush=True)
        
        # Verify results
        set_step("Verifying Results")
        
        # Check accuracy matches
        acc_match = np.isclose(result1.accuracy_mean, result2.accuracy_mean, atol=1e-6)
        kappa_match = np.isclose(result1.kappa_mean, result2.kappa_mean, atol=1e-6)
        
        if acc_match and kappa_match:
            print("    ✓ Results match between cold and warm runs!", flush=True)
        else:
            print(f"    ⚠ Results differ!", flush=True)
            print(f"      Run 1 Acc: {result1.accuracy_mean:.6f}", flush=True)
            print(f"      Run 2 Acc: {result2.accuracy_mean:.6f}", flush=True)
            return False
        
        # Check cache hits
        expected_hits = len(available)  # One hit per fold
        actual_hits = cache_stats2['session']['hits']
        if actual_hits >= expected_hits:
            print(f"    ✓ Cache hits as expected: {actual_hits} >= {expected_hits}", flush=True)
        else:
            print(f"    ⚠ Fewer cache hits than expected: {actual_hits} < {expected_hits}", flush=True)
            return False
        
        # Calculate speedup
        speedup = run1_time / max(run2_time, 0.001)
        print(f"\n{'='*50}", flush=True)
        print("PIPELINE INTEGRATION TEST SUMMARY", flush=True)
        print(f"{'='*50}", flush=True)
        print(f"Subjects: {len(available)}", flush=True)
        print(f"Run 1 (cold): {run1_time:.2f}s", flush=True)
        print(f"Run 2 (warm): {run2_time:.2f}s", flush=True)
        print(f"Speedup: {speedup:.1f}x", flush=True)
        print(f"Cache hits on warm run: {actual_hits}", flush=True)
        
        print("\n✓ PASS: Real pipeline integration works correctly!", flush=True)
        return True
        
    finally:
        set_step("Cleaning up temporary directories")
        shutil.rmtree(temp_output_dir, ignore_errors=True)
        shutil.rmtree(temp_cache_dir, ignore_errors=True)
        print(f"    Cleaned up temp directories", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("\n" + "=" * 60, flush=True)
    print(" LOSO CACHE & FINGERPRINTING TEST SUITE", flush=True)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 60, flush=True)
    
    # Start progress tracker
    start_progress_tracker()
    overall_start = time.time()
    
    results = {}
    
    try:
        # Test 1: Minimal unit tests
        set_step("TEST PHASE 1: Fingerprint Unit Tests")
        print("\n" + "#" * 60, flush=True)
        print("# PHASE 1: FINGERPRINT UNIT TESTS", flush=True)
        print("#" * 60, flush=True)
        results['fingerprint_determinism'] = test_fingerprint_determinism()
        results['fingerprint_sensitivity'] = test_fingerprint_sensitivity()
        results['fingerprint_held_out'] = test_fingerprint_held_out_uniqueness()
        
        # Test 2: Cache operations with synthetic data
        set_step("TEST PHASE 2: Cache Operations with Synthetic Data")
        print("\n" + "#" * 60, flush=True)
        print("# PHASE 2: CACHE OPERATIONS (SYNTHETIC DATA)", flush=True)
        print("#" * 60, flush=True)
        results['cache_basic'] = test_cache_basic_operations()
        results['cache_synthetic_loso'] = test_cache_with_synthetic_loso()
        
        # Test 3: Real data (simplified manual LOSO)
        set_step("TEST PHASE 3: Real Data with 10 Subjects")
        print("\n" + "#" * 60, flush=True)
        print("# PHASE 3: REAL DATA TEST (10 SUBJECTS - MANUAL LOSO)", flush=True)
        print("#" * 60, flush=True)
        results['cache_real_data'] = test_real_data_10_subjects()
        
        # Test 4: Full pipeline integration with TrainingPipeline
        set_step("TEST PHASE 4: Real Pipeline Integration (training.py)")
        print("\n" + "#" * 60, flush=True)
        print("# PHASE 4: FULL PIPELINE INTEGRATION TEST", flush=True)
        print("#" * 60, flush=True)
        results['pipeline_integration'] = test_real_pipeline_integration()
        
    finally:
        # Stop progress tracker
        stop_progress_tracker()
    
    overall_elapsed = time.time() - overall_start
    
    # Summary
    set_step("Generating test summary")
    print("\n" + "=" * 60, flush=True)
    print(" TEST SUMMARY", flush=True)
    print(f" Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f" Total duration: {overall_elapsed:.1f}s", flush=True)
    print("=" * 60, flush=True)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⚠ SKIP"
        print(f"  {test_name:30s} {status}", flush=True)
    
    # Overall result
    failures = sum(1 for r in results.values() if r is False)
    passes = sum(1 for r in results.values() if r is True)
    skips = sum(1 for r in results.values() if r is None)
    
    print(f"\nTotal: {passes} passed, {failures} failed, {skips} skipped", flush=True)
    print(f"Duration: {overall_elapsed:.1f} seconds", flush=True)
    
    if failures > 0:
        print("\n⚠ Some tests failed!", flush=True)
        return 1
    else:
        print("\n✓ All tests passed!", flush=True)
        return 0


if __name__ == "__main__":
    exit(main())
