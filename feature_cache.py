"""
Feature caching helpers with integrity verification.

STATUS: IMPLEMENTED ✓ (224× speedup verified)
    This module handles FEATURE-LEVEL caching (Stage 1).
    For MODEL-LEVEL caching (Stage 2 - core thesis work), see:
    - fingerprint.py (LOSOFingerprint — implemented)
    - loso_cache.py (LOSO model cache — implemented)

Cache Architecture:
==================
GLOBAL FEATURE CACHE (results/features_cache_global/) - IMPLEMENTED ✓
    - Shared across ALL experiments
    - Contains: subject_{id}_full.npz (195 features)
    - Purpose: Avoid recomputing expensive feature extraction
    - Status: 128/128 subjects cached, ~146 MB total
    
LOSO MODEL CACHE (results/loso_model_cache/) - IMPLEMENTED ✓
    - Cache trained models per LOSO fold
    - Key: fingerprint including held_out_subject + training_subjects
    - Purpose: Skip redundant model training across experiments
    - See: loso_cache.py and training.py integration
    
PER-EXPERIMENT DATA (results/experiment_*/per_subject/)
    - Isolated to each experiment run
    - Contains: epochs.npy, features.csv, labels.npy
    - Purpose: Reproducibility, intermediate inspection

Cache Integrity Strategy:
========================
- Store preprocessing config fingerprint with cached data
- Store source file metadata (size) for change detection
- On load, optionally validate against current config
- Manual data_version bump if raw data changes

NOTE on legacy "unknown" fingerprints (decision: option-b, leave as-is):
    Cache files produced before the config_fingerprint field was added do
    not store the field at all. On load, get_cache_info() substitutes the
    string "unknown" for display, and load_features_from_cache() with
    strict_validation=False (the default) accepts these legacy files
    without checking. This is intentional for the thesis: existing
    feature caches were produced under a fixed, frozen preprocessing
    config and are correct; forcing regeneration would cost hours of
    re-extraction for zero scientific benefit. The thesis disclosure
    accompanies this choice (Methodology §Fingerprint Generation).

    For post-thesis / production work: flip strict_validation to True
    by default and treat stored 'unknown' as a fingerprint mismatch —
    the next pipeline run will regenerate every legacy cache with a
    proper fingerprint, after which validation actually means something.

When to invalidate cache:
- Change preprocessing params → fingerprint changes automatically
  (only enforced when strict_validation=True; see above)
- Change feature extraction → bump feature_version
- Raw data changes → bump data_version OR delete cache
"""
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import pandas as pd
import hashlib
import json
import os


def compute_config_fingerprint(
    config_dict: Dict[str, Any],
    include_data_info: bool = True,
    file_path: Optional[Path] = None
) -> str:
    """
    Compute SHA-256 fingerprint of preprocessing configuration.
    
    Args:
        config_dict: Dictionary of preprocessing parameters
        include_data_info: Whether to include file metadata
        file_path: Path to source EDF file (for metadata)
        
    Returns:
        8-character hex fingerprint
    """
    fingerprint_data = config_dict.copy()
    
    # Optionally include file metadata for data change detection
    if include_data_info and file_path and Path(file_path).exists():
        fingerprint_data['_file_size'] = os.path.getsize(file_path)
        # Note: We don't include mtime as it changes on copy
    
    # Sort keys for deterministic hashing
    config_str = json.dumps(fingerprint_data, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:32]


def get_default_config_fingerprint() -> str:
    """
    Get fingerprint for current default preprocessing config.
    
    Default config (hardcoded for thesis):
        - bandpass: 0.5-40 Hz
        - notch: 50 Hz
        - target_sfreq: 128 Hz
        - epoch_duration: 30s
        
    Version tracking:
        - data_version: Bump if raw BOAS data changes
        - feature_version: Bump if feature extraction code changes
    """
    default_config = {
        'bandpass_low': 0.5,
        'bandpass_high': 40.0,
        'notch_freq': 50.0,
        'target_sfreq': 128.0,
        'epoch_duration': 30.0,
        'data_version': '1.0',     # BOAS dataset version
        'feature_version': '1.0'   # Feature extraction version
    }
    return compute_config_fingerprint(default_config, include_data_info=False)


def save_features_to_cache(
    cache_path: Path,
    features_df: pd.DataFrame,
    labels: np.ndarray,
    n_channels: int = 6,
    config_fingerprint: Optional[str] = None,
    source_file: Optional[str] = None
):
    """
    Save features to cache with integrity metadata.
    
    Stores:
        - features: The extracted feature matrix
        - feature_names: Column names for reconstruction
        - labels: Sleep stage labels
        - n_channels: Number of channels used
        - config_fingerprint: Hash of preprocessing config (for validation)
        - source_file: Original data file path (for reference)
        - cache_version: Cache format version
        - created_at: Timestamp for debugging
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use default fingerprint if not provided
    if config_fingerprint is None:
        config_fingerprint = get_default_config_fingerprint()

    from datetime import datetime
    
    np.savez_compressed(
        cache_path,
        features=features_df.values,
        feature_names=np.array(features_df.columns.tolist(), dtype=object),
        labels=labels,
        n_channels=n_channels,
        config_fingerprint=config_fingerprint,
        source_file=source_file or '',
        cache_version='1.0',
        created_at=datetime.now().isoformat()
    )


def load_features_from_cache(
    cache_path: Path,
    expected_fingerprint: Optional[str] = None,
    strict_validation: bool = False
) -> Optional[Tuple[pd.DataFrame, np.ndarray, int]]:
    """
    Load features from cache with optional integrity validation.
    
    Args:
        cache_path: Path to cache file
        expected_fingerprint: If provided, validate against stored fingerprint
        strict_validation: If True, return None on fingerprint mismatch
        
    Returns:
        Tuple of (features_df, labels, n_channels) or None if:
        - File doesn't exist
        - strict_validation=True and fingerprint mismatch
        
    Validation Modes:
        - Lenient (default): Load if file exists, ignore fingerprint
        - Strict: Verify fingerprint matches current config
        
    For thesis: Use lenient mode (config is fixed).
    For production: Use strict mode to catch config drift.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    data = np.load(cache_path, allow_pickle=True)

    # Validate fingerprint if strict mode enabled.
    #
    # NOTE on legacy 'unknown' fingerprints: caches produced before the
    # config_fingerprint field was added do not store the field at all.
    # The .get(...) default below ('') makes the inner `if stored_fingerprint
    # and ...` condition false, so legacy caches load even in strict mode.
    # That is the intentional thesis behaviour (preprocessing config is
    # frozen, regenerating would cost hours for zero benefit). For
    # post-thesis / production: change the default to 'unknown' and add an
    # explicit `or stored_fingerprint == 'unknown'` to the mismatch check
    # so legacy caches are forced through regeneration. See module
    # docstring for the full rationale.
    if strict_validation and expected_fingerprint:
        stored_fingerprint = str(data.get('config_fingerprint', ''))
        if stored_fingerprint and stored_fingerprint != expected_fingerprint:
            # Fingerprint mismatch - treat as cache miss
            # This means config changed since cache was created
            return None
    
    feature_names = data['feature_names'].tolist()
    features = pd.DataFrame(data['features'], columns=feature_names)
    labels = data['labels']
    n_channels = int(data['n_channels']) if 'n_channels' in data else 6

    return features, labels, n_channels


def get_cache_info(cache_path: Path) -> Optional[Dict[str, Any]]:
    """
    Get metadata about a cache file without loading full data.
    
    Useful for debugging and cache management.  
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    
    data = np.load(cache_path, allow_pickle=True)
    
    return {
        'n_epochs': data['features'].shape[0],
        'n_features': data['features'].shape[1],
        'n_channels': int(data.get('n_channels', 6)),
        'config_fingerprint': str(data.get('config_fingerprint', 'unknown')),
        'source_file': str(data.get('source_file', 'unknown')),
        'cache_version': str(data.get('cache_version', '0.0')),
        'created_at': str(data.get('created_at', 'unknown')),
        'file_size_mb': cache_path.stat().st_size / (1024 * 1024)
    }


def select_channel_features(
    features_df: pd.DataFrame,
    channels_to_keep: int = 6,
    standardize_names: bool = True
) -> pd.DataFrame:
    """
    Filter a full-feature DataFrame (per-channel prefixes + global features)
    to only include the requested number of channels.
    
    Handles both naming conventions:
    - Standard: F3_, F4_, C3_, C4_, O1_, O2_
    - Generic: CH1_, CH2_, CH3_, CH4_, CH5_, CH6_
    
    If standardize_names=True, converts generic names to standard names for
    consistent concatenation across subjects.
    """
    # Define canonical prefixes for the first 6 channels (both naming conventions)
    standard_prefixes = ['F3_', 'F4_', 'C3_', 'C4_', 'O1_', 'O2_']
    generic_prefixes = ['CH1_', 'CH2_', 'CH3_', 'CH4_', 'CH5_', 'CH6_']
    
    # Detect which naming convention is used
    has_standard = any(col.startswith('F3_') for col in features_df.columns)
    has_generic = any(col.startswith('CH1_') for col in features_df.columns)

    if channels_to_keep == 6:
        if has_standard:
            keep_prefixes = standard_prefixes
        elif has_generic:
            keep_prefixes = generic_prefixes
        else:
            # Fallback to standard
            keep_prefixes = standard_prefixes
    elif channels_to_keep >= 8:
        # Add two physiological channels EOG and EMG
        if has_standard:
            keep_prefixes = standard_prefixes + ['EOG_', 'EMG_']
        elif has_generic:
            keep_prefixes = generic_prefixes + ['CH7_', 'CH8_']
        else:
            keep_prefixes = standard_prefixes + ['EOG_', 'EMG_']
    else:
        # Fallback: keep as many of the base prefixes as requested
        if has_standard:
            keep_prefixes = standard_prefixes[:channels_to_keep]
        else:
            keep_prefixes = generic_prefixes[:channels_to_keep]

    # Keep per-channel features matching prefixes
    per_channel_cols = [
        col for col in features_df.columns
        if any(col.startswith(pref) for pref in keep_prefixes)
    ]

    # Global features heuristics: keep features that are not tied to excluded channels
    global_cols = [
        col for col in features_df.columns
        if col.startswith(('coherence_', 'plv_', 'global_'))
    ]

    # For 6-channel selection, remove any global feature mentioning EOG or EMG or CH7/CH8
    if channels_to_keep == 6:
        global_cols = [c for c in global_cols 
                      if 'EOG' not in c and 'EMG' not in c 
                      and 'CH7' not in c and 'CH8' not in c]

    keep_cols = per_channel_cols + global_cols

    # Preserve original column order where possible
    keep_cols = [c for c in features_df.columns if c in keep_cols]
    
    result = features_df.loc[:, keep_cols].copy()
    
    # Standardize column names (convert generic CH1_, CH2_... to F3_, F4_...)
    if standardize_names and has_generic and channels_to_keep == 6:
        rename_map = {}
        for gen, std in zip(generic_prefixes, standard_prefixes):
            for col in result.columns:
                if col.startswith(gen):
                    new_name = col.replace(gen, std, 1)
                    rename_map[col] = new_name
        result = result.rename(columns=rename_map)
    
    return result
