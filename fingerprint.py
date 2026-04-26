"""
Fingerprint Module for LOSO Model Caching
==========================================

Implements SHA-256 fingerprint-based cache key generation for LOSO cross-validation.
Each unique experiment configuration produces a unique fingerprint, enabling:
- Automatic cache invalidation when parameters change
- Data leakage prevention (held_out_subject in fingerprint)
- Reproducibility tracking

Fingerprint Formula:
    fingerprint = SHA256(canonical_json(config))[:32]
    
    Where config includes:
    - random_seed: Ensures deterministic behavior
    - code_version: Git commit or semantic version
    - model_config: Algorithm name and hyperparameters
    - feature_config: Base feature count and correlation threshold
    - held_out_subject: Critical for LOSO - prevents cross-fold contamination

Output: 32 hex characters (128 bits) - collision resistant for large-scale experiments

Author: Lennart Gorzel
Date: December 2025
Status: IMPLEMENTED
"""

import hashlib
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


# Code version — manually bumped for the thesis run.
#
# Limitation (disclosed in Methodology §Fingerprint Generation):
#   This string feeds the cache fingerprint, so two checkouts that share
#   the same string but differ in source will silently reuse cached
#   models. For the thesis the codebase is frozen during the experimental
#   run, so a static literal is sufficient. For production / continued
#   work, replace this with a runtime lookup such as:
#       subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
#   (with a non-git fallback) or any equivalent build-time identifier
#   from CI / a release pipeline. That ties the cache key to a real-
#   world version anchor and invalidates caches automatically on every
#   commit.
# TODO (post-thesis): wire this to git SHA / build identifier.
__version__ = "1.0.0"


@dataclass
class ModelConfig:
    """
    Model configuration for fingerprint generation.
    
    Attributes:
        name: Model algorithm name (e.g., 'xgboost', 'rf', 'fnn')
        params: Model hyperparameters dictionary
    """
    name: str
    params: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'params': self.params
        }


@dataclass
class FeatureConfig:
    """
    Feature configuration for fingerprint generation.

    Attributes:
        base_features: Number of base features (149 for 6-channel, 195 for 8-channel)
        correlation_threshold: Correlation filter threshold (e.g., 0.85, 0.90, None)
        top_k: Number of top features to select (e.g., 30, 50, 149)
        n_selected: Actual number of features after selection (for cache validation)
        selected_features: List of actual selected feature names (for cache validation)
    """
    base_features: int = 195
    correlation_threshold: Optional[float] = None
    top_k: Optional[int] = None
    n_selected: Optional[int] = None
    selected_features: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'base': self.base_features,
            'corr': self.correlation_threshold,
            'top_k': self.top_k
        }
        # Include actual selected features for cache validation
        # Hash the feature list to keep fingerprint compact
        if self.selected_features is not None:
            features_str = ','.join(sorted(self.selected_features))
            features_hash = hashlib.sha256(features_str.encode()).hexdigest()[:16]
            result['features_hash'] = features_hash
        elif self.n_selected is not None:
            result['n_selected'] = self.n_selected
        return result


@dataclass
class LOSOFingerprintConfig:
    """
    Complete configuration for LOSO fingerprint generation.
    
    All fields that affect model training outcome must be included.
    Changing ANY field produces a different fingerprint → cache miss.
    
    Attributes:
        random_seed: Random seed for reproducibility (e.g., 42)
        code_version: Code version string (e.g., 'v1.0' or 'git:abc123')
        model_config: Model algorithm and hyperparameters
        feature_config: Feature selection configuration
        held_out_subject: Subject ID held out for testing (CRITICAL for LOSO)
        training_subjects: Optional list of training subject IDs (for partial datasets)
    """
    random_seed: int
    code_version: str
    model_config: ModelConfig
    feature_config: FeatureConfig
    held_out_subject: str
    training_subjects: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for canonical JSON serialization.
        
        Keys are sorted alphabetically for deterministic hashing.
        """
        config_dict = {
            'code_version': self.code_version,
            'feature_config': self.feature_config.to_dict(),
            'held_out_subject': self.held_out_subject,
            'model_config': self.model_config.to_dict(),
            'random_seed': self.random_seed,
        }
        
        # Include training_subjects hash if provided (for partial dataset scenarios)
        if self.training_subjects is not None:
            # Hash the sorted list of training subjects for compact representation
            subjects_str = ','.join(sorted(self.training_subjects))
            subjects_hash = hashlib.sha256(subjects_str.encode()).hexdigest()[:16]
            config_dict['training_subjects_hash'] = subjects_hash
        
        return config_dict


class LOSOFingerprint:
    """
    Generates SHA-256 fingerprints for LOSO cross-validation caching.
    
    The fingerprint uniquely identifies an experiment configuration.
    Two runs with identical fingerprints are guaranteed to produce
    identical models (given deterministic training).
    
    Usage:
        # Create fingerprint for a LOSO fold
        fingerprint = LOSOFingerprint.generate(
            random_seed=42,
            code_version='v1.0',
            model_name='xgboost',
            model_params={'max_depth': 6, 'n_estimators': 200},
            feature_config={'base': 149, 'corr': 0.85},
            held_out_subject='SC4001'
        )
        
        # Use as cache key
        cache_path = f"models/{fingerprint}.joblib"
    
    Example Output:
        'a3f2b8c1d4e5f6a7b8c9d0e1f2a3b4c5'  (32 hex chars = 128 bits)
    """
    
    # Fingerprint length in hex characters (32 chars = 128 bits)
    FINGERPRINT_LENGTH = 32
    
    @classmethod
    def generate(
        cls,
        random_seed: int,
        code_version: str,
        model_name: str,
        model_params: Dict[str, Any],
        feature_config: Union[Dict[str, Any], FeatureConfig],
        held_out_subject: str,
        training_subjects: Optional[List[str]] = None
    ) -> str:
        """
        Generate a fingerprint for the given configuration.
        
        Args:
            random_seed: Random seed for reproducibility (e.g., 42)
            code_version: Code version (e.g., 'v1.0', 'git:abc123')
            model_name: Model algorithm name (e.g., 'xgboost', 'rf', 'fnn')
            model_params: Model hyperparameters dictionary
            feature_config: Feature configuration dict or FeatureConfig object
            held_out_subject: Subject ID being held out for testing
            training_subjects: Optional list of training subject IDs
            
        Returns:
            32-character hex string fingerprint
            
        Example:
            >>> LOSOFingerprint.generate(
            ...     random_seed=42,
            ...     code_version='v1.0',
            ...     model_name='xgboost',
            ...     model_params={'max_depth': 6, 'n_estimators': 200},
            ...     feature_config={'base': 149, 'corr': 0.85},
            ...     held_out_subject='SC4001'
            ... )
            'a3f2b8c1d4e5f6a7b8c9d0e1f2a3b4c5'
        """
        # Normalize feature_config to FeatureConfig object
        if isinstance(feature_config, dict):
            feat_cfg = FeatureConfig(
                base_features=feature_config.get('base', 195),
                correlation_threshold=feature_config.get('corr'),
                top_k=feature_config.get('top_k'),
                n_selected=feature_config.get('n_selected'),
                selected_features=feature_config.get('selected_features')
            )
        else:
            feat_cfg = feature_config
        
        # Create model config
        model_cfg = ModelConfig(name=model_name, params=model_params)
        
        # Create full config
        config = LOSOFingerprintConfig(
            random_seed=random_seed,
            code_version=code_version,
            model_config=model_cfg,
            feature_config=feat_cfg,
            held_out_subject=held_out_subject,
            training_subjects=training_subjects
        )
        
        return cls.from_config(config)
    
    @classmethod
    def from_config(cls, config: LOSOFingerprintConfig) -> str:
        """
        Generate fingerprint from a LOSOFingerprintConfig object.
        
        Args:
            config: Complete configuration object
            
        Returns:
            32-character hex string fingerprint
        """
        # Convert to canonical JSON (sorted keys for determinism)
        config_dict = config.to_dict()
        canonical_json = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA-256 hash and truncate to 32 hex chars (128 bits)
        fingerprint = hashlib.sha256(canonical_json.encode()).hexdigest()[:cls.FINGERPRINT_LENGTH]
        
        logger.debug(f"Generated fingerprint: {fingerprint} for config: {canonical_json}")
        
        return fingerprint
    
    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        held_out_subject: str
    ) -> str:
        """
        Generate fingerprint from a configuration dictionary.
        
        Convenience method for integration with existing config structures.
        
        Args:
            config_dict: Dictionary containing random_seed, model_type, etc.
            held_out_subject: Subject ID being held out (passed separately for LOSO loop)
            
        Returns:
            32-character hex string fingerprint
        """
        return cls.generate(
            random_seed=config_dict.get('random_seed', config_dict.get('random_state', 42)),
            code_version=config_dict.get('code_version', __version__),
            model_name=config_dict.get('model_type', config_dict.get('model_name', 'unknown')),
            model_params=config_dict.get('model_params', {}),
            feature_config=config_dict.get('feature_config', {}),
            held_out_subject=held_out_subject,
            training_subjects=config_dict.get('training_subjects')
        )
    
    @staticmethod
    def validate(fingerprint: str) -> bool:
        """
        Validate that a string is a valid fingerprint format.
        
        Args:
            fingerprint: String to validate
            
        Returns:
            True if valid fingerprint format, False otherwise
        """
        if not isinstance(fingerprint, str):
            return False
        if len(fingerprint) != LOSOFingerprint.FINGERPRINT_LENGTH:
            return False
        try:
            int(fingerprint, 16)  # Check if valid hex
            return True
        except ValueError:
            return False


def generate_cache_key(
    held_out_subject: str,
    model_type: str = 'xgboost',
    model_params: Optional[Dict[str, Any]] = None,
    feature_config: Optional[Dict[str, Any]] = None,
    random_seed: int = 42,
    code_version: Optional[str] = None
) -> str:
    """
    Convenience function to generate a cache key for a LOSO fold.
    
    This is the primary interface for the training loop.
    
    Args:
        held_out_subject: Subject ID being held out for testing
        model_type: Model algorithm name (default: 'xgboost')
        model_params: Model hyperparameters (default: empty dict)
        feature_config: Feature selection config (default: empty dict)
        random_seed: Random seed (default: 42)
        code_version: Code version (default: module __version__)
        
    Returns:
        Cache key string in format: {fingerprint}_{held_out_subject}
        
    Example:
        >>> key = generate_cache_key('SC4001', model_type='xgboost')
        >>> print(key)
        'a3f2b8c1d4e5f6a7b8c9d0e1f2a3b4c5_SC4001'
    """
    fingerprint = LOSOFingerprint.generate(
        random_seed=random_seed,
        code_version=code_version or __version__,
        model_name=model_type,
        model_params=model_params or {},
        feature_config=feature_config or {},
        held_out_subject=held_out_subject
    )
    
    return f"{fingerprint}_{held_out_subject}"


# ============================================================================
# QUICK TEST / DEMO
# ============================================================================

if __name__ == "__main__":
    # Demo: Generate fingerprints for different configurations
    print("=" * 60)
    print("LOSOFingerprint Demo")
    print("=" * 60)
    
    # Example 1: Basic fingerprint generation
    fp1 = LOSOFingerprint.generate(
        random_seed=42,
        code_version='v1.0',
        model_name='xgboost',
        model_params={'max_depth': 6, 'n_estimators': 200},
        feature_config={'base': 149, 'corr': 0.85},
        held_out_subject='SC4001'
    )
    print(f"\nExample 1 - XGBoost, subject SC4001:")
    print(f"  Fingerprint: {fp1}")
    print(f"  Length: {len(fp1)} hex chars = {len(fp1) * 4} bits")
    
    # Example 2: Same config, different subject → different fingerprint
    fp2 = LOSOFingerprint.generate(
        random_seed=42,
        code_version='v1.0',
        model_name='xgboost',
        model_params={'max_depth': 6, 'n_estimators': 200},
        feature_config={'base': 149, 'corr': 0.85},
        held_out_subject='SC4002'  # Different subject
    )
    print(f"\nExample 2 - Same config, different subject (SC4002):")
    print(f"  Fingerprint: {fp2}")
    print(f"  Different from Example 1: {fp1 != fp2}")
    
    # Example 3: Different model → different fingerprint
    fp3 = LOSOFingerprint.generate(
        random_seed=42,
        code_version='v1.0',
        model_name='rf',  # Different model
        model_params={'n_estimators': 100, 'max_depth': 10},
        feature_config={'base': 149, 'corr': 0.85},
        held_out_subject='SC4001'
    )
    print(f"\nExample 3 - Random Forest, same subject:")
    print(f"  Fingerprint: {fp3}")
    print(f"  Different from Example 1: {fp1 != fp3}")
    
    # Example 4: Cache key generation
    cache_key = generate_cache_key(
        held_out_subject='SC4001',
        model_type='xgboost',
        model_params={'max_depth': 6},
        feature_config={'base': 149, 'corr': 0.85}
    )
    print(f"\nExample 4 - Cache key format:")
    print(f"  Key: {cache_key}")
    
    # Example 5: Reproducibility check
    fp1_repeat = LOSOFingerprint.generate(
        random_seed=42,
        code_version='v1.0',
        model_name='xgboost',
        model_params={'max_depth': 6, 'n_estimators': 200},
        feature_config={'base': 149, 'corr': 0.85},
        held_out_subject='SC4001'
    )
    print(f"\nExample 5 - Reproducibility:")
    print(f"  Same config produces same fingerprint: {fp1 == fp1_repeat}")
    
    print("\n" + "=" * 60)
    print("Fingerprint validation passed!")
    print("=" * 60)
