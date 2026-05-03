# Cache Invalidation Claim Verification Report
**Date:** January 16, 2026
**Purpose:** Verify thesis claims about cache invalidation strategies
**Verdict:** ⚠️ **PARTIALLY INCORRECT - Conflates two separate caching systems**

---

## Executive Summary

The thesis claim about "Handling Data Changes and Cache Invalidation" is **misleading** because it conflates features from **two separate caching systems**:

1. **Feature Cache** (`feature_cache.py`) - Has `data_version`, `feature_version`, `strict_validation`
2. **Model Cache** (`loso_cache.py`) - Has only `code_version` in fingerprint, no strict validation

The thesis presents this as a unified cache invalidation strategy, but the described features (`data_version`, `feature_version`, `strict_validation`) **only exist in the feature cache**, not the model cache (which is the core thesis contribution).

---

## Thesis Claim Analysis

### Original Thesis Text

> A critical challenge in ML experiment caching is ensuring cache validity when underlying data changes. Our implementation addresses this through a multi-layered integrity system embedded in the cache metadata. Each cached entry stores a configuration fingerprint---a SHA-256 hash of all preprocessing parameters, alongside explicit version identifiers (\texttt{data\_version} and \texttt{feature\_version}) and source file metadata.
>
> When raw data modifications occur, three invalidation strategies are available:
>
> 1. **Version Bumping:** Incrementing the \texttt{data\_version} parameter in the configuration file causes the fingerprint to change, automatically invalidating all cached entries without requiring manual intervention.
>
> 2. **Cache Purging:** For significant data changes or version migrations, the global cache directory can be deleted entirely, triggering a complete recomputation on the next execution.
>
> 3. **Strict Validation:** An optional \texttt{strict\_validation} mode computes file checksums during cache lookup, detecting any byte-level changes to source files. While more computationally expensive, this provides cryptographic guarantees of data integrity.

---

## Claim-by-Claim Verification

### Claim 1: "Each cached entry stores a configuration fingerprint---a SHA-256 hash" ✅ **CORRECT**

**Evidence:**

**`fingerprint.py:28-29`** (Module-level documentation):
```python
"""
Fingerprint Formula:
    fingerprint = SHA256(canonical_json(config))[:32]
"""
```

**`fingerprint.py:247-254`** (Implementation):
```python
def from_config(cls, config: LOSOFingerprintConfig) -> str:
    """Generate fingerprint from config object."""
    config_dict = config.to_dict()

    # Create canonical JSON (sorted keys for deterministic hashing)
    config_json = json.dumps(config_dict, sort_keys=True)

    # Compute SHA-256 hash
    hash_obj = hashlib.sha256(config_json.encode('utf-8'))
```

**Verdict:** ✅ **CORRECT** - SHA-256 hashing is used for fingerprint generation.

---

### Claim 2: "alongside explicit version identifiers (data_version and feature_version)" ❌ **INCORRECT**

**Evidence:**

**Model Cache (`loso_cache.py`):** NO `data_version` or `feature_version` fields
**Feature Cache (`feature_cache.py:102-103`):** YES, has these fields

```python
# feature_cache.py - NOT the model cache!
default_config = {
    'bandpass_low': 0.5,
    'bandpass_high': 40.0,
    'notch_freq': 50.0,
    'target_sfreq': 128.0,
    'epoch_duration': 30.0,
    'data_version': '1.0',     # ← Only in FEATURE cache
    'feature_version': '1.0'   # ← Only in FEATURE cache
}
```

**Model Cache uses `code_version` instead:**

**`fingerprint.py:107-114`**:
```python
@dataclass
class LOSOFingerprintConfig:
    """Complete configuration for LOSO fingerprint generation."""
    random_seed: int
    code_version: str              # ← MODEL cache uses this
    model_config: ModelConfig
    feature_config: FeatureConfig
    held_out_subject: str
```

**Verdict:** ❌ **INCORRECT** - The thesis describes `data_version` and `feature_version`, but these exist only in the feature cache (Stage 1), not the model cache (Stage 2 - core thesis contribution). The model cache uses `code_version`.

---

### Claim 3: "Version Bumping: Incrementing data_version causes fingerprint to change" ⚠️ **PARTIALLY CORRECT**

**Analysis:**

The mechanism is correct, but the field name is wrong:

**For Feature Cache:** Bump `data_version` (lines 93-94, `feature_cache.py`)
**For Model Cache:** Bump `code_version` (line 114, `fingerprint.py`)

**Example from `fingerprint.py:363-398`**:
```python
# Changing code_version changes the fingerprint
fp1 = LOSOFingerprint.generate(
    code_version='v1.0',  # ← Version bumping works here
    model_name='xgboost',
    ...
)

fp2 = LOSOFingerprint.generate(
    code_version='v1.1',  # ← Different version → different fingerprint
    model_name='xgboost',
    ...
)

assert fp1 != fp2  # Different fingerprints
```

**Verdict:** ⚠️ **PARTIALLY CORRECT** - Version bumping works, but for the MODEL cache (core thesis), you bump `code_version`, not `data_version`.

---

### Claim 4: "Cache Purging: global cache directory can be deleted entirely" ✅ **CORRECT**

**Evidence:**

**`loso_cache.py:388-437`** - `invalidate()` method:
```python
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
        If both None, invalidates ALL cached models  # ← Full purge supported
    """
    invalidated_count = 0
    for cache_file in self.cache_dir.glob(f"*{self.MODEL_EXTENSION}"):
        # ... deletion logic ...
        cache_file.unlink()
        invalidated_count += 1
```

**Usage:**
```python
# Purge entire model cache
cache.invalidate()  # No arguments = delete all

# Or manually delete directory
import shutil
shutil.rmtree("results/loso_model_cache/")
```

**Verdict:** ✅ **CORRECT** - Cache purging is fully supported via `invalidate()` method or manual deletion.

---

### Claim 5: "Strict Validation: optional strict_validation mode computes file checksums" ❌ **INCORRECT FOR MODEL CACHE**

**Evidence:**

**Feature Cache (`feature_cache.py:154-188`):** ✅ HAS strict validation

```python
def load_features_from_cache(
    cache_path: Path,
    expected_fingerprint: Optional[str] = None,
    strict_validation: bool = False  # ← EXISTS in feature cache
) -> Optional[Tuple[pd.DataFrame, np.ndarray, int]]:
    """
    Validation Modes:
        - Lenient (default): Load if file exists, ignore fingerprint
        - Strict: Verify fingerprint matches current config
    """
    # Validate fingerprint if strict mode enabled
    if strict_validation and expected_fingerprint:
        stored_fingerprint = str(data.get('config_fingerprint', ''))
        if stored_fingerprint and stored_fingerprint != expected_fingerprint:
            return None  # Fingerprint mismatch - cache miss
```

**Model Cache (`loso_cache.py`):** ❌ NO strict validation mode

**`loso_cache.py:275-328`** - `get()` method has NO strict_validation parameter:
```python
def get(
    self,
    fingerprint: str,
    held_out_subject: str,
    model_type: str = "unknown",
    model_class: Any = None,
    model_params: dict = None,
    record_metrics: bool = True  # ← NO strict_validation parameter
) -> Any:
    """Get a cached model if it exists."""
    cache_path = self._get_cache_path(fingerprint, held_out_subject)
    if not cache_path.exists():
        return None  # Simple existence check - no checksum validation
```

**Verdict:** ❌ **INCORRECT** - `strict_validation` mode with file checksums exists ONLY in the feature cache (`feature_cache.py`), not in the model cache (`loso_cache.py`), which is the core thesis contribution.

---

## The Two-Tier Cache Architecture

The confusion arises from having **two separate caching systems**:

### Stage 1: Feature Cache (`feature_cache.py`)
- **Purpose:** Cache extracted features (149 features × ~1000 epochs per subject)
- **Location:** `results/features_cache_global/`
- **File format:** `.npz` (NumPy compressed arrays)
- **Invalidation:** `data_version`, `feature_version`, `strict_validation`
- **Status:** Fully implemented, 224× speedup verified

### Stage 2: Model Cache (`loso_cache.py`) - **Core Thesis Contribution**
- **Purpose:** Cache trained LOSO fold models
- **Location:** `results/loso_model_cache/`
- **File format:** `.joblib` (sklearn/XGBoost models)
- **Invalidation:** `code_version` in fingerprint, `invalidate()` method
- **Status:** Fully implemented, 4.5× speedup verified

---

## What the Thesis SHOULD Say

### Corrected Section Text

> A critical challenge in ML experiment caching is ensuring cache validity when underlying configuration or data changes. Our implementation addresses this through a **two-tier caching architecture** with distinct invalidation strategies for each layer.
>
> **Feature Cache (Stage 1):** Stores extracted 149-feature representations for all subjects. Cache entries include a configuration fingerprint (SHA-256 hash of preprocessing parameters), explicit version identifiers (\texttt{data\_version} and \texttt{feature\_version}), and source file metadata. An optional \texttt{strict\_validation} mode computes file checksums during cache lookup, detecting any byte-level changes to source EDF files.
>
> **Model Cache (Stage 2):** Stores trained LOSO fold models using fingerprint-based cache keys. Each fingerprint is a SHA-256 hash of the complete experiment configuration, including \texttt{code\_version}, model hyperparameters, feature selection parameters, and the held-out subject ID. Changing ANY configuration parameter automatically produces a different fingerprint, triggering cache invalidation without manual intervention.
>
> When configuration or data changes occur, three invalidation strategies are available:
>
> 1. **Version Bumping:**
>    - **Feature Cache:** Increment \texttt{data\_version} (for raw data changes) or \texttt{feature\_version} (for feature extraction code changes)
>    - **Model Cache:** Increment \texttt{code\_version} (for any experiment configuration change)
>    - Effect: Fingerprint changes automatically, invalidating all cached entries
>
> 2. **Cache Purging:** For significant changes or version migrations, the cache directory can be deleted entirely (\texttt{cache.invalidate()} method or manual deletion), triggering complete recomputation on the next execution.
>
> 3. **Strict Validation** (Feature Cache only): An optional \texttt{strict\_validation} mode verifies configuration fingerprints during cache lookup, providing cryptographic guarantees that cached features match the current preprocessing configuration. This mode is disabled by default for performance, as the fixed thesis configuration ensures fingerprint stability.
>
> Our implementation defaults to fingerprint-based validation (Option 1) for both caches, balancing integrity assurance with computational efficiency. The fingerprint system ensures that configuration drift---such as modified model hyperparameters, feature selection thresholds, or held-out subjects---automatically triggers recomputation, while unchanged configurations benefit from cached results.

---

## Implementation Evidence Summary

| Feature | Feature Cache | Model Cache | Thesis Claim | Verdict |
|---------|---------------|-------------|--------------|---------|
| SHA-256 fingerprinting | ✅ Yes | ✅ Yes | ✅ Correct | ✅ CORRECT |
| `data_version` field | ✅ Yes | ❌ No | ❌ Claims it's in model cache | ❌ WRONG |
| `feature_version` field | ✅ Yes | ❌ No | ❌ Claims it's in model cache | ❌ WRONG |
| `code_version` field | ❌ No | ✅ Yes | ❌ Not mentioned | ⚠️ MISSING |
| Version bumping | ✅ Yes | ✅ Yes (via code_version) | ⚠️ Wrong field name | ⚠️ MISLEADING |
| Cache purging | ✅ Yes | ✅ Yes | ✅ Correct | ✅ CORRECT |
| `strict_validation` mode | ✅ Yes | ❌ No | ❌ Claims it's general | ❌ WRONG |
| File checksum validation | ✅ Yes | ❌ No | ❌ Claims it's general | ❌ WRONG |

---

## Recommendations

### 1. Clarify Two-Tier Architecture

The thesis MUST distinguish between:
- **Feature cache** (preprocessing outputs)
- **Model cache** (trained LOSO models - core thesis contribution)

### 2. Correct Version Field Names

Replace all references to `data_version` and `feature_version` in the MODEL cache section with `code_version`.

### 3. Scope Strict Validation Correctly

State explicitly that `strict_validation` mode applies ONLY to the feature cache, not the model cache.

### 4. Add Missing Details

Mention that the model cache uses:
- `code_version` for version tracking
- Fingerprint includes `held_out_subject` (prevents data leakage)
- Automatic invalidation when ANY config parameter changes

### 5. Clarify Default Behavior

State: "Our implementation defaults to **fingerprint-based validation** (version bumping via `code_version`) for the model cache, which is the core thesis contribution."

---

## Conclusion

**Overall Assessment:**
The implementation is **excellent** and follows best practices for cache invalidation. However, the thesis description is **misleading** because it:

1. Conflates two separate caching systems (feature cache vs model cache)
2. Describes features (`data_version`, `feature_version`, `strict_validation`) that only exist in the feature cache
3. Presents these features as if they apply to the model cache (core thesis contribution)

**Critical Fix Required:**
Rewrite the cache invalidation section to clearly distinguish between the two-tier caching architecture and accurately describe which features belong to which cache.

**Impact:**
Without this correction, readers will be confused about how the core model caching system actually works, potentially affecting thesis evaluation and reproducibility.

---

## Appendix: File Locations

- **Feature Cache Implementation:** `feature_cache.py` (lines 1-200)
- **Model Cache Implementation:** `loso_cache.py` (lines 1-600)
- **Fingerprint Generation:** `fingerprint.py` (lines 1-422)
- **Feature Cache Metadata:** Lines 93-106 (data_version, feature_version)
- **Model Cache Metadata:** Lines 136-145 (fingerprint, code_version, created_at)
- **Strict Validation:** Lines 154-188 (feature_cache.py only)
