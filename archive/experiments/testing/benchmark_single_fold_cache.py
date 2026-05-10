#!/usr/bin/env python3
"""
Single-Fold LOSO Cache Benchmark
================================

Benchmarks one real LOSO fold per configuration so you can verify:
- cold training time
- disk cache load time
- prediction time after cache load

This is intended as a sanity check for the cache path when a full 128-fold run
is too slow to inspect interactively.

Default scope:
- thesis grid for the currently implemented models (XGBoost + Random Forest)
- optional FNN via --include-fnn if PyTorch is available

The script uses a dedicated benchmark cache directory so it does not interfere
with your normal experiment cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cross_validation import CVFold, LOSOCrossValidator
from feature_cache import load_features_from_cache, select_channel_features
from feature_selection import FeatureSelectionPipeline
from fingerprint import LOSOFingerprint, __version__ as FINGERPRINT_VERSION
from loso_cache import LOSOModelCache
from models import create_model
from training import TrainingConfig, create_thesis_grid, create_training_grid


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


DEFAULT_CACHE_DIR = Path("./results/loso_model_cache_benchmark")
DEFAULT_GLOBAL_CACHE_DIR = Path("./results/features_cache_global")


def _subject_sort_key(cache_file: Path) -> int:
    subject_id = cache_file.stem.replace("subject_", "").replace("_full", "")
    try:
        return int(subject_id)
    except ValueError:
        return 10**9


def load_cached_dataset(cache_dir: Path, n_channels: int = 6) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load all cached subject features from disk."""
    if not cache_dir.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_dir}")

    subject_files = sorted(cache_dir.glob("subject_*_full.npz"), key=_subject_sort_key)
    if not subject_files:
        raise FileNotFoundError(f"No subject caches found in {cache_dir}")

    all_features: List[pd.DataFrame] = []
    all_labels: List[np.ndarray] = []
    all_subject_ids: List[str] = []

    for cache_file in subject_files:
        subject_id = cache_file.stem.replace("subject_", "").replace("_full", "")
        cached = load_features_from_cache(cache_file)
        if cached is None:
            logger.warning(f"Skipping unreadable cache: {cache_file.name}")
            continue

        features_df, labels, _ = cached
        features_df = select_channel_features(features_df, channels_to_keep=n_channels)
        all_features.append(features_df)
        all_labels.append(labels)
        all_subject_ids.extend([f"Subject_{subject_id}"] * len(labels))

    if not all_features:
        raise RuntimeError(f"No usable cached subjects found in {cache_dir}")

    return (
        pd.concat(all_features, ignore_index=True),
        np.concatenate(all_labels),
        np.array(all_subject_ids),
    )


def get_fold_for_subject(features_df: pd.DataFrame, labels: np.ndarray, subject_ids: np.ndarray, target_subject: str) -> CVFold:
    """Return the LOSO fold for the requested held-out subject."""
    cv = LOSOCrossValidator(verbose=False)
    for fold in cv.split(features_df, labels, subject_ids):
        if fold.test_subject == target_subject:
            return fold
    raise ValueError(f"Subject not found in LOSO split: {target_subject}")


def build_thesis_configs(include_fnn: bool = False) -> List[TrainingConfig]:
    """Build the thesis-style configuration grid for the implemented models."""
    if include_fnn:
        return create_training_grid(
            models=["xgboost", "random_forest", "fnn"],
            correlation_thresholds=[0.75, 0.90, None],
            top_k_features=[30, 50, None],
            random_state=42,
            use_hybrid=True,
            selection_method="anova",
            scope="global",
        )

    return create_thesis_grid(
        random_state=42,
        use_hybrid=True,
        selection_method="anova",
        scope="global",
    )


def _fit_feature_selection(
    config: TrainingConfig,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    X_full: pd.DataFrame,
    y_full: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], float]:
    """Apply the same global feature-selection path used by the training pipeline."""
    fs_start = time.time()

    if config.feature_selection.correlation_threshold is None and config.feature_selection.top_k_features is None:
        selected_features = list(X_full.columns)
        return X_train[selected_features], X_test[selected_features], selected_features, time.time() - fs_start

    fs_pipeline = FeatureSelectionPipeline(config.feature_selection)
    fs_pipeline.fit(X_full, y_full)
    selected_features = fs_pipeline.get_selected_features()
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    return X_train_selected, X_test_selected, selected_features, time.time() - fs_start


def run_single_config_benchmark(
    config: TrainingConfig,
    fold: CVFold,
    features_df: pd.DataFrame,
    labels: np.ndarray,
    cache_dir: Path,
    clear_matching_cache: bool = True,
) -> Dict[str, Any]:
    """Run cold and warm timing for one config on one LOSO fold."""
    X_train = features_df.iloc[fold.train_indices].copy()
    y_train = labels[fold.train_indices]
    X_test = features_df.iloc[fold.test_indices].copy()
    y_test = labels[fold.test_indices]

    X_train_selected, X_test_selected, selected_features, fs_time = _fit_feature_selection(
        config, X_train, y_train, X_test, features_df, labels
    )

    fingerprint = LOSOFingerprint.generate(
        random_seed=config.random_state,
        code_version=FINGERPRINT_VERSION,
        model_name=config.model_type,
        model_params=config.model_params,
        feature_config={
            "base": X_train.shape[1],
            "corr": config.feature_selection.correlation_threshold,
            "top_k": config.feature_selection.top_k_features,
            "n_selected": X_train_selected.shape[1],
            "selected_features": selected_features,
        },
        held_out_subject=str(fold.test_subject),
    )

    model_cache = LOSOModelCache(cache_dir=str(cache_dir), enable_registry=True, estimated_training_time=120.0)
    if clear_matching_cache:
        model_cache.invalidate(fingerprint=fingerprint, held_out_subject=str(fold.test_subject))

    cold_cache_path = model_cache._get_cache_path(fingerprint, str(fold.test_subject))

    cold_model = create_model(config.model_type, config.model_params, config.random_state)
    cold_train_start = time.time()
    cold_model.fit(X_train_selected.values, y_train)
    cold_train_time = time.time() - cold_train_start

    cold_pred_start = time.time()
    cold_pred = cold_model.predict(X_test_selected.values)
    try:
        cold_model.predict_proba(X_test_selected.values)
    except Exception:
        pass
    cold_pred_time = time.time() - cold_pred_start

    cold_cache_write_start = time.time()
    model_cache.put(
        fingerprint,
        str(fold.test_subject),
        cold_model,
        model_type=config.model_type,
        training_time=cold_train_time,
    )
    cold_cache_write_time = time.time() - cold_cache_write_start

    cold_total_time = fs_time + cold_train_time + cold_pred_time + cold_cache_write_time

    warm_cache_start = time.time()
    warm_model = model_cache.get(
        fingerprint,
        str(fold.test_subject),
        model_type=config.model_type,
    )
    warm_cache_load_time = time.time() - warm_cache_start

    if warm_model is None:
        raise RuntimeError(f"Cache load failed for {config.get_config_id()} / {fold.test_subject}")

    warm_pred_start = time.time()
    warm_pred = warm_model.predict(X_test_selected.values)
    try:
        warm_model.predict_proba(X_test_selected.values)
    except Exception:
        pass
    warm_pred_time = time.time() - warm_pred_start

    warm_total_time = fs_time + warm_cache_load_time + warm_pred_time

    predictions_match = bool(np.array_equal(cold_pred, warm_pred))

    return {
        "config_id": config.get_config_id(),
        "model_type": config.model_type,
        "held_out_subject": str(fold.test_subject),
        "fingerprint": fingerprint,
        "cache_path": str(cold_cache_path),
        "feature_selection": asdict(config.feature_selection),
        "n_train": int(len(X_train_selected)),
        "n_test": int(len(X_test_selected)),
        "n_selected_features": int(X_train_selected.shape[1]),
        "feature_selection_time_seconds": round(fs_time, 4),
        "cold": {
            "train_time_seconds": round(cold_train_time, 4),
            "predict_time_seconds": round(cold_pred_time, 4),
            "cache_write_time_seconds": round(cold_cache_write_time, 4),
            "total_time_seconds": round(cold_total_time, 4),
            "accuracy": float((cold_pred == y_test).mean()),
            "predictions_match_warm": predictions_match,
        },
        "warm": {
            "cache_load_time_seconds": round(warm_cache_load_time, 4),
            "predict_time_seconds": round(warm_pred_time, 4),
            "total_time_seconds": round(warm_total_time, 4),
            "cache_hit": True,
            "accuracy": float((warm_pred == y_test).mean()),
        },
        "speedup_cold_vs_warm": round(cold_total_time / warm_total_time, 2) if warm_total_time > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one LOSO fold per config with cold vs warm cache timing")
    parser.add_argument("--feature-cache-dir", type=str, default=str(DEFAULT_GLOBAL_CACHE_DIR))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--subject-id", type=str, default=None, help="Held-out subject, e.g. Subject_1")
    parser.add_argument("--n-channels", type=int, default=6, choices=[6, 8])
    parser.add_argument("--include-fnn", action="store_true", help="Include the FNN model if PyTorch is installed")
    parser.add_argument("--keep-cache", action="store_true", help="Do not clear the benchmark cache before running")
    args = parser.parse_args()

    feature_cache_dir = Path(args.feature_cache_dir)
    cache_dir = Path(args.cache_dir)

    logger.info("=" * 78)
    logger.info("Single-Fold LOSO Cache Benchmark")
    logger.info("=" * 78)
    logger.info(f"Feature cache dir: {feature_cache_dir}")
    logger.info(f"Benchmark cache dir: {cache_dir}")

    if cache_dir.exists() and not args.keep_cache:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    features_df, labels, subject_ids = load_cached_dataset(feature_cache_dir, n_channels=args.n_channels)
    unique_subjects = sorted({str(subject_id) for subject_id in subject_ids}, key=lambda value: int(value.split("_")[-1]))

    target_subject = args.subject_id or unique_subjects[0]
    if target_subject not in unique_subjects:
        raise ValueError(f"Subject not found in cached data: {target_subject}")

    fold = get_fold_for_subject(features_df, labels, subject_ids, target_subject)
    logger.info(f"Held-out subject: {fold.test_subject}")
    logger.info(f"Train epochs: {fold.n_train:,}")
    logger.info(f"Test epochs:  {fold.n_test:,}")

    configs = build_thesis_configs(include_fnn=args.include_fnn)
    logger.info(f"Configurations: {len(configs)}")

    results: List[Dict[str, Any]] = []
    for index, config in enumerate(configs, 1):
        logger.info("")
        logger.info(f"[{index}/{len(configs)}] {config.get_config_id()}")
        try:
            result = run_single_config_benchmark(
                config=config,
                fold=fold,
                features_df=features_df,
                labels=labels,
                cache_dir=cache_dir,
                clear_matching_cache=True,
            )
            results.append(result)

            logger.info(
                f"  cold={result['cold']['total_time_seconds']:.2f}s | "
                f"warm={result['warm']['total_time_seconds']:.2f}s | "
                f"load={result['warm']['cache_load_time_seconds']:.3f}s | "
                f"speedup={result['speedup_cold_vs_warm']}x"
            )
        except Exception as exc:
            logger.error(f"  FAILED: {exc}")
            results.append({
                "config_id": config.get_config_id(),
                "model_type": config.model_type,
                "status": "error",
                "error": str(exc),
            })

    output = {
        "timestamp": datetime.now().isoformat(),
        "feature_cache_dir": str(feature_cache_dir),
        "cache_dir": str(cache_dir),
        "held_out_subject": target_subject,
        "n_channels": args.n_channels,
        "n_subjects": int(len(unique_subjects)),
        "n_configs": len(configs),
        "results": results,
    }

    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"single_fold_cache_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    logger.info("")
    logger.info(f"Saved benchmark report to {output_path}")


if __name__ == "__main__":
    main()
