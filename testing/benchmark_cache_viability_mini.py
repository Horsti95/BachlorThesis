#!/usr/bin/env python3
"""
Mini Cache Viability Benchmark
==============================

This is a compact verification harness for the cache viability methodology.

It runs a small subject subset, a reduced configuration set, and records:
- cold training time
- warm cache load time
- cache file size
- speedup ratio

Intended use:
- sanity-check the benchmark pipeline quickly
- compare relative behavior across representative configs
- validate that cache size and warm-load time move as expected

Important:
- The historical table with SVM, AdaBoost, LightGBM, etc. cannot be reproduced
  from the current repository because those models are not implemented here.
- This script verifies the methodology on the models that do exist in the codebase.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cross_validation import CVFold, LOSOCrossValidator
from feature_cache import load_features_from_cache, select_channel_features
from feature_selection import FeatureSelectionConfig
from fingerprint import LOSOFingerprint, __version__ as FINGERPRINT_VERSION
from loso_cache import LOSOModelCache
from models import create_model
from training import TrainingConfig


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


DEFAULT_FEATURE_CACHE_DIR = Path("./results/features_cache_global")
DEFAULT_CACHE_DIR = Path("./results/mini_viability_cache")


@dataclass(frozen=True)
class LegacyModelSpec:
    model_type: str
    category: str
    params: Dict[str, Any]
    label: str


LEGACY_MODEL_SPECS: List[LegacyModelSpec] = [
    LegacyModelSpec(
        model_type="svm_linear",
        category="svm",
        params={"kernel": "linear", "C": 1.0, "probability": True, "class_weight": "balanced"},
        label="svm_linear",
    ),
    LegacyModelSpec(
        model_type="adaboost",
        category="boosting",
        params={"n_estimators": 100, "learning_rate": 1.0},
        label="adaboost",
    ),
    LegacyModelSpec(
        model_type="gradient_boosting",
        category="boosting",
        params={"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
        label="gradient_boosting",
    ),
    LegacyModelSpec(
        model_type="logistic_regression",
        category="linear",
        params={"max_iter": 1000, "solver": "lbfgs", "multi_class": "auto", "class_weight": "balanced"},
        label="logistic_regression",
    ),
    LegacyModelSpec(
        model_type="decision_tree",
        category="tree",
        params={"class_weight": "balanced"},
        label="decision_tree",
    ),
    LegacyModelSpec(
        model_type="catboost",
        category="boosting",
        params={"iterations": 200, "depth": 6, "learning_rate": 0.1},
        label="catboost",
    ),
    LegacyModelSpec(
        model_type="ridge_classifier",
        category="linear",
        params={"alpha": 1.0, "class_weight": "balanced"},
        label="ridge_classifier",
    ),
    LegacyModelSpec(
        model_type="naive_bayes",
        category="probabilistic",
        params={},
        label="naive_bayes",
    ),
    LegacyModelSpec(
        model_type="xgboost",
        category="boosting",
        params={"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        label="xgboost",
    ),
    LegacyModelSpec(
        model_type="svm_rbf",
        category="svm",
        params={"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True, "class_weight": "balanced"},
        label="svm_rbf",
    ),
    LegacyModelSpec(
        model_type="lightgbm",
        category="boosting",
        params={"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 31},
        label="lightgbm",
    ),
    LegacyModelSpec(
        model_type="random_forest",
        category="tree_ensemble",
        params={"n_estimators": 200, "max_depth": None, "class_weight": "balanced"},
        label="random_forest",
    ),
    LegacyModelSpec(
        model_type="knn_5",
        category="distance",
        params={"n_neighbors": 5, "weights": "uniform"},
        label="knn_5",
    ),
    LegacyModelSpec(
        model_type="extra_trees",
        category="tree_ensemble",
        params={"n_estimators": 200, "max_depth": None, "class_weight": "balanced"},
        label="extra_trees",
    ),
    LegacyModelSpec(
        model_type="knn_10",
        category="distance",
        params={"n_neighbors": 10, "weights": "uniform"},
        label="knn_10",
    ),
]


def _subject_sort_key(cache_file: Path) -> int:
    subject_id = cache_file.stem.replace("subject_", "").replace("_full", "")
    try:
        return int(subject_id)
    except ValueError:
        return 10**9


def load_cached_dataset(cache_dir: Path, max_subjects: int, n_channels: int = 6) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not cache_dir.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_dir}")

    subject_files = sorted(cache_dir.glob("subject_*_full.npz"), key=_subject_sort_key)[:max_subjects]
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

    return (
        pd.concat(all_features, ignore_index=True),
        np.concatenate(all_labels),
        np.array(all_subject_ids),
    )


def get_fold_for_subject(features_df: pd.DataFrame, labels: np.ndarray, subject_ids: np.ndarray, target_subject: str) -> CVFold:
    cv = LOSOCrossValidator(verbose=False)
    for fold in cv.split(features_df, labels, subject_ids):
        if fold.test_subject == target_subject:
            return fold
    raise ValueError(f"Subject not found in LOSO split: {target_subject}")


def build_mini_configs() -> List[Dict[str, Any]]:
    """Build a one-row-per-model verification set matching the historical table."""
    no_op_fs = FeatureSelectionConfig(
        correlation_threshold=None,
        top_k_features=None,
        selection_method="anova",
        scope="global",
        random_state=42,
        use_hybrid=True,
    )

    configs: List[Dict[str, Any]] = []
    for spec in LEGACY_MODEL_SPECS:
        configs.append({
            "spec": spec,
            "config": TrainingConfig(
                model_type=spec.model_type,
                model_params=spec.params,
                feature_selection=no_op_fs,
                random_state=42,
            ),
        })
    return configs


def _fit_feature_selection(
    config: TrainingConfig,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    X_full: pd.DataFrame,
    y_full: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], float]:
    fs_start = time.time()
    selected_features = list(X_full.columns)
    return X_train[selected_features], X_test[selected_features], selected_features, time.time() - fs_start


def run_config_once(
    config: TrainingConfig,
    category: str,
    fold: CVFold,
    features_df: pd.DataFrame,
    labels: np.ndarray,
    cache_dir: Path,
) -> Dict[str, Any]:
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
    model_cache.invalidate(fingerprint=fingerprint, held_out_subject=str(fold.test_subject))
    cache_path = model_cache._get_cache_path(fingerprint, str(fold.test_subject))

    cold_model = create_model(config.model_type, config.model_params, config.random_state)
    cold_train_start = time.time()
    cold_model.fit(X_train_selected.values, y_train)
    cold_train_time = time.time() - cold_train_start

    cold_predict_start = time.time()
    cold_pred = cold_model.predict(X_test_selected.values)
    try:
        cold_model.predict_proba(X_test_selected.values)
    except Exception:
        pass
    cold_predict_time = time.time() - cold_predict_start

    cold_cache_write_start = time.time()
    model_cache.put(
        fingerprint,
        str(fold.test_subject),
        cold_model,
        model_type=config.model_type,
        training_time=cold_train_time,
    )
    cold_cache_write_time = time.time() - cold_cache_write_start

    cache_size_bytes = cache_path.stat().st_size if cache_path.exists() else 0

    warm_load_start = time.time()
    warm_model = model_cache.get(fingerprint, str(fold.test_subject), model_type=config.model_type)
    warm_cache_load_time = time.time() - warm_load_start

    if warm_model is None:
        raise RuntimeError(f"Warm cache load failed for {config.get_config_id()} / {fold.test_subject}")

    warm_predict_start = time.time()
    warm_pred = warm_model.predict(X_test_selected.values)
    try:
        warm_model.predict_proba(X_test_selected.values)
    except Exception:
        pass
    warm_predict_time = time.time() - warm_predict_start

    cold_total_time = fs_time + cold_train_time + cold_predict_time + cold_cache_write_time
    warm_total_time = fs_time + warm_cache_load_time + warm_predict_time

    return {
        "config_id": config.get_config_id(),
        "model_type": config.model_type,
        "category": category,
        "held_out_subject": str(fold.test_subject),
        "cache_path": str(cache_path),
        "cache_size_bytes": int(cache_size_bytes),
        "cache_size_mb": round(cache_size_bytes / (1024 * 1024), 3),
        "feature_selection_time_seconds": round(fs_time, 4),
        "cold": {
            "train_time_seconds": round(cold_train_time, 4),
            "predict_time_seconds": round(cold_predict_time, 4),
            "cache_write_time_seconds": round(cold_cache_write_time, 4),
            "total_time_seconds": round(cold_total_time, 4),
            "accuracy": float((cold_pred == y_test).mean()),
        },
        "warm": {
            "cache_load_time_seconds": round(warm_cache_load_time, 4),
            "predict_time_seconds": round(warm_predict_time, 4),
            "total_time_seconds": round(warm_total_time, 4),
            "accuracy": float((warm_pred == y_test).mean()),
        },
        "speedup_cold_vs_warm": round(cold_total_time / warm_total_time, 2) if warm_total_time > 0 else None,
        "predictions_match": bool(np.array_equal(cold_pred, warm_pred)),
        "mb_per_second_saved": round(cache_size_bytes / (1024 * 1024) / max(cold_total_time - warm_total_time, 1e-9), 4),
    }


def _verdict(mb_per_second_saved: float) -> str:
    if mb_per_second_saved < 0.5:
        return "VIABLE"
    if mb_per_second_saved <= 2.0:
        return "BORDERLINE"
    return "NOT_VIABLE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini verification benchmark for cache viability")
    parser.add_argument("--feature-cache-dir", type=str, default=str(DEFAULT_FEATURE_CACHE_DIR))
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--subjects", type=int, default=20, help="Number of cached subjects to include")
    parser.add_argument("--subject-id", type=str, default=None, help="Held-out subject, e.g. Subject_1")
    parser.add_argument("--n-channels", type=int, default=6, choices=[6, 8])
    parser.add_argument("--keep-cache", action="store_true", help="Do not clear the benchmark cache before running")
    parser.add_argument("--save-csv", action="store_true", help="Save a CSV summary")
    args = parser.parse_args()

    feature_cache_dir = Path(args.feature_cache_dir)
    cache_dir = Path(args.cache_dir)

    logger.info("=" * 78)
    logger.info("Mini Cache Viability Benchmark")
    logger.info("=" * 78)
    logger.info(f"Feature cache dir: {feature_cache_dir}")
    logger.info(f"Benchmark cache dir: {cache_dir}")
    logger.info(f"Subjects used: {args.subjects}")
    logger.info(f"Models: {len(LEGACY_MODEL_SPECS)} historical table entries")

    if cache_dir.exists() and not args.keep_cache:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    features_df, labels, subject_ids = load_cached_dataset(feature_cache_dir, max_subjects=args.subjects, n_channels=args.n_channels)
    unique_subjects = sorted({str(subject_id) for subject_id in subject_ids}, key=lambda value: int(value.split("_")[-1]))

    target_subject = args.subject_id or unique_subjects[0]
    if target_subject not in unique_subjects:
        raise ValueError(f"Subject not found in cached data: {target_subject}")

    fold = get_fold_for_subject(features_df, labels, subject_ids, target_subject)
    logger.info(f"Held-out subject: {fold.test_subject}")
    logger.info(f"Train epochs: {fold.n_train:,}")
    logger.info(f"Test epochs:  {fold.n_test:,}")

    base_configs = build_mini_configs()
    logger.info(f"Configurations: {len(base_configs)}")

    results: List[Dict[str, Any]] = []
    for index, config in enumerate(base_configs, 1):
        logger.info("")
        spec = config["spec"]
        model_config = config["config"]
        logger.info(f"[{index}/{len(base_configs)}] {spec.label}")
        try:
            result = run_config_once(
                config=model_config,
                category=spec.category,
                fold=fold,
                features_df=features_df,
                labels=labels,
                cache_dir=cache_dir,
            )
            result["verdict"] = _verdict(result["mb_per_second_saved"])
            result["status"] = "success"
            results.append(result)
            logger.info(
                f"  cold={result['cold']['total_time_seconds']:.2f}s | "
                f"warm={result['warm']['total_time_seconds']:.2f}s | "
                f"load={result['warm']['cache_load_time_seconds']:.3f}s | "
                f"size={result['cache_size_mb']:.2f}MB | mb/s-saved={result['mb_per_second_saved']:.4f} | {result['verdict']}"
            )
        except Exception as exc:
            logger.warning(f"  skipped: {exc}")
            results.append({
                "config_id": model_config.get_config_id(),
                "model_type": spec.model_type,
                "category": spec.category,
                "status": "skipped",
                "error": str(exc),
            })

    output = {
        "timestamp": datetime.now().isoformat(),
        "feature_cache_dir": str(feature_cache_dir),
        "cache_dir": str(cache_dir),
        "held_out_subject": target_subject,
        "subjects_used": int(len(unique_subjects)),
        "models": [spec.label for spec in LEGACY_MODEL_SPECS],
        "results": results,
    }

    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"mini_cache_viability_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    if args.save_csv:
        csv_path = results_dir / f"mini_cache_viability_{timestamp}.csv"
        pd.DataFrame(results).to_csv(csv_path, index=False)
        logger.info(f"Saved CSV report to {csv_path}")

    if results:
        print("\n" + "=" * 110)
        print("CACHE VIABILITY MINI SUMMARY")
        print("=" * 110)
        print(f"{'Model':<24} {'Category':<15} {'Cold/fold':<12} {'Warm/fold':<12} {'Speedup':<10} {'Cache/fold':<12} {'MB/s-saved':<12} {'Verdict':<11} {'Acc':<8}")
        print("-" * 110)
        for result in results:
            if result.get("status") != "success":
                print(f"{result.get('model_type', 'unknown'):<24} {result.get('category', 'unknown'):<15} {'SKIPPED':<12} {'SKIPPED':<12} {'-':<10} {'-':<12} {'-':<12} {'SKIPPED':<11} {'-':<8}")
                continue
            print(
                f"{result['model_type']:<24} "
                f"{result['category']:<15} "
                f"{result['cold']['total_time_seconds']:<12.3f} "
                f"{result['warm']['total_time_seconds']:<12.3f} "
                f"{result['speedup_cold_vs_warm']:<10} "
                f"{result['cache_size_mb']:<12.2f} "
                f"{result['mb_per_second_saved']:<12.4f} "
                f"{result['verdict']:<11} "
                f"{result['warm']['accuracy']:<8.4f}"
            )
        print("-" * 110)

    logger.info("")
    logger.info(f"Saved benchmark report to {output_path}")


if __name__ == "__main__":
    main()