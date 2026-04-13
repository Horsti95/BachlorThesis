#!/usr/bin/env python3
"""
RF Cache Interaction Benchmark
==============================

Benchmarks interaction effects across:
- RF tree count: n_estimators (e.g., 25, 50, 100, 200, 400)
- feature selection: top-k (30, 50, all)
- correlation threshold: (0.75, 0.90, None)
- subject amount: (3, 30, all available)

The script intentionally supports limited fold sampling per setting so you can
run a proof-of-concept without executing full LOSO for every combination.

Outputs:
- JSON with raw runs + aggregated summaries
- Optional CSV for easier analysis in pandas/Excel

Focus metrics:
- cold total time
- warm total time
- warm cache load time
- cache size (MB)
- time saved
- MB per second saved (cache viability proxy)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cross_validation import CVFold, LOSOCrossValidator
from feature_cache import load_features_from_cache, select_channel_features
from feature_selection import FeatureSelectionConfig, FeatureSelectionPipeline
from fingerprint import LOSOFingerprint, __version__ as FINGERPRINT_VERSION
from loso_cache import LOSOModelCache
from models import create_model
from training import TrainingConfig


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


DEFAULT_FEATURE_CACHE_DIR = Path("./results/features_cache_global")
DEFAULT_CACHE_DIR = Path("./results/loso_model_cache_rf_interactions")


def _subject_sort_key(cache_file: Path) -> int:
    subject_id = cache_file.stem.replace("subject_", "").replace("_full", "")
    try:
        return int(subject_id)
    except ValueError:
        return 10**9


def load_cached_dataset(
    cache_dir: Path, max_subjects: Optional[int], n_channels: int = 6
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not cache_dir.exists():
        raise FileNotFoundError(f"Feature cache not found: {cache_dir}")

    subject_files = sorted(cache_dir.glob("subject_*_full.npz"), key=_subject_sort_key)
    if not subject_files:
        raise FileNotFoundError(f"No subject caches found in {cache_dir}")

    if max_subjects is not None:
        subject_files = subject_files[:max_subjects]

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


def choose_folds(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    max_folds: int,
) -> List[CVFold]:
    cv = LOSOCrossValidator(verbose=False)
    folds = list(cv.split(features_df, labels, subject_ids))
    if max_folds <= 0 or max_folds >= len(folds):
        return folds
    return folds[:max_folds]


def _fit_feature_selection(
    fs_config: FeatureSelectionConfig,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    X_full: pd.DataFrame,
    y_full: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], float]:
    fs_start = time.time()

    if fs_config.correlation_threshold is None and fs_config.top_k_features is None:
        selected_features = list(X_full.columns)
        return (
            X_train[selected_features],
            X_test[selected_features],
            selected_features,
            time.time() - fs_start,
        )

    fs_pipeline = FeatureSelectionPipeline(fs_config)
    fs_pipeline.fit(X_full, y_full)
    selected_features = fs_pipeline.get_selected_features()
    return (
        X_train[selected_features],
        X_test[selected_features],
        selected_features,
        time.time() - fs_start,
    )


def run_single_fold(
    model_config: TrainingConfig,
    fs_config: FeatureSelectionConfig,
    fold: CVFold,
    features_df: pd.DataFrame,
    labels: np.ndarray,
    cache_dir: Path,
) -> Dict[str, Any]:
    X_train = features_df.iloc[fold.train_indices].copy()
    y_train = labels[fold.train_indices]
    X_test = features_df.iloc[fold.test_indices].copy()
    y_test = labels[fold.test_indices]

    X_train_selected, X_test_selected, selected_features, fs_time = (
        _fit_feature_selection(fs_config, X_train, y_train, X_test, features_df, labels)
    )

    fingerprint = LOSOFingerprint.generate(
        random_seed=model_config.random_state,
        code_version=FINGERPRINT_VERSION,
        model_name=model_config.model_type,
        model_params=model_config.model_params,
        feature_config={
            "base": X_train.shape[1],
            "corr": fs_config.correlation_threshold,
            "top_k": fs_config.top_k_features,
            "n_selected": X_train_selected.shape[1],
            "selected_features": selected_features,
        },
        held_out_subject=str(fold.test_subject),
    )

    model_cache = LOSOModelCache(
        cache_dir=str(cache_dir),
        enable_registry=True,
        estimated_training_time=120.0,
    )

    # Delete any existing cache for this exact config to force cold training
    model_cache.invalidate(
        fingerprint=fingerprint, held_out_subject=str(fold.test_subject)
    )
    cache_path = model_cache._get_cache_path(fingerprint, str(fold.test_subject))

    # --- COLD RUN: train from scratch ---
    cold_model = create_model(
        model_config.model_type, model_config.model_params, model_config.random_state
    )

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

    # Write to cache
    cold_cache_write_start = time.time()
    model_cache.put(
        fingerprint,
        str(fold.test_subject),
        cold_model,
        model_type=model_config.model_type,
        training_time=cold_train_time,
    )
    cold_cache_write_time = time.time() - cold_cache_write_start

    if not cache_path.exists():
        raise RuntimeError(f"Cache file missing after write: {cache_path}")

    cache_size_bytes = cache_path.stat().st_size

    # --- WARM RUN: load from cache ---
    warm_load_start = time.time()
    warm_model = model_cache.get(
        fingerprint,
        str(fold.test_subject),
        model_type=model_config.model_type,
    )
    warm_cache_load_time = time.time() - warm_load_start

    if warm_model is None:
        raise RuntimeError(f"Warm cache load failed for fingerprint={fingerprint}")

    warm_predict_start = time.time()
    warm_pred = warm_model.predict(X_test_selected.values)
    try:
        warm_model.predict_proba(X_test_selected.values)
    except Exception:
        pass
    warm_predict_time = time.time() - warm_predict_start

    # --- Compute metrics ---
    cold_total_time = fs_time + cold_train_time + cold_predict_time + cold_cache_write_time
    warm_total_time = fs_time + warm_cache_load_time + warm_predict_time
    time_saved_seconds = max(cold_total_time - warm_total_time, 0.0)
    cache_size_mb = cache_size_bytes / (1024 * 1024)
    mb_per_second_saved = cache_size_mb / max(time_saved_seconds, 1e-9)

    return {
        "held_out_subject": str(fold.test_subject),
        "n_train": int(len(X_train_selected)),
        "n_test": int(len(X_test_selected)),
        "n_selected_features": int(X_train_selected.shape[1]),
        "corr": fs_config.correlation_threshold,
        "top_k": fs_config.top_k_features,
        "cold_total_time_seconds": float(cold_total_time),
        "cold_train_time_seconds": float(cold_train_time),
        "warm_total_time_seconds": float(warm_total_time),
        "warm_cache_load_time_seconds": float(warm_cache_load_time),
        "feature_selection_time_seconds": float(fs_time),
        "cache_write_time_seconds": float(cold_cache_write_time),
        "cache_size_mb": float(cache_size_mb),
        "time_saved_seconds": float(time_saved_seconds),
        "speedup_cold_vs_warm": float(cold_total_time / warm_total_time)
        if warm_total_time > 0
        else None,
        "mb_per_second_saved": float(mb_per_second_saved),
        "accuracy_cold": float((cold_pred == y_test).mean()),
        "accuracy_warm": float((warm_pred == y_test).mean()),
        "predictions_match": bool(np.array_equal(cold_pred, warm_pred)),
    }


def summarize_runs(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"status": "empty"}

    group_cols = ["subject_count", "corr", "top_k", "tree_count"]

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            folds_evaluated=("held_out_subject", "count"),
            selected_features_mean=("n_selected_features", "mean"),
            cold_total_mean=("cold_total_time_seconds", "mean"),
            warm_total_mean=("warm_total_time_seconds", "mean"),
            load_time_mean=("warm_cache_load_time_seconds", "mean"),
            cache_size_mb_mean=("cache_size_mb", "mean"),
            time_saved_mean=("time_saved_seconds", "mean"),
            speedup_mean=("speedup_cold_vs_warm", "mean"),
            mb_per_second_saved_mean=("mb_per_second_saved", "mean"),
            predictions_match_rate=("predictions_match", "mean"),
        )
        .reset_index()
    )

    # Correlation matrix across numeric dimensions
    numeric_cols = [
        "subject_count",
        "corr",
        "top_k",
        "tree_count",
        "n_selected_features",
        "cold_total_time_seconds",
        "warm_total_time_seconds",
        "warm_cache_load_time_seconds",
        "cache_size_mb",
        "time_saved_seconds",
        "speedup_cold_vs_warm",
        "mb_per_second_saved",
    ]

    corr_df = df[numeric_cols].copy()
    corr_df["corr"] = corr_df["corr"].fillna(-1.0)
    corr_df["top_k"] = corr_df["top_k"].fillna(9999)
    correlation_matrix = corr_df.corr(numeric_only=True).round(4).to_dict()

    return {
        "n_runs": int(len(df)),
        "grouped_summary": grouped.to_dict(orient="records"),
        "correlation_matrix": correlation_matrix,
    }


def parse_list_with_all(raw_values: List[str]) -> List[Optional[int]]:
    parsed: List[Optional[int]] = []
    for raw in raw_values:
        if raw.lower() == "all":
            parsed.append(None)
        else:
            parsed.append(int(raw))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RF cache interactions across tree_count, top_k, corr, subject_count"
    )
    parser.add_argument(
        "--feature-cache-dir", type=str, default=str(DEFAULT_FEATURE_CACHE_DIR)
    )
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--n-channels", type=int, default=6, choices=[6, 8])

    parser.add_argument(
        "--tree-counts", type=int, nargs="+", default=[25, 50, 100, 200, 400]
    )
    parser.add_argument(
        "--top-k",
        type=str,
        nargs="+",
        default=["30", "50", "all"],
        help="Use integers and/or 'all'",
    )
    parser.add_argument(
        "--corr",
        type=str,
        nargs="+",
        default=["0.75", "0.90", "none"],
        help="Use floats and/or 'none'",
    )
    parser.add_argument(
        "--subject-counts",
        type=str,
        nargs="+",
        default=["3", "30", "all"],
        help="Use integers and/or 'all'",
    )

    parser.add_argument(
        "--folds-per-setting",
        type=int,
        default=3,
        help="Limit LOSO folds per setting (default: 3)",
    )
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()

    feature_cache_dir = Path(args.feature_cache_dir)
    cache_dir = Path(args.cache_dir)

    if cache_dir.exists() and not args.keep_cache:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    top_k_values = parse_list_with_all(args.top_k)

    corr_values: List[Optional[float]] = []
    for raw in args.corr:
        if raw.lower() == "none":
            corr_values.append(None)
        else:
            corr_values.append(float(raw))

    subject_counts = parse_list_with_all(args.subject_counts)

    # Count total combinations
    total_combos = (
        len(args.tree_counts)
        * len(top_k_values)
        * len(corr_values)
        * len(subject_counts)
    )
    total_runs = total_combos * args.folds_per_setting

    logger.info("=" * 90)
    logger.info("RF Cache Interaction Benchmark")
    logger.info("=" * 90)
    logger.info(f"  feature_cache_dir: {feature_cache_dir}")
    logger.info(f"  cache_dir:         {cache_dir}")
    logger.info(f"  tree_counts:       {args.tree_counts}")
    logger.info(f"  top_k:             {top_k_values}")
    logger.info(f"  corr:              {corr_values}")
    logger.info(f"  subject_counts:    {subject_counts}")
    logger.info(f"  folds_per_setting: {args.folds_per_setting}")
    logger.info(f"  total combos:      {total_combos}")
    logger.info(f"  total runs:        {total_runs}")
    logger.info("=" * 90)

    all_rows: List[Dict[str, Any]] = []
    run_counter = 0

    for subject_count in subject_counts:
        features_df, labels, subject_ids = load_cached_dataset(
            cache_dir=feature_cache_dir,
            max_subjects=subject_count,
            n_channels=args.n_channels,
        )

        selected_subject_count = int(len(np.unique(subject_ids)))
        folds = choose_folds(
            features_df, labels, subject_ids, max_folds=args.folds_per_setting
        )

        logger.info("")
        logger.info(
            f"Subject set: requested={subject_count} | actual={selected_subject_count} "
            f"| folds={len(folds)} | epochs={len(labels):,}"
        )

        for corr in corr_values:
            for top_k in top_k_values:
                fs_config = FeatureSelectionConfig(
                    correlation_threshold=corr,
                    top_k_features=top_k,
                    selection_method="anova",
                    scope="global",
                    random_state=42,
                )

                for tree_count in args.tree_counts:
                    model_config = TrainingConfig(
                        model_type="random_forest",
                        model_params={"n_estimators": int(tree_count)},
                        feature_selection=fs_config,
                        random_state=42,
                    )

                    fold_times = []
                    for fold in folds:
                        row = run_single_fold(
                            model_config=model_config,
                            fs_config=fs_config,
                            fold=fold,
                            features_df=features_df,
                            labels=labels,
                            cache_dir=cache_dir,
                        )
                        row["subject_count"] = selected_subject_count
                        row["tree_count"] = int(tree_count)
                        all_rows.append(row)
                        fold_times.append(row["cold_train_time_seconds"])
                        run_counter += 1

                    avg_cold = np.mean(fold_times)
                    last = all_rows[-1]
                    logger.info(
                        f"  [{run_counter:>4}/{total_runs}] "
                        f"subj={selected_subject_count:>3} corr={str(corr):>4} "
                        f"k={str(top_k):>4} trees={tree_count:>3} | "
                        f"cold={avg_cold:.1f}s cache={last['cache_size_mb']:.1f}MB "
                        f"mb/s={last['mb_per_second_saved']:.3f} "
                        f"match={last['predictions_match']}"
                    )

    df = pd.DataFrame(all_rows)
    summary = summarize_runs(df)

    output = {
        "timestamp": datetime.now().isoformat(),
        "params": {
            "feature_cache_dir": str(feature_cache_dir),
            "cache_dir": str(cache_dir),
            "n_channels": args.n_channels,
            "tree_counts": args.tree_counts,
            "top_k": top_k_values,
            "corr": corr_values,
            "subject_counts_requested": subject_counts,
            "folds_per_setting": args.folds_per_setting,
        },
        "summary": summary,
        "runs": all_rows,
    }

    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = results_dir / f"rf_cache_interactions_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    logger.info(f"\nSaved JSON: {json_path}")

    if args.save_csv:
        csv_path = results_dir / f"rf_cache_interactions_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        grouped_path = results_dir / f"rf_cache_interactions_summary_{timestamp}.csv"
        pd.DataFrame(summary.get("grouped_summary", [])).to_csv(
            grouped_path, index=False
        )
        logger.info(f"Saved CSV: {csv_path}")
        logger.info(f"Saved grouped summary: {grouped_path}")

    logger.info(f"Total runs: {len(df)}")
    logger.info(
        f"Predictions match rate: {df['predictions_match'].mean():.1%}"
    )


if __name__ == "__main__":
    main()
