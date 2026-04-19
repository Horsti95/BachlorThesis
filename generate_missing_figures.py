"""
Generate Missing Thesis Figures
================================

Generates the two figures that require the full feature cache:
  A. Sleep stage distribution bar chart (fig_sleep_stage_distribution.pdf)
  B. Aggregated confusion matrix for best XGBoost config (fig_confusion_matrix.pdf)

Prerequisites:
  - Feature cache populated at results/features_cache_global/  (128 subjects)
  - Model cache populated (warm run already executed for best XGBoost config)
  - Run AFTER generate_thesis_figures.py

Usage:
    python generate_missing_figures.py

Output:
    thesis/figures/fig_sleep_stage_distribution.pdf
    thesis/figures/fig_confusion_matrix.pdf
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

REPO = Path(__file__).resolve().parent
FIG_DIR = REPO / "thesis" / "figures"
CACHE_DIR = REPO / "results" / "features_cache_global"
WARM_RUN_DIR = REPO / "results" / "training_20260408_181311_full" / "training_results"

CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
CLASS_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]

BEST_CONFIG = {
    "model_type": "xgboost",
    "correlation_threshold": None,
    "top_k_features": None,
    "selection_method": "anova",
    "scope": "global",
    "random_state": 42,
}


def check_feature_cache() -> bool:
    if not CACHE_DIR.exists():
        return False
    subject_dirs = list(CACHE_DIR.glob("SC*"))
    return len(subject_dirs) >= 128


def load_all_labels() -> np.ndarray:
    """Load labels from all cached subjects."""
    sys.path.insert(0, str(REPO))
    from feature_cache import load_features_from_cache

    subject_ids = sorted([d.name for d in CACHE_DIR.iterdir() if d.is_dir()])
    print(f"  Loading labels from {len(subject_ids)} subjects...")
    all_labels = []
    for sid in subject_ids:
        npz_files = list((CACHE_DIR / sid).glob("*.npz"))
        if npz_files:
            data = np.load(npz_files[0], allow_pickle=True)
            if "labels" in data:
                all_labels.append(data["labels"])
    return np.concatenate(all_labels) if all_labels else np.array([])


def fig_sleep_distribution(labels: np.ndarray) -> None:
    """Bar chart: sleep stage distribution across all 128 subjects."""
    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()

    stages = [stage_map.get(int(u), str(u)) for u in unique if int(u) in stage_map]
    cnts = [c for u, c in zip(unique, counts) if int(u) in stage_map]
    pcts = [100.0 * c / total for c in cnts]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(stages, cnts, color=CLASS_COLORS[: len(stages)], alpha=0.85,
                  edgecolor="black", linewidth=0.6)

    for bar, pct, cnt in zip(bars, pcts, cnts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f"{pct:.1f}%\n({cnt:,})",
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Sleep Stage")
    ax.set_ylabel("Number of 30-second Epochs")
    ax.set_title("Sleep Stage Distribution — BOAS Dataset (128 subjects, 128-fold LOSO)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, max(cnts) * 1.18)
    fig.tight_layout()

    out = FIG_DIR / "fig_sleep_stage_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [A] Saved {out.name}")


def fig_confusion_matrix() -> None:
    """
    Aggregate confusion matrix across all 128 LOSO folds for best XGBoost config.
    Runs the warm-cache pipeline (fast: ~7s) to collect fold predictions.
    """
    sys.path.insert(0, str(REPO))
    from run_training import load_cached_features
    from training import TrainingPipeline, TrainingConfig
    from feature_selection import FeatureSelectionConfig
    from cross_validation import LOSOCrossValidator
    from sklearn.metrics import confusion_matrix

    subject_ids = sorted([d.name for d in CACHE_DIR.iterdir() if d.is_dir()])
    print(f"  Loading features for {len(subject_ids)} subjects...")
    features_df, labels, _ = load_cached_features(subject_ids, CACHE_DIR)

    fs_config = FeatureSelectionConfig(
        correlation_threshold=BEST_CONFIG["correlation_threshold"],
        top_k_features=BEST_CONFIG["top_k_features"],
        selection_method=BEST_CONFIG["selection_method"],
        scope=BEST_CONFIG["scope"],
        random_state=BEST_CONFIG["random_state"],
    )
    config = TrainingConfig(
        model_type=BEST_CONFIG["model_type"],
        feature_selection=fs_config,
        random_state=BEST_CONFIG["random_state"],
    )

    cv = LOSOCrossValidator()
    folds = cv.get_folds(features_df, labels)

    pipeline = TrainingPipeline(features_df, labels, use_tqdm=True)
    result = pipeline.run(config, folds)

    y_true_all, y_pred_all = [], []
    for fold_result in result.fold_results:
        y_true_all.extend(fold_result.y_test.tolist())
        y_pred_all.extend(fold_result.y_pred.tolist())

    cm = confusion_matrix(y_true_all, y_pred_all)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, data, fmt, title in [
        (axes[0], cm,      "d",    "Counts"),
        (axes[1], cm_norm, ".2f", "Row-normalised (recall per class)"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    vmin=0, vmax=(1.0 if fmt == ".2f" else None),
                    ax=ax, square=True, cbar_kws={"shrink": 0.8})
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)

    fig.suptitle(
        "Aggregated Confusion Matrix — XGBoost (corr=None, k=None, 149 features)\n"
        "128-fold LOSO, 128 subjects",
        fontsize=12,
    )
    fig.tight_layout()

    out = FIG_DIR / "fig_confusion_matrix.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [B] Saved {out.name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not check_feature_cache():
        print(
            "ERROR: Feature cache not found or incomplete at:\n"
            f"  {CACHE_DIR}\n\n"
            "Run this script on the machine where the full feature cache is populated\n"
            "(the benchmark desktop used for the thesis experiments).\n"
            "Expected: 128 subject directories under results/features_cache_global/"
        )
        sys.exit(1)

    print("Generating missing thesis figures...")
    labels = load_all_labels()
    if len(labels) == 0:
        print("ERROR: Could not load labels from feature cache.")
        sys.exit(1)

    print(f"  Total epochs loaded: {len(labels):,}")
    fig_sleep_distribution(labels)
    fig_confusion_matrix()
    print("\nDone. Both figures saved to thesis/figures/")


if __name__ == "__main__":
    main()
