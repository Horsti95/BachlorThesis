"""
Generate Missing Thesis Figures
================================

Generates the confusion matrix figure (fig_confusion_matrix.pdf).

Uses the best XGBoost configuration (corr=None, top-k=None, 149 features)
over all 128 LOSO folds to produce an aggregated confusion matrix consistent
with the performance numbers in Tab 4.18 (PC1 rerun data).

Prerequisites:
  - Feature cache populated (128 subject_N_full.npz files)
  - Strongly recommended: model cache from a previous full training run
    (warm run ~7s instead of cold run ~43 min)

Usage (5090 with model cache — fastest, ~7s):
    python generate_missing_figures.py \\
        --cache  "/path/to/features_cache_global" \\
        --model-cache "/path/to/loso_model_cache"

Usage (PC1 cold run — ~43 min):
    python generate_missing_figures.py \\
        --cache "C:/Users/DerHo/Desktop/Data/features_cache_global"

Output:
    thesis/figures/fig_confusion_matrix.pdf
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

REPO    = Path(__file__).resolve().parents[2]
FIG_DIR = REPO / "thesis" / "figures"

DEFAULT_CACHE = REPO / "results" / "features_cache_global"

CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]

BEST_CONFIG = {
    "model_type": "xgboost",
    "correlation_threshold": None,
    "top_k_features": None,
    "selection_method": "anova",
    "scope": "global",
    "random_state": 42,
}


def get_subject_ids(cache_dir: Path) -> list:
    files = sorted(cache_dir.glob("subject_*_full.npz"))
    ids = []
    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            ids.append(int(parts[1]))
    return sorted(ids)


def check_cache(cache_dir: Path) -> bool:
    if not cache_dir.exists():
        return False
    return len(list(cache_dir.glob("subject_*_full.npz"))) >= 128


def fig_confusion_matrix(cache_dir: Path, model_cache_dir: Path = None) -> None:
    import tempfile
    sys.path.insert(0, str(REPO))
    from run_training import load_cached_features
    from training import TrainingPipeline, TrainingConfig
    from feature_selection import FeatureSelectionConfig
    from sklearn.metrics import confusion_matrix

    # Always use a persistent model cache so interrupted runs resume from cache
    if model_cache_dir is None:
        model_cache_dir = cache_dir.parent / "confusion_matrix_model_cache"
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    already = len(list(model_cache_dir.glob("*.joblib")))
    print(f"  Model cache: {model_cache_dir}")
    print(f"  Already cached folds: {already} (will skip these on warm run)")

    subject_ids = get_subject_ids(cache_dir)
    print(f"  Found {len(subject_ids)} subjects in cache.")
    print(f"  Loading features...")
    features_df, labels, subject_id_array = load_cached_features(subject_ids, cache_dir)

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

    out_dir = cache_dir.parent / "confusion_matrix_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = TrainingPipeline(
        features_df, labels, subject_id_array,
        output_dir=out_dir,
        model_cache_dir=str(model_cache_dir),
        enable_model_cache=True,
    )
    print("  Running LOSO pipeline (cached folds load in ~0.2s each)...")
    result = pipeline.run_single_config(config, show_progress=True)

    y_true_all, y_pred_all = [], []
    for fold_result in result.fold_results:
        y_true_all.extend(fold_result.y_true.tolist())
        y_pred_all.extend(fold_result.y_pred.tolist())

    cm      = confusion_matrix(y_true_all, y_pred_all)
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

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_confusion_matrix.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(DEFAULT_CACHE),
                        help="Path to features_cache_global directory")
    parser.add_argument("--model-cache", default=None,
                        help="Path to model cache dir from a previous training run "
                             "(warm run ~7s instead of cold run ~43 min)")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    model_cache_dir = Path(args.model_cache) if args.model_cache else None

    if not check_cache(cache_dir):
        print(f"ERROR: Feature cache not found or incomplete at:\n  {cache_dir}")
        print("Expected: 128 files named subject_N_full.npz")
        print('Usage: python generate_missing_figures.py --cache "C:/path/to/features_cache_global"')
        sys.exit(1)

    print(f"Feature cache: {cache_dir}")
    fig_confusion_matrix(cache_dir, model_cache_dir)
    print("\nDone. Commit thesis/figures/fig_confusion_matrix.pdf and push.")


if __name__ == "__main__":
    main()
