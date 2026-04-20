"""
Generate Missing Thesis Figures
================================

Generates the confusion matrix figure (fig_confusion_matrix.pdf).
The sleep stage distribution figure is now handled by generate_distribution_figure.py.

Prerequisites:
  - Feature cache populated (128 subject_N_full.npz files)
  - Model cache populated (warm run already executed for best XGBoost config)

Usage:
    python generate_missing_figures.py
    python generate_missing_figures.py --cache "C:/Users/DerHo/Desktop/BachlorThesis/results/features_cache_global"

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

REPO    = Path(__file__).resolve().parent
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
    """Extract numeric subject IDs from subject_N_full.npz filenames."""
    files = sorted(cache_dir.glob("subject_*_full.npz"))
    ids = []
    for f in files:
        parts = f.stem.split("_")  # subject_N_full
        if len(parts) >= 2 and parts[1].isdigit():
            ids.append(int(parts[1]))
    return sorted(ids)


def check_cache(cache_dir: Path) -> bool:
    if not cache_dir.exists():
        return False
    return len(list(cache_dir.glob("subject_*_full.npz"))) >= 128


def fig_confusion_matrix(cache_dir: Path) -> None:
    import tempfile
    sys.path.insert(0, str(REPO))
    from run_training import load_cached_features
    from training import TrainingPipeline, TrainingConfig
    from feature_selection import FeatureSelectionConfig
    from sklearn.metrics import confusion_matrix

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

    with tempfile.TemporaryDirectory() as tmp:
        pipeline = TrainingPipeline(
            features_df, labels, subject_id_array,
            output_dir=Path(tmp),
        )
        print("  Running warm-cache LOSO pipeline (~7s)...")
        result = pipeline.run_single_config(config)

    y_true_all, y_pred_all = [], []
    for fold_result in result.fold_results:
        y_true_all.extend(fold_result.y_test.tolist())
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
    args = parser.parse_args()

    cache_dir = Path(args.cache)

    if not check_cache(cache_dir):
        print(f"ERROR: Feature cache not found or incomplete at:\n  {cache_dir}")
        print("Expected: 128 files named subject_N_full.npz")
        print('Usage: python generate_missing_figures.py --cache "C:/path/to/features_cache_global"')
        sys.exit(1)

    print(f"Using cache: {cache_dir}")
    fig_confusion_matrix(cache_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
