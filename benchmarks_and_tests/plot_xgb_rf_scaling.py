"""
plot_xgb_rf_scaling.py - Visualize XGB+RF Cache Speedup vs Subject Count
========================================================================

Consumes the summary CSV produced by benchmark_xgb_rf_scaling.ps1 and produces
a two-panel figure suitable for inclusion in Chapter 4 (RQ3 Scalability):

    Left panel : per-fold cold and warm time vs subject count (log y).
                 Cold grows; warm stays roughly flat.
    Right panel: speedup ratio vs subject count (linear y, log x).
                 Both XGB and RF curves trend upward.

The figure visually demonstrates that cache speedup is not a fixed property
of the model but grows with the dataset, which is the substantive claim of
RQ3 along the subject axis.

Usage (from project root):
    python benchmarks_and_tests/plot_xgb_rf_scaling.py \
        --summary results/xgb_rf_scaling_<...>/xgb_rf_scaling_summary.csv \
        --out thesis/figures/fig_xgb_rf_subject_scaling.pdf

If --summary points to multiple CSVs (comma-separated), the script overlays
them as separate machine tags (e.g. PC1 vs RTX 5090) using line styles.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
}
MODEL_COLORS = {
    "xgboost": "#0072B2",
    "random_forest": "#D55E00",
}
TAG_LINESTYLES = {
    "": "-",
    "PC1": "-",
    "5090": "--",
}


def load_summary(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        # Ensure required columns are present
        required = ["Subjects", "Model", "ColdPerFold_s", "WarmPerFold_s", "Speedup"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            sys.exit(f"ERROR: {p} missing columns: {missing}")
        if "Tag" not in df.columns:
            df["Tag"] = ""
        df["Tag"] = df["Tag"].fillna("").astype(str)
        df["__source__"] = str(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def plot_scaling(df: pd.DataFrame, out_path: Path, show: bool):
    fig, (ax_time, ax_sp) = plt.subplots(1, 2, figsize=(11, 4.5))

    tags = sorted(df["Tag"].unique())

    # ----- Left panel: cold/warm per-fold time -----
    for model in ["xgboost", "random_forest"]:
        for tag in tags:
            sub = df[(df["Model"] == model) & (df["Tag"] == tag)].sort_values("Subjects")
            if sub.empty:
                continue
            ls = TAG_LINESTYLES.get(tag, "-")
            label_suffix = f" [{tag}]" if tag else ""

            ax_time.plot(
                sub["Subjects"], sub["ColdPerFold_s"],
                marker="o", linestyle=ls,
                color=MODEL_COLORS[model],
                label=f"{MODEL_LABELS[model]} cold{label_suffix}",
            )
            ax_time.plot(
                sub["Subjects"], sub["WarmPerFold_s"],
                marker="s", linestyle=ls,
                color=MODEL_COLORS[model], alpha=0.55,
                label=f"{MODEL_LABELS[model]} warm{label_suffix}",
            )

    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("Subjects in LOSO sweep")
    ax_time.set_ylabel("Per-fold time (seconds, log)")
    ax_time.set_title("(a) Cold vs. warm per-fold time")
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.legend(fontsize=8, loc="best")

    # ----- Right panel: speedup -----
    for model in ["xgboost", "random_forest"]:
        for tag in tags:
            sub = df[(df["Model"] == model) & (df["Tag"] == tag)].sort_values("Subjects")
            if sub.empty:
                continue
            ls = TAG_LINESTYLES.get(tag, "-")
            label_suffix = f" [{tag}]" if tag else ""
            ax_sp.plot(
                sub["Subjects"], sub["Speedup"],
                marker="o", linestyle=ls, linewidth=2,
                color=MODEL_COLORS[model],
                label=f"{MODEL_LABELS[model]}{label_suffix}",
            )
            # Annotate first and last points with their speedup
            for x, y in [
                (sub["Subjects"].iloc[0], sub["Speedup"].iloc[0]),
                (sub["Subjects"].iloc[-1], sub["Speedup"].iloc[-1]),
            ]:
                ax_sp.annotate(
                    f"{y:.0f}×",
                    xy=(x, y), xytext=(4, 6),
                    textcoords="offset points",
                    fontsize=9, fontweight="bold",
                    color=MODEL_COLORS[model],
                )

    ax_sp.set_xscale("log")
    ax_sp.set_xlabel("Subjects in LOSO sweep")
    ax_sp.set_ylabel("Cache speedup ratio (×)")
    ax_sp.set_title("(b) Speedup grows with dataset size")
    ax_sp.grid(True, which="both", alpha=0.3)
    ax_sp.legend(fontsize=9, loc="best")

    fig.suptitle(
        "XGBoost and Random Forest cache speedup vs. subject count",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def print_table(df: pd.DataFrame):
    """Print a wide table grouped by model and tag for the thesis text."""
    print()
    print("=" * 80)
    print("XGB + RF SUBJECT-AXIS SCALING")
    print("=" * 80)
    for tag in sorted(df["Tag"].unique()):
        if tag:
            print(f"\n[Tag: {tag}]")
        for model in ["xgboost", "random_forest"]:
            sub = df[(df["Model"] == model) & (df["Tag"] == tag)].sort_values("Subjects")
            if sub.empty:
                continue
            print(f"\n  {MODEL_LABELS[model]}:")
            print(f"    {'Subjects':>10} {'Cold/fold':>12} {'Warm/fold':>12} "
                  f"{'Speedup':>10} {'Cache/fold':>14}")
            for _, r in sub.iterrows():
                print(f"    {int(r['Subjects']):>10} "
                      f"{r['ColdPerFold_s']:>10.3f}s  "
                      f"{r['WarmPerFold_s']:>10.3f}s  "
                      f"{r['Speedup']:>8.1f}×  "
                      f"{r['CachePerFold_MB']:>10.2f} MB")
    print()


def main():
    parser = argparse.ArgumentParser(description="Plot XGB+RF subject-axis scaling")
    parser.add_argument("--summary", required=True,
                        help="Path to xgb_rf_scaling_summary.csv "
                             "(comma-separate multiple paths to overlay machines)")
    parser.add_argument("--out", default="thesis/figures/fig_xgb_rf_subject_scaling.pdf",
                        help="Output PDF path")
    parser.add_argument("--show", action="store_true", help="Show figure interactively")
    args = parser.parse_args()

    paths = [Path(p.strip()) for p in args.summary.split(",")]
    for p in paths:
        if not p.exists():
            sys.exit(f"ERROR: summary CSV not found: {p}")

    df = load_summary(paths)
    print_table(df)
    plot_scaling(df, Path(args.out), show=args.show)


if __name__ == "__main__":
    main()
