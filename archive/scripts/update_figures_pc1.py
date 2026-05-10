"""
Update Fig 4.5 (speedup bar) and Fig 4.12 (crossover) with PC1 rerun data.

Data source: benchmark_results/pc1_rerun_20260425_summary.csv
  - 18 configurations (2 models × 3 corr thresholds × 3 feature counts)
  - 128-fold LOSO, PC1 hardware (Intel desktop)

Cache sizes come from the original RTX-5090 XGB/RF CSVs (model size is
hardware-independent; only training time and warm-load time differ).

Usage:
    python update_figures_pc1.py          # save to thesis/figures/
    python update_figures_pc1.py --show   # save and display

Output:
    thesis/figures/fig1_speedup_bar.pdf   (replaces old figure)
    thesis/figures/fig3b_crossover.pdf    (replaces old figure)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO       = Path(__file__).resolve().parents[2]
FIG_DIR    = REPO / "thesis" / "figures"
RESULTS    = REPO / "results"
BENCH      = REPO / "benchmark_results"

PC1_CSV    = BENCH / "pc1_rerun_20260425_summary.csv"
XGB_CSV    = RESULTS / "xgb_cache_evaluation_20260314_151728.csv"
RF_CSV     = RESULTS / "rf_cache_evaluation_full_9configs.csv"

# Empirically measured n_features for each (corr, top_k) combo on BOAS dataset
_N_FEATURES = {
    (0.75, 30):   30,
    (0.75, 50):   48,
    (0.75, None): 48,
    (0.9,  30):   30,
    (0.9,  50):   50,
    (0.9,  None): 84,
    (None, 30):   30,
    (None, 50):   50,
    (None, None): 149,
}


def _load_pc1() -> pd.DataFrame:
    df = pd.read_csv(PC1_CSV)
    # Normalise correlation_threshold (CSV stores NaN for None)
    df["corr_key"] = df["correlation_threshold"].apply(
        lambda x: None if (pd.isna(x) or str(x).lower() == "nan") else float(x)
    )
    df["topk_key"] = df["top_k_features"].apply(
        lambda x: None if (pd.isna(x) or str(x).lower() == "nan") else int(x)
    )
    df["n_features"] = df.apply(
        lambda r: _N_FEATURES.get((r["corr_key"], r["topk_key"]), 0), axis=1
    )
    # Per-fold times (128 folds)
    df["cold_per_fold_s"] = df["cold_seconds"] / 128
    df["warm_per_fold_s"] = df["warm_seconds"] / 128
    return df


def _cache_sizes_by_nf() -> dict:
    """Return {(model, n_features): mean_cache_mb} from original RTX 5090 CSVs."""
    xgb = pd.read_csv(XGB_CSV)
    rf  = pd.read_csv(RF_CSV)
    sizes = {}
    for nf, grp in xgb.groupby("n_features"):
        sizes[("xgboost", int(nf))] = grp["cache_size_mb"].mean()
    for nf, grp in rf.groupby("n_features"):
        sizes[("random_forest", int(nf))] = grp["cache_size_mb"].mean()
    return sizes


# =====================================================================
# Fig 1 — Speedup Bar Chart
# =====================================================================
def fig1_speedup_bar(show: bool):
    df = _load_pc1()

    xgb = df[df["model"] == "xgboost"]
    rf  = df[df["model"] == "random_forest"]

    data = {
        "XGBoost\n(9 configs)": {
            "cold_h": xgb["cold_seconds"].sum() / 3600,
            "warm_min": xgb["warm_seconds"].sum() / 60,
            "speedup": xgb["cold_seconds"].sum() / xgb["warm_seconds"].sum(),
        },
        "Random Forest\n(9 configs)": {
            "cold_h": rf["cold_seconds"].sum() / 3600,
            "warm_min": rf["warm_seconds"].sum() / 60,
            "speedup": rf["cold_seconds"].sum() / rf["warm_seconds"].sum(),
        },
    }

    labels    = list(data.keys())
    cold_vals = [d["cold_h"] * 60 for d in data.values()]  # minutes
    warm_vals = [d["warm_min"]     for d in data.values()]

    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, cold_vals, w, label="Cold (no cache)", color="#D55E00")
    ax.bar(x + w / 2, warm_vals, w, label="Warm (cached)",   color="#0072B2")

    ax.set_yscale("log")
    ax.set_ylim(0.5, max(cold_vals) * 3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_ylabel("Total time (minutes, log scale)")
    ax.set_title("Cache Speedup: Cold Start vs. Cached Execution (128 LOSO folds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for i, lbl in enumerate(labels):
        sp = data[lbl]["speedup"]
        ax.annotate(
            f"{sp:.0f}×\nspeedup",
            xy=(x[i] - w / 2, cold_vals[i]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=10,
            color="#222222",
        )

    fig.tight_layout()
    out = FIG_DIR / "fig1_speedup_bar.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    if show:
        plt.show()
    plt.close(fig)


# =====================================================================
# Fig 3b — Compute-to-I/O Crossover
# =====================================================================
def fig3b_crossover(show: bool):
    df    = _load_pc1()
    sizes = _cache_sizes_by_nf()

    xgb_df = df[df["model"] == "xgboost"]
    rf_df  = df[df["model"] == "random_forest"]

    # Average per unique feature count across correlation thresholds
    xgb_avg = (
        xgb_df.groupby("n_features")[["cold_per_fold_s", "warm_per_fold_s"]]
        .mean()
        .reset_index()
        .sort_values("n_features")
    )
    rf_avg = (
        rf_df.groupby("n_features")[["cold_per_fold_s", "warm_per_fold_s"]]
        .mean()
        .reset_index()
        .sort_values("n_features")
    )

    # Attach per-fold cache sizes (total / 128 folds)
    xgb_avg["cache_per_fold_mb"] = xgb_avg["n_features"].apply(
        lambda nf: sizes.get(("xgboost", nf), np.nan) / 128
    )
    rf_avg["cache_per_fold_mb"] = rf_avg["n_features"].apply(
        lambda nf: sizes.get(("random_forest", nf), np.nan) / 128
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- XGBoost panel ---
    ax1.plot(xgb_avg["n_features"], xgb_avg["cold_per_fold_s"], "o-",
             color="#D55E00", label="Cold (re-train)", linewidth=2, markersize=7)
    ax1.plot(xgb_avg["n_features"], xgb_avg["warm_per_fold_s"], "s-",
             color="#0072B2", label="Warm (cache load)", linewidth=2, markersize=7)
    ax1.fill_between(xgb_avg["n_features"], xgb_avg["warm_per_fold_s"],
                     xgb_avg["cold_per_fold_s"], alpha=0.15, color="#0072B2",
                     label="Time saved")
    for _, row in xgb_avg.iterrows():
        if not np.isnan(row["cache_per_fold_mb"]):
            ax1.annotate(f"{row['cache_per_fold_mb']:.1f} MB",
                         (row["n_features"], row["cold_per_fold_s"]),
                         textcoords="offset points", xytext=(0, 10),
                         fontsize=7.5, ha="center", color="#555555",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                   ec="none", alpha=0.7))
    ax1.set_xlabel("Number of features")
    ax1.set_ylabel("Time per fold (seconds)")
    ax1.set_title("XGBoost — ~1.5 MB/fold, loads in ~0.2s")
    ax1.legend(fontsize=8, loc="center left")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v:.2f}" if v < 1 else f"{v:.0f}"))

    # --- RF panel ---
    ax2.plot(rf_avg["n_features"], rf_avg["cold_per_fold_s"], "o-",
             color="#D55E00", label="Cold (re-train)", linewidth=2, markersize=7)
    ax2.plot(rf_avg["n_features"], rf_avg["warm_per_fold_s"], "s-",
             color="#0072B2", label="Warm (cache load)", linewidth=2, markersize=7)
    ax2.fill_between(rf_avg["n_features"], rf_avg["warm_per_fold_s"],
                     rf_avg["cold_per_fold_s"], alpha=0.15, color="#E69F00",
                     label="Small gap = I/O bottleneck")

    rf_min = rf_avg["cache_per_fold_mb"].min()
    rf_max = rf_avg["cache_per_fold_mb"].max()
    ax2.annotate(
        f"Cache: {rf_min:.0f}–{rf_max:.0f} MB/fold\n(200 trees, size ≈ const.)",
        xy=(rf_avg["n_features"].iloc[-1], rf_avg["cold_per_fold_s"].iloc[-1]),
        xytext=(0.97, 0.85), textcoords="axes fraction",
        fontsize=8, ha="right", color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#999999", lw=1),
    )
    ax2.set_xlabel("Number of features")
    ax2.set_ylabel("Time per fold (seconds)")
    rf_avg_cache = rf_avg["cache_per_fold_mb"].mean()
    ax2.set_title(f"Random Forest — ~{rf_avg_cache:.0f} MB/fold, loads in ~3s")
    ax2.legend(fontsize=8, loc="upper left")

    fig.suptitle("XGBoost vs. Random Forest: Why Model Size Determines Cache Viability",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = FIG_DIR / "fig3b_crossover.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Update figures with PC1 rerun data")
    parser.add_argument("--show", action="store_true", help="Display figures after saving")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Updating figures with PC1 rerun data ({PC1_CSV.name})...")

    fig1_speedup_bar(args.show)
    fig3b_crossover(args.show)

    print("Done.")


if __name__ == "__main__":
    main()
