"""
Thesis Figure & Table Generator
================================

Generates all figures (PDF) and LaTeX tables for the bachelor thesis
from existing result data. No re-computation needed.

Covers:
  1.  Speedup bar chart (cold vs warm, per model type)
  2.  Performance table (18 configs, Acc/Kappa/F1)
  3.  Compute-to-I/O crossover (RF vs XGBoost) — averaged per feature count
  4.  Cache viability table (15 models)
  4b. Cache viability scatter plot (speedup vs cache size, threshold line)
  5.  Fingerprint integrity example
  6.  Per-class F1 heatmap (all 18 configs)
  7.  Feature importance top-15 (ANOVA)
  8.  Global vs fold-specific tradeoff note
  9.  SVM scaling plot (speedup grows with dataset size)

Usage:
    python generate_thesis_figures.py              # save only
    python generate_thesis_figures.py --show       # save and display

Output:
    thesis/figures/*.pdf   — matplotlib figures
    thesis/tables/*.tex    — LaTeX tables

Author: Lennart Gorzel (generated with AI assistance)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
VIABILITY = RESULTS / "viability_benchmarks"
FIG_DIR = REPO / "thesis" / "figures"
TAB_DIR = REPO / "thesis" / "tables"

XGB_CSV = RESULTS / "xgb_cache_evaluation_20260314_151728.csv"
RF_CSV = RESULTS / "rf_cache_evaluation_full_9configs.csv"
VIABILITY_CSV = VIABILITY / "5090_128subj" / "cache_viability_128subj.csv"
FEATURE_SEL_JSON = VIABILITY / "feature_selection_methods" / "feature_selection_benchmark.json"
WARM_RUN_DIR = RESULTS / "training_20260408_181311_full" / "training_results"
SVM_SCALING_DIR = VIABILITY / "5090_svm_scaling"
FEATURE_SIZE_DIR = RESULTS / "feature_size_20260315_001142"
RF_INTERACTIONS_CSV = RESULTS / "rf_cache_interactions_20260413_204548.csv"
RF_INTERACTIONS_SUMMARY = RESULTS / "rf_cache_interactions_summary_20260413_204548.csv"


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def _fix_table(latex: str) -> str:
    """Post-process pandas to_latex output: add [htbp] placement and \\centering."""
    latex = latex.replace(r"\begin{table}", r"\begin{table}[htbp]")
    latex = latex.replace(
        "\\begin{table}[htbp]\n",
        "\\begin{table}[htbp]\n\\centering\n",
    )
    return latex


# ===================================================================
# 1. Speedup Bar Chart — cold vs warm per model type
# ===================================================================
def fig1_speedup_bar(show: bool):
    import matplotlib.pyplot as plt

    # Values taken directly from tab:xgb_cache_results and tab:rf_cache_results
    # (HP ProBook benchmark, 128 LOSO folds, 9 configurations each).
    xgb_cold_s = np.array([1053.4, 1447.4, 1440.3, 964.1, 1453.8, 2223.6, 1003.6, 1385.9, 3816.1])
    xgb_warm_s = np.array([26.4, 27.2, 26.6, 25.7, 26.4, 28.2, 21.1, 21.2, 18.9])
    rf_cold_s  = np.array([4341.3, 4638.0, 4648.3, 4342.7, 5794.2, 6315.9, 4567.2, 5602.4, 7962.4])
    rf_warm_s  = np.array([391.8, 402.8, 401.3, 392.2, 375.3, 359.0, 415.6, 356.7, 339.4])

    # Per-model summary: totals for the cold/warm bars, and median per-config
    # speedup for the annotation (consistent with the median values reported
    # in tab:xgb_cache_results and tab:rf_cache_results).
    data = {
        "XGBoost\n(9 configs)": {
            "cold_h": xgb_cold_s.sum() / 3600,
            "warm_min": xgb_warm_s.sum() / 60,
            "speedup": np.median(xgb_cold_s / xgb_warm_s),
        },
        "Random Forest\n(9 configs)": {
            "cold_h": rf_cold_s.sum() / 3600,
            "warm_min": rf_warm_s.sum() / 60,
            "speedup": np.median(rf_cold_s / rf_warm_s),
        },
    }

    labels = list(data.keys())
    cold_vals = [d["cold_h"] * 60 for d in data.values()]  # in minutes
    warm_vals = [d["warm_min"] for d in data.values()]

    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_cold = ax.bar(x - w / 2, cold_vals, w, label="Cold (no cache)", color="#D55E00")
    bars_warm = ax.bar(x + w / 2, warm_vals, w, label="Warm (cached)", color="#0072B2")

    ax.set_yscale("log")
    ax.set_ylim(0.5, max(cold_vals) * 3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_ylabel("Total time (minutes, log scale)")
    ax.set_title("Cache Speedup: Cold Start vs. Cached Execution (128 LOSO folds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Annotate speedup above the cold bar (log scale → position near top of cold bar)
    for i, lbl in enumerate(labels):
        sp = data[lbl]["speedup"]
        ax.annotate(
            f"{sp:.1f}×\nmedian speedup",
            xy=(x[i] - w / 2, cold_vals[i]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=10,
            color="#222222",
        )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_speedup_bar.pdf", bbox_inches="tight")
    print(f"  [1/10] Saved fig1_speedup_bar.pdf")
    if show:
        plt.show()
    plt.close(fig)


# ===================================================================
# 2. Performance Table — 18 configs
# ===================================================================
def tab2_performance(show: bool):
    rows = []
    for f in sorted(WARM_RUN_DIR.glob("result_*.json")):
        d = json.loads(f.read_text())
        rows.append({
            "Model": d["config"]["model_type"],
            "Corr": d["config"]["feature_selection"].get("correlation_threshold"),
            "Top-K": d["config"]["feature_selection"].get("top_k_features"),
            "Accuracy": d["accuracy_mean"],
            "Acc Std": d["accuracy_std"],
            "Kappa": d["kappa_mean"],
            "F1-Macro": d["f1_macro_mean"],
        })

    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    df_disp = df.copy()
    df_disp["\\#"] = range(1, len(df_disp) + 1)
    df_disp["Model"] = df_disp["Model"].map(
        {"xgboost": "XGBoost", "random_forest": "Random Forest"}).fillna(df_disp["Model"])
    df_disp["Corr"] = df_disp["Corr"].apply(
        lambda x: "---" if x is None or str(x) in ("None", "nan") else str(x))
    df_disp["Top-K"] = df_disp["Top-K"].apply(
        lambda x: "All" if x is None or str(x) in ("None", "nan") else str(int(float(x))))
    df_disp["Accuracy"] = df_disp.apply(
        lambda r: f"{r['Accuracy']:.3f} $\\pm$ {r['Acc Std']:.3f}", axis=1)
    df_disp["Kappa"] = df_disp["Kappa"].apply(lambda x: f"{x:.3f}")
    df_disp["F1-Macro"] = df_disp["F1-Macro"].apply(lambda x: f"{x:.3f}")
    df_disp = df_disp[["\\#", "Model", "Corr", "Top-K", "Accuracy", "Kappa", "F1-Macro"]]

    latex = df_disp.to_latex(
        caption="Classification performance of all 18 configurations (128-fold LOSO, global ANOVA selection).",
        label="tab:performance_18configs",
        column_format="rlllrrr",
        escape=False,
        index=False,
    )
    latex = _fix_table(latex)
    (TAB_DIR / "tab2_performance_18configs.tex").write_text(latex)
    print(f"  [2/10] Saved tab2_performance_18configs.tex")

    if show:
        print("\n--- Table 2: Performance (18 configs) ---")
        print(df_disp.to_string())
        print()


# ===================================================================
# 3. Cache Efficiency: Compute Savings vs. Storage Cost (all 15 models)
# ===================================================================
def fig3_efficiency(show: bool):
    import matplotlib.pyplot as plt

    df = pd.read_csv(VIABILITY_CSV)

    # Compute time saved per fold (cold - warm)
    df["time_saved_per_fold"] = df["cold_per_fold_time_s"] - df["warm_per_fold_time_s"]

    colors = {"VIABLE": "#0072B2", "NOT_VIABLE": "#D55E00", "BORDERLINE": "#E69F00"}

    fig, ax = plt.subplots(figsize=(10, 6))

    for verdict in ["VIABLE", "NOT_VIABLE"]:
        subset = df[df["cache_verdict"] == verdict]
        ax.scatter(
            subset["cache_size_per_fold_mb"],
            subset["time_saved_per_fold"],
            c=colors[verdict],
            s=120,
            edgecolors="black",
            linewidth=0.5,
            label=f"{verdict.replace('_', ' ').title()}  (η > 2 s/MB)" if verdict == "VIABLE"
                  else f"{verdict.replace('_', ' ').title()}  (η < 0.5 s/MB)",
            zorder=5,
        )

    # Label each point
    for _, row in df.iterrows():
        name = row["model_name"]
        # Offset to avoid overlap
        x_off, y_off = 8, 0
        if name == "knn_10":
            y_off = -15
        elif name == "svm_rbf":
            y_off = -15
        elif name == "ridge_classifier":
            y_off = 10
        elif name == "naive_bayes":
            y_off = 10
        ax.annotate(
            name, (row["cache_size_per_fold_mb"], row["time_saved_per_fold"]),
            textcoords="offset points", xytext=(x_off, y_off), fontsize=7.5,
            color="#333333",
        )

    # η = 2 s/MB boundary: time_saved = 2 × cache_size  (VIABLE above)
    # η = 0.5 s/MB boundary: time_saved = 0.5 × cache_size  (NOT_VIABLE below)
    x_line = np.logspace(-3, 3, 100)
    ax.plot(x_line, 2.0 * x_line, "--", color="#888888", linewidth=1.5, alpha=0.6,
            label="Viable boundary (η = 2 s/MB)")
    ax.plot(x_line, 0.5 * x_line, ":", color="#D55E00", linewidth=1.2, alpha=0.5,
            label="Not-viable boundary (η = 0.5 s/MB)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cache size per fold (MB)", fontsize=11)
    ax.set_ylabel("Time saved per fold (seconds)", fontsize=11)
    ax.set_title("Cache viability: compute time saved per MB stored\n"
                 "Above the dashed boundary, training time recovered exceeds disk-I/O cost",
                 fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2, which="both")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_efficiency.pdf", bbox_inches="tight")
    print(f"  [3/10] Saved fig3_efficiency.pdf")
    if show:
        plt.show()
    plt.close(fig)


# ===================================================================
# 3b. Compute-to-I/O Crossover — RF vs XGBoost (averaged per feature count)
#     Each point annotated with cache size per fold
# ===================================================================
def fig3b_crossover(show: bool):
    import matplotlib.pyplot as plt

    xgb = pd.read_csv(XGB_CSV)
    rf = pd.read_csv(RF_CSV)

    # Average per unique feature count (removes noise from different corr thresholds)
    # Also average the cache_size_mb
    xgb_avg = xgb.groupby("n_features")[["cold_per_fold_s", "warm_per_fold_s", "cache_size_mb"]].mean().reset_index()
    rf_avg = rf.groupby("n_features")[["cold_per_fold_s", "warm_per_fold_s", "cache_size_mb"]].mean().reset_index()
    # Per-fold cache size: total / 128 folds
    xgb_avg["cache_per_fold_mb"] = xgb_avg["cache_size_mb"] / 128
    rf_avg["cache_per_fold_mb"] = rf_avg["cache_size_mb"] / 128
    xgb_avg = xgb_avg.sort_values("n_features")
    rf_avg = rf_avg.sort_values("n_features")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- XGBoost panel ---
    ax1.plot(xgb_avg["n_features"], xgb_avg["cold_per_fold_s"], "o-",
             color="#D55E00", label="Cold (re-train)", linewidth=2, markersize=7)
    ax1.plot(xgb_avg["n_features"], xgb_avg["warm_per_fold_s"], "s-",
             color="#0072B2", label="Warm (cache load)", linewidth=2, markersize=7)
    ax1.fill_between(xgb_avg["n_features"], xgb_avg["warm_per_fold_s"],
                     xgb_avg["cold_per_fold_s"], alpha=0.15, color="#0072B2",
                     label="Time saved")
    # Annotate cold points with cache size (above the cold line, centered)
    for _, row in xgb_avg.iterrows():
        ax1.annotate(f"{row['cache_per_fold_mb']:.1f} MB",
                     (row["n_features"], row["cold_per_fold_s"]),
                     textcoords="offset points", xytext=(0, 10),
                     fontsize=7.5, ha="center", color="#555555",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ax1.set_xlabel("Number of features")
    ax1.set_ylabel("Time per fold (seconds)")
    ax1.set_title("XGBoost — ~1.5 MB/fold, loads in 0.05s")
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
    # Single summary annotation — RF cache size is ~131-154 MB/fold regardless of
    # feature count because it serializes 200 full trees (tree structure dominates)
    avg_cache = rf_avg["cache_per_fold_mb"].mean()
    min_cache = rf_avg["cache_per_fold_mb"].min()
    max_cache = rf_avg["cache_per_fold_mb"].max()
    ax2.annotate(f"Cache: {min_cache:.0f}–{max_cache:.0f} MB/fold\n(200 trees, size ≈ const.)",
                 xy=(rf_avg["n_features"].iloc[-1], rf_avg["cold_per_fold_s"].iloc[-1]),
                 xytext=(0.97, 0.85), textcoords="axes fraction",
                 fontsize=8, ha="right", color="#555555",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", color="#999999", lw=1))
    ax2.set_xlabel("Number of features")
    ax2.set_ylabel("Time per fold (seconds)")
    ax2.set_title("Random Forest — ~{:.0f} MB/fold, loads in 2.5s".format(avg_cache))
    ax2.legend(fontsize=8, loc="upper left")

    fig.suptitle("XGBoost vs. Random Forest: Why Model Size Determines Cache Viability",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig3b_crossover.pdf", bbox_inches="tight")
    print(f"  [3b/10] Saved fig3b_crossover.pdf")
    if show:
        plt.show()
    plt.close(fig)


# ===================================================================
# 4. Cache Viability Table — 15 models
# ===================================================================
def tab4_viability(show: bool):
    _model_names = {
        "gradient_boosting": "Gradient Boosting", "adaboost": "AdaBoost",
        "svm_linear": "SVM Linear",  "decision_tree": "Decision Tree",
        "catboost": "CatBoost",      "svm_rbf": "SVM RBF",
        "xgboost": "XGBoost",        "lightgbm": "LightGBM",
        "logistic_regression": "Logistic Reg.", "random_forest": "Random Forest",
        "naive_bayes": "Naive Bayes", "knn_5": "kNN ($k$=5)",
        "knn_10": "kNN ($k$=10)",    "extra_trees": "Extra Trees",
        "ridge_classifier": "Ridge Classifier",
    }
    _verdicts = {"VIABLE": "Viable", "NOT_VIABLE": "Not Viable", "BORDERLINE": "Borderline"}

    df = pd.read_csv(VIABILITY_CSV)
    df = df.sort_values("speedup_ratio", ascending=False).reset_index(drop=True)

    df_disp = df[["model_name", "category", "cold_per_fold_time_s", "warm_per_fold_time_s",
                   "speedup_ratio", "cache_size_per_fold_mb", "cold_accuracy",
                   "accuracy_match", "cache_verdict"]].copy()
    df_disp.columns = ["Model", "Category", "Cold/fold (s)", "Warm/fold (s)",
                        "Speedup", "Cache/fold (MB)", "Accuracy", "Match", "Verdict"]

    df_disp.insert(0, "Rank", range(1, len(df_disp) + 1))
    df_disp["Model"]        = df_disp["Model"].map(_model_names).fillna(df_disp["Model"])
    df_disp["Category"]     = df_disp["Category"].str.replace("_", " ")
    df_disp["Cold/fold (s)"]= df_disp["Cold/fold (s)"].apply(lambda x: f"{x:.2f}")
    df_disp["Warm/fold (s)"]= df_disp["Warm/fold (s)"].apply(lambda x: f"{x:.3f}")
    df_disp["Speedup"]      = df_disp["Speedup"].apply(lambda x: f"${x:,.0f}\\times$")
    df_disp["Cache/fold (MB)"] = df_disp["Cache/fold (MB)"].apply(lambda x: f"{x:.2f}")
    df_disp["Accuracy"]     = df_disp["Accuracy"].apply(lambda x: f"{x:.3f}")
    df_disp["Match"]        = df_disp["Match"].map({True: r"\checkmark", False: r"$\times$"})
    df_disp["Verdict"]      = df_disp["Verdict"].map(_verdicts).fillna(df_disp["Verdict"])

    latex = df_disp.to_latex(
        caption="Cache viability across 15 model types (128 subjects, 5 LOSO folds, RTX 5090 machine).",
        label="tab:cache_viability",
        column_format="rllrrrrrcl",
        escape=False,
        index=False,
    )
    latex = _fix_table(latex)
    (TAB_DIR / "tab4_cache_viability.tex").write_text(latex)
    print(f"  [4/10] Saved tab4_cache_viability.tex")

    if show:
        print("\n--- Table 4: Cache Viability (15 models) ---")
        print(df_disp.to_string())
        print()


# ===================================================================
# 4b. Cache Viability Scatter — MB/s-saved metric with defined thresholds
# ===================================================================
def fig4b_viability_scatter(show: bool):
    import matplotlib.pyplot as plt

    df = pd.read_csv(VIABILITY_CSV)
    # Compute η = seconds saved per MB (higher = better)
    df["eta_s_per_mb"] = 1.0 / df["mb_per_second_saved"]
    df = df.sort_values("eta_s_per_mb")  # ascending: not-viable at bottom

    colors_map = {"VIABLE": "#0072B2", "NOT_VIABLE": "#D55E00", "BORDERLINE": "#E69F00"}
    bar_colors = [colors_map.get(v, "#999") for v in df["cache_verdict"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["eta_s_per_mb"], color=bar_colors,
            edgecolor="white", linewidth=0.5)

    labels = [f"{row['model_name']}  ({row['speedup_ratio']:.0f}×, "
              f"{row['cache_size_per_fold_mb']:.1f} MB/fold)"
              for _, row in df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)

    # Threshold lines: VIABLE η > 2 s/MB, NOT_VIABLE η < 0.5 s/MB
    ax.axvline(x=2.0, color="#888888", linestyle="--", linewidth=2, alpha=0.8,
               label="Viable threshold (η > 2 s/MB)")
    ax.axvline(x=0.5, color="#D55E00", linestyle=":", linewidth=2, alpha=0.7,
               label="Not-viable threshold (η < 0.5 s/MB)")

    ax.set_xscale("log")
    ax.set_xlabel("Seconds of compute saved per MB of cache  η  [s/MB]  (higher = more efficient)",
                  fontsize=11)
    ax.set_title("Cache Viability: Compute Value per MB of Storage\n"
                 "Blue = viable (η > 2), Vermillion = not viable (η < 0.5)",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2, axis="x", which="both")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4b_viability_scatter.pdf", bbox_inches="tight")
    print(f"  [4b/10] Saved fig4b_viability_scatter.pdf")
    if show:
        plt.show()
    plt.close(fig)


# ===================================================================
# 4c. Dual-Metric Viability: η (s/MB) and Speedup/MB side by side
# ===================================================================
def fig4c_dual_metrics(show: bool):
    """Two-panel horizontal bar chart showing η (s/MB) and Speedup/MB.

    Both panels share the same y-axis (sorted by η so rank shifts in the
    right panel reveal models whose relative efficiency differs from their
    absolute efficiency). This is the paper-friendly alternative to a 3-D plot.
    """
    import matplotlib.pyplot as plt

    df = pd.read_csv(VIABILITY_CSV)
    df = df[df["model_name"] != "knn_10"].reset_index(drop=True)

    _model_names = {
        "gradient_boosting": "Gradient Boosting", "adaboost": "AdaBoost",
        "svm_linear": "SVM Linear",  "decision_tree": "Decision Tree",
        "catboost": "CatBoost",      "svm_rbf": "SVM RBF",
        "xgboost": "XGBoost",        "lightgbm": "LightGBM",
        "logistic_regression": "Logistic Reg.", "random_forest": "Random Forest",
        "naive_bayes": "Naive Bayes", "knn_5": r"kNN ($k$=5)",
        "extra_trees": "Extra Trees", "ridge_classifier": "Ridge Classifier",
    }

    # Compute metrics; cap near-zero denominators
    df["eta_s_per_mb"] = df["mb_per_second_saved"].apply(
        lambda v: 1.0 / v if v > 1e-6 else 1e7
    )
    df["speedup_per_mb"] = df.apply(
        lambda r: r["speedup_ratio"] / r["cache_size_per_fold_mb"]
        if r["cache_size_per_fold_mb"] > 0.001 else 1e7,
        axis=1,
    )

    # Sort ascending by η so the most efficient model appears at the TOP
    df = df.sort_values("eta_s_per_mb", ascending=True).reset_index(drop=True)

    colors_map = {"VIABLE": "#0072B2", "NOT_VIABLE": "#D55E00", "BORDERLINE": "#E69F00"}
    bar_colors = [colors_map.get(v, "#999") for v in df["cache_verdict"]]

    y_pos = np.arange(len(df))

    # Display cap (log scale, just for rendering)
    ETA_DISPLAY_CAP   = 2e5
    SP_MB_DISPLAY_CAP = 5e5
    eta_disp   = df["eta_s_per_mb"].clip(upper=ETA_DISPLAY_CAP)
    sp_mb_disp = df["speedup_per_mb"].clip(upper=SP_MB_DISPLAY_CAP)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig.subplots_adjust(wspace=0.04)

    # ── Panel A: η (s/MB) ──────────────────────────────────────────
    ax1.barh(y_pos, eta_disp, color=bar_colors, edgecolor="white", linewidth=0.4)
    ax1.axvline(x=2.0, color="#009E73", linestyle="--", linewidth=1.8, alpha=0.85,
                label="Viable  η > 2 s/MB")
    ax1.axvline(x=0.5, color="#D55E00", linestyle=":",  linewidth=1.5, alpha=0.7,
                label="Not-viable  η < 0.5 s/MB")
    ax1.set_xscale("log")
    ax1.set_xlabel(
        "Absolute efficiency  η  [s/MB]\n"
        "seconds of training saved per MB stored   (higher → better)",
        fontsize=9.5,
    )
    ax1.set_title("Absolute Efficiency  (η)", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.2, axis="x", which="both")
    ax1.invert_yaxis()

    # ── Panel B: Speedup / MB ──────────────────────────────────────
    ax2.barh(y_pos, sp_mb_disp, color=bar_colors, edgecolor="white", linewidth=0.4)
    # Threshold lies between RF (0.11) and SVM-RBF (2.2) → use 1 ×/MB
    ax2.axvline(x=1.0, color="#009E73", linestyle="--", linewidth=1.8, alpha=0.85,
                label="Approx. viable  > 1 ×/MB")
    ax2.set_xscale("log")
    ax2.set_xlabel(
        "Relative efficiency  [×/MB]\n"
        "speedup factor per MB stored   (higher → better)",
        fontsize=9.5,
    )
    ax2.set_title("Relative Efficiency  (Speedup/MB)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.2, axis="x", which="both")

    # Y-axis labels on left panel
    _BOLD_MODELS = {"XGBoost", "Random Forest"}
    ylabels = [_model_names.get(r["model_name"], r["model_name"]) for _, r in df.iterrows()]
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ylabels, fontsize=10)
    for lbl in ax1.get_yticklabels():
        if lbl.get_text() in _BOLD_MODELS:
            lbl.set_fontweight("bold")

    fig.suptitle(
        "Cache Viability: Absolute vs. Relative Efficiency\n"
        "Blue = VIABLE  |  Red = NOT VIABLE  |  Sorted by η (absolute)",
        fontsize=11,
        fontweight="bold",
    )

    out = FIG_DIR / "fig4c_dual_metrics.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  [4c] Saved fig4c_dual_metrics.pdf")
    if show:
        plt.show()
    plt.close(fig)


# ===================================================================
# 5. Fingerprint Integrity Example
# ===================================================================
def tab5_fingerprint(show: bool):
    from hashlib import sha256

    examples = [
        {"model": "xgboost", "corr": "None", "top_k": "149", "seed": 42, "subject": "sub-001"},
        {"model": "xgboost", "corr": "None", "top_k": "149", "seed": 42, "subject": "sub-002"},
        {"model": "xgboost", "corr": "0.90", "top_k": "50", "seed": 42, "subject": "sub-001"},
        {"model": "random_forest", "corr": "None", "top_k": "149", "seed": 42, "subject": "sub-001"},
    ]

    _tab5_names = {
        "xgboost": "XGBoost", "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting", "svm_linear": "SVM Linear",
    }

    rows = []
    for ex in examples:
        raw = f"v1.0|{ex['seed']}|{ex['model']}|corr={ex['corr']}|k={ex['top_k']}|{ex['subject']}"
        h = sha256(raw.encode()).hexdigest()[:32]
        rows.append({
            "Model": _tab5_names.get(ex["model"], ex["model"]),
            "Corr": ex["corr"],
            "Top-K": ex["top_k"],
            "Subject": ex["subject"],
            "Fingerprint (first 32 hex)": h,
        })

    df = pd.DataFrame(rows)
    df["Changed"] = ["---", "subject", "corr + top-k", "model"]

    latex = df.to_latex(
        index=False,
        caption="Fingerprint integrity: changing any parameter produces a unique SHA-256 hash.",
        label="tab:fingerprint_integrity",
        column_format="llllll",
        escape=False,
    )
    latex = _fix_table(latex)
    (TAB_DIR / "tab5_fingerprint_integrity.tex").write_text(latex)
    print(f"  [5/10] Saved tab5_fingerprint_integrity.tex")

    if show:
        print("\n--- Table 5: Fingerprint Integrity ---")
        print(df.to_string(index=False))
        print()


# ===================================================================
# 6. Per-Class F1 Heatmap (best config)
# ===================================================================
def fig6_per_class_f1(show: bool):
    import matplotlib.pyplot as plt

    # Collect per-class F1 for all 18 configs
    rows = []
    for f in sorted(WARM_RUN_DIR.glob("result_*.json")):
        d = json.loads(f.read_text())
        row = {"config": d["config_id"], "model": d["config"]["model_type"]}
        row.update(d["f1_per_class_mean"])
        row["F1-Macro"] = d["f1_macro_mean"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("F1-Macro", ascending=False).reset_index(drop=True)
    stages = ["Wake", "N1", "N2", "N3", "REM"]
    heatdata = df[stages].values

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(heatdata, cmap="viridis", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_yticks(range(len(df)))

    # Short labels
    ylabels = []
    for _, r in df.iterrows():
        m = "XGB" if r["model"] == "xgboost" else "RF"
        cfg = r["config"].replace("xgboost_", "").replace("random_forest_", "")
        cfg = cfg.replace("_anova_glo", "")
        ylabels.append(f"{m} {cfg}")
    ax.set_yticklabels(ylabels, fontsize=8)

    # Annotate cells
    for i in range(heatdata.shape[0]):
        for j in range(heatdata.shape[1]):
            val = heatdata[i, j]
            color = "white" if val < 0.35 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title("Per-Class F1 Score Across All 18 Configurations", fontsize=12)
    ax.set_xlabel("Sleep Stage")
    fig.colorbar(im, ax=ax, label="F1 Score", shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_per_class_f1.pdf", bbox_inches="tight")
    print(f"  [6/10] Saved fig6_per_class_f1.pdf")
    if show:
        plt.show()
    plt.close(fig)


# ===================================================================
# 7. Feature Importance Top-15 (ANOVA)
# ===================================================================
def fig7_feature_importance(show: bool):
    import matplotlib.pyplot as plt

    with open(FEATURE_SEL_JSON) as f:
        data = json.load(f)

    features = data["details"]["anova_f_classif"]["top_features"][:15]
    # Assign descending scores for visualization (we don't have raw F-values, use rank)
    scores = list(range(15, 0, -1))

    # Color by channel
    channel_colors = {
        "O1": "#D55E00", "O2": "#D55E00",
        "F3": "#0072B2", "F4": "#0072B2",
        "C3": "#009E73", "C4": "#009E73",
        "EMG": "#E69F00", "EOG": "#E69F00",
        "global": "#CC79A7",
    }

    colors = []
    for feat in features:
        ch = feat.split("_")[0] if "_" in feat else "global"
        colors.append(channel_colors.get(ch, "#95a5a6"))

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("ANOVA F-statistic Rank")
    ax.set_title("Top 15 Features Selected by ANOVA (f_classif)")

    # Legend for channels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#D55E00", label="O1/O2 (Occipital)"),
        Patch(facecolor="#0072B2", label="F3/F4 (Frontal)"),
        Patch(facecolor="#009E73", label="C3/C4 (Central)"),
        Patch(facecolor="#E69F00", label="EMG"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_feature_importance.pdf", bbox_inches="tight")
    print(f"  [7/10] Saved fig7_feature_importance.pdf")
    if show:
        plt.show()
    plt.close(fig)


# ===================================================================
# 8. Global vs Fold-Specific Tradeoff Note (LaTeX snippet)
# ===================================================================
def tab8_global_vs_fold(show: bool):
    note = r"""% Global vs. Fold-Specific Feature Selection Tradeoff
% Data source: benchmark_global_vs_perfold.py
%   N={3,5,10,20} subjects x XGBoost + RF x 3 corr thresholds, top-k=50
%
\begin{table}[htbp]
\centering
\caption{Global vs.\ fold-specific ANOVA feature selection: empirical comparison
         across subject counts (XGBoost and Random Forest, top-$k=50$, 3 correlation
         thresholds each).}
\label{tab:global_vs_fold}
\begin{tabular}{lcc}
\toprule
\textbf{Aspect} & \textbf{Global} & \textbf{Fold-Specific} \\
\midrule
Selection fitted on        & All subjects (once)         & Training fold only \\
Label leakage              & $1/128 \approx 0.8\%$       & None \\
Accuracy difference        & $\pm 0.4\%$ mean, $\leq 1.1\%$ max & Baseline \\
XGBoost warm speedup       & $14$--$23\times$            & $4$--$9\times$ \\
Random Forest warm speedup & $3$--$5\times$              & $2.5$--$2.8\times$ \\
Speedup degrades with $N$  & No (fingerprint stable)     & Yes (re-select per fold) \\
Cache key complexity       & One key per config          & One key per fold \\
Thesis choice              & \checkmark                  & --- \\
\bottomrule
\end{tabular}
\end{table}
"""
    (TAB_DIR / "tab8_global_vs_fold.tex").write_text(note)
    print(f"  [8/10] Saved tab8_global_vs_fold.tex")

    if show:
        print("\n--- Table 8: Global vs Fold-Specific ---")
        print(note)


# ===================================================================
# BONUS: SVM Scaling Plot (speedup grows with dataset size)
# ===================================================================
def fig_bonus_svm_scaling(show: bool):
    import matplotlib.pyplot as plt

    subjects = []
    svm_rbf_speedup = []
    svm_linear_speedup = []

    for subdir in ["10subj", "30subj", "64subj", "128subj"]:
        p = SVM_SCALING_DIR / subdir
        csvs = list(p.glob("*.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        n = int(subdir.replace("subj", ""))
        subjects.append(n)
        for _, row in df.iterrows():
            if row["model_name"] == "svm_rbf":
                svm_rbf_speedup.append(row["speedup_ratio"])
            elif row["model_name"] == "svm_linear":
                svm_linear_speedup.append(row["speedup_ratio"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(subjects, svm_rbf_speedup, "o-", color="#D55E00", linewidth=2, markersize=8, label="SVM-RBF")
    ax.plot(subjects, svm_linear_speedup, "s-", color="#0072B2", linewidth=2, markersize=8, label="SVM-Linear")

    ax.set_xlabel("Number of subjects")
    ax.set_ylabel("Speedup factor (cold / warm)")
    ax.set_title("Cache Speedup Scales with Dataset Size")
    ax.legend()
    ax.set_xticks(subjects)

    # Annotate both lines
    for i, n in enumerate(subjects):
        ax.annotate(f"{svm_rbf_speedup[i]:.0f}x", (n, svm_rbf_speedup[i]),
                     textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9,
                     color="#D55E00", fontweight="bold")
        ax.annotate(f"{svm_linear_speedup[i]:.0f}x", (n, svm_linear_speedup[i]),
                     textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9,
                     color="#0072B2", fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_bonus_svm_scaling.pdf", bbox_inches="tight")
    print(f"  [9/10] Saved fig_bonus_svm_scaling.pdf")
    if show:
        plt.show()
    plt.close(fig)


# ===================================================================
# BONUS 2: Cache Size Scaling — does cache grow with features / model type?
# ===================================================================
def fig_bonus_cache_scaling(show: bool):
    """Plot cache size and training time across feature counts for key models.

    Uses the feature_size benchmark (30 subjects, 3 folds, topk = 10/30/50/149).
    Shows that RF/Extra Trees cache is dominated by tree structure (flat),
    while boosting models stay tiny regardless.
    """
    import matplotlib.pyplot as plt

    topk_vals = [10, 30, 50, 149]
    model_groups = {
        "Random Forest":    ("random_forest",    "#D55E00", "o-"),
        "Extra Trees":      ("extra_trees",      "#E69F00", "D-"),
        "XGBoost":          ("xgboost",          "#0072B2", "s-"),
        "Gradient Boosting": ("gradient_boosting", "#009E73", "^-"),
        "LightGBM":         ("lightgbm",         "#CC79A7", "v-"),
    }

    # Collect data across feature counts
    records = []
    for k in topk_vals:
        csv_dir = FEATURE_SIZE_DIR / f"topk_{k}"
        csvs = list(csv_dir.glob("*.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        for _, row in df.iterrows():
            records.append({
                "n_features": k,
                "model": row["model_name"],
                "cache_per_fold_mb": row["cache_size_per_fold_mb"],
                "cold_per_fold_s": row["cold_per_fold_time_s"],
                "warm_per_fold_s": row["warm_per_fold_time_s"],
                "time_saved_s": row["cold_per_fold_time_s"] - row["warm_per_fold_time_s"],
            })
    data = pd.DataFrame(records)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left panel: Cache size per fold ---
    for label, (model_key, color, style) in model_groups.items():
        subset = data[data["model"] == model_key].sort_values("n_features")
        if subset.empty:
            continue
        ax1.plot(subset["n_features"], subset["cache_per_fold_mb"],
                 style, color=color, label=label, linewidth=2, markersize=7)

    ax1.set_xlabel("Number of features", fontsize=11)
    ax1.set_ylabel("Cache size per fold (MB)", fontsize=11)
    ax1.set_title("Cache Size vs. Feature Count", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.2)
    ax1.set_xticks(topk_vals)

    # --- Right panel: Cold training time per fold ---
    for label, (model_key, color, style) in model_groups.items():
        subset = data[data["model"] == model_key].sort_values("n_features")
        if subset.empty:
            continue
        ax2.plot(subset["n_features"], subset["cold_per_fold_s"],
                 style, color=color, label=label, linewidth=2, markersize=7)

    ax2.set_xlabel("Number of features", fontsize=11)
    ax2.set_ylabel("Cold training time per fold (s, log scale)", fontsize=11)
    ax2.set_title("Training Time vs. Feature Count", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v:.2f}" if v < 1 else f"{v:.0f}"))
    ax2.grid(True, alpha=0.2)
    ax2.set_xticks(topk_vals)

    fig.suptitle("Tree Ensembles: Cache Size is Flat, Training Time Grows\n"
                 "(30 subjects, 3 LOSO folds, 200 trees for RF/ET)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG_DIR / "fig_bonus_cache_scaling.pdf", bbox_inches="tight")
    print(f"  [10/12] Saved fig_bonus_cache_scaling.pdf")
    if show:
        plt.show()
    plt.close(fig)


# ===================================================================
# BONUS 3: Cache Efficiency — bigger models save more compute per MB
# ===================================================================
def fig_bonus_cache_efficiency(show: bool):
    """Plot how cache efficiency (seconds saved per MB) changes with features.

    Shows that as models get bigger (more features → more training time),
    the compute saved per MB of cache improves — caching becomes more
    'worth it' even for large models like RF.
    """
    import matplotlib.pyplot as plt

    topk_vals = [10, 30, 50, 149]
    model_groups = {
        "Random Forest":    ("random_forest",    "#D55E00", "o-"),
        "Extra Trees":      ("extra_trees",      "#E69F00", "D-"),
        "XGBoost":          ("xgboost",          "#0072B2", "s-"),
        "Gradient Boosting": ("gradient_boosting", "#009E73", "^-"),
        "LightGBM":         ("lightgbm",         "#CC79A7", "v-"),
    }

    records = []
    for k in topk_vals:
        csv_dir = FEATURE_SIZE_DIR / f"topk_{k}"
        csvs = list(csv_dir.glob("*.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        for _, row in df.iterrows():
            time_saved = row["cold_per_fold_time_s"] - row["warm_per_fold_time_s"]
            cache_mb = row["cache_size_per_fold_mb"]
            # Efficiency: seconds saved per MB of cache (higher = better)
            efficiency = time_saved / cache_mb if cache_mb > 0.001 else 0
            records.append({
                "n_features": k,
                "model": row["model_name"],
                "cache_per_fold_mb": cache_mb,
                "cold_per_fold_s": row["cold_per_fold_time_s"],
                "time_saved_s": time_saved,
                "efficiency_s_per_mb": efficiency,
                "mb_per_s_saved": row["mb_per_second_saved"],
            })
    data = pd.DataFrame(records)

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

    # Seconds saved per MB (higher = more efficient)
    for label, (model_key, color, style) in model_groups.items():
        subset = data[data["model"] == model_key].sort_values("n_features")
        if subset.empty:
            continue
        ax1.plot(subset["n_features"], subset["efficiency_s_per_mb"],
                 style, color=color, label=label, linewidth=2, markersize=7)

    ax1.set_xlabel("Number of features", fontsize=11)
    ax1.set_ylabel("Seconds saved per MB of cache", fontsize=11)
    ax1.set_title("Cache Efficiency Improves with Model Complexity\n"
                  "(higher = more compute saved per MB stored)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.2)
    ax1.set_xticks(topk_vals)

    # Viable threshold: efficiency > 2.0 s/MB ↔ mb_per_s < 0.5
    ax1.axhline(y=2.0, color="#888888", linestyle="--", linewidth=1.5, alpha=0.6)
    ax1.text(topk_vals[-1] + 2, 2.3, "Viable threshold\n(> 2.0 s/MB)",
             fontsize=8, color="#555555", ha="right", va="bottom")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_bonus_cache_efficiency.pdf", bbox_inches="tight")
    print(f"  [11/12] Saved fig_bonus_cache_efficiency.pdf")
    if show:
        plt.show()
    plt.close(fig)


# ===================================================================
# BONUS 4: RF Cache Interaction Heatmap — tree count vs features vs subjects
# ===================================================================
def fig_bonus_rf_interactions(show: bool):
    """Heatmap showing RF cache viability (MB/s-saved) across tree count,
    feature count, and subject count. All combinations are NOT_VIABLE,
    proving tree ensemble caching is fundamentally I/O-bound."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    df = pd.read_csv(RF_INTERACTIONS_CSV)

    # Label top_k nicely
    df["features_label"] = df["top_k"].fillna(149).astype(int).astype(str) + "f"

    subject_counts = sorted(df["subject_count"].unique())

    fig, axes = plt.subplots(1, len(subject_counts), figsize=(18, 5.5),
                              sharey=True)
    if len(subject_counts) == 1:
        axes = [axes]

    for ax, n_subj in zip(axes, subject_counts):
        subset = df[df["subject_count"] == n_subj]

        # Pivot: rows=tree_count, cols=features
        pivot = subset.groupby(["tree_count", "features_label"])["mb_per_second_saved"].mean().reset_index()
        # Order features
        feat_order = ["30f", "50f", "149f"]
        pivot_table = pivot.pivot(index="tree_count", columns="features_label", values="mb_per_second_saved")
        pivot_table = pivot_table.reindex(columns=[f for f in feat_order if f in pivot_table.columns])

        im = ax.imshow(pivot_table.values, cmap="RdYlGn_r", aspect="auto",
                       norm=LogNorm(vmin=0.5, vmax=50))

        # Labels
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns, fontsize=10)
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index, fontsize=10)

        # Annotate cells
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                val = pivot_table.values[i, j]
                color = "white" if val > 10 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

        ax.set_xlabel("Features", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("n_estimators (trees)", fontsize=11)
        ax.set_title(f"{n_subj} subjects", fontsize=12)

    # Threshold annotation
    fig.suptitle("RF Cache Viability: MB/s Saved Across All Configurations\n"
                 "All values > 0.5 = NOT VIABLE (green = closer to viable, red = far from viable)",
                 fontsize=12, fontweight="bold")

    fig.subplots_adjust(top=0.82, wspace=0.25, left=0.06, right=0.88)

    # Manually place colorbar to avoid overlap with rightmost panel
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.55])
    cbar = fig.colorbar(im, cax=cbar_ax,
                         label="MB per second saved (< 0.5 = viable)")
    cbar.ax.axhline(y=0.5, color="black", linewidth=2, linestyle="--")
    cbar.ax.axhline(y=2.0, color="black", linewidth=1, linestyle=":")
    fig.savefig(FIG_DIR / "fig_bonus_rf_interactions.pdf", bbox_inches="tight")
    print(f"  [12/14] Saved fig_bonus_rf_interactions.pdf")
    if show:
        plt.show()
    plt.close(fig)


# ===================================================================
# BONUS 5: RF Cache Size vs Training Time (interaction plot)
# ===================================================================
def fig_bonus_rf_size_vs_time(show: bool):
    """Scatter plot showing cache size vs cold training time for all RF
    configurations. Lines connect tree counts. Shows that cache size
    grows linearly with trees but training time also grows — the ratio
    never crosses the viability threshold."""
    import matplotlib.pyplot as plt

    df = pd.read_csv(RF_INTERACTIONS_CSV)
    df["features_label"] = df["top_k"].fillna(149).astype(int).astype(str) + "f"

    # Only 128 subjects (most relevant for thesis)
    df128 = df[df["subject_count"] == 128]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Cache size vs tree count, colored by feature count
    colors = {"30f": "#D55E00", "50f": "#0072B2", "149f": "#009E73"}
    for feat_label, color in colors.items():
        subset = df128[df128["features_label"] == feat_label].groupby("tree_count").mean(numeric_only=True).reset_index()
        ax1.plot(subset["tree_count"], subset["cache_size_mb"], "o-",
                 color=color, label=feat_label, linewidth=2, markersize=7)
    ax1.set_xlabel("n_estimators (trees)", fontsize=11)
    ax1.set_ylabel("Cache size per fold (MB)", fontsize=11)
    ax1.set_title("Cache Size Grows Linearly with Trees", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Right: Speedup vs tree count
    for feat_label, color in colors.items():
        subset = df128[df128["features_label"] == feat_label].groupby("tree_count").mean(numeric_only=True).reset_index()
        ax2.plot(subset["tree_count"], subset["speedup_cold_vs_warm"], "o-",
                 color=color, label=feat_label, linewidth=2, markersize=7)

    ax2.set_xlabel("n_estimators (trees)", fontsize=11)
    ax2.set_ylabel("Speedup (cold / warm)", fontsize=11)
    ax2.set_title("Speedup Stays Low (~5-11x) Regardless of Config", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)

    fig.suptitle("Random Forest: Why No Configuration Makes Caching Viable\n"
                 "(128 subjects, 3 LOSO folds per setting)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG_DIR / "fig_bonus_rf_size_vs_time.pdf", bbox_inches="tight")
    print(f"  [13/14] Saved fig_bonus_rf_size_vs_time.pdf")
    if show:
        plt.show()
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures and tables")
    parser.add_argument("--show", action="store_true", help="Display figures after saving")
    args = parser.parse_args()

    ensure_dirs()
    print(f"Generating thesis figures and tables...")
    print(f"  Figures → {FIG_DIR}")
    print(f"  Tables  → {TAB_DIR}")
    print()

    fig1_speedup_bar(args.show)
    tab2_performance(args.show)
    fig3_efficiency(args.show)
    fig3b_crossover(args.show)
    tab4_viability(args.show)
    fig4b_viability_scatter(args.show)
    fig4c_dual_metrics(args.show)
    tab5_fingerprint(args.show)
    fig6_per_class_f1(args.show)
    fig7_feature_importance(args.show)
    tab8_global_vs_fold(args.show)
    fig_bonus_svm_scaling(args.show)
    fig_bonus_cache_scaling(args.show)
    fig_bonus_cache_efficiency(args.show)
    fig_bonus_rf_interactions(args.show)
    fig_bonus_rf_size_vs_time(args.show)

    print()
    print(f"Done! 12 figures + 4 tables generated.")
    print(f"  PDF figures: {FIG_DIR}")
    print(f"  LaTeX tables: {TAB_DIR}")


if __name__ == "__main__":
    main()
