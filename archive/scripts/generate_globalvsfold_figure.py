"""
Generate fig_global_vs_perfold.pdf from benchmark results.
Run after benchmark_global_vs_perfold.py has produced the CSV, OR
use the hardcoded data below (collected 2026-04-20).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[2] / "thesis" / "figures"

# ── Hardcoded benchmark results ──────────────────────────────────────────────
# Source: benchmark_global_vs_perfold.py, top_k=50, corr=None/0.75/0.90
# Accuracy values are averages across the 3 corr thresholds (cold runs)
DATA = {
    "n": [3, 5, 10, 20],
    "xgb": {
        "global_acc":   [0.7441, 0.7360, 0.7360, 0.7940],
        "perfold_acc":  [0.7382, 0.7359, 0.7398, 0.7931],
        "global_warm":  [14.3,   18.8,   22.2,   22.9],
        "perfold_warm": [9.0,    8.7,    6.6,    4.4],
    },
    "rf": {
        "global_acc":   [0.7184, 0.7237, 0.7189, 0.7814],
        "perfold_acc":  [0.7287, 0.7202, 0.7171, 0.7780],
        "global_warm":  [2.9,    3.1,    3.7,    4.7],
        "perfold_warm": [2.6,    2.5,    2.6,    2.8],
    },
}

C_GLOBAL  = "#0072B2"   # blue
C_PERFOLD = "#D55E00"   # vermillion
C_XGB     = "#0072B2"
C_RF      = "#E69F00"

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n = DATA["n"]
    x = np.arange(len(n))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Left: accuracy difference (global - per_fold) ─────────────────────
    ax = axes[0]
    xgb_diff = [g - p for g, p in zip(DATA["xgb"]["global_acc"], DATA["xgb"]["perfold_acc"])]
    rf_diff  = [g - p for g, p in zip(DATA["rf"]["global_acc"],  DATA["rf"]["perfold_acc"])]

    bars_xgb = ax.bar(x - w/2, xgb_diff, w, color=C_XGB,  label="XGBoost", alpha=0.9)
    bars_rf  = ax.bar(x + w/2, rf_diff,  w, color=C_RF,   label="Random Forest", alpha=0.9)

    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={v}" for v in n])
    ax.set_ylabel("Accuracy: global − per-fold")
    ax.set_title("Accuracy Difference\n(positive = global wins, negative = per-fold wins)",
                 fontsize=10)
    ax.legend(fontsize=9)

    # Annotate each bar with its value
    for bar in list(bars_xgb) + list(bars_rf):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + (0.0005 if h >= 0 else -0.0015),
                f"{h:+.3f}", ha="center", va="bottom" if h >= 0 else "top",
                fontsize=7.5, color="#333333")

    # ── Right: warm-run speedup comparison ───────────────────────────────
    ax2 = axes[1]
    ax2.plot(n, DATA["xgb"]["global_warm"],  "o-", color=C_XGB,  linewidth=2, markersize=7,
             label="XGBoost — global")
    ax2.plot(n, DATA["xgb"]["perfold_warm"], "o--", color=C_XGB, linewidth=2, markersize=7,
             alpha=0.55, label="XGBoost — per-fold")
    ax2.plot(n, DATA["rf"]["global_warm"],   "s-", color=C_RF,   linewidth=2, markersize=7,
             label="RF — global")
    ax2.plot(n, DATA["rf"]["perfold_warm"],  "s--", color=C_RF,  linewidth=2, markersize=7,
             alpha=0.55, label="RF — per-fold")

    # Annotate end points
    for vals, color, dy in [
        (DATA["xgb"]["global_warm"],  C_XGB, +1.0),
        (DATA["xgb"]["perfold_warm"], C_XGB, -1.5),
        (DATA["rf"]["global_warm"],   C_RF,  +0.3),
        (DATA["rf"]["perfold_warm"],  C_RF,  -0.5),
    ]:
        ax2.annotate(f"{vals[-1]:.1f}×",
                     xy=(n[-1], vals[-1]), xytext=(2, dy),
                     textcoords="offset points",
                     fontsize=8, color=color, fontweight="bold")

    ax2.set_xlabel("Number of subjects")
    ax2.set_ylabel("Warm-run speedup (cold / warm)")
    ax2.set_title("Warm-Run Speedup: Global vs. Per-Fold Scope\n"
                  "(per-fold XGBoost speedup degrades as N grows)", fontsize=10)
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.set_xticks(n)
    ax2.grid(True, alpha=0.2)

    fig.suptitle("Global vs. Per-Fold ANOVA: Accuracy Impact and Cache Speedup",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = FIG_DIR / "fig_global_vs_perfold.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


if __name__ == "__main__":
    main()
