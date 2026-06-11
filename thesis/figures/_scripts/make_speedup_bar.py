"""Generate the cache speedup figure (fig1_speedup_bar).

Each bar's height is the median per-configuration warm-run speedup
(median of cold_i / warm_i across the 9 configurations), so the bar height
equals the factor printed above it -- no totals-vs-median mismatch. The inside
-bar label names the cold and warm execution times for the representative
(median) configuration.

Data source: tab:xgb_cache_results and tab:rf_cache_results
(HP ProBook benchmark, 128 LOSO folds, 9 configurations each).

Output:
    thesis/figures/fig1_speedup_bar.pdf   (+ /tmp PNG preview)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[1]
PREVIEW = Path("/tmp")

# --- Raw per-configuration timings (seconds) -------------------------------
xgb_cold = np.array([1053.4, 1447.4, 1440.3, 964.1, 1453.8, 2223.6, 1003.6, 1385.9, 3816.1])
xgb_warm = np.array([26.4, 27.2, 26.6, 25.7, 26.4, 28.2, 21.1, 21.2, 18.9])
rf_cold = np.array([4341.3, 4638.0, 4648.3, 4342.7, 5794.2, 6315.9, 4567.2, 5602.4, 7962.4])
rf_warm = np.array([391.8, 402.8, 401.3, 392.2, 375.3, 359.0, 415.6, 356.7, 339.4])

# Colour-blind safe (Wong) palette, coloured by model.
C_XGB = "#0072B2"  # blue
C_RF = "#D55E00"   # vermillion


def summarise(cold, warm):
    """Median per-config speedup and the cold/warm times of that median config."""
    sp = cold / warm
    med_idx = int(np.argsort(sp)[len(sp) // 2])  # odd n -> exact median element
    return {
        "median": float(np.median(sp)),
        "cold_rep": float(cold[med_idx]),
        "warm_rep": float(warm[med_idx]),
    }


XGB = summarise(xgb_cold, xgb_warm)
RF = summarise(rf_cold, rf_warm)


def make_figure():
    models = ["XGBoost", "Random Forest"]
    medians = [XGB["median"], RF["median"]]
    colors = [C_XGB, C_RF]
    reps = [(XGB["cold_rep"], XGB["warm_rep"]), (RF["cold_rep"], RF["warm_rep"])]

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.bar(x, medians, 0.55, color=colors, zorder=3)

    for i, b in enumerate(bars):
        h = b.get_height()
        # Headline factor on top of each bar.
        ax.annotate(f"{h:.1f}×", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    va="bottom", fontweight="bold", fontsize=16, color=colors[i])
        # Cold and warm execution times of the median config (inside the bar).
        c, w = reps[i]
        ax.annotate(f"Median config:\nCold {c:.0f} s\nWarm {w:.0f} s",
                    xy=(b.get_x() + b.get_width() / 2, h / 2),
                    ha="center", va="center", fontsize=9.5, color="white",
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Speedup vs. cold start (×)")
    ax.set_ylim(0, max(medians) * 1.22)
    ax.set_title("Cache Speedup per Model (median across 9 configs, 128 LOSO folds)")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    pdf = FIG_DIR / "fig1_speedup_bar.pdf"
    png = PREVIEW / "fig1_speedup_bar.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    plt.close(fig)


if __name__ == "__main__":
    print("XGBoost: median %.1fx, rep cold %.0f -> warm %.0f s"
          % (XGB["median"], XGB["cold_rep"], XGB["warm_rep"]))
    print("RandomForest: median %.1fx, rep cold %.0f -> warm %.0f s"
          % (RF["median"], RF["cold_rep"], RF["warm_rep"]))
    make_figure()
