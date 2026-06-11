"""Generate clearer speedup figures (v2 and v3) for the cache benchmark.

Motivation
----------
The original ``fig1_speedup_bar.pdf`` plots *summed total* cold/warm time on a
log axis but annotates the *median per-configuration* speedup. Those are two
different numbers (e.g. XGBoost: total ratio 66.7x vs. median ratio 54.1x), so
the visible bar gap does not match the printed factor. Both versions below fix
this by deriving everything from the per-configuration speedups and reporting a
single, consistent definition: the median of (cold_i / warm_i).

Data source: tab:xgb_cache_results and tab:rf_cache_results
(HP ProBook benchmark, 128 LOSO folds, 9 configurations each) -- identical to
the arrays used in archive/scripts/generate_thesis_figures.py::fig1_speedup_bar.

Outputs:
    thesis/figures/fig1_speedup_bar_v2.pdf   (+ /tmp PNG preview)
    thesis/figures/fig1_speedup_bar_v3.pdf   (+ /tmp PNG preview)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
    """Per-config speedups plus the index of the median (representative) config."""
    sp = cold / warm
    med_idx = int(np.argsort(sp)[len(sp) // 2])  # odd n -> exact median element
    return {
        "sp": sp,
        "median": float(np.median(sp)),
        "lo": float(sp.min()),
        "hi": float(sp.max()),
        "med_idx": med_idx,
        "cold_rep": float(cold[med_idx]),
        "warm_rep": float(warm[med_idx]),
        "cold_total_min": float(cold.sum() / 60),
        "warm_total_min": float(warm.sum() / 60),
    }


XGB = summarise(xgb_cold, xgb_warm)
RF = summarise(rf_cold, rf_warm)


def _save(fig, name):
    pdf = FIG_DIR / f"{name}.pdf"
    png = PREVIEW / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    plt.close(fig)


# ===========================================================================
# v2 -- Direct speedup-factor bars. Bar height == the cited number.
# ===========================================================================
def make_v2():
    models = ["XGBoost", "Random Forest"]
    medians = [XGB["median"], RF["median"]]
    colors = [C_XGB, C_RF]
    reps = [(XGB["cold_rep"], XGB["warm_rep"]), (RF["cold_rep"], RF["warm_rep"])]

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.bar(x, medians, 0.55, color=colors, zorder=3)

    # 1x reference: caching gives no benefit below this line.
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1, zorder=2)
    ax.annotate("1× = no speedup", xy=(1.45, 1.0), xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=8.5, color="#555555")

    # Headline factor on top of each bar; representative time transform below it.
    for i, b in enumerate(bars):
        h = b.get_height()
        ax.annotate(f"{h:.1f}×", xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    va="bottom", fontweight="bold", fontsize=15, color=colors[i])
        c, w = reps[i]
        ax.annotate(f"{c:.0f} s → {w:.0f} s\nper configuration",
                    xy=(b.get_x() + b.get_width() / 2, h / 2),
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Warm-run speedup (×)")
    ax.set_ylim(0, max(medians) * 1.25)
    ax.set_title("Cache Speedup per Model (median across 9 configs, 128 LOSO folds)")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    _save(fig, "fig1_speedup_bar_v2")


# ===========================================================================
# v3 -- Before/after slope chart. Drop on the log axis == speedup; the median
#       config is highlighted and its endpoints give exactly the cited factor.
# ===========================================================================
def make_v3():
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xpos = {"cold": 0, "warm": 1}

    def plot_model(s, cold, warm, color, label):
        # Faint lines: every configuration (shows spread, all consistent).
        for ci, wi in zip(cold, warm):
            ax.plot([0, 1], [ci, wi], color=color, alpha=0.18, linewidth=1.2, zorder=2)
        # Bold line: the representative (median-speedup) configuration.
        c, w = s["cold_rep"], s["warm_rep"]
        ax.plot([0, 1], [c, w], color=color, alpha=1.0, linewidth=2.6, zorder=4,
                marker="o", markersize=7, label=label)
        # Speedup label on the midpoint of the bold connector (geometric mean y).
        ymid = np.sqrt(c * w)
        ax.annotate(f"{s['median']:.1f}× faster", xy=(0.5, ymid),
                    xytext=(0, 0), textcoords="offset points", ha="center",
                    va="center", fontsize=11, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1.2),
                    zorder=5)

    plot_model(XGB, xgb_cold, xgb_warm, C_XGB, "XGBoost")
    plot_model(RF, rf_cold, rf_warm, C_RF, "Random Forest")

    ax.set_yscale("log")
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Cold\n(no cache)", "Warm\n(cached)"], fontsize=11)
    ax.set_ylabel("Time per configuration (s, log scale)")
    ax.set_title("Cold vs. Cached Training Time per Configuration (128 LOSO folds)")
    ax.grid(axis="y", alpha=0.3, which="both")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Legend: model colours + meaning of bold vs. faint lines.
    handles = [
        Line2D([0], [0], color=C_XGB, lw=2.6, marker="o", label="XGBoost (median config)"),
        Line2D([0], [0], color=C_RF, lw=2.6, marker="o", label="Random Forest (median config)"),
        Line2D([0], [0], color="#888888", lw=1.2, alpha=0.5, label="Individual configs (9 each)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    _save(fig, "fig1_speedup_bar_v3")


if __name__ == "__main__":
    print("XGBoost: median %.1fx (range %.1f-%.1f), rep %.0f->%.0f s"
          % (XGB["median"], XGB["lo"], XGB["hi"], XGB["cold_rep"], XGB["warm_rep"]))
    print("RandomForest: median %.1fx (range %.1f-%.1f), rep %.0f->%.0f s"
          % (RF["median"], RF["lo"], RF["hi"], RF["cold_rep"], RF["warm_rep"]))
    make_v2()
    make_v3()
