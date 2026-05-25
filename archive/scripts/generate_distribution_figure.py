"""
Generate sleep stage distribution figure from hardcoded BOAS dataset counts.
No feature cache required — runs on any machine with matplotlib.

Output: thesis/figures/fig_sleep_stage_distribution.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIG_DIR = REPO / "thesis" / "figures"

# BOAS dataset — 128 subjects, ~120K epochs
STAGES  = ["Wake", "N1",   "N2",    "N3",   "REM"]
COUNTS  = [19137,  4462,   72181,   5225,   18754]
# Okabe-Ito colorblind-safe palette (Wake, N1, N2, N3, REM)
COLORS  = ["#E69F00", "#56B4E9", "#0072B2", "#009E73", "#CC79A7"]

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(COUNTS)
    pcts  = [100.0 * c / total for c in COUNTS]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(STAGES, COUNTS, color=COLORS, alpha=0.85,
                  edgecolor="black", linewidth=0.6)

    for bar, pct, cnt in zip(bars, pcts, COUNTS):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.004,
                f"{pct:.1f}%\n({cnt:,})",
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Sleep Stage")
    ax.set_ylabel("Number of 30-second Epochs")
    ax.set_title(
        "Sleep Stage Distribution — BOAS Dataset\n"
        "128 subjects · 119,759 epochs · 30-second windows"
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.set_ylim(0, max(COUNTS) * 1.2)
    fig.tight_layout()

    out = FIG_DIR / "fig_sleep_stage_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
