"""
Generate two pipeline architecture figures for the thesis.

  fig_pipeline_overview.pdf   — full end-to-end pipeline (horizontal)
  fig_fingerprint_zoom.pdf    — fingerprint generation & cache lookup (vertical)

Output: thesis/figures/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[2] / "thesis" / "figures"

# Okabe-Ito palette roles
C_DATA    = "#0072B2"   # blue   — data / processing
C_CACHE   = "#E69F00"   # orange — cache layers
C_COLD    = "#D55E00"   # vermillion — cold / train path
C_WARM    = "#009E73"   # green  — warm / hit path
C_FINGER  = "#CC79A7"   # purple — fingerprint
C_EVAL    = "#56B4E9"   # sky    — evaluation
C_GRAY    = "#BBBBBB"   # light gray — decision diamond fill
C_TEXT    = "#222222"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def box(ax, cx, cy, w, h, color, text, fontsize=8.2, text_color="white", style="round,pad=0.08"):
    patch = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle=style, linewidth=0.8,
                           edgecolor="#444444", facecolor=color, zorder=3)
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            zorder=4, linespacing=1.4,
            multialignment="center")


def diamond(ax, cx, cy, w, h, color, text, fontsize=7.5):
    xs = [cx, cx+w/2, cx, cx-w/2, cx]
    ys = [cy+h/2, cy, cy-h/2, cy, cy+h/2]
    ax.fill(xs, ys, color=color, zorder=3, linewidth=0.8, edgecolor="#444444")
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=C_TEXT, zorder=4,
            multialignment="center")


def arrow(ax, x0, y0, x1, y1, color="#555555", label="", lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12), zorder=2)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+0.03, my, label, fontsize=7, color=color,
                ha="left", va="center", zorder=5)


# ─────────────────────────────────────────────────────────────
# Figure 1: Full pipeline overview
# ─────────────────────────────────────────────────────────────

def fig_pipeline_overview():
    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(-0.15, 3.8)
    ax.axis("off")

    Y_MAIN = 1.8
    BH = 1.15
    BW = 1.55

    nodes = [
        (1.0,  Y_MAIN, C_DATA,    "BOAS Dataset\n128 subjects\n~120K epochs\n6 EEG channels"),
        (3.0,  Y_MAIN, C_DATA,    "Preprocessing\n0.5–40 Hz bandpass\nNotch 50 Hz\n128 Hz · 30 s epochs"),
        (5.0,  Y_MAIN, C_DATA,    "Feature Extraction\n149 features/epoch\nTime · Freq · Complexity\n~53 min cold"),
        (7.15, Y_MAIN, C_CACHE,   "Layer 1 Cache\nNPZ per subject\nConfig-independent\n~146 MB total"),
        (9.3,  Y_MAIN, C_FINGER,  "LOSO Split\n128 folds\n+ Fingerprint\nSHA-256 key"),
        (11.4, Y_MAIN, C_CACHE,   "Layer 2 Cache\nJoblib per fold\nFingerprint-keyed\nXGB ~185 MB / run"),
        (13.3, Y_MAIN, C_EVAL,    "Evaluation\nAccuracy · \u03ba\nMacro-F1\nPer-class F1"),
    ]

    for cx, cy, color, text in nodes:
        box(ax, cx, cy, BW, BH, color, text, fontsize=7.8)

    # Main horizontal arrows
    xs = [n[0] for n in nodes]
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + BW/2, Y_MAIN, xs[i+1] - BW/2, Y_MAIN)

    # ── Warm bypass 1: Layer 1 hit → skip extraction (arc BELOW)
    Y_BELOW = Y_MAIN - BH/2 - 0.38
    # horizontal line below Feature Extraction → Layer 2 Cache
    ax.annotate("",
                xy=(nodes[5][0] - BW/2, Y_MAIN - BH/2),
                xytext=(nodes[2][0] + BW/2, Y_MAIN - BH/2),
                arrowprops=dict(arrowstyle="-|>", color=C_WARM, lw=1.5,
                                connectionstyle=f"arc,angleA=-90,angleB=-90,armA=30,armB=30,rad=0",
                                mutation_scale=11), zorder=2)
    ax.text(8.2, Y_BELOW, "Layer 1 hit \u2192 skip re-extraction",
            fontsize=7.2, color=C_WARM, ha="center", va="top")

    # ── Warm bypass 2: Layer 2 hit → skip training (arc ABOVE)
    Y_ABOVE = Y_MAIN + BH/2 + 0.38
    ax.annotate("",
                xy=(nodes[6][0] - BW/2, Y_MAIN + BH/2),
                xytext=(nodes[4][0] + BW/2, Y_MAIN + BH/2),
                arrowprops=dict(arrowstyle="-|>", color=C_WARM, lw=1.5,
                                connectionstyle="arc,angleA=90,angleB=90,armA=30,armB=30,rad=0",
                                mutation_scale=11), zorder=2)
    ax.text(11.35, Y_ABOVE, "Layer 2 hit \u2192 skip training (\u223c7 s warm run)",
            fontsize=7.2, color=C_WARM, ha="center", va="bottom")

    # Cold label
    ax.text(nodes[5][0], Y_MAIN - BH/2 - 0.08,
            "cold: train + store", fontsize=7, color=C_COLD,
            ha="center", va="top", style="italic")

    # Legend
    legend_items = [
        mpatches.Patch(facecolor=C_DATA,   label="Data / Processing"),
        mpatches.Patch(facecolor=C_CACHE,  label="Cache layer"),
        mpatches.Patch(facecolor=C_FINGER, label="Fingerprinting / LOSO"),
        mpatches.Patch(facecolor=C_EVAL,   label="Evaluation"),
        mpatches.Patch(facecolor=C_WARM,   label="Warm path (cache hit)"),
        mpatches.Patch(facecolor=C_COLD,   label="Cold path (retrain)"),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=3,
              fontsize=7.5, framealpha=0.92,
              bbox_to_anchor=(0.5, -0.02))

    ax.set_title("Fingerprint-Based Caching Pipeline \u2014 End-to-End Overview",
                 fontsize=11, fontweight="bold", pad=8)

    fig.tight_layout()
    out = FIG_DIR / "fig_pipeline_overview.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


# ─────────────────────────────────────────────────────────────
# Figure 2: Fingerprint zoom
# ─────────────────────────────────────────────────────────────

def fig_fingerprint_zoom():
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 10)
    ax.axis("off")

    BW, BH = 4.8, 0.78
    CX = 3.5
    ARROW_X = CX

    steps = [
        # (y_center, color, text)
        (9.2, C_DATA,
         "Experiment Configuration\n"
         "model_type · hyperparameters · random_seed\n"
         "corr_threshold · top_k · selected_feature_names\n"
         "code_version · held_out_subject"),
        (7.6, C_FINGER,
         "Canonical Serialisation\n"
         "json.dumps(config, sort_keys=True,\n"
         "separators=(',', ':'))  →  deterministic string"),
        (6.1, C_FINGER,
         "SHA-256 Hashing\n"
         "hashlib.sha256(json_str.encode('utf-8'))\n"
         "→ truncate to 32 hex chars (128-bit key)"),
        (4.6, C_FINGER,
         "Cache Key Formation\n"
         "key = fingerprint[:32] + '_sub-' + subject_id\n"
         "e.g.  a3f2...d1_sub-42"),
    ]

    for cy, color, text in steps:
        box(ax, CX, cy, BW, BH*1.1, color, text, fontsize=8)

    # arrows between config steps
    ys = [s[0] for s in steps]
    for i in range(len(ys)-1):
        arrow(ax, ARROW_X, ys[i] - BH*1.1/2,
                  ARROW_X, ys[i+1] + BH*1.1/2)

    # Decision diamond
    DY = 3.0
    diamond(ax, CX, DY, 3.2, 0.9, C_GRAY, "Cache hit?\n(key found in cache dir)")
    arrow(ax, ARROW_X, ys[-1] - BH*1.1/2, ARROW_X, DY + 0.45)

    # HIT branch (left)
    HIT_X = 1.6
    box(ax, HIT_X, 1.7, 2.4, 0.75, C_WARM,
        "Load model\n.joblib (~58 ms XGB)", fontsize=8)
    ax.annotate("", xy=(HIT_X, 1.7+0.375),
                xytext=(CX-1.6, DY),
                arrowprops=dict(arrowstyle="-|>", color=C_WARM,
                                lw=1.5, mutation_scale=12), zorder=2)
    ax.text(HIT_X, DY+0.1, "HIT", fontsize=8, color=C_WARM,
            ha="center", fontweight="bold")

    # MISS branch (right)
    MISS_X = 5.4
    box(ax, MISS_X, 2.2, 2.4, 0.75, C_COLD,
        "Train model\n(cold: seconds–minutes)", fontsize=8)
    box(ax, MISS_X, 1.2, 2.4, 0.75, C_COLD,
        "Serialise + store\n.joblib in cache dir", fontsize=8)
    ax.annotate("", xy=(MISS_X, 2.2+0.375),
                xytext=(CX+1.6, DY),
                arrowprops=dict(arrowstyle="-|>", color=C_COLD,
                                lw=1.5, mutation_scale=12), zorder=2)
    ax.text(MISS_X, DY+0.1, "MISS", fontsize=8, color=C_COLD,
            ha="center", fontweight="bold")
    arrow(ax, MISS_X, 2.2-0.375, MISS_X, 1.2+0.375, color=C_COLD)

    # Convergence to Predict
    box(ax, CX, 0.45, BW*0.72, 0.62, C_EVAL,
        "model.predict(X_test)  →  fold predictions", fontsize=8)
    arrow(ax, HIT_X,  1.7-0.375, HIT_X,  0.45+0.31, color=C_WARM)
    arrow(ax, MISS_X, 1.2-0.375, MISS_X, 0.45+0.31, color=C_COLD)

    ax.set_title("Fingerprint Generation and Cache Lookup\n(per LOSO fold)",
                 fontsize=11, fontweight="bold", y=0.99)

    fig.tight_layout()
    out = FIG_DIR / "fig_fingerprint_zoom.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_pipeline_overview()
    fig_fingerprint_zoom()
    print("Done.")
