"""Generate XGB+RF scaling figure: speedup and cache size vs subject count.

Data source: scaling experiment summary (XGB+RF, N in {10,30,64,128}).
Output: thesis/figures/fig_xgbrf_scaling.pdf
"""
from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "fig_xgbrf_scaling.pdf"

subjects = [10, 30, 64, 128]
xgb_speedup = [35.6, 43.1, 52.0, 56.3]
rf_speedup = [4.9, 8.7, 10.9, 13.0]
xgb_cache_mb = [1.03, 1.30, 1.42, 1.50]
rf_cache_mb = [11.64, 36.04, 71.88, 152.42]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: speedup
ax1.plot(subjects, xgb_speedup, "o-", color="#1f77b4", linewidth=2, markersize=7, label="XGBoost")
ax1.plot(subjects, rf_speedup, "s-", color="#d62728", linewidth=2, markersize=7, label="Random Forest")
for x, y in zip(subjects, xgb_speedup):
    ax1.annotate(f"{y:.1f}×", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#1f77b4")
for x, y in zip(subjects, rf_speedup):
    ax1.annotate(f"{y:.1f}×", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#d62728")
ax1.set_xlabel("Number of subjects")
ax1.set_ylabel("Warm-run speedup (×)")
ax1.set_title("(a) Speedup grows with dataset size")
ax1.set_xticks(subjects)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper left")

# Right: cache size (log scale)
ax2.plot(subjects, xgb_cache_mb, "o-", color="#1f77b4", linewidth=2, markersize=7, label="XGBoost")
ax2.plot(subjects, rf_cache_mb, "s-", color="#d62728", linewidth=2, markersize=7, label="Random Forest")
for x, y in zip(subjects, xgb_cache_mb):
    ax2.annotate(f"{y:.2f} MB", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#1f77b4")
for x, y in zip(subjects, rf_cache_mb):
    ax2.annotate(f"{y:.1f} MB", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#d62728")
ax2.set_xlabel("Number of subjects")
ax2.set_ylabel("Cache size per fold (MB, log)")
ax2.set_title("(b) Cache size: XGB sublinear, RF linear")
ax2.set_xticks(subjects)
ax2.set_yscale("log")
ax2.grid(True, alpha=0.3, which="both")
ax2.legend(loc="upper left")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"Wrote {OUT}")
