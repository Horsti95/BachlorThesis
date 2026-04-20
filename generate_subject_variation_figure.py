"""
Generate per-subject sleep stage variation figure for LOSO justification.

Loads label distributions from the feature cache, verifies the 4 extreme
subjects, and generates thesis/figures/fig_subject_variation.pdf.

Run on the benchmark desktop (needs results/features_cache_global/).
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO      = Path(__file__).resolve().parent
FIG_DIR   = REPO / "thesis" / "figures"
CACHE_DIR = REPO / "results" / "features_cache_global"

STAGES  = ["Wake", "N1", "N2", "N3", "REM"]
COLORS  = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
LABEL_MAP = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

# Claimed extreme subjects — verified by script at runtime
CLAIMED = {
    "SC4114": ("N2-heavy",   {"N2": 79.96}),
    "SC4034": ("Wake-heavy", {"Wake": 51.53}),
    "SC4002": ("N3-heavy",   {"N3": 22.56}),
    "SC4052": ("REM-heavy",  {"REM": 24.49}),
}


def load_subject_labels(subject_dir: Path) -> np.ndarray:
    npz_files = list(subject_dir.glob("*.npz"))
    if not npz_files:
        return np.array([])
    data = np.load(npz_files[0], allow_pickle=True)
    return data["labels"] if "labels" in data else np.array([])


def subject_distribution(labels: np.ndarray) -> dict:
    total = len(labels)
    if total == 0:
        return {}
    dist = {}
    for code, name in LABEL_MAP.items():
        dist[name] = 100.0 * np.sum(labels == code) / total
    return dist


def find_extreme_subjects(cache_dir: Path) -> dict:
    subject_dirs = sorted(cache_dir.iterdir())
    distributions = {}
    for d in subject_dirs:
        if not d.is_dir():
            continue
        labels = load_subject_labels(d)
        if len(labels) > 0:
            distributions[d.name] = subject_distribution(labels)

    extremes = {
        "N2-heavy":   max(distributions, key=lambda s: distributions[s].get("N2", 0)),
        "Wake-heavy": max(distributions, key=lambda s: distributions[s].get("Wake", 0)),
        "N3-heavy":   max(distributions, key=lambda s: distributions[s].get("N3", 0)),
        "REM-heavy":  max(distributions, key=lambda s: distributions[s].get("REM", 0)),
    }
    return extremes, distributions


def verify_claims(extremes: dict, distributions: dict) -> list:
    """Compare found extremes against claimed subjects. Returns rows for the figure."""
    print("\n=== Verification ===")
    rows = []
    claim_map = {v[0]: (k, v[1]) for k, v in CLAIMED.items()}

    for label, found_sid in extremes.items():
        claimed_sid, claimed_vals = claim_map[label]
        dist = distributions[found_sid]
        stat_key = list(claimed_vals.keys())[0]
        claimed_pct = claimed_vals[stat_key]
        actual_pct  = dist.get(stat_key, 0.0)
        match = "OK" if abs(actual_pct - claimed_pct) < 0.5 else "MISMATCH"
        print(f"  {label}: found={found_sid} claimed={claimed_sid} | "
              f"{stat_key}: claimed={claimed_pct:.2f}% actual={actual_pct:.2f}% [{match}]")
        rows.append((label, found_sid, dist))
    return rows


def generate_figure(rows: list) -> None:
    n_subjects = len(rows)
    x = np.arange(len(STAGES))
    width = 0.18
    offsets = np.linspace(-(n_subjects - 1) / 2, (n_subjects - 1) / 2, n_subjects) * width

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (label, sid, dist) in enumerate(rows):
        vals = [dist.get(s, 0.0) for s in STAGES]
        bars = ax.bar(x + offsets[i], vals, width,
                      label=f"{sid} ({label})",
                      color=[c + "cc" for c in COLORS],
                      edgecolor="black", linewidth=0.5)

    # Global average line
    global_pcts = [16.0, 3.7, 60.3, 4.4, 15.7]
    for xi, gp in zip(x, global_pcts):
        ax.hlines(gp, xi - 0.4, xi + 0.4,
                  colors="black", linestyles="--", linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_xlabel("Sleep Stage")
    ax.set_ylabel("Percentage of Epochs (%)")
    ax.set_title(
        "Inter-Subject Variability in Sleep Stage Distribution\n"
        "Four extreme subjects from BOAS dataset (dashed line = dataset average)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 90)
    fig.tight_layout()

    out = FIG_DIR / "fig_subject_variation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


def main():
    if not CACHE_DIR.exists() or len(list(CACHE_DIR.glob("SC*"))) < 10:
        print(f"ERROR: Feature cache not found at {CACHE_DIR}")
        print("Run on the benchmark desktop with the full feature cache.")
        sys.exit(1)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading labels from {CACHE_DIR} ...")
    extremes, distributions = find_extreme_subjects(CACHE_DIR)

    rows = verify_claims(extremes, distributions)

    print("\n=== Full distributions for figure subjects ===")
    for label, sid, dist in rows:
        pcts = "  ".join(f"{s}={dist.get(s,0):.1f}%" for s in STAGES)
        print(f"  {sid} ({label}): {pcts}")

    generate_figure(rows)


if __name__ == "__main__":
    main()
