"""
Generate per-subject sleep stage variation figure for LOSO justification.

Reads raw BOAS annotation files (stage_hum column), finds the 4 extreme
subjects, verifies claimed stats, and generates fig_subject_variation.pdf.

Usage:
    python generate_subject_variation_figure.py
    python generate_subject_variation_figure.py --data "D:/Data"

Reads from: <DATA_DIR>/sub-N/eeg/sub-N_task-Sleep_acq-psg_events.txt
Output:     thesis/figures/fig_subject_variation.pdf
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO    = Path(__file__).resolve().parents[2]
FIG_DIR = REPO / "thesis" / "figures"

# Default data location — override with --data argument
DEFAULT_DATA = Path(r"C:\Users\DerHo\Desktop\Data")

STAGES         = ["Wake", "N1",   "N2",    "N3",    "REM"]
# Okabe-Ito colorblind-safe palette
COLORS         = ["#E69F00", "#56B4E9", "#0072B2", "#009E73", "#CC79A7"]
SUBJECT_COLORS = ["#D55E00", "#0072B2", "#E69F00", "#CC79A7"]  # one per subject
LABEL_MAP = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
VALID     = set(LABEL_MAP.keys())   # excludes 8=Disconnection, -2=Artifact

# Global BOAS averages (hardcoded from full 128-subject count)
GLOBAL_AVG = {"Wake": 16.0, "N1": 3.7, "N2": 60.3, "N3": 4.4, "REM": 15.7}

# Claimed extreme subjects
CLAIMED = {
    "N2-heavy":   ("sub-114", "N2",   79.96),
    "Wake-heavy": ("sub-34",  "Wake", 51.53),
    "N3-heavy":   ("sub-2",   "N3",   22.56),
    "REM-heavy":  ("sub-119", "REM",  25.17),
}


def load_subject_distribution(events_file: Path) -> dict | None:
    try:
        df = pd.read_csv(events_file, sep="\t")
        if "stage_hum" not in df.columns:
            return None
        labels = df["stage_hum"].values
        labels = labels[np.isin(labels, list(VALID))]
        if len(labels) == 0:
            return None
        total = len(labels)
        return {name: 100.0 * np.sum(labels == code) / total
                for code, name in LABEL_MAP.items()}
    except Exception as e:
        print(f"  Warning: could not read {events_file.name}: {e}")
        return None


def load_all_subjects(data_dir: Path) -> dict:
    distributions = {}
    subject_dirs = sorted(data_dir.glob("sub-*"))
    if not subject_dirs:
        print(f"ERROR: No sub-* directories found in {data_dir}")
        sys.exit(1)
    for sub_dir in subject_dirs:
        sid = sub_dir.name
        events_files = list((sub_dir / "eeg").glob(f"{sid}_*_events.txt"))
        if not events_files:
            continue
        dist = load_subject_distribution(events_files[0])
        if dist:
            distributions[sid] = dist
    print(f"Loaded {len(distributions)} subjects from {data_dir}")
    return distributions


def find_and_verify(distributions: dict) -> list:
    extremes = {
        "N2-heavy":   max(distributions, key=lambda s: distributions[s].get("N2",   0)),
        "Wake-heavy": max(distributions, key=lambda s: distributions[s].get("Wake", 0)),
        "N3-heavy":   max(distributions, key=lambda s: distributions[s].get("N3",   0)),
        "REM-heavy":  max(distributions, key=lambda s: distributions[s].get("REM",  0)),
    }

    print("\n=== Verification ===")
    rows = []
    for profile, found_sid in extremes.items():
        claimed_sid, stage_key, claimed_pct = CLAIMED[profile]
        actual_pct = distributions[found_sid].get(stage_key, 0.0)
        match = "OK" if abs(actual_pct - claimed_pct) < 1.0 else "MISMATCH"
        print(f"  {profile:12s}: found={found_sid:8s} claimed={claimed_sid:8s} | "
              f"{stage_key}={actual_pct:.2f}% (claimed {claimed_pct:.2f}%) [{match}]")
        rows.append((profile, found_sid, distributions[found_sid]))

    print("\n=== Full distributions ===")
    for profile, sid, dist in rows:
        line = "  ".join(f"{s}={dist.get(s,0):.1f}%" for s in STAGES)
        print(f"  {sid} ({profile}): {line}")

    return rows


def generate_figure(rows: list) -> None:
    n = len(rows)
    x = np.arange(len(STAGES))
    width = 0.16
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (profile, sid, dist) in enumerate(rows):
        vals = [dist.get(s, 0.0) for s in STAGES]
        ax.bar(x + offsets[i], vals, width,
               label=f"{sid} ({profile})",
               color=SUBJECT_COLORS[i], alpha=0.82,
               edgecolor="black", linewidth=0.5)

    # Dataset average as dashed reference lines
    for xi, stage in enumerate(STAGES):
        gp = GLOBAL_AVG[stage]
        ax.hlines(gp, xi - 0.42, xi + 0.42,
                  colors="black", linestyles="--", linewidth=1.1)

    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_xlabel("Sleep Stage")
    ax.set_ylabel("Percentage of Epochs (%)")
    ax.set_title(
        "Inter-Subject Variability in Sleep Stage Distribution\n"
        "Four extreme subjects · BOAS dataset · dashed line = dataset average"
    )
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_ylim(0, 88)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_subject_variation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="Path to BOAS Data directory containing sub-* folders")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Usage: python generate_subject_variation_figure.py --data <path>")
        sys.exit(1)

    distributions = load_all_subjects(data_dir)
    rows = find_and_verify(distributions)
    generate_figure(rows)


if __name__ == "__main__":
    main()
