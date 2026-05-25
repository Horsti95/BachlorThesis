"""
Rebuild feature cache (Stage 1 only) without running model training.

Repopulates results/features_cache_global/ from raw EDF data.
Already-cached subjects are skipped automatically (incremental).

Usage:
    python rebuild_feature_cache.py --data-path "C:\\Users\\DerHo\\Desktop\\Data"
    python rebuild_feature_cache.py --data-path "C:\\Users\\DerHo\\Desktop\\Data" --n-subjects 20
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CACHE_DIR = Path("results/features_cache_global")


def main():
    parser = argparse.ArgumentParser(description="Rebuild feature cache (Stage 1 only)")
    parser.add_argument("--data-path", required=True,
                        help="Path to BOAS dataset root (containing sub-1/, sub-2/, ...)")
    parser.add_argument("--n-subjects", type=int, default=128,
                        help="Number of subjects to process (default: 128)")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: data-path not found: {data_path}")
        sys.exit(1)

    subjects = [str(i) for i in range(1, args.n_subjects + 1)]

    existing = len(list(CACHE_DIR.glob("subject_*_full.npz"))) if CACHE_DIR.exists() else 0
    to_extract = args.n_subjects - existing
    print(f"Feature cache: {existing}/{args.n_subjects} subjects already cached")
    if to_extract <= 0:
        print("Cache already complete — nothing to do.")
        return

    print(f"Extracting {to_extract} missing subjects...")

    from run_full_pipeline import run_stage1_feature_extraction
    t0 = time.time()
    result = run_stage1_feature_extraction(
        data_path=str(data_path),
        subjects=subjects,
    )
    elapsed = time.time() - t0

    cached_now = len(list(CACHE_DIR.glob("subject_*_full.npz")))
    print(f"\nDone in {elapsed / 60:.1f} min")
    print(f"  Cache hits  : {result.get('cache_hits', '?')} (skipped)")
    print(f"  Cache misses: {result.get('cache_misses', '?')} (extracted)")
    print(f"  Total cached: {cached_now}/{args.n_subjects} subjects")
    print(f"  Cache dir   : {CACHE_DIR.resolve()}")


if __name__ == "__main__":
    main()
