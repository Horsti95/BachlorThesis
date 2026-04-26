"""
Page-cache effect quantification + cross-machine spot check.

Runs three experiments back-to-back on the same machine:

  1. XGBoost best config (corr=None, top_k=None) — cold -> warm WITHOUT flush
     (current default behaviour; warm load benefits from OS page cache)
  2. XGBoost best config — cold -> FLUSH PAGE CACHE -> warm
     (warm load forced to read from SSD)
  3. Random Forest best config — cold -> warm (no flush)
     (RF cache exceeds typical RAM, so page cache effect is minimal anyway)

Output: results/pagecache_comparison/comparison_summary.json + .csv

Usage:
  python run_pagecache_comparison.py --data-path "C:/path/to/Data"
  python run_pagecache_comparison.py --data-path "C:/path/to/Data" --subjects 128
"""
import argparse
import gc
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from run_full_pipeline import (
    LOSO_CACHE_DIR,
    run_stage1_feature_extraction,
    run_stage2_training,
)

RESULTS_DIR = Path("results/pagecache_comparison")


def clear_loso_cache() -> int:
    if LOSO_CACHE_DIR.exists():
        n = len(list(LOSO_CACHE_DIR.glob("*.joblib")))
        shutil.rmtree(LOSO_CACHE_DIR)
        return n
    return 0


def flush_page_cache(target_gb: float) -> float:
    """
    Force OS to evict page cache by allocating a large numpy array.

    Allocating ~`target_gb` GB pressures Windows/Linux into freeing
    page-cache memory to satisfy the request. After deallocation, the
    page cache is largely empty and subsequent file reads come from disk.
    """
    import numpy as np

    print(f"Flushing OS page cache (allocating {target_gb:.0f} GB)...")
    started = time.time()
    n_floats = int(target_gb * (1024 ** 3) // 8)
    big = np.zeros(n_floats, dtype=np.float64)
    big += 1.0
    del big
    gc.collect()
    elapsed = time.time() - started
    print(f"  flush complete in {elapsed:.1f}s")
    return elapsed


def run_one(
    label: str,
    subjects,
    output_dir: Path,
    model: str,
    flush_before_warm: bool,
    flush_gb: float,
):
    print("\n" + "=" * 80)
    print(f"RUN: {label}")
    print("=" * 80)

    deleted = clear_loso_cache()
    print(f"Cleared {deleted} cached models before COLD")

    cold_start = time.time()
    cold = run_stage2_training(
        subjects=subjects,
        output_dir=output_dir,
        experiment_name=f"cold_{label}",
        enable_model_cache=True,
        models=[model],
        correlation_thresholds=[None],
        top_k_features=[None],
    )
    cold_seconds = time.time() - cold_start
    print(f"  COLD: {cold_seconds:.1f}s")

    flush_seconds = 0.0
    if flush_before_warm:
        flush_seconds = flush_page_cache(flush_gb)

    warm_start = time.time()
    warm = run_stage2_training(
        subjects=subjects,
        output_dir=output_dir,
        experiment_name=f"warm_{label}",
        enable_model_cache=True,
        models=[model],
        correlation_thresholds=[None],
        top_k_features=[None],
    )
    warm_seconds = time.time() - warm_start
    print(f"  WARM: {warm_seconds:.1f}s")
    speedup = cold_seconds / warm_seconds if warm_seconds > 0 else 0.0
    print(f"  SPEEDUP: {speedup:.1f}x")

    cold_acc = (cold.get("best_result") or {}).get("accuracy_mean")
    warm_acc = (warm.get("best_result") or {}).get("accuracy_mean")

    return {
        "label": label,
        "model": model,
        "flush_before_warm": flush_before_warm,
        "flush_seconds": round(flush_seconds, 2),
        "cold_seconds": round(cold_seconds, 3),
        "warm_seconds": round(warm_seconds, 3),
        "speedup_factor": round(speedup, 2),
        "cold_best_accuracy": cold_acc,
        "warm_best_accuracy": warm_acc,
        "cold_cache_metrics": cold.get("model_cache_metrics", {}),
        "warm_cache_metrics": warm.get("model_cache_metrics", {}),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Quantify OS page-cache contribution to warm-run speed; spot-check XGB+RF on host"
    )
    parser.add_argument("--data-path", required=True, help="Path to BOAS dataset root")
    parser.add_argument("--subjects", type=int, default=128, help="Number of subjects (default: 128)")
    parser.add_argument(
        "--flush-gb",
        type=float,
        default=50.0,
        help="GB to allocate to evict page cache (default: 50; tune to RAM size)",
    )
    parser.add_argument(
        "--skip-rf",
        action="store_true",
        help="Skip the Random Forest run (only compare XGB with/without flush)",
    )
    args = parser.parse_args()

    subjects = [str(i) for i in range(1, args.subjects + 1)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Stage 1 once
    print("\n[Stage 1] Ensuring feature cache is ready...")
    stage1_start = time.time()
    run_stage1_feature_extraction(
        data_path=args.data_path,
        subjects=subjects,
        experiment_name=f"pagecache_stage1_{timestamp}",
    )
    print(f"  Stage 1 done in {time.time() - stage1_start:.1f}s")

    runs = []

    runs.append(
        run_one(
            "xgb_no_flush",
            subjects,
            output_dir,
            model="xgboost",
            flush_before_warm=False,
            flush_gb=args.flush_gb,
        )
    )

    runs.append(
        run_one(
            "xgb_with_flush",
            subjects,
            output_dir,
            model="xgboost",
            flush_before_warm=True,
            flush_gb=args.flush_gb,
        )
    )

    if not args.skip_rf:
        runs.append(
            run_one(
                "rf_no_flush",
                subjects,
                output_dir,
                model="random_forest",
                flush_before_warm=False,
                flush_gb=args.flush_gb,
            )
        )

    summary = {
        "timestamp": timestamp,
        "n_subjects": len(subjects),
        "flush_gb_target": args.flush_gb,
        "runs": runs,
    }

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    csv_path = output_dir / "comparison_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("label,model,flush_before_warm,cold_seconds,warm_seconds,speedup_factor,cold_best_accuracy\n")
        for r in runs:
            f.write(
                f"{r['label']},{r['model']},{r['flush_before_warm']},"
                f"{r['cold_seconds']},{r['warm_seconds']},{r['speedup_factor']},"
                f"{r['cold_best_accuracy']}\n"
            )

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    for r in runs:
        flush_tag = "FLUSH" if r["flush_before_warm"] else "no flush"
        print(
            f"  {r['label']:<22} {flush_tag:<10}  "
            f"cold={r['cold_seconds']:.0f}s  warm={r['warm_seconds']:.1f}s  "
            f"speedup={r['speedup_factor']:.1f}x"
        )

    if len(runs) >= 2 and runs[0]["model"] == runs[1]["model"]:
        no_flush_warm = runs[0]["warm_seconds"]
        with_flush_warm = runs[1]["warm_seconds"]
        ratio = with_flush_warm / no_flush_warm if no_flush_warm > 0 else 0
        print(
            f"\n  Page-cache contribution: warm time grows {ratio:.2f}x when flushed "
            f"({no_flush_warm:.1f}s -> {with_flush_warm:.1f}s)"
        )

    print(f"\n  Summary: {summary_path}")
    print(f"  CSV:     {csv_path}")


if __name__ == "__main__":
    main()
