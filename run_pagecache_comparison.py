"""
Page-cache effect quantification + cross-machine spot check.

Two modes:

  Default (cross-device check, e.g. on the 5090):
      cold -> warm (no flush)        for XGB and RF best configs

  --with-flush (full page-cache test, e.g. on PC1):
      cold -> warm (no flush) -> FLUSH PAGE CACHE -> warm (with flush)
      for XGB and RF best configs

The single-cold-double-warm structure lets us measure how much of the
"warm" speedup comes from realistic OS page caching vs true SSD reads,
without paying for two cold runs per model.

Output: results/pagecache_comparison/run_<ts>/comparison_summary.{json,csv}

Usage:
  # 5090: cross-device comparison (no flush, both models)
  python run_pagecache_comparison.py --data-path "C:/path/Data"

  # PC1: full page-cache test (both models, with flush)
  python run_pagecache_comparison.py --data-path "C:/path/Data" --with-flush --flush-gb 10

  # XGB only:
  python run_pagecache_comparison.py --data-path "C:/path/Data" --with-flush --models xgboost
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
    """
    import numpy as np

    print(f"  Flushing OS page cache (allocating {target_gb:.0f} GB)...")
    started = time.time()
    n_floats = int(target_gb * (1024 ** 3) // 8)
    big = np.zeros(n_floats, dtype=np.float64)
    big += 1.0
    del big
    gc.collect()
    elapsed = time.time() - started
    print(f"    flush complete in {elapsed:.1f}s")
    return elapsed


def time_run(label, subjects, output_dir, model):
    started = time.time()
    result = run_stage2_training(
        subjects=subjects,
        output_dir=output_dir,
        experiment_name=label,
        enable_model_cache=True,
        models=[model],
        correlation_thresholds=[None],
        top_k_features=[None],
    )
    elapsed = time.time() - started
    return elapsed, result


def measure_model(model: str, subjects, output_dir: Path, with_flush: bool, flush_gb: float):
    print("\n" + "=" * 80)
    print(f"MODEL: {model.upper()}")
    print("=" * 80)

    deleted = clear_loso_cache()
    print(f"  Cleared {deleted} cached models before COLD")

    print(f"\n  [1/{'3' if with_flush else '2'}] COLD run...")
    cold_seconds, cold_result = time_run(f"cold_{model}", subjects, output_dir, model)
    print(f"    COLD: {cold_seconds:.1f}s")

    print(f"\n  [2/{'3' if with_flush else '2'}] WARM run (no flush)...")
    warm_no_flush_seconds, warm_no_flush_result = time_run(
        f"warm_noflush_{model}", subjects, output_dir, model
    )
    print(f"    WARM (no flush): {warm_no_flush_seconds:.1f}s")
    print(f"    Speedup vs cold: {cold_seconds/warm_no_flush_seconds:.1f}x")

    flush_seconds = 0.0
    warm_with_flush_seconds = None
    warm_with_flush_result = None
    if with_flush:
        print(f"\n  [3/3] Flushing page cache then WARM run (with flush)...")
        flush_seconds = flush_page_cache(flush_gb)
        warm_with_flush_seconds, warm_with_flush_result = time_run(
            f"warm_flushed_{model}", subjects, output_dir, model
        )
        print(f"    WARM (with flush): {warm_with_flush_seconds:.1f}s")
        print(f"    Speedup vs cold: {cold_seconds/warm_with_flush_seconds:.1f}x")

    return {
        "model": model,
        "cold_seconds": round(cold_seconds, 3),
        "warm_no_flush_seconds": round(warm_no_flush_seconds, 3),
        "warm_with_flush_seconds": (
            round(warm_with_flush_seconds, 3) if warm_with_flush_seconds is not None else None
        ),
        "flush_seconds": round(flush_seconds, 2),
        "speedup_no_flush": round(cold_seconds / warm_no_flush_seconds, 2)
        if warm_no_flush_seconds > 0
        else 0.0,
        "speedup_with_flush": (
            round(cold_seconds / warm_with_flush_seconds, 2)
            if warm_with_flush_seconds is not None and warm_with_flush_seconds > 0
            else None
        ),
        "page_cache_factor": (
            round(warm_with_flush_seconds / warm_no_flush_seconds, 2)
            if warm_with_flush_seconds is not None and warm_no_flush_seconds > 0
            else None
        ),
        "cold_best_accuracy": (cold_result.get("best_result") or {}).get("accuracy_mean"),
        "warm_no_flush_accuracy": (warm_no_flush_result.get("best_result") or {}).get(
            "accuracy_mean"
        ),
        "warm_with_flush_accuracy": (
            (warm_with_flush_result.get("best_result") or {}).get("accuracy_mean")
            if warm_with_flush_result
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Page-cache effect + cross-machine comparison (XGB and RF best configs)"
    )
    parser.add_argument("--data-path", required=True, help="Path to BOAS dataset root")
    parser.add_argument(
        "--subjects", type=int, default=128, help="Number of subjects (default: 128)"
    )
    parser.add_argument(
        "--with-flush",
        action="store_true",
        help="Add a third warm run with OS page cache flushed (PC1 mode)",
    )
    parser.add_argument(
        "--flush-gb",
        type=float,
        default=50.0,
        help="GB to allocate to evict page cache (default: 50; tune to RAM size)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="xgboost,random_forest",
        help="Comma-separated models to test (default: xgboost,random_forest)",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    subjects = [str(i) for i in range(1, args.subjects + 1)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    print(f"Mode:   {'WITH flush' if args.with_flush else 'NO flush'}")
    print(f"Models: {models}")
    print(f"N subj: {len(subjects)}")

    print("\n[Stage 1] Ensuring feature cache is ready...")
    stage1_start = time.time()
    run_stage1_feature_extraction(
        data_path=args.data_path,
        subjects=subjects,
        experiment_name=f"pagecache_stage1_{timestamp}",
    )
    print(f"  Stage 1 done in {time.time() - stage1_start:.1f}s")

    runs = []
    for model in models:
        runs.append(
            measure_model(
                model=model,
                subjects=subjects,
                output_dir=output_dir,
                with_flush=args.with_flush,
                flush_gb=args.flush_gb,
            )
        )

    summary = {
        "timestamp": timestamp,
        "n_subjects": len(subjects),
        "with_flush": args.with_flush,
        "flush_gb_target": args.flush_gb if args.with_flush else None,
        "models": models,
        "runs": runs,
    }

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    csv_path = output_dir / "comparison_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "model,cold_seconds,warm_no_flush_seconds,warm_with_flush_seconds,"
            "speedup_no_flush,speedup_with_flush,page_cache_factor,"
            "cold_accuracy\n"
        )
        for r in runs:
            f.write(
                f"{r['model']},{r['cold_seconds']},{r['warm_no_flush_seconds']},"
                f"{r['warm_with_flush_seconds']},"
                f"{r['speedup_no_flush']},{r['speedup_with_flush']},"
                f"{r['page_cache_factor']},{r['cold_best_accuracy']}\n"
            )

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    for r in runs:
        print(f"\n  {r['model'].upper()}:")
        print(f"    cold:                 {r['cold_seconds']:.0f}s")
        print(
            f"    warm (no flush):      {r['warm_no_flush_seconds']:.1f}s "
            f"(speedup {r['speedup_no_flush']:.1f}x)"
        )
        if r["warm_with_flush_seconds"] is not None:
            print(
                f"    warm (with flush):    {r['warm_with_flush_seconds']:.1f}s "
                f"(speedup {r['speedup_with_flush']:.1f}x)"
            )
            print(
                f"    page-cache factor:    {r['page_cache_factor']:.2f}x slower when flushed"
            )

    print(f"\n  Summary: {summary_path}")
    print(f"  CSV:     {csv_path}")


if __name__ == "__main__":
    main()
