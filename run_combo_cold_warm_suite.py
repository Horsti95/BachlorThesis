"""
Run thesis combinations one-by-one with cold/warm benchmarking.

Workflow per combination:
1) Clear LOSO model cache
2) Run combo cold (cache miss expected)
3) Run same combo warm (cache hit expected)
4) Save combo report and print comparison
5) Clear cache before next combination

Optionally, after all 18 combinations, run additional test/exploration scripts.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from run_full_pipeline import (
    LOSO_CACHE_DIR,
    RESULTS_DIR,
    THESIS_CORRELATION,
    THESIS_MODELS,
    THESIS_TOP_K,
    check_data,
    check_environment,
    collect_system_info,
    run_stage1_feature_extraction,
    run_stage2_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run each thesis combo as cold+warm benchmark with cache reset between combos"
    )

    size_group = parser.add_mutually_exclusive_group(required=False)
    size_group.add_argument("--quick", action="store_true", help="3 subjects")
    size_group.add_argument("--pilot", action="store_true", help="10 subjects")
    size_group.add_argument("--full", action="store_true", help="128 subjects")

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to BOAS dataset root (required unless using --extras-only)",
    )
    parser.add_argument(
        "--run-extras",
        action="store_true",
        help="After all 18 combos, run test/exploration scripts",
    )
    parser.add_argument(
        "--extras-only",
        metavar="OUTPUT_DIR",
        default=None,
        help="Skip benchmark; run only the extra scripts and write logs into OUTPUT_DIR",
    )
    parser.add_argument(
        "--extra-timeout-seconds",
        type=int,
        default=300,
        help="Timeout per extra script in seconds (default: 300; 0 = no timeout)",
    )
    parser.add_argument(
        "--benchmarks-only",
        action="store_true",
        help="When running extras, skip generate_*.py plot scripts and run only benchmark/test scripts",
    )
    return parser.parse_args()


def determine_subjects(args: argparse.Namespace) -> List[str]:
    if args.quick:
        return [str(i) for i in range(1, 4)]
    if args.pilot:
        return [str(i) for i in range(1, 11)]
    return [str(i) for i in range(1, 129)]


def clear_loso_cache() -> int:
    if LOSO_CACHE_DIR.exists():
        n = len(list(LOSO_CACHE_DIR.glob("*.joblib")))
        shutil.rmtree(LOSO_CACHE_DIR)
        return n
    return 0


def safe_val(value: Optional[object]) -> str:
    if value is None:
        return "none"
    return str(value).replace(".", "p")


def make_combo_name(model: str, corr: Optional[float], top_k: Optional[int]) -> str:
    return f"{model}_corr{safe_val(corr)}_topk{safe_val(top_k)}"


def run_combo_cold_warm(
    subjects: List[str],
    output_root: Path,
    combo_index: int,
    combo_total: int,
    model: str,
    corr: Optional[float],
    top_k: Optional[int],
) -> Dict:
    combo_name = make_combo_name(model, corr, top_k)
    combo_dir = output_root / f"{combo_index:02d}_{combo_name}"
    combo_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 80)
    print(f"[{combo_index}/{combo_total}] {combo_name}")
    print("-" * 80)

    deleted_before = clear_loso_cache()
    print(f"  Cleared cache before COLD run: {deleted_before} files")

    cold_start = time.time()
    cold = run_stage2_training(
        subjects=subjects,
        output_dir=combo_dir,
        experiment_name=f"cold_{combo_name}",
        enable_model_cache=True,
        models=[model],
        correlation_thresholds=[corr],
        top_k_features=[top_k],
    )
    cold_seconds = time.time() - cold_start

    warm_start = time.time()
    warm = run_stage2_training(
        subjects=subjects,
        output_dir=combo_dir,
        experiment_name=f"warm_{combo_name}",
        enable_model_cache=True,
        models=[model],
        correlation_thresholds=[corr],
        top_k_features=[top_k],
    )
    warm_seconds = time.time() - warm_start

    speedup = (cold_seconds / warm_seconds) if warm_seconds > 0 else 0.0
    saved = cold_seconds - warm_seconds

    cold_cache = cold.get("model_cache_metrics", {})
    warm_cache = warm.get("model_cache_metrics", {})

    print(
        f"  COLD: {cold_seconds:.1f}s | hits={cold_cache.get('hits', 0)} misses={cold_cache.get('misses', 0)}"
    )
    print(
        f"  WARM: {warm_seconds:.1f}s | hits={warm_cache.get('hits', 0)} misses={warm_cache.get('misses', 0)}"
    )
    print(f"  SPEEDUP: {speedup:.2f}x | SAVED: {saved:.1f}s")

    deleted_after = clear_loso_cache()
    print(f"  Cleared cache after WARM run: {deleted_after} files")

    report = {
        "combo_index": combo_index,
        "combo_name": combo_name,
        "config": {
            "model": model,
            "correlation_threshold": corr,
            "top_k_features": top_k,
        },
        "cold": {
            "seconds": round(cold_seconds, 3),
            "n_configs": cold.get("n_configs", 0),
            "n_folds": cold.get("n_folds", 0),
            "total_runs": cold.get("total_runs", 0),
            "cache": cold_cache,
            "best_result": cold.get("best_result"),
        },
        "warm": {
            "seconds": round(warm_seconds, 3),
            "n_configs": warm.get("n_configs", 0),
            "n_folds": warm.get("n_folds", 0),
            "total_runs": warm.get("total_runs", 0),
            "cache": warm_cache,
            "best_result": warm.get("best_result"),
        },
        "comparison": {
            "speedup_factor": round(speedup, 4),
            "time_saved_seconds": round(saved, 3),
            "time_saved_percent": round((saved / cold_seconds) * 100, 2) if cold_seconds > 0 else 0.0,
        },
        "cache_cleanup": {
            "deleted_before_cold": deleted_before,
            "deleted_after_warm": deleted_after,
        },
    }

    with open(combo_dir / "combo_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def write_summary_csv(summary_rows: List[Dict], csv_path: Path) -> None:
    fieldnames = [
        "combo_index",
        "combo_name",
        "model",
        "correlation_threshold",
        "top_k_features",
        "cold_seconds",
        "warm_seconds",
        "speedup_factor",
        "time_saved_seconds",
        "cold_hits",
        "cold_misses",
        "warm_hits",
        "warm_misses",
        "cold_best_accuracy",
        "warm_best_accuracy",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def find_extra_scripts(repo_root: Path, benchmarks_only: bool = False) -> List[Path]:
    candidates: List[Path] = []

    benchmark_patterns = [
        "benchmarks_and_tests/*.py",
        "testing/*.py",
        "model_tryouts/*.py",
    ]
    plot_patterns = [
        "generate_*.py",
    ]

    for pattern in benchmark_patterns:
        candidates.extend(repo_root.glob(pattern))
    if not benchmarks_only:
        for pattern in plot_patterns:
            candidates.extend(repo_root.glob(pattern))

    ignore_names = {
        "__init__.py",
        "run_combo_cold_warm_suite.py",
        # always skip — overwrites manually updated tab2 with stale CSV data
        "generate_thesis_figures.py",
        # always skip — requires --subjects/--cache-dir args and runs hours of training
        "benchmark_njobs.py",
    }

    unique_sorted = sorted({p.resolve() for p in candidates if p.name not in ignore_names})
    return [Path(p) for p in unique_sorted]


def run_extra_scripts(
    repo_root: Path, output_root: Path, timeout_seconds: int, benchmarks_only: bool = False
) -> Dict:
    scripts = find_extra_scripts(repo_root, benchmarks_only=benchmarks_only)
    logs_dir = output_root / "extras_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    mode_label = "BENCHMARKS ONLY" if benchmarks_only else "ALL SCRIPTS"
    print("\n" + "=" * 80)
    print(f"RUNNING EXTRA SCRIPTS ({len(scripts)} found, {mode_label})")
    print("=" * 80)

    results = []
    for i, script_path in enumerate(scripts, start=1):
        rel = script_path.relative_to(repo_root)
        print(f"[{i}/{len(scripts)}] {rel}")

        cmd = [sys.executable, str(script_path)]
        started = time.time()
        timed_out = False

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=(timeout_seconds if timeout_seconds > 0 else None),
            )
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = -1
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + "\n[TIMEOUT]"
        except KeyboardInterrupt:
            print(f"\n  -> INTERRUPTED — skipping remaining extra scripts")
            break

        elapsed = time.time() - started
        log_base = rel.as_posix().replace("/", "__").replace(".py", "")

        with open(logs_dir / f"{log_base}.stdout.log", "w", encoding="utf-8") as f_out:
            f_out.write(stdout)
        with open(logs_dir / f"{log_base}.stderr.log", "w", encoding="utf-8") as f_err:
            f_err.write(stderr)

        result = {
            "script": str(rel.as_posix()),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "elapsed_seconds": round(elapsed, 2),
            "stdout_log": str((logs_dir / f"{log_base}.stdout.log").resolve().relative_to(repo_root).as_posix()),
            "stderr_log": str((logs_dir / f"{log_base}.stderr.log").resolve().relative_to(repo_root).as_posix()),
        }
        results.append(result)

        status = "OK" if exit_code == 0 and not timed_out else "FAIL"
        print(f"  -> {status} ({elapsed:.1f}s)")

    ok_count = sum(1 for r in results if r["exit_code"] == 0 and not r["timed_out"])
    fail_count = len(results) - ok_count
    return {
        "total_scripts": len(results),
        "ok": ok_count,
        "failed": fail_count,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    # --extras-only does not need --data-path or a size flag
    if not args.extras_only:
        if not args.data_path:
            print("error: --data-path is required unless using --extras-only")
            sys.exit(1)
        if not (args.quick or args.pilot or args.full):
            print("error: one of --quick / --pilot / --full is required unless using --extras-only")
            sys.exit(1)

    subjects = determine_subjects(args)

    if args.extras_only:
        output_root = Path(args.extras_only)
        output_root.mkdir(parents=True, exist_ok=True)
        extras_result = run_extra_scripts(
            repo_root=repo_root,
            output_root=output_root,
            timeout_seconds=args.extra_timeout_seconds,
            benchmarks_only=args.benchmarks_only,
        )
        with open(output_root / "extras_summary.json", "w", encoding="utf-8") as f:
            json.dump(extras_result, f, indent=2, default=str)
        print(f"Extras: {extras_result['ok']} OK / {extras_result['failed']} failed")
        print(f"Logs:   {output_root / 'extras_summary.json'}")
        return

    issues = check_environment()
    if issues:
        print("Missing dependencies:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nInstall first: pip install -r requirements.txt")
        sys.exit(1)

    data_status = check_data(args.data_path, len(subjects))
    if not data_status.get("valid", False):
        print(f"Data check failed: {data_status.get('error', 'unknown error')}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = RESULTS_DIR / f"combo_cold_warm_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("COMBO COLD/WARM SUITE")
    print("=" * 80)
    print(f"Subjects: {len(subjects)}")
    print(f"Output:   {output_root}")
    print(f"Grid:     {len(THESIS_MODELS)} x {len(THESIS_CORRELATION)} x {len(THESIS_TOP_K)}")

    # Stage 1 once so Stage 2 can load cached features for all folds/combos.
    print("\n[Stage 1] Ensuring feature cache is ready...")
    stage1_started = time.time()
    stage1_stats = run_stage1_feature_extraction(
        data_path=args.data_path,
        subjects=subjects,
        experiment_name=f"combo_suite_stage1_{timestamp}",
    )
    stage1_elapsed = time.time() - stage1_started
    print(
        f"  Feature cache: hits={stage1_stats.get('cache_hits', 0)} "
        f"misses={stage1_stats.get('cache_misses', 0)} ({stage1_elapsed:.1f}s)"
    )

    combos = list(itertools.product(THESIS_MODELS, THESIS_CORRELATION, THESIS_TOP_K))

    combo_reports: List[Dict] = []
    summary_rows: List[Dict] = []

    suite_started = time.time()
    for idx, (model, corr, top_k) in enumerate(combos, start=1):
        report = run_combo_cold_warm(
            subjects=subjects,
            output_root=output_root,
            combo_index=idx,
            combo_total=len(combos),
            model=model,
            corr=corr,
            top_k=top_k,
        )
        combo_reports.append(report)

        summary_rows.append(
            {
                "combo_index": idx,
                "combo_name": report["combo_name"],
                "model": model,
                "correlation_threshold": corr,
                "top_k_features": top_k,
                "cold_seconds": report["cold"]["seconds"],
                "warm_seconds": report["warm"]["seconds"],
                "speedup_factor": report["comparison"]["speedup_factor"],
                "time_saved_seconds": report["comparison"]["time_saved_seconds"],
                "cold_hits": report["cold"]["cache"].get("hits", 0),
                "cold_misses": report["cold"]["cache"].get("misses", 0),
                "warm_hits": report["warm"]["cache"].get("hits", 0),
                "warm_misses": report["warm"]["cache"].get("misses", 0),
                "cold_best_accuracy": (report["cold"].get("best_result") or {}).get("accuracy_mean", ""),
                "warm_best_accuracy": (report["warm"].get("best_result") or {}).get("accuracy_mean", ""),
            }
        )

    suite_elapsed = time.time() - suite_started

    summary_json = {
        "timestamp": timestamp,
        "system_info": collect_system_info(),
        "subjects": subjects,
        "n_subjects": len(subjects),
        "stage1": {
            "elapsed_seconds": round(stage1_elapsed, 2),
            "stats": stage1_stats,
        },
        "suite": {
            "elapsed_seconds": round(suite_elapsed, 2),
            "n_combinations": len(combos),
            "combinations": combo_reports,
        },
    }

    summary_json_path = output_root / "combo_suite_summary.json"
    summary_csv_path = output_root / "combo_suite_summary.csv"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, default=str)
    write_summary_csv(summary_rows, summary_csv_path)

    extras_result = None
    if args.run_extras:
        extras_result = run_extra_scripts(
            repo_root=repo_root,
            output_root=output_root,
            timeout_seconds=args.extra_timeout_seconds,
            benchmarks_only=args.benchmarks_only,
        )
        with open(output_root / "extras_summary.json", "w", encoding="utf-8") as f:
            json.dump(extras_result, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    avg_speedup = (
        sum(r["comparison"]["speedup_factor"] for r in combo_reports) / len(combo_reports)
        if combo_reports
        else 0.0
    )
    total_saved = sum(r["comparison"]["time_saved_seconds"] for r in combo_reports)

    print(f"Combinations completed: {len(combo_reports)}")
    print(f"Average speedup:        {avg_speedup:.2f}x")
    print(f"Total time saved:       {total_saved:.1f}s ({total_saved/60:.1f} min)")
    print(f"Summary JSON:           {summary_json_path}")
    print(f"Summary CSV:            {summary_csv_path}")

    if extras_result is not None:
        print(
            f"Extras:                 {extras_result['ok']} OK / "
            f"{extras_result['failed']} failed (of {extras_result['total_scripts']})"
        )
        print(f"Extras summary:         {output_root / 'extras_summary.json'}")


if __name__ == "__main__":
    main()