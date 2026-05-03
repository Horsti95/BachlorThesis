#!/usr/bin/env python3
"""Quick status check for running benchmark."""

import sys
from pathlib import Path
from datetime import datetime
import time

TERMINAL_ID = "a90a6fb1-ff70-4507-9900-a0a313fa87e3"

def check_status():
    """Check benchmark progress from terminal output."""
    print("\n" + "="*70)
    print(f"  Benchmark Status Check - {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)
    
    # Check if benchmark results exist
    results_files = list(Path("results").glob("benchmark_results_*.json"))
    if results_files:
        print("✅ BENCHMARK COMPLETE!")
        latest = sorted(results_files)[-1]
        print(f"Results saved: {latest}")
        print(f"File size: {latest.stat().st_size / 1024:.1f} KB")
        return True
    
    # Check if cache directory growing (indicator of active training)
    cache_dir = Path("results/loso_model_cache")
    if cache_dir.exists():
        joblib_count = len(list(cache_dir.glob("*.joblib")))
        print(f"📊 Cached models: {joblib_count}")
        
        if joblib_count > 0:
            print(f"✓ Training in progress (models being cached)")
        else:
            print(f"⏳ Waiting for first model to cache...")
    
    print("\nTerminal ID: " + TERMINAL_ID)
    print("To check output: get_terminal_output(id='" + TERMINAL_ID + "')")
    print("\n" + "="*70)
    return False

if __name__ == "__main__":
    is_complete = check_status()
    sys.exit(0 if is_complete else 1)
