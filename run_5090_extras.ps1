# =============================================================================
#  run_5090_extras.ps1  —  Extra benchmarks + figures on the RTX-5090 machine
# =============================================================================
#
#  Run from the BachlorThesis repo root:
#
#      cd "C:\Users\DerHo\Desktop\BachlorThesis_V4"
#      .\run_5090_extras.ps1
#
#  Optional flags:
#    -ModelCache  "path\to\loso_model_cache"   # skip confusion matrix cold run
#    -NoPush                                   # skip git push at end
#
#  What this runs (in order):
#    1.  Check feature cache is populated
#    2.  Regenerate viability figures from existing 5090 CSV (instant)
#    3.  Generate global-vs-perfold figure from hardcoded data (instant)
#    4.  ANOVA vs MI benchmark              (~5 min,  10 subjects)
#    5.  Feature selection benchmark        (~10 min,  5 subjects)
#    6.  Global vs per-fold benchmark       (~20 min, up to 20 subjects)
#    7.  Re-generate global-vs-perfold figure with fresh CSV data
#    8.  Confusion matrix figure            (~7 s warm / ~43 min cold)
#    9.  git commit + push
#
#  NOTE: update_figures_pc1.py (speedup bar, crossover) requires PC1 timing
#  data and is NOT included here.
# =============================================================================

param(
    [string]$ModelCache = "",
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
$CACHE = "results\features_cache_global"

function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Run-Python([string[]]$args_list, [string]$label) {
    Write-Host "  > python $($args_list -join ' ')" -ForegroundColor DarkGray
    python @args_list
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$label exited with code $LASTEXITCODE — continuing"
    }
}

# ── pre-flight ────────────────────────────────────────────────────────────────
if (-not (Test-Path "generate_thesis_figures.py")) {
    Write-Error "Run this from the BachlorThesis repo root."
    exit 1
}
try { python --version | Out-Null } catch { Write-Error "Python not on PATH"; exit 1 }

# ── Step 1: feature cache check ───────────────────────────────────────────────
Write-Step "Step 1 / 8  —  Feature cache check"
$cached = (Get-ChildItem "$CACHE\subject_*_full.npz" -ErrorAction SilentlyContinue).Count
Write-Host "  Found $cached / 128 subjects in $CACHE" -ForegroundColor Yellow
if ($cached -lt 10) {
    Write-Error "Feature cache is nearly empty ($cached files). Run rebuild_feature_cache.py first."
    exit 1
}
if ($cached -lt 128) {
    Write-Warning "Only $cached / 128 subjects cached — benchmarks will run on available subjects only."
}

# ── Step 2: regenerate viability figures (5090 CSV already in repo) ───────────
Write-Step "Step 2 / 8  —  Viability figures from existing 5090 data (instant)"
Run-Python @("generate_thesis_figures.py") -label "Thesis figures"

# ── Step 3: global-vs-perfold figure (hardcoded data, no benchmark needed) ───
Write-Step "Step 3 / 8  —  Global-vs-perfold figure (hardcoded data, instant)"
Run-Python @("generate_globalvsfold_figure.py") -label "Global vs perfold figure"

# ── Step 4: ANOVA vs MI (~5 min) ──────────────────────────────────────────────
Write-Step "Step 4 / 8  —  ANOVA vs MI benchmark  (~5 min)"
Run-Python @("benchmarks_and_tests\benchmark_anova_vs_mi.py") -label "ANOVA vs MI"

# ── Step 5: feature selection benchmark (~10 min) ─────────────────────────────
Write-Step "Step 5 / 8  —  Feature selection benchmark  (~10 min)"
Run-Python @("benchmarks_and_tests\benchmark_feature_selection.py") -label "Feature selection"

# ── Step 6: global vs per-fold benchmark (~20 min) ────────────────────────────
Write-Step "Step 6 / 8  —  Global vs per-fold benchmark  (~20 min)"
Run-Python @("benchmark_global_vs_perfold.py", "--cache", $CACHE) -label "Global vs per-fold"

# ── Step 7: regenerate global-vs-perfold figure with fresh CSV data ───────────
Write-Step "Step 7 / 8  —  Global-vs-perfold figure (fresh data)"
Run-Python @("generate_globalvsfold_figure.py") -label "Global vs perfold figure (fresh)"

# ── Step 8: confusion matrix ──────────────────────────────────────────────────
Write-Step "Step 8 / 8  —  Confusion matrix figure"
if ($ModelCache -ne "" -and (Test-Path $ModelCache)) {
    $modelCount = (Get-ChildItem "$ModelCache\*.joblib" -ErrorAction SilentlyContinue).Count
    Write-Host "  Model cache: $ModelCache ($modelCount models)" -ForegroundColor Green
    Run-Python @("generate_missing_figures.py", "--cache", $CACHE, "--model-cache", $ModelCache) `
               -label "Confusion matrix (warm)"
} else {
    Write-Host "  No model cache provided — cold run (~43 min for XGBoost best config)" -ForegroundColor Yellow
    Write-Host "  Tip: pass -ModelCache 'path\to\loso_model_cache' to use a cached run" -ForegroundColor DarkGray
    Run-Python @("generate_missing_figures.py", "--cache", $CACHE) -label "Confusion matrix (cold)"
}

# ── commit + push ─────────────────────────────────────────────────────────────
Write-Step "git commit + push"

git add "thesis\figures\" `
        "thesis\tables\" `
        "results\benchmark\" `
        "results\global_vs_perfold_benchmark.csv" `
        "results\global_vs_perfold_benchmark.json" 2>$null

$branch  = git rev-parse --abbrev-ref HEAD
$datestr = Get-Date -Format "yyyyMMdd"
git commit -m "5090 extras: benchmarks + figures regenerated $datestr"

if (-not $NoPush) {
    Write-Host "  Pushing to $branch ..." -ForegroundColor Cyan
    git push -u origin $branch
}

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host "  ALL DONE" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green
