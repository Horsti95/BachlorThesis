# =============================================================================
#  run_5090_extras.ps1  —  Extra benchmarks + figures on the RTX-5090 machine
# =============================================================================
#
#  Run from the BachlorThesis repo root (no venv needed — activate first):
#
#      cd "C:\Users\DerHo\Desktop\BachlorThesis_V6"
#      .\.venv\Scripts\Activate.ps1        # activate venv
#      .\run_5090_extras.ps1 -ModelCache "results\loso_model_cache"
#
#  Optional flags:
#    -ModelCache  "path\to\loso_model_cache"   # confusion matrix warm (~7s)
#    -NoPush                                   # skip git push at end
#
#  What this runs (in order):
#    1.  Check feature cache is populated
#    2.  Regenerate viability figures from existing 5090 CSV   (instant)
#    3.  Generate global-vs-perfold figure (hardcoded data)    (instant)
#    4.  ANOVA vs MI benchmark              (~5 min,  10 subjects)
#    5.  Feature selection benchmark        (~10 min,  5 subjects)
#    6.  Global vs per-fold benchmark       (~20 min, up to 20 subjects)
#    7.  Re-generate global-vs-perfold figure with fresh CSV
#    8.  Confusion matrix figure            (~7 s warm / ~43 min cold)
#    9.  git commit + push
#
#  NOTE: update_figures_pc1.py (speedup bar, crossover) needs PC1 timing
#  data — run on PC1 once it is available.
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

function Run-Python {
    param([string[]]$ArgList, [string]$Label)
    Write-Host "  > python $($ArgList -join ' ')" -ForegroundColor DarkGray
    python @ArgList
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$Label exited with code $LASTEXITCODE — continuing"
    }
}

# ── pre-flight ────────────────────────────────────────────────────────────────
if (-not (Test-Path "generate_thesis_figures.py")) {
    Write-Error "Run this from the BachlorThesis repo root."
    exit 1
}
try { python --version | Out-Null } catch { Write-Error "Python not on PATH — activate your venv first: .\.venv\Scripts\Activate.ps1"; exit 1 }

# ── Step 1: feature cache check ───────────────────────────────────────────────
Write-Step "Step 1 / 8  —  Feature cache check"
$cached = (Get-ChildItem "$CACHE\subject_*_full.npz" -ErrorAction SilentlyContinue).Count
Write-Host "  Found $cached / 128 subjects in $CACHE" -ForegroundColor Yellow
if ($cached -lt 10) {
    Write-Error "Feature cache nearly empty — only $cached files found. Run rebuild_feature_cache.py first."
    exit 1
}
if ($cached -lt 128) {
    Write-Warning "Only $cached / 128 subjects cached — benchmarks run on available subjects only."
}

# ── Step 2: regenerate viability figures (5090 CSV already in repo) ───────────
Write-Step "Step 2 / 8  —  Viability figures from existing 5090 data (instant)"
Run-Python -ArgList @("generate_thesis_figures.py") -Label "Thesis figures"

# ── Step 3: global-vs-perfold figure (hardcoded data) ────────────────────────
Write-Step "Step 3 / 8  —  Global-vs-perfold figure (hardcoded data, instant)"
Run-Python -ArgList @("generate_globalvsfold_figure.py") -Label "Global vs perfold figure"

# ── Step 4: ANOVA vs MI (~5 min) ──────────────────────────────────────────────
Write-Step "Step 4 / 8  —  ANOVA vs MI benchmark  (~5 min)"
Run-Python -ArgList @("benchmarks_and_tests\benchmark_anova_vs_mi.py") -Label "ANOVA vs MI"

# ── Step 5: feature selection benchmark (~10 min) ─────────────────────────────
Write-Step "Step 5 / 8  —  Feature selection benchmark  (~10 min)"
Run-Python -ArgList @("benchmarks_and_tests\benchmark_feature_selection.py") -Label "Feature selection"

# ── Step 6: global vs per-fold benchmark (~20 min) ────────────────────────────
Write-Step "Step 6 / 8  —  Global vs per-fold benchmark  (~20 min)"
Run-Python -ArgList @("benchmark_global_vs_perfold.py", "--cache", $CACHE) -Label "Global vs per-fold"

# ── Step 7: regenerate global-vs-perfold figure with fresh CSV ───────────────
Write-Step "Step 7 / 8  —  Global-vs-perfold figure (fresh data)"
Run-Python -ArgList @("generate_globalvsfold_figure.py") -Label "Global vs perfold figure (fresh)"

# ── Step 8: confusion matrix ──────────────────────────────────────────────────
Write-Step "Step 8 / 8  —  Confusion matrix figure"
if ($ModelCache -ne "" -and (Test-Path $ModelCache)) {
    $modelCount = (Get-ChildItem "$ModelCache\*.joblib" -ErrorAction SilentlyContinue).Count
    Write-Host "  Model cache: $ModelCache ($modelCount models found)" -ForegroundColor Green
    Run-Python -ArgList @("generate_missing_figures.py", "--cache", $CACHE, "--model-cache", $ModelCache) `
               -Label "Confusion matrix (warm)"
} else {
    Write-Host "  No model cache — cold run (~43 min for XGBoost best config)" -ForegroundColor Yellow
    Write-Host "  Tip: pass -ModelCache 'results\loso_model_cache' to use cached models" -ForegroundColor DarkGray
    Run-Python -ArgList @("generate_missing_figures.py", "--cache", $CACHE) -Label "Confusion matrix (cold)"
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
