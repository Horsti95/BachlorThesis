# =============================================================================
#  run_pc1_rerun.ps1  —  Full benchmark re-run on PC1
# =============================================================================
#
#  Run from the BachlorThesis repo root, e.g.:
#
#      cd "C:\Users\DerHo\Documents\BachlorThesis"
#      .\run_pc1_rerun.ps1 -DataPath "C:\Users\DerHo\Desktop\Data"
#
#  Flags:
#    -SkipSvmScaling   Skip the ~8-hour SVM scaling benchmark
#    -SkipConfMatrix   Skip confusion matrix (needs 128-subject model cache)
#    -NoPush           Don't git push at the end
#
#  What this script does (in order):
#    1.  git pull
#    2.  Rebuild feature cache for 128 subjects  (~1.5 h if empty)
#    3.  ANOVA vs MI benchmark                   (~5 min,  10 subjects)
#    4.  Feature selection benchmark             (~10 min, 5 subjects)
#    5.  Global vs per-fold benchmark            (~20 min, up to 20 subj.)
#    6.  Generate global-vs-perfold figure
#    7.  Confusion matrix figure                 (~7 s warm / ~43 min cold)
#    8.  Update speedup bar + crossover figures
#    9.  [Optional] SVM scaling benchmark        (~8 h)
#   10.  git commit + push
# =============================================================================

param(
    [Parameter(Mandatory = $true)]
    [string]$DataPath,

    [switch]$SkipSvmScaling,
    [switch]$SkipConfMatrix,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
$CACHE = "results\features_cache_global"

# ── helpers ───────────────────────────────────────────────────────────────────
function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Run-Python([string[]]$args_list, [string]$label, [switch]$Fatal) {
    Write-Host "  > python $($args_list -join ' ')" -ForegroundColor DarkGray
    python @args_list
    if ($LASTEXITCODE -ne 0) {
        if ($Fatal) { Write-Error "$label failed"; exit 1 }
        Write-Warning "$label exited with code $LASTEXITCODE — continuing"
    }
}

# ── pre-flight checks ─────────────────────────────────────────────────────────
if (-not (Test-Path "run_combo_cold_warm_suite.py")) {
    Write-Error "Run this script from the BachlorThesis repo root (where run_combo_cold_warm_suite.py lives)."
    exit 1
}
if (-not (Test-Path $DataPath)) {
    Write-Error "DataPath not found: $DataPath"
    exit 1
}
try { python --version | Out-Null } catch { Write-Error "Python not on PATH"; exit 1 }

# ── Step 1: git pull ──────────────────────────────────────────────────────────
Write-Step "Step 1 / 10  —  git pull"
git pull
if ($LASTEXITCODE -ne 0) { Write-Error "git pull failed"; exit 1 }

# ── Step 2: feature cache ─────────────────────────────────────────────────────
Write-Step "Step 2 / 10  —  Feature cache (128 subjects)"
$cached = (Get-ChildItem "$CACHE\subject_*_full.npz" -ErrorAction SilentlyContinue).Count
Write-Host "  Found $cached / 128 subjects in cache" -ForegroundColor Yellow

if ($cached -lt 128) {
    $missing = 128 - $cached
    Write-Host "  Extracting $missing missing subjects (~1.5 h if starting from zero)..." -ForegroundColor Yellow
    Run-Python @("rebuild_feature_cache.py", "--data-path", $DataPath, "--n-subjects", "128") `
               -label "Feature cache rebuild" -Fatal
} else {
    Write-Host "  Cache complete — skipping extraction" -ForegroundColor Green
}

# ── Step 3: ANOVA vs MI ───────────────────────────────────────────────────────
Write-Step "Step 3 / 10  —  ANOVA vs MI benchmark  (~5 min, 10 subjects)"
Run-Python @("benchmarks_and_tests\benchmark_anova_vs_mi.py") -label "ANOVA vs MI"

# ── Step 4: Feature selection ─────────────────────────────────────────────────
Write-Step "Step 4 / 10  —  Feature selection benchmark  (~10 min)"
Run-Python @("benchmarks_and_tests\benchmark_feature_selection.py") -label "Feature selection"

# ── Step 5: Global vs per-fold ────────────────────────────────────────────────
Write-Step "Step 5 / 10  —  Global vs per-fold benchmark  (~20 min)"
Run-Python @("benchmark_global_vs_perfold.py", "--cache", $CACHE) -label "Global vs per-fold"

# ── Step 6: Global vs per-fold figure ────────────────────────────────────────
Write-Step "Step 6 / 10  —  Generate global-vs-perfold figure"
Run-Python @("generate_globalvsfold_figure.py") -label "Global vs perfold figure"

# ── Step 7: Confusion matrix ──────────────────────────────────────────────────
if (-not $SkipConfMatrix) {
    Write-Step "Step 7 / 10  —  Confusion matrix figure"
    Write-Host "  Warm (~7 s if model cache exists) or cold (~43 min)" -ForegroundColor Yellow
    $modelCacheDir = "results\features_cache_global\..\loso_model_cache"
    if (Test-Path $modelCacheDir) {
        Run-Python @("generate_missing_figures.py", "--cache", $CACHE, "--model-cache", $modelCacheDir) `
                   -label "Confusion matrix"
    } else {
        Run-Python @("generate_missing_figures.py", "--cache", $CACHE) `
                   -label "Confusion matrix"
    }
} else {
    Write-Host "`nStep 7 / 10  —  Confusion matrix skipped (-SkipConfMatrix)" -ForegroundColor DarkGray
}

# ── Step 8: Update speedup bar + crossover figures ────────────────────────────
Write-Step "Step 8 / 10  —  Update fig1_speedup_bar + fig3b_crossover (PC1 data)"
Run-Python @("update_figures_pc1.py") -label "Update figures"

# ── Step 9: SVM scaling (optional) ───────────────────────────────────────────
if (-not $SkipSvmScaling) {
    Write-Step "Step 9 / 10  —  SVM scaling benchmark  (~8 hours)"
    Write-Host "  Results -> results\svm_scaling_<timestamp>\" -ForegroundColor Yellow
    Write-Host "  Press Ctrl+C to skip this step." -ForegroundColor Yellow
    & ".\benchmarks_and_tests\benchmark_svm_scaling.ps1"
    if ($LASTEXITCODE -ne 0) { Write-Warning "SVM scaling exited with code $LASTEXITCODE" }
} else {
    Write-Host "`nStep 9 / 10  —  SVM scaling skipped (-SkipSvmScaling)" -ForegroundColor DarkGray
}

# ── Step 10: git commit + push ────────────────────────────────────────────────
Write-Step "Step 10 / 10  —  Commit results"

# Stage all result and figure files
git add "results\benchmark\" `
        "results\global_vs_perfold_benchmark.csv" `
        "results\global_vs_perfold_benchmark.json" `
        "thesis\figures\" `
        "benchmark_results\" 2>$null

# Also add any new svm_scaling directories
git add "results\svm_scaling_*\" 2>$null

$branch  = git rev-parse --abbrev-ref HEAD
$datestr = Get-Date -Format "yyyyMMdd"
$msg     = "PC1 benchmark re-run $datestr — benchmarks, figures updated"
git commit -m $msg

if (-not $NoPush) {
    Write-Host "  Pushing to $branch ..." -ForegroundColor Cyan
    git push -u origin $branch
    if ($LASTEXITCODE -ne 0) { Write-Warning "Push failed — run: git push -u origin $branch" }
}

Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host "  ALL DONE" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green
