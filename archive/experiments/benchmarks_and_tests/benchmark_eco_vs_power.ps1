# =============================================================================
# benchmark_eco_vs_power.ps1 - Eco Mode vs Power Mode Caching Strategy
# =============================================================================
# Thesis Question: "Is a lightweight cache config viable, or do you need full
#                   caching to get meaningful speedups?"
#
# Eco Mode:   Few fast models, 3 folds, top_k=30 features  (laptop-friendly)
# Power Mode: All models, 5 folds, all 149 features         (workstation)
#
# This benchmark proves that caching benefits scale with compute cost:
#   - Eco mode: small absolute savings (but still viable for expensive models)
#   - Power mode: large absolute savings (caching is essential)
#
# Usage:
#   .\benchmarks_and_tests\benchmark_eco_vs_power.ps1
#   .\benchmarks_and_tests\benchmark_eco_vs_power.ps1 -Subjects 64
#
# Output:
#   results\eco_vs_power_<timestamp>\  (eco + power CSVs + comparison)
# =============================================================================

param(
    [int]$Subjects = 30,
    [string]$OutputBase = "results\eco_vs_power"
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsDir = "$OutputBase`_$timestamp"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " ECO vs POWER MODE BENCHMARK" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Subjects: $Subjects"
Write-Host "  Output  : $resultsDir"
Write-Host ""
Write-Host "  ECO MODE:   3 folds, top_k=30, fast models only" -ForegroundColor Green
Write-Host "  POWER MODE: 5 folds, all 149 features, all models" -ForegroundColor Red
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null

$overallStart = Get-Date

# -------------------------------------------------------------------------
# ECO MODE: lightweight config — only fast models, fewer features, fewer folds
# -------------------------------------------------------------------------
$ecoDir = "$resultsDir\eco_mode"
Write-Host ">>> ECO MODE: top_k=30, 3 folds, fast models <<<" -ForegroundColor Green
Write-Host ("-" * 50)

$ecoStart = [System.Diagnostics.Stopwatch]::StartNew()

# Eco mode runs each fast model category separately to avoid slow models
$ecoModels = @("random_forest", "extra_trees", "knn_5", "logistic_regression", "ridge_classifier", "naive_bayes", "decision_tree", "lightgbm", "xgboost", "adaboost")

foreach ($model in $ecoModels) {
    python model_tryouts/benchmark_cache_all_models.py `
        --subjects $Subjects `
        --max-folds 3 `
        --top-k 30 `
        --filter $model `
        --output-dir "$ecoDir\$model"
}

$ecoStart.Stop()
$ecoTime = $ecoStart.Elapsed.TotalSeconds
Write-Host "  ECO completed in $([math]::Round($ecoTime / 60, 1)) min" -ForegroundColor Green

# -------------------------------------------------------------------------
# POWER MODE: full config — all models, all features, more folds
# -------------------------------------------------------------------------
$powerDir = "$resultsDir\power_mode"
Write-Host ""
Write-Host ">>> POWER MODE: all 149 features, 5 folds, ALL models <<<" -ForegroundColor Red
Write-Host ("-" * 50)

$powerStart = [System.Diagnostics.Stopwatch]::StartNew()

python model_tryouts/benchmark_cache_all_models.py `
    --subjects $Subjects `
    --max-folds 5 `
    --output-dir $powerDir

$powerStart.Stop()
$powerTime = $powerStart.Elapsed.TotalSeconds
Write-Host "  POWER completed in $([math]::Round($powerTime / 60, 1)) min" -ForegroundColor Red

# -------------------------------------------------------------------------
# COMPARISON
# -------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " ECO vs POWER COMPARISON" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Parse eco results (one CSV per model subfolder)
$ecoRows = @()
foreach ($model in $ecoModels) {
    $csvFiles = Get-ChildItem -Path "$ecoDir\$model" -Filter "cache_viability_*.csv" -Recurse -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending
    if ($csvFiles.Count -gt 0) {
        $data = Import-Csv $csvFiles[0].FullName
        foreach ($row in $data) {
            $speedup = 0.0; $mbps = 0.0
            if ($row.speedup_ratio -match '^[\d.]+$') { $speedup = [math]::Round([double]$row.speedup_ratio, 1) }
            elseif ($row.speedup_ratio -eq 'inf') { $speedup = 99999.0 }
            if ($row.mb_per_second_saved -match '^[\d.]+$') { $mbps = [math]::Round([double]$row.mb_per_second_saved, 4) }
            elseif ($row.mb_per_second_saved -eq 'inf') { $mbps = 99999.0 }

            $ecoRows += [PSCustomObject]@{
                Model           = $row.model_name
                ColdPerFold_s   = [math]::Round([double]$row.cold_per_fold_time_s, 3)
                WarmPerFold_s   = [math]::Round([double]$row.warm_per_fold_time_s, 3)
                Speedup         = $speedup
                CachePerFold_MB = [math]::Round([double]$row.cache_size_per_fold_mb, 2)
                Verdict         = $row.cache_verdict
            }
        }
    }
}

# Parse power results
$powerRows = @()
$powerCsvFiles = Get-ChildItem -Path $powerDir -Filter "cache_viability_*.csv" -Recurse -ErrorAction SilentlyContinue |
                 Sort-Object LastWriteTime -Descending
if ($powerCsvFiles.Count -gt 0) {
    $data = Import-Csv $powerCsvFiles[0].FullName
    foreach ($row in $data) {
        $speedup = 0.0; $mbps = 0.0
        if ($row.speedup_ratio -match '^[\d.]+$') { $speedup = [math]::Round([double]$row.speedup_ratio, 1) }
        elseif ($row.speedup_ratio -eq 'inf') { $speedup = 99999.0 }
        if ($row.mb_per_second_saved -match '^[\d.]+$') { $mbps = [math]::Round([double]$row.mb_per_second_saved, 4) }
        elseif ($row.mb_per_second_saved -eq 'inf') { $mbps = 99999.0 }

        $powerRows += [PSCustomObject]@{
            Model           = $row.model_name
            ColdPerFold_s   = [math]::Round([double]$row.cold_per_fold_time_s, 3)
            WarmPerFold_s   = [math]::Round([double]$row.warm_per_fold_time_s, 3)
            Speedup         = $speedup
            CachePerFold_MB = [math]::Round([double]$row.cache_size_per_fold_mb, 2)
            Verdict         = $row.cache_verdict
        }
    }
}

# Side-by-side table
$header = "{0,-22} {1,14} {2,10} {3,10} {4,14} {5,10} {6,10}" -f `
    "Model", "ECO cold/fold", "ECO spd", "ECO MB", "PWR cold/fold", "PWR spd", "PWR MB"
Write-Host $header
Write-Host ("-" * 92)

foreach ($pRow in ($powerRows | Sort-Object { $_.ColdPerFold_s } -Descending)) {
    $eRow = $ecoRows | Where-Object { $_.Model -eq $pRow.Model }
    $ecoCold = if ($eRow) { "{0,12:N3}s" -f $eRow.ColdPerFold_s } else { "    (skipped)" }
    $ecoSpd  = if ($eRow) { "{0,8:N1}x" -f $eRow.Speedup } else { "       -" }
    $ecoMB   = if ($eRow) { "{0,8:N2}" -f $eRow.CachePerFold_MB } else { "       -" }

    $line = "{0,-22} {1,14} {2,10} {3,10} {4,12:N3}s {5,8:N1}x {6,8:N2}" -f `
        $pRow.Model, $ecoCold, $ecoSpd, $ecoMB, $pRow.ColdPerFold_s, $pRow.Speedup, $pRow.CachePerFold_MB

    # Color by power mode verdict
    $color = switch ($pRow.Verdict) {
        "VIABLE"     { "Green" }
        "BORDERLINE" { "Yellow" }
        default      { "White" }
    }
    Write-Host $line -ForegroundColor $color
}

Write-Host ("-" * 92)
Write-Host ""
Write-Host "TIMING:" -ForegroundColor Cyan
Write-Host "  Eco mode total:   $([math]::Round($ecoTime / 60, 1)) min"
Write-Host "  Power mode total: $([math]::Round($powerTime / 60, 1)) min"
Write-Host "  Ratio:            $([math]::Round($powerTime / [math]::Max($ecoTime, 1), 1))x slower in power mode"
Write-Host ""
Write-Host "THESIS INSIGHT:" -ForegroundColor Yellow
Write-Host "  Eco mode (fewer features + fewer folds) = faster benchmarks, smaller cache." -ForegroundColor Yellow
Write-Host "  Power mode shows which models TRULY need caching at full scale." -ForegroundColor Yellow
Write-Host "  Models that are 'VIABLE' in power but 'TOO_FAST' in eco = scaling-dependent." -ForegroundColor Yellow
Write-Host ""

$overallElapsed = ((Get-Date) - $overallStart).TotalSeconds
Write-Host "Total benchmark time: $([math]::Round($overallElapsed / 60, 1)) minutes" -ForegroundColor Cyan

# Save comparison
$compRows = @()
foreach ($pRow in $powerRows) {
    $eRow = $ecoRows | Where-Object { $_.Model -eq $pRow.Model }
    $compRows += [PSCustomObject]@{
        Model              = $pRow.Model
        Eco_ColdPerFold_s  = if ($eRow) { $eRow.ColdPerFold_s } else { "N/A" }
        Eco_Speedup        = if ($eRow) { $eRow.Speedup } else { "N/A" }
        Eco_CacheMB        = if ($eRow) { $eRow.CachePerFold_MB } else { "N/A" }
        Eco_Verdict        = if ($eRow) { $eRow.Verdict } else { "SKIPPED" }
        Power_ColdPerFold_s = $pRow.ColdPerFold_s
        Power_Speedup       = $pRow.Speedup
        Power_CacheMB       = $pRow.CachePerFold_MB
        Power_Verdict       = $pRow.Verdict
    }
}

$compPath = "$resultsDir\eco_vs_power_comparison.csv"
$compRows | Export-Csv -Path $compPath -NoTypeInformation
Write-Host "Comparison CSV: $compPath" -ForegroundColor Green
Write-Host ""
