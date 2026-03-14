# =============================================================================
# benchmark_feature_size.ps1 - Feature Count Impact on Cache Viability
# =============================================================================
# Thesis Question: "How does feature count affect training time and cache size?"
#
# Tests all models with different top_k feature selections.
# Fewer features = smaller models = faster training = smaller cache.
# This also covers correlation threshold effects (same mechanism).
#
# Usage:
#   .\benchmarks_and_tests\benchmark_feature_size.ps1
#
# Output:
#   results\feature_size_<timestamp>\  (one CSV per top_k + summary)
# =============================================================================

param(
    [int[]]$TopKValues = @(10, 30, 50, 149),
    [int]$Subjects = 30,
    [int]$MaxFolds = 3,
    [string]$OutputBase = "results\feature_size"
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsDir = "$OutputBase`_$timestamp"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " FEATURE SIZE IMPACT BENCHMARK" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Top-K values   : $($TopKValues -join ', ')"
Write-Host "  Subjects       : $Subjects"
Write-Host "  Max folds/model: $MaxFolds"
Write-Host "  Output         : $resultsDir"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null

$summaryRows = @()
$overallStart = Get-Date

foreach ($k in $TopKValues) {
    $runDir = "$resultsDir\topk_${k}"
    Write-Host ""
    Write-Host ">>> Feature Size: top_k=$k, $Subjects subjects, $MaxFolds folds <<<" -ForegroundColor Yellow
    Write-Host ("-" * 50)

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    # 149 = all features (no selection)
    if ($k -ge 149) {
        python model_tryouts/benchmark_cache_all_models.py `
            --subjects $Subjects `
            --max-folds $MaxFolds `
            --output-dir $runDir
    } else {
        python model_tryouts/benchmark_cache_all_models.py `
            --subjects $Subjects `
            --max-folds $MaxFolds `
            --top-k $k `
            --output-dir $runDir
    }

    $sw.Stop()
    Write-Host "  Completed in $([math]::Round($sw.Elapsed.TotalSeconds / 60, 1)) min" -ForegroundColor Green

    # Parse CSV
    $csvFiles = Get-ChildItem -Path $runDir -Filter "cache_viability_*.csv" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending
    if ($csvFiles.Count -gt 0) {
        $data = Import-Csv $csvFiles[0].FullName
        foreach ($row in $data) {
            $summaryRows += [PSCustomObject]@{
                TopK            = $k
                Model           = $row.model_name
                Category        = $row.category
                ColdPerFold_s   = [math]::Round([double]$row.cold_per_fold_time_s, 3)
                WarmPerFold_s   = [math]::Round([double]$row.warm_per_fold_time_s, 3)
                Speedup         = [math]::Round([double]$row.speedup_ratio, 1)
                CachePerFold_MB = [math]::Round([double]$row.cache_size_per_fold_mb, 2)
                MB_per_s_saved  = [math]::Round([double]$row.mb_per_second_saved, 4)
                Verdict         = $row.cache_verdict
            }
        }
    }
}

$overallElapsed = ((Get-Date) - $overallStart).TotalSeconds

# Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " FEATURE SIZE IMPACT SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$models = $summaryRows | Select-Object -ExpandProperty Model -Unique
Write-Host ("{0,-22} {1,>12} {2,>12} {3,>12} {4,>12}" -f `
    "Model", ($TopKValues | ForEach-Object { "k=$_" }))
Write-Host ("-" * 80)

foreach ($model in $models) {
    $modelRows = $summaryRows | Where-Object { $_.Model -eq $model } | Sort-Object TopK
    $line = "{0,-22}" -f $model
    foreach ($row in $modelRows) {
        $line += " {0,12:N3}s" -f $row.ColdPerFold_s
    }
    Write-Host $line
}

Write-Host ("-" * 80)
Write-Host ""
Write-Host "THESIS NOTE: If times barely change with fewer features," -ForegroundColor Yellow
Write-Host "  then feature count has minimal impact on cache viability." -ForegroundColor Yellow
Write-Host "  If times drop significantly -> fewer features = faster training = less cache benefit." -ForegroundColor Yellow
Write-Host ""
Write-Host "Total time: $([math]::Round($overallElapsed / 60, 1)) minutes" -ForegroundColor Cyan

$summaryPath = "$resultsDir\feature_size_summary.csv"
$summaryRows | Export-Csv -Path $summaryPath -NoTypeInformation
Write-Host "Summary CSV: $summaryPath" -ForegroundColor Green
Write-Host ""
