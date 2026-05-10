# =============================================================================
# benchmark_svm_scaling.ps1 - SVM Cache Viability Scaling Analysis
# =============================================================================
# Thesis Question: "Does SVM caching become MORE viable as dataset size grows?"
#
# SVM training scales O(n^2)-O(n^3) with data size, but the cached model
# (support vectors) stays relatively small. This script proves that
# mb_per_s_saved DECREASES as subjects increase --> cache becomes more viable.
#
# Usage:
#   .\benchmarks_and_tests\benchmark_svm_scaling.ps1
#
# Output:
#   results\svm_scaling_<timestamp>\  (one CSV per subject count + summary)
# =============================================================================

param(
    [int[]]$SubjectCounts = @(10, 30, 64, 128),
    [int]$MaxFolds = 3,
    [string]$OutputBase = "results\svm_scaling"
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsDir = "$OutputBase`_$timestamp"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " SVM CACHE VIABILITY SCALING BENCHMARK" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Subject counts : $($SubjectCounts -join ', ')"
Write-Host "  Max folds/model: $MaxFolds"
Write-Host "  Output         : $resultsDir"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null

$summaryRows = @()
$overallStart = Get-Date

foreach ($n in $SubjectCounts) {
    $runDir = "$resultsDir\svm_${n}subj"
    Write-Host ""
    Write-Host ">>> SVM Scaling: $n subjects, $MaxFolds folds <<<" -ForegroundColor Yellow
    Write-Host ("-" * 50)

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    python model_tryouts/benchmark_cache_all_models.py `
        --subjects $n `
        --max-folds $MaxFolds `
        --filter "svm" `
        --output-dir $runDir

    $sw.Stop()
    $elapsed = $sw.Elapsed.TotalSeconds

    Write-Host "  Completed in $([math]::Round($elapsed, 1))s" -ForegroundColor Green

    # Parse the CSV output for summary
    $csvFiles = Get-ChildItem -Path $runDir -Filter "cache_viability_*.csv" | Sort-Object LastWriteTime -Descending
    if ($csvFiles.Count -gt 0) {
        $data = Import-Csv $csvFiles[0].FullName
        foreach ($row in $data) {
            $summaryRows += [PSCustomObject]@{
                Subjects       = $n
                Model          = $row.model_name
                ColdPerFold_s  = [math]::Round([double]$row.cold_per_fold_time_s, 3)
                WarmPerFold_s  = [math]::Round([double]$row.warm_per_fold_time_s, 3)
                Speedup        = [math]::Round([double]$row.speedup_ratio, 1)
                CachePerFold_MB = [math]::Round([double]$row.cache_size_per_fold_mb, 2)
                MB_per_s_saved = [math]::Round([double]$row.mb_per_second_saved, 4)
                Verdict        = $row.cache_verdict
            }
        }
    }
}

$overallElapsed = ((Get-Date) - $overallStart).TotalSeconds

# Print scaling summary table
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " SVM SCALING SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host ("{0,-12} {1,-16} {2,>12} {3,>12} {4,>9} {5,>12} {6,>12} {7,-12}" -f `
    "Subjects", "Model", "Cold/fold", "Warm/fold", "Speedup", "Cache/fold", "MB/s-saved", "Verdict")
Write-Host ("-" * 100)

foreach ($row in $summaryRows) {
    $color = switch ($row.Verdict) {
        "VIABLE"     { "Green" }
        "BORDERLINE" { "Yellow" }
        "TOO_FAST"   { "DarkGray" }
        default      { "Red" }
    }
    Write-Host ("{0,-12} {1,-16} {2,12:N3}s {3,12:N3}s {4,8:N1}x {5,10:N2} MB {6,12:N4} {7,-12}" -f `
        $row.Subjects, $row.Model, $row.ColdPerFold_s, $row.WarmPerFold_s, `
        $row.Speedup, $row.CachePerFold_MB, $row.MB_per_s_saved, $row.Verdict) -ForegroundColor $color
}

Write-Host ("-" * 100)
Write-Host ""

# Key insight: show how MB/s-saved changes with scale for each model
Write-Host "KEY INSIGHT - MB/s-saved trend (lower = more viable):" -ForegroundColor Cyan
$models = $summaryRows | Select-Object -ExpandProperty Model -Unique
foreach ($model in $models) {
    $modelRows = $summaryRows | Where-Object { $_.Model -eq $model }
    $trend = ($modelRows | ForEach-Object { "$($_.Subjects)subj=$($_.MB_per_s_saved)" }) -join " -> "
    Write-Host "  $model : $trend"
}

Write-Host ""
Write-Host "Total benchmark time: $([math]::Round($overallElapsed / 60, 1)) minutes" -ForegroundColor Cyan

# Save summary CSV
$summaryPath = "$resultsDir\svm_scaling_summary.csv"
$summaryRows | Export-Csv -Path $summaryPath -NoTypeInformation
Write-Host "Summary CSV: $summaryPath" -ForegroundColor Green
Write-Host ""
