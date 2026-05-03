# =============================================================================
# benchmark_data_scaling.ps1 - How does training time scale with more data?
# =============================================================================
# Thesis Question: "Which models benefit MOST from caching as data grows?"
#
# Runs ALL 15 models at 3, 5, 30, 128 subjects to show:
#   - Which model families have O(n^2+) scaling (huge cache benefit)
#   - Which stay O(n) (cache benefit is marginal)
#
# Usage:
#   .\benchmarks_and_tests\benchmark_data_scaling.ps1
#   .\benchmarks_and_tests\benchmark_data_scaling.ps1 -SubjectCounts @(5,30,64)
#
# Output:
#   results\data_scaling_<timestamp>\  (one CSV per subject count + summary)
# =============================================================================

param(
    [int[]]$SubjectCounts = @(3, 5, 30, 128),
    [int]$MaxFolds = 3,
    [string]$OutputBase = "results\data_scaling"
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsDir = "$OutputBase`_$timestamp"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " DATA SCALING BENCHMARK - All Models" -ForegroundColor Cyan
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
    $runDir = "$resultsDir\all_${n}subj"
    Write-Host ""
    Write-Host ">>> Data Scaling: $n subjects, $MaxFolds folds, ALL models <<<" -ForegroundColor Yellow
    Write-Host ("-" * 50)

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    python model_tryouts/benchmark_cache_all_models.py `
        --subjects $n `
        --max-folds $MaxFolds `
        --output-dir $runDir

    $sw.Stop()
    $elapsed = $sw.Elapsed.TotalSeconds

    Write-Host "  Completed in $([math]::Round($elapsed / 60, 1)) min" -ForegroundColor Green

    # Parse CSV for summary
    $csvFiles = Get-ChildItem -Path $runDir -Filter "cache_viability_*.csv" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending
    if ($csvFiles.Count -gt 0) {
        $data = Import-Csv $csvFiles[0].FullName
        foreach ($row in $data) {
            # Handle "inf" / "nan" values from Python CSV output
            $speedup = 0.0; $mbps = 0.0
            if ($row.speedup_ratio -match '^[\d.]+$') { $speedup = [math]::Round([double]$row.speedup_ratio, 1) }
            elseif ($row.speedup_ratio -eq 'inf') { $speedup = 99999.0 }
            if ($row.mb_per_second_saved -match '^[\d.]+$') { $mbps = [math]::Round([double]$row.mb_per_second_saved, 4) }
            elseif ($row.mb_per_second_saved -eq 'inf') { $mbps = 99999.0 }

            $summaryRows += [PSCustomObject]@{
                Subjects        = $n
                Model           = $row.model_name
                Category        = $row.category
                ColdPerFold_s   = [math]::Round([double]$row.cold_per_fold_time_s, 3)
                WarmPerFold_s   = [math]::Round([double]$row.warm_per_fold_time_s, 3)
                Speedup         = $speedup
                CachePerFold_MB = [math]::Round([double]$row.cache_size_per_fold_mb, 2)
                MB_per_s_saved  = $mbps
                Verdict         = $row.cache_verdict
            }
        }
    }
}

$overallElapsed = ((Get-Date) - $overallStart).TotalSeconds

# Print scaling summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " DATA SCALING SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Group by model, show how cold_time scales
$models = $summaryRows | Select-Object -ExpandProperty Model -Unique
# Header: model name + one column per subject count + scale factor
$header = "{0,-22}" -f "Model"
foreach ($sc in $SubjectCounts) { $header += " {0,12}" -f "${sc}subj" }
$header += " {0,14}" -f "Scale factor"
Write-Host $header
Write-Host ("-" * (22 + 13 * $SubjectCounts.Count + 14))

foreach ($model in $models) {
    $modelRows = $summaryRows | Where-Object { $_.Model -eq $model } | Sort-Object Subjects
    $times = $modelRows | ForEach-Object { $_.ColdPerFold_s }
    $line = "{0,-22}" -f $model
    foreach ($t in $times) {
        $line += " {0,10:N3}s" -f $t
    }
    # Scale factor: last / first
    if ($times.Count -ge 2 -and $times[0] -gt 0) {
        $scale = $times[-1] / $times[0]
        $line += " {0,11:N1}x" -f $scale
    }
    # Color: high scale factor = caching more important
    $color = if ($scale -gt 10) { "Red" } elseif ($scale -gt 3) { "Yellow" } else { "Green" }
    Write-Host $line -ForegroundColor $color
}

Write-Host ("-" * 90)
Write-Host ""
Write-Host "Color guide:" -ForegroundColor White
Write-Host "  RED    = Scale >10x  -> Caching is ESSENTIAL for this model" -ForegroundColor Red
Write-Host "  YELLOW = Scale 3-10x -> Caching provides significant benefit" -ForegroundColor Yellow
Write-Host "  GREEN  = Scale <3x   -> Model scales well, caching is optional" -ForegroundColor Green
Write-Host ""
Write-Host "Total benchmark time: $([math]::Round($overallElapsed / 60, 1)) minutes" -ForegroundColor Cyan

# Save summary CSV
$summaryPath = "$resultsDir\data_scaling_summary.csv"
$summaryRows | Export-Csv -Path $summaryPath -NoTypeInformation
Write-Host "Summary CSV: $summaryPath" -ForegroundColor Green
Write-Host ""
