# =============================================================================
# benchmark_xgb_rf_scaling.ps1 - XGBoost + Random Forest Subject-Axis Scaling
# =============================================================================
# Thesis Question: "Does XGB/RF cache speedup grow with subject count?"
#
# RQ3 (Scalability) was previously answered using SVM as the theoretically clean
# test case. This benchmark extends the answer to the primary classifiers
# (XGBoost and Random Forest) along the SUBJECT axis (not the feature axis).
#
# At each subject count (10, 30, 64, 128) the script runs:
#   - 1 cold LOSO pass (no cache, must train every fold)
#   - 1 warm LOSO pass (cache hit on every fold)
# for both xgboost and random_forest, and records per-fold times, cache size,
# and speedup ratio. A summary CSV is written for the plot script.
#
# Usage:
#   .\benchmarks_and_tests\benchmark_xgb_rf_scaling.ps1
#   .\benchmarks_and_tests\benchmark_xgb_rf_scaling.ps1 -SubjectCounts @(10,30,64,128) -MaxFolds 5
#   .\benchmarks_and_tests\benchmark_xgb_rf_scaling.ps1 -Tag "5090"     # tag the output dir
#
# Output:
#   results\xgb_rf_scaling_<tag>_<timestamp>\
#       xgboost_<n>subj\cache_viability_*.csv
#       random_forest_<n>subj\cache_viability_*.csv
#       xgb_rf_scaling_summary.csv     <-- consumed by plot_xgb_rf_scaling.py
# =============================================================================

param(
    [int[]]$SubjectCounts = @(10, 30, 64, 128),
    [int]$MaxFolds = 5,
    [string]$Tag = "",
    [string]$OutputBase = "results\xgb_rf_scaling"
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$tagPart = if ($Tag) { "_$Tag" } else { "" }
$resultsDir = "$OutputBase$tagPart`_$timestamp"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " XGB + RF SUBJECT-AXIS SCALING BENCHMARK" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Subject counts : $($SubjectCounts -join ', ')"
Write-Host "  Max folds/cell : $MaxFolds"
Write-Host "  Models         : xgboost, random_forest"
Write-Host "  Tag            : $(if ($Tag) { $Tag } else { '<none>' })"
Write-Host "  Output         : $resultsDir"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null

$summaryRows = @()
$overallStart = Get-Date
$models = @("xgboost", "random_forest")

foreach ($n in $SubjectCounts) {
    foreach ($model in $models) {
        $runDir = "$resultsDir\${model}_${n}subj"
        Write-Host ""
        Write-Host ">>> $model | $n subjects | $MaxFolds folds <<<" -ForegroundColor Yellow
        Write-Host ("-" * 60)

        $sw = [System.Diagnostics.Stopwatch]::StartNew()

        python model_tryouts/benchmark_cache_all_models.py `
            --subjects $n `
            --max-folds $MaxFolds `
            --filter $model `
            --output-dir $runDir

        $sw.Stop()
        $elapsed = $sw.Elapsed.TotalSeconds
        Write-Host "  Completed in $([math]::Round($elapsed, 1))s" -ForegroundColor Green

        # Parse the CSV output
        $csvFiles = Get-ChildItem -Path $runDir -Filter "cache_viability_*.csv" -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending
        if ($csvFiles.Count -gt 0) {
            $data = Import-Csv $csvFiles[0].FullName
            foreach ($row in $data) {
                # Filter out anything that snuck in (defensive: only keep target model)
                if ($row.model_name -ne $model) { continue }

                $speedup = 0.0; $mbps = 0.0
                if ($row.speedup_ratio -match '^[\d.]+$') { $speedup = [math]::Round([double]$row.speedup_ratio, 2) }
                elseif ($row.speedup_ratio -eq 'inf') { $speedup = 99999.0 }
                if ($row.mb_per_second_saved -match '^[\d.]+$') { $mbps = [math]::Round([double]$row.mb_per_second_saved, 4) }
                elseif ($row.mb_per_second_saved -eq 'inf') { $mbps = 99999.0 }

                $summaryRows += [PSCustomObject]@{
                    Subjects        = $n
                    Model           = $row.model_name
                    NFoldsRun       = [int]$row.n_folds_run
                    ColdTotal_s     = [math]::Round([double]$row.cold_total_time_s, 2)
                    WarmTotal_s     = [math]::Round([double]$row.warm_total_time_s, 2)
                    ColdPerFold_s   = [math]::Round([double]$row.cold_per_fold_time_s, 3)
                    WarmPerFold_s   = [math]::Round([double]$row.warm_per_fold_time_s, 3)
                    Speedup         = $speedup
                    CacheTotal_MB   = [math]::Round([double]$row.cache_size_mb, 2)
                    CachePerFold_MB = [math]::Round([double]$row.cache_size_per_fold_mb, 2)
                    MB_per_s_saved  = $mbps
                    AccuracyMatch   = $row.accuracy_match
                    Verdict         = $row.cache_verdict
                    Tag             = $Tag
                    Timestamp       = $timestamp
                }
            }
        } else {
            Write-Host "  WARNING: no CSV output found in $runDir" -ForegroundColor Red
        }
    }
}

$overallElapsed = ((Get-Date) - $overallStart).TotalSeconds

# -----------------------------------------------------------------------------
# Print summary table grouped by model
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " XGB + RF SCALING SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host ("{0,-16} {1,-10} {2,>12} {3,>12} {4,>9} {5,>14} {6,-12}" -f `
    "Model", "Subjects", "Cold/fold", "Warm/fold", "Speedup", "Cache/fold", "Verdict")
Write-Host ("-" * 90)

foreach ($row in $summaryRows | Sort-Object Model, Subjects) {
    $color = switch ($row.Verdict) {
        "VIABLE"     { "Green" }
        "BORDERLINE" { "Yellow" }
        "TOO_FAST"   { "DarkGray" }
        default      { "Red" }
    }
    Write-Host ("{0,-16} {1,10} {2,12:N3}s {3,12:N3}s {4,8:N1}x {5,11:N2} MB {6,-12}" -f `
        $row.Model, $row.Subjects, $row.ColdPerFold_s, $row.WarmPerFold_s, `
        $row.Speedup, $row.CachePerFold_MB, $row.Verdict) -ForegroundColor $color
}

Write-Host ("-" * 90)
Write-Host ""

# Per-model speedup trend (the headline RQ3 result)
Write-Host "SPEEDUP TREND vs SUBJECT COUNT:" -ForegroundColor Cyan
foreach ($model in $models) {
    $rows = $summaryRows | Where-Object { $_.Model -eq $model } | Sort-Object Subjects
    if ($rows.Count -gt 0) {
        $trend = ($rows | ForEach-Object { "$($_.Subjects)subj=$($_.Speedup)x" }) -join "  ->  "
        Write-Host "  $model : $trend"
    }
}

Write-Host ""
Write-Host "Total benchmark time: $([math]::Round($overallElapsed / 60, 1)) minutes" -ForegroundColor Cyan

# Save summary CSV (consumed by the plot script)
$summaryPath = "$resultsDir\xgb_rf_scaling_summary.csv"
$summaryRows | Export-Csv -Path $summaryPath -NoTypeInformation
Write-Host ""
Write-Host "Summary CSV: $summaryPath" -ForegroundColor Green
Write-Host "Plot it with: python benchmarks_and_tests\plot_xgb_rf_scaling.py --summary `"$summaryPath`"" -ForegroundColor Green
Write-Host ""
