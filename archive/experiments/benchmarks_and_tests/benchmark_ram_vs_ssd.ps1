# =============================================================================
# benchmark_ram_vs_ssd.ps1 - RAM Disk vs SSD Cache Loading Comparison
# =============================================================================
# Thesis Question: "Does cache backend (RAM vs SSD) significantly affect
#                   warm-load performance for LOSO model caching?"
#
# On the 5090 machine with 64GB RAM, we can create a RAM disk and compare
# warm-load times against NVMe SSD. If the difference is negligible,
# SSD caching is sufficient (simpler, persistent across reboots).
#
# Prerequisites (run ONCE as admin before this script):
#   # Option A: ImDisk (free, lightweight)
#   imdisk -a -s 16G -m R: -p "/fs:ntfs /q /y"
#
#   # Option B: PowerShell with Storage module (Windows Server / Pro)
#   New-VHD -Path C:\ramdisk.vhdx -SizeBytes 16GB -Dynamic
#   Mount-VHD -Path C:\ramdisk.vhdx
#   # Then format and assign R:
#
# Usage:
#   .\benchmarks_and_tests\benchmark_ram_vs_ssd.ps1
#   .\benchmarks_and_tests\benchmark_ram_vs_ssd.ps1 -RamDiskPath "R:\"
#   .\benchmarks_and_tests\benchmark_ram_vs_ssd.ps1 -Subjects 64 -MaxFolds 3
#
# Output:
#   results\ram_vs_ssd_<timestamp>\  (comparison CSV + console table)
# =============================================================================

param(
    [string]$RamDiskPath = "R:\cache_bench",
    [string]$SsdPath     = "results\ram_vs_ssd_ssd_cache",
    [int]$Subjects       = 30,
    [int]$MaxFolds       = 3,
    [string]$ModelFilter  = "",        # empty = all models
    [string]$OutputBase  = "results\ram_vs_ssd"
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$resultsDir = "$OutputBase`_$timestamp"

# Validate RAM disk exists
if (-not (Test-Path (Split-Path $RamDiskPath -Qualifier))) {
    Write-Host "ERROR: RAM disk drive not found at $(Split-Path $RamDiskPath -Qualifier)" -ForegroundColor Red
    Write-Host ""
    Write-Host "To create a RAM disk (run as Administrator):" -ForegroundColor Yellow
    Write-Host "  # Using ImDisk (install from https://sourceforge.net/projects/imdisk-toolkit/):"
    Write-Host '  imdisk -a -s 16G -m R: -p "/fs:ntfs /q /y"'
    Write-Host ""
    Write-Host "  # Or pass a different path:" -ForegroundColor Yellow
    Write-Host '  .\benchmark_ram_vs_ssd.ps1 -RamDiskPath "D:\ramdisk\cache"'
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " RAM vs SSD CACHE LOADING BENCHMARK" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  RAM disk path : $RamDiskPath"
Write-Host "  SSD path      : $SsdPath"
Write-Host "  Subjects      : $Subjects"
Write-Host "  Max folds     : $MaxFolds"
Write-Host "  Model filter  : $(if ($ModelFilter) { $ModelFilter } else { 'ALL' })"
Write-Host "  Output        : $resultsDir"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null

$filterArg = if ($ModelFilter) { "--filter", $ModelFilter } else { @() }

# ── Run 1: SSD backend ──
Write-Host ">>> PHASE 1: SSD Cache Backend <<<" -ForegroundColor Yellow
Write-Host ("-" * 50)

$ssdOutDir = "$resultsDir\ssd"
$swSsd = [System.Diagnostics.Stopwatch]::StartNew()

python model_tryouts/benchmark_cache_all_models.py `
    --subjects $Subjects `
    --max-folds $MaxFolds `
    --output-dir $ssdOutDir `
    @filterArg

$swSsd.Stop()
Write-Host "  SSD run: $([math]::Round($swSsd.Elapsed.TotalSeconds, 1))s" -ForegroundColor Green

# ── Run 2: RAM disk backend ──
Write-Host ""
Write-Host ">>> PHASE 2: RAM Disk Cache Backend <<<" -ForegroundColor Yellow
Write-Host ("-" * 50)

$ramOutDir = "$resultsDir\ram"
# Point the benchmark's cache temp dir to RAM disk
$env:BENCH_CACHE_OVERRIDE = $RamDiskPath
$swRam = [System.Diagnostics.Stopwatch]::StartNew()

python model_tryouts/benchmark_cache_all_models.py `
    --subjects $Subjects `
    --max-folds $MaxFolds `
    --output-dir $ramOutDir `
    @filterArg

$swRam.Stop()
$env:BENCH_CACHE_OVERRIDE = $null
Write-Host "  RAM run: $([math]::Round($swRam.Elapsed.TotalSeconds, 1))s" -ForegroundColor Green

# ── Compare Results ──
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " RAM vs SSD COMPARISON" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$ssdCsv = Get-ChildItem -Path $ssdOutDir -Filter "cache_viability_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$ramCsv = Get-ChildItem -Path $ramOutDir -Filter "cache_viability_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($ssdCsv -and $ramCsv) {
    $ssdData = Import-Csv $ssdCsv.FullName
    $ramData = Import-Csv $ramCsv.FullName

    $comparison = @()

    Write-Host ""
    Write-Host ("{0,-22} {1,>12} {2,>12} {3,>12} {4,>14}" -f `
        "Model", "SSD warm/f", "RAM warm/f", "RAM speedup", "Significant?")
    Write-Host ("-" * 80)

    foreach ($ssdRow in $ssdData) {
        $ramRow = $ramData | Where-Object { $_.model_name -eq $ssdRow.model_name }
        if (-not $ramRow) { continue }

        $ssdWarm = [double]$ssdRow.warm_per_fold_time_s
        $ramWarm = [double]$ramRow.warm_per_fold_time_s

        $ramVsSsd = if ($ramWarm -gt 0) { $ssdWarm / $ramWarm } else { 0 }
        $significant = if ($ramVsSsd -gt 2.0) { "YES - RAM wins" }
                       elseif ($ramVsSsd -gt 1.2) { "MARGINAL" }
                       else { "NO - SSD is fine" }

        $color = switch -Wildcard ($significant) {
            "YES*"  { "Green" }
            "MARG*" { "Yellow" }
            default { "White" }
        }

        Write-Host ("{0,-22} {1,12:N3}s {2,12:N3}s {3,11:N2}x {4,-14}" -f `
            $ssdRow.model_name, $ssdWarm, $ramWarm, $ramVsSsd, $significant) -ForegroundColor $color

        $comparison += [PSCustomObject]@{
            Model           = $ssdRow.model_name
            Category        = $ssdRow.category
            SSD_warm_s      = [math]::Round($ssdWarm, 3)
            RAM_warm_s      = [math]::Round($ramWarm, 3)
            RAM_vs_SSD_x    = [math]::Round($ramVsSsd, 2)
            Significant     = $significant
            SSD_cold_s      = [math]::Round([double]$ssdRow.cold_per_fold_time_s, 3)
            RAM_cold_s      = [math]::Round([double]$ramRow.cold_per_fold_time_s, 3)
            SSD_cache_MB    = [math]::Round([double]$ssdRow.cache_size_per_fold_mb, 2)
            SSD_verdict     = $ssdRow.cache_verdict
            RAM_verdict     = $ramRow.cache_verdict
        }
    }

    Write-Host ("-" * 80)

    # Thesis conclusion
    $avgSpeedup = ($comparison | Measure-Object -Property RAM_vs_SSD_x -Average).Average
    Write-Host ""
    if ($avgSpeedup -lt 1.2) {
        Write-Host "CONCLUSION: RAM disk provides negligible benefit ($([math]::Round($avgSpeedup, 2))x avg)." -ForegroundColor Yellow
        Write-Host "  -> SSD caching is sufficient. RAM disk adds complexity for no gain." -ForegroundColor Yellow
        Write-Host "  -> Thesis note: 'NVMe SSD loading is fast enough; RAM disk unnecessary.'" -ForegroundColor Yellow
    }
    elseif ($avgSpeedup -lt 2.0) {
        Write-Host "CONCLUSION: RAM disk shows marginal improvement ($([math]::Round($avgSpeedup, 2))x avg)." -ForegroundColor Yellow
        Write-Host "  -> Worth mentioning in thesis but not a strong recommendation." -ForegroundColor Yellow
    }
    else {
        Write-Host "CONCLUSION: RAM disk significantly faster ($([math]::Round($avgSpeedup, 2))x avg)." -ForegroundColor Green
        Write-Host "  -> For time-critical pipelines, RAM caching is worth the complexity." -ForegroundColor Green
    }

    # Save comparison CSV
    $compPath = "$resultsDir\ram_vs_ssd_comparison.csv"
    $comparison | Export-Csv -Path $compPath -NoTypeInformation
    Write-Host ""
    Write-Host "Comparison CSV: $compPath" -ForegroundColor Green
}
else {
    Write-Host "ERROR: Could not find result CSVs to compare" -ForegroundColor Red
}

Write-Host ""
Write-Host "Total benchmark time: $([math]::Round(($swSsd.Elapsed.TotalSeconds + $swRam.Elapsed.TotalSeconds) / 60, 1)) minutes" -ForegroundColor Cyan
Write-Host ""
