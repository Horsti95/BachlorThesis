<#
.SYNOPSIS
    Smoke-test the codebase: compile, --help, 3-subject pipeline, thesis tests.
.DESCRIPTION
    Continues past failures so one broken step never stops the rest. Writes
    everything (stdout + stderr) to a single timestamped log file in the repo
    root: verify_<yyyyMMdd_HHmmss>.txt

    Sequence:
      1. Compile every *.py at repo root (catches syntax errors, no deps).
      2. --help on the three CLI entry points (argparse builds OK).
      3. Two no-data thesis tests (test_imports.py, test_loso_cache_fixes.py).
      4. If $DataPath exists: 3-subject pipeline run, then the two
         data-dependent thesis tests (test_cache_comprehensive.py uses the
         feature cache; test_ram_cache_comparison.py uses the LOSO model
         cache produced by the previous step).

    The data-dependent steps are skipped automatically (and recorded as SKIP
    in the summary) if the BOAS dataset is not at $DataPath.
.PARAMETER DataPath
    Path to the BOAS dataset root. Default: C:\Users\DerHo\Desktop\Data
.PARAMETER Python
    Python executable to use. Default: python
.EXAMPLE
    .\verify_pipeline.ps1
.EXAMPLE
    .\verify_pipeline.ps1 -DataPath "D:\BOAS"
.EXAMPLE
    .\verify_pipeline.ps1 -Python "py -3.12"
#>
param(
    [string]$DataPath = "C:\Users\DerHo\Desktop\Data",
    [string]$Python   = "python"
)

$ErrorActionPreference = 'Continue'

$repo      = (Get-Location).Path
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile   = Join-Path $repo "verify_$timestamp.txt"
$results   = @()

# Make repo root importable so archive/tests/*.py find core modules.
$env:PYTHONPATH = $repo

# Force UTF-8 everywhere: switch the console code page to UTF-8 (65001) and
# tell Python to use UTF-8 for stdin/stdout/stderr. Without this, the Unicode
# chars printed by the pipeline (-> arrows, check marks, etc.) crash on the
# default cp1252 PowerShell console.
try { chcp 65001 | Out-Null } catch { }
$env:PYTHONUTF8       = "1"
$env:PYTHONIOENCODING = "utf-8"
# Ensure PowerShell itself decodes external-command output as UTF-8.
# (Windows PowerShell 5.x defaults [Console]::OutputEncoding to cp1252 even
# after chcp 65001, which corrupts Python's UTF-8 bytes before they reach the
# pipeline.)
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    [Console]::InputEncoding  = [System.Text.Encoding]::UTF8
} catch { }

# Pre-touch the log file with no BOM so PowerShell 5.x doesn't add Default
# encoding bytes to it later.
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
[System.IO.File]::WriteAllText($logFile, '', $utf8NoBom)

function Tee-Utf8 {
    # Stream-style helper that mimics `Tee-Object -Append -Encoding UTF8`
    # but works on Windows PowerShell 5.1 (where Tee-Object has no -Encoding).
    # Writes each pipeline item to the host AND appends it to $Path as UTF-8.
    param(
        [Parameter(Mandatory=$true, Position=0)]
        [string]$Path,
        [Parameter(ValueFromPipeline=$true)]
        $InputObject
    )
    begin {
        $enc = New-Object System.Text.UTF8Encoding $false
        $nl  = [System.Environment]::NewLine
    }
    process {
        $line = if ($null -eq $InputObject) { '' } else { "$InputObject" }
        [Console]::Out.WriteLine($line)
        [System.IO.File]::AppendAllText($Path, $line + $nl, $enc)
    }
}

function Log {
    param([string]$Text)
    $Text | Tee-Utf8 $logFile
}

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Cmd
    )
    $bar = '=' * 72
    Log ""
    Log $bar
    Log "STEP: $Name"
    Log "TIME: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Log $bar

    $global:LASTEXITCODE = 0
    try {
        & $Cmd 2>&1 | Tee-Utf8 $logFile
    } catch {
        Log "EXCEPTION: $_"
        if ($LASTEXITCODE -eq 0) { $global:LASTEXITCODE = 1 }
    }
    $exit = $LASTEXITCODE
    if ($null -eq $exit) { $exit = 0 }

    $status = if ($exit -eq 0) { 'PASS' } else { "FAIL($exit)" }
    Log "[$status] $Name"
    return [pscustomobject]@{ Name = $Name; Exit = $exit; Status = $status }
}

# --- header ---------------------------------------------------------------
Log "Verification log"
Log "Started:  $(Get-Date)"
Log "Repo:     $repo"
Log "DataPath: $DataPath"
Log "Python:   $(& $Python --version 2>&1)"
Log "PYTHONPATH=$env:PYTHONPATH"

# --- 1. compile every root .py --------------------------------------------
$results += Run-Step "Compile every root *.py" {
    $files = Get-ChildItem -File -Filter '*.py' | ForEach-Object FullName
    & $Python -m py_compile @files
}

# --- 2. --help on entry points --------------------------------------------
$results += Run-Step "run_experiment.py --help"    { & $Python run_experiment.py --help }
$results += Run-Step "run_full_pipeline.py --help" { & $Python run_full_pipeline.py --help }
$results += Run-Step "run_training.py --help"      { & $Python run_training.py --help }

# --- 3. data-free thesis tests --------------------------------------------
$results += Run-Step "test_imports.py" {
    & $Python (Join-Path $repo "archive\tests\test_imports.py")
}
$results += Run-Step "test_loso_cache_fixes.py" {
    & $Python (Join-Path $repo "archive\tests\test_loso_cache_fixes.py")
}

# --- 4. data-dependent tests (skip if BOAS not present) -------------------
if (Test-Path $DataPath) {
    $results += Run-Step "Pipeline 3-subject (--quick-test)" {
        & $Python run_experiment.py --quick-test --data-path $DataPath
    }
    $results += Run-Step "test_cache_comprehensive.py" {
        & $Python (Join-Path $repo "archive\tests\test_cache_comprehensive.py")
    }
    $results += Run-Step "test_ram_cache_comparison.py" {
        & $Python (Join-Path $repo "archive\tests\test_ram_cache_comparison.py")
    }
} else {
    Log ""
    Log "[SKIP] BOAS data not found at: $DataPath"
    Log "       Skipping pipeline + data-dependent tests."
    $results += [pscustomobject]@{ Name = "Pipeline 3-subject (--quick-test)"; Exit = -1; Status = 'SKIP' }
    $results += [pscustomobject]@{ Name = "test_cache_comprehensive.py";       Exit = -1; Status = 'SKIP' }
    $results += [pscustomobject]@{ Name = "test_ram_cache_comparison.py";      Exit = -1; Status = 'SKIP' }
}

# --- summary --------------------------------------------------------------
$bar = '=' * 72
Log ""
Log $bar
Log "SUMMARY"
Log $bar
foreach ($r in $results) {
    Log ('{0,-10} {1}' -f $r.Status, $r.Name)
}

$pass = ($results | Where-Object { $_.Status -eq 'PASS'    }).Count
$fail = ($results | Where-Object { $_.Status -like 'FAIL*' }).Count
$skip = ($results | Where-Object { $_.Status -eq 'SKIP'    }).Count

Log $bar
Log ("Total: {0}   Pass: {1}   Fail: {2}   Skip: {3}" -f $results.Count, $pass, $fail, $skip)
Log ("Log:   {0}" -f $logFile)
Log ("Done:  {0}" -f (Get-Date))

if ($fail -gt 0) { exit 1 } else { exit 0 }
