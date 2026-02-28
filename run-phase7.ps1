<#
.SYNOPSIS
    Run COUNTERCASE Phase 7 -- Evaluation, Ablation, and Research Writeup.

.DESCRIPTION
    Runs all Phase 7 evaluation steps: retrieval evaluation with t-tests,
    counterfactual evaluation, explanation faithfulness, human eval template,
    test set creation, and result aggregation.

.PARAMETER Steps
    Comma-separated list of steps to run. Default: all.
    Valid steps: retrieval, counterfactual, faithfulness, human-template,
                 create-testset, merge-testset, split-testset, aggregate

.PARAMETER TestSet
    Path to the test set JSON (default: countercase/evaluation/data/test_set.json).

.PARAMETER OutputDir
    Directory for evaluation results (default: countercase/evaluation/results).

.PARAMETER CaseTexts
    Path to case_texts.json for citation-based test set creation.

.PARAMETER Cutoff
    Year cutoff for train/test split (default: 2020).

.EXAMPLE
    .\run-phase7.ps1
    .\run-phase7.ps1 -Steps "retrieval,counterfactual"
    .\run-phase7.ps1 -Steps "create-testset" -CaseTexts countercase/data/case_texts.json
#>

param(
    [string]$Steps = "all",
    [string]$TestSet = "countercase/evaluation/data/test_set.json",
    [string]$OutputDir = "countercase/evaluation/results",
    [string]$CaseTexts = "countercase/data/case_texts.json",
    [string]$CitationIndex = "countercase/data/citation_index.json",
    [int]$Cutoff = 2020
)

$ErrorActionPreference = "Stop"

# --- Resolve steps ---
$allSteps = @(
    "retrieval", "counterfactual", "faithfulness",
    "human-template", "create-testset", "merge-testset",
    "split-testset", "aggregate"
)

if ($Steps -eq "all") {
    $runSteps = $allSteps
} else {
    $runSteps = $Steps -split "," | ForEach-Object { $_.Trim() }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  COUNTERCASE Phase 7 Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Steps    : $($runSteps -join ', ')" -ForegroundColor Yellow
Write-Host "Test Set : $TestSet"
Write-Host "Output   : $OutputDir"
Write-Host ""

# Ensure output directory exists
New-Item -Path $OutputDir -ItemType Directory -Force | Out-Null

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$passed = 0
$failed = 0

function Run-Step {
    param([string]$Name, [scriptblock]$Block)
    Write-Host "--- [$Name] ---" -ForegroundColor Green
    $stepSw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        & $Block
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "Exit code $LASTEXITCODE" }
        $stepSw.Stop()
        Write-Host "  OK ($([math]::Round($stepSw.Elapsed.TotalSeconds, 1))s)" -ForegroundColor Green
        $script:passed++
    } catch {
        $stepSw.Stop()
        Write-Host "  FAILED: $_" -ForegroundColor Red
        $script:failed++
    }
    Write-Host ""
}

# --- 1. Retrieval evaluation ---
if ($runSteps -contains "retrieval") {
    Run-Step "Retrieval Evaluation" {
        python -c @"
from countercase.evaluation.eval_harness import EvalHarness
h = EvalHarness()
ts = h.load_test_set('$($TestSet -replace '\\','/')')
h.run_full_evaluation(ts, output_dir='$($OutputDir -replace '\\','/')')
"@
    }
}

# --- 2. Counterfactual evaluation ---
if ($runSteps -contains "counterfactual") {
    Run-Step "Counterfactual Evaluation" {
        python -m countercase.evaluation.counterfactual_eval `
            --eval-set countercase/evaluation/data/cf_eval_set.json `
            --output "$OutputDir/counterfactual_eval.json"
    }
}

# --- 3. Explanation faithfulness ---
if ($runSteps -contains "faithfulness") {
    Run-Step "Explanation Faithfulness" {
        python -m countercase.evaluation.explanation_eval faithfulness `
            --input countercase/evaluation/data/explanations.json `
            --output "$OutputDir/faithfulness.json"
    }
}

# --- 4. Human evaluation template ---
if ($runSteps -contains "human-template") {
    Run-Step "Human Eval Template" {
        python -m countercase.evaluation.explanation_eval gen-template `
            --input countercase/evaluation/data/explanations.json `
            --output "$OutputDir/human_eval_template.csv"
    }
}

# --- 5. Create test set from citations ---
if ($runSteps -contains "create-testset") {
    Run-Step "Create Citation Test Set" {
        $createArgs = @(
            "-m", "countercase.evaluation.create_test_set", "citations",
            "--case-texts", $CaseTexts,
            "--output", "countercase/evaluation/data/citation_test_set.json"
        )
        if (Test-Path $CitationIndex) {
            $createArgs += "--citation-index"
            $createArgs += $CitationIndex
        }
        python @createArgs
    }
}

# --- 6. Merge test sets ---
if ($runSteps -contains "merge-testset") {
    Run-Step "Merge Test Sets" {
        python -m countercase.evaluation.create_test_set merge `
            countercase/evaluation/data/test_set.json `
            countercase/evaluation/data/citation_test_set.json `
            --output countercase/evaluation/data/merged.json
    }
}

# --- 7. Split test set by year ---
if ($runSteps -contains "split-testset") {
    Run-Step "Split Test Set (cutoff=$Cutoff)" {
        python -m countercase.evaluation.create_test_set split `
            countercase/evaluation/data/merged.json `
            --cutoff $Cutoff
    }
}

# --- 8. Aggregate all results ---
if ($runSteps -contains "aggregate") {
    Run-Step "Aggregate Results" {
        python -m countercase.evaluation.aggregate_results `
            --results-dir $OutputDir
    }
}

# --- Summary ---
$sw.Stop()
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Phase 7 Complete" -ForegroundColor Cyan
Write-Host "  Passed: $passed  Failed: $failed" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })
Write-Host "  Total time: $([math]::Round($sw.Elapsed.TotalSeconds, 1))s" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# List output files
Write-Host ""
Write-Host "Output files:" -ForegroundColor Yellow
Get-ChildItem -Path $OutputDir -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object Name, @{N="Size";E={"{0:N1} KB" -f ($_.Length/1KB)}}, LastWriteTime |
    Format-Table -AutoSize

if ($failed -gt 0) { exit 1 }
