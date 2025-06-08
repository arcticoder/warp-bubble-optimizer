# Automated Warp Bubble Power Pipeline Execution Script (PowerShell)
# ================================================================
#
# This script runs the complete end-to-end power pipeline including:
# 1. Ghost/Phantom EFT energy source instantiation
# 2. Parameter space sweep (radius × velocity)
# 3. CMA-ES optimization of metric ansatz
# 4. Re-simulation with optimized parameters
# 5. Optional 3D mesh validation
#
# Usage:
#   .\bin\power_pipeline.ps1 [OPTIONS]
#
# Options:
#   -Quick           Run quick test with reduced parameter space
#   -Full            Run full comprehensive sweep (default)
#   -Validate        Include 3D mesh validation
#   -NoPlots         Skip plot generation
#   -OutputDir DIR   Output directory (default: power_pipeline_results)
#
# Authors: LQG-ANEC Research Team
# Date: June 8, 2025

param(
    [switch]$Quick = $false,
    [switch]$Full = $false,
    [switch]$Validate = $false,
    [switch]$NoPlots = $false,
    [string]$OutputDir = "power_pipeline_results",
    [switch]$Help = $false
)

# Show help if requested
if ($Help) {
    Write-Host "Usage: .\bin\power_pipeline.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Quick           Run quick test with reduced parameter space"
    Write-Host "  -Full            Run full comprehensive sweep (default)"
    Write-Host "  -Validate        Include 3D mesh validation"
    Write-Host "  -NoPlots         Skip plot generation"
    Write-Host "  -OutputDir DIR   Output directory (default: power_pipeline_results)"
    Write-Host "  -Help            Show this help message"
    exit 0
}

# Set error action preference
$ErrorActionPreference = "Stop"

# Script configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$PythonScript = Join-Path $ProjectRoot "src\power_pipeline.py"

# Default settings
if (-not $Full -and -not $Quick) {
    $Full = $true  # Default to full sweep
}

$GeneratePlots = -not $NoPlots

# Setup
Write-Host "🚀 Warp Bubble Power Pipeline Automation" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Project root: $ProjectRoot"
Write-Host "Output directory: $OutputDir"
Write-Host "Quick mode: $Quick"
Write-Host "Validation: $Validate"
Write-Host "Generate plots: $GeneratePlots"
Write-Host ""

# Check dependencies
Write-Host "📋 Checking dependencies..." -ForegroundColor Blue

try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion"
} catch {
    Write-Host "❌ Python not found. Please install Python 3.7+" -ForegroundColor Red
    exit 1
}

try {
    python -c "import numpy, pandas, matplotlib" 2>$null
    Write-Host "✅ Required packages available"
} catch {
    Write-Host "❌ Missing required Python packages. Installing..." -ForegroundColor Yellow
    pip install numpy pandas matplotlib scipy
}

if (-not (Test-Path $PythonScript)) {
    Write-Host "❌ Power pipeline script not found: $PythonScript" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Dependencies check completed" -ForegroundColor Green
Write-Host ""

# Set up environment
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src;$ProjectRoot\..\lqg-anec-framework\src;$env:PYTHONPATH"

# Create output directory
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# Run the power pipeline
Write-Host "🔄 Starting power pipeline execution..." -ForegroundColor Blue
Write-Host ""

if ($Quick) {
    Write-Host "⚡ Running in quick mode (reduced parameter space)" -ForegroundColor Yellow
    $pythonCode = @"
import sys
sys.path.insert(0, 'src')
from power_pipeline import WarpBubblePowerPipeline

# Quick test configuration
pipeline = WarpBubblePowerPipeline(output_dir='$OutputDir')
results = pipeline.run_full_pipeline(
    radii=[10.0, 20.0],              # 2 radii
    speeds=[1000, 5000],             # 2 speeds  
    optimize_target='auto',
    validate_best=$($Validate.ToString().ToLower())
)

print('🎯 Quick Pipeline Results:')
if results['optimization_result'].get('success'):
    opt = results['optimization_result']
    print(f'   Energy: {opt["final_energy_negative"]:.3e} J')
    print(f'   Stability: {opt["final_stability"]:.3f}')
    print(f'   Solver: {opt["solver"]}')
print(f'   Runtime: {results["total_runtime"]:.1f} seconds')
"@
    python -c $pythonCode
} else {
    Write-Host "� Running full comprehensive sweep" -ForegroundColor Blue
    $pythonCode = @"
import sys
sys.path.insert(0, 'src')
from power_pipeline import WarpBubblePowerPipeline

# Full configuration
pipeline = WarpBubblePowerPipeline(output_dir='$OutputDir')
results = pipeline.run_full_pipeline(
    radii=[5.0, 10.0, 20.0, 50.0],       # 4 radii
    speeds=[1000, 5000, 10000, 50000],   # 4 speeds
    optimize_target='auto',
    validate_best=$($Validate.ToString().ToLower())
)

print('🎯 Full Pipeline Results:')
if results['optimization_result'].get('success'):
    opt = results['optimization_result']
    print(f'   Energy: {opt["final_energy_negative"]:.3e} J')
    print(f'   Stability: {opt["final_stability"]:.3f}')
    print(f'   Solver: {opt["solver"]}')
    print(f'   Configuration: R={opt["radius"]}m, v={opt["speed"]}c')
print(f'   Runtime: {results["total_runtime"]:.1f} seconds')
"@
    python -c $pythonCode
}

# Check if results exist
if (Test-Path "power_pipeline_results") {
    Write-Host "✅ Pipeline completed successfully" -ForegroundColor Green
    Write-Host "📁 Results available in: power_pipeline_results/" -ForegroundColor Cyan
    
    # List generated files
    Write-Host "📄 Generated files:" -ForegroundColor Cyan
    Get-ChildItem "power_pipeline_results" | Format-Table Name, Length, LastWriteTime
    
    # Show key results if available
    if (Test-Path "power_pipeline_results/pipeline_results.json") {
        Write-Host "🎯 Key Results:" -ForegroundColor Yellow
        python -c "
import json, sys
try:
    with open('power_pipeline_results/pipeline_results.json', 'r') as f:
        results = json.load(f)
    
    opt_result = results.get('optimization_result', {})
    if opt_result.get('success'):
        print(f'   Solver: {opt_result.get(\"solver\", \"unknown\")}')
        print(f'   Energy: {opt_result.get(\"final_energy_negative\", 0):.3e} J')
        print(f'   Stability: {opt_result.get(\"final_stability\", 0):.3f}')
        print(f'   Runtime: {results.get(\"total_runtime\", 0):.1f} s')
    else:
        print('   Optimization did not complete successfully')
        
except Exception as e:
    print(f'   Could not parse results: {e}')
"
    }
    
} else {
    Write-Host "❌ Pipeline failed - no results directory found" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🏆 Power pipeline complete!" -ForegroundColor Green
Write-Host "📂 Open power_pipeline_results/ to view all outputs" -ForegroundColor Cyan

# Option to open results folder
$response = Read-Host "Open results folder? (y/N)"
if ($response -eq "y" -or $response -eq "Y") {
    Start-Process "power_pipeline_results"
}
