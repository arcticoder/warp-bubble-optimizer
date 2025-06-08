# Flight Power Pipeline Automation - PowerShell Script
# Usage: .\flight_power_pipeline.ps1 [target_distance_ly] [output_dir]

param(
    [double]$TargetDistance = 4.37,  # Proxima Centauri distance
    [string]$OutputDir = "flight_results"
)

Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "🚀 WARP BUBBLE FLIGHT POWER PIPELINE - AUTOMATED" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "📍 Project root: $ProjectRoot" -ForegroundColor Cyan
Write-Host "🎯 Target distance: $TargetDistance light years" -ForegroundColor Cyan
Write-Host "📁 Output directory: $OutputDir" -ForegroundColor Cyan
Write-Host ""

# Change to project directory
Set-Location $ProjectRoot

# Check if Python is available
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Please install Python 3.7+ and try again." -ForegroundColor Red
    exit 1
}

# Check if the flight pipeline script exists
$FlightScriptPath = Join-Path $ProjectRoot "scripts\flight_power_pipeline.py"
if (-not (Test-Path $FlightScriptPath)) {
    Write-Host "❌ Flight pipeline script not found at scripts\flight_power_pipeline.py" -ForegroundColor Red
    exit 1
}

Write-Host "🚀 Running flight power pipeline..." -ForegroundColor Yellow
$Command = "python scripts\flight_power_pipeline.py --target-distance $TargetDistance --output-dir $OutputDir"
Write-Host "Command: $Command" -ForegroundColor Gray
Write-Host ""

# Run the flight pipeline
try {
    & python scripts\flight_power_pipeline.py --target-distance $TargetDistance --output-dir $OutputDir
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ FLIGHT POWER PIPELINE COMPLETED SUCCESSFULLY" -ForegroundColor Green
        Write-Host "----------------------------------------" -ForegroundColor Green
        Write-Host "📁 Results location: $OutputDir\" -ForegroundColor Cyan
        Write-Host "📊 Power sweep: $OutputDir\flight_power_sweep.csv" -ForegroundColor Cyan
        Write-Host "🚀 Flight profile: $OutputDir\flight_power_profile.json" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "📈 Next steps:" -ForegroundColor Yellow
        Write-Host "   • Review CSV data for trajectory optimization" -ForegroundColor White
        Write-Host "   • Integrate JSON profile into mission planners" -ForegroundColor White
        Write-Host "   • Scale energy budgets for realistic missions" -ForegroundColor White
        Write-Host "   • Plan test flight validation campaigns" -ForegroundColor White
    } else {
        Write-Host "❌ Flight power pipeline failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "❌ Error running flight power pipeline: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
