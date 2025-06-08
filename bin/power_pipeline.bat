@echo off
REM Automated Warp Bubble Power Pipeline Execution Script (Windows Batch)
REM =====================================================================
REM
REM This script runs the complete end-to-end power pipeline including:
REM 1. Ghost/Phantom EFT energy source instantiation
REM 2. Parameter space sweep (radius × velocity)
REM 3. CMA-ES optimization of metric ansatz
REM 4. Re-simulation with optimized parameters
REM 5. Optional 3D mesh validation
REM
REM Usage:
REM   bin\power_pipeline.bat [quick|full] [validate] [no-plots]
REM
REM Arguments:
REM   quick      Run quick test with reduced parameter space
REM   full       Run full comprehensive sweep (default)
REM   validate   Include 3D mesh validation
REM   no-plots   Skip plot generation
REM
REM Authors: LQG-ANEC Research Team
REM Date: June 8, 2025

setlocal enabledelayedexpansion

REM Configuration
set "OUTPUT_DIR=power_pipeline_results"
set "VALIDATE=false"
set "QUICK_MODE=false"
set "GENERATE_PLOTS=true"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :setup
if /I "%~1"=="quick" (
    set "QUICK_MODE=true"
    shift
    goto :parse_args
)
if /I "%~1"=="full" (
    set "QUICK_MODE=false"
    shift
    goto :parse_args
)
if /I "%~1"=="validate" (
    set "VALIDATE=true"
    shift
    goto :parse_args
)
if /I "%~1"=="no-plots" (
    set "GENERATE_PLOTS=false"
    shift
    goto :parse_args
)
if /I "%~1"=="help" (
    echo Usage: bin\power_pipeline.bat [quick^|full] [validate] [no-plots]
    echo.
    echo Arguments:
    echo   quick      Run quick test with reduced parameter space
    echo   full       Run full comprehensive sweep ^(default^)
    echo   validate   Include 3D mesh validation
    echo   no-plots   Skip plot generation
    echo   help       Show this help message
    exit /b 0
)
echo Unknown argument: %~1
echo Use 'help' for usage information
exit /b 1

:setup
REM Setup
echo 🚀 Warp Bubble Power Pipeline Automation
echo ========================================
echo Project root: %CD%
echo Output directory: %OUTPUT_DIR%
echo Quick mode: %QUICK_MODE%
echo Validation: %VALIDATE%
echo Generate plots: %GENERATE_PLOTS%
echo.

REM Check dependencies
echo 📋 Checking dependencies...

python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.7+
    exit /b 1
)

python -c "import numpy, pandas, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo ❌ Missing required Python packages. Installing...
    pip install numpy pandas matplotlib scipy
)

if not exist "src\power_pipeline.py" (
    echo ❌ Power pipeline script not found: src\power_pipeline.py
    exit /b 1
)

echo ✅ Dependencies check completed
echo.

REM Set up environment
set "PYTHONPATH=%CD%\src;%CD%\..\lqg-anec-framework\src;%PYTHONPATH%"

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Run the power pipeline
echo 🔄 Starting power pipeline execution...
echo.

if "%QUICK_MODE%"=="true" (
    echo ⚡ Running in quick mode ^(reduced parameter space^)
    python -c "
import sys
sys.path.insert(0, 'src')
from power_pipeline import WarpBubblePowerPipeline

# Quick test configuration
pipeline = WarpBubblePowerPipeline(output_dir='%OUTPUT_DIR%')
results = pipeline.run_full_pipeline(
    radii=[10.0, 20.0],              # 2 radii
    speeds=[1000, 5000],             # 2 speeds  
    optimize_target='auto',
    validate_best=%VALIDATE%
)

print('🎯 Quick Pipeline Results:')
if results['optimization_result'].get('success'):
    opt = results['optimization_result']
    print(f'   Energy: {opt[\"final_energy_negative\"]:.3e} J')
    print(f'   Stability: {opt[\"final_stability\"]:.3f}')
    print(f'   Solver: {opt[\"solver\"]}')
print(f'   Runtime: {results[\"total_runtime\"]:.1f} seconds')
"
) else (
    echo 🔄 Running full comprehensive sweep
    python -c "
import sys
sys.path.insert(0, 'src')
from power_pipeline import WarpBubblePowerPipeline

# Full configuration
pipeline = WarpBubblePowerPipeline(output_dir='%OUTPUT_DIR%')
results = pipeline.run_full_pipeline(
    radii=[5.0, 10.0, 20.0, 50.0],       # 4 radii
    speeds=[1000, 5000, 10000, 50000],   # 4 speeds
    optimize_target='auto',
    validate_best=%VALIDATE%
)

print('🎯 Full Pipeline Results:')
if results['optimization_result'].get('success'):
    opt = results['optimization_result']
    print(f'   Energy: {opt[\"final_energy_negative\"]:.3e} J')
    print(f'   Stability: {opt[\"final_stability\"]:.3f}')
    print(f'   Solver: {opt[\"solver\"]}')
    print(f'   Configuration: R={opt[\"radius\"]}m, v={opt[\"speed\"]}c')
print(f'   Runtime: {results[\"total_runtime\"]:.1f} seconds')
"
)

REM Check results
echo.
echo 📊 Pipeline Results Summary
echo ==========================

if exist "%OUTPUT_DIR%\power_sweep.csv" (
    echo ✅ Parameter sweep completed: %OUTPUT_DIR%\power_sweep.csv
    for /f %%i in ('type "%OUTPUT_DIR%\power_sweep.csv" ^| find /c /v ""') do set "NUM_CONFIGS=%%i"
    set /a NUM_CONFIGS=!NUM_CONFIGS!-1
    echo    Configurations tested: !NUM_CONFIGS!
) else (
    echo ❌ Parameter sweep file not found
)

if exist "%OUTPUT_DIR%\pipeline_results.json" (
    echo ✅ Pipeline results saved: %OUTPUT_DIR%\pipeline_results.json
) else (
    echo ❌ Pipeline results file not found
)

REM List optimization results
set "OPTIM_COUNT=0"
for %%f in ("%OUTPUT_DIR%\optimization_*.json") do (
    if exist "%%f" set /a OPTIM_COUNT+=1
)
if !OPTIM_COUNT! gtr 0 (
    echo ✅ Optimization results: !OPTIM_COUNT! solver^(s^) optimized
    for %%f in ("%OUTPUT_DIR%\optimization_*.json") do (
        echo    - %%~nxf
    )
) else (
    echo ⚠️  No optimization results found
)

REM List validation results
if "%VALIDATE%"=="true" (
    set "VALID_COUNT=0"
    for %%f in ("%OUTPUT_DIR%\validation_*.json") do (
        if exist "%%f" set /a VALID_COUNT+=1
    )
    if !VALID_COUNT! gtr 0 (
        echo ✅ Validation results: !VALID_COUNT! configuration^(s^) validated
        for %%f in ("%OUTPUT_DIR%\validation_*.json") do (
            echo    - %%~nxf
        )
    ) else (
        echo ⚠️  No validation results found
    )
)

REM List plots
if "%GENERATE_PLOTS%"=="true" (
    set "PLOT_COUNT=0"
    for %%f in ("%OUTPUT_DIR%\*.png") do (
        if exist "%%f" set /a PLOT_COUNT+=1
    )
    if !PLOT_COUNT! gtr 0 (
        echo ✅ Visualization plots: !PLOT_COUNT! plot^(s^) generated
        for %%f in ("%OUTPUT_DIR%\*.png") do (
            echo    - %%~nxf
        )
    ) else (
        echo ⚠️  No plots found
    )
)

echo.
echo 📁 All results saved to: %CD%\%OUTPUT_DIR%

REM Optional: Open results directory
if exist "%OUTPUT_DIR%" (
    echo 💡 Opening results directory...
    start "" explorer "%CD%\%OUTPUT_DIR%"
)

echo.
echo 🏆 Power pipeline automation completed successfully!
echo.
echo Next steps:
echo   1. Review power_sweep.csv for parameter space analysis
echo   2. Check optimization_*.json for best configurations
echo   3. Examine *.png plots for visual analysis
echo   4. Use pipeline_results.json for programmatic access

if "%VALIDATE%"=="false" (
    echo.
    echo 💡 Tip: Run with 'validate' for 3D mesh validation
)

endlocal
