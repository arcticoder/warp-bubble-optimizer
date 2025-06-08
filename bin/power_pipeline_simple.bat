@echo off
REM ========================================
REM Warp Bubble Power Pipeline Automation
REM Batch script for Windows systems
REM ========================================

setlocal enabledelayedexpansion

REM Default settings
set "OUTPUT_DIR=power_pipeline_results"
set "QUICK_MODE=false"
set "VALIDATE=false"
set "GENERATE_PLOTS=true"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto setup
if /i "%~1"=="quick" (
    set "QUICK_MODE=true"
    shift
    goto parse_args
)
if /i "%~1"=="full" (
    set "QUICK_MODE=false"
    shift
    goto parse_args
)
if /i "%~1"=="validate" (
    set "VALIDATE=true"
    shift
    goto parse_args
)
if /i "%~1"=="no-plots" (
    set "GENERATE_PLOTS=false"
    shift
    goto parse_args
)
if /i "%~1"=="help" (
    echo.
    echo Warp Bubble Power Pipeline Automation
    echo ====================================
    echo.
    echo Usage: power_pipeline.bat [options]
    echo.
    echo Options:
    echo   quick      Run quick test (2x2 parameter space)
    echo   full       Run full comprehensive sweep (default)
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
echo Warp Bubble Power Pipeline Automation
echo ========================================
echo Project root: %CD%
echo Output directory: %OUTPUT_DIR%
echo Quick mode: %QUICK_MODE%
echo Validation: %VALIDATE%
echo Generate plots: %GENERATE_PLOTS%
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+
    exit /b 1
)

REM Check src/power_pipeline.py
if not exist "src\power_pipeline.py" (
    echo Error: src\power_pipeline.py not found
    exit /b 1
)

echo Dependencies check completed
echo.

REM Set up environment
set "PYTHONPATH=%CD%\src;%CD%\..\lqg-anec-framework\src;%PYTHONPATH%"

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Run the power pipeline
echo Starting power pipeline execution...
echo.

if "%QUICK_MODE%"=="true" (
    echo Running in quick mode (reduced parameter space)
    if "%VALIDATE%"=="true" (
        python src\power_pipeline.py --quick --output-dir "%OUTPUT_DIR%"
    ) else (
        python src\power_pipeline.py --quick --output-dir "%OUTPUT_DIR%" --no-validation
    )
) else (
    echo Running in full mode (complete parameter space)
    if "%VALIDATE%"=="true" (
        python src\power_pipeline.py --output-dir "%OUTPUT_DIR%"
    ) else (
        python src\power_pipeline.py --output-dir "%OUTPUT_DIR%" --no-validation
    )
)

if errorlevel 1 (
    echo.
    echo Pipeline execution failed!
    exit /b 1
)

echo.
echo Pipeline completed successfully!
echo Results saved to: %OUTPUT_DIR%
echo.

REM List generated files
echo Generated files:
for %%f in ("%OUTPUT_DIR%\*") do echo   %%~nxf

REM Optional validation
if "%VALIDATE%"=="true" (
    echo.
    echo Running 3D mesh validation...
    python src\power_pipeline.py --validate-only --output-dir "%OUTPUT_DIR%"
)

REM Generate plots if requested
if "%GENERATE_PLOTS%"=="true" (
    echo.
    echo Generating visualization plots...
    python src\power_pipeline.py --plot-only --output-dir "%OUTPUT_DIR%"
)

echo.
echo ========================================
echo Warp Bubble Pipeline Complete
echo ========================================
echo.
echo Next steps:
echo   - Analyze results in: %OUTPUT_DIR%
echo   - View plots: %OUTPUT_DIR%\*.png
echo   - Read summary: %OUTPUT_DIR%\pipeline_results.json
echo.
echo Programmatic usage:
echo   python -c "from src.power_pipeline import WarpBubblePowerPipeline; pipeline = WarpBubblePowerPipeline(output_dir='%OUTPUT_DIR%'); results = pipeline.run_full_pipeline()"

endlocal
