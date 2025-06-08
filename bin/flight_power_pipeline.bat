@echo off
REM Flight Power Pipeline Automation - Windows Batch Script
REM Usage: flight_power_pipeline.bat [target_distance_ly] [output_dir]

echo ============================================================
echo 🚀 WARP BUBBLE FLIGHT POWER PIPELINE - AUTOMATED
echo ============================================================

REM Set default parameters
set TARGET_DISTANCE=%1
set OUTPUT_DIR=%2
if "%TARGET_DISTANCE%"=="" set TARGET_DISTANCE=4.37
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=flight_results

REM Get project root directory
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

echo 📍 Project root: %PROJECT_ROOT%
echo 🎯 Target distance: %TARGET_DISTANCE% light years
echo 📁 Output directory: %OUTPUT_DIR%
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.7+ and try again.
    pause
    exit /b 1
)

REM Check if flight pipeline script exists
if not exist "scripts\flight_power_pipeline.py" (
    echo ❌ Flight pipeline script not found at scripts\flight_power_pipeline.py
    pause
    exit /b 1
)

echo 🚀 Running flight power pipeline...
echo Command: python scripts\flight_power_pipeline.py --target-distance %TARGET_DISTANCE% --output-dir %OUTPUT_DIR%
echo.

REM Run the flight pipeline
python scripts\flight_power_pipeline.py --target-distance %TARGET_DISTANCE% --output-dir %OUTPUT_DIR%

if %errorlevel% equ 0 (
    echo.
    echo ✅ FLIGHT POWER PIPELINE COMPLETED SUCCESSFULLY
    echo ----------------------------------------
    echo 📁 Results location: %OUTPUT_DIR%\
    echo 📊 Power sweep: %OUTPUT_DIR%\flight_power_sweep.csv
    echo 🚀 Flight profile: %OUTPUT_DIR%\flight_power_profile.json
    echo.
    echo 📈 Next steps:
    echo    • Review CSV data for trajectory optimization
    echo    • Integrate JSON profile into mission planners
    echo    • Scale energy budgets for realistic missions
    echo    • Plan test flight validation campaigns
) else (
    echo ❌ Flight power pipeline failed with exit code %errorlevel%
    pause
    exit /b %errorlevel%
)

pause
