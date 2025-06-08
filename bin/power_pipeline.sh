#!/usr/bin/env bash
#
# Automated Warp Bubble Power Pipeline Execution Script
# =====================================================
#
# This script runs the complete end-to-end power pipeline including:
# 1. Ghost/Phantom EFT energy source instantiation
# 2. Parameter space sweep (radius × velocity)
# 3. CMA-ES optimization of metric ansatz
# 4. Re-simulation with optimized parameters
# 5. Optional 3D mesh validation
#
# Usage:
#   ./bin/power_pipeline.sh [OPTIONS]
#
# Options:
#   --quick          Run quick test with reduced parameter space
#   --full           Run full comprehensive sweep (default)
#   --validate       Include 3D mesh validation
#   --no-plots       Skip plot generation
#   --output DIR     Output directory (default: power_pipeline_results)
#
# Authors: LQG-ANEC Research Team
# Date: June 8, 2025
#

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$PROJECT_ROOT/src/power_pipeline.py"

# Default settings
OUTPUT_DIR="power_pipeline_results"
VALIDATE=false
QUICK_MODE=false
GENERATE_PLOTS=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --full)
            QUICK_MODE=false
            shift
            ;;
        --validate)
            VALIDATE=true
            shift
            ;;
        --no-plots)
            GENERATE_PLOTS=false
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick          Run quick test with reduced parameter space"
            echo "  --full           Run full comprehensive sweep (default)"
            echo "  --validate       Include 3D mesh validation"
            echo "  --no-plots       Skip plot generation"
            echo "  --output DIR     Output directory (default: power_pipeline_results)"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Setup
echo "🚀 Warp Bubble Power Pipeline Automation"
echo "========================================"
echo "Project root: $PROJECT_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo "Quick mode: $QUICK_MODE"
echo "Validation: $VALIDATE"
echo "Generate plots: $GENERATE_PLOTS"
echo ""

# Check dependencies
echo "📋 Checking dependencies..."

if ! python3 -c "import numpy, pandas, matplotlib" 2>/dev/null; then
    echo "❌ Missing required Python packages. Installing..."
    pip install numpy pandas matplotlib scipy
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "❌ Power pipeline script not found: $PYTHON_SCRIPT"
    exit 1
fi

echo "✅ Dependencies check completed"
echo ""

# Set up environment
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/../lqg-anec-framework/src:$PYTHONPATH"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the power pipeline
echo "🔄 Starting power pipeline execution..."
echo ""

if [[ "$QUICK_MODE" == true ]]; then
    echo "⚡ Running in quick mode (reduced parameter space)"
    python3 -c "
import sys
sys.path.insert(0, 'src')
from power_pipeline import WarpBubblePowerPipeline

# Quick test configuration
pipeline = WarpBubblePowerPipeline(output_dir='$OUTPUT_DIR')
results = pipeline.run_full_pipeline(
    radii=[10.0, 20.0],              # 2 radii
    speeds=[1000, 5000],             # 2 speeds  
    optimize_target='auto',
    validate_best=$VALIDATE
)

print('🎯 Quick Pipeline Results:')
if results['optimization_result'].get('success'):
    opt = results['optimization_result']
    print(f'   Energy: {opt[\"final_energy_negative\"]:.3e} J')
    print(f'   Stability: {opt[\"final_stability\"]:.3f}')
    print(f'   Solver: {opt[\"solver\"]}')
print(f'   Runtime: {results[\"total_runtime\"]:.1f} seconds')
"
else
    echo "🔄 Running full comprehensive sweep"
    python3 -c "
import sys
sys.path.insert(0, 'src')
from power_pipeline import WarpBubblePowerPipeline

# Full configuration
pipeline = WarpBubblePowerPipeline(output_dir='$OUTPUT_DIR')
results = pipeline.run_full_pipeline(
    radii=[5.0, 10.0, 20.0, 50.0],       # 4 radii
    speeds=[1000, 5000, 10000, 50000],   # 4 speeds
    optimize_target='auto',
    validate_best=$VALIDATE
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
fi

# Check results
echo ""
echo "📊 Pipeline Results Summary"
echo "=========================="

if [[ -f "$OUTPUT_DIR/power_sweep.csv" ]]; then
    echo "✅ Parameter sweep completed: $OUTPUT_DIR/power_sweep.csv"
    NUM_CONFIGS=$(tail -n +2 "$OUTPUT_DIR/power_sweep.csv" | wc -l)
    echo "   Configurations tested: $NUM_CONFIGS"
else
    echo "❌ Parameter sweep file not found"
fi

if [[ -f "$OUTPUT_DIR/pipeline_results.json" ]]; then
    echo "✅ Pipeline results saved: $OUTPUT_DIR/pipeline_results.json"
else
    echo "❌ Pipeline results file not found"
fi

# List optimization results
OPTIM_FILES=$(find "$OUTPUT_DIR" -name "optimization_*.json" 2>/dev/null | wc -l)
if [[ $OPTIM_FILES -gt 0 ]]; then
    echo "✅ Optimization results: $OPTIM_FILES solver(s) optimized"
    find "$OUTPUT_DIR" -name "optimization_*.json" -exec basename {} \; | sed 's/^/   - /'
else
    echo "⚠️  No optimization results found"
fi

# List validation results
if [[ "$VALIDATE" == true ]]; then
    VALID_FILES=$(find "$OUTPUT_DIR" -name "validation_*.json" 2>/dev/null | wc -l)
    if [[ $VALID_FILES -gt 0 ]]; then
        echo "✅ Validation results: $VALID_FILES configuration(s) validated"
        find "$OUTPUT_DIR" -name "validation_*.json" -exec basename {} \; | sed 's/^/   - /'
    else
        echo "⚠️  No validation results found"
    fi
fi

# List plots
if [[ "$GENERATE_PLOTS" == true ]]; then
    PLOT_FILES=$(find "$OUTPUT_DIR" -name "*.png" 2>/dev/null | wc -l)
    if [[ $PLOT_FILES -gt 0 ]]; then
        echo "✅ Visualization plots: $PLOT_FILES plot(s) generated"
        find "$OUTPUT_DIR" -name "*.png" -exec basename {} \; | sed 's/^/   - /'
    else
        echo "⚠️  No plots found"
    fi
fi

echo ""
echo "📁 All results saved to: $(realpath "$OUTPUT_DIR")"

echo ""
echo "🏆 Power pipeline automation completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Review power_sweep.csv for parameter space analysis"
echo "  2. Check optimization_*.json for best configurations"
echo "  3. Examine *.png plots for visual analysis"
echo "  4. Use pipeline_results.json for programmatic access"

if [[ "$VALIDATE" == false ]]; then
    echo ""
    echo "💡 Tip: Run with --validate for 3D mesh validation"
fi
