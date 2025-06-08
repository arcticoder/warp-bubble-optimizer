#!/bin/bash
# Flight Power Pipeline Automation - Bash Script
# Usage: ./flight_power_pipeline.sh [target_distance_ly] [output_dir]

echo "============================================================"
echo "🚀 WARP BUBBLE FLIGHT POWER PIPELINE - AUTOMATED"
echo "============================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default parameters
TARGET_DISTANCE=${1:-4.37}  # Proxima Centauri distance
OUTPUT_DIR=${2:-"flight_results"}

echo "📍 Project root: $PROJECT_ROOT"
echo "🎯 Target distance: $TARGET_DISTANCE light years"
echo "📁 Output directory: $OUTPUT_DIR"
echo

# Change to project directory
cd "$PROJECT_ROOT"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found! Please install Python 3.7+ and try again."
    exit 1
fi

# Check if the flight pipeline script exists
if [ ! -f "scripts/flight_power_pipeline.py" ]; then
    echo "❌ Flight pipeline script not found at scripts/flight_power_pipeline.py"
    exit 1
fi

echo "🚀 Running flight power pipeline..."
echo "Command: python scripts/flight_power_pipeline.py --target-distance $TARGET_DISTANCE --output-dir $OUTPUT_DIR"
echo

# Run the flight pipeline
python scripts/flight_power_pipeline.py \
    --target-distance "$TARGET_DISTANCE" \
    --output-dir "$OUTPUT_DIR"

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo
    echo "✅ FLIGHT POWER PIPELINE COMPLETED SUCCESSFULLY"
    echo "----------------------------------------"
    echo "📁 Results location: $OUTPUT_DIR/"
    echo "📊 Power sweep: $OUTPUT_DIR/flight_power_sweep.csv"
    echo "🚀 Flight profile: $OUTPUT_DIR/flight_power_profile.json"
    echo
    echo "📈 Next steps:"
    echo "   • Review CSV data for trajectory optimization"
    echo "   • Integrate JSON profile into mission planners"
    echo "   • Scale energy budgets for realistic missions"
    echo "   • Plan test flight validation campaigns"
else
    echo "❌ Flight power pipeline failed with exit code $?"
    exit 1
fi
