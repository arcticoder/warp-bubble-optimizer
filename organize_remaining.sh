#!/bin/bash
# Comprehensive file organization for remaining root .py files

set -e

cd /home/sherri3/Code/asciimath/warp-bubble-optimizer

# Create directories
mkdir -p demos examples scripts tools

echo "Moving remaining files..."

# Integration/control files → scripts
for file in integrated_*.py enhanced_*.py warp_bubble_power_pipeline*.py; do
    [ -f "$file" ] && git mv "$file" scripts/ 2>/dev/null && echo "  → scripts/$file"
done

# Breakthrough/physics demos → demos
for file in breakthrough_*.py physics_*.py; do
    [ -f "$file" ] && git mv "$file" demos/ 2>/dev/null && echo "  → demos/$file"
done

# Parameter scan files → scripts
for file in parameter_scan*.py comprehensive_parameter*.py; do
    [ -f "$file" ] && git mv "$file" scripts/ 2>/dev/null && echo "  → scripts/$file"
done

# Gaussian optimize variants → examples
for file in gaussian_optimize*.py; do
    [ -f "$file" ] && git mv "$file" examples/ 2>/dev/null && echo "  → examples/$file"
done

# Optimization/benchmark files → examples
for file in bayes_opt*.py hybrid_*.py comprehensive_benchmark*.py new_ansatz*.py optimization_summary*.py; do
    [ -f "$file" ] && git mv "$file" examples/ 2>/dev/null && echo "  → examples/$file"
done

# Validation/verification → scripts
for file in stress_tensor*.py h_infinity*.py; do
    [ -f "$file" ] && git mv "$file" scripts/ 2>/dev/null && echo "  → scripts/$file"
done

# Summary/status files → tools
for file in FINAL_*.py MVP_*.py final_*.py next_steps.py prepare_mvp*.py; do
    [ -f "$file" ] && git mv "$file" tools/ 2>/dev/null && echo "  → tools/$file"
done

# Utility files → tools
for file in gpu_check.py progress_tracker.py traceability_check.py visualize_*.py fix_*.py fidelity_runner.py; do
    [ -f "$file" ] && git mv "$file" tools/ 2>/dev/null && echo "  → tools/$file"
done

# Debug/test files → tools
for file in debug_*.py test_impulse_vnv.py; do
    [ -f "$file" ] && git mv "$file" tools/ 2>/dev/null && echo "  → tools/$file"
done

# UQ/simulation files → scripts
for file in uq_*.py analog_sim.py enhanced_qi_constraint.py enhanced_bubble_dynamics.py; do
    [ -f "$file" ] && git mv "$file" scripts/ 2>/dev/null && echo "  → scripts/$file"
done

echo ""
echo "Remaining root .py files:"
find . -maxdepth 1 -name "*.py" -type f | wc -l
