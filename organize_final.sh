#!/bin/bash
# Final cleanup of root .py files

set -e

cd /home/sherri3/Code/asciimath/warp-bubble-optimizer

echo "Final file organization..."

# Summary files → tools
for file in COMPLETE_DIGITAL_TWIN_FINAL_SUMMARY.py DIGITAL_TWIN_SUMMARY.py SETUP_COMPLETE.py matplotlib_fix_summary.py; do
    [ -f "$file" ] && git mv "$file" tools/ 2>/dev/null && echo "  → tools/$file"
done

# Validation files → scripts
for file in comprehensive_theoretical_validation.py metric_backreaction_analysis.py metric_stability_validation.py diagnostic_gut_polymer.py; do
    [ -f "$file" ] && git mv "$file" scripts/ 2>/dev/null && echo "  → scripts/$file"
done

# Evolution/simulation files → scripts
for file in evolve_*.py fdtd_*.py phenomenology_simulation_framework.py; do
    [ -f "$file" ] && git mv "$file" scripts/ 2>/dev/null && echo "  → scripts/$file"
done

# Parameter/sweep files → scripts
for file in parameter_space_sweep.py; do
    [ -f "$file" ] && git mv "$file" scripts/ 2>/dev/null && echo "  → scripts/$file"
done

# Optimizer examples → examples
for file in ultimate_bspline_optimizer.py spline_refine_jax.py qi_constraint.py; do
    [ -f "$file" ] && git mv "$file" examples/ 2>/dev/null && echo "  → examples/$file"
done

# Demo files → demos
for file in refined_breakthrough_demo.py simple_atmospheric_demo.py; do
    [ -f "$file" ] && git mv "$file" demos/ 2>/dev/null && echo "  → demos/$file"
done

# Benchmark/test files → examples
for file in ultimate_benchmark_suite.py test_jax_acceleration.py; do
    [ -f "$file" ] && git mv "$file" examples/ 2>/dev/null && echo "  → examples/$file"
done

# Keep setup.py and conftest.py in root (they should stay there)

echo ""
echo "Final count of root .py files:"
find . -maxdepth 1 -name "*.py" -type f
echo ""
echo "Organization complete! Files should now be organized into:"
echo "  demos/     - Demo and showcase scripts"
echo "  examples/  - Example optimizers and benchmarks"  
echo "  scripts/   - Analysis, validation, and simulation scripts"
echo "  tools/     - Utilities and diagnostic tools"
echo "  Root kept: setup.py, conftest.py (required in root)"
