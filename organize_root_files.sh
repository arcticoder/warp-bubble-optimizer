#!/bin/bash
# Script to organize root .py files into appropriate subdirectories

set -e

# Create organization directories if they don't exist
mkdir -p demos
mkdir -p scripts
mkdir -p examples

echo "Organizing root Python files..."

# Move demo files
for file in demo_*.py; do
    if [ -f "$file" ]; then
        git mv "$file" demos/ 2>/dev/null || echo "  Skipped $file (not tracked or already exists)"
    fi
done

# Move test files (that aren't already in tests/)
for file in test_*.py SIMPLE_*_VALIDATION.py COMPLETE_*_VALIDATION.py *_test.py; do
    if [ -f "$file" ]; then
        git mv "$file" tests/ 2>/dev/null || echo "  Skipped $file (not tracked or already exists)"
    fi
done

# Move analysis/check scripts
for file in analyze_*.py check_*.py validate_*.py verify_*.py; do
    if [ -f "$file" ]; then
        git mv "$file" scripts/ 2>/dev/null || echo "  Skipped $file (not tracked or already exists)"
    fi
done

# Move simulation scripts
for file in simulate_*.py sim_*.py; do
    if [ -f "$file" ]; then
        git mv "$file" scripts/ 2>/dev/null || echo "  Skipped $file (not tracked or already exists)"
    fi
done

# Move optimizer scripts (keep core library files, move examples)
for file in *_optimizer.py *_optimize.py optimize_*.py; do
    if [ -f "$file" ]; then
        # Check if it's a library file or example
        if grep -q "if __name__ == .__main__." "$file"; then
            git mv "$file" examples/ 2>/dev/null || echo "  Skipped $file (not tracked or already exists)"
        fi
    fi
done

# Move run/execution scripts
for file in run_*.py; do
    if [ -f "$file" ]; then
        git mv "$file" scripts/ 2>/dev/null || echo "  Skipped $file (not tracked or already exists)"
    fi
done

# Move example/benchmark scripts
for file in benchmark_*.py example_*.py quick_*.py; do
    if [ -f "$file" ]; then
        git mv "$file" examples/ 2>/dev/null || echo "  Skipped $file (not tracked or already exists)"
    fi
done

echo ""
echo "Organization complete!"
echo ""
echo "Files moved to:"
echo "  demos/     - demo_*.py files"
echo "  tests/     - test_*.py and validation files"
echo "  scripts/   - analyze_*.py, check_*.py, run_*.py, simulate_*.py"
echo "  examples/  - benchmark_*.py, example_*.py, quick_*.py, optimizer examples"
echo ""
echo "Remaining root .py files (if any):"
find . -maxdepth 1 -name "*.py" -type f | wc -l
