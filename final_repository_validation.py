#!/usr/bin/env python3
"""
Final Validation Script for warp-bubble-optimizer Repository

This script performs a comprehensive validation of the setup and readiness
for advanced metric ansatz development.
"""

import os
import sys
import subprocess
from pathlib import Path

def header(text):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")

def check_directory_structure():
    """Check that all required directories and files exist."""
    header("CHECKING REPOSITORY STRUCTURE")
    
    required_dirs = [
        "src/warp_qft",
        "src/warp_qft/metrics", 
        "docs",
        "scripts"
    ]
    
    required_files = [
        "src/warp_qft/__init__.py",
        "src/warp_qft/field_algebra.py",
        "src/warp_qft/metrics/van_den_broeck_natario.py",
        "requirements.txt",
        "setup.py",
        "README.md"
    ]
    
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✓ All required directories present")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files present")
    
    return True

def check_imports():
    """Test critical imports."""
    header("TESTING IMPORTS")
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    try:
        import warp_qft
        print("✓ warp_qft package imports successfully")
        
        from warp_qft.field_algebra import PolymerFieldAlgebra
        print("✓ PolymerFieldAlgebra imports successfully")
        
        from warp_qft.metrics.van_den_broeck_natario import van_den_broeck_shape
        print("✓ van_den_broeck_shape imports successfully")
        
        from warp_qft.metric_ansatz_development import MetricAnsatzBuilder
        print("✓ MetricAnsatzBuilder imports successfully")
        
        from warp_qft.variational_optimizer import VariationalOptimizer
        print("✓ VariationalOptimizer imports successfully")
        
        from warp_qft.numerical_integration import WarpBubbleIntegrator
        print("✓ WarpBubbleIntegrator imports successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def check_numerical_functionality():
    """Test numerical calculations."""
    header("TESTING NUMERICAL FUNCTIONALITY")
    
    sys.path.insert(0, 'src')
    
    try:
        import numpy as np
        from warp_qft.metrics.van_den_broeck_natario import van_den_broeck_shape
        
        # Test Van den Broeck shape function
        r_values = np.array([0.05, 0.1, 0.5, 1.0, 1.5])
        R_int, R_ext = 1.0, 0.1
        
        f_values = van_den_broeck_shape(r_values, R_int, R_ext)
        print(f"✓ Van den Broeck shape function calculated: {f_values}")
        
        # Basic validation
        assert f_values[0] == 1.0, "Interior should be 1.0"
        assert f_values[-1] == 0.0, "Exterior should be 0.0"
        assert all(0 <= f <= 1 for f in f_values), "Values should be in [0,1]"
        
        print("✓ Van den Broeck function validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Numerical functionality error: {e}")
        return False

def check_dependencies():
    """Check that all dependencies are installed."""
    header("CHECKING DEPENDENCIES")
    
    required_packages = {
        'numpy': '1.21.0',
        'scipy': '1.7.0', 
        'matplotlib': '3.5.0',
        'sympy': '1.9.0',
        'pytest': '6.0'
    }
    
    try:
        for package, min_version in required_packages.items():
            __import__(package)
            print(f"✓ {package} is installed")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def run_tests():
    """Run the test suite."""
    header("RUNNING TEST SUITE")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'test_repository.py', '-v'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ All tests passed!")
            return True
        else:
            print(f"❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Run comprehensive validation."""
    print("Warp Bubble Optimizer - Repository Validation")
    print("=" * 60)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Dependencies", check_dependencies), 
        ("Imports", check_imports),
        ("Numerical Functionality", check_numerical_functionality),
        ("Test Suite", run_tests)
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check failed with error: {e}")
            results[name] = False
    
    # Summary
    header("VALIDATION SUMMARY")
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{name:25} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 Repository validation SUCCESSFUL!")
        print("The warp-bubble-optimizer repository is ready for:")
        print("• Advanced metric ansatz development")
        print("• Novel geometric optimization")
        print("• Variational energy minimization")
        print("• LQG-QI enhanced warp bubble design")
        print("\nNext steps:")
        print("1. Begin systematic metric ansatz exploration")
        print("2. Implement novel soliton-like profiles")
        print("3. Apply variational optimization to minimize energy")
        print("4. Integrate latest theoretical advances")
    else:
        print(f"\n❌ Repository validation FAILED")
        print("Please address the issues above before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
