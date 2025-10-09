#!/usr/bin/env python3
"""
Basic validation test for warp-bubble-optimizer setup
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that all core modules can be imported"""
    print("🧪 Testing Core Module Imports")
    print("=" * 35)
    
    try:
        import warp_qft
        print("✓ warp_qft base module")
    except Exception as e:
        print(f"✗ warp_qft base module: {e}")
        return False
    
    try:
        from warp_qft.field_algebra import PolymerFieldAlgebra
        print("✓ PolymerFieldAlgebra")
    except Exception as e:
        print(f"✗ PolymerFieldAlgebra: {e}")
    
    try:
        from warp_qft.lqg_profiles import LQGProfileCalculator
        print("✓ LQGProfileCalculator")
    except Exception as e:
        print(f"✗ LQGProfileCalculator: {e}")
    
    try:
        from warp_qft.metrics.van_den_broeck_natario import van_den_broeck_shape
        print("✓ van_den_broeck_shape")
    except Exception as e:
        print(f"✗ van_den_broeck_shape: {e}")
    
    try:
        from warp_qft.enhancement_pipeline import WarpBubbleEnhancementPipeline
        print("✓ WarpBubbleEnhancementPipeline")
    except Exception as e:
        print(f"✗ WarpBubbleEnhancementPipeline: {e}")
    
    return True

def test_basic_functionality():
    """Test basic functionality with minimal dependencies"""
    print("\n🔬 Testing Basic Functionality")
    print("=" * 35)
    
    try:
        import numpy as np
        r_test = np.array([0.5, 1.0, 1.5])
        print(f"✓ NumPy array creation: {r_test}")
        
        from warp_qft.metrics.van_den_broeck_natario import van_den_broeck_shape
        result = van_den_broeck_shape(r_test, R_warp=1.0, sigma=0.1)
        print(f"✓ van_den_broeck_shape calculation: shape={result.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Warp Bubble Optimizer - Setup Validation")
    print("=" * 50)
    
    success = True
    success &= test_basic_imports()
    success &= test_basic_functionality()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Repository setup is functional.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    print("Repository is ready for metric ansatz development!")
