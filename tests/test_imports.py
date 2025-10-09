#!/usr/bin/env python3
"""
Simple import test for warp-bubble-optimizer
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test basic imports to verify setup."""
    try:
        print("Testing imports...")
        
        # Test basic package import
        import warp_qft
        print("✓ warp_qft package imported")
        
        # Test specific modules
        from warp_qft.field_algebra import PolymerFieldAlgebra
        print("✓ PolymerFieldAlgebra imported")
        
        from warp_qft.metrics.van_den_broeck_natario import van_den_broeck_shape
        print("✓ van_den_broeck_shape imported")
        
        # Test array functionality
        import numpy as np
        r_values = np.array([0.5, 1.0, 1.5, 2.0])
        f_values = van_den_broeck_shape(r_values, 1.0, 0.1)
        print(f"✓ Van den Broeck shape function calculated: {f_values}")
        
        print("\nAll imports successful! Repository is ready for development.")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
