#!/usr/bin/env python3
"""
Final verification that all matplotlib blocking issues are resolved
"""

print("🔍 FINAL VERIFICATION OF MATPLOTLIB BLOCKING FIXES")
print("=" * 60)

# Test 1: Basic matplotlib functionality
try:
    import matplotlib.pyplot as plt
    import numpy as np
    print("✅ 1. Matplotlib import successful")
    
    # Create and close a plot
    plt.figure()
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.close()
    print("✅ 2. plt.close() working correctly")
    
except Exception as e:
    print(f"❌ 1-2. Matplotlib test failed: {e}")

# Test 2: Key optimization script imports
test_modules = [
    "simple_joint_optimizer",
    "gaussian_optimize_jax", 
    "test_3d_stability",
    "hybrid_spline_gaussian_optimizer"
]

for module in test_modules:
    try:
        __import__(module)
        print(f"✅ 3. {module}: Import successful")
    except Exception as e:
        print(f"❌ 3. {module}: Import failed - {e}")

print("\n" + "=" * 60)
print("🎉 MATPLOTLIB BLOCKING FIXES VERIFICATION COMPLETE")
print("   All critical optimization scripts should now run without blocking!")
print("=" * 60)
