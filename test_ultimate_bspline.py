#!/usr/bin/env python3
"""
Test script for the ultimate B-spline optimizer
"""
import sys
import traceback

print("🧪 Testing Ultimate B-Spline Optimizer Import")
print("=" * 50)

try:
    print("1. Testing basic imports...")
    import numpy as np
    print("   ✅ NumPy imported successfully")
    
    import matplotlib.pyplot as plt
    print("   ✅ Matplotlib imported successfully")
    
    try:
        import jax
        import jax.numpy as jnp
        print("   ✅ JAX imported successfully")
    except ImportError:
        print("   ❌ JAX not available")
        
    try:
        import cma
        print("   ✅ CMA-ES imported successfully")
    except ImportError:
        print("   ❌ CMA-ES not available")
    
    print("\n2. Testing ultimate_bspline_optimizer import...")
    from ultimate_bspline_optimizer import UltimateBSplineOptimizer
    print("   ✅ UltimateBSplineOptimizer imported successfully")
    
    print("\n3. Testing optimizer initialization...")
    optimizer = UltimateBSplineOptimizer(
        n_control_points=8,  # Smaller for testing
        R_bubble=100.0,
        stability_penalty_weight=1e4,
        surrogate_assisted=False,  # Disable for testing
        verbose=True
    )
    print("   ✅ Optimizer initialized successfully")
    
    print("\n4. Testing parameter initialization...")
    params = optimizer.initialize_parameters('physics_informed')
    print(f"   ✅ Parameters initialized: {len(params)} parameters")
    print(f"   μ = {params[0]:.3f}, G_geo = {params[1]:.3e}")
    
    print("\n5. Testing shape function...")
    r_test = jnp.linspace(0, 150, 10)
    f_test = optimizer.shape_function(r_test, params)
    print(f"   ✅ Shape function computed: min={jnp.min(f_test):.3f}, max={jnp.max(f_test):.3f}")
    
    print("\n6. Testing objective function...")
    obj_val = optimizer.objective_function(params)
    print(f"   ✅ Objective function: {obj_val:.3e}")
    
    print("\n✅ ALL TESTS PASSED!")
    print("🚀 Ultimate B-Spline Optimizer is ready for use!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
