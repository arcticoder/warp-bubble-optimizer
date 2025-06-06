#!/usr/bin/env python3
"""
Test script for the ACCELERATED gaussian_optimize.py

This script demonstrates usage of the accelerated 3-Gaussian optimization
and allows quick testing of individual components.
"""

if __name__ == "__main__":
    from gaussian_optimize import (
        test_single_optimization, 
        run_quick_scan,
        benchmark_integration_methods,
        monitor_optimization_performance,
        optimize_gaussian_ansatz_robust
    )
    
    print("🧪 TESTING ACCELERATED GAUSSIAN OPTIMIZATION")
    print("=" * 50)
    
    # Test 1: Robust single optimization
    print("\n1️⃣ Testing robust optimization...")
    try:
        result = optimize_gaussian_ansatz_robust(mu_val=1e-6, G_geo_val=1e-5)
        if result:
            print(f"   ✅ Success: E₋ = {result['energy_J']:.3e} J")
            print(f"   Strategy: {result['strategy']}")
            print(f"   Time: {result['total_time']:.1f}s")
        else:
            print("   ❌ All strategies failed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Integration method benchmark
    print("\n2️⃣ Benchmarking integration methods...")
    benchmark_integration_methods([0.5, 0.3, 0.1, 0.3, 0.6, 0.2, 0.2, 0.9, 0.1])
    
    # Test 3: Performance monitoring
    print("\n3️⃣ Performance monitoring...")
    monitor_optimization_performance()
    
    print("\n✅ Core tests completed!")
    print("\nTo run full optimization:")
    print("   python gaussian_optimize.py")
