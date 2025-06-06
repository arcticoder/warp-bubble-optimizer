#!/usr/bin/env python3
"""
ACCELERATED GAUSSIAN OPTIMIZATION TEST SUITE

Tests all acceleration strategies for the Gaussian ansatz optimizer:
1. 4-Gaussian vs 3-Gaussian comparison
2. Vectorized integration benchmark
3. Parallel DE performance
4. Hybrid ansatz validation  
5. CMA-ES vs DE comparison (if available)
6. Physics constraint validation

Usage: python test_accelerated_gaussian.py
"""
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from gaussian_optimize import *

def test_vectorized_integration():
    """
    Test vectorized integration performance vs scipy.quad
    """
    print("🔬 INTEGRATION ACCELERATION TEST")
    print("-" * 40)
    
    # Test parameters (reasonable 4-Gaussian configuration)
    test_params = [
        0.4, 0.2, 0.08,   # Gaussian 1: A=0.4, r0=0.2, σ=0.08
        0.3, 0.5, 0.12,   # Gaussian 2: A=0.3, r0=0.5, σ=0.12
        0.2, 0.7, 0.10,   # Gaussian 3: A=0.2, r0=0.7, σ=0.10
        0.1, 0.9, 0.08    # Gaussian 4: A=0.1, r0=0.9, σ=0.08
    ]
    
    # Test with different grid sizes
    grid_sizes = [200, 500, 800, 1000, 1500]
    timing_results = {}
    
    for N in grid_sizes:
        # Update grid
        global N_points, r_grid, dr, vol_weights
        N_points = N
        r_grid = np.linspace(0.0, R, N_points)
        dr = r_grid[1] - r_grid[0]
        vol_weights = 4.0 * np.pi * r_grid**2
        
        # Time vectorized method
        start = time.time()
        for _ in range(100):  # Multiple runs for accurate timing
            energy_fast = E_negative_gauss_fast(test_params)
        fast_time = (time.time() - start) / 100
        
        timing_results[N] = {
            'fast_time': fast_time,
            'energy': energy_fast
        }
        
        print(f"N={N:4d}: {fast_time*1000:.3f} ms/call, E₋={energy_fast:.3e} J")
    
    # Time scipy.quad for comparison (single call - it's very slow)
    print("\n📊 scipy.quad comparison:")
    start = time.time()
    energy_slow = E_negative_gauss_slow(test_params)
    slow_time = time.time() - start
    print(f"scipy.quad: {slow_time*1000:.1f} ms/call, E₋={energy_slow:.3e} J")
    
    # Calculate speedup
    fast_time_800 = timing_results[800]['fast_time']
    speedup = slow_time / fast_time_800
    print(f"\n⚡ SPEEDUP: {speedup:.1f}× faster with N=800 grid")
    
    return timing_results

def test_parallel_vs_sequential():
    """
    Test parallel DE performance vs sequential
    """
    print("\n🔬 PARALLEL OPTIMIZATION TEST")
    print("-" * 40)
    
    # Set test parameters
    global mu0, G_geo, M_gauss
    mu0 = 1e-6
    G_geo = 1e-5
    M_gauss = 4
    
    bounds = get_optimization_bounds()
    
    # Test sequential DE
    print("🔸 Sequential DE...")
    start = time.time()
    result_seq = differential_evolution(
        objective_gauss,
        bounds,
        strategy='best1bin',
        maxiter=50,  # Reduced for testing
        popsize=8,
        tol=1e-6,
        seed=42,
        workers=1  # Force sequential
    )
    seq_time = time.time() - start
    
    # Test parallel DE
    print("🔸 Parallel DE...")
    start = time.time()
    result_par = differential_evolution(
        objective_gauss,
        bounds,
        strategy='best1bin',
        maxiter=50,
        popsize=8,
        tol=1e-6,
        seed=42,
        workers=-1  # Use all cores
    )
    par_time = time.time() - start
    
    print(f"\nSequential: {seq_time:.1f}s, E₋={result_seq.fun:.3e}")
    print(f"Parallel:   {par_time:.1f}s, E₋={result_par.fun:.3e}")
    print(f"⚡ Parallel speedup: {seq_time/par_time:.2f}×")
    
    return seq_time, par_time

def test_ansatz_comparison():
    """
    Compare 3-Gaussian vs 4-Gaussian performance
    """
    print("\n🔬 ANSATZ COMPARISON TEST")
    print("-" * 40)
    
    global mu0, G_geo, M_gauss
    mu0 = 1e-6
    G_geo = 1e-5
    
    results = {}
    
    for M in [3, 4]:
        print(f"\n🔸 Testing {M}-Gaussian ansatz...")
        M_gauss = M
        
        start = time.time()
        result = optimize_gaussian_ansatz_fast(mu0, G_geo)
        total_time = time.time() - start
        
        if result:
            results[f'{M}-Gaussian'] = {
                'energy': result['energy_J'],
                'time': total_time,
                'params': result['params']
            }
            
            print(f"   ✅ E₋ = {result['energy_J']:.3e} J")
            print(f"   ⏱️  Time: {total_time:.1f}s")
        else:
            print(f"   ❌ Optimization failed")
    
    # Compare results
    if '3-Gaussian' in results and '4-Gaussian' in results:
        improvement = abs(results['4-Gaussian']['energy']) / abs(results['3-Gaussian']['energy'])
        print(f"\n📊 4-Gaussian improvement: {improvement:.3f}× better energy")
        
        time_ratio = results['4-Gaussian']['time'] / results['3-Gaussian']['time']
        print(f"📊 Time ratio: {time_ratio:.2f}× longer for 4-Gaussian")
    
    return results

def test_hybrid_ansatz():
    """
    Test hybrid polynomial+Gaussian ansatz
    """
    print("\n🔬 HYBRID ANSATZ TEST")
    print("-" * 40)
    
    global mu0, G_geo
    mu0 = 1e-6
    G_geo = 1e-5
    
    # Test hybrid bounds and functions
    hybrid_bounds = get_hybrid_bounds(M_gauss_hybrid=2)
    print(f"🔸 Hybrid ansatz: 2 Gaussians + polynomial transition")
    print(f"   Parameters: {len(hybrid_bounds)} (vs {3*4}=12 for 4-Gaussian)")
    
    # Test a simple hybrid configuration
    test_hybrid_params = [
        0.1,    # r0: flat core radius
        0.6,    # r1: start of Gaussian region  
        -1.0,   # a1: linear polynomial coefficient
        0.5,    # a2: quadratic polynomial coefficient
        0.4, 0.7, 0.15,  # Gaussian 1
        0.3, 0.9, 0.12   # Gaussian 2
    ]
    
    # Test hybrid functions
    r_test = np.linspace(0, R, 100)
    f_vals = f_hybrid_vectorized(r_test, test_hybrid_params, enable_hybrid=True)
    fp_vals = f_hybrid_prime_vectorized(r_test, test_hybrid_params, enable_hybrid=True)
    
    print(f"🔸 Function evaluation test:")
    print(f"   f(0) = {f_vals[0]:.3f} (should ≈ 1)")
    print(f"   f(R) = {f_vals[-1]:.3f} (should ≈ 0)")
    print(f"   max|f'| = {np.max(np.abs(fp_vals)):.3f}")
    
    # Calculate energy
    energy_hybrid = E_negative_hybrid(test_hybrid_params)
    print(f"   E₋ = {energy_hybrid:.3e} J")
    
    # Plot hybrid profile
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(r_test, f_vals, 'b-', linewidth=2, label='Hybrid f(r)')
    plt.axvline(x=test_hybrid_params[0], color='r', linestyle='--', alpha=0.7, label='r₀')
    plt.axvline(x=test_hybrid_params[1], color='g', linestyle='--', alpha=0.7, label='r₁')
    plt.xlabel('r (m)')
    plt.ylabel('f(r)')
    plt.title('Hybrid Ansatz Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2) 
    plt.plot(r_test, fp_vals, 'r-', linewidth=2, label="f'(r)")
    plt.axvline(x=test_hybrid_params[0], color='r', linestyle='--', alpha=0.7)
    plt.axvline(x=test_hybrid_params[1], color='g', linestyle='--', alpha=0.7)
    plt.xlabel('r (m)')
    plt.ylabel("f'(r)")
    plt.title('Hybrid Ansatz Derivative')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_hybrid_ansatz.png', dpi=300, bbox_inches='tight')
    print(f"📊 Hybrid ansatz profile saved to 'test_hybrid_ansatz.png'")
    
    return energy_hybrid

def test_cma_es_availability():
    """
    Test if CMA-ES is available and compare with DE
    """
    print("\n🔬 CMA-ES AVAILABILITY TEST")
    print("-" * 40)
    
    try:
        import cma
        print("✅ CMA-ES available")
        
        # Quick CMA-ES test
        global mu0, G_geo, M_gauss
        mu0 = 1e-6
        G_geo = 1e-5
        M_gauss = 3  # Use 3-Gaussian for faster test
        
        bounds = get_optimization_bounds()
        
        print("🔸 Testing CMA-ES optimization...")
        start = time.time()
        result_cma = optimize_with_cma_es(bounds, sigma0=0.1)
        cma_time = time.time() - start
        
        if result_cma:
            print(f"   ✅ CMA-ES: E₋ = {result_cma['energy_J']:.3e} J in {cma_time:.1f}s")
        else:
            print("   ❌ CMA-ES optimization failed")
        
        return True, result_cma
        
    except ImportError:
        print("❌ CMA-ES not available. Install with: pip install cma")
        return False, None

def test_physics_constraints():
    """
    Test physics-informed constraints (QI, smoothness, etc.)
    """
    print("\n🔬 PHYSICS CONSTRAINTS TEST")
    print("-" * 40)
    
    # Test parameters that might violate constraints
    test_cases = {
        'good': [0.4, 0.2, 0.08, 0.3, 0.5, 0.12, 0.2, 0.7, 0.10, 0.1, 0.9, 0.08],
        'spiky': [1.0, 0.5, 0.01, 0.8, 0.5, 0.01, 0.6, 0.5, 0.01, 0.4, 0.5, 0.01],  # Very narrow
        'violation': [2.0, 0.0, 0.1, 1.5, 0.5, 0.1, 1.0, 0.8, 0.1, 0.5, 0.9, 0.1]   # Large amplitudes
    }
    
    global mu0, G_geo, M_gauss
    mu0 = 1e-6
    G_geo = 1e-5
    M_gauss = 4
    
    for case_name, params in test_cases.items():
        print(f"\n🔸 Testing '{case_name}' parameters:")
        
        # QI constraint
        rho0 = rho_eff_gauss_vectorized(0.0, params)
        qi_bound = - (hbar * np.sinc(mu0 / np.pi)) / (12.0 * np.pi * tau**2)
        qi_violation = max(0.0, -(rho0 - qi_bound))
        
        # Boundary conditions
        f0 = f_gaussian_vectorized(0.0, params)
        fR = f_gaussian_vectorized(R, params)
        
        # Curvature penalty
        curv_penalty = curvature_penalty(params, enable_hybrid=False)
        
        print(f"   ρ(0) = {rho0:.3e}, QI bound = {qi_bound:.3e}")
        print(f"   QI violation: {qi_violation:.3e}")
        print(f"   f(0) = {f0:.3f}, f(R) = {fR:.3f}")
        print(f"   Curvature penalty: {curv_penalty:.3e}")
        
        # Total penalty
        total_penalty = penalty_gauss(params)
        print(f"   Total penalty: {total_penalty:.3e}")

def run_comprehensive_test():
    """
    Run all acceleration tests
    """
    print("🚀 ACCELERATED GAUSSIAN OPTIMIZATION TEST SUITE")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: Integration speedup
    print("\n" + "="*60)
    test_results['integration'] = test_vectorized_integration()
    
    # Test 2: Parallel vs sequential
    print("\n" + "="*60)
    seq_time, par_time = test_parallel_vs_sequential()
    test_results['parallel'] = {'seq_time': seq_time, 'par_time': par_time}
    
    # Test 3: Ansatz comparison
    print("\n" + "="*60)
    test_results['ansatz'] = test_ansatz_comparison()
    
    # Test 4: Hybrid ansatz
    print("\n" + "="*60)
    test_results['hybrid'] = test_hybrid_ansatz()
    
    # Test 5: CMA-ES availability
    print("\n" + "="*60)
    cma_available, cma_result = test_cma_es_availability()
    test_results['cma_es'] = {'available': cma_available, 'result': cma_result}
    
    # Test 6: Physics constraints
    print("\n" + "="*60)
    test_physics_constraints()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    if 'integration' in test_results:
        fast_time = test_results['integration'][800]['fast_time']
        print(f"✅ Vectorized integration: {fast_time*1000:.2f} ms/call (N=800)")
    
    if 'parallel' in test_results:
        speedup = test_results['parallel']['seq_time'] / test_results['parallel']['par_time']
        print(f"✅ Parallel DE speedup: {speedup:.2f}×")
    
    if 'ansatz' in test_results and '4-Gaussian' in test_results['ansatz']:
        energy_4g = test_results['ansatz']['4-Gaussian']['energy']
        print(f"✅ 4-Gaussian best energy: {energy_4g:.3e} J")
    
    if 'hybrid' in test_results:
        energy_hybrid = test_results['hybrid']
        print(f"✅ Hybrid ansatz energy: {energy_hybrid:.3e} J")
    
    print(f"✅ CMA-ES available: {test_results['cma_es']['available']}")
    
    # Save test results
    with open('acceleration_test_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in test_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                                           for k, v in value.items()}
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2)
    
    print("💾 Test results saved to 'acceleration_test_results.json'")
    
    return test_results

if __name__ == "__main__":
    # Run comprehensive test suite
    results = run_comprehensive_test()
    
    print("\n🏁 ACCELERATION TEST SUITE COMPLETE")
    print("\n🔮 RECOMMENDATIONS:")
    
    if 'integration' in results:
        print("1. Use N=800-1000 grid points for optimal speed/accuracy balance")
    
    if 'parallel' in results:
        speedup = results['parallel']['seq_time'] / results['parallel']['par_time']
        if speedup > 1.5:
            print("2. Parallel DE provides significant speedup - always use workers=-1")
        else:
            print("2. Parallel DE speedup limited - CPU may be single-core or memory-bound")
    
    if 'ansatz' in results:
        if '4-Gaussian' in results['ansatz'] and '3-Gaussian' in results['ansatz']:
            improvement = abs(results['ansatz']['4-Gaussian']['energy']) / abs(results['ansatz']['3-Gaussian']['energy'])
            if improvement > 1.1:
                print("3. 4-Gaussian provides meaningful improvement - recommended")
            else:
                print("3. 4-Gaussian improvement marginal - 3-Gaussian may be sufficient")
    
    if results['cma_es']['available']:
        print("4. CMA-ES available - try for difficult optimization problems")
    else:
        print("4. Consider installing CMA-ES: pip install cma")
    
    print("5. Always validate final results with 3+1D stability test")
