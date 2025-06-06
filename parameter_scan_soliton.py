#!/usr/bin/env python3
"""
Parameter scan for soliton ansatz optimization
Explores different values of μ (polymer parameter) and R_ext/R_int ratios
"""

import numpy as np
import json
from scipy.optimize import minimize, differential_evolution
from soliton_optimize import (
    objective_soliton, E_negative_soliton, bounds, init_params,
    M_soliton, beta_back, hbar, c, G, tau, v, R
)

def parameter_scan_soliton():
    """Scan over polymer parameter μ and geometric ratio"""
    
    # Parameter ranges to scan
    mu_values = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 2e-5]
    R_ratios = [1e-4, 1e-5, 1e-6, 1e-7]  # Different geometric reduction factors
    
    results = []
    
    print("🔍 COMPREHENSIVE PARAMETER SCAN")
    print("=" * 60)
    print(f"Scanning {len(mu_values)} μ values × {len(R_ratios)} R_ratios = {len(mu_values)*len(R_ratios)} configurations")
    print()
    
    for i, mu_val in enumerate(mu_values):
        for j, R_ratio in enumerate(R_ratios):
            print(f"[{i*len(R_ratios)+j+1:2d}/{len(mu_values)*len(R_ratios)}] μ={mu_val:.1e}, R_ratio={R_ratio:.1e}", end=" → ")
            
            # Update global parameters
            import soliton_optimize
            soliton_optimize.mu0 = mu_val
            soliton_optimize.G_geo = R_ratio
            
            try:
                # Quick optimization with fewer iterations for scanning
                res = differential_evolution(
                    objective_soliton,
                    bounds=bounds,
                    maxiter=20,
                    popsize=15,
                    tol=1e-4,
                    seed=42
                )
                
                if res.success:
                    energy = E_negative_soliton(res.x)
                    print(f"E₋ = {energy:.2e} J")
                    
                    result = {
                        'mu': mu_val,
                        'R_ratio': R_ratio,
                        'energy_J': energy,
                        'parameters': res.x.tolist(),
                        'success': True
                    }
                else:
                    print("FAILED")
                    result = {
                        'mu': mu_val,
                        'R_ratio': R_ratio,
                        'energy_J': 0.0,
                        'parameters': [],
                        'success': False
                    }
                    
            except Exception as e:
                print(f"ERROR: {e}")
                result = {
                    'mu': mu_val,
                    'R_ratio': R_ratio,
                    'energy_J': 0.0,
                    'parameters': [],
                    'success': False,
                    'error': str(e)
                }
            
            results.append(result)
    
    print("\n" + "=" * 60)
    print("📊 SCAN RESULTS SUMMARY")
    print("-" * 30)
    
    # Find best configuration
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_result = min(successful_results, key=lambda x: abs(x['energy_J']))
        print(f"🏆 BEST CONFIGURATION:")
        print(f"   μ = {best_result['mu']:.1e}")
        print(f"   R_ratio = {best_result['R_ratio']:.1e}")
        print(f"   E₋ = {best_result['energy_J']:.3e} J")
        
        # Performance analysis
        energies = [abs(r['energy_J']) for r in successful_results if r['energy_J'] != 0]
        if energies:
            print(f"\n📈 PERFORMANCE STATISTICS:")
            print(f"   Best energy: {min(energies):.3e} J")
            print(f"   Worst energy: {max(energies):.3e} J")
            print(f"   Range factor: {max(energies)/min(energies):.1f}×")
            print(f"   Successful configs: {len(successful_results)}/{len(results)}")
    else:
        print("❌ No successful optimizations found")
    
    # Save detailed results
    scan_data = {
        'scan_parameters': {
            'mu_values': mu_values,
            'R_ratios': R_ratios,
            'total_configurations': len(results)
        },
        'results': results,
        'best_result': best_result if successful_results else None,
        'enhancement_factors': {
            'beta_backreaction': beta_back,
            'base_polymer_scale': 1e-6,
            'bubble_radius': R
        }
    }
    
    with open('soliton_parameter_scan.json', 'w') as f:
        json.dump(scan_data, f, indent=2)
    
    print(f"\n💾 Results saved to 'soliton_parameter_scan.json'")
    
    return results, best_result if successful_results else None

def create_scan_visualization(results):
    """Create visualization of scan results"""
    try:
        import matplotlib.pyplot as plt
        
        # Extract data for plotting
        successful_results = [r for r in results if r['success'] and r['energy_J'] != 0]
        if not successful_results:
            print("No data to plot")
            return
            
        mu_vals = [r['mu'] for r in successful_results]
        R_vals = [r['R_ratio'] for r in successful_results]
        energies = [abs(r['energy_J']) for r in successful_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Energy vs μ
        ax1.loglog(mu_vals, energies, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Polymer Parameter μ')
        ax1.set_ylabel('|E₋| (J)')
        ax1.set_title('Soliton Energy vs Polymer Parameter')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Energy vs R_ratio
        ax2.loglog(R_vals, energies, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Geometric Reduction Factor')
        ax2.set_ylabel('|E₋| (J)')
        ax2.set_title('Soliton Energy vs Geometric Factor')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('soliton_parameter_scan_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Scan visualization saved as 'soliton_parameter_scan_results.png'")
        
    except ImportError:
        print("Matplotlib not available for visualization")
    except Exception as e:
        print(f"Visualization error: {e}")

if __name__ == "__main__":
    results, best_config = parameter_scan_soliton()
    
    if best_config:
        print(f"\n🎯 OPTIMAL SOLITON CONFIGURATION IDENTIFIED")
        print(f"   Consider using μ = {best_config['mu']:.1e}")
        print(f"   and R_ratio = {best_config['R_ratio']:.1e}")
        print(f"   for best soliton performance")
        
        create_scan_visualization(results)
    
    print("\n🏁 Parameter scan complete!")
