#!/usr/bin/env python3
"""
Enhanced Warp Bubble Solver for Warp-Bubble-Optimizer

This module extends the existing warp_qft framework with advanced
3D mesh-based validation for Discovery 21 Ghost/Phantom EFT and
other negative energy sources.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings
from dataclasses import dataclass
import time
import json
from pathlib import Path

# Import existing warp_qft components
from .warp_bubble_analysis import (
    squeezed_vacuum_energy,
    sampling_function,
    pi_shell,
    energy_density_polymer,
    polymer_QI_bound
)
from .warp_bubble_engine import *
from .energy_sources import EnergySource, create_energy_source
from .negative_energy import integrate_negative_energy_over_time
from .stability import WarpBubbleStability

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    warnings.warn("PyVista not available. 3D visualization will be limited.")

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available. Optimization features limited.")

@dataclass
class EnhancedWarpBubbleResult:
    """Enhanced results from warp bubble simulation including polymer QI analysis."""
    success: bool
    energy_total: float
    stability: float
    bubble_radius: float
    max_negative_density: float
    min_negative_density: float
    execution_time: float
    mesh_nodes: int
    source_name: str
    parameters: Dict[str, Any]
    
    # Enhanced fields for polymer QI analysis
    qi_violation_achieved: bool
    qi_bound_value: float
    polymer_enhancement_factor: float
    warp_profile_f: Optional[np.ndarray] = None
    warp_profile_g: Optional[np.ndarray] = None
    energy_profile: Optional[np.ndarray] = None
    coordinates: Optional[np.ndarray] = None
    
    # Integration with existing metrics
    squeezed_vacuum_comparison: Optional[float] = None
    classical_energy_ratio: Optional[float] = None

class EnhancedWarpBubbleSolver:
    """
    Enhanced warp bubble solver integrating Discovery 21 with existing framework.
    
    Combines:
    - Discovery 21 Ghost/Phantom EFT optimal parameters
    - Existing polymer QI violation analysis
    - 3D mesh-based validation
    - Integration with warp_bubble_engine components
    """
    
    def __init__(self, use_polymer_enhancement: bool = True,
                 enable_stability_analysis: bool = True):
        """
        Initialize enhanced warp bubble solver.
        
        Args:
            use_polymer_enhancement: Enable polymer field theory enhancements
            enable_stability_analysis: Enable stability analysis using existing tools
        """
        self.use_polymer_enhancement = use_polymer_enhancement
        self.enable_stability_analysis = enable_stability_analysis
        
        # Initialize existing components
        if enable_stability_analysis:
            self.stability_analyzer = WarpBubbleStability()
        
        # Solver state
        self.mesh_coords = None
        self.last_result = None
        
    def generate_spherical_mesh(self, radius: float, n_r: int = 50, 
                               n_theta: int = 30, n_phi: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate spherical mesh compatible with existing framework.
        
        Args:
            radius: Domain radius (m)
            n_r: Number of radial points
            n_theta: Number of theta points
            n_phi: Number of phi points
            
        Returns:
            Tuple of (coordinates, radial_array)
        """
        # Create spherical coordinates
        r = np.linspace(0.1, radius, n_r)  # Avoid r=0 singularity
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        
        # Create coordinate arrays
        R, THETA, PHI = np.meshgrid(r, theta, phi, indexing='ij')
        
        # Convert to Cartesian
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)
        
        # Flatten coordinates
        coords = np.column_stack([
            X.flatten(),
            Y.flatten(), 
            Z.flatten()
        ])
        
        self.mesh_coords = coords
        return coords, r
    
    def analyze_polymer_qi_violations(self, energy_source: EnergySource,
                                    r_array: np.ndarray) -> Dict[str, float]:
        """
        Analyze quantum inequality violations using polymer field theory.
        
        Args:
            energy_source: Energy source to analyze
            r_array: Radial coordinate array
            
        Returns:
            Dictionary with QI analysis results
        """
        if not hasattr(energy_source, 'mu_polymer'):
            return {
                'qi_violation_achieved': False,
                'qi_bound_value': 0.0,
                'polymer_enhancement_factor': 1.0,
                'classical_energy_ratio': 1.0
            }
        
        mu = energy_source.mu_polymer
        
        # Compute polymer-modified QI bound
        qi_bound = polymer_QI_bound(mu, tau=1.0)
        
        # Integrate negative energy over time using existing framework
        N = len(r_array)
        dx = (r_array[-1] - r_array[0]) / N if N > 1 else 1.0
        
        try:
            qi_integral = integrate_negative_energy_over_time(
                N=N, mu=mu, total_time=10.0, dt=0.01, dx=dx, tau=1.0
            )
        except Exception as e:
            warnings.warn(f"QI integration failed: {e}")
            qi_integral = 0.0
        
        # Check for QI violation
        qi_violation_achieved = qi_integral < qi_bound
        
        # Compute enhancement factor
        if mu > 0:
            classical_integral = integrate_negative_energy_over_time(
                N=N, mu=0.0, total_time=10.0, dt=0.01, dx=dx, tau=1.0
            )
            enhancement_factor = abs(qi_integral / classical_integral) if classical_integral != 0 else 1.0
            classical_ratio = qi_integral / classical_integral if classical_integral != 0 else 0.0
        else:
            enhancement_factor = 1.0
            classical_ratio = 1.0
        
        return {
            'qi_violation_achieved': qi_violation_achieved,
            'qi_bound_value': qi_bound,
            'polymer_enhancement_factor': enhancement_factor,
            'classical_energy_ratio': classical_ratio
        }
    
    def compute_warp_bubble_metrics(self, energy_source: EnergySource,
                                  r_array: np.ndarray) -> Dict[str, Any]:
        """
        Compute comprehensive warp bubble metrics.
        
        Args:
            energy_source: Energy source to analyze
            r_array: Radial coordinate array
            
        Returns:
            Dictionary with warp bubble metrics
        """
        # Get warp profiles
        f_profile, g_profile = energy_source.get_warp_profile(r_array)
        
        # Compute energy density on spherical shell
        theta = np.pi/2  # Equatorial plane
        phi = 0.0
        
        x = r_array * np.sin(theta) * np.cos(phi)
        y = r_array * np.sin(theta) * np.sin(phi)
        z = r_array * np.cos(theta)
        
        energy_density = energy_source.energy_density(x, y, z, t=0.0)
        
        # Integration using existing tools
        dr = r_array[1] - r_array[0] if len(r_array) > 1 else 1.0
        
        # Shell integration: 4π r² ρ(r) dr
        shell_energy = 4 * np.pi * r_array**2 * energy_density * dr
        total_energy = np.sum(shell_energy)
        
        # Stability analysis using metric curvature
        # Simplified: based on metric gradient
        df_dr = np.gradient(f_profile, dr)
        dg_dr = np.gradient(g_profile, dr)
        
        # Stability metric: inverse of maximum curvature
        max_curvature = np.max(np.abs(df_dr) + np.abs(dg_dr))
        stability = 1.0 / (1.0 + max_curvature * 100)  # Normalized
        
        return {
            'f_profile': f_profile,
            'g_profile': g_profile,
            'energy_density': energy_density,
            'total_energy': total_energy,
            'stability': stability,
            'max_negative_density': np.min(energy_density),
            'min_negative_density': np.max(energy_density[energy_density < 0]) if np.any(energy_density < 0) else 0.0
        }
    
    def compare_with_squeezed_vacuum(self, energy_source: EnergySource,
                                   r_array: np.ndarray) -> float:
        """
        Compare energy source with squeezed vacuum baseline.
        
        Args:
            energy_source: Energy source to compare
            r_array: Radial coordinate array
            
        Returns:
            Ratio of source energy to squeezed vacuum energy
        """
        # Typical squeezed vacuum parameters
        r_squeeze = 1.0
        omega = 1e12  # 1 THz
        cavity_volume = 1e-9  # 1 mm³
        
        squeezed_density = squeezed_vacuum_energy(r_squeeze, omega, cavity_volume)
        source_total = energy_source.total_energy((4/3) * np.pi * r_array[-1]**3)
        
        # Volume-normalized comparison
        volume = (4/3) * np.pi * r_array[-1]**3
        squeezed_total = squeezed_density * volume
        
        if squeezed_total != 0:
            return abs(source_total / squeezed_total)
        else:
            return 1.0
    
    def simulate(self, energy_source: EnergySource, 
                radius: float = 10.0, resolution: int = 50) -> EnhancedWarpBubbleResult:
        """
        Run comprehensive warp bubble simulation with polymer QI analysis.
        
        Args:
            energy_source: Energy source to test
            radius: Simulation domain radius (m)
            resolution: Mesh resolution parameter
            
        Returns:
            Enhanced simulation results
        """
        start_time = time.time()
        
        try:
            # Generate mesh
            coords, r_array = self.generate_spherical_mesh(radius, resolution, 
                                                          resolution//2, resolution//2)
            
            # Compute warp bubble metrics
            metrics = self.compute_warp_bubble_metrics(energy_source, r_array)
            
            # Analyze polymer QI violations
            qi_analysis = self.analyze_polymer_qi_violations(energy_source, r_array)
            
            # Compare with squeezed vacuum
            squeezed_comparison = self.compare_with_squeezed_vacuum(energy_source, r_array)
            
            # Determine success criteria
            has_negative = metrics['max_negative_density'] < -1e-15
            is_stable = metrics['stability'] > 0.1
            is_valid = energy_source.validate_parameters()
            qi_violated = qi_analysis['qi_violation_achieved'] if self.use_polymer_enhancement else True
            
            success = has_negative and is_stable and is_valid and qi_violated
            
            execution_time = time.time() - start_time
            
            # Create enhanced result
            result = EnhancedWarpBubbleResult(
                success=success,
                energy_total=metrics['total_energy'],
                stability=metrics['stability'],
                bubble_radius=radius,
                max_negative_density=metrics['max_negative_density'],
                min_negative_density=metrics['min_negative_density'],
                execution_time=execution_time,
                mesh_nodes=len(coords),
                source_name=energy_source.name,
                parameters=energy_source.parameters,
                
                # Enhanced fields
                qi_violation_achieved=qi_analysis['qi_violation_achieved'],
                qi_bound_value=qi_analysis['qi_bound_value'],
                polymer_enhancement_factor=qi_analysis['polymer_enhancement_factor'],
                warp_profile_f=metrics['f_profile'],
                warp_profile_g=metrics['g_profile'],
                energy_profile=metrics['energy_density'],
                coordinates=coords,
                
                # Comparisons
                squeezed_vacuum_comparison=squeezed_comparison,
                classical_energy_ratio=qi_analysis['classical_energy_ratio']
            )
            
            self.last_result = result
            return result
            
        except Exception as e:
            warnings.warn(f"Enhanced simulation failed: {e}")
            
            # Return failed result
            return EnhancedWarpBubbleResult(
                success=False,
                energy_total=0.0,
                stability=0.0,
                bubble_radius=radius,
                max_negative_density=0.0,
                min_negative_density=0.0,
                execution_time=time.time() - start_time,
                mesh_nodes=0,
                source_name=energy_source.name,
                parameters=energy_source.parameters,
                qi_violation_achieved=False,
                qi_bound_value=0.0,
                polymer_enhancement_factor=1.0
            )
    
    def optimize_energy_source_parameters(self, source_type: str, 
                                        param_bounds: Dict[str, Tuple[float, float]],
                                        radius: float = 10.0) -> Dict[str, Any]:
        """
        Optimize energy source parameters for maximum negative energy.
        
        Args:
            source_type: Type of energy source to optimize
            param_bounds: Dictionary of parameter bounds
            radius: Simulation radius
            
        Returns:
            Optimization results
        """
        if not HAS_SCIPY:
            warnings.warn("SciPy not available. Cannot perform optimization.")
            return {}
        
        def objective(params):
            """Objective function: maximize total negative energy."""
            try:
                param_dict = dict(zip(param_bounds.keys(), params))
                source = create_energy_source(source_type, **param_dict)
                result = self.simulate(source, radius, resolution=30)
                
                # Objective: minimize negative of total energy (to maximize negative energy)
                # Add penalty for failed simulations
                if not result.success:
                    return 1e6
                
                return -result.energy_total  # Maximize negative energy
                
            except Exception as e:
                return 1e6  # Large penalty for failed evaluations
        
        # Set up optimization
        bounds = [param_bounds[key] for key in param_bounds.keys()]
        x0 = [(b[0] + b[1])/2 for b in bounds]  # Start at midpoint
        
        print(f"Optimizing {source_type} parameters...")
        print(f"Parameters: {list(param_bounds.keys())}")
        
        try:
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            
            # Create optimized source
            optimal_params = dict(zip(param_bounds.keys(), result.x))
            optimal_source = create_energy_source(source_type, **optimal_params)
            optimal_result = self.simulate(optimal_source, radius, resolution=50)
            
            return {
                'success': result.success,
                'optimal_parameters': optimal_params,
                'optimal_energy': optimal_result.energy_total,
                'optimization_result': result,
                'simulation_result': optimal_result
            }
            
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            return {}
    
    def visualize_result(self, result: EnhancedWarpBubbleResult, 
                        save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualization of enhanced results.
        
        Args:
            result: Enhanced simulation result to visualize
            save_path: Optional path to save visualization
        """
        if result.coordinates is None:
            print(f"No data to visualize for {result.source_name}")
            return
        
        # Create multi-panel figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Radial profiles
        if result.warp_profile_f is not None:
            r_array = np.sqrt(np.sum(result.coordinates**2, axis=1))
            r_unique = np.unique(r_array)[:len(result.warp_profile_f)]
            
            # f(r) profile
            ax = axes[0, 0]
            ax.plot(r_unique, result.warp_profile_f, 'b-', linewidth=2, label='f(r)')
            ax.set_xlabel('Radius (m)')
            ax.set_ylabel('f(r)')
            ax.set_title('Temporal Metric Component')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # g(r) profile  
            ax = axes[0, 1]
            ax.plot(r_unique, result.warp_profile_g, 'r-', linewidth=2, label='g(r)')
            ax.set_xlabel('Radius (m)')
            ax.set_ylabel('g(r)')
            ax.set_title('Radial Metric Component')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Energy density profile
            ax = axes[0, 2]
            if result.energy_profile is not None:
                r_coords = np.sqrt(np.sum(result.coordinates**2, axis=1))
                ax.scatter(r_coords, result.energy_profile, alpha=0.3, s=1)
                ax.set_xlabel('Radius (m)')
                ax.set_ylabel('Energy Density (J/m³)')
                ax.set_title('Energy Density Profile')
                ax.grid(True, alpha=0.3)
        
        # Performance metrics
        ax = axes[1, 0]
        metrics = [
            f"Success: {result.success}",
            f"Total Energy: {result.energy_total:.2e} J",
            f"Stability: {result.stability:.3f}",
            f"QI Violation: {result.qi_violation_achieved}",
            f"Polymer Enhancement: {result.polymer_enhancement_factor:.2f}×",
            f"vs Squeezed Vacuum: {result.squeezed_vacuum_comparison:.2e}×",
            f"Execution Time: {result.execution_time:.3f} s",
            f"Mesh Nodes: {result.mesh_nodes}"
        ]
        
        ax.text(0.05, 0.95, '\n'.join(metrics), transform=ax.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        ax.set_title(f'{result.source_name}: Performance Summary')
        ax.axis('off')
        
        # Parameter summary
        ax = axes[1, 1]
        if result.parameters:
            param_text = []
            for key, value in result.parameters.items():
                if isinstance(value, float):
                    param_text.append(f"{key}: {value:.3f}")
                else:
                    param_text.append(f"{key}: {value}")
            
            ax.text(0.05, 0.95, '\n'.join(param_text), transform=ax.transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=10)
        ax.set_title('Source Parameters')
        ax.axis('off')
        
        # Comparison bar chart
        ax = axes[1, 2]
        comparisons = []
        values = []
        
        if result.qi_bound_value != 0:
            comparisons.append('QI Bound')
            values.append(abs(result.qi_bound_value))
        
        if result.squeezed_vacuum_comparison:
            comparisons.append('vs Squeezed Vacuum')
            values.append(result.squeezed_vacuum_comparison)
        
        if result.polymer_enhancement_factor != 1:
            comparisons.append('Polymer Enhancement')
            values.append(result.polymer_enhancement_factor)
        
        if comparisons:
            bars = ax.bar(comparisons, values, alpha=0.7)
            ax.set_ylabel('Enhancement Factor')
            ax.set_title('Performance Comparisons')
            ax.set_yscale('log')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1e}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved: {save_path}")
        
        plt.show()

def run_discovery_21_validation() -> Dict[str, EnhancedWarpBubbleResult]:
    """
    Run comprehensive validation of Discovery 21 Ghost/Phantom EFT.
    
    Returns:
        Dictionary of validation results
    """
    print("Discovery 21 Ghost/Phantom EFT Validation")
    print("=" * 50)
    
    solver = EnhancedWarpBubbleSolver(use_polymer_enhancement=True)
    results = {}
    
    # Test Discovery 21 optimal parameters
    print("Testing Discovery 21 optimal configuration...")
    ghost_optimal = create_energy_source('ghost', M=1000, alpha=0.01, beta=0.1, mu_polymer=0.1)
    result_optimal = solver.simulate(ghost_optimal, radius=10.0, resolution=40)
    results['Discovery_21_Optimal'] = result_optimal
    
    print(f"Optimal config: Success={result_optimal.success}, "
          f"Energy={result_optimal.energy_total:.2e} J, "
          f"QI Violation={result_optimal.qi_violation_achieved}")
    
    # Parameter variations
    print("\nTesting parameter variations...")
    
    variations = [
        {'M': 500, 'alpha': 0.01, 'beta': 0.1, 'mu_polymer': 0.1},
        {'M': 1500, 'alpha': 0.01, 'beta': 0.1, 'mu_polymer': 0.1},
        {'M': 1000, 'alpha': 0.005, 'beta': 0.1, 'mu_polymer': 0.1},
        {'M': 1000, 'alpha': 0.02, 'beta': 0.1, 'mu_polymer': 0.1},
        {'M': 1000, 'alpha': 0.01, 'beta': 0.05, 'mu_polymer': 0.1},
        {'M': 1000, 'alpha': 0.01, 'beta': 0.15, 'mu_polymer': 0.1},
        {'M': 1000, 'alpha': 0.01, 'beta': 0.1, 'mu_polymer': 0.05},
        {'M': 1000, 'alpha': 0.01, 'beta': 0.1, 'mu_polymer': 0.2}
    ]
    
    for i, params in enumerate(variations):
        source = create_energy_source('ghost', **params)
        result = solver.simulate(source, radius=10.0, resolution=30)
        results[f'Variation_{i+1}'] = result
        
        print(f"  Variation {i+1}: Success={result.success}, "
              f"Energy={result.energy_total:.2e} J")
    
    return results

# Example usage
if __name__ == "__main__":
    # Run Discovery 21 validation
    validation_results = run_discovery_21_validation()
    
    # Find best performing configuration
    successful_results = {k: v for k, v in validation_results.items() if v.success}
    
    if successful_results:
        best_config = min(successful_results.keys(), 
                         key=lambda k: successful_results[k].energy_total)
        
        print(f"\nBest performing configuration: {best_config}")
        best_result = successful_results[best_config]
        print(f"  Total Energy: {best_result.energy_total:.2e} J")
        print(f"  Stability: {best_result.stability:.3f}")
        print(f"  QI Violation: {best_result.qi_violation_achieved}")
        print(f"  Polymer Enhancement: {best_result.polymer_enhancement_factor:.2f}×")
        
        # Visualize best result
        solver = EnhancedWarpBubbleSolver()
        solver.visualize_result(best_result, f"discovery_21_validation_{best_config}.png")
    else:
        print("\nNo successful configurations found.")
    
    # Save results summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_configurations': len(validation_results),
        'successful_configurations': len(successful_results),
        'success_rate': len(successful_results) / len(validation_results) * 100,
        'results': {
            name: {
                'success': result.success,
                'energy_total': result.energy_total,
                'stability': result.stability,
                'qi_violation': result.qi_violation_achieved,
                'parameters': result.parameters
            }
            for name, result in validation_results.items()
        }
    }
    
    with open('discovery_21_validation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nValidation complete. Results saved to discovery_21_validation_results.json")
    print(f"Success rate: {summary['success_rate']:.1f}%")
