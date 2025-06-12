"""
Unified Gauge Theory Polymer Corrections for Warp Bubble Metrics

Implementation of GUT-polymer modifications to warp bubble metrics and
ANEC integral calculations for stability analysis.

This module integrates the unified gauge polymerization framework with 
warp bubble metrics, replacing all occurrences of curvature Φ with the 
polymer-corrected form: Φ + sin(μ F^a_μν)/μ.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from scipy.integrate import quad, simpson

# Import from unified_gut_polymerization if available
try:
    # First, try to add the path to access the package if needed
    import os
    import sys
    gut_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "unified-gut-polymerization"))
    if gut_path not in sys.path:
        sys.path.append(gut_path)
        
    from unified_gut_polymerization.core import UnifiedGaugePolymerization, GUTConfig
    from unified_gut_polymerization.running_coupling import RunningCouplingInstanton
    GUT_POLYMER_AVAILABLE = True
except ImportError:
    GUT_POLYMER_AVAILABLE = False


class GUTPolymerMetricCorrections:
    """
    Implementation of GUT-polymer corrections to warp bubble metrics.
    
    This class modifies warp bubble metric functions by replacing curvature Φ
    with a polymer-corrected version that includes GUT field strength terms.
    """
    
    def __init__(self, 
                group: str = 'SU5',
                polymer_scale_mu: float = 0.1,
                field_strength: float = 1.0,
                energy_scale: float = 1e12):  # GeV
        """
        Initialize GUT-polymer metric corrections.
        
        Args:
            group: GUT symmetry group ('SU5', 'SO10', or 'E6')
            polymer_scale_mu: Polymer scale parameter
            field_strength: Gauge field strength parameter (dimensionless)
            energy_scale: Energy scale in GeV (for running coupling)
        """
        self.group = group
        self.mu = polymer_scale_mu
        self.F_strength = field_strength
        self.energy_scale = energy_scale
        
        # Initialize GUT polymer tools if available
        if GUT_POLYMER_AVAILABLE:
            self.gut_config = GUTConfig(group=group, polymer_length=polymer_scale_mu)
            self.gut_polymer = UnifiedGaugePolymerization(self.gut_config)
            self.running_coupling = RunningCouplingInstanton(group=group)
        else:
            print("Warning: unified_gut_polymerization package not available.")
            print("Using simplified polymer corrections.")
    
    def polymer_correction_factor(self, mu: float, F: float) -> float:
        """
        Calculate sin(μF)/μ polymer correction factor.
        
        Args:
            mu: Polymer scale parameter
            F: Field strength value
            
        Returns:
            Polymer correction factor
        """
        if abs(mu * F) < 1e-10:
            # Taylor expansion for small values
            return F * (1 - (mu * F)**2 / 6 + (mu * F)**4 / 120)
        return np.sin(mu * F) / mu
    
    def polymer_modified_curvature(self, 
                                 curvature_phi: float, 
                                 r: float = 0.0,
                                 theta: float = 0.0,
                                 phi: float = 0.0) -> float:
        """
        Modify curvature with polymer gauge field correction.
        
        The modification replaces:
        Φ → Φ + sin(μ F^a_μν)/μ
        
        Args:
            curvature_phi: Original curvature scalar Φ
            r, theta, phi: Optional spacetime coordinates
            
        Returns:
            Modified curvature value
        """        # Get running gauge coupling at the energy scale
        if GUT_POLYMER_AVAILABLE:
            alpha = self.running_coupling.running_coupling(self.energy_scale)
            print(f"Using unified_gut_polymerization package for {self.group}")
            print(f"Coupling α = {alpha:.6f} at energy {self.energy_scale/1e9:.1f} TeV")
            
            # Get polymer metrics from the package
            if hasattr(self.gut_polymer, 'compute_polymer_factor'):
                poly_factor = self.gut_polymer.compute_polymer_factor(self.F_strength)
                print(f"GUT polymer factor: {poly_factor:.6f}")
        else:
            # Approximate values if package not available
            gut_couplings = {'SU5': 1/25, 'SO10': 1/24, 'E6': 1/23}
            alpha = gut_couplings.get(self.group, 1/24)
            print(f"Using fallback coupling for {self.group}: α = {alpha:.6f}")
        
        # Calculate effective field strength (simplified model)
        # F^a_μν ∼ sqrt(α) * F_strength
        effective_F = np.sqrt(alpha) * self.F_strength
        
        # Apply polymer modification
        correction = self.polymer_correction_factor(self.mu, effective_F)
        
        return curvature_phi + correction
    
    def modify_alcubierre_warp_function(self, 
                                      original_function: Callable, 
                                      **kwargs) -> Callable:
        """
        Return a modified version of a warp function with polymer corrections.
        
        Args:
            original_function: Original warp function f(r)
            **kwargs: Additional parameters to pass to original function
            
        Returns:
            Modified warp function with polymer corrections
        """
        def modified_function(r, *args, **kw):
            # Combine kwargs from init and call
            all_kwargs = {**kwargs, **kw}
            
            # Get original value
            original_value = original_function(r, *args, **all_kwargs)
            
            # Convert warp function to approximate curvature scalar
            approx_curvature = original_value * 10.0  # Scale factor for demonstration
            
            # Apply polymer modification to curvature
            modified_curvature = self.polymer_modified_curvature(approx_curvature, r)
            
            # Convert back to warp function
            return modified_curvature / 10.0  # Inverse of scale factor
        
        return modified_function


class ANECIntegralCalculator:
    """
    Calculator for the Averaged Null Energy Condition (ANEC) integral 
    with unified GUT-polymer corrections.
    
    The ANEC integral is:
    ∫_γ T_μν k^μ k^ν ds
    
    where T_μν is the stress-energy tensor, k^μ is a null vector,
    and the integration is along a null geodesic γ.
    """
    
    def __init__(self, 
                gut_polymer: GUTPolymerMetricCorrections,
                num_points: int = 1000,
                integration_range: float = 10.0):
        """
        Initialize ANEC integral calculator.
        
        Args:
            gut_polymer: GUT polymer metric corrections object
            num_points: Number of points for numerical integration
            integration_range: Integration range (-range to +range)
        """
        self.gut_polymer = gut_polymer
        self.num_points = num_points
        self.range = integration_range
    
    def stress_energy_tensor(self, 
                           r: float, 
                           field_strength: float = 1.0) -> Dict[str, float]:
        """
        Compute polymer-modified stress-energy tensor components.
        
        For a gauge field F^a_μν, the stress tensor is:
        T_μν = (1/4π) [F^a_μα F^a_ν^α - (1/4)g_μν F^a_αβ F^a^αβ]
        
        Args:
            r: Radial coordinate
            field_strength: Gauge field strength
            
        Returns:
            Dictionary with tensor components
        """
        # Calculate the field strength tensor components (simplified)
        F_01 = field_strength * np.exp(-r**2)  # Example field profile
        F_02 = 0.5 * field_strength * np.exp(-r**2)
        F_12 = 0.3 * field_strength * np.exp(-r**2/2)
        
        # Apply polymer corrections
        F_01_poly = self.gut_polymer.polymer_modified_curvature(F_01, r)
        F_02_poly = self.gut_polymer.polymer_modified_curvature(F_02, r)
        F_12_poly = self.gut_polymer.polymer_modified_curvature(F_12, r)
        
        # F^2 invariant (simplified)
        F_squared = F_01_poly**2 - F_02_poly**2 - F_12_poly**2
        
        # Compute stress tensor components (simplified model)
        # T_μν = (1/4π) [F_μα F_ν^α - (1/4)g_μν F^2]
        prefactor = 1.0 / (4.0 * np.pi)
        
        T00 = prefactor * (F_01_poly**2 + F_02_poly**2 + 0.25 * F_squared)
        T01 = prefactor * (F_01_poly * F_12_poly)
        T02 = prefactor * (F_02_poly * F_12_poly)
        T11 = prefactor * (F_01_poly**2 - 0.25 * F_squared)
        T22 = prefactor * (F_02_poly**2 - 0.25 * F_squared)
        T12 = prefactor * (F_01_poly * F_02_poly)
        
        return {
            'T00': T00,
            'T01': T01,
            'T02': T02,
            'T11': T11,
            'T22': T22,
            'T12': T12
        }
    
    def null_vector(self, r: float) -> np.ndarray:
        """
        Generate a null vector field for ANEC integral.
        
        Args:
            r: Radial coordinate
            
        Returns:
            Null vector k^μ = (1, 1, 0, 0) (time + radial)
        """
        # Simple null vector (time + radial)
        return np.array([1.0, 1.0, 0.0, 0.0])
    
    def compute_anec_integral(self, field_strength: float = 1.0) -> float:
        """
        Compute the ANEC integral for polymer-modified stress tensor.
        
        Args:
            field_strength: Gauge field strength parameter
            
        Returns:
            ANEC integral value
        """
        # Set up integration grid
        r_values = np.linspace(-self.range, self.range, self.num_points)
        dr = 2.0 * self.range / (self.num_points - 1)
        
        # Initialize integral
        anec_integral = 0.0
        
        # Integrate T_μν k^μ k^ν along the null geodesic
        for r in r_values:
            # Get stress tensor at this point
            T = self.stress_energy_tensor(r, field_strength)
            
            # Get null vector
            k = self.null_vector(r)
            
            # Compute T_μν k^μ k^ν (simplified to 2D for clarity)
            # Full calculation would use proper tensor contraction
            stress_projection = T['T00']*k[0]*k[0] + 2*T['T01']*k[0]*k[1] + T['T11']*k[1]*k[1]
            
            # Add to integral
            anec_integral += stress_projection * dr
        
        return anec_integral
    
    def stability_condition_h_infinity(self, 
                                     field_strength_range: np.ndarray,
                                     reference_integral: Optional[float] = None) -> Dict:
        """
        Compute H-infinity stability margins for ANEC integral.
        
        Args:
            field_strength_range: Array of field strength values to scan
            reference_integral: Optional reference ANEC value (stable classical solution)
            
        Returns:
            Dictionary with stability analysis
        """
        # Compute ANEC integrals for each field strength
        anec_values = np.array([
            self.compute_anec_integral(F) for F in field_strength_range
        ])
        
        # If no reference provided, use the minimum value
        if reference_integral is None:
            reference_integral = np.min(anec_values)
        
        # Compute stability margins
        stability_margins = anec_values / reference_integral if reference_integral != 0 else np.inf
        
        # Find optimal field strength
        if len(field_strength_range) > 0:
            optimal_idx = np.argmin(stability_margins)
            optimal_field = field_strength_range[optimal_idx]
            optimal_margin = stability_margins[optimal_idx]
        else:
            optimal_field = 0.0
            optimal_margin = np.inf
        
        return {
            'field_strength_values': field_strength_range,
            'anec_values': anec_values,
            'stability_margins': stability_margins,
            'optimal_field_strength': optimal_field,
            'optimal_margin': optimal_margin,
            'is_stable': np.any(stability_margins < 1.0),
            'reference_integral': reference_integral
        }
    
    def plot_stability_margins(self, analysis_results: Dict):
        """
        Plot stability margins from H-infinity analysis.
        
        Args:
            analysis_results: Results from stability_condition_h_infinity
        """
        field_values = analysis_results['field_strength_values']
        margins = analysis_results['stability_margins']
        
        plt.figure(figsize=(10, 6))
        plt.plot(field_values, margins, 'b-', linewidth=2)
        plt.axhline(y=1.0, color='r', linestyle='--', label='Stability Threshold')
        
        plt.xlabel('Field Strength Parameter')
        plt.ylabel('H-infinity Stability Margin')
        plt.title('Warp Bubble Stability Analysis with GUT-Polymer Corrections')
        plt.grid(True)
        plt.legend()
        
        # Mark optimal point
        optimal_field = analysis_results['optimal_field_strength']
        optimal_margin = analysis_results['optimal_margin']
        plt.plot([optimal_field], [optimal_margin], 'ro', markersize=10)
        plt.annotate(f'Optimal: {optimal_margin:.3f}', 
                    xy=(optimal_field, optimal_margin),
                    xytext=(optimal_field+0.1, optimal_margin+0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.tight_layout()
        plt.savefig('gut_polymer_stability_margins.png', dpi=300)
        
        return plt
