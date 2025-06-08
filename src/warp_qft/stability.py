"""
Stability Analysis and Ford-Roman Bounds

Analysis of quantum inequality violations and stability of negative energy states.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def ford_roman_bounds(energy_density: float, spatial_scale: float, 
                     temporal_scale: Optional[float] = None) -> Dict:
    """
    Compute Ford-Roman quantum inequality bounds.
    
    The Ford-Roman inequality constrains negative energy:
    ∫ ρ(t) f(t) dt ≥ -C / (τ²)
    
    where f(t) is a sampling function, τ is its characteristic width,
    and C is a constant depending on the field type.
    
    Args:
        energy_density: Peak negative energy density
        spatial_scale: Characteristic spatial scale
        temporal_scale: Characteristic temporal scale (if None, estimated)
        
    Returns:
        Dictionary with Ford-Roman bounds and analysis
    """
    if temporal_scale is None:
        # Estimate based on light-crossing time
        temporal_scale = spatial_scale  # c = 1 units
    
    # Classical Ford-Roman bound for scalar field
    # The constant C ≈ ℏc/(12π) for a massless scalar field
    hbar_c = 1.0  # Natural units
    C_scalar = hbar_c / (12 * np.pi)
    
    # Maximum negative energy integral allowed
    max_negative_integral = -C_scalar / (temporal_scale**2)
    
    # Convert to maximum density for given spatial extent
    max_negative_density = max_negative_integral / spatial_scale
    
    # Violation occurs when magnitude of negative energy density exceeds bound
    violates_bound = abs(energy_density) > abs(max_negative_density) if energy_density < 0 else False
    
    # Violation factor - higher means more severe violation
    violation_factor = abs(energy_density / max_negative_density) if max_negative_density != 0 else np.inf
    
    return {
        "ford_roman_bound": max_negative_density,
        "max_negative_integral": max_negative_integral,
        "spatial_scale": spatial_scale,
        "temporal_scale": temporal_scale,
        "energy_density": energy_density,
        "violates_bound": violates_bound,
        "violation_factor": violation_factor,
        "bound_type": "classical_ford_roman"
    }


def polymer_modified_bounds(energy_density: float, spatial_scale: float,
                          polymer_scale: float, temporal_scale: Optional[float] = None) -> Dict:
    """
    Compute polymer-modified Ford-Roman bounds.
    
    In the polymer representation, the commutation relations are modified,
    which can relax the Ford-Roman inequalities under certain conditions.
    
    Args:
        energy_density: Peak negative energy density
        spatial_scale: Characteristic spatial scale  
        polymer_scale: Polymer parameter μ̄
        temporal_scale: Characteristic temporal scale
        
    Returns:
        Dictionary with polymer-modified bounds
    """
    # Get classical bounds first
    classical_bounds = ford_roman_bounds(energy_density, spatial_scale, temporal_scale)
    
    if polymer_scale == 0:
        return {**classical_bounds, "bound_type": "polymer_classical_limit"}
    
    # Polymer modification factor - enhanced by sinc function
    # For μ̄ > 0, this factor makes the bound more negative
    polymer_factor = 1.0 + (polymer_scale * np.pi)**2 / 6  # Second-order correction
    
    # Make bound more negative with polymer effects
    max_negative_density_polymer = classical_bounds["ford_roman_bound"] * polymer_factor
    max_negative_integral_polymer = classical_bounds["max_negative_integral"] * polymer_factor
    
    # Check violation against polymer-modified bound
    violates_bound = abs(energy_density) > abs(max_negative_density_polymer) if energy_density < 0 else False
    
    return {
        "ford_roman_bound": max_negative_density_polymer,
        "max_negative_integral": max_negative_integral_polymer,
        "spatial_scale": spatial_scale,
        "temporal_scale": temporal_scale,
        "polymer_scale": polymer_scale,
        "energy_density": energy_density,
        "violates_bound": violates_bound,
        "violation_factor": abs(energy_density / max_negative_density_polymer) if max_negative_density_polymer != 0 else np.inf,
        "enhancement_factor": polymer_factor,
        "bound_type": "polymer_modified"
    }


def violation_duration(energy_density: float, spatial_scale: float,
                      polymer_scale: float = 0.0) -> Dict:
    """
    Compute how long negative energy can persist before violating bounds.
    
    Args:
        energy_density: Negative energy density
        spatial_scale: Spatial extent of negative energy region
        polymer_scale: Polymer parameter μ̄
        
    Returns:
        Duration analysis results
    """
    if energy_density >= 0:
        return {
            "max_duration": np.inf,
            "classical_duration": np.inf,
            "polymer_duration": np.inf,
            "violation_type": "no_negative_energy"
        }
    
    # Classical maximum duration (from Ford-Roman bound)
    # τ_max ~ √(C / |ρ| Δx)
    hbar_c = 1.0
    C_scalar = hbar_c / (12 * np.pi)
    
    classical_duration = np.sqrt(C_scalar / (abs(energy_density) * spatial_scale))
    
    # Polymer-modified duration
    if polymer_scale > 0:
        sinc_factor = np.sinc(polymer_scale / np.pi)
        C_polymer = sinc_factor * hbar_c / (12 * np.pi)
        
        # Additional discretization effects
        discretization_factor = 1 + 0.5 * polymer_scale**2
        C_effective = C_polymer * discretization_factor
        
        polymer_duration = np.sqrt(C_effective / (abs(energy_density) * spatial_scale))
    else:
        polymer_duration = classical_duration
    
    # Enhancement factor
    enhancement = polymer_duration / classical_duration
    
    return {
        "classical_duration": classical_duration,
        "polymer_duration": polymer_duration, 
        "max_duration": polymer_duration,
        "enhancement_factor": enhancement,
        "energy_density": energy_density,
        "spatial_scale": spatial_scale,
        "polymer_scale": polymer_scale,
        "violation_type": "duration_limited"
    }


def stability_phase_diagram(polymer_scale_range: Tuple[float, float] = (0, 1.0),
                           energy_range: Tuple[float, float] = (-2.0, 0),
                           spatial_scale: float = 0.1,
                           resolution: int = 50) -> Dict:
    """
    Generate a stability phase diagram in (μ̄, ρ_neg) space.
    
    Args:
        polymer_scale_range: Range of polymer parameters to scan
        energy_range: Range of negative energy densities
        spatial_scale: Fixed spatial scale
        resolution: Grid resolution for each axis
        
    Returns:
        Phase diagram data
    """
    mu_min, mu_max = polymer_scale_range
    rho_min, rho_max = energy_range
    
    mu_grid = np.linspace(mu_min, mu_max, resolution)
    rho_grid = np.linspace(rho_min, rho_max, resolution)
    
    MU, RHO = np.meshgrid(mu_grid, rho_grid)
    
    # Compute stability for each point
    classical_stable = np.zeros_like(MU, dtype=bool)
    polymer_stable = np.zeros_like(MU, dtype=bool)
    enhancement_map = np.zeros_like(MU)
    
    for i in range(resolution):
        for j in range(resolution):
            mu = MU[i, j]
            rho = RHO[i, j]
            
            # Classical bounds
            classical_bounds = ford_roman_bounds(rho, spatial_scale)
            classical_stable[i, j] = not classical_bounds["violates_bound"]
            
            # Polymer bounds
            polymer_bounds = polymer_modified_bounds(rho, spatial_scale, mu)
            polymer_stable[i, j] = not polymer_bounds["violates_bound"]
            enhancement_map[i, j] = polymer_bounds["enhancement_factor"]
    
    # Identify regions
    stable_regions = {
        "classically_stable": classical_stable,
        "polymer_stable": polymer_stable,
        "polymer_only_stable": polymer_stable & ~classical_stable,
        "unstable": ~polymer_stable
    }
    
    return {
        "mu_grid": mu_grid,
        "rho_grid": rho_grid,
        "MU": MU,
        "RHO": RHO,
        "stable_regions": stable_regions,
        "enhancement_map": enhancement_map,
        "spatial_scale": spatial_scale,
        "resolution": resolution
    }


def quantum_pressure_analysis(energy_density: float, polymer_scale: float,
                             spatial_scale: float) -> Dict:
    """
    Analyze quantum pressure effects in the polymer representation.
    
    The discrete nature of the polymer representation can create
    quantum pressure effects that help stabilize negative energy.
    
    Args:
        energy_density: Energy density value
        polymer_scale: Polymer parameter μ̄  
        spatial_scale: Characteristic length scale
        
    Returns:
        Quantum pressure analysis
    """
    if polymer_scale == 0:
        return {
            "quantum_pressure": 0.0,
            "stabilization_strength": 0.0,
            "pressure_type": "classical_limit"
        }
    
    # Quantum pressure from discretization uncertainty
    # Δp ~ ℏ/Δx, where Δx ~ polymer_scale in natural units
    uncertainty_pressure = 1.0 / (polymer_scale**2)
    
    # Polymer-specific corrections
    # The sin(μ̄p)/μ̄ factor creates effective pressure
    polymer_pressure = uncertainty_pressure * np.sin(polymer_scale)**2
    
    # Stabilization strength (dimensionless)
    stabilization = polymer_pressure / (abs(energy_density) + 1e-10)
    
    # Critical polymer scale for stabilization
    critical_scale = np.sqrt(abs(energy_density)) if energy_density < 0 else 0
    
    return {
        "quantum_pressure": polymer_pressure,
        "uncertainty_pressure": uncertainty_pressure,
        "stabilization_strength": stabilization,
        "critical_polymer_scale": critical_scale,
        "stabilization_achieved": polymer_scale > critical_scale,
        "pressure_type": "polymer_quantum"
    }


def stability_analysis(negative_energy: float, spatial_scale: float, 
                      polymer_scale: float) -> Dict:
    """
    Perform stability analysis for given parameters.
    
    Args:
        negative_energy: Peak negative energy density
        spatial_scale: Spatial extent of negative energy region
        polymer_scale: Polymer parameter μ̄
        
    Returns:
        Stability analysis results
    """
    # Classical bounds
    classical_bounds = ford_roman_bounds(negative_energy, spatial_scale)
    
    # Polymer-modified bounds
    polymer_bounds = polymer_modified_bounds(negative_energy, spatial_scale, polymer_scale)
    
    # Violation duration
    duration_info = violation_duration(negative_energy, spatial_scale, polymer_scale)
    
    return {
        "classical_bounds": classical_bounds,
        "polymer_bounds": polymer_bounds,
        "duration_info": duration_info
    }


def lqg_modified_bounds(energy_density: float, spatial_scale: float,
                       flight_time: float, C_lqg: Optional[float] = None,
                       temporal_scale: Optional[float] = None) -> Dict:
    """
    Compute LQG-modified quantum inequality bounds.
    
    The LQG-modified bound is stricter than Ford-Roman:
    E_- ≥ -C_LQG / T^4
    
    where C_LQG << C (classical Ford-Roman constant) and T is the sampling/flight time.
    This provides a more restrictive lower bound on negative energy.
    
    Args:
        energy_density: Peak negative energy density
        spatial_scale: Characteristic spatial scale
        flight_time: Sampling/flight time duration T
        C_lqg: LQG modification constant (if None, use default)
        temporal_scale: Characteristic temporal scale (if None, use flight_time)
        
    Returns:
        Dictionary with LQG-modified bounds and analysis
    """
    if temporal_scale is None:
        temporal_scale = flight_time
    
    # Default LQG constant - much smaller than classical Ford-Roman
    if C_lqg is None:
        hbar_c = 1.0  # Natural units
        C_classical = hbar_c / (12 * np.pi)  # Classical Ford-Roman constant
        # LQG modification makes bound stricter by factor of ~100-1000
        C_lqg = C_classical / 100.0  # C_LQG << C
    
    # LQG-modified bound: E_- ≥ -C_LQG / T^4
    max_negative_integral = -C_lqg / (flight_time**4)
    
    # Convert to maximum density for given spatial extent
    max_negative_density = max_negative_integral / spatial_scale
    
    # Violation occurs when magnitude of negative energy density exceeds bound
    violates_bound = abs(energy_density) > abs(max_negative_density) if energy_density < 0 else False
    
    # Violation factor - higher means more severe violation
    violation_factor = abs(energy_density / max_negative_density) if max_negative_density != 0 else np.inf
    
    # Compare with classical Ford-Roman bound
    classical_bounds = ford_roman_bounds(energy_density, spatial_scale, temporal_scale)
    strictness_factor = abs(max_negative_density / classical_bounds["ford_roman_bound"]) if classical_bounds["ford_roman_bound"] != 0 else np.inf
    
    return {
        "lqg_bound": max_negative_density,
        "max_negative_integral": max_negative_integral,
        "spatial_scale": spatial_scale,
        "temporal_scale": temporal_scale,
        "flight_time": flight_time,
        "C_lqg": C_lqg,
        "energy_density": energy_density,
        "violates_bound": violates_bound,
        "violation_factor": violation_factor,
        "strictness_factor": strictness_factor,  # How much stricter than Ford-Roman
        "bound_type": "lqg_modified"
    }


def enforce_lqg_bound(computed_energy: float, spatial_scale: float, 
                      flight_time: float, C_lqg: Optional[float] = None) -> float:
    """
    Enforce LQG-modified quantum inequality bound on computed negative energy.
    
    This function ensures that any computed negative energy E_- satisfies:
    E_- ≥ -C_LQG / T^4
    
    If the computed energy violates this bound, it is clamped to the bound value.
    
    Args:
        computed_energy: Raw computed negative energy
        spatial_scale: Characteristic spatial scale
        flight_time: Sampling/flight time duration T
        C_lqg: LQG modification constant (if None, use default)
        
    Returns:
        Energy value respecting LQG bound
    """
    if computed_energy >= 0:
        return computed_energy
    
    # Get LQG bound
    lqg_bounds = lqg_modified_bounds(computed_energy, spatial_scale, flight_time, C_lqg)
    lqg_limit = lqg_bounds["lqg_bound"]
    
    # Enforce bound: E_- ≥ -C_LQG/T^4
    # Since lqg_limit is negative, we want max(computed_energy, lqg_limit)
    return max(computed_energy, lqg_limit)


class WarpBubbleStability:
    """
    Warp bubble stability analyzer.
    
    Provides methods for analyzing stability of warp bubble configurations
    and quantum inequality violations.
    """
    
    def __init__(self):
        """Initialize the stability analyzer."""
        self.logger = logging.getLogger(__name__)
        
    def analyze_stability(self, energy_density: float, spatial_scale: float, 
                         temporal_scale: Optional[float] = None) -> Dict:
        """
        Analyze stability of a warp bubble configuration.
        
        Args:
            energy_density: Peak energy density
            spatial_scale: Characteristic spatial scale
            temporal_scale: Characteristic temporal scale
            
        Returns:
            Dictionary with stability analysis results
        """
        # Use existing Ford-Roman bounds analysis
        ford_roman_result = ford_roman_bounds(energy_density, spatial_scale, temporal_scale)
        
        # Add additional stability metrics
        stability_result = {
            **ford_roman_result,
            "overall_stable": not ford_roman_result["violates_bound"],
            "stability_score": 1.0 if not ford_roman_result["violates_bound"] else 0.5,
            "analysis_method": "ford_roman_bounds"
        }
        
        return stability_result
        
    def check_qi_violation(self, energy_density: float, spatial_scale: float) -> bool:
        """
        Check if configuration violates quantum inequalities.
        
        Args:
            energy_density: Energy density to check
            spatial_scale: Spatial scale
            
        Returns:
            True if quantum inequalities are violated
        """
        bounds = ford_roman_bounds(energy_density, spatial_scale)
        return bounds["violates_bound"]