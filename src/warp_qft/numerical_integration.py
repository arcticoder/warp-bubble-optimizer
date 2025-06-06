#!/usr/bin/env python3
"""
Numerical Integration Utilities for Warp Bubble Calculations

This module provides specialized numerical integration routines for computing
energy integrals, stress-energy tensors, and geometric quantities in warp bubble
spacetimes.
"""

import numpy as np
from scipy.integrate import quad, dblquad, tplquad, simpson
try:
    from scipy.integrate import trapezoid as trapz
except ImportError:
    from scipy.integrate import trapz
from scipy.special import sph_harm, gamma, factorial
from typing import Callable, Tuple, Dict, List, Optional, Any
import warnings

class WarpBubbleIntegrator:
    """
    Specialized numerical integrator for warp bubble calculations.
    """
    
    def __init__(self, 
                 integration_method: str = 'adaptive',
                 tolerance: float = 1e-8,
                 max_subdivisions: int = 50):
        """
        Initialize the integrator.
        
        Args:
            integration_method: 'adaptive', 'simpson', 'trapezoid'
            tolerance: Integration tolerance
            max_subdivisions: Maximum number of subdivisions for adaptive methods
        """
        self.method = integration_method
        self.tolerance = tolerance
        self.max_subdivisions = max_subdivisions
    
    def energy_density_integral(self, 
                               energy_density: Callable,
                               r_range: Tuple[float, float],
                               theta_range: Tuple[float, float] = (0, np.pi),
                               phi_range: Tuple[float, float] = (0, 2*np.pi)) -> float:
        """
        Integrate energy density over spatial volume.
        
        Args:
            energy_density: Function ρ(r, θ, φ) in spherical coordinates
            r_range: Radial integration bounds (r_min, r_max)
            theta_range: Polar angle bounds
            phi_range: Azimuthal angle bounds
            
        Returns:
            Total energy
        """
        def integrand(phi, theta, r):
            """Volume element in spherical coordinates: r² sin(θ) dr dθ dφ"""
            return energy_density(r, theta, phi) * r**2 * np.sin(theta)
        
        if self.method == 'adaptive':
            result, error = tplquad(
                integrand,
                r_range[0], r_range[1],
                lambda r: theta_range[0], lambda r: theta_range[1],
                lambda r, theta: phi_range[0], lambda r, theta: phi_range[1],
                epsabs=self.tolerance
            )
            return result
        else:
            raise NotImplementedError(f"Method {self.method} not implemented for 3D integrals")
    
    def surface_integral(self,
                        surface_density: Callable,
                        radius: float,
                        theta_range: Tuple[float, float] = (0, np.pi),
                        phi_range: Tuple[float, float] = (0, 2*np.pi)) -> float:
        """
        Integrate over spherical surface at fixed radius.
        
        Args:
            surface_density: Function σ(θ, φ) on the sphere
            radius: Sphere radius
            theta_range: Polar angle bounds
            phi_range: Azimuthal angle bounds
            
        Returns:
            Surface integral
        """
        def integrand(phi, theta):
            """Surface element: r² sin(θ) dθ dφ"""
            return surface_density(theta, phi) * radius**2 * np.sin(theta)
        
        result, error = dblquad(
            integrand,
            theta_range[0], theta_range[1],
            lambda theta: phi_range[0], lambda theta: phi_range[1],
            epsabs=self.tolerance
        )
        return result
    
    def radial_integral(self,
                       radial_function: Callable,
                       r_range: Tuple[float, float],
                       weight_function: Optional[Callable] = None) -> float:
        """
        Integrate radial function with optional weight.
        
        Args:
            radial_function: Function f(r)
            r_range: Integration bounds
            weight_function: Optional weight w(r)
            
        Returns:
            Integral value
        """
        if weight_function is None:
            weight_function = lambda r: 1.0
        
        def integrand(r):
            return radial_function(r) * weight_function(r)
        
        if self.method == 'adaptive':
            result, error = quad(
                integrand,
                r_range[0], r_range[1],
                epsabs=self.tolerance,
                limit=self.max_subdivisions
            )
            return result
        elif self.method == 'simpson':
            r_vals = np.linspace(r_range[0], r_range[1], 1001)  # Odd number for Simpson
            y_vals = integrand(r_vals)
            return simpson(y_vals, x=r_vals)
        elif self.method == 'trapezoid':
            r_vals = np.linspace(r_range[0], r_range[1], 1001)
            y_vals = integrand(r_vals)
            return trapz(y_vals, x=r_vals)
        else:
            raise ValueError(f"Unknown integration method: {self.method}")

class EnergyMomentumCalculator:
    """
    Calculate energy-momentum tensor components and derived quantities.
    """
    
    def __init__(self, metric_calculator: Callable):
        """
        Initialize with metric calculation function.
        
        Args:
            metric_calculator: Function that returns metric tensor given coordinates
        """
        self.metric_calculator = metric_calculator
        self.integrator = WarpBubbleIntegrator()
    
    def stress_energy_00(self, coordinates: Tuple) -> float:
        """
        Calculate T^00 component (energy density).
        
        Args:
            coordinates: (t, r, θ, φ) coordinates
            
        Returns:
            Energy density at the point
        """
        # This would use Einstein field equations: G_μν = 8πT_μν
        # Placeholder implementation
        t, r, theta, phi = coordinates
        
        # Get metric at this point
        g = self.metric_calculator(coordinates)
        
        # Calculate Einstein tensor (simplified placeholder)
        # In practice, this requires computing Christoffel symbols, Ricci tensor, etc.
        G_00 = self._calculate_einstein_tensor_component(g, 0, 0)
        
        # T^00 = G^00 / (8π)
        return G_00 / (8 * np.pi)
    
    def _calculate_einstein_tensor_component(self, metric: np.ndarray, mu: int, nu: int) -> float:
        """
        Calculate Einstein tensor component (placeholder).
        
        In a full implementation, this would:
        1. Compute Christoffel symbols from metric derivatives
        2. Calculate Riemann tensor
        3. Contract to get Ricci tensor and scalar
        4. Form Einstein tensor G_μν = R_μν - (1/2)g_μν R
        """
        # Placeholder: return small value for now
        return 1e-10
    
    def total_energy(self, 
                    integration_bounds: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate total energy by integrating T^00.
        
        Args:
            integration_bounds: Dictionary with 'r', 'theta', 'phi' bounds
            
        Returns:
            Total energy
        """
        def energy_density(r, theta, phi):
            coordinates = (0, r, theta, phi)  # t=0 slice
            return self.stress_energy_00(coordinates)
        
        return self.integrator.energy_density_integral(
            energy_density,
            integration_bounds['r'],
            integration_bounds.get('theta', (0, np.pi)),
            integration_bounds.get('phi', (0, 2*np.pi))
        )

class GeometricQuantityCalculator:
    """
    Calculate geometric quantities: curvature, volume, etc.
    """
    
    def __init__(self):
        self.integrator = WarpBubbleIntegrator()
    
    def proper_volume(self,
                     metric_determinant: Callable,
                     bounds: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate proper volume using metric determinant.
        
        Args:
            metric_determinant: Function returning √|g|
            bounds: Integration bounds for each coordinate
            
        Returns:
            Proper volume
        """
        def volume_element(r, theta, phi):
            return metric_determinant(r, theta, phi)
        
        return self.integrator.energy_density_integral(
            volume_element,
            bounds['r'],
            bounds.get('theta', (0, np.pi)),
            bounds.get('phi', (0, 2*np.pi))
        )
    
    def curvature_scalar_integral(self,
                                 ricci_scalar: Callable,
                                 metric_determinant: Callable,
                                 bounds: Dict[str, Tuple[float, float]]) -> float:
        """
        Integrate Ricci scalar over volume.
        
        Args:
            ricci_scalar: Function R(r, θ, φ)
            metric_determinant: Function √|g|(r, θ, φ)
            bounds: Integration bounds
            
        Returns:
            Integrated curvature
        """
        def integrand(r, theta, phi):
            return ricci_scalar(r, theta, phi) * metric_determinant(r, theta, phi)
        
        return self.integrator.energy_density_integral(
            integrand,
            bounds['r'],
            bounds.get('theta', (0, np.pi)),
            bounds.get('phi', (0, 2*np.pi))
        )

class SpecializedIntegrals:
    """
    Specialized integrals for specific warp bubble calculations.
    """
    
    @staticmethod
    def van_den_broeck_energy_integral(R_warp: float, 
                                      sigma: float,
                                      r_max: float = 10.0) -> float:
        """
        Specialized integral for Van den Broeck metric energy.
        
        Args:
            R_warp: Warp bubble radius
            sigma: Wall thickness parameter
            r_max: Maximum integration radius
            
        Returns:
            Energy integral value
        """
        def integrand(r):
            # Van den Broeck shape function derivative
            xs = (r - R_warp) / sigma
            if abs(xs) > 10:  # Avoid overflow
                return 0.0
            
            tanh_xs = np.tanh(xs)
            sech_xs = 1 / np.cosh(xs)
            
            # Energy density ~ (df/dr)²
            df_dr = (1 / sigma) * sech_xs**2
            
            # Volume element in spherical coordinates
            return 4 * np.pi * r**2 * df_dr**2
        
        result, _ = quad(integrand, 0, r_max, epsabs=1e-10)
        return result
    
    @staticmethod
    def lqg_correction_integral(mu_parameter: float,
                               R_warp: float,
                               sigma: float) -> float:
        """
        Integral for LQG polymer corrections.
        
        Args:
            mu_parameter: LQG area parameter
            R_warp: Warp bubble radius
            sigma: Wall thickness
            
        Returns:
            LQG correction factor
        """
        def integrand(r):
            # LQG correction with sinc(πμ) profile
            if mu_parameter * r == 0:
                sinc_factor = 1.0
            else:
                sinc_factor = np.sin(np.pi * mu_parameter * r) / (np.pi * mu_parameter * r)
            
            # Volume element
            return 4 * np.pi * r**2 * sinc_factor**2
        
        result, _ = quad(integrand, 0, 5 * R_warp, epsabs=1e-10)
        return result

def create_energy_calculator(metric_type: str) -> Callable:
    """
    Factory function for creating energy calculators.
    
    Args:
        metric_type: Type of metric ('van_den_broeck', 'natario', 'custom')
        
    Returns:
        Energy calculation function
    """
    if metric_type == 'van_den_broeck':
        def vdb_energy_calculator(params):
            R_warp, sigma = params[:2]
            return SpecializedIntegrals.van_den_broeck_energy_integral(R_warp, sigma)
        return vdb_energy_calculator
        
    elif metric_type == 'custom':
        def custom_energy_calculator(params):
            # Generic energy calculator
            calculator = EnergyMomentumCalculator(lambda coords: np.eye(4))
            bounds = {'r': (0, 10), 'theta': (0, np.pi), 'phi': (0, 2*np.pi)}
            return calculator.total_energy(bounds)
        return custom_energy_calculator
        
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")
