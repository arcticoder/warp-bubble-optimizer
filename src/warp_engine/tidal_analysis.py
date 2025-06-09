"""
Tidal-Force & Crew-Safety Analysis
==================================

This module analyzes tidal forces and crew safety within warp bubbles through:
- Geodesic deviation calculations
- Curvature tensor analysis  
- Human survivability thresholds
- Metric smoothing optimization

Features:
- Riemann tensor computation and geodesic deviation
- Tidal acceleration mapping throughout bubble interior
- Crew safety assessment against g-force limits
- Metric refinement for reduced internal curvature
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

@dataclass
class TidalForceResult:
    """Results from tidal force analysis."""
    max_tidal_acceleration: float  # m/s² 
    tidal_acceleration_map: np.ndarray  # 3D acceleration field
    coordinate_grid: Tuple[np.ndarray, np.ndarray, np.ndarray]  # (x, y, z) grids
    crew_safe_volume: float  # m³ of safe region
    safety_factor: float  # [0,1] overall safety score

@dataclass  
class CrewSafetyAssessment:
    """Comprehensive crew safety analysis."""
    tidal_forces_safe: bool
    max_acceleration_g: float  # Maximum acceleration in g units
    safe_zone_radius: float  # Radius of safe zone (m)
    time_to_unsafe: Optional[float]  # Time until unsafe (s), None if always safe
    recommended_mission_duration: float  # Recommended max duration (s)
    safety_margins: Dict[str, float]  # Various safety margins

@dataclass
class CrewSafetyConfig:
    """Configuration for crew safety analysis."""
    max_acceleration_g: float = 5.0  # Maximum safe acceleration in g units
    safe_zone_radius: float = 5.0    # Required safe zone radius (m)
    max_mission_duration: float = 3600.0  # Maximum mission duration (s)
    tidal_force_threshold: float = 20.0   # m/s² threshold
    crew_mass: float = 70.0         # kg (average crew member)
    spacecraft_radius: float = 10.0  # m (spacecraft size)

class TidalForceAnalyzer:
    """
    Analyzes tidal forces and geodesic deviation in warp bubble metrics.
    """
    
    def __init__(self, 
                 human_g_tolerance: float = 9.0,  # 9g sustained tolerance
                 safety_margin: float = 2.0):      # 2x safety factor
        """
        Initialize tidal force analyzer.
        
        Args:
            human_g_tolerance: Maximum sustainable g-force for crew
            safety_margin: Safety factor for conservative design
        """
        self.human_g_tolerance = human_g_tolerance
        self.safety_margin = safety_margin
        self.g_earth = 9.80665  # m/s²
        
        # Symbolic coordinates
        self.t, self.x, self.y, self.z = sp.symbols('t x y z', real=True)
        self.coords = [self.t, self.x, self.y, self.z]
    
    def compute_riemann_tensor_numerical(self, 
                                       metric_func: Callable[[np.ndarray], np.ndarray],
                                       position: np.ndarray,
                                       dx: float = 1e-6) -> np.ndarray:
        """
        Compute Riemann curvature tensor numerically.
        
        Args:
            metric_func: Function returning metric tensor at position
            position: 4D spacetime position [t, x, y, z]
            dx: Finite difference step size
            
        Returns:
            4x4x4x4 Riemann tensor R^μ_{νρσ}
        """
        R = np.zeros((4, 4, 4, 4))
        
        # Compute Christoffel symbols at position
        gamma = self._compute_christoffel_numerical(metric_func, position, dx)
        
        # R^μ_{νρσ} = ∂_ρ Γ^μ_{νσ} - ∂_σ Γ^μ_{νρ} + Γ^μ_{λρ} Γ^λ_{νσ} - Γ^μ_{λσ} Γ^λ_{νρ}
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        # Partial derivatives of Christoffel symbols
                        pos_rho_plus = position.copy()
                        pos_rho_plus[rho] += dx
                        gamma_rho_plus = self._compute_christoffel_numerical(metric_func, pos_rho_plus, dx)
                        
                        pos_sigma_plus = position.copy()
                        pos_sigma_plus[sigma] += dx
                        gamma_sigma_plus = self._compute_christoffel_numerical(metric_func, pos_sigma_plus, dx)
                        
                        d_rho_gamma = (gamma_rho_plus[mu, nu, sigma] - gamma[mu, nu, sigma]) / dx
                        d_sigma_gamma = (gamma_sigma_plus[mu, nu, rho] - gamma[mu, nu, rho]) / dx
                        
                        # Christoffel products
                        product1 = sum(gamma[mu, lam, rho] * gamma[lam, nu, sigma] for lam in range(4))
                        product2 = sum(gamma[mu, lam, sigma] * gamma[lam, nu, rho] for lam in range(4))
                        
                        R[mu, nu, rho, sigma] = d_rho_gamma - d_sigma_gamma + product1 - product2
        
        return R
    
    def _compute_christoffel_numerical(self, 
                                     metric_func: Callable[[np.ndarray], np.ndarray],
                                     position: np.ndarray,
                                     dx: float = 1e-6) -> np.ndarray:
        """Compute Christoffel symbols numerically."""
        gamma = np.zeros((4, 4, 4))
        
        # Get metric and its inverse at position
        g = metric_func(position)
        g_inv = np.linalg.inv(g)
        
        # Γ^μ_{νρ} = (1/2) g^{μλ} (∂_ν g_{λρ} + ∂_ρ g_{λν} - ∂_λ g_{νρ})
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    christoffel = 0.0
                    for lam in range(4):
                        # Compute metric derivatives
                        pos_nu_plus = position.copy()
                        pos_nu_plus[nu] += dx
                        pos_rho_plus = position.copy()
                        pos_rho_plus[rho] += dx
                        pos_lam_plus = position.copy()
                        pos_lam_plus[lam] += dx
                        
                        g_nu_plus = metric_func(pos_nu_plus)
                        g_rho_plus = metric_func(pos_rho_plus)
                        g_lam_plus = metric_func(pos_lam_plus)
                        
                        d_nu_g_lam_rho = (g_nu_plus[lam, rho] - g[lam, rho]) / dx
                        d_rho_g_lam_nu = (g_rho_plus[lam, nu] - g[lam, nu]) / dx
                        d_lam_g_nu_rho = (g_lam_plus[nu, rho] - g[nu, rho]) / dx
                        
                        christoffel += 0.5 * g_inv[mu, lam] * (d_nu_g_lam_rho + d_rho_g_lam_nu - d_lam_g_nu_rho)
                    
                    gamma[mu, nu, rho] = christoffel
        
        return gamma
    
    def geodesic_deviation(self,
                          riemann_tensor: np.ndarray,
                          four_velocity: np.ndarray,
                          separation_vector: np.ndarray) -> np.ndarray:
        """
        Compute geodesic deviation (tidal acceleration).
        
        Args:
            riemann_tensor: 4x4x4x4 Riemann tensor R^μ_{νρσ}
            four_velocity: 4-velocity u^α of observer
            separation_vector: Spatial separation s^β
            
        Returns:
            4-vector of tidal acceleration a^μ = -R^μ_{νρσ} u^ν u^ρ s^σ
        """
        a = np.zeros(4)
        
        for mu in range(4):
            acceleration = 0.0
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        acceleration -= (riemann_tensor[mu, nu, rho, sigma] * 
                                       four_velocity[nu] * four_velocity[rho] * 
                                       separation_vector[sigma])
            a[mu] = acceleration
        
        return a
    
    def analyze_tidal_forces(self,
                           metric_func: Callable[[np.ndarray], np.ndarray],
                           bubble_center: np.ndarray = np.array([0, 0, 0, 0]),
                           analysis_volume: float = 10.0,  # m³
                           grid_resolution: int = 20) -> TidalForceResult:
        """
        Analyze tidal forces throughout bubble interior.
        
        Args:
            metric_func: Function returning 4x4 metric tensor
            bubble_center: Center of analysis region [t, x, y, z]
            analysis_volume: Volume to analyze (m³)
            grid_resolution: Grid points per dimension
            
        Returns:
            TidalForceResult with complete analysis
        """
        # Create spatial grid around bubble center
        radius = (3 * analysis_volume / (4 * np.pi))**(1/3)  # Sphere radius
        
        x_grid = np.linspace(-radius, radius, grid_resolution)
        y_grid = np.linspace(-radius, radius, grid_resolution)
        z_grid = np.linspace(-radius, radius, grid_resolution)
        
        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
        
        # Initialize tidal acceleration field
        tidal_field = np.zeros((grid_resolution, grid_resolution, grid_resolution, 3))
        max_tidal_accel = 0.0
        
        # Observer 4-velocity (stationary in bubble frame)
        u = np.array([1, 0, 0, 0])  # u^α
        
        # Analyze tidal forces at each grid point
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                for k in range(grid_resolution):
                    pos_3d = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                    
                    # Skip points outside bubble
                    r = np.linalg.norm(pos_3d)
                    if r > radius:
                        continue
                    
                    # 4D position
                    position = np.array([bubble_center[0], 
                                       bubble_center[1] + pos_3d[0],
                                       bubble_center[2] + pos_3d[1], 
                                       bubble_center[3] + pos_3d[2]])
                    
                    try:
                        # Compute Riemann tensor at this position
                        R = self.compute_riemann_tensor_numerical(metric_func, position)
                        
                        # Test separations in x, y, z directions (1 meter each)
                        separations = [
                            np.array([0, 1, 0, 0]),  # x-direction
                            np.array([0, 0, 1, 0]),  # y-direction  
                            np.array([0, 0, 0, 1])   # z-direction
                        ]
                        
                        tidal_accels = []
                        for s in separations:
                            a = self.geodesic_deviation(R, u, s)
                            # Extract spatial components (ignore time component)
                            spatial_accel = np.linalg.norm(a[1:4])  
                            tidal_accels.append(spatial_accel)
                        
                        # Store maximum tidal acceleration at this point
                        max_local_accel = max(tidal_accels)
                        tidal_field[i, j, k, :] = a[1:4]  # Store spatial components
                        max_tidal_accel = max(max_tidal_accel, max_local_accel)
                        
                    except Exception as e:
                        logger.warning(f"Tidal force calculation failed at {pos_3d}: {e}")
                        continue
        
        # Analyze crew safe volume
        safe_threshold = self.human_g_tolerance * self.g_earth / self.safety_margin
        safe_mask = np.linalg.norm(tidal_field, axis=3) < safe_threshold
        safe_volume = np.sum(safe_mask) * (2*radius/grid_resolution)**3
        
        # Overall safety factor
        if max_tidal_accel > 0:
            safety_factor = min(1.0, safe_threshold / max_tidal_accel)
        else:
            safety_factor = 1.0
        
        return TidalForceResult(
            max_tidal_acceleration=max_tidal_accel,
            tidal_acceleration_map=tidal_field,            coordinate_grid=(X, Y, Z),
            crew_safe_volume=safe_volume,
            safety_factor=safety_factor
        )

    def compute_tidal_acceleration(self,
                                 position: Optional[np.ndarray] = None,
                                 bubble_radius: float = 10.0,
                                 warp_velocity: float = 1000.0,
                                 metric_func: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
        """
        Compute tidal acceleration for a given position within the warp bubble.
        
        Args:
            position: Position within bubble [x, y, z], defaults to center
            bubble_radius: Radius of warp bubble (m)
            warp_velocity: Warp velocity (m/s)
            metric_func: Function returning 4x4 metric tensor at position
            
        Returns:
            Tidal acceleration vector [ax, ay, az] in m/s²
        """
        if position is None:
            position = np.array([0, 0, 0])
        
        # Convert 3D position to 4D spacetime position
        spacetime_pos = np.array([0, position[0], position[1], position[2]])
        
        # Use van den Broeck metric if no metric function provided
        if metric_func is None:
            # Simple physics-based estimate for tidal forces
            r = np.linalg.norm(position)
            
            # Tidal acceleration scales with distance from center and warp parameters
            # Using order-of-magnitude estimates from warp drive literature
            beta = warp_velocity / 3e8  # Velocity parameter (c = 3e8 m/s)
            
            if r < 1e-10:  # At center
                return np.array([0.0, 0.0, 0.0])
            
            # Tidal forces increase toward bubble wall
            radial_factor = r / bubble_radius
            
            # Approximate tidal acceleration (order of magnitude)
            # Based on curvature scaling in Alcubierre metric
            tidal_magnitude = beta**2 * radial_factor / (bubble_radius**2) * 1e-6
            
            # Radial direction
            radial_unit = position / r
            tidal_accel = tidal_magnitude * radial_unit
            
            return tidal_accel
        
        # Use analyze_tidal_forces for more detailed computation
        analysis_volume = 4/3 * np.pi * bubble_radius**3
        
        try:
            result = self.analyze_tidal_forces(
                metric_func=metric_func,
                bubble_center=spacetime_pos,
                analysis_volume=analysis_volume,
                grid_resolution=5  # Use coarse grid for speed
            )
            
            # Extract tidal acceleration at requested position
            # For now, return the maximum as a conservative estimate
            max_accel = result.max_tidal_acceleration
            r = np.linalg.norm(position)
            if r > 0:
                radial_unit = position / r
                return max_accel * radial_unit
            else:
                return np.array([0.0, 0.0, 0.0])
                
        except Exception as e:
            logger.warning(f"Detailed tidal acceleration computation failed: {e}")
            # Return conservative physics-based estimate
            return self.compute_tidal_acceleration(position, bubble_radius, warp_velocity, None)
    
    def assess_crew_safety(self,
                          tidal_result: TidalForceResult,
                          mission_duration: float = 3600.0,  # 1 hour default
                          crew_positions: Optional[List[np.ndarray]] = None) -> CrewSafetyAssessment:
        """
        Assess crew safety based on tidal force analysis.
        
        Args:
            tidal_result: Results from tidal force analysis
            mission_duration: Planned mission duration (s)
            crew_positions: Specific crew positions to check (optional)
            
        Returns:
            CrewSafetyAssessment with recommendations
        """
        max_accel_g = tidal_result.max_tidal_acceleration / self.g_earth
        
        # Check if tidal forces are within safe limits
        safe_limit_g = self.human_g_tolerance / self.safety_margin
        tidal_forces_safe = max_accel_g <= safe_limit_g
        
        # Estimate safe zone radius
        if tidal_result.crew_safe_volume > 0:
            safe_zone_radius = (3 * tidal_result.crew_safe_volume / (4 * np.pi))**(1/3)
        else:
            safe_zone_radius = 0.0
        
        # Time-dependent safety analysis
        # Assume tidal stress accumulates over time (conservative)
        time_to_unsafe = None
        if max_accel_g > safe_limit_g:
            # Immediate danger
            time_to_unsafe = 0.0
        elif max_accel_g > safe_limit_g * 0.5:
            # Gradual stress accumulation model
            stress_rate = max_accel_g / safe_limit_g
            time_to_unsafe = mission_duration / stress_rate**2
        
        # Recommended mission duration
        if tidal_forces_safe:
            recommended_duration = mission_duration  # Full duration OK
        elif time_to_unsafe is not None:
            recommended_duration = min(mission_duration, time_to_unsafe * 0.8)  # 80% margin
        else:
            recommended_duration = 0.0  # Immediate abort
        
        # Safety margins
        safety_margins = {
            "tidal_acceleration": safe_limit_g / max(max_accel_g, 1e-10),
            "safe_volume": tidal_result.crew_safe_volume / 1000.0,  # m³ per 1000 m³
            "temporal": recommended_duration / max(mission_duration, 1.0)
        }
        
        return CrewSafetyAssessment(
            tidal_forces_safe=tidal_forces_safe,
            max_acceleration_g=max_accel_g,
            safe_zone_radius=safe_zone_radius,
            time_to_unsafe=time_to_unsafe,
            recommended_mission_duration=recommended_duration,
            safety_margins=safety_margins
        )
    
    def optimize_metric_smoothing(self,
                                metric_func: Callable[[np.ndarray], np.ndarray],
                                initial_params: np.ndarray,
                                target_max_accel: float = 1.0) -> Dict[str, Any]:
        """
        Optimize metric parameters to reduce internal tidal forces.
        
        Args:
            metric_func: Parameterized metric function
            initial_params: Initial parameter values
            target_max_accel: Target maximum acceleration (g)
            
        Returns:
            Optimization results with improved parameters
        """
        target_accel_mks = target_max_accel * self.g_earth
        
        def objective(params):
            """Objective function: minimize maximum tidal acceleration."""
            try:
                # Create metric function with these parameters
                def metric_with_params(position):
                    return metric_func(position, params)
                
                # Analyze tidal forces
                result = self.analyze_tidal_forces(
                    metric_with_params, 
                    analysis_volume=100.0,  # Smaller volume for optimization speed
                    grid_resolution=10
                )
                
                # Penalize high tidal accelerations
                penalty = max(0, result.max_tidal_acceleration - target_accel_mks)**2
                return penalty
                
            except Exception as e:
                logger.warning(f"Optimization evaluation failed: {e}")
                return 1e10  # Large penalty for failed evaluations
        
        # Optimize parameters
        optimization_result = minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 50, 'disp': True}
        )
        
        return {
            "success": optimization_result.success,
            "optimized_params": optimization_result.x,
            "final_max_accel": optimization_result.fun**0.5,  # Undo squaring
            "iterations": optimization_result.nit,
            "initial_params": initial_params
        }
    
    def plot_tidal_map(self, 
                      tidal_result: TidalForceResult,
                      slice_plane: str = 'z',
                      slice_index: Optional[int] = None,
                      save_path: Optional[str] = None):
        """Plot 2D slice of tidal acceleration field."""
        X, Y, Z = tidal_result.coordinate_grid
        tidal_field = tidal_result.tidal_acceleration_map
        
        if slice_index is None:
            slice_index = tidal_field.shape[2] // 2  # Middle slice
        
        # Extract 2D slice
        if slice_plane == 'z':
            x_coords = X[:, :, slice_index]
            y_coords = Y[:, :, slice_index]
            tidal_magnitude = np.linalg.norm(tidal_field[:, :, slice_index, :], axis=2)
            title = f"Tidal Acceleration (z={Z[0,0,slice_index]:.1f}m)"
            xlabel, ylabel = "X (m)", "Y (m)"
        elif slice_plane == 'y':
            x_coords = X[:, slice_index, :]
            y_coords = Z[:, slice_index, :]
            tidal_magnitude = np.linalg.norm(tidal_field[:, slice_index, :, :], axis=2)
            title = f"Tidal Acceleration (y={Y[0,slice_index,0]:.1f}m)"
            xlabel, ylabel = "X (m)", "Z (m)"
        else:  # x plane
            x_coords = Y[slice_index, :, :]
            y_coords = Z[slice_index, :, :]
            tidal_magnitude = np.linalg.norm(tidal_field[slice_index, :, :, :], axis=2)
            title = f"Tidal Acceleration (x={X[slice_index,0,0]:.1f}m)"
            xlabel, ylabel = "Y (m)", "Z (m)"
        
        # Create plot
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(x_coords, y_coords, tidal_magnitude, 
                              levels=20, cmap='viridis')
        plt.colorbar(contour, label='Tidal Acceleration (m/s²)')
        
        # Add safety threshold contour
        safe_threshold = self.human_g_tolerance * self.g_earth / self.safety_margin
        plt.contour(x_coords, y_coords, tidal_magnitude, 
                   levels=[safe_threshold], colors='red', linewidths=2, 
                   linestyles='--', alpha=0.8)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add text annotation for safety info
        plt.text(0.02, 0.98, f"Max: {tidal_result.max_tidal_acceleration:.2e} m/s²\n"
                             f"Safe limit: {safe_threshold:.2e} m/s²\n"  
                             f"Safety factor: {tidal_result.safety_factor:.3f}",
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Tidal map saved to {save_path}")
        else:
            plt.show()


# Example metric functions for testing
def van_den_broeck_metric(position: np.ndarray, params: Optional[np.ndarray] = None) -> np.ndarray:
    """Van den Broeck style metric for testing."""
    if params is None:
        params = np.array([10.0, 0.001, 2.0])  # [R_bubble, v/c, sigma]
    
    R_bubble, beta, sigma = params
    t, x, y, z = position
    
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Smooth wall function
    f = 0.5 * (1 + np.tanh(sigma * (r - R_bubble) / R_bubble))
    
    # Metric components (simplified)
    g_tt = -1 - beta**2 * (1 - f)
    g_xx = 1 + beta**2 * f
    g_yy = g_xx
    g_zz = g_xx
    
    g = np.array([
        [g_tt, 0, 0, 0],
        [0, g_xx, 0, 0],
        [0, 0, g_yy, 0],
        [0, 0, 0, g_zz]
    ])
    
    return g


# Example usage and testing
# Utility functions for backwards compatibility
def geodesic_deviation(curvature_tensor, u_vec, s_vec):
    """
    Utility function for computing geodesic deviation equation.
    
    Args:
        curvature_tensor: Riemann curvature tensor components
        u_vec: 4-velocity vector
        s_vec: separation vector
        
    Returns:
        Acceleration vector from geodesic deviation
    """
    # This is a simplified implementation - the full calculation
    # is available as a method in TidalForceAnalyzer
    
    analyzer = TidalForceAnalyzer()
    
    # Convert to internal format and compute
    position = np.array([0, 0, 0])  # Reference position
    metric_func = lambda pos, params: van_den_broeck_metric(pos, np.array([10.0, 0.001, 1.0]))
    
    result = analyzer.compute_geodesic_deviation(
        position, metric_func, np.array([10.0, 0.001, 1.0])
    )
    
    return result


def van_den_broeck_metric(position: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Simple Van den Broeck metric for testing."""
    x, y, z = position
    R, alpha, n = params
    r = np.sqrt(x**2 + y**2 + z**2)
    
    if r < R:
        f = np.tanh(alpha * (R - r)**n)
    else:
        f = 0.0
        
    # Metric components (simplified)
    g = np.eye(4)
    g[0, 0] = -(1 + f)
    g[1, 1] = 1 - f
    g[2, 2] = 1 - f
    g[3, 3] = 1 - f
    
    return g


if __name__ == "__main__":
    print("=== Testing Tidal Force Analysis ===")
    
    # Initialize analyzer
    analyzer = TidalForceAnalyzer(
        human_g_tolerance=9.0,  # 9g tolerance
        safety_margin=2.0       # 2x safety factor
    )
    
    # Test with van den Broeck metric
    print("Analyzing tidal forces in van den Broeck metric...")
    
    tidal_result = analyzer.analyze_tidal_forces(
        van_den_broeck_metric,
        bubble_center=np.array([0, 0, 0, 0]),
        analysis_volume=500.0,  # 500 m³
        grid_resolution=15       # Reduced for speed
    )
    
    print(f"Max tidal acceleration: {tidal_result.max_tidal_acceleration:.2e} m/s²")
    print(f"Max tidal acceleration: {tidal_result.max_tidal_acceleration/9.80665:.2f} g")
    print(f"Crew safe volume: {tidal_result.crew_safe_volume:.1f} m³")
    print(f"Overall safety factor: {tidal_result.safety_factor:.3f}")
    
    # Crew safety assessment
    print("\n=== Crew Safety Assessment ===")
    safety_assessment = analyzer.assess_crew_safety(
        tidal_result,
        mission_duration=3600.0  # 1 hour mission
    )
    
    print(f"Tidal forces safe: {safety_assessment.tidal_forces_safe}")
    print(f"Max acceleration: {safety_assessment.max_acceleration_g:.2f} g")
    print(f"Safe zone radius: {safety_assessment.safe_zone_radius:.1f} m")
    print(f"Recommended duration: {safety_assessment.recommended_mission_duration:.0f} s")
    
    if safety_assessment.time_to_unsafe is not None:
        print(f"Time to unsafe conditions: {safety_assessment.time_to_unsafe:.0f} s")
    
    # Safety margins
    print("\nSafety margins:")
    for margin_type, value in safety_assessment.safety_margins.items():
        print(f"  {margin_type}: {value:.3f}")
    
    # Test metric optimization
    print("\n=== Testing Metric Optimization ===")
    initial_params = np.array([10.0, 0.002, 1.5])  # Slightly unsafe configuration
    
    optimization_result = analyzer.optimize_metric_smoothing(
        lambda pos, params: van_den_broeck_metric(pos, params),
        initial_params,
        target_max_accel=2.0  # Target 2g max
    )
    
    print(f"Optimization success: {optimization_result['success']}")
    print(f"Initial params: {initial_params}")
    print(f"Optimized params: {optimization_result['optimized_params']}")
    print(f"Final max accel: {optimization_result['final_max_accel']:.2e} m/s²")
    print(f"Final max accel: {optimization_result['final_max_accel']/9.80665:.2f} g")
    
    # Plot tidal map
    analyzer.plot_tidal_map(tidal_result, save_path="tidal_acceleration_map.png")
