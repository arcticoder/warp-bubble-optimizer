#!/usr/bin/env python3
"""
3+1D Time Evolution Prototype for Warp Bubble Stability

This module implements a simplified 3+1D finite-difference evolution of 
the scalar field φ(x,t) and its conjugate momentum π(x,t) to test 
time-dependent stability of optimized ansätze.

Starting with the Klein-Gordon-like equation:
∂²φ/∂t² - ∇²φ + V_eff(φ, ∇φ) = 0

Where V_eff includes the effective potential from the energy functional.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, Dict, Callable, Optional
import logging
from numba import jit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WarpBubble3D1Evolution:
    """
    3+1D evolution of warp bubble field configurations.
    
    Implements finite-difference time evolution of:
    ∂φ/∂t = π
    ∂π/∂t = ∇²φ - V'(φ)
    
    where φ represents the warp factor field and π is its conjugate momentum.
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int, int] = (64, 64, 64),
                 box_size: Tuple[float, float, float] = (10.0, 10.0, 10.0),
                 dt: float = 0.01,
                 courant_factor: float = 0.5):
        """
        Initialize the evolution system.
        
        Args:
            grid_size: Number of grid points in (x, y, z)
            box_size: Physical size of simulation box
            dt: Time step
            courant_factor: Courant condition safety factor
        """
        self.nx, self.ny, self.nz = grid_size
        self.Lx, self.Ly, self.Lz = box_size
        
        # Spatial grid
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz
        
        # Time step (check Courant condition)
        dx_min = min(self.dx, self.dy, self.dz)
        dt_max = courant_factor * dx_min / np.sqrt(3)  # 3D Courant condition
        self.dt = min(dt, dt_max)
        
        logger.info(f"Grid: {grid_size}, Box: {box_size}")
        logger.info(f"Spatial resolution: dx={self.dx:.3f}, dy={self.dy:.3f}, dz={self.dz:.3f}")
        logger.info(f"Time step: {self.dt:.6f} (max stable: {dt_max:.6f})")
        
        # Create coordinate arrays
        self.x = np.linspace(-self.Lx/2, self.Lx/2, self.nx)
        self.y = np.linspace(-self.Ly/2, self.Ly/2, self.ny)
        self.z = np.linspace(-self.Lz/2, self.Lz/2, self.nz)
        
        # 3D coordinate meshes
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        
        # Field arrays
        self.phi = np.zeros((self.nx, self.ny, self.nz))      # Scalar field
        self.pi = np.zeros((self.nx, self.ny, self.nz))       # Conjugate momentum
        self.phi_new = np.zeros((self.nx, self.ny, self.nz))  # Updated field
        self.pi_new = np.zeros((self.nx, self.ny, self.nz))   # Updated momentum
        
        # Evolution history
        self.time = 0.0
        self.step = 0
        self.energy_history = []
        self.max_field_history = []
        
    @staticmethod
    @jit(nopython=True)
    def laplacian_3d(field: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
        """
        Compute 3D Laplacian using finite differences with periodic boundaries.
        
        Args:
            field: 3D field array
            dx, dy, dz: Grid spacing in each dimension
            
        Returns:
            3D Laplacian of the field
        """
        nx, ny, nz = field.shape
        laplacian = np.zeros_like(field)
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Periodic boundary conditions
                    im1 = (i - 1) % nx
                    ip1 = (i + 1) % nx
                    jm1 = (j - 1) % ny
                    jp1 = (j + 1) % ny
                    km1 = (k - 1) % nz
                    kp1 = (k + 1) % nz
                    
                    # Second derivatives
                    d2_dx2 = (field[ip1, j, k] - 2*field[i, j, k] + field[im1, j, k]) / (dx**2)
                    d2_dy2 = (field[i, jp1, k] - 2*field[i, j, k] + field[i, jm1, k]) / (dy**2)
                    d2_dz2 = (field[i, j, kp1] - 2*field[i, j, k] + field[i, j, km1]) / (dz**2)
                    
                    laplacian[i, j, k] = d2_dx2 + d2_dy2 + d2_dz2
        
        return laplacian
    
    def effective_potential(self, phi: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Effective potential for warp bubble dynamics.
        
        V_eff(φ, r) = m²φ²/2 + λφ⁴/4 + V_bubble(r)φ
        
        where V_bubble(r) represents the background warp bubble potential.
        
        Args:
            phi: Field values
            r: Radial coordinate
            
        Returns:
            Effective potential values
        """
        # Field mass and self-interaction
        m_squared = 0.1  # Field mass squared
        lambda_4 = 0.01  # Quartic coupling
        
        # Background bubble potential (attractive near r=R_bubble)
        R_bubble = 2.5
        V_bubble = -1.0 * np.exp(-((r - R_bubble) / 0.5)**2)
        
        # Total potential
        V_eff = 0.5 * m_squared * phi**2 + 0.25 * lambda_4 * phi**4 + V_bubble * phi
        
        return V_eff
    
    def potential_derivative(self, phi: np.ndarray, r: np.ndarray) -> np.ndarray:
        """
        Derivative of effective potential dV_eff/dφ.
        
        Args:
            phi: Field values
            r: Radial coordinate
            
        Returns:
            Potential derivative
        """
        m_squared = 0.1
        lambda_4 = 0.01
        R_bubble = 2.5
        V_bubble = -1.0 * np.exp(-((r - R_bubble) / 0.5)**2)
        
        dV_dphi = m_squared * phi + lambda_4 * phi**3 + V_bubble
        
        return dV_dphi
    
    def initialize_warp_bubble(self, ansatz_type: str = 'gaussian', **kwargs):
        """
        Initialize field configuration with optimized ansatz.
        
        Args:
            ansatz_type: Type of initial ansatz ('gaussian', 'polynomial', 'soliton')
            **kwargs: Ansatz parameters
        """
        if ansatz_type == 'gaussian':
            # Gaussian warp bubble
            amplitude = kwargs.get('amplitude', 1.0)
            width = kwargs.get('width', 1.0)
            center = kwargs.get('center', 2.5)
            
            self.phi = amplitude * np.exp(-((self.R - center) / width)**2)
            
        elif ansatz_type == 'polynomial':
            # Polynomial profile from optimization
            coeffs = kwargs.get('coeffs', [1.0, -0.5, 0.2, -0.05])
            R_max = kwargs.get('R_max', 5.0)
            
            r_norm = self.R / R_max
            self.phi = np.zeros_like(self.R)
            
            for i, coeff in enumerate(coeffs):
                self.phi += coeff * r_norm**i
            
            # Apply cutoff
            cutoff = np.exp(-((self.R - R_max) / (0.1 * R_max))**2)
            self.phi *= cutoff
            
        elif ansatz_type == 'soliton':
            # Soliton-like profile
            amplitude = kwargs.get('amplitude', 1.0)
            width = kwargs.get('width', 1.0)
            center = kwargs.get('center', 2.5)
            
            self.phi = amplitude / np.cosh((self.R - center) / width)**2
            
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")
        
        # Initialize momentum to zero (static initial configuration)
        self.pi.fill(0.0)
        
        logger.info(f"Initialized {ansatz_type} warp bubble")
        logger.info(f"Initial field range: [{np.min(self.phi):.6f}, {np.max(self.phi):.6f}]")
    
    def compute_energy(self) -> Dict[str, float]:
        """
        Compute total energy and its components.
        
        Returns:
            Dictionary with energy components
        """
        # Kinetic energy: ∫ π²/2 d³x
        kinetic = 0.5 * np.sum(self.pi**2) * self.dx * self.dy * self.dz
        
        # Gradient energy: ∫ |∇φ|²/2 d³x
        grad_x = np.gradient(self.phi, self.dx, axis=0)
        grad_y = np.gradient(self.phi, self.dy, axis=1)
        grad_z = np.gradient(self.phi, self.dz, axis=2)
        gradient = 0.5 * np.sum(grad_x**2 + grad_y**2 + grad_z**2) * self.dx * self.dy * self.dz
        
        # Potential energy: ∫ V_eff(φ, r) d³x
        V_eff = self.effective_potential(self.phi, self.R)
        potential = np.sum(V_eff) * self.dx * self.dy * self.dz
        
        # Total energy
        total = kinetic + gradient + potential
        
        return {
            'kinetic': kinetic,
            'gradient': gradient,
            'potential': potential,
            'total': total
        }
    
    def evolve_step(self):
        """
        Perform one time evolution step using leapfrog integration.
        """
        # Compute Laplacian
        laplacian = self.laplacian_3d(self.phi, self.dx, self.dy, self.dz)
        
        # Compute potential derivative
        dV_dphi = self.potential_derivative(self.phi, self.R)
        
        # Update equations:
        # φ(t+dt) = φ(t) + dt * π(t)
        # π(t+dt) = π(t) + dt * (∇²φ(t) - dV/dφ(t))
        
        self.phi_new[:] = self.phi + self.dt * self.pi
        self.pi_new[:] = self.pi + self.dt * (laplacian - dV_dphi)
        
        # Update fields
        self.phi[:] = self.phi_new
        self.pi[:] = self.pi_new
        
        # Update time
        self.time += self.dt
        self.step += 1
    
    def evolve(self, total_time: float, save_interval: int = 10) -> Dict:
        """
        Evolve the system for a given total time.
        
        Args:
            total_time: Total evolution time
            save_interval: Steps between data saves
            
        Returns:
            Evolution data dictionary
        """
        n_steps = int(total_time / self.dt)
        
        logger.info(f"Starting evolution for {total_time:.2f} time units ({n_steps} steps)")
        
        # Storage for snapshots
        snapshots = []
        times = []
        
        for step in range(n_steps):
            # Evolve one step
            self.evolve_step()
            
            # Compute diagnostics
            if step % save_interval == 0:
                energy = self.compute_energy()
                max_field = np.max(np.abs(self.phi))
                
                self.energy_history.append(energy['total'])
                self.max_field_history.append(max_field)
                
                # Save snapshot
                snapshots.append(self.phi.copy())
                times.append(self.time)
                
                if step % (save_interval * 10) == 0:
                    logger.info(f"Step {step}/{n_steps}, t={self.time:.3f}, "
                              f"E_total={energy['total']:.6f}, max|φ|={max_field:.6f}")
        
        logger.info("Evolution complete!")
        
        return {
            'times': np.array(times),
            'snapshots': snapshots,
            'energy_history': np.array(self.energy_history),
            'max_field_history': np.array(self.max_field_history),
            'final_time': self.time,
            'total_steps': self.step
        }
    
    def analyze_stability(self, evolution_data: Dict) -> Dict:
        """
        Analyze stability of the evolved configuration.
        
        Args:
            evolution_data: Results from evolve() method
            
        Returns:
            Stability analysis results
        """
        times = evolution_data['times']
        energy_history = evolution_data['energy_history']
        max_field_history = evolution_data['max_field_history']
        
        # Energy conservation check
        energy_drift = np.abs(energy_history[-1] - energy_history[0]) / np.abs(energy_history[0])
        
        # Field growth analysis
        field_growth_rate = (np.log(max_field_history[-1]) - np.log(max_field_history[0])) / times[-1]
        
        # Stability assessment
        is_stable = (energy_drift < 0.1) and (field_growth_rate < 1.0)
        
        # Collapse/dispersion analysis
        initial_size = np.sum(np.abs(evolution_data['snapshots'][0]) > 0.1 * np.max(evolution_data['snapshots'][0]))
        final_size = np.sum(np.abs(evolution_data['snapshots'][-1]) > 0.1 * np.max(evolution_data['snapshots'][-1]))
        size_ratio = final_size / initial_size
        
        collapse_tendency = "dispersion" if size_ratio > 1.5 else ("collapse" if size_ratio < 0.5 else "stable")
        
        return {
            'energy_drift': energy_drift,
            'field_growth_rate': field_growth_rate,
            'is_stable': is_stable,
            'size_ratio': size_ratio,
            'collapse_tendency': collapse_tendency,
            'assessment': 'STABLE' if is_stable else 'UNSTABLE'
        }
    
    def plot_evolution(self, evolution_data: Dict, save_path: Optional[str] = None):
        """
        Plot evolution results.
        
        Args:
            evolution_data: Results from evolve() method
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        times = evolution_data['times']
        
        # Energy evolution
        axes[0, 0].plot(times, evolution_data['energy_history'])
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Total Energy')
        axes[0, 0].set_title('Energy Conservation')
        axes[0, 0].grid(True)
        
        # Maximum field amplitude
        axes[0, 1].semilogy(times, evolution_data['max_field_history'])
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Max |φ|')
        axes[0, 1].set_title('Field Amplitude Evolution')
        axes[0, 1].grid(True)
        
        # Initial field profile (slice through center)
        center_slice = evolution_data['snapshots'][0][:, self.ny//2, self.nz//2]
        axes[1, 0].plot(self.x, center_slice, label='Initial')
        
        # Final field profile
        final_slice = evolution_data['snapshots'][-1][:, self.ny//2, self.nz//2]
        axes[1, 0].plot(self.x, final_slice, label='Final', linestyle='--')
        
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('φ(x, 0, 0)')
        axes[1, 0].set_title('Field Profile Evolution (Central Slice)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 2D field visualization (z=0 slice)
        im = axes[1, 1].imshow(evolution_data['snapshots'][-1][:, :, self.nz//2],
                              extent=[-self.Lx/2, self.Lx/2, -self.Ly/2, self.Ly/2],
                              cmap='RdBu_r', aspect='equal')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        axes[1, 1].set_title('Final Field Configuration (z=0)')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved evolution plot to {save_path}")
        
        plt.close()  # Close instead of show to prevent blocking


def test_ansatz_stability(ansatz_params: Dict, evolution_time: float = 10.0) -> Dict:
    """
    Test stability of a given ansatz configuration.
    
    Args:
        ansatz_params: Dictionary with ansatz parameters
        evolution_time: Time to evolve the system
        
    Returns:
        Stability test results
    """
    # Initialize evolution system
    evolver = WarpBubble3D1Evolution(
        grid_size=(32, 32, 32),  # Reduced for faster computation
        box_size=(8.0, 8.0, 8.0),
        dt=0.005
    )
    
    # Set up initial conditions
    evolver.initialize_warp_bubble(**ansatz_params)
    
    # Evolve system
    evolution_data = evolver.evolve(evolution_time, save_interval=20)
    
    # Analyze stability
    stability = evolver.analyze_stability(evolution_data)
    
    # Generate plots
    evolver.plot_evolution(evolution_data, f"stability_test_{ansatz_params.get('ansatz_type', 'default')}.png")
    
    return {
        'ansatz_params': ansatz_params,
        'evolution_data': evolution_data,
        'stability_analysis': stability,
        'evolver': evolver
    }


def main():
    """
    Main demonstration of 3+1D evolution framework.
    """
    print("🌌 3+1D Warp Bubble Evolution Framework")
    print("=" * 50)
    
    # Test different ansatz configurations
    test_configurations = [
        {
            'ansatz_type': 'gaussian',
            'amplitude': 0.5,
            'width': 1.0,
            'center': 2.5
        },
        {
            'ansatz_type': 'polynomial',
            'coeffs': [1.0, -0.9, 0.16, 0.65, 0.09],  # From optimization
            'R_max': 4.0
        },
        {
            'ansatz_type': 'soliton',
            'amplitude': 0.8,
            'width': 0.8,
            'center': 2.5
        }
    ]
    
    results = {}
    
    for i, config in enumerate(test_configurations):
        print(f"\n{i+1}. Testing {config['ansatz_type']} ansatz...")
        
        try:
            result = test_ansatz_stability(config, evolution_time=5.0)
            results[config['ansatz_type']] = result
            
            stability = result['stability_analysis']
            print(f"   Status: {stability['assessment']}")
            print(f"   Energy drift: {stability['energy_drift']:.6f}")
            print(f"   Growth rate: {stability['field_growth_rate']:.6f}")
            print(f"   Tendency: {stability['collapse_tendency']}")
            
        except Exception as e:
            print(f"   Error in evolution: {e}")
            results[config['ansatz_type']] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*50)
    print("STABILITY SUMMARY")
    print("="*50)
    
    stable_ansatze = []
    for ansatz_type, result in results.items():
        if 'stability_analysis' in result:
            status = result['stability_analysis']['assessment']
            print(f"{ansatz_type.ljust(15)}: {status}")
            if status == 'STABLE':
                stable_ansatze.append(ansatz_type)
        else:
            print(f"{ansatz_type.ljust(15)}: ERROR")
    
    if stable_ansatze:
        print(f"\n✅ Stable ansätze found: {', '.join(stable_ansatze)}")
        print("These configurations show promise for practical warp bubble implementations.")
    else:
        print("\n⚠️ No stable configurations found in this test.")
        print("Consider adjusting parameters or using different ansatz forms.")


if __name__ == "__main__":
    main()
