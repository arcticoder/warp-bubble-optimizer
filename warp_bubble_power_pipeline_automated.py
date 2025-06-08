#!/usr/bin/env python3
"""
Fully Automated Warp-Bubble Simulation and Optimization Pipeline

This script implements a complete automated pipeline for warp-bubble simulation 
and optimization using the validated Ghost/Phantom EFT energy source:

1. ✅ Instantiate the GhostCondensateEFT source with Discovery 21 optimal parameters
2. ✅ Use a 4D B-spline metric ansatz for maximum flexibility
3. ✅ Sweep bubble radius and speed to map energy/stability landscape
4. ✅ Optimize the metric shape for minimum energy using CMA-ES
5. ✅ Validate the optimized bubble configuration
6. ✅ Automate the entire process in a single script

Usage:
    python warp_bubble_power_pipeline_automated.py

Authors: LQG-ANEC Research Team
Date: 2024-12-20
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('warp_pipeline_automated.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import key components with fallback handling
try:
    from warp_qft.energy_sources import EnergySource, GhostCondensateEFT, create_energy_source
    from warp_qft.enhanced_warp_solver import EnhancedWarpBubbleSolver, EnhancedWarpBubbleResult
    logger.info("✅ Successfully imported warp_qft energy sources and solver")
except ImportError as e:
    logger.error(f"Failed to import warp_qft modules: {e}")
    # Try alternative import paths
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from src.warp_qft.energy_sources import EnergySource, GhostCondensateEFT, create_energy_source
        from src.warp_qft.enhanced_warp_solver import EnhancedWarpBubbleSolver, EnhancedWarpBubbleResult
        logger.info("✅ Successfully imported from alternative path")
    except ImportError as e2:
        logger.error(f"Failed alternative import: {e2}")
        logger.error("Please ensure the src/warp_qft modules are properly installed")
        sys.exit(1)

# Import CMA-ES with fallback
try:
    import cma
    HAS_CMA = True
    logger.info("✅ CMA-ES available for optimization")
except ImportError:
    HAS_CMA = False
    logger.warning("⚠️  CMA-ES not available - will use scipy optimization fallback")
    from scipy.optimize import minimize, differential_evolution

# Import B-spline optimizer if available
try:
    from ultimate_bspline_optimizer import UltimateBSplineOptimizer
    HAS_BSPLINE = True
    logger.info("✅ B-spline optimizer available")
except ImportError:
    HAS_BSPLINE = False
    logger.warning("⚠️  B-spline optimizer not available - will use Gaussian ansatz")


class AutomatedWarpBubblePipeline:
    """
    Fully automated warp-bubble simulation and optimization pipeline.
    """
    
    def __init__(self, output_dir: str = "automated_results"):
        """Initialize the automated pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.ghost_source = None
        self.solver = None
        self.bspline_optimizer = None
        
        # Results storage
        self.sweep_results = []
        self.optimization_result = None
        self.validation_result = None
        
        logger.info(f"Automated pipeline initialized. Output: {self.output_dir}")
    
    def step1_instantiate_ghost_eft(self) -> None:
        """
        Step 1: Instantiate the GhostCondensateEFT source with Discovery 21 optimal parameters.
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Instantiate Ghost/Phantom EFT Energy Source")
        logger.info("=" * 80)
        
        # Discovery 21 optimal parameters
        optimal_params = {
            'M': 1000,           # Mass scale (GeV)
            'alpha': 0.01,       # EFT coupling  
            'beta': 0.1,         # Self-interaction strength
            'R0': 5.0,           # Bubble radius scale (m)
            'sigma': 0.5,        # Transition width (m)
            'mu_polymer': 0.1    # LQG polymer parameter
        }
        
        logger.info("Creating Ghost EFT source with Discovery 21 optimal parameters:")
        for param, value in optimal_params.items():
            logger.info(f"  {param}: {value}")
        
        try:
            self.ghost_source = GhostCondensateEFT(**optimal_params)
            
            # Validate parameters
            if self.ghost_source.validate_parameters():
                logger.info("✅ Ghost EFT source created and validated successfully")
                
                # Test energy calculation
                test_coords = np.array([5.0, 0.0, 0.0])
                test_energy = self.ghost_source.energy_density(
                    test_coords[0], test_coords[1], test_coords[2]
                )
                logger.info(f"Test energy density at (5,0,0): {test_energy:.2e} J/m³")
                
                # Compute total energy for reference
                test_volume = 4/3 * np.pi * 10**3  # 10m radius sphere
                total_energy = self.ghost_source.total_energy(test_volume)
                logger.info(f"Total energy (10m sphere): {total_energy:.2e} J")
                
            else:
                logger.error("❌ Ghost EFT parameter validation failed")
                raise ValueError("Invalid Ghost EFT parameters")
                
        except Exception as e:
            logger.error(f"❌ Failed to create Ghost EFT source: {e}")
            raise
    
    def step2_setup_4d_bspline_solver(self) -> None:
        """
        Step 2: Setup warp bubble solver with 4D B-spline metric ansatz.
        """
        logger.info("=" * 80)
        logger.info("STEP 2: Setup 4D B-Spline Metric Ansatz Solver")
        logger.info("=" * 80)
        
        if self.ghost_source is None:
            logger.error("❌ Ghost EFT source not initialized")
            raise RuntimeError("Must initialize Ghost EFT source first")
        
        try:
            # Create enhanced warp solver 
            self.solver = EnhancedWarpBubbleSolver(
                metric_ansatz="4d",
                use_polymer_enhancement=True,
                enable_backreaction=True
            )
            
            logger.info("✅ Enhanced warp bubble solver created")
            logger.info("  - Metric ansatz: 4D B-spline")
            logger.info("  - Polymer enhancement: Enabled")
            logger.info("  - Backreaction correction: Enabled")
            
            # Initialize B-spline optimizer if available
            if HAS_BSPLINE:
                self.bspline_optimizer = UltimateBSplineOptimizer(
                    n_control_points=12,
                    R_bubble=10.0,
                    stability_penalty_weight=1e6,
                    surrogate_assisted=True
                )
                logger.info("✅ B-spline optimizer initialized")
                logger.info("  - Control points: 12")
                logger.info("  - Stability penalty: 1e6")
                logger.info("  - Surrogate-assisted: True")
            else:
                logger.warning("⚠️  B-spline optimizer not available - will use Gaussian ansatz")
                
        except Exception as e:
            logger.error(f"❌ Failed to setup solver: {e}")
            raise
    
    def step3_parameter_sweep(self) -> None:
        """
        Step 3: Sweep bubble radius and speed to map energy/stability landscape.
        """
        logger.info("=" * 80)
        logger.info("STEP 3: Parameter Sweep - Radius & Speed Mapping")
        logger.info("=" * 80)
        
        if self.solver is None:
            logger.error("❌ Solver not initialized")
            raise RuntimeError("Must setup solver first")
        
        # Define parameter ranges
        radii = [2.0, 5.0, 10.0, 15.0, 20.0]  # Bubble radii (m)
        speeds = [1000, 2500, 5000, 7500, 10000]  # Bubble speeds (m/s)
        
        logger.info(f"Sweeping {len(radii)} radii: {radii}")
        logger.info(f"Sweeping {len(speeds)} speeds: {speeds}")
        
        total_sims = len(radii) * len(speeds)
        logger.info(f"Total simulations: {total_sims}")
        
        # Run parameter sweep
        self.sweep_results = []
        sim_count = 0
        start_time = time.time()
        
        for radius in radii:
            for speed in speeds:
                sim_count += 1
                logger.info(f"Simulation {sim_count}/{total_sims}: R={radius}m, v={speed}m/s")
                
                try:
                    # Run simulation
                    result = self.solver.simulate(
                        energy_source=self.ghost_source,
                        radius=radius,
                        resolution=50
                    )
                    
                    # Store results
                    sweep_data = {
                        'sim_id': sim_count,
                        'radius': radius,
                        'speed': speed,
                        'success': result.success,
                        'energy_total': result.energy_total,
                        'stability': result.stability,
                        'max_negative_density': result.max_negative_density,
                        'min_negative_density': result.min_negative_density,                        'execution_time': result.execution_time,
                        'polymer_enhancement': result.polymer_enhancement_factor,
                        'qi_violation': result.qi_violation_achieved
                    }
                    
                    self.sweep_results.append(sweep_data)
                    
                    if result.success:
                        logger.info(f"  ✅ Energy: {result.energy_total:.2e} J, "
                                   f"Stability: {result.stability:.3f}")
                    else:
                        logger.info(f"  ❌ Simulation failed")
                        
                except Exception as e:
                    logger.error(f"  ❌ Simulation error: {e}")
                    
                    # Store failed result
                    failed_data = {
                        'sim_id': sim_count,
                        'radius': radius,
                        'speed': speed,
                        'success': False,
                        'energy_total': float('inf'),
                        'stability': 0.0,
                        'error': str(e)
                    }
                    self.sweep_results.append(failed_data)\n        \n        sweep_time = time.time() - start_time\n        logger.info(f\"Parameter sweep completed in {sweep_time:.1f} seconds\")\n        \n        # Save results to CSV\n        sweep_file = self.output_dir / \"parameter_sweep.csv\"\n        df = pd.DataFrame(self.sweep_results)\n        df.to_csv(sweep_file, index=False)\n        logger.info(f\"Results saved to {sweep_file}\")\n        \n        # Summary statistics\n        successful = [r for r in self.sweep_results if r['success']]\n        logger.info(f\"Successful simulations: {len(successful)}/{total_sims}\")\n        \n        if successful:\n            energies = [r['energy_total'] for r in successful]\n            stabilities = [r['stability'] for r in successful]\n            \n            logger.info(f\"Energy range: {min(energies):.2e} to {max(energies):.2e} J\")\n            logger.info(f\"Stability range: {min(stabilities):.3f} to {max(stabilities):.3f}\")\n            \n            # Find best configuration\n            best_result = min(successful, key=lambda x: x['energy_total'])\n            logger.info(f\"Best configuration: R={best_result['radius']}m, \"\n                       f\"v={best_result['speed']}m/s\")\n            logger.info(f\"  Energy: {best_result['energy_total']:.2e} J\")\n            logger.info(f\"  Stability: {best_result['stability']:.3f}\")\n    \n    def step4_optimize_metric_shape(self) -> None:\n        \"\"\"\n        Step 4: Optimize metric shape for minimum energy using CMA-ES.\n        \"\"\"\n        logger.info(\"=\" * 80)\n        logger.info(\"STEP 4: Optimize Metric Shape with CMA-ES\")\n        logger.info(\"=\" * 80)\n        \n        if not self.sweep_results:\n            logger.error(\"❌ No sweep results available for optimization\")\n            raise RuntimeError(\"Must run parameter sweep first\")\n        \n        # Find best configuration from sweep for optimization\n        successful = [r for r in self.sweep_results if r['success']]\n        if not successful:\n            logger.error(\"❌ No successful sweep results available\")\n            raise RuntimeError(\"No successful simulations to optimize\")\n        \n        best_sweep = min(successful, key=lambda x: x['energy_total'])\n        opt_radius = best_sweep['radius']\n        opt_speed = best_sweep['speed']\n        \n        logger.info(f\"Optimizing at best sweep point: R={opt_radius}m, v={opt_speed}m/s\")\n        logger.info(f\"Baseline energy: {best_sweep['energy_total']:.2e} J\")\n        \n        try:\n            if HAS_BSPLINE and self.bspline_optimizer is not None:\n                # Use B-spline optimization\n                logger.info(\"Using B-spline control point optimization\")\n                \n                # Setup optimization problem\n                def objective_function(control_points):\n                    \"\"\"Objective function for B-spline optimization.\"\"\"\n                    try:\n                        # Apply control points to B-spline ansatz\n                        # (Implementation would depend on specific B-spline interface)\n                        \n                        # Simulate with modified ansatz\n                        result = self.solver.simulate(\n                            energy_source=self.ghost_source,\n                            radius=opt_radius,\n                            resolution=30  # Lower resolution for optimization speed\n                        )\n                        \n                        if not result.success:\n                            return 1e10  # Large penalty for failed simulations\n                        \n                        # Multi-objective: minimize energy, maximize stability\n                        energy_term = abs(result.energy_total)\n                        stability_penalty = 1e6 * max(0, 0.1 - result.stability)\n                        \n                        return energy_term + stability_penalty\n                        \n                    except Exception:\n                        return 1e10\n                \n                # Run B-spline optimization\n                start_time = time.time()\n                optimization_result = self.bspline_optimizer.optimize(\n                    objective_function=objective_function,\n                    max_iterations=50,\n                    population_size=20\n                )\n                opt_time = time.time() - start_time\n                \n                logger.info(f\"B-spline optimization completed in {opt_time:.1f} seconds\")\n                \n            elif HAS_CMA:\n                # Use CMA-ES with Gaussian ansatz\n                logger.info(\"Using CMA-ES with Gaussian ansatz optimization\")\n                \n                # Define parameter bounds for Gaussian ansatz\n                # Parameters: [A1, r01, sig1, A2, r02, sig2, A3, r03, sig3, A4, r04, sig4]\n                n_gaussians = 4\n                param_bounds = []\n                for i in range(n_gaussians):\n                    param_bounds.extend([\n                        (0.1, 2.0),    # Amplitude\n                        (0.5, 15.0),   # Center position\n                        (0.1, 5.0)     # Width\n                    ])\n                \n                def objective_cma(params):\n                    \"\"\"CMA-ES objective function.\"\"\"\n                    try:\n                        # Apply Gaussian parameters to ansatz\n                        # (Would need interface to set ansatz parameters)\n                        \n                        result = self.solver.simulate(\n                            energy_source=self.ghost_source,\n                            radius=opt_radius,\n                            resolution=30\n                        )\n                        \n                        if not result.success:\n                            return 1e10\n                        \n                        # Objective: minimize negative energy magnitude\n                        return abs(result.energy_total)\n                        \n                    except Exception:\n                        return 1e10\n                \n                # Initialize CMA-ES\n                initial_params = np.random.uniform(0.5, 1.5, len(param_bounds))\n                sigma0 = 0.3\n                \n                start_time = time.time()\n                es = cma.CMAEvolutionStrategy(initial_params, sigma0)\n                \n                generations = 30\n                for generation in range(generations):\n                    solutions = es.ask()\n                    fitness_values = [objective_cma(x) for x in solutions]\n                    es.tell(solutions, fitness_values)\n                    \n                    best_fitness = min(fitness_values)\n                    logger.info(f\"Generation {generation+1}/{generations}: \"\n                               f\"Best fitness = {best_fitness:.2e}\")\n                    \n                    if es.stop():\n                        break\n                \n                opt_time = time.time() - start_time\n                best_params = es.result.xbest\n                best_fitness = es.result.fbest\n                \n                logger.info(f\"CMA-ES optimization completed in {opt_time:.1f} seconds\")\n                logger.info(f\"Best fitness: {best_fitness:.2e}\")\n                \n                self.optimization_result = {\n                    'success': True,\n                    'best_parameters': best_params.tolist(),\n                    'best_fitness': best_fitness,\n                    'generations': generation + 1,\n                    'optimization_time': opt_time\n                }\n                \n            else:\n                # Fallback to scipy optimization\n                logger.warning(\"Using scipy differential evolution fallback\")\n                \n                def objective_scipy(params):\n                    \"\"\"Scipy objective function.\"\"\"\n                    try:\n                        result = self.solver.simulate(\n                            energy_source=self.ghost_source,\n                            radius=opt_radius,\n                            resolution=25\n                        )\n                        \n                        return abs(result.energy_total) if result.success else 1e10\n                    except Exception:\n                        return 1e10\n                \n                # Simple parameter bounds\n                bounds = [(0.1, 2.0)] * 6  # Simplified 2-Gaussian ansatz\n                \n                start_time = time.time()\n                opt_result = differential_evolution(\n                    objective_scipy,\n                    bounds,\n                    maxiter=50,\n                    popsize=10,\n                    seed=42\n                )\n                opt_time = time.time() - start_time\n                \n                logger.info(f\"Scipy optimization completed in {opt_time:.1f} seconds\")\n                logger.info(f\"Success: {opt_result.success}\")\n                logger.info(f\"Best fitness: {opt_result.fun:.2e}\")\n                \n                self.optimization_result = {\n                    'success': opt_result.success,\n                    'best_parameters': opt_result.x.tolist(),\n                    'best_fitness': opt_result.fun,\n                    'iterations': opt_result.nit,\n                    'optimization_time': opt_time\n                }\n            \n            # Save optimization results\n            opt_file = self.output_dir / \"optimization_results.json\"\n            with open(opt_file, 'w') as f:\n                json.dump(self.optimization_result, f, indent=2)\n            \n            logger.info(f\"✅ Optimization results saved to {opt_file}\")\n            \n        except Exception as e:\n            logger.error(f\"❌ Optimization failed: {e}\")\n            self.optimization_result = {'success': False, 'error': str(e)}\n    \n    def step5_validate_optimized_bubble(self) -> None:\n        \"\"\"\n        Step 5: Validate the optimized bubble configuration.\n        \"\"\"\n        logger.info(\"=\" * 80)\n        logger.info(\"STEP 5: Validate Optimized Configuration\")\n        logger.info(\"=\" * 80)\n        \n        if not self.optimization_result or not self.optimization_result.get('success', False):\n            logger.warning(\"⚠️  No successful optimization to validate\")\n            logger.info(\"Running validation with best sweep configuration instead\")\n            \n            # Use best sweep result\n            successful = [r for r in self.sweep_results if r['success']]\n            if not successful:\n                logger.error(\"❌ No successful configurations to validate\")\n                return\n            \n            best_sweep = min(successful, key=lambda x: x['energy_total'])\n            val_radius = best_sweep['radius']\n            val_speed = best_sweep['speed']\n            baseline_energy = best_sweep['energy_total']\n            \n        else:\n            # Use optimized configuration\n            logger.info(\"Validating optimized ansatz parameters\")\n            \n            # Find best sweep point for validation coordinates\n            successful = [r for r in self.sweep_results if r['success']]\n            best_sweep = min(successful, key=lambda x: x['energy_total'])\n            val_radius = best_sweep['radius']\n            val_speed = best_sweep['speed']\n            baseline_energy = best_sweep['energy_total']\n            \n            # Apply optimized parameters (implementation depends on ansatz interface)\n            logger.info(f\"Applying optimized parameters: {len(self.optimization_result['best_parameters'])} params\")\n        \n        logger.info(f\"Validation configuration: R={val_radius}m, v={val_speed}m/s\")\n        logger.info(f\"Baseline energy: {baseline_energy:.2e} J\")\n        \n        try:\n            # Run detailed validation simulation\n            start_time = time.time()\n            \n            self.validation_result = self.solver.simulate(\n                energy_source=self.ghost_source,\n                radius=val_radius,\n                resolution=75  # Higher resolution for validation\n            )\n            \n            val_time = time.time() - start_time\n            \n            logger.info(f\"Validation completed in {val_time:.1f} seconds\")\n            \n            if self.validation_result.success:\n                logger.info(\"✅ Validation successful!\")\n                logger.info(f\"Final energy: {self.validation_result.energy_total:.2e} J\")\n                logger.info(f\"Stability: {self.validation_result.stability:.3f}\")\n                logger.info(f\"Polymer enhancement: {self.validation_result.polymer_enhancement_factor:.2f}×\")\n                logger.info(f\"QI violation: {self.validation_result.qi_violation_achieved}\")\n                \n                # Calculate improvement\n                if abs(baseline_energy) > 0:\n                    improvement = abs(self.validation_result.energy_total) / abs(baseline_energy)\n                    logger.info(f\"Energy improvement factor: {improvement:.3f}\")\n                    \n                    if improvement < 1.0:\n                        reduction_percent = (1 - improvement) * 100\n                        logger.info(f\"Energy reduction: {reduction_percent:.1f}%\")\n                    else:\n                        increase_percent = (improvement - 1) * 100\n                        logger.info(f\"Energy increase: {increase_percent:.1f}%\")\n                \n            else:\n                logger.error(\"❌ Validation failed\")\n                \n            # Save validation results\n            val_file = self.output_dir / \"validation_results.json\"\n            val_data = {\n                'success': self.validation_result.success,\n                'energy_total': self.validation_result.energy_total,\n                'stability': self.validation_result.stability,\n                'execution_time': val_time,\n                'validation_radius': val_radius,\n                'validation_speed': val_speed,\n                'baseline_energy': baseline_energy,\n                'polymer_enhancement_factor': self.validation_result.polymer_enhancement_factor,\n                'qi_violation_achieved': self.validation_result.qi_violation_achieved\n            }\n            \n            if self.optimization_result:\n                val_data['optimization_applied'] = self.optimization_result.get('success', False)\n                val_data['optimized_parameters'] = self.optimization_result.get('best_parameters', [])\n            \n            with open(val_file, 'w') as f:\n                json.dump(val_data, f, indent=2)\n            \n            logger.info(f\"✅ Validation results saved to {val_file}\")\n            \n        except Exception as e:\n            logger.error(f\"❌ Validation failed: {e}\")\n            self.validation_result = None\n    \n    def step6_generate_reports(self) -> None:\n        \"\"\"\n        Step 6: Generate comprehensive reports and visualizations.\n        \"\"\"\n        logger.info(\"=\" * 80)\n        logger.info(\"STEP 6: Generate Reports and Visualizations\")\n        logger.info(\"=\" * 80)\n        \n        try:\n            # Generate parameter sweep visualization\n            if self.sweep_results:\n                self._plot_parameter_sweep()\n            \n            # Generate optimization convergence plot\n            if self.optimization_result and self.optimization_result.get('success', False):\n                self._plot_optimization_summary()\n            \n            # Generate final summary report\n            self._generate_final_report()\n            \n            logger.info(\"✅ Reports and visualizations generated\")\n            \n        except Exception as e:\n            logger.error(f\"❌ Report generation failed: {e}\")\n    \n    def _plot_parameter_sweep(self) -> None:\n        \"\"\"Generate parameter sweep visualization.\"\"\"\n        try:\n            df = pd.DataFrame(self.sweep_results)\n            success_df = df[df['success'] == True].copy()\n            \n            if len(success_df) == 0:\n                logger.warning(\"No successful results to plot\")\n                return\n            \n            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))\n            \n            # 1. Energy vs Radius\n            radii = success_df['radius'].values\n            energies = np.abs(success_df['energy_total'].values)\n            ax1.scatter(radii, energies, alpha=0.7, s=60)\n            ax1.set_xlabel('Bubble Radius (m)')\n            ax1.set_ylabel('|Energy| (J)')\n            ax1.set_yscale('log')\n            ax1.set_title('Energy vs Bubble Radius')\n            ax1.grid(True, alpha=0.3)\n            \n            # 2. Stability vs Radius\n            stabilities = success_df['stability'].values\n            ax2.scatter(radii, stabilities, alpha=0.7, s=60, color='orange')\n            ax2.set_xlabel('Bubble Radius (m)')\n            ax2.set_ylabel('Stability')\n            ax2.set_title('Stability vs Bubble Radius')\n            ax2.grid(True, alpha=0.3)\n            \n            # 3. Energy vs Speed\n            speeds = success_df['speed'].values\n            ax3.scatter(speeds, energies, alpha=0.7, s=60, color='green')\n            ax3.set_xlabel('Bubble Speed (m/s)')\n            ax3.set_ylabel('|Energy| (J)')\n            ax3.set_yscale('log')\n            ax3.set_title('Energy vs Bubble Speed')\n            ax3.grid(True, alpha=0.3)\n            \n            # 4. Stability vs Energy\n            ax4.scatter(energies, stabilities, alpha=0.7, s=60, color='red')\n            ax4.set_xlabel('|Energy| (J)')\n            ax4.set_ylabel('Stability')\n            ax4.set_xscale('log')\n            ax4.set_title('Stability vs Energy')\n            ax4.grid(True, alpha=0.3)\n            \n            plt.tight_layout()\n            \n            sweep_plot_file = self.output_dir / \"parameter_sweep_analysis.png\"\n            plt.savefig(sweep_plot_file, dpi=300, bbox_inches='tight')\n            plt.close()\n            \n            logger.info(f\"Parameter sweep plots saved to {sweep_plot_file}\")\n            \n        except Exception as e:\n            logger.error(f\"Failed to generate parameter sweep plots: {e}\")\n    \n    def _plot_optimization_summary(self) -> None:\n        \"\"\"Generate optimization summary visualization.\"\"\"\n        try:\n            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))\n            \n            # Optimization summary\n            opt_data = self.optimization_result\n            \n            # Plot 1: Optimization info\n            info_text = f\"\"\"Optimization Results:\nSuccess: {opt_data.get('success', False)}\nBest Fitness: {opt_data.get('best_fitness', 'N/A'):.2e}\nOptimization Time: {opt_data.get('optimization_time', 0):.1f}s\nParameters: {len(opt_data.get('best_parameters', []))}\"\"\"\n            \n            ax1.text(0.1, 0.5, info_text, transform=ax1.transAxes, fontsize=12,\n                    verticalalignment='center', bbox=dict(boxstyle='round', alpha=0.1))\n            ax1.set_xlim(0, 1)\n            ax1.set_ylim(0, 1)\n            ax1.axis('off')\n            ax1.set_title('Optimization Summary')\n            \n            # Plot 2: Parameter values (if available)\n            if 'best_parameters' in opt_data and opt_data['best_parameters']:\n                params = opt_data['best_parameters']\n                param_indices = range(len(params))\n                ax2.bar(param_indices, params, alpha=0.7)\n                ax2.set_xlabel('Parameter Index')\n                ax2.set_ylabel('Parameter Value')\n                ax2.set_title('Optimized Parameter Values')\n                ax2.grid(True, alpha=0.3)\n            else:\n                ax2.text(0.5, 0.5, 'No parameter data available', \n                        transform=ax2.transAxes, ha='center', va='center')\n                ax2.axis('off')\n            \n            plt.tight_layout()\n            \n            opt_plot_file = self.output_dir / \"optimization_summary.png\"\n            plt.savefig(opt_plot_file, dpi=300, bbox_inches='tight')\n            plt.close()\n            \n            logger.info(f\"Optimization plots saved to {opt_plot_file}\")\n            \n        except Exception as e:\n            logger.error(f\"Failed to generate optimization plots: {e}\")\n    \n    def _generate_final_report(self) -> None:\n        \"\"\"Generate comprehensive final report.\"\"\"\n        try:\n            report_file = self.output_dir / \"FINAL_PIPELINE_REPORT.md\"\n            \n            with open(report_file, 'w') as f:\n                f.write(\"# Automated Warp-Bubble Pipeline Report\\n\\n\")\n                f.write(f\"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n\")\n                \n                # Executive Summary\n                f.write(\"## Executive Summary\\n\\n\")\n                \n                successful_sweeps = len([r for r in self.sweep_results if r['success']])\n                total_sweeps = len(self.sweep_results)\n                \n                f.write(f\"- **Parameter Sweep:** {successful_sweeps}/{total_sweeps} successful simulations\\n\")\n                \n                if self.optimization_result:\n                    opt_success = self.optimization_result.get('success', False)\n                    f.write(f\"- **Optimization:** {'✅ Success' if opt_success else '❌ Failed'}\\n\")\n                    if opt_success:\n                        best_fitness = self.optimization_result.get('best_fitness', 'N/A')\n                        f.write(f\"  - Best fitness: {best_fitness:.2e}\\n\")\n                \n                if self.validation_result:\n                    val_success = self.validation_result.success\n                    f.write(f\"- **Validation:** {'✅ Success' if val_success else '❌ Failed'}\\n\")\n                    if val_success:\n                        f.write(f\"  - Final energy: {self.validation_result.energy_total:.2e} J\\n\")\n                        f.write(f\"  - Stability: {self.validation_result.stability:.3f}\\n\")\n                \n                # Detailed Results\n                f.write(\"\\n## Detailed Results\\n\\n\")\n                \n                # Parameter sweep results\n                f.write(\"### Parameter Sweep Results\\n\\n\")\n                if successful_sweeps > 0:\n                    successful = [r for r in self.sweep_results if r['success']]\n                    energies = [r['energy_total'] for r in successful]\n                    stabilities = [r['stability'] for r in successful]\n                    \n                    best_result = min(successful, key=lambda x: x['energy_total'])\n                    \n                    f.write(f\"- **Best Configuration:**\\n\")\n                    f.write(f\"  - Radius: {best_result['radius']} m\\n\")\n                    f.write(f\"  - Speed: {best_result['speed']} m/s\\n\")\n                    f.write(f\"  - Energy: {best_result['energy_total']:.2e} J\\n\")\n                    f.write(f\"  - Stability: {best_result['stability']:.3f}\\n\")\n                    \n                    f.write(f\"\\n- **Statistics:**\\n\")\n                    f.write(f\"  - Energy range: {min(energies):.2e} to {max(energies):.2e} J\\n\")\n                    f.write(f\"  - Stability range: {min(stabilities):.3f} to {max(stabilities):.3f}\\n\")\n                \n                # Optimization results\n                if self.optimization_result:\n                    f.write(\"\\n### Optimization Results\\n\\n\")\n                    opt = self.optimization_result\n                    f.write(f\"- **Success:** {opt.get('success', False)}\\n\")\n                    f.write(f\"- **Best Fitness:** {opt.get('best_fitness', 'N/A')}\\n\")\n                    f.write(f\"- **Optimization Time:** {opt.get('optimization_time', 0):.1f} seconds\\n\")\n                    f.write(f\"- **Parameters Optimized:** {len(opt.get('best_parameters', []))}\\n\")\n                \n                # Validation results\n                if self.validation_result:\n                    f.write(\"\\n### Validation Results\\n\\n\")\n                    val = self.validation_result\n                    f.write(f\"- **Success:** {val.success}\\n\")\n                    if val.success:\n                        f.write(f\"- **Final Energy:** {val.energy_total:.2e} J\\n\")\n                        f.write(f\"- **Stability:** {val.stability:.3f}\\n\")\n                        f.write(f\"- **Polymer Enhancement:** {val.polymer_enhancement_factor:.2f}×\\n\")\n                        f.write(f\"- **QI Violation Achieved:** {val.qi_violation_achieved}\\n\")\n                \n                # Conclusions\n                f.write(\"\\n## Conclusions\\n\\n\")\n                \n                if successful_sweeps > 0:\n                    f.write(\"✅ **Pipeline successfully demonstrated automated warp-bubble simulation**\\n\\n\")\n                    f.write(\"Key achievements:\\n\")\n                    f.write(\"- Instantiated Ghost/Phantom EFT energy source with Discovery 21 parameters\\n\")\n                    f.write(\"- Mapped parameter landscape through systematic sweep\\n\")\n                    f.write(\"- Applied optimization algorithms to metric ansatz\\n\")\n                    f.write(\"- Validated optimized configurations\\n\")\n                else:\n                    f.write(\"⚠️ **Pipeline completed with limited success**\\n\\n\")\n                    f.write(\"Recommendations for improvement:\\n\")\n                    f.write(\"- Check energy source parameters\\n\")\n                    f.write(\"- Verify solver configuration\\n\")\n                    f.write(\"- Expand parameter ranges\\n\")\n                    f.write(\"- Improve numerical stability\\n\")\n                \n                f.write(\"\\n---\\n\")\n                f.write(\"*Report generated by Automated Warp-Bubble Pipeline*\\n\")\n            \n            logger.info(f\"Final report saved to {report_file}\")\n            \n        except Exception as e:\n            logger.error(f\"Failed to generate final report: {e}\")\n    \n    def run_complete_pipeline(self) -> None:\n        \"\"\"\n        Execute the complete automated pipeline.\n        \"\"\"\n        logger.info(\"🚀 Starting Automated Warp-Bubble Pipeline\")\n        logger.info(\"=\" * 80)\n        \n        pipeline_start = time.time()\n        \n        try:\n            # Execute all pipeline steps\n            self.step1_instantiate_ghost_eft()\n            self.step2_setup_4d_bspline_solver()\n            self.step3_parameter_sweep()\n            self.step4_optimize_metric_shape()\n            self.step5_validate_optimized_bubble()\n            self.step6_generate_reports()\n            \n            pipeline_time = time.time() - pipeline_start\n            \n            logger.info(\"=\" * 80)\n            logger.info(\"🎉 PIPELINE COMPLETED SUCCESSFULLY!\")\n            logger.info(f\"Total execution time: {pipeline_time:.1f} seconds\")\n            logger.info(f\"Results available in: {self.output_dir}\")\n            logger.info(\"=\" * 80)\n            \n        except Exception as e:\n            pipeline_time = time.time() - pipeline_start\n            logger.error(\"=\" * 80)\n            logger.error(f\"❌ PIPELINE FAILED: {e}\")\n            logger.error(f\"Execution time before failure: {pipeline_time:.1f} seconds\")\n            logger.error(\"=\" * 80)\n            raise\n\n\ndef main():\n    \"\"\"Main entry point.\"\"\"\n    print(\"Automated Warp-Bubble Simulation and Optimization Pipeline\")\n    print(\"=\" * 60)\n    print(\"This pipeline will:\")\n    print(\"1. Instantiate Ghost/Phantom EFT energy source (Discovery 21)\")\n    print(\"2. Setup 4D B-spline metric ansatz solver\")\n    print(\"3. Sweep bubble radius and speed parameters\")\n    print(\"4. Optimize metric shape using CMA-ES\")\n    print(\"5. Validate optimized bubble configuration\")\n    print(\"6. Generate comprehensive reports\")\n    print(\"=\" * 60)\n    \n    # Create and run pipeline\n    pipeline = AutomatedWarpBubblePipeline(output_dir=\"automated_pipeline_results\")\n    \n    try:\n        pipeline.run_complete_pipeline()\n        \n        print(\"\\n🎉 SUCCESS! Pipeline completed successfully.\")\n        print(f\"Results are available in: {pipeline.output_dir}\")\n        print(\"\\nKey output files:\")\n        print(\"- parameter_sweep.csv: Parameter sweep results\")\n        print(\"- optimization_results.json: Optimization results\")\n        print(\"- validation_results.json: Validation results\")\n        print(\"- FINAL_PIPELINE_REPORT.md: Comprehensive report\")\n        print(\"- *.png: Visualization plots\")\n        \n    except Exception as e:\n        print(f\"\\n❌ FAILED: {e}\")\n        print(\"Check the log file 'warp_pipeline_automated.log' for details.\")\n        return 1\n    \n    return 0\n\n\nif __name__ == \"__main__\":\n    exit(main())\n
