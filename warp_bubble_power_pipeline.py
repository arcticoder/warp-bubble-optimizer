#!/usr/bin/env python3
"""
Warp Bubble Power Pipeline: Complete Automation

This script implements the complete automated pipeline for warp bubble 
simulation, optimization, and validation as described in the roadmap:

1. Integrate Discovery 21 Ghost/Phantom EFT energy source
2. Choose and configure metric ansatz (4D B-spline)
3. Sweep bubble radius & speed parameters
4. Optimize metric shape for minimum energy using CMA-ES
5. Validate optimized bubble configuration
6. Generate comprehensive reports and visualizations

Usage:
    python warp_bubble_power_pipeline.py [--config config.json] [--output results/]
"""

import sys
import argparse
import json
import csv
from pathlib import Path
import time
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('warp_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import warp bubble components
try:
    from src.warp_qft.integrated_warp_solver import (
        WarpBubbleSolver, 
        create_optimal_ghost_solver,
        WarpSimulationResult
    )
    from src.warp_qft.cmaes_optimization import (
        CMAESOptimizer,
        create_4d_optimizer,
        create_hybrid_optimizer,
        OptimizationResult
    )
    from src.warp_qft.energy_sources import GhostCondensateEFT
except ImportError:
    # Fallback for different import structure
    try:
        from warp_qft.integrated_warp_solver import (
            WarpBubbleSolver,
            create_optimal_ghost_solver, 
            WarpSimulationResult
        )
        from warp_qft.cmaes_optimization import (
            CMAESOptimizer,
            create_4d_optimizer,
            create_hybrid_optimizer,
            OptimizationResult
        )
        from warp_qft.energy_sources import GhostCondensateEFT
    except ImportError:
        logger.error("Failed to import warp_qft modules. Check installation.")
        sys.exit(1)

# Visualization imports (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
except ImportError:
    HAS_PLOTTING = False
    logger.warning("Matplotlib/seaborn not available. Plots will be skipped.")


class WarpBubblePowerPipeline:
    """
    Complete automated pipeline for warp bubble power analysis.
    
    Integrates parameter sweeps, optimization, and validation into
    a single automated workflow.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "results"):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.solver = None
        self.optimizer = None
        
        # Results storage
        self.sweep_results = []
        self.optimization_result = None
        self.validation_result = None
        
        logger.info(f"Pipeline initialized. Output dir: {self.output_dir}")
    
    def step1_initialize_components(self) -> None:
        """
        Step 1: Initialize Ghost EFT energy source and solver.
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Initialize Ghost/Phantom EFT Components")
        logger.info("=" * 60)
        
        # Get energy source parameters from config
        source_config = self.config.get('energy_source', {})
        
        # Create optimal Ghost EFT source with Discovery 21 parameters
        ghost_params = {
            'M': source_config.get('M', 1000),
            'alpha': source_config.get('alpha', 0.01),
            'beta': source_config.get('beta', 0.1),
            'R0': source_config.get('R0', 5.0),
            'sigma': source_config.get('sigma', 0.2),
            'mu_polymer': source_config.get('mu_polymer', 0.1)
        }
        
        logger.info(f"Creating Ghost EFT source with parameters: {ghost_params}")
        
        ghost_source = GhostCondensateEFT(**ghost_params)
        
        # Create integrated solver
        metric_ansatz = self.config.get('metric_ansatz', '4d')
        enable_backreaction = self.config.get('enable_backreaction', True)
        enable_stability = self.config.get('enable_stability', True)
        
        self.solver = WarpBubbleSolver(
            metric_ansatz=metric_ansatz,
            energy_source=ghost_source,
            enable_backreaction=enable_backreaction,
            enable_stability=enable_stability
        )
        
        logger.info(f"Solver created with ansatz: {metric_ansatz}")
        logger.info(f"Backreaction enabled: {enable_backreaction}")
        logger.info(f"Stability analysis enabled: {enable_stability}")
        
        # Save initial configuration
        config_file = self.output_dir / "pipeline_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Configuration saved to {config_file}")
    
    def step2_parameter_sweep(self) -> None:
        """
        Step 2: Sweep bubble radius and speed parameters.
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Parameter Sweep (Radius & Speed)")
        logger.info("=" * 60)
          # Get sweep parameters
        sweep_config = self.config.get('parameter_sweep', {})
        
        radii = sweep_config.get('radii', [5.0, 10.0, 20.0])
        speeds = sweep_config.get('speeds', [1000, 5000, 10000])
        
        logger.info(f"Sweeping {len(radii)} radii: {radii}")
        logger.info(f"Sweeping {len(speeds)} speeds: {speeds}")
        
        total_sims = len(radii) * len(speeds)
        logger.info(f"Total simulations: {total_sims}")
        
        # Run sweep
        self.sweep_results = []
        sim_count = 0
        
        for radius in radii:
            for speed in speeds:
                sim_count += 1
                logger.info(f"Simulation {sim_count}/{total_sims}: R={radius}m, v={speed}m/s")
                
                try:
                    result = self.solver.simulate(radius=radius, speed=speed)
                    
                    # Store result with sweep parameters
                    sweep_data = result.to_dict()
                    sweep_data.update({
                        'sim_id': sim_count,
                        'radius_param': radius,
                        'speed_param': speed
                    })
                    
                    self.sweep_results.append(sweep_data)
                    
                    logger.info(f"  → Energy: {result.energy_total:.2e} J, "
                               f"Stability: {result.stability:.3f}")
                    
                except Exception as e:
                    logger.error(f"  → Simulation failed: {e}")
                    
                    # Store failed result
                    failed_data = {
                        'sim_id': sim_count,
                        'radius_param': radius,
                        'speed_param': speed,
                        'success': False,
                        'energy_total_J': float('inf'),
                        'stability': 0.0,
                        'error': str(e)
                    }
                    self.sweep_results.append(failed_data)
        
        # Save sweep results
        sweep_file = self.output_dir / "power_sweep.csv"
        df = pd.DataFrame(self.sweep_results)
        df.to_csv(sweep_file, index=False)
        
        logger.info(f"Parameter sweep completed. Results saved to {sweep_file}")
        
        # Print summary
        successful_sims = sum(1 for r in self.sweep_results if r.get('success', False))
        logger.info(f"Successful simulations: {successful_sims}/{total_sims}")
        
        if successful_sims > 0:
            energies = [r['energy_total_J'] for r in self.sweep_results 
                       if r.get('success', False) and np.isfinite(r['energy_total_J'])]
            if energies:
                min_energy = min(energies)
                max_energy = max(energies)
                logger.info(f"Energy range: {min_energy:.2e} to {max_energy:.2e} J")
    
    def step3_optimize_ansatz(self) -> None:
        """
        Step 3: Optimize metric ansatz parameters using CMA-ES.
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Ansatz Optimization with CMA-ES")
        logger.info("=" * 60)
        
        # Get optimization parameters
        opt_config = self.config.get('optimization', {})
        
        # Choose optimizer based on ansatz type
        ansatz_type = self.config.get('metric_ansatz', '4d')
        fixed_radius = opt_config.get('fixed_radius', 10.0)
        fixed_speed = opt_config.get('fixed_speed', 5000.0)
        
        logger.info(f"Optimizing {ansatz_type} ansatz at R={fixed_radius}m, v={fixed_speed}m/s")
        
        if ansatz_type == '4d':
            self.optimizer = create_4d_optimizer(
                self.solver, fixed_radius, fixed_speed
            )
        elif ansatz_type == 'hybrid':
            self.optimizer = create_hybrid_optimizer(
                self.solver, fixed_radius, fixed_speed
            )
        else:
            logger.error(f"Unknown ansatz type: {ansatz_type}")
            return
        
        # Set optimization weights if provided
        energy_weight = opt_config.get('energy_weight', 1.0)
        stability_weight = opt_config.get('stability_weight', 0.5)
        
        self.optimizer.energy_weight = energy_weight
        self.optimizer.stability_weight = stability_weight
        
        logger.info(f"Optimization weights - Energy: {energy_weight}, Stability: {stability_weight}")
        
        # Run optimization
        generations = opt_config.get('generations', 50)
        pop_size = opt_config.get('population_size', 20)
        sigma0 = opt_config.get('initial_step_size', 0.3)
        seed = opt_config.get('random_seed', None)
        
        logger.info(f"CMA-ES parameters - Generations: {generations}, "
                   f"Population: {pop_size}, σ₀: {sigma0}")
        
        start_time = time.time()
        
        self.optimization_result = self.optimizer.optimize(
            generations=generations,
            pop_size=pop_size,
            sigma0=sigma0,
            seed=seed
        )
        
        opt_time = time.time() - start_time
        
        logger.info(f"Optimization completed in {opt_time:.1f} seconds")
        logger.info(f"Success: {self.optimization_result.success}")
        logger.info(f"Best score: {self.optimization_result.best_score:.3e}")
        logger.info(f"Best energy: {self.optimization_result.best_energy:.2e} J")
        logger.info(f"Best stability: {self.optimization_result.best_stability:.3f}")
        logger.info(f"Function evaluations: {self.optimization_result.function_evaluations}")
        
        # Save optimization results
        opt_file = self.output_dir / "optimization_results.json"
        self.optimizer.save_results(self.optimization_result, str(opt_file))
        
        logger.info(f"Optimization results saved to {opt_file}")
    
    def step4_validate_optimized_bubble(self) -> None:
        """
        Step 4: Validate the optimized bubble configuration.
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Validate Optimized Configuration")
        logger.info("=" * 60)
        
        if self.optimization_result is None:
            logger.error("No optimization result available for validation")
            return
        
        if not self.optimization_result.success:
            logger.warning("Optimization was not successful, validating anyway")
        
        # Apply optimized parameters
        best_params = self.optimization_result.best_parameters
        logger.info(f"Applying optimized parameters: {best_params}")
        
        self.solver.set_ansatz_parameters(best_params)
        
        # Run detailed validation simulation
        validation_config = self.config.get('validation', {})
        radius = validation_config.get('radius', 10.0)
        speed = validation_config.get('speed', 5000.0)
        
        logger.info(f"Running validation at R={radius}m, v={speed}m/s")
        
        start_time = time.time()
        
        self.validation_result = self.solver.simulate(
            radius=radius,
            speed=speed,
            detailed_analysis=True
        )
        
        val_time = time.time() - start_time
        
        logger.info(f"Validation completed in {val_time:.1f} seconds")
        logger.info(f"Validation success: {self.validation_result.success}")
        
        if self.validation_result.success:
            logger.info(f"Final energy requirement: {self.validation_result.energy_total:.2e} J")
            logger.info(f"Energy reduction factor: {self.validation_result.energy_reduction_factor:.3f}")
            logger.info(f"Energy reduction: {(1-self.validation_result.energy_reduction_factor)*100:.1f}%")
            logger.info(f"Stability score: {self.validation_result.stability:.3f}")
            logger.info(f"Execution time: {self.validation_result.execution_time:.3f} s")
            
            # Compare with original energy
            original_energy = self.validation_result.energy_original
            final_energy = self.validation_result.energy_total
            total_reduction = (original_energy - final_energy) / original_energy * 100
            
            logger.info(f"Total energy reduction: {total_reduction:.1f}%")
            
        else:
            logger.error("Validation simulation failed")
            for warning in self.validation_result.warnings:
                logger.warning(f"  {warning}")
        
        # Save validation results
        val_file = self.output_dir / "validation_results.json"
        val_data = self.validation_result.to_dict()
        val_data['optimized_parameters'] = best_params
        
        with open(val_file, 'w') as f:
            json.dump(val_data, f, indent=2)
        
        logger.info(f"Validation results saved to {val_file}")
    
    def step5_generate_visualizations(self) -> None:
        """
        Step 5: Generate plots and visualizations.
        """
        logger.info("=" * 60)
        logger.info("STEP 5: Generate Visualizations")
        logger.info("=" * 60)
        
        if not HAS_PLOTTING:
            logger.warning("Plotting libraries not available. Skipping visualizations.")
            return
        
        # 1. Parameter sweep heatmap
        if self.sweep_results:
            self._plot_parameter_sweep()
        
        # 2. Optimization convergence
        if self.optimization_result:
            self._plot_optimization_convergence()
        
        # 3. Summary report
        self._generate_summary_report()
        
        logger.info("Visualizations completed")
    
    def _plot_parameter_sweep(self) -> None:
        """Generate parameter sweep visualization."""
        try:
            df = pd.DataFrame(self.sweep_results)
            
            # Filter successful simulations
            success_df = df[df['success'] == True].copy()
            
            if len(success_df) == 0:
                logger.warning("No successful simulations for plotting")
                return
            
            # Create pivot table for heatmap
            pivot_energy = success_df.pivot(
                index='radius_param', 
                columns='speed_param', 
                values='energy_total_J'
            )
            
            pivot_stability = success_df.pivot(
                index='radius_param',
                columns='speed_param', 
                values='stability'
            )
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Energy heatmap
            sns.heatmap(
                np.log10(np.abs(pivot_energy)), 
                annot=True, 
                fmt='.1f',
                cmap='viridis',
                ax=ax1,
                cbar_kws={'label': 'log₁₀|Energy| (J)'}
            )
            ax1.set_title('Energy Requirements vs Radius & Speed')
            ax1.set_xlabel('Speed (m/s)')
            ax1.set_ylabel('Radius (m)')
            
            # Stability heatmap
            sns.heatmap(
                pivot_stability,
                annot=True,
                fmt='.2f', 
                cmap='RdYlGn',
                ax=ax2,
                cbar_kws={'label': 'Stability Score'}
            )
            ax2.set_title('Stability vs Radius & Speed')
            ax2.set_xlabel('Speed (m/s)')
            ax2.set_ylabel('Radius (m)')
            
            plt.tight_layout()
            
            sweep_plot_file = self.output_dir / "parameter_sweep_heatmap.png"
            plt.savefig(sweep_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Parameter sweep heatmap saved to {sweep_plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to create parameter sweep plot: {e}")
    
    def _plot_optimization_convergence(self) -> None:
        """Generate optimization convergence plot."""
        try:
            convergence_history = self.optimization_result.convergence_history
            
            if not convergence_history:
                logger.warning("No convergence history available")
                return
            
            plt.figure(figsize=(10, 6))
            
            generations = np.arange(len(convergence_history))
            plt.plot(generations, convergence_history, 'b-', linewidth=2)
            plt.xlabel('Function Evaluations')
            plt.ylabel('Objective Value')
            plt.title('CMA-ES Optimization Convergence')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            # Mark best point
            best_idx = np.argmin(convergence_history)
            plt.plot(best_idx, convergence_history[best_idx], 'ro', markersize=8, 
                    label=f'Best: {convergence_history[best_idx]:.2e}')
            plt.legend()
            
            convergence_plot_file = self.output_dir / "optimization_convergence.png"
            plt.savefig(convergence_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Convergence plot saved to {convergence_plot_file}")
            
        except Exception as e:
            logger.error(f"Failed to create convergence plot: {e}")
    
    def _generate_summary_report(self) -> None:
        """Generate summary report."""
        report_file = self.output_dir / "pipeline_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("WARP BUBBLE POWER PIPELINE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Configuration
            f.write("CONFIGURATION:\n")
            f.write(f"Energy Source: {self.config.get('energy_source', {})}\n")
            f.write(f"Metric Ansatz: {self.config.get('metric_ansatz', 'unknown')}\n")
            f.write(f"Backreaction: {self.config.get('enable_backreaction', False)}\n")
            f.write(f"Stability: {self.config.get('enable_stability', False)}\n\n")
            
            # Parameter sweep results
            if self.sweep_results:
                successful = sum(1 for r in self.sweep_results if r.get('success', False))
                total = len(self.sweep_results)
                f.write(f"PARAMETER SWEEP:\n")
                f.write(f"Total simulations: {total}\n")
                f.write(f"Successful: {successful}\n")
                f.write(f"Success rate: {successful/total*100:.1f}%\n\n")
            
            # Optimization results
            if self.optimization_result:
                f.write("OPTIMIZATION RESULTS:\n")
                f.write(f"Success: {self.optimization_result.success}\n")
                f.write(f"Best energy: {self.optimization_result.best_energy:.2e} J\n")
                f.write(f"Best stability: {self.optimization_result.best_stability:.3f}\n")
                f.write(f"Function evaluations: {self.optimization_result.function_evaluations}\n")
                f.write(f"Execution time: {self.optimization_result.execution_time:.1f} s\n")
                f.write(f"Best parameters: {self.optimization_result.best_parameters}\n\n")
            
            # Validation results
            if self.validation_result:
                f.write("VALIDATION RESULTS:\n")
                f.write(f"Success: {self.validation_result.success}\n")
                f.write(f"Final energy: {self.validation_result.energy_total:.2e} J\n")
                f.write(f"Energy reduction: {(1-self.validation_result.energy_reduction_factor)*100:.1f}%\n")
                f.write(f"Stability: {self.validation_result.stability:.3f}\n")
                f.write(f"Warnings: {len(self.validation_result.warnings)}\n")
        
        logger.info(f"Summary report saved to {report_file}")
    
    def run_complete_pipeline(self) -> None:
        """
        Run the complete automated pipeline.
        """
        logger.info("🚀 Starting Warp Bubble Power Pipeline")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        pipeline_start = time.time()
        
        try:
            # Execute all pipeline steps
            self.step1_initialize_components()
            self.step2_parameter_sweep()
            self.step3_optimize_ansatz()
            self.step4_validate_optimized_bubble()
            self.step5_generate_visualizations()
            
            total_time = time.time() - pipeline_start
            
            logger.info("=" * 60)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Total execution time: {total_time:.1f} seconds")
            logger.info(f"Results saved to: {self.output_dir}")
            
            # Final summary
            if self.validation_result and self.validation_result.success:
                logger.info(f"🎯 FINAL RESULTS:")
                logger.info(f"   Energy requirement: {self.validation_result.energy_total:.2e} J")
                logger.info(f"   Energy reduction: {(1-self.validation_result.energy_reduction_factor)*100:.1f}%")
                logger.info(f"   Stability score: {self.validation_result.stability:.3f}")
                
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if Path(config_file).exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"Config file {config_file} not found. Using defaults.")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default pipeline configuration."""
    return {
        "energy_source": {
            "M": 1000,
            "alpha": 0.01,
            "beta": 0.1,
            "R0": 5.0,
            "sigma": 0.2,
            "mu_polymer": 0.1
        },
        "metric_ansatz": "4d",
        "enable_backreaction": True,
        "enable_stability": True,
        "parameter_sweep": {
            "radii": [5.0, 10.0, 20.0],
            "speeds": [1000, 5000, 10000]
        },
        "optimization": {
            "fixed_radius": 10.0,
            "fixed_speed": 5000.0,
            "generations": 30,
            "population_size": 15,
            "initial_step_size": 0.3,
            "energy_weight": 1.0,
            "stability_weight": 0.5,
            "random_seed": 42
        },
        "validation": {
            "radius": 10.0,
            "speed": 5000.0
        }
    }


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Automated Warp Bubble Power Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick power sweep for initial analysis
  python warp_bubble_power_pipeline.py --quick --output-dir results/quick_sweep
  
  # Validate existing results
  python warp_bubble_power_pipeline.py --validate-only --output-dir results/quick_sweep
  
  # Full optimization pipeline
  python warp_bubble_power_pipeline.py --config production_config.json --output-dir results/full_optimization
  
  # Extract optimal parameters from previous run
  python warp_bubble_power_pipeline.py --extract-params --output-dir results/full_optimization
        """
    )
    
    parser.add_argument(
        '--config', 
        default='pipeline_config.json',
        help='Configuration file (JSON format)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    
    mode_group.add_argument(
        '--quick',
        action='store_true',
        help='Run quick parameter sweep only (no optimization)'
    )
    
    mode_group.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate existing optimization results only'
    )
    
    mode_group.add_argument(
        '--extract-params',
        action='store_true',
        help='Extract optimal parameters from existing results'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite existing results'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Load configuration
    config = load_config(args.config)
    
    # Create pipeline
    pipeline = WarpBubblePowerPipeline(config, args.output_dir)
    
    # Run based on mode
    if args.quick:
        pipeline.run_quick_sweep()
    elif args.validate_only:
        pipeline.run_validation_only()
    elif args.extract_params:
        pipeline.extract_optimal_parameters()
    else:
        pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
