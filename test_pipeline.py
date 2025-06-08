#!/usr/bin/env python3
"""
Test Script for Warp Bubble Power Pipeline

This script tests the core components of the pipeline to ensure
everything is working correctly before running the full automation.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        # Core pipeline components
        from src.warp_qft.integrated_warp_solver import (
            WarpBubbleSolver, 
            create_optimal_ghost_solver
        )
        from src.warp_qft.cmaes_optimization import (
            CMAESOptimizer,
            create_4d_optimizer
        )
        from src.warp_qft.energy_sources import GhostCondensateEFT
        
        logger.info("✅ Core pipeline imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        
        # Try alternative import structure
        try:
            from warp_qft.integrated_warp_solver import (
                WarpBubbleSolver,
                create_optimal_ghost_solver
            )
            logger.info("✅ Alternative import structure works")
            return True
        except ImportError as e2:
            logger.error(f"❌ Alternative imports also failed: {e2}")
            return False

def test_ghost_source():
    """Test Ghost EFT energy source creation."""
    logger.info("Testing Ghost EFT energy source...")
    
    try:
        from src.warp_qft.energy_sources import GhostCondensateEFT
        
        # Create source with Discovery 21 parameters
        ghost = GhostCondensateEFT(
            M=1000,
            alpha=0.01,
            beta=0.1,
            R0=5.0,
            sigma=0.2,
            mu_polymer=0.1
        )
        
        # Test energy density calculation
        import numpy as np
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.0, 0.0, 0.0])
        z = np.array([0.0, 0.0, 0.0])
        
        energy_density = ghost.energy_density(x, y, z, t=0.0)
        
        logger.info(f"✅ Ghost EFT source created successfully")
        logger.info(f"   Energy density shape: {energy_density.shape}")
        logger.info(f"   Energy density range: [{np.min(energy_density):.2e}, {np.max(energy_density):.2e}]")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ghost source test failed: {e}")
        return False

def test_solver_creation():
    """Test integrated solver creation."""
    logger.info("Testing integrated solver creation...")
    
    try:
        from src.warp_qft.integrated_warp_solver import create_optimal_ghost_solver
        
        # Create solver
        solver = create_optimal_ghost_solver()
        
        logger.info(f"✅ Solver created successfully")
        logger.info(f"   Metric ansatz: {solver.metric_ansatz}")
        logger.info(f"   Energy source: {solver.energy_source.name}")
        logger.info(f"   Backreaction enabled: {solver.enable_backreaction}")
        logger.info(f"   Stability enabled: {solver.enable_stability}")
        
        return True, solver
        
    except Exception as e:
        logger.error(f"❌ Solver creation failed: {e}")
        return False, None

def test_simple_simulation(solver):
    """Test a simple simulation."""
    logger.info("Testing simple simulation...")
    
    try:
        # Run simple simulation
        result = solver.simulate(radius=5.0, speed=1000.0)
        
        logger.info(f"✅ Simulation completed")
        logger.info(f"   Success: {result.success}")
        logger.info(f"   Energy: {result.energy_total:.2e} J")
        logger.info(f"   Stability: {result.stability:.3f}")
        logger.info(f"   Execution time: {result.execution_time:.3f} s")
        
        return result.success
        
    except Exception as e:
        logger.error(f"❌ Simulation failed: {e}")
        return False

def test_optimization_setup():
    """Test optimization setup."""
    logger.info("Testing optimization setup...")
    
    try:
        from src.warp_qft.integrated_warp_solver import create_optimal_ghost_solver
        from src.warp_qft.cmaes_optimization import create_4d_optimizer
        
        # Create solver and optimizer
        solver = create_optimal_ghost_solver()
        optimizer = create_4d_optimizer(solver, fixed_radius=5.0, fixed_speed=1000.0)
        
        logger.info(f"✅ Optimizer created successfully")
        logger.info(f"   Parameters to optimize: {optimizer.param_names}")
        logger.info(f"   Parameter bounds: {optimizer.bounds}")
        logger.info(f"   Fixed radius: {optimizer.fixed_radius}")
        logger.info(f"   Fixed speed: {optimizer.fixed_speed}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimization setup failed: {e}")
        return False

def test_pipeline_config():
    """Test pipeline configuration loading."""
    logger.info("Testing pipeline configuration...")
    
    try:
        import json
        from pathlib import Path
        
        config_file = Path("pipeline_config.json")
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            logger.info(f"✅ Configuration loaded successfully")
            logger.info(f"   Energy source params: {config.get('energy_source', {})}")
            logger.info(f"   Metric ansatz: {config.get('metric_ansatz', 'unknown')}")
            logger.info(f"   Sweep radii: {config.get('parameter_sweep', {}).get('radii', [])}")
            
            return True
        else:
            logger.warning(f"⚠️  Config file not found: {config_file}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Config test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Running Warp Bubble Pipeline Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Ghost Source Test", test_ghost_source),
        ("Solver Creation Test", test_solver_creation),
        ("Optimization Setup Test", test_optimization_setup),
        ("Pipeline Config Test", test_pipeline_config)
    ]
    
    results = []
    solver = None
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 {test_name}")
        logger.info("-" * 30)
        
        if test_name == "Solver Creation Test":
            success, solver = test_func()
        else:
            success = test_func()
        
        results.append((test_name, success))
        
        if not success:
            logger.warning(f"⚠️  {test_name} failed - some features may not work")
    
    # Run simulation test if solver was created
    if solver is not None:
        logger.info(f"\n🔍 Simple Simulation Test")
        logger.info("-" * 30)
        sim_success = test_simple_simulation(solver)
        results.append(("Simple Simulation Test", sim_success))
    
    # Summary
    logger.info("\n📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for name, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! Pipeline should work correctly.")
    elif passed >= total * 0.8:
        logger.info("⚠️  Most tests passed. Pipeline should work with some limitations.")
    else:
        logger.info("❌ Many tests failed. Check dependencies and installation.")

if __name__ == "__main__":
    main()
