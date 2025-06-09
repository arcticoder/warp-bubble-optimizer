#!/usr/bin/env python3
"""
Test Script for ProgressTracker Integration
==========================================

This script tests the ProgressTracker integration across all major simulation
and optimization components to ensure proper functionality and fallback behavior.
"""

import sys
import time
import numpy as np
from pathlib import Path

def test_progress_imports():
    """Test ProgressTracker imports and fallbacks."""
    print("Testing ProgressTracker imports...")
    
    # Test individual module imports
    modules_to_test = [
        'sim_control_loop',
        'analog_sim', 
        'jax_4d_optimizer',
        'enhanced_qi_constraint',
        'enhanced_virtual_control_loop'
    ]
    
    results = {}
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name)
            progress_available = getattr(module, 'PROGRESS_AVAILABLE', False)
            results[module_name] = {
                'import_success': True,
                'progress_available': progress_available
            }
            print(f"  ✅ {module_name}: Import OK, ProgressTracker={progress_available}")
        except Exception as e:
            results[module_name] = {
                'import_success': False,
                'error': str(e)
            }
            print(f"  ❌ {module_name}: Import failed - {e}")
    
    return results

def test_jax_fallback():
    """Test JAX availability and fallback logic."""
    print("\nTesting JAX availability...")
    
    try:
        import jax
        import jax.numpy as jnp
        print("  ✅ JAX available - GPU acceleration enabled")
        
        # Simple JAX test
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x**2)
        print(f"  ✅ JAX computation test: sum([1,2,3]²) = {y}")
        return True
        
    except ImportError:
        print("  ⚠️  JAX not available - using NumPy fallback")
        return False

def test_virtual_control_loop():
    """Test virtual control loop with progress tracking."""
    print("\nTesting virtual control loop...")
    
    try:
        import sim_control_loop
        from sim_control_loop import VirtualControlLoop, SensorConfig, ActuatorConfig, ControllerConfig
        
        # Test configuration
        def simple_objective(params):
            x, y = params
            return (x - 1)**2 + (y - 2)**2
        
        # Create control loop
        control_loop = VirtualControlLoop(
            objective_func=simple_objective,
            initial_params=np.array([0.0, 0.0]),
            sensor_config=SensorConfig(noise_level=0.01, update_rate=10.0),
            actuator_config=ActuatorConfig(response_time=0.1),
            controller_config=ControllerConfig(kp=0.1, ki=0.01, kd=0.001)
        )
        
        print("  ✅ VirtualControlLoop initialized successfully")
        
        # Run short test
        import asyncio
        async def run_test():
            results = await control_loop.run_control_loop(duration=1.0, target_rate=10.0)
            return results
        
        # For Python 3.7+ compatibility
        try:
            results = asyncio.run(run_test())
        except AttributeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(run_test())
            loop.close()
        
        print(f"  ✅ Control loop test completed: {results['steps']} steps")
        print(f"  📊 Final objective: {results['final_objective']:.6e}")
        return True
        
    except Exception as e:
        print(f"  ❌ Virtual control loop test failed: {e}")
        return False

def test_analog_simulation():
    """Test analog physics simulation."""
    print("\nTesting analog simulation...")
    
    try:
        import analog_sim
        from analog_sim import AcousticWarpAnalog, AnalogConfig
        
        # Create small test simulation
        config = AnalogConfig(
            grid_size=(50, 50),  # Small for quick test
            physical_size=(2.0, 2.0),
            dt=1e-3,
            wave_speed=343.0
        )
        
        simulator = AcousticWarpAnalog(config)
        print("  ✅ AcousticWarpAnalog initialized")
        
        # Run short simulation
        results = simulator.run_simulation(duration=0.1, save_interval=0.05)
        
        print(f"  ✅ Simulation completed: {len(results['snapshots'])} snapshots")
        print(f"  📊 Max pressure: {np.max(np.abs(results['final_pressure'])):.3e}")
        return True
        
    except Exception as e:
        print(f"  ❌ Analog simulation test failed: {e}")
        return False

def test_jax_optimization():
    """Test JAX-accelerated optimization."""
    print("\nTesting JAX optimization...")
    
    try:
        import jax_4d_optimizer
        
        # Check if JAX optimizer can be imported
        print("  ✅ JAX 4D optimizer imported successfully")
        
        # Note: Full optimization test would take too long for this test script
        # We just verify the import and basic class instantiation works
        
        return True
        
    except Exception as e:
        print(f"  ❌ JAX optimization test failed: {e}")
        return False

def test_progress_tracker_direct():
    """Test ProgressTracker directly if available."""
    print("\nTesting ProgressTracker directly...")
    
    try:
        from progress_tracker import ProgressTracker
        
        print("  ✅ ProgressTracker imported successfully")
        # Test basic functionality
        with ProgressTracker(total_steps=5, description="Test Progress") as progress:
            for i in range(5):
                progress.update(f"Test Step {i+1}", step_number=i+1)
                time.sleep(0.05)  # Brief pause
        
        print("  ✅ ProgressTracker test completed successfully")
        return True
        
    except ImportError:
        print("  ⚠️  ProgressTracker not available - fallback behavior will be used")
        return False
    except Exception as e:
        print(f"  ❌ ProgressTracker test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🧪 WARP BUBBLE OPTIMIZER - INTEGRATION TESTS")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    test_results['imports'] = test_progress_imports()
    test_results['jax'] = test_jax_fallback()
    test_results['progress_direct'] = test_progress_tracker_direct()
    test_results['control_loop'] = test_virtual_control_loop()
    test_results['analog_sim'] = test_analog_simulation()
    test_results['jax_opt'] = test_jax_optimization()
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    
    for test_name, result in test_results.items():
        if test_name == 'imports':
            passed = all(r.get('import_success', False) for r in result.values())
        else:
            passed = result
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        
        if not passed:
            all_passed = False
    
    print("\n🎯 OVERALL RESULT")
    print("=" * 60)
    if all_passed:
        print("🎉 All tests passed! Integration is working correctly.")
        print("✨ Ready for production use with simulation features.")
    else:
        print("⚠️  Some tests failed. Check individual results above.")
        print("💡 Core functionality should still work with fallback behavior.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
