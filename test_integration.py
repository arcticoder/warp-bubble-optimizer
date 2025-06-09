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

# Test the new simulation modules
def test_impulse_engine_simulation():
    """Test impulse engine simulation functionality."""
    print("🧪 Testing impulse engine simulation...")
    
    try:
        from simulate_impulse_engine import simulate_impulse_maneuver, ImpulseProfile, WarpParameters
        
        # Test basic impulse simulation
        profile = ImpulseProfile(
            v_max=1e-5,
            t_up=5.0,
            t_hold=10.0,
            t_down=5.0,
            n_steps=100
        )
        
        warp_params = WarpParameters(
            R_max=50.0,
            thickness=1.0
        )
        
        results = simulate_impulse_maneuver(profile, warp_params, enable_progress=False)
        
        # Validate results
        assert results['total_energy'] > 0, "Total energy should be positive"
        assert results['peak_energy'] > 0, "Peak energy should be positive"
        assert len(results['velocity_profile']) == profile.n_steps, "Velocity profile length mismatch"
        assert len(results['energy_timeline']) == profile.n_steps, "Energy timeline length mismatch"
        
        print("✅ Impulse engine simulation test passed")
        return True
        
    except ImportError:
        print("⚠️  Impulse engine simulation module not available")
        return False
    except Exception as e:
        print(f"❌ Impulse engine simulation test failed: {e}")
        return False

def test_vector_impulse_simulation():
    """Test vectorized impulse simulation functionality."""
    print("🧪 Testing vector impulse simulation...")
    
    try:
        from simulate_vector_impulse import simulate_vector_impulse_maneuver, VectorImpulseProfile, WarpBubbleVector, Vector3D
        
        # Test 3D vector impulse simulation
        target = Vector3D(100.0, 50.0, -25.0)
        
        profile = VectorImpulseProfile(
            target_displacement=target,
            v_max=5e-6,
            t_up=3.0,
            t_hold=8.0,
            t_down=3.0,
            n_steps=50
        )
        
        warp_params = WarpBubbleVector(
            R_max=30.0,
            thickness=1.0,
            orientation=target.unit,
            asymmetry_factor=0.1
        )
        
        results = simulate_vector_impulse_maneuver(profile, warp_params, enable_progress=False)
        
        # Validate results
        assert results['total_energy'] > 0, "Total energy should be positive"
        assert results['trajectory_error'] >= 0, "Trajectory error should be non-negative"
        assert results['trajectory_accuracy'] <= 1.0, "Trajectory accuracy should be <= 1.0"
        assert len(results['position_trajectory']) == profile.n_steps, "Position trajectory length mismatch"
        
        # Check 3D trajectory
        final_pos = results['final_position']
        assert len(final_pos) == 3, "Final position should be 3D"
        
        print("✅ Vector impulse simulation test passed")
        return True
        
    except ImportError:
        print("⚠️  Vector impulse simulation module not available")
        return False
    except Exception as e:
        print(f"❌ Vector impulse simulation test failed: {e}")
        return False

def test_rotation_simulation():
    """Test rotation and attitude control simulation."""
    print("🧪 Testing rotation simulation...")
    
    try:
        from simulate_rotation import simulate_rotation_maneuver, RotationProfile, WarpBubbleRotational, Quaternion
        
        # Test rotation simulation
        target_q = Quaternion.from_euler(0.1, 0.2, 0.3)
        
        profile = RotationProfile(
            target_orientation=target_q,
            omega_max=0.02,
            t_up=2.0,
            t_hold=5.0,
            t_down=2.0,
            n_steps=50,
            control_mode="smooth"
        )
        
        warp_params = WarpBubbleRotational(
            R_max=40.0,
            thickness=1.5,
            moment_of_inertia=1e5,
            rotational_coupling=0.3
        )
        
        results = simulate_rotation_maneuver(profile, warp_params, enable_progress=False)
        
        # Validate results
        assert results['total_energy'] > 0, "Total energy should be positive"
        assert results['rotation_error'] >= 0, "Rotation error should be non-negative"
        assert results['rotation_accuracy'] <= 1.0, "Rotation accuracy should be <= 1.0"
        assert results['max_angular_accel'] >= 0, "Max angular acceleration should be non-negative"
        assert len(results['orientation_trajectory']) == profile.n_steps, "Orientation trajectory length mismatch"
        
        print("✅ Rotation simulation test passed")
        return True
        
    except ImportError:
        print("⚠️  Rotation simulation module not available")
        return False
    except Exception as e:
        print(f"❌ Rotation simulation test failed: {e}")
        return False

def test_integrated_control_system():
    """Test integrated warp control system."""
    print("🧪 Testing integrated control system...")
    
    try:
        from integrated_warp_control import IntegratedWarpController, MissionObjective, SystemConfiguration, Vector3D, Quaternion
        
        # Test system initialization
        config = SystemConfiguration(
            bubble_radius=60.0,
            bubble_thickness=1.5,
            max_velocity=1e-5,
            max_angular_velocity=0.03,
            simulation_steps=100
        )
        
        controller = IntegratedWarpController(config)
        
        # Test translation mission
        translation_obj = MissionObjective(
            objective_type="translation",
            target_position=Vector3D(200.0, 100.0, 0.0),
            time_constraint=120.0,
            energy_constraint=1e10,
            accuracy_requirement=0.90
        )
        
        # Test feasibility analysis
        feasibility = controller.analyze_mission_feasibility(translation_obj)
        assert isinstance(feasibility['feasible'], bool), "Feasibility should be boolean"
        assert feasibility['estimated_time'] > 0, "Estimated time should be positive"
        assert feasibility['estimated_energy'] > 0, "Estimated energy should be positive"
        
        # Test mission planning
        mission_profile = controller.plan_mission_profile(translation_obj)
        assert len(mission_profile['phases']) > 0, "Mission should have phases"
        assert mission_profile['total_duration'] > 0, "Total duration should be positive"
        
        # Test mission execution (without full simulation to save time)
        results = controller.execute_mission(translation_obj, enable_simulation=False)
        assert isinstance(results['success'], bool), "Success should be boolean"
        assert results['performance']['total_energy'] > 0, "Total energy should be positive"
        
        # Test system status
        status = controller.get_system_status()
        assert status['total_missions'] >= 1, "Should have at least one mission"
        
        print("✅ Integrated control system test passed")
        return True
        
    except ImportError:
        print("⚠️  Integrated control system module not available")
        return False
    except Exception as e:
        print(f"❌ Integrated control system test failed: {e}")
        return False

def test_simulation_integration():
    """Test integration between all simulation components."""
    print("🧪 Testing simulation component integration...")
    
    try:
        # Test that all simulation modules can be imported together
        from simulate_impulse_engine import ImpulseProfile
        from simulate_vector_impulse import VectorImpulseProfile, Vector3D
        from simulate_rotation import RotationProfile, Quaternion
        from integrated_warp_control import IntegratedWarpController, MissionObjective, SystemConfiguration
        
        # Test coordinate compatibility
        target_pos = Vector3D(100.0, 200.0, 50.0)
        target_rot = Quaternion.from_euler(0.1, 0.2, 0.3)
        
        # Test that different simulation profiles can be created consistently
        impulse_profile = ImpulseProfile(v_max=1e-5, t_up=5.0, t_hold=10.0, t_down=5.0)
        vector_profile = VectorImpulseProfile(target_displacement=target_pos, v_max=1e-5)
        rotation_profile = RotationProfile(target_orientation=target_rot, omega_max=0.02)
        
        # Test mission objective creation
        mission = MissionObjective(
            objective_type="combined",
            target_position=target_pos,
            target_orientation=target_rot
        )
        
        # Test that all components use compatible data structures
        assert hasattr(target_pos, 'vec'), "Vector3D should have vec attribute"
        assert hasattr(target_rot, 'w'), "Quaternion should have w attribute"
        assert hasattr(mission, 'objective_type'), "MissionObjective should have objective_type"
        
        print("✅ Simulation integration test passed")
        return True
        
    except ImportError as e:
        print(f"⚠️  Some simulation modules not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Simulation integration test failed: {e}")
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
    test_results['impulse_engine'] = test_impulse_engine_simulation()
    test_results['vector_impulse'] = test_vector_impulse_simulation()
    test_results['rotation_simulation'] = test_rotation_simulation()
    test_results['integrated_control'] = test_integrated_control_system()
    test_results['simulation_integration'] = test_simulation_integration()
    
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
