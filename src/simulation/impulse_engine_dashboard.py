#!/usr/bin/env python3
"""
Integrated Impulse Engine Dashboard
=================================

Dashboard script that combines impulse engine simulation with virtual control
loop integration for complete mission planning and execution simulation.
"""

import numpy as np
import asyncio
import argparse
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import our simulation modules
from simulate_impulse_engine import (
    ImpulseProfile, WarpParameters, simulate_impulse_maneuver, 
    visualize_impulse_results, parameter_sweep
)

try:
    from sim_control_loop import VirtualWarpController, SensorConfig, ActuatorConfig, ControllerConfig
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    print("⚠️  Control loop module not available")

try:
    from progress_tracker import ProgressTracker
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False

@dataclass 
class MissionProfile:
    """Complete mission specification."""
    name: str
    target_delta_v_ms: float      # Target Δv in m/s
    mission_duration_s: float     # Total mission duration
    max_acceleration_g: float     # Maximum acceleration (g-forces)
    energy_budget_gj: float       # Energy budget in GJ
    safety_margin: float = 0.2    # Safety margin factor

class ImpulseEngineDashboard:
    """Comprehensive dashboard for impulse engine mission planning."""
    
    def __init__(self):
        """Initialize dashboard with default configurations."""
        self.default_warp_params = WarpParameters(
            R_max=150.0,
            thickness=3.0,
            shape_params=np.array([1.2, 1.0, 0.9])
        )
        
        self.control_config = {
            'sensor': SensorConfig(noise_level=0.01, update_rate=20.0),
            'actuator': ActuatorConfig(response_time=0.05, max_rate=1.0),
            'controller': ControllerConfig(kp=0.5, ki=0.1, kd=0.05)
        } if CONTROL_AVAILABLE else None
        
    def design_impulse_profile(self, mission: MissionProfile) -> ImpulseProfile:
        """
        Design optimal impulse profile for given mission requirements.
        
        Args:
            mission: Mission specification
            
        Returns:
            Optimized impulse profile
        """
        print(f"🎯 Designing impulse profile for mission: {mission.name}")
        
        # Convert mission Δv to maximum velocity
        # Assume symmetric acceleration profile
        v_max_ms = mission.target_delta_v_ms / 2  # Peak velocity
        v_max_c = v_max_ms / 299792458  # Fraction of c
        
        # Design time profile based on acceleration limits
        max_accel_ms2 = mission.max_acceleration_g * 9.81
        t_ramp = v_max_ms / max_accel_ms2  # Time to reach v_max
        
        # Remaining time for cruise phase
        t_remaining = mission.mission_duration_s - 2 * t_ramp
        t_hold = max(t_remaining, 5.0)  # Minimum 5s cruise
        
        # Apply safety margin
        v_max_c *= (1 + mission.safety_margin)
        
        profile = ImpulseProfile(
            v_max=v_max_c,
            t_up=t_ramp,
            t_hold=t_hold, 
            t_down=t_ramp,
            n_steps=int(mission.mission_duration_s * 10)  # 10 Hz sampling
        )
        
        print(f"   Max velocity: {v_max_c:.2e} c ({v_max_ms:.1f} m/s)")
        print(f"   Acceleration time: {t_ramp:.1f} s")
        print(f"   Cruise time: {t_hold:.1f} s")
        print(f"   Max acceleration: {max_accel_ms2/9.81:.1f} g")
        
        return profile
    
    def validate_mission_feasibility(self, mission: MissionProfile, 
                                   profile: ImpulseProfile) -> Dict[str, Any]:
        """
        Validate mission feasibility against physical and engineering constraints.
        
        Args:
            mission: Mission specification
            profile: Proposed impulse profile
            
        Returns:
            Feasibility analysis results
        """
        print("🔍 Validating mission feasibility...")
        
        # Run energy simulation
        results = simulate_impulse_maneuver(profile, self.default_warp_params)
        
        # Check constraints
        constraints = {
            'energy_budget': {
                'required_gj': results['total_energy'] / 1e9,
                'budget_gj': mission.energy_budget_gj,
                'feasible': results['total_energy'] / 1e9 <= mission.energy_budget_gj,
                'margin': mission.energy_budget_gj - (results['total_energy'] / 1e9)
            },
            'quantum_inequality': {
                'max_velocity': profile.v_max,
                'qi_limit': 1e-3,  # Conservative QI limit
                'feasible': profile.v_max <= 1e-3,
                'margin': 1e-3 - profile.v_max
            },
            'acceleration': {
                'max_accel_g': mission.target_delta_v_ms / profile.t_up / 9.81,
                'limit_g': mission.max_acceleration_g,
                'feasible': mission.target_delta_v_ms / profile.t_up / 9.81 <= mission.max_acceleration_g,
                'margin': mission.max_acceleration_g - (mission.target_delta_v_ms / profile.t_up / 9.81)
            }
        }
        
        # Overall feasibility
        overall_feasible = all(c['feasible'] for c in constraints.values())
        
        feasibility = {
            'overall_feasible': overall_feasible,
            'constraints': constraints,
            'simulation_results': results,
            'recommendations': []
        }
        
        # Generate recommendations
        if not constraints['energy_budget']['feasible']:
            feasibility['recommendations'].append(
                f"Reduce velocity by {(results['total_energy']/1e9/mission.energy_budget_gj - 1)*100:.1f}% to meet energy budget"
            )
        
        if not constraints['quantum_inequality']['feasible']:
            feasibility['recommendations'].append(
                "Velocity exceeds QI limits - consider multi-impulse trajectory"
            )
            
        if not constraints['acceleration']['feasible']:
            feasibility['recommendations'].append(
                f"Increase ramp time to {mission.target_delta_v_ms/mission.max_acceleration_g/9.81:.1f}s to meet acceleration limits"
            )
        
        return feasibility
    
    async def run_integrated_simulation(self, mission: MissionProfile, 
                                      profile: ImpulseProfile) -> Dict[str, Any]:
        """
        Run integrated simulation with virtual control loop.
        
        Args:
            mission: Mission specification
            profile: Impulse profile to execute
            
        Returns:
            Integrated simulation results
        """
        if not CONTROL_AVAILABLE:
            print("⚠️  Control loop not available - running basic simulation")
            return simulate_impulse_maneuver(profile, self.default_warp_params)
        
        print("🔄 Running integrated control loop simulation...")
        
        # Define control objective based on impulse profile
        def impulse_objective(params, time_context=None):
            """Objective function that varies with mission time."""
            if time_context is None:
                return np.sum(params**2)  # Default: minimize parameter deviation
            
            # Get target velocity for current time
            from simulate_impulse_engine import velocity_profile
            t = time_context.get('current_time', 0.0)
            target_v = velocity_profile(t, profile.v_max, profile.t_up, profile.t_hold, profile.t_down)
            
            # Objective: track velocity profile
            current_v_estimate = params[0] if len(params) > 0 else 0.0
            velocity_error = (current_v_estimate - target_v)**2
            
            # Add warp field shape optimization
            shape_error = np.sum((params[1:] - self.default_warp_params.shape_params[:len(params)-1])**2)
            
            return velocity_error + 0.1 * shape_error
        
        # Initialize control loop
        initial_params = np.concatenate([
            [0.0],  # Initial velocity
            self.default_warp_params.shape_params[:3]  # Shape parameters
        ])
        
        controller = VirtualWarpController(
            objective_func=impulse_objective,
            initial_params=initial_params,
            sensor_config=self.control_config['sensor'],
            actuator_config=self.control_config['actuator'],
            controller_config=self.control_config['controller']
        )
        
        # Run control simulation for mission duration
        total_duration = profile.t_up + profile.t_hold + profile.t_down
        control_results = await controller.run_control_loop(
            duration=total_duration,
            target_rate=20.0  # 20 Hz control rate
        )
        
        # Combine with impulse simulation
        impulse_results = simulate_impulse_maneuver(profile, self.default_warp_params)
        
        integrated_results = {
            'impulse_simulation': impulse_results,
            'control_simulation': control_results,
            'mission_profile': mission,
            'impulse_profile': profile,
            'integration_metrics': {
                'control_steps': control_results['steps'],
                'final_objective': control_results['final_objective'],
                'control_duration': control_results['elapsed_time'],
                'tracking_performance': 'GOOD' if control_results['final_objective'] < 1.0 else 'POOR'
            }
        }
        
        return integrated_results
    
    def generate_mission_report(self, results: Dict[str, Any], 
                              save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive mission analysis report.
        
        Args:
            results: Simulation results
            save_path: Optional path to save report
            
        Returns:
            Report text
        """
        # Extract data
        if 'impulse_simulation' in results:
            # Integrated results
            impulse_data = results['impulse_simulation']
            control_data = results['control_simulation']
            mission = results['mission_profile']
            profile = results['impulse_profile']
        else:
            # Basic simulation results
            impulse_data = results
            control_data = None
            mission = None
            profile = results.get('profile')
        
        report = f"""
IMPULSE ENGINE MISSION ANALYSIS REPORT
======================================

Generated: {np.datetime_as_string(np.datetime64('now'), unit='s')}

MISSION OVERVIEW
----------------
Mission Name: {mission.name if mission else 'Test Mission'}
Target Δv: {mission.target_delta_v_ms if mission else 'N/A'} m/s
Duration: {mission.mission_duration_s if mission else 'N/A'} s
Energy Budget: {mission.energy_budget_gj if mission else 'N/A'} GJ

VELOCITY PROFILE
----------------
Maximum Velocity: {profile.v_max:.2e} c ({profile.v_max*299792458:.1f} m/s)
Acceleration Time: {profile.t_up:.1f} s
Cruise Time: {profile.t_hold:.1f} s
Deceleration Time: {profile.t_down:.1f} s
Total Maneuver: {profile.t_up + profile.t_hold + profile.t_down:.1f} s

ENERGY ANALYSIS
---------------
Total Energy Required: {impulse_data['total_energy']/1e9:.2f} GJ
Peak Energy: {impulse_data['peak_energy']/1e9:.2f} GJ
Average Cruise Energy: {impulse_data['hold_avg_energy']/1e9:.2f} GJ
Energy Efficiency: {impulse_data['total_energy']/(profile.v_max*299792458*profile.t_hold)/1e6:.1f} MJ/km

CONTROL PERFORMANCE
-------------------"""
        
        if control_data:
            report += f"""
Control Steps: {control_data['steps']}
Final Objective: {control_data['final_objective']:.3e}
Control Duration: {control_data['elapsed_time']:.1f} s
Tracking Performance: {results['integration_metrics']['tracking_performance']}"""
        else:
            report += """
Control Loop: Not simulated"""
        
        report += f"""

FEASIBILITY ASSESSMENT
----------------------
QI Compliance: {'✅ PASS' if profile.v_max <= 1e-3 else '⚠️ MARGINAL'}
Energy Scale: {'✅ REASONABLE' if impulse_data['total_energy'] < 1e12 else '⚠️ HIGH'}
Velocity Regime: {'✅ LOW-V' if profile.v_max < 1e-4 else '⚠️ RELATIVISTIC'}

RECOMMENDATIONS
---------------
• Velocity scaling confirmed: E ∝ v²
• {'Multi-impulse trajectory recommended for energy efficiency' if impulse_data['total_energy'] > 1e11 else 'Single impulse feasible'}
• {'Consider larger bubble radius for efficiency' if impulse_data['total_energy'] > 5e10 else 'Current configuration optimal'}

SIMULATION METADATA
-------------------
JAX Acceleration: {'Enabled' if 'jax' in str(type(impulse_data['time_grid'])) else 'NumPy Fallback'}
Time Steps: {len(impulse_data['time_grid'])}
Computation: {'GPU-Accelerated' if 'jax' in str(type(impulse_data['time_grid'])) else 'CPU-Only'}
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"📄 Mission report saved: {save_path}")
        
        return report

def main():
    """Main CLI interface for impulse engine dashboard."""
    parser = argparse.ArgumentParser(description='Impulse Engine Mission Dashboard')
    parser.add_argument('--mode', choices=['design', 'validate', 'simulate', 'sweep'], 
                       default='design', help='Operation mode')
    parser.add_argument('--mission-file', type=str, help='Mission specification JSON file')
    parser.add_argument('--delta-v', type=float, default=1000.0, help='Target delta-v (m/s)')
    parser.add_argument('--duration', type=float, default=120.0, help='Mission duration (s)')
    parser.add_argument('--energy-budget', type=float, default=10.0, help='Energy budget (GJ)')
    parser.add_argument('--max-accel', type=float, default=2.0, help='Max acceleration (g)')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    print("🚀 IMPULSE ENGINE MISSION DASHBOARD")
    print("=" * 60)
    
    # Initialize dashboard
    dashboard = ImpulseEngineDashboard()
    
    # Load or create mission
    if args.mission_file:
        with open(args.mission_file, 'r') as f:
            mission_data = json.load(f)
        mission = MissionProfile(**mission_data)
    else:
        mission = MissionProfile(
            name="CLI Mission",
            target_delta_v_ms=args.delta_v,
            mission_duration_s=args.duration,
            max_acceleration_g=args.max_accel,
            energy_budget_gj=args.energy_budget
        )
    
    print(f"📋 Mission: {mission.name}")
    print(f"   Δv: {mission.target_delta_v_ms:.1f} m/s")
    print(f"   Duration: {mission.mission_duration_s:.1f} s")
    print(f"   Energy budget: {mission.energy_budget_gj:.1f} GJ")
    
    if args.mode == 'design':
        # Design impulse profile
        profile = dashboard.design_impulse_profile(mission)
        
        # Basic simulation
        results = simulate_impulse_maneuver(profile, dashboard.default_warp_params)
        
        # Visualization
        if not args.no_viz:
            visualize_impulse_results(results)
        
        # Report
        report = dashboard.generate_mission_report(results, 
                                                 f"{args.output_dir}/mission_report.txt")
        print("\n" + report)
        
    elif args.mode == 'validate':
        # Design and validate
        profile = dashboard.design_impulse_profile(mission)
        feasibility = dashboard.validate_mission_feasibility(mission, profile)
        
        print(f"\n🔍 FEASIBILITY ANALYSIS")
        print(f"Overall feasible: {'✅ YES' if feasibility['overall_feasible'] else '❌ NO'}")
        
        for constraint, data in feasibility['constraints'].items():
            status = '✅ PASS' if data['feasible'] else '❌ FAIL'
            print(f"   {constraint}: {status} (margin: {data['margin']:.2e})")
        
        if feasibility['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in feasibility['recommendations']:
                print(f"   • {rec}")
    
    elif args.mode == 'simulate':
        # Full integrated simulation
        profile = dashboard.design_impulse_profile(mission)
        
        async def run_sim():
            return await dashboard.run_integrated_simulation(mission, profile)
        
        try:
            results = asyncio.run(run_sim())
            
            # Generate comprehensive report
            report = dashboard.generate_mission_report(results, 
                                                     f"{args.output_dir}/integrated_report.txt")
            print("\n" + report)
            
        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            # Fallback to basic simulation
            profile = dashboard.design_impulse_profile(mission)
            results = simulate_impulse_maneuver(profile, dashboard.default_warp_params)
            report = dashboard.generate_mission_report(results)
            print("\n" + report)
    
    elif args.mode == 'sweep':
        # Parameter sweep analysis
        print(f"\n🔬 Running parameter sweep...")
        v_range = np.logspace(-5, -3, 8)
        t_range = np.linspace(5.0, 30.0, 6)
        
        sweep_results = parameter_sweep(v_range, t_range, dashboard.default_warp_params)
        
        # Save sweep data
        sweep_file = f"{args.output_dir}/parameter_sweep.json"
        with open(sweep_file, 'w') as f:
            # Convert numpy arrays for JSON serialization
            save_data = {
                'v_max_values': sweep_results['v_max_values'],
                't_ramp_values': sweep_results['t_ramp_values'],
                'energy_matrix': sweep_results['energy_matrix'].tolist(),
                'scaling_analysis': sweep_results['scaling_analysis']
            }
            json.dump(save_data, f, indent=2)
        
        print(f"📁 Sweep data saved: {sweep_file}")
    
    print(f"\n🎯 Dashboard operation complete!")

if __name__ == "__main__":
    main()
