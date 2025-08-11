#!/usr/bin/env python3
"""
Shared pytest fixtures and collection filters (tests scope only).
"""
import asyncio
import os
from pathlib import Path
import pytest

from integrated_impulse_control import (
    IntegratedImpulseController, MissionWaypoint, ImpulseEngineConfig
)
from src.simulation.simulate_vector_impulse import Vector3D
from simulate_rotation import Quaternion


# Skip legacy/heavy script-style tests and optional-deps suites during collection
def pytest_ignore_collect(collection_path: Path, config):
    try:
        name = os.path.basename(str(collection_path))
        # Legacy and optional-deps tests that should not run in CI
        skip_names = {
            "test_ultimate_bspline.py",
            "test_lqg_bounds_focused.py",
            "test_pipeline.py",
            "test_solver_debug.py",
            "test_ghost_eft.py",          # pulls in sympy/warp_qft
            "test_jax_cpu.py",            # requires jax on CI
            "test_soliton_3d_stability.py"  # heavy sympy/numpy stack
        }
        if name in skip_names:
            return True
        p = str(collection_path)
        if "/site-packages/" in p or p.endswith("internal_test_util/test_harnesses.py"):
            return True
    except Exception:
        return False
    return False


@pytest.fixture(scope="function")
def trajectory_plan():
    """Provide a minimal, feasible trajectory plan for mission execution tests."""
    config = ImpulseEngineConfig(
        max_velocity=1e-4,
        max_angular_velocity=0.1,
        energy_budget=1e12,
    )
    controller = IntegratedImpulseController(config)

    waypoints = [
        MissionWaypoint(
            position=Vector3D(0.0, 0.0, 0.0),
            orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
            dwell_time=1.0,
        ),
        MissionWaypoint(
            position=Vector3D(100.0, 0.0, 0.0),
            orientation=Quaternion.from_euler(0.0, 0.0, 0.0),
            dwell_time=1.0,
            approach_speed=5e-5,
        ),
    ]

    plan = controller.plan_impulse_trajectory(waypoints, optimize_energy=True)
    return plan


@pytest.fixture(scope="function")
def mission_results(trajectory_plan):
    """Execute a short mission to generate results for reporting tests."""
    config = ImpulseEngineConfig(
        max_velocity=1e-4,
        max_angular_velocity=0.1,
        energy_budget=1e12,
    )
    controller = IntegratedImpulseController(config)

    async def run():
        return await controller.execute_impulse_mission(trajectory_plan, enable_feedback=False)

    return asyncio.run(run())
