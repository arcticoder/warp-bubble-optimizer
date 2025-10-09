"""
Top-level import shim for UltimateBSplineOptimizer to satisfy tests that do:
    from ultimate_bspline_optimizer import UltimateBSplineOptimizer

This re-exports the implementation from src/optimization/ultimate_bspline_optimizer.py
without changing package layout.
"""
from src.optimization.ultimate_bspline_optimizer import UltimateBSplineOptimizer  # noqa: F401
