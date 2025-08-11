"""
Moved: gaussian_optimize now lives under src/optimization/gaussian_optimize.py

This shim re-exports the implementation to preserve backward-compatible imports:
    from gaussian_optimize import ...
"""
from src.optimization.gaussian_optimize import *  # noqa: F401,F403
