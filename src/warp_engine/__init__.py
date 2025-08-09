"""
Warp Engine: Complete Implementation Framework
===========================================

Import-safe package initializer. Heavy submodules (that may require optional
dependencies like JAX/SymPy) are not imported eagerly to keep test collection
robust in minimal environments. Import the needed submodules directly, e.g.:

    from src.warp_engine.backreaction import EinsteinSolver, BackreactionAnalyzer
    from src.warp_engine.dynamic_sim import DynamicBubbleSimulator

"""

__version__ = "1.0.0"
__author__ = "Warp Engine Development Team"

# Keep __all__ minimal to avoid triggering imports; submodules should be
# imported explicitly by consumers. Metadata only.
__all__ = ['__version__', '__author__']
