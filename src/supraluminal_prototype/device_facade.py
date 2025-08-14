from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceFacade:
    """Mock facade for external EM coil, laser, and plasma devices.

    Provides minimal behavior to simulate hardware interaction at API level.
    """

    initialized: bool = False

    def initialize_coil(self, power: float) -> bool:
        self.initialized = power > 0
        return self.initialized

    def set_laser_frequency(self, freq: float) -> float:
        # Simple mock: slight calibration offset
        return freq * 1.05

    def set_plasma_density(self, n0: float, R_shell: float, width: float) -> dict:
        # Simplified: echo key params
        return {"n": n0, "R": R_shell}

    def read_field_state(self) -> dict:
        # Mocked field state consistent with tests' expectations
        return {"amplitude": 0.6, "error": 0.3}
