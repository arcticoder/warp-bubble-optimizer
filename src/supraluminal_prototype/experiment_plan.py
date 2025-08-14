from __future__ import annotations

from dataclasses import dataclass

from .device_facade import DeviceFacade


@dataclass
class ExperimentPlan:
    laser_power: float
    coil_freq: float
    plasma_density: float
    r_shell: float
    width: float

    def generate_plan(self) -> dict:
        return {
            'laser_power': self.laser_power,
            'coil_freq': self.coil_freq,
            'plasma_params': {
                'n0': self.plasma_density,
                'R': self.r_shell,
                'width': self.width,
            }
        }

    def validate_plan(self, facade: DeviceFacade) -> bool:
        ok = facade.initialize_coil(self.laser_power)
        _ = facade.set_laser_frequency(self.coil_freq)
        _ = facade.set_plasma_density(self.plasma_density, self.r_shell, self.width)
        state = facade.read_field_state()
        return bool(ok and state.get('error', 1.0) < 0.45)
