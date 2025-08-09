from typing import Literal


def compute_smearing_energy(P_peak: float, t_ramp: float, t_cruise: float,
                            shape: Literal['linear', 'triangle'] = 'linear') -> float:
    """
    Compute total energy for acceleration+deceleration with temporal smearing plus cruise.

    Contract:
    - Inputs: P_peak [W], t_ramp [s] per phase, t_cruise [s], shape in {'linear','triangle'}
    - Output: Energy [J] for accel + cruise + decel assuming identical ramps
    - Error modes: ValueError on negative times or unsupported shape

    Model assumptions:
    - Accel and decel ramps are mirror images (same t_ramp and shape)
    - Linear: power rises 0→P_peak over t_ramp, average P = P_peak/2 per ramp
    - Triangle: accel+decel as single triangle (half-area identical to two linear ramps)
    - Cruise consumes P_peak for t_cruise (max-power segment)
    """
    if t_ramp < 0 or t_cruise < 0 or P_peak < 0:
        raise ValueError("Inputs must be non-negative")
    shape_norm = str(shape).lower()
    if shape_norm not in ("linear", "triangle"):
        raise ValueError("Unsupported shape")

    # Energy for one linear ramp: area of triangle = 0.5 * base * height
    E_ramp = 0.5 * t_ramp * P_peak
    # Two ramps (accel + decel)
    E_ramps = 2.0 * E_ramp
    # Cruise at peak power
    E_cruise = P_peak * t_cruise
    return E_ramps + E_cruise
