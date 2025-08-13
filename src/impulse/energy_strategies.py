"""Pluggable translation energy estimation strategies.

The default analytical model was previously embedded in
IntegratedImpulseController._estimate_translation_energy.
It is now factored into strategy classes so tests can inject
alternative empirical models and validate monotonicity / bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .integrated_impulse_control import VectorImpulseProfile  # type: ignore


class TranslationEnergyStrategy(Protocol):
    def estimate(self, profile: "VectorImpulseProfile") -> float:  # pragma: no cover - structural
        """Return estimated translation energy for the profile (J)."""
        ...


@dataclass
class QuadraticVelocityDistanceStrategy:
    """Analytical heuristic: E ∝ v_max^2 * displacement_magnitude.

    Constant chosen to align with prior baseline behaviour so legacy
    tests remain stable. k_factor can be tuned (or randomized in UQ).
    """
    k_factor: float = 1e11

    def estimate(self, profile: "VectorImpulseProfile") -> float:
        return self.k_factor * profile.v_max ** 2 * profile.target_displacement.magnitude


@dataclass
class EmpiricalScalingStrategy:
    """Empirical extension applying ramp/hold temporal weighting.

    Adds mild dependence on duty cycle so that for same displacement
    higher v_max with shorter hold still trends correctly.
    """
    base: TranslationEnergyStrategy
    ramp_weight: float = 0.6
    hold_weight: float = 0.4

    def estimate(self, profile: "VectorImpulseProfile") -> float:
        t_total = profile.t_up + profile.t_hold + profile.t_down
        if t_total <= 0:
            return 0.0
        ramp_frac = (profile.t_up + profile.t_down) / t_total
        hold_frac = profile.t_hold / t_total
        weight = self.ramp_weight * ramp_frac + self.hold_weight * hold_frac
        return self.base.estimate(profile) * (0.9 + 0.2 * weight)


DEFAULT_STRATEGY = QuadraticVelocityDistanceStrategy()

__all__ = [
    "TranslationEnergyStrategy",
    "QuadraticVelocityDistanceStrategy",
    "EmpiricalScalingStrategy",
    "DEFAULT_STRATEGY",
]
