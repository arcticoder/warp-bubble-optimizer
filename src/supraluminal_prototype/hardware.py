from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CoilDriver:
    max_current: float
    hysteresis: float = 0.0  # simple fractional hysteresis model

    def command(self, normalized: float) -> float:
        """
        Map normalized [0,1] command to current [A] with simple hysteresis.
        For V&V we keep it nearly linear.
        """
        x = max(0.0, min(1.0, float(normalized)))
        return (1.0 - self.hysteresis) * x * self.max_current
