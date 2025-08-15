from __future__ import annotations
from dataclasses import dataclass
from typing import List
import yaml


@dataclass
class RingParams:
    outer_diameter_m: float
    inner_diameter_m: float
    cross_section: str
    num_rings: int
    spacing_m: float
    coil_turns: int
    max_current_A: float


def load_ring_params(path: str) -> RingParams:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    required = [
        'outer_diameter_m','inner_diameter_m','cross_section','num_rings','spacing_m','coil_turns','max_current_A'
    ]
    for k in required:
        if k not in data:
            raise ValueError(f"Missing required ring param: {k}")
    return RingParams(
        outer_diameter_m=float(data['outer_diameter_m']),
        inner_diameter_m=float(data['inner_diameter_m']),
        cross_section=str(data['cross_section']),
        num_rings=int(data['num_rings']),
        spacing_m=float(data['spacing_m']),
        coil_turns=int(data['coil_turns']),
        max_current_A=float(data['max_current_A']),
    )
