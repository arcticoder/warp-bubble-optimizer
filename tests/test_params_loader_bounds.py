from __future__ import annotations
import io, textwrap
import pytest
from supraluminal_prototype.params_loader import load_ring_params


def test_params_loader_bounds_valid(tmp_path):
    yml = textwrap.dedent('''
    outer_diameter_m: 1.0
    inner_diameter_m: 0.5
    cross_section: circular
    num_rings: 4
    spacing_m: 0.1
    coil_turns: 100
    max_current_A: 10.0
    ''')
    p = tmp_path / 'rings.yaml'
    p.write_text(yml)
    rp = load_ring_params(str(p))
    assert rp.outer_diameter_m > rp.inner_diameter_m


@pytest.mark.parametrize('field, value', [
    ('inner_diameter_m', 1.5),
    ('num_rings', 0),
    ('coil_turns', 0),
    ('max_current_A', 0.0),
    ('spacing_m', -0.1),
])
def test_params_loader_bounds_invalid(tmp_path, field, value):
    yml = {
        'outer_diameter_m': 1.0,
        'inner_diameter_m': 0.5,
        'cross_section': 'circular',
        'num_rings': 4,
        'spacing_m': 0.1,
        'coil_turns': 100,
        'max_current_A': 10.0,
    }
    yml[field] = value
    import yaml
    p = tmp_path / 'rings.yaml'
    p.write_text(yaml.safe_dump(yml))
    with pytest.raises(ValueError):
        _ = load_ring_params(str(p))
