# HTS Coil Optimization Algorithms

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**Computational optimization algorithms for superconducting magnet design and plasma physics applications.**

This repository provides multi-objective optimization algorithms used by the [hts-coils](https://github.com/DawsonInstitute/hts-coils) framework for:
- High-beta plasma confinement parameter optimization
- REBCO superconducting magnet design
- Multi-objective electromagnetic field optimization
- Computational validation and uncertainty quantification

The algorithms are integrated with HTS coil simulations through the `src/warp/` directory in the hts-coils repository, supporting peer-reviewed preprints on plasma physics and superconducting magnet applications.

### Primary Integration: HTS Coil Optimization

**Key Applications in HTS Framework:**
- Multi-objective optimization for magnetic field uniformity and thermal stability
- JAX-accelerated electromagnetic field calculations
- Monte Carlo uncertainty quantification for manufacturing tolerances
- Validation framework for REBCO paper reproduction

**Related Publications:**
- "Computational Analysis of High-Temperature Superconducting Coils for High-Beta Plasma Confinement" (https://github.com/DawsonInstitute/hts-coils/blob/main/papers/warp/hts_plasma_confinement.tex)
- "Validation Framework for Lentz Soliton Formation in High-Beta Plasma" (https://github.com/DawsonInstitute/hts-coils/blob/main/papers/warp/soliton_validation.tex)

---

## Experimental Research Components

> **Note**: This repository also contains experimental warp field research code not currently integrated with Dawson Institute preprints. See [EXPERIMENTAL_COMPONENTS.md](EXPERIMENTAL_COMPONENTS.md) for complete documentation of advanced optimization algorithms, theoretical frameworks, and simulation capabilities.

## Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Run tests
pytest
```

## Contributing

We welcome collaboration on hardware scaling and theoretical integration:

- Hardware: laser–coil synchronization, plasma chamber design, envelope tracking
- Theory: LQG corrections with sinc(πμ), Natário zero-expansion, positive-energy soliton profiles

Start here:

- Read CONTRIBUTING.md
- Fork the repo, create a feature branch, add tests (pytest), and open a PR

Focus areas that need help now:

- Device facades and experiment plans for 2035–2050 prototypes
- UQ extensions and CI artifact dashboards
- Cross-repo schema consumers for mission/perf analytics

## Testing and CI

- Local quick run: use the lightweight tests and filters already configured in `pytest.ini` and `conftest.py`.
  - Site-packages and heavy script-style tests are excluded from collection.
  - Focused runs: `pytest -k vector_impulse -q` or `pytest tests/test_vnv_vector_impulse.py -q`.
  - Markers: use `-m "not slow"` to skip slow tests.
- Dependencies: install `requirements-test.txt` for a consistent environment.
- CI: GitHub Actions workflow runs the test suite on Ubuntu with Python 3.11 using the pinned test requirements.

## Mission timeline logging (--timeline-log)

The mission CLI supports an optional `--timeline-log` argument to record planning and execution milestones. The log can be written as CSV (default) or JSONL (if the path ends with `.jsonl`).

- CSV header: `iso_time,t_rel_s,event,segment_id,planned_value,actual_value`
- JSONL fields: `{ iso_time, t_rel_s, event, segment_id, planned_value, actual_value }`

Events currently emitted:
- `plan_created` (t_rel_s = 0, planned_value = total planned energy J)
- `rehearsal_complete` or `dry_run_abort_complete` (non-executing paths)
- `mission_complete` (actual_value = total energy used J)

Example usage:

```bash
PYTHONPATH=src python -m impulse.mission_cli \
    --waypoints examples/waypoints/simple.json \
    --export mission.json \
    --timeline-log mission_timeline.csv \
    --rehearsal --hybrid simulate-first --seed 123
```


## Scope, Validation & Limitations

- Scope: The materials and numeric outputs in this repository are research-stage examples and depend on implementation choices, parameter settings, and numerical tolerances.
- Validation: Reproducibility artifacts (scripts, raw outputs, seeds, and environment details) are provided in `docs/` or `examples/` where available; reproduce analyses with parameter sweeps and independent environments to assess robustness.
- Limitations: Results are sensitive to modeling choices and discretization. Independent verification, sensitivity analyses, and peer review are recommended before using these results for engineering or policy decisions.
