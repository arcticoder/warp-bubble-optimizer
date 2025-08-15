# HIL Loopback Readme and Checklists

This document guides a simple hardware-in-the-loop (HIL) loopback simulation using the supraluminal prototype modules.

## Objectives

- Exercise drivers → scheduler → estimation → fault detection path
- Verify timing, basic bounds, and error handling

## Pre-flight Checklist

- [ ] Python 3.12 environment with `requirements.txt` and `requirements-test.txt` installed
- [ ] Deterministic seed set if comparing runs (`WARP_SEED=123`)
- [ ] Power/thermal safety limits configured in params

## Loopback Procedure

1. Load ring params YAML via `params_loader.load_ring_params`
2. Generate a short current profile using `control_phase.generate_current_profile`
3. Use `drivers` facades to feed the scheduler
4. Run `estimation.EKF` predict/update with synthetic sensors
5. Check `fault_detection` thresholds

## Acceptance Criteria

- No faults for nominal profile
- EKF innovation decreases after update step
- Thermal model stays within configured bounds

## Post-run Checklist

- [ ] Save logs and perf.csv (if generated)
- [ ] Archive artifacts
- [ ] File issues for anomalies
