# Seed Reproducibility and Environment Variables

This project aims for deterministic behavior where feasible.

## Seeds

- CLI option: many scripts accept `--seed` to set deterministic random streams.
- Environment variables set by the CLI (mission):
  - `WARP_SEED`: exported when `--seed` is provided
  - `PYTHONHASHSEED`: set by test harness/CI as needed

## Recommendations

- Always record `--seed` in experiment logs
- Set `WARP_SEED` in CI workflows for traceability
- Avoid time-dependent randomness in core logic; use `np.random.default_rng(seed)`

## Example

```
PYTHONPATH=src python -m impulse.mission_cli \
  --waypoints examples/waypoints/simple.json \
  --seed 123 --rehearsal --export mission.json --timeline-log mission_timeline.csv
```
