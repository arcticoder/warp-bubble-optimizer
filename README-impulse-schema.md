# Impulse Mission JSON Schema

This repository now ships a JSON Schema for the mission export produced by the integrated impulse controller.

- Location: `schemas/impulse.mission.v1.json`
- ID: `impulse.mission.v1`
- Version: `1`

Highlights:
- Plan metadata: energy/time estimates, safety margin, hybrid planning knobs.
- Results: per-segment summaries with `kinds: [translation|rotation]` and compact translation/rotation summaries.
- Optional `planning_cache` persists planning-time simulation cache for reproducibility.
- Optional `meta` with a config snapshot when `verbose_export=True`.

Validation: if you have `jsonschema` installed, the test `test_json_schema_file_validation_if_available` will validate exports against this schema.
