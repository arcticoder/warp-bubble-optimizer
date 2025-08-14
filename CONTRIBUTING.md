# Contributing

Thanks for your interest in contributing!

## Quick start

1. Fork this repository
2. Create a feature branch from `main`
3. Set up a virtual environment and install dependencies
4. Make your change with tests
5. Run tests locally (`pytest -q`)
6. Open a Pull Request

## Testing

- Keep tests fast and deterministic
- Use `pytest -m quick` for smoke runs in CI and locally
- Add unit tests for new public behavior and schemas

## Code style

- Prefer small, focused modules in `src/`
- Add minimal docs/comments where rationale isn’t obvious
- Keep CI green; fix or skip flaky tests with justification

## Areas to help

- Hardware mock facades and experiment planning
- UQ sampling and artifact dashboards
- Schema consumers across repos for mission/perf analytics

## Communication

- Open an issue for design changes
- Link to related research/resources when applicable

Thanks again — together we’ll advance 52c-class FTL simulation integrity.
