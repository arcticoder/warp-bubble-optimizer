# CI Best Practices
- Trigger `mission-validate.yml` before `traceability-badge.yml`.
- Check logs with `gh run view <run-id> --log`.
- Ensure `public/.nojekyll` exists for Pages.
- Verify artifacts via `gh run download <run-id> --name 40eridani-artifacts`.
- Verify PNGs: `ls artifacts/40eridani_*.png`.
- Debug plotting: `PYTHONPATH=src python -m src.uq_validation.plot_uq_records`.

## Zenodo Debugging

- Debug Zenodo: `curl -s -H "Authorization: Bearer $ZENODO_TOKEN" https://zenodo.org/api/deposit/depositions/<deposit-id>`
- Verify Zenodo DOI: `curl -s https://zenodo.org/api/records/<your-doi>`
