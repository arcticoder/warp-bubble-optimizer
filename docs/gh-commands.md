# GitHub CLI handy commands

Use these to debug CI artifacts and Pages publishing.

- List recent artifacts
  gh run list --limit 10
  gh run view <run-id> --log
  gh run download <run-id> -n 40eridani-artifacts -D ./downloaded

- Inspect Pages deployments
  gh api repos/${{owner}}/${{repo}}/pages/deployments | jq

- Verify Pages content locally
  python -m http.server -d public 8080

Notes:
- Artifacts expire; ensure name matches exactly: 40eridani-artifacts
- Pages deploys may lag by a minute; hard refresh if you see 404s

# GitHub CLI Commands
- List runs: `gh run list --repo arcticoder/warp-bubble-optimizer --workflow mission-validate.yml --limit 10`
- View logs: `gh run view <run-id> --repo arcticoder/warp-bubble-optimizer --log`
- Download artifacts: `gh run download <run-id> --repo arcticoder/warp-bubble-optimizer --name 40eridani-artifacts`
- Trigger workflow: `gh workflow run mission-validate.yml --ref main --repo arcticoder/warp-bubble-optimizer`
- Check Pages: `curl -s -I https://arcticoder.github.io/warp-bubble-optimizer/40eridani_energy.png`

- Trigger Zenodo upload: `gh workflow run zenodo-upload.yml --ref main --repo arcticoder/warp-bubble-optimizer`
- Check Pages energy plot: `curl -s -I https://arcticoder.github.io/warp-bubble-optimizer/40eridani_energy.png`
