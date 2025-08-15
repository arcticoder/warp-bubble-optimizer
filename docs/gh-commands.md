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
