# Archived build and probe logs from PR #56

This orphan branch holds the 782 `*.log` files that were removed from
`stabilization/0.4.0-baseline` (removed from tree after commit `65aad5f`), at
their original repository paths:

| Path prefix | Files |
|---|---|
| `Docs/superpowers/execution/2026-09-18-vectorcore-stabilization/reports/` | 736 |
| `Docs/audits/2026-09-14/evidence/` | 46 |

They were force-added past the repository's `*.log` ignore rule and are
build, test and probe output only. The bytes here are unchanged from the PR
tree; `removed-logs.sha256` lists each file's SHA-256 as recorded in the PR
at the time of removal. Verify with:

```
shasum -a 256 -c removed-logs.sha256
```

The same list is kept in the library tree at
`Docs/superpowers/execution/2026-09-18-vectorcore-stabilization/reports/removed-logs.sha256`,
and the 46 audit-evidence hashes are also recorded under `archived_artifacts`
in `Docs/audits/2026-09-14/evidence/manifest.json`.
