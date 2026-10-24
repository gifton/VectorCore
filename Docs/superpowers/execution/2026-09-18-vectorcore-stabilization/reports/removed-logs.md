# Removed build and probe logs

On 2026-10-23 the 782 `*.log` files that this branch had force-added past the
repository's `*.log` ignore rule (`.gitignore:14`) were removed from the tree.
They were build, test and probe output: 105,132 lines, 37 percent of the PR's
added lines, read by no script, test or workflow.

| Path prefix | Files | Lines |
|---|---|---|
| `Docs/superpowers/execution/2026-09-18-vectorcore-stabilization/reports/` | 736 | 100,466 |
| `Docs/audits/2026-09-14/evidence/` | 46 | 4,666 |

The bytes are preserved unchanged, at the same repository paths, on the orphan
branch `archive/pr56-build-logs` (commit `b2709f5b8c6a171a843813a968e5ca3cfa2e94d3`).
`removed-logs.sha256` beside this file lists each file's SHA-256 as it stood in
the tree at removal. To read one:

```
git fetch origin archive/pr56-build-logs
git show b2709f5b8c6a171a843813a968e5ca3cfa2e94d3:<original path>
```

To verify the whole set, check out that commit and run
`shasum -a 256 -c removed-logs.sha256`.

Reports, closures and briefs in this directory that cite a `reports/*.log`
path refer to the archived file. The 46 audit-evidence hashes are also kept
under `archived_artifacts` in `Docs/audits/2026-09-14/evidence/manifest.json`.
