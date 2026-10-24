# Archived execution ledger of the 2026-09-18 stabilization program

This orphan branch holds `Docs/superpowers/execution/2026-09-18-vectorcore-stabilization/`
exactly as it stood at PR #56 head `65aad5f`: 1,177 files, including the 736
build and probe logs under `reports/`. It is the complete working record of
the program (task briefs, per-task and milestone reports, verification JSON,
diffs, logs, acceptance closures and the top-level summaries).

On 2026-10-23 the library tree kept only the 12 top-level summaries and
`closures/` (with machine-specific paths rewritten in the summaries) and
removed the rest; see `ARCHIVE.md` in that directory on the PR branch.
A first step had already moved the 782 `*.log` files to the sibling branch
`archive/pr56-build-logs` (commit `b2709f5b8c6a171a843813a968e5ca3cfa2e94d3`);
those bytes are also here.

`manifest.sha256` lists the SHA-256 of every file. Verify with:

```
shasum -a 256 -c manifest.sha256
```
