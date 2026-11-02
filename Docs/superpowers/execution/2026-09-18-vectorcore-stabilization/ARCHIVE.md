# Archived program records

This directory is the retained summary of the 2026-09-18 stabilization program.
The program's full working record (task briefs, per-task and milestone reports,
worker build logs) was removed from the repository tree on 2026-10-23 in two
steps. The review of PR #56 (finding S1) had measured the directory at 51.6
percent of the PR's added lines.

## Where the files are

Both archives are orphan branches of this repository. Fetch one with
`git fetch origin <branch>` and read a file with `git show <commit>:<path>`.

| Branch | Commit | Contents |
|---|---|---|
| `archive/pr56-execution-ledger` | `220641304cc570753d8facc7552c9d49fa81e66c` | this whole directory as it stood at PR head `65aad5f`: 1,177 files, logs included |
| `archive/pr56-build-logs` | `b2709f5b8c6a171a843813a968e5ca3cfa2e94d3` | the 782 `*.log` files only (the first removal step) |

`ARCHIVE.sha256` beside this file lists the SHA-256 of every file in the
original directory. In a checkout of the ledger archive commit,
`shasum -a 256 -c ARCHIVE.sha256` verifies all 1,177.

## What was removed and what stays

| Path | Files | Lines | Status |
|---|---|---|---|
| `reports/` worker and milestone reports, verification JSON, diffs | 299 | 34,710 | archived |
| `reports/` build and probe logs | 736 | 100,466 | archived |
| `briefs/` task dispatch briefs | 70 | 3,233 | archived |
| `closures/` per-task acceptance records | 62 | 2,807 | retained, byte-identical |
| 12 top-level summaries (`progress.md`, `state.json`, `DESIGN_DECISIONS.md`, `DOWNSTREAM_UPGRADES.md`, handoffs, policy) | 12 | 4,918 | retained, machine paths rewritten |

Relative references to `briefs/…` and `reports/…` in the retained files, and
the contract `source` notes in `Scripts/ci/claim_ledger.json` that name
briefs, refer to archived files.

## Machine paths

The retained top-level files recorded absolute paths from the machine the
program ran on. They were rewritten as below; the archive holds the original
bytes.

| Original | Rewritten |
|---|---|
| `<repo>` | `<repo>` |
| `<integration-worktree>` | `<integration-worktree>` |
| `<pr-worktree>` | `<pr-worktree>` |
| any other `<home>` | `<home>` |
| `<tmp>` | `<tmp>` |
| `<agents>/<agent>` | `<agents>/<agent>` |

`closures/*.json` were not rewritten: the claim ledger cites each closure by
the SHA-256 of its bytes (67 references), so 29 closure files keep 31 lines of
original paths.
