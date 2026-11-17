# Evidence publication and recovery

Current documentation uses portable paths. Large qualification profiles and
evidence manifests are represented by small references to their original bytes.
This keeps workstation paths and generated inventories out of the current source
tree while retaining the exact inputs to earlier reviews.

This is a current-files cleanup. Existing commits, branches and tags have not
been rewritten. Historical copies can still contain machine-specific paths, and
the Git repository's historical storage has not been reduced. A private snapshot
of the original tree is also retained by the maintainer.

## Exact profiles and manifests

An archive reference contains `artifactReference=1`, an immutable `gitCommit`, a
repository-relative `gitPath`, the original `bytes` count and its `sha256`.
`Scripts/ci/archived_evidence.py` reads the Git blob as data and verifies both size
and SHA-256 before returning it. It rejects duplicate JSON keys and nested
references. No historical helpers are imported or executed.

The qualification checkers accept these references as well as original inline
profiles. All existing source, compiler, receipt, numerical and diagnostic checks
still apply to the original data. The hosted verifier requires the reference to
be tracked unchanged in the reviewed policy commit. Its report records the
original profile digest and, when applicable, the reference digest separately.

Local verification stays offline, including in partial clones: Git's implicit
object fetching is disabled. Archive references require Git with
`--no-lazy-fetch` support; inline retained JSON has no Git dependency. If a shallow
or partial checkout lacks the pinned data, materialize it with an explicit fetch:

```sh
python3 -B Scripts/ci/archived_evidence.py \
  Scripts/ci/qualification_profiles/macos-27.0.1-arm64-0.4.0-hosted-2026-11-16.json \
  --fetch --output "$EVIDENCE_ROOT/reviewed-native-profile.json"
```

The destination must not already exist. `--fetch` retrieves only the pinned
commit from `origin`; it does not check out or execute that revision. The hosted
verification workflow explicitly enables this fetch for shallow policy
checkouts. Missing objects, incorrect hashes or unavailable archives fail
verification; a summary or a green workflow cannot substitute for the data.

For a source export without Git history, provide the retained original profile
directly to `compiler_qualification.py --profile`. The historical
`bounded_native_qualification.py` also accepts `--profile`; its default remains
the adjacent reviewed selection. Raw receipts and source inputs are still
required at the explicitly supplied roots.

These Git references reuse evidence that was already published. Do not publish
new private captures merely to create an archive reference. Keep fresh raw
profiles outside the repository and explicitly review their storage and
publication before using them in hosted qualification. The existing 30-day CI
artifact retention is not a permanent archive. Preserve the pinned commits and
private backups; a future history purge requires a separate storage migration.

## Portable presentation records

[The publication catalog](evidence-publication.json) lists the original files
whose paths or bulky inventories were replaced, their Git locations, byte counts,
checksums and whether
the current copy is a reference or a portable presentation. This catalog records
the originals; it is not a fresh qualification result.

Paths in presentation JSON, Markdown and replay examples use these placeholders:

| Placeholder | Meaning |
|---|---|
| `${SOURCE_ROOT}` | The source checkout identified by the record's commit |
| `${EVIDENCE_ROOT}` | The maintainer's retained evidence collections |
| `${ARCHIVE_ROOT}` | Archive indexes and relocation records |
| `${WORKSPACE_ROOT}` | Related checkouts used during the investigation |
| `${CAPTURE_ROOT}` / `${SYSTEM_TEMP}` | Temporary capture or runtime locations |
| `${BETA_DEVELOPER_DIR}` | The separately selected beta Xcode developer directory |
| `${CI_HOME}` / `${USER_HOME}` | A runner or user's home, when needed as context |

Set appropriate local values before adapting an example command. These edited
presentation files are not byte-identical raw receipts and must not be passed to
a verifier in place of the retained originals. Raw records still have their
original paths and hashes; their existing explicit root remapping remains the
verification mechanism. Replay helpers under `Docs` use
`VECTORCORE_EVIDENCE_ROOT` for their output location.

To retrieve a presentation file's original, read its commit/path from the catalog
and use `git show COMMIT:PATH`, then compare its size and SHA-256 with that record.
No new compiler, platform, candidate or release admission follows from this
cleanup. Existing scoped results retain `compilerProfile=known-failing` and
`releaseAdmitted=false`.

`PublicPathTests` scans current tracked contents in the normal repository-checks
job. It rejects developer home paths and captured temporary roots. Standard
system paths and the existing synthetic benchmark path fixture remain allowed;
changing compiled test inputs is not required for this publication cleanup.
