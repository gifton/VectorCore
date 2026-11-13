# Hosted qualification and preserved CI reachability

The user selected two-stage evidence review. The new manual
`compiler-qualification.yml` workflow collects evidence for an immutable
candidate, then verifies a separately reviewed bundle in a later dispatch.
Local workflow tests do not establish a successful hosted run.

## Collection and review

`mode=collect` takes the full candidate commit. The policy/helper checkout uses
the dispatch's `github.sha`; the candidate checkout uses the requested commit.
The helper verifies clean checkouts, the actual Xcode 27.2 build 27B52,
frontend hash, and arm64 architecture before running the candidate's collectors.
The explicit developer directory is `/Applications/Xcode_27.2.app/Contents/Developer`.
The runner label does not grant acceptance. Python 3.12 or later is required.

The simulator job requires an explicit installed runtime identifier, creates
its own simulator, and records the actual runtime/SDK/target identities. It
does not select another attached device. Native collection retains full suites,
consumer matrices, original/residual diagnostics, migration contracts and
C-on/off routes. Simulator collection retains the four app-hosted cells.
Collection deliberately leaves both platform jobs and their aggregate
unqualified, even if collection and artifact upload complete successfully.

Review the raw evidence before committing a profile to the policy revision.
Use literal numerical contracts; never derive expected correct values from the
new output. Each profile adds a `reviewedArtifact` object with these exact keys:

```json
{
  "runId": "<collection run ID>",
  "repository": "gifton/VectorCore",
  "candidateCommit": "<full candidate commit>",
  "policyCommit": "<full collection workflow commit>",
  "platform": "native or ios-simulator",
  "name": "native-evidence or ios-simulator-evidence",
  "sha256": "<SHA-256 of evidence.tar.gz>"
}
```

These values must match the retained `collection.json`; the digest must match
the archive bytes. Record the newly reviewed environment in the ordinary
qualification profile, including the complete source manifest. Local manifests
conservatively include three existing `.DS_Store` files; a clean hosted checkout
can therefore have a different full-tree digest while its tracked compiled
inputs remain fixed by the same candidate commit. Review that difference
explicitly instead of reusing a local profile.

`mode=verify` downloads named artifacts from the exact run ID in the same
repository. The selected profiles must be tracked, unchanged JSON files inside
the workflow checkout's `Scripts/ci/qualification_profiles/` directory; paths
and symlinks cannot escape it. Archive digests and provenance are checked before
extraction. Only directories and regular files are extracted. No archived
helper or binary is executed. The shared checker verifies retained receipts,
products, numerical observations and the selected candidate's source tree.

A verified job succeeds only with scoped `accepted`, `known-failing`, an empty
problem list and `releaseAdmitted=false`. The report records both the collection
policy revision and the later reviewed policy revision. Missing, failed,
cancelled or skipped platform jobs reject `qualification-required` through
the existing `check_required.py`. Upload success is never a substitute.

## Reachability

| Gate or route | Reachability after this change |
| --- | --- |
| Lost-store canary in Debug | The original strict expectation remains active |
| Lost-store canary with compiler ≥6.4.1, including adopted 6.4.2 | The original strict expectation remains active; exact diagnostic failure is retained and assessed outside the test process |
| Compiler 6.4.0 and pre-6.4 optimized branches | Existing `withKnownIssue` behavior is unchanged; the new workflow does not certify these branches |
| Captured-variable and supported numerical tests | Assertions are unchanged; known diagnostic names do not waive failures |
| Four precision timing workloads | All require the existing exact `VECTORCORE_TEST_EXTENDED=1` opt-in; the five ordinary precision tests remain ungated |
| Optional C routes | Native collection still invokes the unchanged four-cell `VC_USE_C_KERNELS` driver with observable route, numerical and tail controls |
| arm64 paths | Reachable on the new native and simulator jobs, with runtime and build identities recorded |
| x86_64 paths | Not reached by the new arm64 jobs. Existing Intel compilation remains in legacy CI; Intel runtime qualification remains open |
| Legacy CI Required | Unchanged seven jobs: test, lint, repository-checks, consumer, platforms, benchmark-contracts, sanitizers |
| Legacy compiler/platform cells | Xcode 16.2/26.3 test cells, simulator/Catalyst library builds, all package products, Intel library/test compilation, sanitizer jobs and benchmark contracts remain unchanged |
| SDK name collision job | Remains outside legacy CI Required, as before |
| New qualification aggregate | Requires successful native and simulator **verify** jobs; collect dispatches cannot satisfy it |
| Full iOS package suites | Retained separately in local candidate qualification; the hosted app matrix does not claim to cover them |
| Physical iOS | Requires its separate app-hosted Release receipt; no device identifier, provisioning material or private profile is uploaded by this workflow |

Minimum-OS execution, Intel runtime, Catalyst/tvOS/visionOS execution, current
sanitizer/performance evidence and final release-reference/admission decisions
remain outside this workflow's acceptance. Existing CI continues to run and
must be assessed separately.

## Introduction and dispatch prerequisite

The GitHub workflow inventory inspected on 2026-11-12 does not yet contain this
workflow. GitHub requires a manual workflow to exist on the default branch
before dispatch. A reviewed integration or a separately approved bootstrap is
therefore required before a first hosted run. No default-branch change, push,
dispatch, tag or release has been performed by this implementation.
[GitHub manual-workflow documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)

The frozen local candidate is
`af315c56c52de737f868fd969e6e1f5ae59eaa48` on
`codex/audit-top-six-2026-10-27`. Collection inputs will be `mode=collect`, that
candidate SHA, and the reviewed installed simulator runtime. Profile paths and
an evidence run ID are supplied only after the collection has been reviewed.
The exact policy commit/ref and introduction action must be prepared and
approved after independent implementation review; this document does not
authorize shared-repository changes.
