# Hosted metadata environment contract repair — wave 3

Hosted run `41940013803`, candidate `90b8d583055fffa729558fb10b96eb7e30e92fda`, passed the native reader setup and all twelve fixed literal-preservation rows in the metadata environment test. Xcode 26.3 Release job `126958111714` failed only the two `/usr/bin/env`-prefixed `DYLD_LIBRARY_PATH` and `DYLD_FRAMEWORK_PATH` rows: the literal value survived, while the test required it to be absent. Their paired unavailable-reason assertions also failed. This is a test expectation defect exposed after the copied-system-binary fixture defect was repaired.

This change compares those two rows with an independent, direct-exec native baseline on the same host. It retains fixed literal expectations for the other twelve rows, executes every row, and rejects baseline execution failures rather than interpreting them as stripping. There is no production change and no inference about hosted SIP configuration.

## Evidence and diagnosis

The source-bound Xcode 26.3 Release summary records Swift Testing 1,757 tests passing with the previously documented known canary issue, and XCTest 208 tests with 28 skips and four assertion failures in this one metadata test. The subsequently completed Xcode 16.2 Debug and Release jobs show the same sole metadata failure and four assertions; both have 1,757 Swift Testing tests passing with no known issues. Their copied logs/summaries and hashes are included in input-provenance.json. Xcode 26.3 Debug was still active at the last coordinator update. `xcode263-release-log.stdout` and `xcode263-release-summary.json` are copied into this evidence directory. The job log SHA-256 is `742c4a969dbcfaa439e55c59dadf1d3939c62b832a0c34c95145b34c719ee0f8`.

Before source edits, an independent local process runner compiled the existing test-owned native reader using the selected developer compiler and ordinary arm64 architecture, then directly executed all fourteen command/key combinations with the literal value in the environment. It did not invoke EnvCapture or a shell. Direct reader execution preserved all seven keys; `/usr/bin/env` followed by the reader stripped the two DYLD keys and preserved the five others. No injection marker appeared. `independent-baseline-result.json` retains reader/source/system-env hashes and results, with each command, environment, deadline, raw output and exit in separate records. This is evidence that the local behavior differs from the hosted observation, not evidence of the reason for the difference.

## Test-owned repair

`RunnerContractTests.swift` retains the seven keys, direct/protected command shapes, shell substitution text, startup-file variables, function-shaped environment value, and injection marker assertions. Both value and unavailable reason are now checked exactly. A host-dependent expectation is permitted only for the two protected DYLD rows, and it must equal a measured baseline; the test does not accept either value or nil from EnvCapture.

The baseline uses the existing bounded, marked XCTest child and takes a separate fixture branch before any EnvCapture call. The branch constructs the environment explicitly and replaces the child through `execve`. The native reader writes exact output bytes to a unique capture file and reports its PID and process group. The parent verifies those match the launcher's child identity, requires ordinary completed/reaped termination, and classifies only:

- Exit 0 with the exact literal plus newline: preserved.
- Exit 1 with an empty native capture: stripped.

Missing native identity/capture, signal termination, timeout, incomplete cleanup, unexpected output, partial-error output, and other exit statuses fail the baseline. The independent path uses no shell, EnvCapture guardian, or production metadata code. Reader compilation and its preflight still use the existing separately verified setup helper.

`ChildProbeLauncher.swift` adds one package-scoped replacement helper. It requires the current process's existing child marker, checks the forwarded marker matches, requires an absolute executable and NUL-free argv, rejects malformed/NUL-containing environment entries, checks allocation, and throws on failed `execve`. It creates no process and does not change selector construction, nested-launch refusal, process-group creation, deadlines, reaping, or cleanup. The existing launch-policy and bounds tests require no exceptions or allowlists.

## Regression and control coverage

The old expectation was factored into a test-owned expectation function without changing its behavior. A new host-policy table supplies both preserved and stripped baselines for each DYLD key. The fresh Debug red run compiled successfully, then failed exactly two assertions for the preserved-baseline cases; the other four focused XCTest cases passed. `red-sources/` and `metadata-red.*` retain that state. The repair makes the expectation depend on the measured baseline; unprotected expectations remain fixed.

Three new XCTest cases provide mechanical coverage:

- `testMetadataEnvironmentBaselineRejectsAmbiguousResults`: both host-policy outcomes for both DYLD keys; exact preserved and stripped native results; eight rejected combinations covering empty/wrong success output, nonempty absent output, unexpected exit, signal, timeout, incomplete cleanup, and missing termination.
- `testMetadataEnvironmentDeterministicPolicyControls`: native preserve, explicit `unsetenv`, and partial-output/exit-7 paths for each DYLD key, through the independent baseline and actual EnvCapture. Preserve and strip require their exact expected observations. Exit 7 must be rejected as a baseline and reported as an exact production command failure. Injection markers remain absent.
- `testMetadataDirectExecRejectsInvalidRequests`: unmarked use, empty/relative/NUL-containing argv, empty/equals/NUL-containing environment keys, NUL-containing values, marker mismatch, and missing executable. These controls fail explicitly rather than executing a truncated command or being interpreted as an absent variable.

The existing deliberate native compiler-error control remains required. The full fourteen-row matrix also validates the native target's PID/group continuity in each direct baseline; the twelve fixed expectations are checked independently of those baselines.

## Verification and remaining boundary

Debug green: five focused XCTest cases pass, zero failures, with an incremental build of 2.88 seconds and total command time 8.37 seconds. The initial fresh Debug red build took 121.01 seconds. Fresh Release passes the same five XCTest cases with zero failures: build 337.22 seconds, total command time 342.91 seconds, recorded in `metadata-release.*`. Each configuration records twenty native direct-exec results: fourteen matrix rows plus six deterministic controls, including both expected exit-7 errors. All eighteen non-error native executions match exact bytes/status and PID/group identity; both error executions are rejected. Debug and Release used fresh wave 3 scratch directories and at most two compiler jobs; wave 1/2 build products were not modified.

Python coverage includes twelve launch-policy tests, twelve child-bounds tests, six CI contract tests, and nine sanitizer-driver tests. The final launch/bounds records bind to both edited Swift files. The driver from `44567c88ae7c6e63e93625f3e05a0d8a01c6c93b` remains untouched. No manifest, workflow, public API, package requirement, numerical gate or production source changed. `git diff --check` passes.

These are focused local Debug/Release validations on macOS 27.0.1 arm64 with the qualified Swift.org 6.4.2 payload and installed Xcode SDK 27.2. They do not replace the required hosted matrix. The next candidate must demonstrate the actual hosted direct baseline and matching EnvCapture result on both pinned Xcode versions/configurations, plus all unchanged sanitizer and downstream gates. Existing known numerical-canary policy is unchanged.

## Evidence

Evidence root: `/private/tmp/vectorcore-qualification-2026-11-04/hosted-fix-3`. Earlier wave evidence and builds are unchanged.

`before-identity.json`, source snapshots, per-command JSON records, pre-edit independent native reader source/binary and compiler output, independent baseline records, `metadata-red.*`, `metadata-green.*`, `metadata-release.*`, direct Python test records, and final built-product/source hashes provide source-bound local evidence. The archive inclusion manifest excludes build caches while retaining their product identities. Full hosted logs remain under the parent's `hosted-run-2` evidence as well as the copied metadata failure input here.
