# iOS app-hosted tooling validation — 2026-11-12

The simulator and physical-device app captures satisfy the adopted bounded
policy on the preserved alpha inputs. Both reports retain
`compilerProfile=known-failing` and `releaseAdmitted=false`. They do not qualify
the separately selected 0.4.0 candidate.

| Scope | Result |
| --- | --- |
| Simulator, iOS 27.0 (24A434), arm64 | Four Debug/Release producer × client cells; 52 app groups, 96 supported rows, 168 control rows; 14 incorrect diagnostic rows remain visible |
| iPhone 16 Pro Max, iOS 27.2 (24B119), arm64 | Release producer/Release client; 13 app groups, 24 supported rows, 42 control rows; seven incorrect diagnostic rows remain visible |
| Full simulator package suites, alpha | Two Debug attempts failed only in the batch-throughput timing test, at 60 and 180 seconds respectively; each recorded 1,829 passed, 163 skipped, one failure. Neither attempt reached Release |

Both app runs used Xcode 27.2 build 27B52, the adopted compiler fingerprint,
the iOS 27.2 SDK, and iOS 17 deployment targets. This does not establish minimum
OS coverage. Simulator execution does not substitute for the physical run.

All supported by-value rows matched their literal numerical expectations.
The optimized original CopyInoutReturnInside, LostStoreObserved, and
ReturnInsideInout diagnostics remain incorrect, as do concrete-inout and the
three residual rows `new.concrete.copied-inout`, `new.inside.copied-inout`, and
`new.aggregate.inout`. The app adapter preserves the helper/probe bodies;
these app-hosted compositions complement the native originals.

The host writes a fresh UUID-bound begin/end record, its own executable hash,
runtime identity and the complete numerical body. A successful install or
launch cannot replace those records. The separate witness target and actual
Swift compile/link commands are checked. The focused qualification suite has
90 passing tests, including stale/missing/duplicate records, platform/SDK,
producer/client, object/module and canonical-path mutations.

## Retained evidence

Canonical evidence and exact profile hashes are in
[ios-tooling-result.json](ios-tooling-result.json). The simulator profile is
[`ios-simulator-27.0-arm64-alpha-2026-11-11.json`](../../../Scripts/ci/qualification_profiles/ios-simulator-27.0-arm64-alpha-2026-11-11.json).
The physical profile stays in the local private archive because its raw
commands contain device and signing identifiers. Public reports omit those
identifiers. Signing, installation and provisioning were explicitly approved.

The archive contains all captured products and receipts. Original temporary
paths remain in receipts; the validator maps them to the archive and to
`/Users/goftin/dev/projects/VectorCore-archive/VectorCore-compiler-acceptance-2026-11-10-evidence/source-alpha-ios-tooling`.
Use that preserved source root when revalidating alpha evidence. Revalidation
against the changed live candidate correctly rejects the benchmark-source
hash mismatch. Never replace the old manifest with the new one.

Two verifier parsing defects were corrected after collection: a linker LTO
output had been mistaken for an input, and joined Swift `-D` conditions were
not recognized. RED→GREEN tests cover both. Profiles pin separately preserved
collector source; the simulator snapshot matches the initial profile's
immutable pins. No app binary, command, adapter, numerical output or oracle
was changed. The final independent review must assess this provenance choice.

Incomplete earlier attempts are retained: the first simulator host lacked
the required scene lifecycle and crashed; the first physical attempt could
not resolve a locked device. Neither supplies qualification evidence.

## Candidate freeze and remaining work

The user approved making only the timing-only Euclidean batch benchmark
opt-in with `VECTORCORE_TEST_EXTENDED=1`. Its body and the ordinary numerical
tests are unchanged. Nine performance-policy tests pass; mutation tests cover
every timing gate and preserve reachability of the five ordinary tests.

The user then selected version `0.4.0`, `prerelease=nil` for qualification.
The complete frozen compiled-input manifest is
[stable-candidate-source-inputs.json](stable-candidate-source-inputs.json),
SHA-256 `50d4861428d63368c309f0e86bfbf376f68a79df08db96a5b164aaaf37abe830`. The commit containing this manifest is the selected
candidate; fresh capture records will identify its full commit SHA. This
selection grants no release, tag, merge, push or hosted-dispatch approval.

Fresh native, full simulator suites, simulator app matrix and physical
Release app captures are still required for that candidate. Hosted collection
and acceptance use the user's two-stage review decision. Minimum-OS, Intel,
other platforms, sanitizer/performance and final release-reference gates
remain separate.
