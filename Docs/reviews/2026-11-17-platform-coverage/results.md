# 0.4.0 platform qualification results

**Recorded 2026-11-18.** The local execution required by the
[adopted platform matrix](decision.md) is complete for candidate
`510ea0259998219ef6c0656a2dfcec76190b518a`: version `0.4.0`, no prerelease suffix,
389 compiled inputs, and an iOS 18.0 deployment minimum. Every scope uses the
adopted local Xcode 27.2 build 27B52 compiler pair. All four consumer checkers
return `accepted / known-failing / releaseAdmitted=false`, with zero problems.

## Executed coverage

| Scope | Debug | Release |
| --- | --- | --- |
| Apple silicon macOS 27.0.1 (26A434), full suites | 1,946 passed; 47 skipped | 1,945 passed; 47 skipped; one known canary failure |
| arm64 iOS 18.0 (22A3351) simulator, full suites | 1,829 passed; 164 skipped | 1,828 passed; 164 skipped; one known canary failure |
| arm64 iOS 27.0 (24A434) simulator, full suites | 1,829 passed; 164 skipped | 1,828 passed; 164 skipped; one known canary failure |

Both simulator runtimes reconcile all three test bundles and all 1,993 declared
methods, including parameterized outcomes, skip reasons, enclosing runner
results and summary failures. The known Release failure remains
`LostStoreCanaryTests.literalArrayStoreIsObservedAfterOutOfLineClosure`: it
observes 2.0 where 1.0 is required. It is not reported as a numerical pass.

| Consumer scope | Result |
| --- | --- |
| Native Debug/Release producer × client | 36 groups; 96 supported rows pass, 166 control rows match; 11 wrong diagnostic rows remain visible |
| Each simulator's Debug/Release producer × client | 52 app groups; 96 supported rows pass, 168 control rows match; 14 wrong diagnostic rows remain visible |
| Physical iPhone 17, stable iOS 27.2 (24B119), Release/Release | 13 app groups; 24 supported rows pass, 42 control rows match; seven wrong diagnostic rows remain visible |

Native migration contracts and all four optional C-on/off × Debug/Release
build cells also passed their reviewed checks. No supported numerical failure
was waived. The [compiler support restrictions](../2026-11-10-compiler-acceptance/decision.md)
still exclude additional descriptor/view inout handoffs through helper or
conformer chains; these results do not establish general compiler correctness.

## Evidence and limits

[Portable scope and digest records](results.json) identify the private profiles,
captures, suite reviews and frozen source snapshot. Raw logs, artifact inventories,
signing/device identifiers and the failed first phone attempt remain private.
That attempt stopped after the host killed an Xcode build process; a fresh full
retry completed. Initial profile/report preparation errors are retained too;
the accepted reports use complete artifact bindings and reviewed reconciliation.

Two previous physical-run summaries incorrectly named iOS 27.2. Their retained
profiles, captures and app frames identify **iOS 27.3 beta (24C5037i)**. The
[alpha tooling summary](../2026-11-10-compiler-acceptance/ios-tooling.md) and
[prior stable-candidate summary](../2026-11-10-compiler-acceptance/stable-candidate.md)
now record that correction; original evidence bytes remain unchanged. The
new physical run above supplies stable-iOS evidence for this candidate.

Minimum-macOS, Intel, tvOS, visionOS and additional Catalyst runtime coverage
remain explicitly deferred. Existing required compilation jobs remain required.
Fresh PR/CI and hosted numerical qualification are still needed for the changed
candidate; the earlier hosted profiles qualify their original inputs only.
Hosted use of the new private evidence requires a separately reviewed storage
and publication arrangement. Broader sanitizer, controlled-performance,
final-release-reference and release-approval gates also remain open.
