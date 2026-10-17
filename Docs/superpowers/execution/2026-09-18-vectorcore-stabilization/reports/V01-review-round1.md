# V01 independent review — round 1

Reviewer: /root/v01_review, Astra/medium. Complete package: V01-review-package.diff, base04f27df and recorded candidate hashes. Spec: issues found. Quality: needs fixes. Root verified the two Important findings against source and accepts them.

Strengths: Correctness.swift:22 rejects shape/empty/nonfinite comparisons; Models.swift:97 validates all four error statistics; BenchmarkHarness.swift:127 propagates thrown bodies through measurement phases; AppMain.swift:472 preserves named failures; RunnerContractTests.swift:64 exercises real runner processes/artifacts.

Important: BatchBench.swift:325–334 discards actual provider reduction and later validates a separate serial computation, also across optimized/generic families and discarded 1536 outputs. A provider returning nonfinite or wrong values without throwing can receive a strict success artifact. Root requires bounded validation of every actual returned observation against an independent aggregate reference, retaining prior failure across warmup/calibration/samples. Existing per-distance checks remain; no tolerance widening.

Important: AppMain.swift:518 suppresses CSV FileHandle.close errors using try?. Root requires checked close in the throwing write path; best-effort cleanup is permitted only after an earlier failure.

Minor: RunnerContractTests.swift:71–75 asserts nonempty failures but not exact original case name/reason. Root requires both assertions.

Reviewer inspected unchanged CSVWriter.swift for the named risk of optional correctness-schema compatibility; it does not consume those fields. No tests rerun. Diff was retrieved in bounded chunks after an initial truncated output. Release was pending at review.

Root additionally received an implementer-observed Release linkage anomaly: the second Swift test runner executed App output rather than the Swift Testing banner despite XCTest pass/exit0. V01 is not accepted on that output. Root authorized a focused diagnosis and a separate benchmark support target plus minimal executable entry if required; the shipped product/CLI behavior and platform/dependencies stay unchanged. Subsequent results and complete diff need re-review.
