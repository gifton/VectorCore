# R03 independent implementation review

## Frozen source verdict

**ACCEPT source/contract review; final runtime acceptance pending.** No concrete source blocker found. Reviewed the complete six-file patch and complete new test and public-consumer files against approved R03. This is not a claim that pending final Debug, Release, ASan, or consumer executions passed.

Binding brief SHA256: `b0a7130d5df896dcf41dc1faaf0b55d02fa304801bfd6ca448a57b47aab1077c`. Base: `076f60b8da6b4cad80a36be73b9506d06eb90bf9`. Full diff SHA256: `ee7b4e931f33d351129d6378ad4a76aba388171a32a2c0edd2cd860480b6e186`.

Independently checked both stated hashes, all six final manifest hashes/sizes, all 268 baseline code hashes against exact git blobs, and all 270 final code inventory hashes against the implementation archive. An in-memory unified-patch reconstruction of every changed/new file produced the exact archived bytes. The complete inventory differences are exactly the four authorized production files, MatrixMetricContractTests.swift, and the public consumer fixture. No MatrixDistance, existing test, Package, numerical normalization/statistics, pool, configuration actor, or scheduling implementation changes appear.

## Contract and integration findings

BatchOperations evaluates the supplied metric on every diagonal in its serial and parallel paths. Its existing optimized-type/Euclidean-or-Cosine GEMM gate remains unchanged; the wrapper replaces only diagonal cells using the direct requested metric. Off-diagonal GEMM arithmetic remains untouched. ExecutionOperations includes self pairs in both existing triangular paths. Mirroring is consistent with the now explicit symmetric custom-metric contract; arbitrary directed-metric equivalence is not promised.

The Dim768 public cosine-similarity convenience retains normalization once, obtains negative-dot scores, then negates them. Finite results are clamped to [-1, 1], while nonfinite results remain nonfinite. This establishes the intended ordinary finite identical/opposite/orthogonal/zero behavior without claiming a new extreme-range normalization result. N01 remains responsible for normalization; the statistics and pool algorithms are unchanged.

All three general zero-query SoA implementations now write indexed elements only in 0..<N, after the existing N>0 guard. Thus empty routes do not dereference an empty base address, and oversized or offset output views retain elements outside the logical candidate prefix. Output-capacity requirements, Safe512 validation, logical byte gate, padded layout, nonzero arithmetic, and route thresholds remain unchanged.

## Test and oracle assessment

The complete new tests cover direct public pairwise results at N0/1/2/99/100/101 for all built-in Float metric families, configured Hamming/Minkowski variants, and a symmetric custom metric with nonzero diagonal. Every cell is checked against both supplied metric evaluation and separately written Double scalar formulas; shape and finiteness are checked before tolerances, with exact direct-result diagonal bit checks. The Double formulas do not invoke the production metric kernels. Chosen ordinary examples and finite-aware tolerances support the scoped sign/zero/metric claim, not near-cancellation GEMM accuracy.

ExecutionOperations and public similarity tests include N31/32 across its existing 1000-cell threshold, plus empty/small cases. Explicit NaN/Inf similarity controls prevent finite clamping from concealing nonfinite results. SoA coverage includes widths 512/768/1536; blocked2, blocked4, tiled, public wrappers, and Safe512; zero and nonzero queries; logical output capacities N and N+2; offset views and bitwise surrounding sentinels; N0/1/3/4/5, tile neighbors, and integer neighbors of the actual 8 MiB logical-byte gate. The insufficient-capacity Safe512 test retains the checked error contract.

All 45 global configuration scenarios execute in fresh filtered children, while the parent only schedules scenarios. Children retain and restore all six configuration fields. Width/type/metric/configuration gate observations and custom-witness call counts establish the intended route controls at source level; the direct MatrixDistance comparison is a supplementary off-diagonal control, not the independent oracle. Default threshold neighbors, disabled routing, and small-threshold neighbors cover all three optimized widths and both GEMM metrics.

The child runner installs its termination handler before launch, captures to a file rather than a pipe, limits execution to 15 seconds and owned-child cleanup to two additional one-second waits, and reads exit status only after termination. Missing body/completion/invocation markers and missed deadlines fail the parent. The parent child-entry no-op must be separated from actual scenario execution in final evidence. Inherited environment preserves sanitizer settings.

The public fixture exercises observable diagonal/sign/zero behavior and the checked SoA suffix contract using ordinary public import. Its final compile/link/run association remains to be verified.

## Remaining evidence gate

Final adjudication must associate the unchanged frozen source with actual final Debug/Release commands and results, all child capture/parent associations and loaded test bundle provenance, focused SoA ASan execution, and public consumer compile/run evidence. Earlier Debug activity is not substituted for a frozen final run. No builds or runtime reruns were performed for this review.

## Follow-up claim-discipline check

A narrow documentation correction remains before final acceptance: the three new public SoA wrapper comments assert logical-prefix writes and suffix preservation without naming their exercising test. Their bodies dispatch to helpers, so the full claim is not a one-step derivation from adjacent implementation. Add an explicit reference to `MatrixMetricContractTests.soaLogicalOutputMatrix` to each comment, or clearly identify the statement as a contract. This does not require arithmetic, routing, or test changes and does not alter the favorable production review.

`R03Calls: @unchecked Sendable` has a complete, directly visible justification: its immutable private NSLock guards every read and mutation of the sole private mutable Int; no mutable state escapes. It is not a compiler-workaround concern and needs no source change.

## Final v2 evidence adjudication — ACCEPT

**Final verdict: ACCEPT R03 v2.** The documentation gap above is resolved. Independently verified full v2 diff SHA256 `b1a4d7c6ef245418ee6f41f76cb741e2f83a4f8f6a5d464a73773fad47e71ac8`, all 270 final source hashes, and the precise v1-to-v2 equivalence: only three identical named-test documentation lines were added to BatchKernels_SoA.swift; removing them restores its v1 hash, and all other 269 code files match v1. Final runtime records correctly refer to the previously reviewed v1 executable code. No comment-only behavioral rerun is claimed or required by the root-approved equivalence.

Verified all 167 worker evidence records against actual retained bytes and sizes. Verified all recorded build artifact hashes (325 Debug, 324 Release, 325 ASan), 133 public linked object/module artifacts, and the standalone public executable. Read the exact final command records and validation driver: the driver checks all code inventory hashes before and after each command and records the frozen diff identity. All five final command/log associations have exit zero and matching hashes.

Final Debug and Release each report 77 functions in 16 suites. Separately checked every one of the 90 final child captures: 45 scenarios per configuration, actual invocation JSON and selected test-bundle argument, matching bundle hash in the configuration's artifact inventory, route/body/completion markers, one-test successful child summary, and exact original parent capture marker/log hash. Their capture set is disjoint from the separately retained 45 intermediate captures. The parent isolatedConfigurationChild entry is a no-op and is not counted as an actual configuration scenario.

Focused ASan selects exactly the three SoA functions and reports five actual invocations (three width arguments plus two individual cases), all passing. It selects no configuration children or parent no-op entries. `ASAN_OPTIONS=detect_leaks=0` is explicit; acceptance concerns the selected valid output-boundary paths, not leak freedom or unrelated routes.

The standalone public fixture compiled/linked with Swift 6 and complete concurrency checking, then exited zero with `R03_PUBLIC_MATRIX_MIGRATION_COMPLETE`. Its exact object arguments and artifact hashes associate that run with the reviewed code. Earlier setup/compiler failures and intermediate successes remain identified separately. No new builds or runtime reruns were performed during adjudication, and no broader numerical, performance, scheduling or downstream claim is inferred.
