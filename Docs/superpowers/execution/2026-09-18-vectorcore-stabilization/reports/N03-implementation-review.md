# N03 independent implementation review

## Verdict and evidence boundary

**ACCEPT SOURCE; final runtime acceptance pending.** No blocking mathematical, scope, routing or oracle defect was found. This review does not claim final Release or final command/log/source association before that evidence arrives. No builds, production/test changes, nested agents or index/commit operations were performed.

Binding brief SHA256: `976f77a65c56db104c7187b98cb4d6404ef8d1cf0cf7834d3543acd8870637a5`. Base: `1c55a85b46f026fbcad8d27face1159000be4501`. Complete patch SHA256: `cfc15ec332f833236e3ed636b753f397c7785b32a9f0633f08686200a9bd9904`.

## Complete scope and provenance

Read the complete seven-file diff, including both existing production hunks, the complete accumulator, test suite, generated fixtures, Decimal generator and all provenance rows. Independently verified all seven final hashes, both existing-file base hashes, the patch hash and brief hash. Reconstructed every patch hunk against the exact base in memory and compared complete resulting files with the frozen implementation archive. All seven canonical backup files match the archive byte-for-byte.

An independent git-object inventory of Sources/Tests/Scripts found 255 tracked files, none missing, exactly the two authorized tracked changes (BatchOperations and SyncBatchOperations), and exactly five authorized new files. This count has the explicit three-directory scope; it does not replace root's broader 256/261 code inventory. Production changes are restricted to statistics bodies/docs, the internal explicit-partition statistics helper and BatchStatistics semantic docs. E03 configuration mutation, N01 normalization, factory/provider dispatch, default thresholds/chunk selection and unrelated Sync methods remain unchanged.

Only BatchOperations.statistics is public among the reviewed statistics entry points. SyncBatchOperations.statistics, Array.batchStatistics and SyncBatchOperations.finiteStatistics are internal routes. Read the unchanged VectorProtocol.magnitude expression and finiteStatistics/filterFinite forwarding; the latter still filters coordinate finiteness and then summarizes retained vectors' delivered magnitudes.

## Mathematical assessment

The accumulator widens each delivered Float exactly to Double. Its update uses delta = x - mean, mean += delta/n and M2 += delta²*(n-1)/n, the centered Welford identity. Its merge uses the Chan correction delta²*nA*(nB/(nA+nB)), with Double conversion before count products. Neither route squares Float magnitudes nor rounds the mean to Float before computing deviation. Population variance is M2/count, followed by Double square root and one Float projection.

Every M2 increment is nonnegative. For finite Float input, differences and squared deviations remain far below Double overflow even with an Int-sized count (the approved conservative bound is below 2^321). Count addition is checked before mutation; production increments and merges additionally derive from disjoint ranges of one actual array. Empty merges preserve identity. Repeated identical finite measurements keep exact zero delta/M2, including maxFinite and subnormal values. The range argument is not an all-input accuracy theorem.

Every add increments the count, including nonfinite measurements and subsequent values after invalidation. Invalidity is sticky through updates and merges. Empty results contain positive zeros; any delivered NaN or infinity yields full count and two NaNs. There is no invalid-variance clamp. Singleton finite deviation is positive zero. Generic magnitude is evaluated once per vector in the statistics loop; its own two-pass coordinate implementation remains outside N03's arithmetic scope.

## Actual routing and partition behavior

The public empty return and serial/parallel threshold are preserved. Below the default threshold the private serial body uses the same accumulator as the internal synchronous route. At/above it, the public route calls the production-used explicit-partition helper with the unchanged optimalChunkSize result. The helper uses start + min(chunkSize, remaining), so a positive Int.max chunk is safe. Tasks carry original chunk indices; the parent sorts those indices before merging, independent of completion order.

The helper deliberately preserves N03's many logical chunks and existing task fan-out; E02 owns subsequent bounded worker scheduling. Ordered merges are deterministic for a fixed input/partition, not bitwise identical across every partition or permutation. The positive internal chunk-size precondition is not a new public configuration knob. No deliberate invalid-chunk child test is claimed by this package.

## Tests and oracle independence

Eight test functions cover public threshold neighbors 999/1000/1024, internal synchronous and Array forwarding, repeated finite values at ordinary/extreme/subnormal ranges, empty/singleton/zero-dimensional/mixed-dimensional inputs, and fixed/optimized ordinary controls. Single-coordinate DynamicVector inputs expose known nonnegative Float magnitudes through the real generic expression; tests do not rely on a shadow magnitude property as a fake protocol witness.

The partition matrix exercises actual helper tasks for chunk sizes 1/2/3/127/256/1024/larger-than-N/Int.max, original/reversed/deterministically shuffled orders, and uneven final chunks. It repeats fixed partitions for exact result-bit checks. Deterministic ordering is also directly established by the adjacent sorted merge code; no controlled completion-order perturbation or scheduler-bound proof is claimed. Direct accumulator split tests supplement, rather than replace, public-route tests.

NaN and both infinity signs occur at first/middle/last positions in serial and parallel-sized arrays, and across explicit partial boundaries and permutations. Finite coordinates whose magnitude overflows are exercised in public, internal Sync, Array and finiteStatistics routes, including a 1000-row public parallel case. finiteStatistics retains coordinate filtering semantics, with explicit filtered-empty and ordinary retained-row controls.

Closed-form two-point expectations use widened counts and an independent weighted mean/deviation equation. The offline generator reconstructs exact nonnegative binary Float inputs, computes a 200-digit Decimal mean then a separate centered second pass, and rounds references by comparing neighboring Float values with ties-to-even. It shares neither the Welford update nor Chan merge with production. The six cases include max range, actual adjacent large Float values, tiny subnormals, mixed exponents, a small outlier and normal-boundary neighbors. The generator's sign-free decoder is appropriate to these explicitly nonnegative magnitude fixtures, not a general signed-Float decoder.

Independently executed the generator's computation in memory and reproduced both the Swift fixture text and JSON provenance byte-for-byte without modifying files. Finite comparisons first assert finiteness and then use the brief's 16 ULP plus 8 least-subnormal allowance; expected positive deviations additionally require a positive result. Empty mean and required exact-zero deviations use bit-pattern checks, preventing a broad near-zero allowance from accepting spurious nonzero identical-value deviation. Nonfinite cases use classification and exact count assertions.

## Pending final adjudication

Root reports five behavioral RED tests/161 issues after a separately retained macro setup failure, and eight focused Debug test functions passing. These reported outcomes have not yet been promoted here to final independent runtime acceptance: the worker's exact final command/evidence package and Release completion remain pending. Final review must associate logs, exits, selected consumers and frozen hashes, while keeping setup failure distinct from behavioral RED. No ASan, throughput, RSS, component-covariance, downstream execution or all-input numerical-accuracy claim is made.

## Final evidence adjudication — ACCEPT

**ACCEPT FINAL frozen v1.** This supersedes the pending-runtime verdict above while preserving the original source review. Independently verified the final evidence inventory SHA256 `061f4a35ca8689c1c2e646ad461a8150b6a3453c5c547e01423106f935a025c2`, all 16 listed artifact hashes/lengths, and all 261 final source inventory entries against the implementation archive. The complete v1 patch and seven-file manifest remain unchanged.

Read the final implementation report and exact five-command records. Both final commands run from the frozen implementation archive with separate N03 Debug/Release scratches, six jobs, and filter BatchStatisticsRangeTests. Each records exit 0 and its actual log reports eight Swift Testing functions in one suite passing (zero selected XCTest cases). Final Debug log SHA256 is `8d911abf41356e6fd4257761b6f6edf989ec4e8250fcb3be6e46a12706a6d9fd`; final Release is `fb5b97710042a33bc5821315b813fc5fa84836b40db42c682a0ce0644fa486a9`. Their source association points to the independently verified frozen v1 manifest.

The retained setup log contains the Testing-macro compilation error and is correctly excluded from behavioral RED. The corrected RED log actually reports five tests/161 issues with exit 1; both RED source records retain the exact base hashes for the two production files. Initial GREEN is separately retained and does not substitute for the final rerun after the two assertion additions. The final report's narrow filter follows its existing-consumer inventory; the new suite itself covers the internal Sync/Array/finiteStatistics consumers. No raw-memory changes, sanitizer result or deliberate trap-child evidence are claimed. Source and final recorded runtime evidence satisfy N03's bounded contract; root integration remains separate.
