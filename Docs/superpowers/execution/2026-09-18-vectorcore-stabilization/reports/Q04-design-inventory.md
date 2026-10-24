# Q04 range-wrapper design inventory

Read-only preparation against shared accepted source; no implementation, builds or runtime measurements. Read WORK_ITEMS Q04 and canonical NUM-22. Q04 depends on Q01/Q03 and requires serialized ownership with Q02/O05 in MixedPrecisionKernels.swift. The existing O03 source verdict remains unchanged.

## Actual routes and semantics

All references below are Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift unless specified.

Nine public wrappers: euclid2 widths512/768/1536 at4660/4678/4696; cosine at4730/4748/4766; dot at4796/4814/4832. Every wrapper currently constructs `candidatesFP16.map { $0.storage }` for the entire array before forwarding its original range. This copies/retains an array of ContiguousArray handles, not every FP16 payload and not an FP32-to-FP16 conversion. NUM-22's O(N) metadata work applies even to a requested empty range; actual optimized allocation/latency still needs measurement.

Three internal @usableFromInline cores begin at4136/4340/4510. They read candidates at global indices range.lowerBound..<upperBound, but write compact output starting at out[0], incrementing outIdx by two then handling an odd tail. Thus result[j] corresponds to candidate[range.lowerBound+j]. An offset output view is relative to its own base; elements beyond range.count must remain untouched. Euclidean is **squared** distance, dot is ordinary positive dot, cosine is1-similarity. No output includes original candidate indices or writes at range.lowerBound in the output view.

The cores borrow two candidate ContiguousArray owners in nested synchronous closures for blocked pairs, or one for the tail. Query storage is FP32 SIMD4; FP16 conversion happens inside the arithmetic only for visited candidates. Query magnitude is currently computed by cosine even for empty range, so eliminating metadata work alone does not eliminate empty-query work. Returning early for a validated empty range can avoid both.

## Current validation, not an invented safe contract

Public FP16 raw constructors enforce exact fixed dimension using guard/fatalError (329 and width siblings), and storage is immutable. Optimized FP32 query shape is already fixed under S01. Wrappers have no independent range/capacity validation. Only Euclidean core has DEBUG-only preconditions for out.count>=range.count and upperBound<=candidate count (4148–4151); it lacks an explicit nonnegative lower-bound guard. Cosine/dot cores have no corresponding upfront checks. Ordinary Array/buffer subscripts may trap later, after work or partial writes. Therefore no consistent existing typed-error or eager validation order can be claimed.

A future brief should bind a common unsafe precondition diagnostic policy for all nine before metadata indexing, query work or output mutation: validate lowerBound>=0 and upperBound<=candidates.count (Range already orders its endpoints), then derive a safe range count, then output capacity. Validate even an empty out-of-bounds range before its early return. Avoid evaluating range.count on extreme negative-to-positive bounds before endpoint validation. Existing APIs are nonthrowing and unsafe-pointer based; adding throwing checked APIs would be a distinct root decision. Do not credit legacy DEBUG-only assertions as Release validation.

## Minimal bounded mechanisms

The smallest ownership-preserving option is: validate; early-return empty; map only `candidatesFP16[range]` to its owned storage handles; pass `0..<range.count` to the existing cores. This uses O(requested count) metadata, leaves all arithmetic/borrow lifetimes unchanged and preserves global input-to-compact-output meaning through the slice. Merely slicing while retaining the old nonzero range would misindex. No FP16 payload conversion or copying is necessary.

An alternative is a generic internal random-access/typed-candidate accessor retaining the original indices and borrowing storage only for selected candidates, avoiding even the O(range.count) handle array. This requires a larger internal-core signature change but no public signature change. Preserve synchronous scoped lifetime throughout; do not extract raw pointers into an array or return them from a borrow. Either mechanism can satisfy NUM-22; root should choose a bounded scope, not presume allocation-free is required.

O05 replaces public escaped fp16Storage getters with explicitly unsafe synchronous withUnsafeFP16BufferPointer. The current range cores already use internal owned ContiguousArray borrows and never use that getter. O05 does not require Q04 to rewrite all internal borrows through the public API. Any new accessor must preserve owner lifetime and must not resurrect the removed escaped-pointer interface. No SoA layout, diagnostics recording or quantization arithmetic changes are necessary.

## Callers and existing tests

Search command: `rg -n 'range_(euclid2|cosine|dot)_mixed' Sources Tests Benchmarks Scripts --glob '*.swift'`. Production Sources contains only the nine declarations and their three core calls: no automatic Operations or tuner caller was found. Benchmarks/VectorCoreBench/Suites/BatchBench.swift:1121–1473 calls Euclidean/cosine across all widths, predominantly full0..<N output. Tests/ComprehensiveTests/MixedPrecisionKernelTests.swift and MixedPrecisionKernelsTests.swift have many public Euclidean/cosine callers; no separate production direct-core consumer was found. Preserve public signatures and valid indexing, so these consumers need no mechanical migration for a slice-rebased implementation. Repeat search at dispatch after concurrent tasks integrate.

MixedPrecisionRangeValidationTests is misleadingly named for this purpose: it validates representable FP16 bit patterns, FP32 range screens and conversion endpoints, **not candidate-index ranges or output buffers**. It supplies no existing nine-wrapper range-boundary proof. New matrix coverage is required rather than assuming that suite closes NUM-22. Existing tests may have scalar FP32 references on unquantized originals; new independent references should use the actual stored FP16 values and preserve squared/dot/cosine semantics.

## Executable work evidence without production targets

For each metric and width, cover N0/1/2/3 and a larger N with valid empty ranges at0/interior/end, nonzero-start singleton, two/three candidates and odd tail. Compare every compact output against independent ordinary finite scalar references, and check exact-bit prefix/suffix sentinels around offset views. Empty output may have nil base and must not be dereferenced. Use unique candidate patterns so accidental rebasing to the wrong candidates cannot pass. Validate malformed endpoint/capacity diagnostics in bounded fresh children and preserve no-write-before-rejection where the new brief promises it.

Use the actual linked Release library with preconstructed candidates/query/output outside measurement. Hold requested range cardinality fixed at0/1/2/3 while increasing unrelated total N; measure allocator event counts/requested bytes with validated no-op/known-allocation controls, exact source/object/executable hashes and identical before/after fixtures. Warm metadata before credited measurements, consume outputs afterward, and separate setup conversion from wrapper invocation. O(N) metadata-array bytes can be visible even with constant allocation-event count, so record bytes as well as events. Do not substitute pointer equality, RSS, a rewritten core or source-only complexity for actual allocation results.

A local immutable accessor/observer seam (if root authorizes one) can count selected-candidate accesses exactly for a generic-core option; do not instrument every candidate with a production global hook. A source-level class check can catch reintroduction of whole-array map across all nine wrappers, but it complements runtime evidence. For the slice-map option, observe O(range.count) metadata and preserved payload owners; no requirement for zero allocations or wall-clock speedup is implied. If the optimizer already removes a baseline allocation, report that outcome rather than forcing it with different optimization flags.

## Separate NaN/activation note

The mixed cosine core's actual policy (4433–4445 and4496–4499) sets similarity0 when denominator is not both >1e-9 and finite, then returns clamped distance1. Consequently zero/tiny norms and nonfinite/overflowed denominators can produce a plausible finite1. This is not universal NaN propagation and should not be silently changed by a wrapper-copy task. Euclidean/dot retain their arithmetic nonfinite behavior; finite fixtures must not hide exceptional outputs in abs/max comparisons.

E06 intentionally activates **FP32 BatchKernels.range_cosine_fused_* / range_euclid_*** through nearest, not these mixed wrappers. Their NaN/tail/blocked policy requires independent route checks; evidence from this mixed cosine implementation cannot establish E06 parity or a shared policy. If Q04 special-value controls reveal an arithmetic inconsistency, return it separately to root before expanding ownership.
