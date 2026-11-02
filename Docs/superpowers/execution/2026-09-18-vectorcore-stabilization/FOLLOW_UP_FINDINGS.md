# Execution findings beyond the original bounded briefs

Historical audit records stay immutable. These entries do not claim runtime reproduction until evidence is recorded.

## EX-02 — Factory nearest-dimension lookup can overflow at negative extremes

Source-established during S02 contract preparation: `VectorTypeFactory.optimalDimension(for:)` computes `abs(supported - approximateSize)` before any bounds guard. Int.min cannot be safely subtracted from a positive supported width. No runtime probe executed yet. Assigned narrowly to S02: clamp outside the supported interval, preserve nearest-width selection and smaller-width ties, and run a separate bounded child-process RED/Debug/Release matrix for Int extremes. No production allocation/workload target or historical audit changes.

## EX-01 — INT8 mutable-borrow throw policy needs explicit family coverage

Status: source-established behavior; contract/repair pending Q01 numerical boundary design. Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift at the three Vector512/768/1536INT8 withUnsafeMutableBufferPointer methods converts to a temporary FP32 vector and re-quantizes only after a successful body return. A thrown body skips all writeback. This is another occurrence of temporary-buffer writeback behavior encountered during O02 design; O02's accepted scope remains TieredStorage only and must not imply all VectorProtocol conformers now share partial-write semantics.

Before Q01 dispatch, root must specify whether quantized borrows commit partial writes on throw and how invalid/nonfinite written values are handled without a new cleanup-time trap. Simply adding defer could activate the existing nonfinite quantization failure path. Add a mechanical three-width normal/throwing mutation matrix with retained-copy and quantization-parameter controls once that contract is chosen. Coordinate with S01 shape guards and Q01 validity policy; do not silently expand O02's currently running source ownership.

## EX-03 — Error construction overflows at negative dimension extremes

S02 baseline encountered a trap while basis(Int.min) constructs an indexOutOfBounds error. Source confirms VectorError.indexOutOfBounds computes dimension-1 solely for max_index metadata, which overflows for Int.min. Root authorizes a narrow S02 scope extension to that helper: preserve original index/dimension in message and .indexOutOfBounds code; max_index is dimension>0 ? dimension-1 : -1 because nonpositive dimensions have no valid index. Direct mechanical error-construction regression over dimension Int.min/-1/0/1/Int.max and extreme indices must establish RED in a bounded child and pass Debug/Release. Historical audit remains unchanged; direct-helper closure is pending S02 evidence.

## EX-04 — Other INT8 metric range paths require separate disposition

Confirmed by the root Release probe in reports/EX04-metric-probe.swift against the actual O04-frozen VectorCore module/objects (source98388fc), with compile/probe exits0 and retained artifact hashes: QuantizedKernels symmetric dotProduct512/768/1536 multiply Float scales before multiplying the integer dot; a tiny scale product can underflow before a representable final dot. Mixed and SoA Euclidean paths square Float differences and can underflow or overflow even when the final norm is representable. The same-scale integer Euclidean gate also permits approximately equal scales, although its formula requires equality. Repairing calibration/conversion/cosine cannot imply these consumers are range-safe. Root must re-derive reachability for newly meaningful subnormal calibration scales and choose a bounded Q01 extension or explicit separate follow-up before claiming numerical consumer closure. The21witnesses span all widths512/768/1536: dot underflow (referenceFloat bit patterns37/55/110 vs actual0), mixed and SoA tiny1e-30 distances returning0 and large1e20 distances returningInf, and scalar/SoA unequal-scale distances returning0 instead of positive approximately2.7e-6...4.7e-6. Exact command, module/object hashes and results are retained in reports/EX04-provenance.json and EX04-results.log. Root assigns a bounded widened-metric/gate extension to Q01; the earlier design candidate alone remains unapproved.

## EX-05 — MemoryPool fallback repeats fresh-memory binding/alignment defects

O04 implementer, reviewer and root independently read Utilities/MemoryPool.swift's last-resort withBuffer fallback: it requests only MemoryLayout<T>.alignment, multiplies count*stride unchecked and calls assumingMemoryBound on freshly allocated raw memory. O04's AlignedMemory repair does not affect that fallback. This is source-established; no forced-failure reproduction or real OOM is claimed. Assign investigation and mechanical fallback coverage to E04 alongside checked count/bucket arithmetic and the initialized-element ownership contract. Do not include this path in O04 closure.

## EX-02 / EX-03 closure disposition

Accepted with S02 (closures/S02.json): separate runtimeRED establishes Int.min lookup/error-helper traps; finalDebug/Release bounded children and fullintegerboundarytables pass. Endpoint clamping prevents lookup subtraction overflow; nonpositive error dimension metadata uses-1 without subtracting. Original source-established entries above are preserved as history.

## EX-01 / EX-04 closure disposition

Accepted with Q01 revision4 (closures/Q01.json). Three-width mutable-borrow tests establish transactional rollback on throw, finite validation/recalibration on normal return, retained-copy isolation and legacy-overflow behavior. All21root EX04 witnesses now pass alongside exact centered-dot and affine-difference cancellation controls, public/generic/mixed/SoA route matrices and bounded result-prefix guards in Debug/Release; valid controls pass ASan. Original findings above remain historical. Generic VectorProtocol/ordinary FP32 metrics, optional C alternatives and downstream/GPU behavior remain outside this closure.

## EX-06 — Test and CI child cleanup could wait indefinitely

Root milestone 04 Debug exposed an older O04 diagnostic expectation after N01 renamed its guard. Inspecting that helper found an unbounded post-deadline wait. A mechanical scan reproduced the same class in four Swift test helpers (one also read a pipe before waiting) and the Python MemoryPool watchdog cleanup. The new regression fails on all five helpers, and a mocked killed-but-unreaped child proves the Python cleanup still lacked a deadline; existing real descendant-group termination remains covered. Root replaces these waits with pre-run termination handlers and bounded execution/cleanup, uses file capture, and throws before interpreting a timeout as an expected trap or exit code. Only the older normalize diagnostic is migrated. Frozen patch and RED/GREEN evidence are EX06-root-v1* and EX06-watchdog-*.log. Independent source/Python/Debug review accepted; final Release association remains pending.

### EX-06 closure disposition

Accepted after independent review and root source-hash verification; final combined Debug and Release each pass 25 XCTest tests plus 260 Swift Testing functions in 28 suites. The five focused Python regressions pass. The mechanical scan closes the identified unbounded wait/pipe API class, not arbitrary future orchestration loops. Source and commands are associated in reports/milestone-04-verification.json.

## EX-07 — Benchmark CLI test driver executes during unittest import

Full CI discovery reproduced an import-time failure: the existing V01 process driver interpreted unittest's `discover` argument as its executable path. Root moved the unchanged driver body into main() behind a standard entry-point guard, preserving all assertions and adding an argument-count usage message. Independent review verified exact body preservation and source/evidence hashes. Discovery now passes all 28 tests, and an explicit run against the actual milestone 04 Release binary passes all CLI contracts. Accepted; source and commands are in reports/EX07-root-manifest.json. Executable-driver CI wiring is still a V03 responsibility; importing the module is not claimed as runtime CLI coverage.

## EX-05 closure disposition

Accepted with E04 (closures/E04.json). Checked count/byte/alignment contracts precede allocation; primary, direct and final Swift raw paths preserve requested alignment, establish fresh binding and retain matching allocation family. Per-instance failure controls exercise both fallbacks with actual allocation/free ledgers and reference-element lifetime checks. Debug/Release and valid-path ASan pass. Genuine allocator exhaustion remains outside the recovery promise; no real OOM reproduction is claimed.

## EX-08 — Benchmark setup mixes allocation families and hides failures

Source-established during V02 inventory: MemoryBench allocates through AlignedMemory (posix_memalign) but releases through UnsafeMutablePointer.deallocate, including both per-iteration allocation and persistent copy/fill buffers. Persistent setup silently substitutes a different Swift allocation family after failure. No supported-use corruption is reproduced. Root assigns a narrow V02 benchmark-only repair: one explicit matching allocator/releaser, no silent fallback, retained initialization/deinitialization counts, and allocation failure reported through the existing benchmark failure boundary. A local injected allocator ledger and actual valid ASan controls establish allocation-family matching and cleanup on partial setup failure; no OOM recovery promise or library allocator change.

## EX-09 — Benchmark metadata acquisition can wait indefinitely

Source-established during V02 inventory: EnvCapture waits unboundedly for runtime git/swiftc processes and captures pipes before safe bounded orchestration. These runtime values also cannot prove what compiled a moved executable. Root assigns bounded file-captured metadata collection with pre-launch handler, execution/cleanup deadlines and explicit unavailable values to V02. Test executable stubs cover normal/error/hang cases; timeout must not read blocking Process status properties. Compile-feature metadata is separately emitted from compiled code, and runtime checkout/toolchain observations are named honestly. This extends the EX-06 orchestration discipline to the tooling runtime; it is not an already observed production hang.

## V02 observed matrix limitation — strict policy remains truthful

Development-03's new benchmark matrix check exposes positive self-distance residuals in the existing Float Gram route. The separately linked bounded diagnostic retains all27pairs for N3/seed0 across512/768/1536, nine input bit-pattern hashes and160 actual linked objects. Root rederives Double and exact rational coordinate-difference references: diagonal references are exactly0, while observed maximum diagonal residuals are2^-15/2^-14/2^-14. Off-diagonal relative maxima are approximately5.45e-8/4.32e-8/8.63e-8. These are actual observations of this diagnostic, not portable backend constants; the earlier measured768case differed.

For a zero reference, the unchanged strict relative statistic divides the absolute error by its 1e-12 denominator floor, so these residuals fail 1e-6 admission. Finite statistics are still valid observations. Root preserves dense fixtures/diagonals/thresholds, requires honest failures and independently verified reporting, and adds only a post-timing test observer. Binary-grid controls remain complementary. This is qualified Float-Gram cancellation context related to NUM-27's numerical-oracle caveat, not a new claim that its KNN acceptance is closed or that MatrixDistance is repaired. Evidence/disposition: reports/V02-root-matrix-diagnostic-verification.json and reports/V02-matrix-disposition-review.md; actual canonical bytes under Sept29/V02/matrix-diagnostic-01.
