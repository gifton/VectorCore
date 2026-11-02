# Component 07: reduced precision, quantization, caches, and tuners

Reviewed revision: `8b02e75bca7055ae11170873d15c06f1538709b7` (main), 2026-09-14. No repository edits or shared builds. Findings below distinguish public opt-in APIs from built-in search routing.

## Coverage and reachability

Read all source in:

- `Sources/VectorCore/Quantization/QuantizationSchemes.swift` (165 lines).
- `Sources/VectorCore/Operations/Kernels/MixedPrecisionAutoTuner.swift` (951).
- `Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift` (190).
- `Sources/VectorCore/Operations/Kernels/NormCache.swift` (159).
- `Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift` (834).
- `Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift` (5,442).
- `Sources/VectorCore/Operations/Kernels/MixedPrecisionValidation.swift` (376).
- `Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift` (426).
- `Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift` (1,475).

Read the relevant generic/dispatch/array-wrapper paths in `BatchKernels_SoA.swift`, particularly lines 1–260 and 660–850; checked all symbols/cross-calls. Read `Vector512Optimized.hash(into:)`, documented package/API/numerical/memory boundaries, `QuantizationTests`, and targeted tests in `QuantizedKernelsTests`, `QuantizedKernelsComprehensiveTests`, `MixedPrecisionPhase3Tests`, `MixedPrecisionKernelTests`, `MixedPrecisionRangeValidationTests`, `MixedPrecisionFuzzingTests`, `KernelAutoTunerTests`, and `MixedPrecisionAutoTunerTests`.

**Routing facts:** `SoAFP16Cache512`, both public tuners, `QuantizedAutoTuner`, direct FP16 kernels and INT8 vector methods are opt-in public APIs. Source search found no cache or mixed/quantized-tuner calls from `Operations.findNearest`, `BatchOperations`, or `BatchKernels_SoA`. The latter is FP32 SoA; `KernelAutoTuner` benchmarks its array wrappers (which rebuild SoA per execution). `MixedPrecisionProvider` is internal, has no production construction/callers outside its declaration, and stores but never consults its `autoTuner`. `NormCache` and `PrecisionValidator` are internal helpers, with no production callers outside their defining source. No matching uses of the reviewed cache/tuner/precision symbols were found in the supplied `VectorIndex/Sources` or `future/VectorAccelerate/Sources` trees. These findings must not be described as defects automatically affecting every default nearest-neighbor call.

## Validation artifacts

`reduced-probes/Probe.swift` is a standalone `@testable import VectorCore` executable linked against the orchestrator's completed Debug object files. The probe itself was compiled with `swiftc -O`; no package rebuild was started. Supporting files: `objects.rsp`, `probe`, `normal.log`, `quantize-trap.log`, `asymmetric-trap.log`, and `zero-cache.log` in `/private/tmp/vectorcore-review-2026-09-14/reduced-probes/`.

Executable normal-mode results:

```text
cache hashes equal=true, input[100]=1.0, cached[100]=0.0
schemes input=[1.0, 2.0], scale=[0.003921569], offset=[-128.0], roundtrip=[1.0, 1.0]
schemes input=[-10.0, -1.0], scale=[0.03529412], offset=[127.0], roundtrip=[-9.0, -0.98823535]
schemes input=[1.5, 1.5], scale=[1e-45], offset=[-128.0], roundtrip=[3.57e-43, 3.57e-43]
adaptive strict zero=0.0, fp32=0.0022631169
adaptive candidate overflow=inf, fp32=1583918.9
diagnostics subnormal=512, underflows=0
quantized tuner first=candidatesINT8, cached=candidatesINT8, fresh=fullFP32
quantized cosine self 1e10=1.0
precision equality nan zero=true
```

Both finite-value quantization crash cases and zero-capacity cache case exited by signal 5. These are observations on this host/build, not inferred benchmark results. The concurrency race is established by conflicting unsynchronized source accesses; no TSan run was performed.

## Prioritized verified defects

### 1. P1 — Cache treats deliberately partial vector hashes as exact data identity

**Category:** correctness / cache architecture. **Location:** `Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:26` (key construction), `:97` (hit); `Sources/VectorCore/Vectors/Vector512Optimized.swift:335`–339 (hash samples only lanes 0,1,126,127).

**Trigger:** cache a zero vector, change element 100 to 1, then request `getOrCreate` for the changed candidate set. Both vectors have identical first/last eight components, so their hashes are *deterministically* equal. Cache key equality compares arrays of these hashes and never actual vector values. The second request returns the old SoA; the probe observes component 100 still zero.

**Impact:** silent stale embeddings and incorrect distances/rankings through the public cache's intended API. Sparse vectors or edits confined to the middle 496 dimensions make this ordinary, not a remote random collision. Actor isolation protects the wrong identity consistently.

**Remedy/test:** retain an immutable candidate identity/version, or compare complete candidate data on matching hashes. Merely hashing all elements reduces collision frequency but does not establish equality. Add an adversarial cache identity test that mutates each scalar component in turn while preserving vector count/order, and verify reconstructed cached vectors, plus reordered/multiple-candidate cases. Existing `MixedPrecisionPhase3Tests.testSoACacheBasic/LRU/HitRate` use ordinary generated data and miss deliberate partial-hash collisions. Confidence: executable reproduction + exact source derivation.

### 2. P2 — QuantizationSchemes calibrates ranges its own affine representation cannot cover

**Category:** numerical correctness / duplicated architecture. **Location:** `Sources/VectorCore/Quantization/QuantizationSchemes.swift:96`–104.

**Trigger:** public `computeQuantizationParams(values:[1,2],strategy:.perVector)`. Scale is 1/255 but zero point is clamped to -128; the representable decoded interval becomes [0,1], so 2 saturates to 1. Negative-only [-10,-1] decodes its low endpoint as -9. Constant 1.5 uses the least subnormal Float scale and decodes essentially zero.

**Impact:** order-one loss for simple finite inputs, far beyond ordinary INT8 rounding noise. This public parameter/calibration API is separate from `LinearQuantizationParams`, whose zero point has already been widened specifically to avoid the same one-sided-range defect. The similarly named test suite mostly exercises LinearQuantizationParams; source search found no test invoking `QuantizationSchemes.computeQuantizationParams`.

**Remedy/test:** consolidate affine calibration semantics; either include zero in the calibration interval before constraining the zero point or support a sufficiently expressive offset/bias. Define a constant-vector mapping. Parameterize round-trip endpoint/constant tests over both public calibration surfaces and positive, negative, crossing-zero, and degenerate intervals. Confidence: executable reproduction using the returned public parameters and their documented decode formula.

### 3. P2 — “Clamping” conversions trap before reaching the clamp on finite inputs

**Category:** robustness / numerical conversion. **Location:** `Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:46`, `:54`.

`LinearQuantizationParams(minValue:-1,maxValue:1).quantize(1e20)` traps converting the scaled Float to Int; the Int8 clamping initializer never runs. Public asymmetric calibration with `minValue=maxValue=1e20` similarly traps converting the zero-point Float to Int before the outer Int32 clamp. Both inputs are finite, and no documented precondition limits these methods to Int-representable intermediates. The scalar quantizer explicitly says clamping is used for safety.

**Impact:** process termination when applying calibrated quantization to a large finite outlier, or calibrating finite constant/high-offset values. NaN/Inf also need an explicit policy but are not needed to demonstrate this defect. Scalar and SIMD quantization paths have different saturation order (SIMD clamps in floating-point first).

**Remedy/test:** validate domain and scale, handle non-finites explicitly, clamp in a suitable floating-point domain before integer conversion, and define fallback/representation for offsets outside Int32. Mechanical tests should cover *all* Float-to-integer conversions in these parameter/quantization implementations: ordinary endpoints, finite outliers, extreme constants, ±infinity and NaN, scalar vs SIMD consistency. Confidence: both crash probes terminated at the standard library Float-to-Int conversion with exact over/under-Int-bound messages.

### 4. P2 — Diagnostics singleton races when recording is toggled during conversion

**Category:** concurrency / Sendable contract. **Location:** `Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:103`, `:146`; writes at `:64`, `:73`.

`recordConversion` and `recordBatchConversion` read mutable `isEnabled` before taking the lock; `enable`/`disable` write it under the lock. The outer `isRecording` check in `MixedPrecisionKernels.recordConversionsIfEnabled` locks and unlocks, then builds a buffer and calls `recordBatchConversion`, so it does not protect that later unlocked read. Conversion on one task while another toggles diagnostics produces a data race in an `@unchecked Sendable` global object. The locked statistics storage does not repair the race on the flag.

**Impact:** undefined concurrent behavior, stale enable state, and recording into a new/reset session after another task disabled/re-enabled diagnostics. Public vector conversion initializers reach this code whenever tracking is enabled.

**Remedy/test:** recheck `isEnabled` inside the same critical section as statistics mutation; use a real atomic only if an unlocked fast-path check is needed, with precise reset semantics. TSan test concurrent public conversion and enable/disable/reset operations. Confidence: direct conflicting-access proof; no TSan measurement claimed.

### 5. P2 — Quantized cosine introduces avoidable overflow despite bounded integer accumulators

**Category:** numerical correctness. **Location:** `Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:754`–761, repeated `:793`–796 and `:825`–828.

**Trigger:** `let q=Vector512INT8(repeating:1e10); q.cosineDistance(to:q)` returns 1, although the decoded vector is finite/nonzero and a vector's cosine distance to itself is zero. Each squared norm is finite (~5.12e22), but the code multiplies the two squared norms in Float before sqrt, overflowing to infinity. The finite dot divided by infinity produces zero similarity.

**Impact:** loses cosine's scale invariance and produces maximally uninformative scores for valid finite quantized data. This is an arithmetic defect after quantization, not a conversion error. The same expression is copied across 512/768/1536 implementations.

**Remedy/test:** for symmetric quantization, positive scales cancel exactly from cosine: compute directly from bounded integer dot/norm accumulators in Double or safely scaled Float. Otherwise avoid multiplying squared magnitudes before square roots. Class-wide scale-invariance tests across all widths, sign/opposite vectors, small/large scales, zero vectors, and asymmetric fallback. Confidence: public API executable reproduction and overflow derivation.

### 6. P2 — Adaptive distance ignores requested tolerance and checks only the query's range

**Category:** API correctness / precision policy. **Location:** `Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:3388`–3414.

`threshold` is never used. Any query whose maximum component magnitude is below 100 selects FP16 for both query and candidate, without inspecting candidate values. A zero query and all-70000 candidate return infinity vs finite FP32 distance 1583918.9. With query=all-1 and candidate=all-1.0001, `threshold:0` still returns zero vs FP32 0.0022631169.

**Impact:** the named adaptive API and its documented error-threshold argument provide no operative accuracy/range control. This is an explicit public call, not default Operations routing. The general numerical doc correctly disclaims blanket FP16 error guarantees; that does not make an ignored configuration argument effective.

**Remedy/test:** define absolute/relative threshold semantics, examine both operands before selecting FP16, use FP32 fallback when representability or the requested bound cannot be established, or remove the ineffective argument and advertise a heuristic-only operation. Add high-candidate/low-query, reversed operands, and same-input/different-threshold tests. Existing adaptive tests use random/moderate values and broadly tolerant comparisons. Confidence: executable reproductions.

### 7. P2 — Quantized auto-tuner cache omits a parameter that changes its answer

**Category:** configuration correctness. **Location:** `Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:1389`–1408.

`accuracyRequirement` influences strategy selection but is absent from the cache key. At operation `dot`, dimension512, candidateCount2000: first request accuracy0.02 yields candidatesINT8; second request accuracy0.2 returns cached candidatesINT8, whereas the identical second request after `clearCache` yields fullFP32.

**Impact:** public strategy selection depends on call history; workloads with different accuracy requirements pollute one another. The heuristic's direction is also counterintuitive (stricter small requirements choose quantization), but that policy issue is separate from this deterministically proved cache inconsistency. No automatic production dispatcher consumes this tuner in this checkout.

**Remedy/test:** include all decision inputs in the cache identity and explicitly define whether accuracyRequirement is max error or minimum accuracy. Parameterize tests varying exactly one decision input and comparing warm-cache vs fresh-cache results. Confidence: executable reproduction.

## Additional verified gaps and engineering opportunities

- **P3, diagnostics classification:** `MixedPrecisionDiagnostics.swift:132`–139 and `:173`–176 put the `abs(value)<1e-10` underflow branch after the broader `abs(value)<6.10352e-5` branch, making it unreachable. 512 conversions of 1e-12 report 512 subnormals and zero underflows, although FP16 conversion is zero. Use the actual converted result (and distinguish rounded-to-normal near the boundary) rather than disjoint approximate magnitude labels. The scalar and batch copies need one shared classification test. Also min/max start at zero, so positive-only data report a minimum never observed.
- **P3, cache capacity:** `SoAFP16Cache.swift:85` accepts maxSize=0 or negative; `store` line133 removes from an empty LRU list and traps. Probe confirms zero. Either precondition positive capacity in initializer or make zero disable retention. The default capacity is positive; this is explicit configuration misuse currently accepted without a stated contract.
- **Performance, source-derived:** all nine public range FP16 wrappers (`MixedPrecisionKernels.swift:4666`, `4684`, `4702`, `4736`, `4754`, `4772`, `4802`, `4820`, `4838`) map/retain the *entire* candidates array before processing the requested range. For N candidates and K independent chunks this adds O(KN) reference processing and O(N) temporary metadata per concurrent call, even for an empty/tiny range. This undermines the API's explicit parallel-chunking purpose. Pass the typed candidate array directly or map only the requested slice; verify empty/single-element range allocation/work is independent of unrelated candidate count. No timing regression measurement is claimed.
- **Performance / retention:** SoAFP16Cache512 caps entries, not bytes. A 512D set of N candidates retains roughly 1024N payload bytes (plus padding/key overhead); 100 one-million-candidate entries can retain ~102.4GB. Its comment claiming ~1KB per 100 vectors is wrong by ~100x (actually ~100KB). Provide a byte budget/admission policy and expose byte statistics. Cached-key generation still allocates/hashes O(N) metadata; LRU firstIndex/remove is O(entry count). Those complexities are derived, not speed estimates.
- **Ownership contract gap:** three public `fp16Storage` getters (`MixedPrecisionKernels.swift:344`, `387`, `430`) return a pointer that escapes `ContiguousArray.withUnsafeBufferPointer`; no scoped API/owner token accompanies it. Temporary vector expressions can leave dangling pointers immediately. These are explicitly unsafe pointer APIs, so do not conflate with a demonstrated use-after-free in a supported caller. Recommend scoped FP16-buffer access and documented owner lifetime; direct library kernels correctly use nested scoped borrows.
- **Public invariant gap:** `Vector512INT8/768INT8/1536INT8` raw storage initializers at `QuantizedKernels.swift:171`, `184`, `197` and public mutable storage permit short/oversized storage with a fixed reported dimension. `toFP32` uses fixed lane counts and unchecked buffer reads; C dot paths use one vector's actual count to read the other. Safe-looking public construction can break unsafe-kernel invariants. Validate count at construction and make storage mutation shape-preserving. This is source-verified, not a executed memory fault; coordinate with the representation reviewer because the same class may exist in FP32 types. General raw-pointer caller obligations do not clearly document the raw INT8 initializer's shape requirement.
- **Internal-only validator bug:** `PrecisionValidator.validateVectorEquality512/768/1536` return true for NaN differences because only `diff > tolerance` is rejected. Probe confirms NaN vector vs zero passes. No production call outside this file, so treat as an unreliable test/validation helper rather than a public runtime regression. Its distance validators already reject non-finites; apply the same defined policy consistently.
- **Tuner architecture:** KernelAutoTuner ignores memoryBudget/latencyTarget in both key and selection, estimates rather than measures footprint, omits hardware/revision identity/expiry from persistent profiles, and claims overrides persist while only profileCache is saved. `selectMixedPrecisionStrategy(.fp32Full)` maps full precision to a nonzero 0.0001 error budget. Its test/CI environments deliberately skip profile-based selection, so normal production cache selection is not exercised by ordinary tests. These are API/design consistency gaps; no claim that they currently reroute Operations calls.
- **Calibration cost:** MixedPrecisionAutoTuner calibrates all strategies for each exact candidate count (125 measured/warmup executions in total plus data generation), synchronously inside its singleton actor; every reduced-precision execution reconverts candidates. At very large N this is substantial first-call latency/transient storage, and benchmark cost models differ from preconverted/cached serving workloads. No timing was measured here.
- **Misleading numerical/performance claims:** MixedPrecisionKernels lines71–80 state that FP32 accumulation prevents catastrophic cancellation and preserves distance ordering; neither follows from conversion/accumulation and the docs explicitly disclaim such guarantees. Many unreferenced speedups/hardware-latency claims, and FP16VectorPool's “lock-free fast path”/“GC pressure” comments, are unsupported or contradicted by the implementation. Rewrite as measured revision/hardware-specific evidence or remove. The immutable FP16 vectors in FP16VectorPool cannot be refilled through a supported API, so it cannot actually reuse allocation for conversion of new values as advertised.

## Strengths and rejected suspicions

- FP16 vectors keep immutable storage and validate exact lengths at construction; raw bit-pattern range validation checks the exponent bits, including NaNs and both infinities. `MixedPrecisionRangeValidationTests` covers all widths and finite boundaries.
- Reviewed software FP16 conversion has explicit normal/subnormal ties-to-even handling, mantissa carry into the next exponent, signed zero, NaN and infinity behavior. No actionable new conversion-bit defect found in this read. Existing tests cover conversion/subnormal cases; an exhaustive new conversion test was not run here.
- FP16 direct dot/Euclidean/mixed cores have explicit scalar/lane remainder loops. Standard and 8-candidate SoA kernels correctly use padded groups and write only actual candidates; no new legal-input tail indexing defect found.
- INT8 squared differences are widened from Int16 to Int32 *before* multiplication; products and norm sums fit for supported 512/768/1536 dimensions. The old Int16 squared-difference overflow is fixed in both pairwise and batch implementations. Do not re-report it.
- Asymmetric and differing-scale INT8 Euclidean comparisons have dequantize-then-compute fallbacks. Approximate scale equality could erase extremely tiny differences but is not elevated above the concrete defects above.
- NormCache contains owned Float arrays, no retained raw pointer keys; no dangling/stale-address cache issue found there. Its caller pairing is unchecked except count, but it has no current production call sites.
- SoAFP16Cache's stored value is immutable COW storage, so actor escape does not itself create shared mutable data races. Its bug is key identity, not value ownership or actor isolation.
- QuantizedAutoTuner protects dictionary access with its lock; its verified failure is cache identity rather than a data race. FP16VectorPool also locks acquire/release/count/drain correctly.
- Output pointer capacity checks are frequently Debug-only; where the public API explicitly requires adequate output capacity, undersized buffers violate the documented unsafe contract. They are not treated as newly proved legal-input corruption.
- FP16 range representability is not an arbitrary score-error guarantee; normal lossy rounding/rank changes near ties are documented limitations. Findings above instead identify deterministic policy omissions, avoidable arithmetic failure, stale cache values, or runtime traps.

## Limits

No full-suite build/test, sustained benchmark, TSan/ASan run, or downstream integration build was performed by this component reviewer; the orchestrator owns those. Standalone executable uses current Debug library objects and optimized probe code. Performance findings are complexity/allocation derivations. Full generic/custom OptimizedVector conformance behavior, serializer validation, and all FP32 kernels lie with adjacent component reviewers. This review found multiple actionable defects, with deterministic SoA cache corruption the clearest priority.
