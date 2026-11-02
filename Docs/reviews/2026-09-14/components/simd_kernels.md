# Component 04: scalar, SIMD, platform, and C kernels

Revision reviewed: `8b02e75bca7055ae11170873d15c06f1538709b7`. Review only; no shared repository modifications or package builds.

## Reviewed surface

Read all eight files in `Sources/VectorCore/Platform/`: `AccelerateSIMDProvider.swift`, `ArraySIMDProvider.swift`, `PlatformConfiguration.swift`, `SIMDExtensions.swift`, `SIMDOperations.swift`, `SIMDProvider.swift`, `SwiftDoubleSIMDProvider.swift`, `SwiftFloatSIMDProvider.swift`; `Interop/CKernels.swift`, `Interop/FeatureFlags.swift`; all C sources/header except the assigned-away LAPACK shim (`common/vc_dispatch.c`, `common/vc_features.c`, `common/vc_scalar.c`, `arm64/vc_arm64.c`, `x86_64/vc_x86.c`, `include/VectorCoreC.h`); `Operations/Kernels/{Dot,Euclidean,Cosine,Manhattan,Normalize}Kernels.swift`, `EdgeCaseHandler.swift`; `Operations/{DistanceMetrics,OptimizedDistanceMetrics,NaNInfinityHandling,VectorNormalization,VectorMath}.swift`; `Optimization/OptimizationAttributes.swift`.

Read package-boundary/API/numerical docs and traced public routing and relevant tests: `SIMDProviderBoundsTests`, `AccelerateArraySIMDProviderParityTests`, `PointerSeamRegressionTests`, normalization portions of `NumericalStabilityTests`, `VectorNormalizationTests`, `VectorDistanceMetricsTests`, `CKernelsSmokeTests`. Searched all source/tests for helper callers and C gates. Inspected downstream `VectorAccelerate/Operations/StableNormalization.swift` because it explicitly describes divergence from Core normalization.

## Verified / high-confidence candidate findings

### 1. P2 correctness: pointer normalization reconstructs the wrong norm after clamping the scale

- **Location:** `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:453`, `:483` (public `normalizeUnchecked`).
- **Trigger:** aligned, finite, nonzero buffer filled with `Float(1e-38)`, dimensions 1, 4, 5, or 512. These meet the documented magnitude > 0 precondition; reciprocal of the true norm is representable.
- **Mechanism:** scaling uses `den = max(maxAbs, leastNormalMagnitude)`, but norm reconstruction uses `maxAbs * sqrt(sumSquares)`. When the clamp is active, this is `trueNorm * maxAbs / den`; output has norm `den / maxAbs` rather than 1.
- **Evidence:** standalone Swift compilation of the exact public pointer method (extracted unchanged into a minimal enum) plus repository SIMD extension produced norms **1.17549455, 1.17549443, 1.17549448, 1.17549466**, respectively. Sources and output: `/private/tmp/vectorcore-review-2026-09-14/simd-probe/{NormalizePointer.swift,main.swift,results.txt}`. Thus this is a 17.5% normalization error, not reduction rounding.
- **Related high-end failure:** for finite input `[Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, ...]`, reconstructed norm overflows, `1/Inf == 0` passes `invMag.isFinite`, and output is all zero. Source-derived; not separately run in this probe. Typed checked normalization helpers at lines 276–309 likewise accept positive infinite magnitudes and return success after zeroing finite components.
- **Downstream evidence:** `../future/VectorAccelerate/Sources/VectorAccelerate/Operations/StableNormalization.swift:56–62` already explicitly names both Core defects and deliberately computes correct normalized output instead. This creates an actual CPU/GPU normalization policy divergence.
- **Coverage gap:** `PointerSeamRegressionTests.subnormalNoPoison` and `NumericalStabilityTests.testNormalizeUncheckedPointerSubnormalNoPoison` assert only that results are finite. They do not check unit norm or unchanged output, even though comments imply a stronger contract.
- **Remedy/test:** normalize in the scaled domain, e.g. `(x * scale) / sqrt(sumSquares)`, with an explicit degenerate-input policy. Test all optimized typed and pointer variants over normal/subnormal/large finite values, tails 1...17 and production dimensions, checking direction and unit norm in Double. Keep the intentionally unchanged deep-subnormal contract explicit if retained.
- **Confidence:** high, reproduced exact arithmetic and public method.

### 2. P2 unsafe API contract: optional pointer alignment is actually mandatory

- **Location:** `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:419`, `:436` and subsequent SIMD loads/stores.
- **Trigger:** valid Float-aligned storage (16-aligned allocation advanced by one Float), count 4, all values 1. Public parameter documentation says “16-byte aligned preferred,” allowing this input.
- **Mechanism:** `UnsafeMutableRawPointer.load(as: SIMD4<Float>.self)` requires SIMD4 alignment. A Float pointer guarantees only Float alignment. All complete SIMD groups take this path.
- **Evidence:** compiled exact pointer method, invoked as `simd-probe/probe misaligned`, exits with **-5**, `Fatal error: load from misaligned raw pointer`. No out-of-bounds allocation was involved. Optimized builds remove diagnostics but do not remove the language precondition.
- **Coverage gap:** `PointerSeamRegressionTests` explicitly allocates aligned copies and says callers must supply aligned memory; the public source contract is weaker. No unaligned seam test found.
- **Remedy/test:** either implement unaligned loads/stores (or scalar fallback), or explicitly require alignment and validate it at the boundary. Exercise offsets 0, 4, 8, 12 bytes and arbitrary tails. Supporting Float-aligned buffers best matches the existing public wording.
- **Confidence:** high, deterministic crash reproduction.

### 3. P2 correctness/bounds: `unrolledLoop` does not honor its public stride parameter

- **Location:** `Sources/VectorCore/Optimization/OptimizationAttributes.swift:91–96`; related missing width validation at `:116–118`.
- **Trigger/evidence:** exact repository implementation compiled in probe, `count: 10`: `by: 1` visits `[0,1,1,2,2,3,...,9,9,10]`; `by: 5` visits `[0,1,2,3,5,6,7,8]`; `by: 8` visits `[0,1,2,3,8,9]`.
- **Mechanism:** body always runs at least twice, at most four times, while index advances by arbitrary stride. `by: 1` duplicates work and calls one index beyond count; >4 silently omits work. Zero stride never advances. `vectorizedLoop(vectorWidth: 0)` has the same nontermination class.
- **Impact:** public callback can read/write out of bounds or silently leave work unperformed. No repository/downstream callers or tests were found, so this is an exposed helper defect, not a demonstrated current internal algorithm failure.
- **Remedy/test:** constrain/document permitted widths, or correctly iterate all requested lanes with specialized cases for common widths. Require positive width and nonnegative count. A mechanical property test should require emitted indices (or flattened ranges) equal `0..<count` exactly across widths 1...17 and boundary counts; explicitly cover rejection of zero/negative widths.
- **Confidence:** high, reproduced; zero-width hang is direct control-flow proof and was not run.

### 4. P2 numerical correctness: reciprocal scalar division manufactures Inf/NaN from finite quotients

- **Locations:** `Sources/VectorCore/Platform/ArraySIMDProvider.swift:181–182`, `:639–640`; `SwiftFloatSIMDProvider.swift:358–365`; `SIMDProvider.swift:210–217` (inherited by SwiftDoubleSIMDProvider).
- **Trigger/evidence:** `DefaultArraySIMDProvider().divide([Float(1e-40), 0], by: Float(1e-40))` gives `[Inf, NaN]`; `SwiftDoubleSIMDProvider.divideByScalar([Double(1e-320), 0], scalar: Double(1e-320), ...)` gives `[Inf, NaN]`. Both expected elementwise quotients are `[1, 0]`. Exact repository source compiled in probe.
- **Mechanism:** `1/scalar` overflows before multiplication even though numerator/scalar is representable. The public SIMD protocol explicitly specifies `result[i] = a[i] / scalar`; no finite reciprocal restriction is documented. This is not ordinary floating-point tolerance divergence.
- **Reachability:** both public array providers, public static Swift providers, and default `Operations.simdProvider` compatibility wrapper. Low-level Accelerate providers use `vDSP_vsdiv`/`vDSP_vsdivD`, but numerical output of those functions for this case was not probed; do not assert cross-platform parity without running it.
- **Remedy/test:** use direct division when the reciprocal is non-finite or otherwise numerically unsuitable; consider direct division as the baseline and measure a safe fast path. Add one conformance matrix spanning Float/Double and all providers, zeros, signed zeros, subnormal divisors, normal extremes, NaN/Inf and dimensions with tails. Existing bounds sweep divides only by 2; Accelerate parity suite does not test scalar division.
- **Confidence:** high, reproduced.

### 5. P2 public option semantics: `safeLog` bypasses NaN/positive-Infinity handling

- **Location:** `Sources/VectorCore/Operations/NaNInfinityHandling.swift:383–398`.
- **Trigger:** `Vector<Dim4>(repeating: .nan).safeLog(options: .strict)` or positive Infinity returns non-finite values without throwing. `.replaceAll` also leaves them; `.propagateNaN` does not propagate an existing NaN to the entire result.
- **Mechanism:** option handling is only in `value <= 0`; both NaN and positive Infinity take the `Foundation.log(value)` branch and return immediately without `handleNonFinite`. `NonFiniteHandling.throwOnNonFinite` is explicitly documented at line 30 as throwing on non-finite values. Compare `safeDivide`, which runs its output through `handleNonFinite`.
- **Evidence/confidence:** high source-derived result, not run against the complete library. No test references to `safeLog` found. Method-specific docs describe options as handling “non-positive values”; this is a possible intended narrower contract, but reusing `NonFiniteHandling.strict` while ignoring NaN/+Inf is misleading and deserves correction or explicit distinction.
- **Remedy/test:** apply the selected non-finite policy to input/result and separately handle the logarithm domain. Cover each options preset for negative finite, zero, NaN, +Inf, -Inf, and mixed vectors; verify `.strict` cannot return non-finite output.

### 6. P2 performance/code quality: nearest-neighbor convenience API fully sorts despite claiming a heap

- **Location:** `Sources/VectorCore/Operations/VectorMath.swift:386–388`, `:408–412`.
- **Trigger:** `VectorMath.nearestNeighbors` with large n and small k (including k=0).
- **Mechanism:** materializes n distances and calls `sorted` on all n before taking prefix(k). Selection is O(n log n), contrary to documented O(n log k) partial heap. k=0 still runs every metric and sorts. Negative k traps at prefix without a documented constraint. The comparator is ordinary Float `<`, so it also lacks the package Top-K NaN order; docs restrict that contract to actual Top-K APIs, so this is not claimed as a package-wide correctness breach.
- **Impact:** avoidable sorting/allocation work in a public search convenience API; source-derived complexity, no timing claim. Relative significance depends on metric cost and n/k.
- **Remedy/test:** use the existing TopKSelection contract, ideally maintain bounded selection while scanning. Define nonpositive-k behavior. Update the docs if full sorting is intentional. Test neighbor membership/tie behavior versus a small reference and coordinate Release performance measurement for large n/small k.
- **Confidence:** high for complexity mismatch; no measured regression.

## Smaller actionable issue

- **P3 factory correctness:** `OptimizedDistanceMetrics.create(named:)` (`OptimizedDistanceMetrics.swift:402`) lowercases every input, while enum `MetricType.dotProduct` has raw value `dotProduct`. Therefore every spelling of dot-product name maps to nil, despite `.create(.dotProduct)` working. One-step derivation; add a case-insensitive round-trip for every `MetricType.allCases`.

## Architecture and performance observations (not verified speed regressions)

- Three distinct numerical surfaces coexist: static pointer providers (Accelerate on Apple), task-local array providers (Swift default), and concrete optimized kernels. This is documented in API_Overview_Map. Do not assume replacing `Operations.simdProvider` redirects optimized distance overloads or all vector arithmetic. A shared cross-surface conformance matrix would catch bugs 1/4 better than provider self-parity.
- `VectorMath` large L1/Manhattan paths create several full arrays (`toArray`, subtract, abs) despite already holding pointers; `variance` creates array/differences/squares and reduces serially. O(d) temporary allocation is source-visible. Use pointer reductions or fused loops if Release allocation profiles justify changing it. Comments claiming single allocation/~10x speedups are not established by adjacent code or named tests.
- SwiftDouble reductions horizontally sum each SIMD chunk into a scalar dependency; SwiftFloat accumulates SIMD lanes across chunks. This is a potential throughput difference, not a measured issue.
- `PlatformConfiguration` assumes x86_64 implies optimal AVX-512 width and Apple ARM64 implies a 128KB L2/AMX, with no runtime detection. It is an internal heuristic layer; no illegal-instruction call is driven directly by these properties. Avoid interpreting these as hardware facts or tuning evidence.
- C FP32 wrappers have no production callers found; optional INT8 paths do. `VC_USE_C_KERNELS` is disabled by default and absent from manifest defines. x86 FP32 functions intentionally delegate to scalar; NEON FP32 and INT8 have bounded fixed chunks/tails. C header contains explicitly marked proposed unimplemented symbols, which is an API hygiene concern, not a current Swift link failure. Header range comment says output `[start..end)` but all implementations write compact `[0..end-start)` and the Swift wrapper documents that compact convention.
- C INT8 accumulator has signed-overflow exposure for enormous lane counts (e.g. 131072 copies of -128), whereas Swift fallback wraps. Current quantized call sites are fixed small dimensions, so no active production overflow finding is made. If generic C lanes become a supported path, specify a representable-sum bound or use a wider/defined accumulator and add maximum-magnitude tests before opening the gate.
- Many optimization helpers are identity functions or normal loads/stores; `prefetchRead` can be optimized away, `prefetchWrite` is a self-assignment, `likely`/`unlikely` do not emit branch metadata. Do not credit measured tuning improvements to their names. They have no identified internal users in this scope.

## Strengths

- The previous SIMD maxima/minima off-by-one class now uses `i + width <= count`; the 1...64 bounds sweep covers every remainder, including reductions that start at index 1.
- Fixed optimized dot/L2/L1 kernels use dimensions whose lane counts are all multiples of 16; no tail is required by their actual entry points. Generic providers and Manhattan distance have explicit tails. No additional bounds defect was found in those loops.
- Accelerate subtraction/division operand order is correct in source; array providers avoid zero-filling outputs that are fully overwritten and explicitly short-circuit empty output allocation.
- C x86 INT8 sign extension plus `madd_epi16` avoids unsigned-signed confusion; NEON vmull/vpadal widens before pairwise accumulation. Architecture dispatch separates baseline code from AVX2 codegen; AVX2 calls are runtime-gated.
- Numerical docs correctly separate computed metric values from Top-K ordering, document GEMM cancellation and metric-specific non-finite behavior, and avoid blanket numerical-error guarantees.

## Rejected suspicions / documented limitations

- Different zero-vector and NaN behavior between generic/optimized cosine is documented as API-specific. It is not a new package-wide propagation violation.
- Direct sum-of-squares distance/magnitude overflow, GEMM cancellation, and pre-normalized input obligations are not automatically new defects under the documented numerical boundaries. The normalization finding above is narrower: an incorrect scaling reconstruction returns a finite but wrong unit norm for valid input.
- Pointer counts/dimensions and range products are unsafe caller obligations; negative or astronomically large counts alone are not used to allege new memory safety bugs. Alignment finding differs because the API explicitly labels that constraint optional.
- `EdgeCaseHandler` graph/SoA arithmetic and epsilon shortcuts appear internal and mostly helper/test-only; no additional publicly reachable failure was established. Its safe batch Euclidean extension consistently returns squared distances even though the name is loose.
- Deep-subnormal typed magnitude currently returns zero rather than its representable true norm; tests explicitly permit this finite-only behavior. This should be revisited as a numerical contract, but is distinguished from the reproduced pointer scaling defect.

## Validation limits

All package Debug/Release tests are owned by the orchestrator. This component ran only standalone Swift source probes in a unique temp directory. The pointer normalizer was extracted unchanged because its original file also depends on concrete vector types; all provider/optimization files compiled directly from the repository. No C architecture execution or throughput benchmark was run. SafeLog/factory/complexity and huge-norm normalization cases are source-derived and clearly marked. Full-library integration confirmation can use the reproductions above.
