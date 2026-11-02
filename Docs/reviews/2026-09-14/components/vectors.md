# Component 01 — vector representations and protocols

Revision: `8b02e75bca7055ae11170873d15c06f1538709b7`. Review only; no repository edits or SwiftPM builds.

## Reviewed surface

Read `Vectors/Vector.swift`, `DynamicVector.swift`, and all four `Vector{384,512,768,1536}Optimized.swift` implementations; `Core/Dimension.swift`, `Core/Operators.swift`; `Factory/VectorTypeFactory.swift`; `Initializers/ConvenienceInitializers.swift`; `Protocols/{CoreProtocols,VectorProtocol,OptimizedVector,VectorFactory,VectorProducer,VectorOperationsProtocol,SIMDStorage}.swift`; `Types/VectorData.swift`; `Operations/VectorCoreOperations.swift`. The optimized variants were additionally normalized by dimension/lane count and diffed, exposing the remaining 384 binding discrepancy.

Followed storage into `Storage/DimensionStorage.swift`, normalization into `Operations/VectorNormalization.swift`, arithmetic into `VectorMath.swift`, and storage forwarding into `Integration/IndexableVector.swift`. Read associated construction, dynamic-vector, arithmetic, arbitrary-dimension, factory, producer, optimized-vector tests and relevant numerical/pointer regression sections. Searched source/tests and sibling VectorIndex/VectorAccelerate sources for affected usage. Read boundary/API/numerical docs.

## Prioritized findings

### V1 — P1 correctness / memory safety: publicly mutable storage defeats the vector shape invariant

**Locations:** `Sources/VectorCore/Vectors/DynamicVector.swift:17`, `:20`, `:24`; `Sources/VectorCore/Vectors/Vector512Optimized.swift:29`, `:150–173`; same storage property and fixed-count pointer construction in 384/768/1536 implementations (384 property at :32). Architectural source: `Protocols/VectorProtocol.swift:38` requires unrestricted get/set storage.

**Trigger:** After valid construction, assign differently sized storage or call `optimized.storage.removeAll()` / `removeLast()`. These are ordinary public Swift operations, with no unsafe annotation or documented storage-shape precondition on the property.

**Mechanism and impact:** DynamicVector caches immutable `dimension` while `toArray()` and buffers return current storage length. Equality compares only the cached number of elements whereas hashing consumes all actual storage. Assigning `[1,2,3]` storage to a two-dimensional `[1,2]` vector makes it equal to the original while producing a different hash. Short replacement also defeats bounds/shape validation: helpers size from scalarCount and read the shorter backing buffer. `resized` directly initializes `min(dimension,newDimension)` elements from the source pointer.

Optimized vectors export a fixed-length Float buffer even after the ContiguousArray is shortened. `toArray()` also unconditionally copies that fixed count, so safe high-level methods can read beyond allocation. Mutable-buffer methods can write beyond allocation. Initializer preconditions do not protect subsequent mutations. `NormalizationHint.storage` forwards this problem (`Integration/IndexableVector.swift:167`).

**Verified evidence:** `vectors-probe.swift` linked against existing Debug objects prints `mutated dynamic shape 2 3 3 equal true hashEqual false`. After replacing Vector512Optimized storage with one SIMD4, its Float buffer advertises `512` elements despite the actual four scalars. No intentional out-of-bounds write was executed; the subsequent overrun is directly derived from the fixed-count view and `toArray()` implementation.

**Remedy/test:** Make shape preservation an enforced public invariant: DynamicVector can derive scalarCount from storage; optimized storage needs a validated setter/fixed-length wrapper, or remove public settable storage from the base protocol and expose scoped element mutation. Validate every path, including wrapper forwarding and direct collection mutation. Add a shared vector-shape test covering equal/short/long storage replacement, buffer lengths, collection count, serialization, equality/hash consistency, and arithmetic. Avoid only patching the DynamicVector initializer.

### V2 — P2 correctness: reciprocal-based scalar division poisons finite subnormal inputs, including checked normalization

**Locations:** `Protocols/VectorProtocol.swift:264–266`, `:402–407`; `Protocols/VectorFactory.swift:87–89`; `Vectors/Vector384Optimized.swift:284–287` and identical division implementation in the three other optimized types. Related direct fast normalization: `Operations/VectorNormalization.swift:28–30`, `:49–51`.

**Trigger:** Let `s = Float.leastNonzeroMagnitude`; divide vector `[s,0]` by `s`, or call DynamicVector's checked `normalized()` on that vector. Expected mathematical quotient is `[1,0]`, exactly representable in Float.

**Mechanism:** `/` computes `lhs * (1/rhs)`. The intermediate reciprocal overflows although the final quotient is representable; outputs become `[Inf,NaN]`. The generic magnitude correctly returns a positive tiny value, so checked normalization returns `.success` with poisoned output. This applies to scalar division on every vector family, even where concrete optimized normalization has its own guards.

**Verified evidence:** Probe prints `tiny normalization [inf, nan]` and `optimized division tiny [inf, nan]`. A `genericNormalize<V: VectorProtocol>` helper also returns `[inf,nan]` for Vector512Optimized, while the direct concrete method returns `.failure`. `normalized` is extension-only rather than a protocol requirement, so generic dispatch bypasses the optimized implementation and its earlier fixes.

**Remedy/test:** Use direct component division when the reciprocal is non-finite; more generally normalize via stable scaled components without requiring a representable reciprocal or full norm. Define and enforce the same checked-normalization contract for direct and generic calls; make intended customization points protocol requirements. Extend the existing subnormal regression class across DynamicVector, Vector<D>, every optimized dimension, direct calls, generic calls, `/`, `inPlaceDivide`, and all normalization variants. Existing `NumericalStabilityTests.testNormalizeSubnormalVectorNoNaN` only exercises concrete optimized normalization; that explains why it misses this surviving instance.

### V3 — P2 correctness: in-place binary methods omit runtime dimension checks

**Locations:** `Protocols/VectorProtocol.swift:181–189`, `:194–202`, `:230–238`; generic `+=`/`-=` at :241–246.

**Trigger:** `DynamicVector([1,2]).inPlaceAdd(DynamicVector([4,5,6]))`, or use a generic `<V: VectorProtocol>` helper applying `+=` to the same operands. Reverse length ordering to expose an out-of-bounds access.

**Mechanism and impact:** The loops always use the receiver's count. Longer RHS values silently truncate; shorter RHS hits a Debug pointer bounds trap and can read past the buffer when checks are optimized away. Unlike corresponding `+`, `-`, and `.*`, these methods contain no dimension precondition and document no unchecked contract. The concrete VectorFactory `+=` overload is checked via combination, but generic protocol dispatch resolves the unchecked default instead.

**Verified evidence:** Probe prints `inplace-long [5.0, 7.0]` and `generic += long [5.0, 7.0]`; subprocess mode `inplace-short` exits `-5` (SIGTRAP). Release out-of-bounds behavior was not executed.

**Remedy/test:** Add shape preconditions before mutation to all three methods and any corresponding checked variants. Mechanical shared tests should enumerate binary mutators and direct/generic operator routes with equal dimensions, empty vectors, shorter RHS, and longer RHS. Existing arithmetic tests exercise fixed dimensions and DynamicVector min/lerp mismatch, not these mutators.

### V4 — P2 code quality / memory binding: 384-dimensional vector missed the scoped rebinding repair

**Locations:** `Vectors/Vector384Optimized.swift:69–72`, `:143–145`, `:156–159`, `:168–171`.

**Trigger:** Normal array construction, `toArray()`, or either Float buffer accessor on Vector384Optimized.

**Mechanism:** These four sites use permanent `bindMemory` between Float and SIMD4<Float>, then subsequently access/deallocate the memory through its original typed owner. The equivalent 512/768/1536 implementations explicitly use scoped `withMemoryRebound` to restore the original binding, with comments naming the TBAA issue. The 384 implementation is the surviving same-class instance. Constructors also need the same alignment assumptions audited across all four variants.

**Evidence/confidence:** Source-verified discrepancy across all four files; no optimizer miscompile was reproduced and none is claimed. The existing `OptimizedVector512Suite.testBufferAccess_RoundTripIsElementwiseExact` explicitly references the prior repair but covers 512 only.

**Remedy/test:** Apply the scoped-lifetime repair consistently and add a mechanical check preventing unscoped typed-owner rebinding in all vector variants, plus shared construction/read/write/COW round trips under Release. Behavioral tests alone cannot prove binding-law compliance.

### V5 — P2 performance: sampled vector hashes collapse structured/sparse keys into one bucket

**Locations:** `Vectors/Vector512Optimized.swift:334–339`; same hash implementation at :334–339 in 384/768/1536.

**Trigger:** Many distinct vectors share their first eight and last eight scalar elements, varying only their interior. Zero-padded feature vectors and sparse vectors are ordinary examples.

**Mechanism and scale:** Only four SIMD4 values are combined. Every such key gets the same full hash, regardless of Hasher randomization. Hash-table insertions/lookups then require linear collision scans, with vector equality work per candidate; constructing a Set can cost O(n²*d) when differences occur late in the interior. This is a performance defect, not a violation of the equal-values-have-equal-hash rule.

**Evidence:** Standalone probe makes two otherwise equal Vector512Optimized values differ at scalar 128 and prints `interior distinct true hashEqual true`. Source proves deterministic collision for all interior-only variation. No timing/speedup claim is made and no affected vector-keyed downstream production dictionary was found in the searched siblings. Existing optimized equality/hash tests deliberately mutate an interior scalar but use only three values, so verify Set correctness without exposing scale.

**Remedy/test:** Hash all equality-significant components (as generic Vector and DynamicVector do), or document/use explicit identity keys where appropriate. Test a large family of interior-varying vectors; a small statistical collision bound can detect this omission, with a coordinated Release benchmark for insertion scaling.

### V6 — P2 validation: throwing batch factory traps before it can reject zero dimensions

**Location:** `Factory/VectorTypeFactory.swift:249–259`.

**Trigger:** `try VectorTypeFactory.batch(dimension: 0, from: [])` (nonempty arrays also fail).

**Mechanism:** First validation is `% dimension`, so zero divides before a VectorError can be thrown. Negative dimensions that divide a nonempty count produce a negative capacity later. There is no public precondition excluding zero; zero-dimensional DynamicVector itself is explicitly supported, which makes this inconsistent boundary particularly easy to reach.

**Verified evidence:** Probe subprocess `batch-zero` exits `-5`, stderr `Division by zero in remainder operation`.

**Remedy/test:** Require a strictly positive batch row dimension before remainder/division. Test zero and negative dimensions with empty/nonempty arrays in both factory test suites; retain positive-dimension empty-batch behavior. `testBatch_NonMultipleCountThrows` currently only covers positive four-dimensional rows.

### V7 — P2 correctness: approximate equality accepts NaN as matching arbitrary data

**Location:** `Protocols/VectorProtocol.swift:550–560`.

**Trigger:** `DynamicVector([Float.nan]).isApproximatelyEqual(to: DynamicVector([42]))`.

**Mechanism:** `abs(NaN - 42) > tolerance` is false, so that component never rejects equality. A NaN tolerance similarly accepts all same-sized vectors. This can cause numerical verification, deduplication, or convergence checks to accept invalid results.

**Verified evidence:** Probe prints `nanApprox true`. No internal caller was found, but this is a public utility available on every vector.

**Remedy/test:** Explicitly define non-finite and tolerance semantics; at minimum reject NaN inputs and NaN/negative tolerance, and choose whether identical signed infinities compare equal. Test NaN in either operand, same/opposite infinities, signed zeros, finite near/far values, and invalid tolerances across a generic conformance helper.

## Architecture / performance opportunities (not additional verified defects)

- Extension-only arithmetic/math methods mean generic `VectorProtocol` code statically selects baseline operations rather than specialized implementations. V2/V3 show actual behavioral consequences; performance also needs measurement. `VectorProtocol` currently bundles Collection/Hashable/Codable/raw storage even for consumers needing only a read-only numeric view. Separate a narrow view from mutation/construction and explicitly declare override points.
- `StaticDimension.Storage` is advertised as selecting optimized storage, but `Vector<D>.Storage` always aliases `DimensionStorage<D,Float>`. Extension availability is then gated on `D.Storage: VectorStorageOperations`, even though that associated storage is not used. This is confusing for custom dimensions; remove the unused customization point or implement it deliberately.
- `VectorTypeFactory` calls its output optimal/optimized but creates generic heap-backed vectors for supported dimensions, not the specialized types. Existing tests explicitly pin this behavior; do not change it silently because routing and source compatibility can change. Document concrete output and benchmark intended workloads before choosing a strategy.
- `VectorCoreOperations` frequently converts both inputs to arrays, allocates provider output, and constructs another owned vector. Dynamic min/max/clamp allocate an initial zero result which is immediately replaced by provider output. These are source-visible allocation opportunities, not measured regressions.
- Generic vector convenience factories (`linspace`, `range`, `generate`, `geometric`, `sparse`) repeatedly mutate storage subscripts. The memory agent owns the enum/COW copy-cost diagnosis; include these call sites in the impact/test matrix rather than reporting the same root cause twice.
- Dimension and optimized-vector doc comments contain performance and robustness claims without named measurement evidence. `magnitudeSquared` claiming sqrt is required to prevent overflow is misleading: if the exact square is not representable, squaring the stable magnitude still overflows. Fixed absolute timing claims should link hardware/configuration/benchmark results or be removed.

## Strengths

- Fixed-vector array constructors check count before storage construction; dedicated SIMD-storage constructors enforce exact lane counts at entry.
- Scoped Float views are implemented consistently across 512/768/1536, and their underlying ContiguousArray supplies normal Swift COW for element mutation.
- The generic norm scales by direct division and detects NaN/Inf separately, avoiding squared-component overflow when the true norm remains representable.
- Dynamic resize, basic shape changes, collection access, factories, and nominal arithmetic have targeted tests. Separate checked convenience initializers preserve ordinary nonthrowing APIs while exposing useful VectorError paths.
- VectorProducer's default single-text implementation propagates errors and rejects empty producer results; those behaviors have direct tests. No additional actionable defect was found in VectorProducer, VectorData, operator declarations, or the small SIMDStorage helpers within their stated contracts.

## Verification and limitations

`vectors-probe.swift` was compiled using `build_probe.py` and existing Debug object files after explicit compiler-cache escalation approval. No package build or production changes. Probe output and subprocess exit codes are above. SIMD hash collision and shape-view mismatch were observed; hash performance and out-of-bounds writes were deliberately not benchmarked/executed. Swift memory rebinding defect is source-derived, not a reproduced optimizer failure. Full-suite validation remains orchestrator-owned.

Rejected/qualified suspicions: exact Float equality is appropriate for Hashable (epsilon would not be transitive); NaN's non-reflexivity follows Float behavior and is distinct from V7. Ordinary copied ContiguousArray vectors preserve value semantics. Invalid custom StaticDimension implementations that mutate their own static size were not treated as library defects. Documented unchecked matrix/normalization limits were not reclassified as newly discovered bugs. No arbitrary-dimension SIMD-tail defect was proven in this assigned component; existing ArbitraryDimensionTests only use 384, so they do not establish odd-dimension behavior.
