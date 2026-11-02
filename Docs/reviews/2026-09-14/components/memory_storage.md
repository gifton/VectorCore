# Component 02: memory, storage, alignment and SoA review

Revision reviewed: `8b02e75bca7055ae11170873d15c06f1538709b7`.

## Outcome

Five actionable public-path findings: compact-tier copies alias, generic compact elements leak, unique dynamic-vector construction performs quadratic memory traffic, large triangular-matrix writes clone the complete matrix, and public generic `alignedCopy` changes initialization semantics with alignment. A sixth group identifies verified/deduced dormant internal defects that should be fixed or removed before exposing new routes. Page-aligned/SoA ownership limitations are already documented; I found no additional actionable layout defect for the four built-in conformers under their stated contracts.

## Coverage

Read all 18 assigned files, 4,557 source lines:

- `Sources/VectorCore/Storage/AlignedDynamicArrayStorage.swift`
- `Sources/VectorCore/Storage/AlignedMemory.swift`
- `Sources/VectorCore/Storage/AlignedValueStorage.swift`
- `Sources/VectorCore/Storage/AlignmentStrategy.swift`
- `Sources/VectorCore/Storage/ArrayStorage.swift`
- `Sources/VectorCore/Storage/COWDynamicStorage.swift`
- `Sources/VectorCore/Storage/DimensionSpecificStorage.swift` (all 13 fixed-size wrappers)
- `Sources/VectorCore/Storage/DimensionStorage.swift`
- `Sources/VectorCore/Storage/HybridStorage.swift`
- `Sources/VectorCore/Storage/SoA.swift`
- `Sources/VectorCore/Storage/SoALayout.swift`
- `Sources/VectorCore/Storage/UnifiedStorage.swift`
- `Sources/VectorCore/Storage/UnifiedVectorBuffer.swift`
- `Sources/VectorCore/Storage/VectorStorage.swift`
- `Sources/VectorCore/Utilities/MemoryUtilities.swift`
- `Sources/VectorCore/Utilities/SIMDMemoryUtilities.swift`
- `Sources/VectorCore/Utilities/TieredStorage.swift`
- `Sources/VectorCore/Interop/PointerBridge.swift`

Additional complete caller read: `Utilities/TriangularMatrix.swift`; relevant construction/mutation/buffer sections of `Vectors/DynamicVector.swift` and `Vectors/Vector.swift`. Searched all package and sibling source call sites to determine reachability. Public `DynamicVector` uses `HybridStorage`, **not** `COWDynamicStorage` despite the latter's documentation. `TieredStorage` has one production owning type: public `TriangularMatrix<T>`. Internal `StorageSlice`, SIMD tuple utilities, and the raw generic sendable wrappers have no production callers in the searched checkout.

Read docs: Package_Boundaries, API_Overview_Map, Numerical_Behavior, Memory_Alignment, SoA_Layout_Contract. Read tests: StorageAlignmentTests, DynamicVectorTests, SoAPageAlignTests, SoALayoutContractTests, UnifiedVectorBufferTests. Searched the entire test tree for COW/TieredStorage/HybridStorage/TriangularMatrix/lifetime coverage: no dedicated tests cover the findings below. SoA batch-kernel test references inspected for storage-boundary use; arithmetic internals belong to the other component.

## Verified public-path findings

### MS-1 — P1 correctness: copying a compact triangular matrix does not isolate mutation

**Location:** `Sources/VectorCore/Utilities/TieredStorage.swift:175`–183 and `:450`–451. Public entry at `Utilities/TriangularMatrix.swift:94`–95 and `:157`–158.

**Trigger:**

```swift
let a = TriangularMatrix<Float>(size: 4, defaultValue: 1)
var b = a
b[0, 1] = 99
// a[0, 1] is now 99, not 1.
```

The compact tier stores a reference to `CompactBuffer`, which contains one `ManagedBuffer`. Copying the enclosing struct shares the `CompactBuffer` object without copying its `buffer` property. The uniqueness test examines that nested `ManagedBuffer` property, whose reference count remains one, so it incorrectly authorizes mutation of shared state. The same defect affects the internal mutable buffer access, `setAt`, and iterator snapshots retaining a matrix copy.

**Range:** default triangular matrix sizes 4...32 (`n*(n-1)/2` is 6...496, inside compact storage's 5...512 elements). TieredStorage itself fails at every compact count, not just one dimension.

**Evidence:** isolated Swift `-O` executable using exact source observed original/copy values `99/99` at n=4,16,32; control n=2,3,33,64,65 remained `1/99`. Internal bulk mutation at counts 5 and 512 similarly changed the supposedly unchanged copy. High confidence; verified behavior.

**Remedy/test:** perform uniqueness checking on the reference actually shared by the value wrapper, using an ownership representation that permits checking that stored reference without creating an extra owner. Add a mechanical tier-boundary matrix of copy tests covering subscript, `setAt`, and bulk mutation (counts 0,1,4,5,512,513,2048,2049; public matrix n=3,4,32,33,64,65). Include an iterator created before mutation. Preserve both ordinary and throwing-closure semantics consistently.

### MS-2 — P2 performance: dynamic-vector repeating construction performs quadratic copying

**Location:** `Storage/HybridStorage.swift:125`–136, `:173`–189, `:224`–228. Public caller `Vectors/DynamicVector.swift:38`–40; resize at `:139`–145.

`HybridStorage.ensureUniquelyReferenced` extracts its heap payload into a local `ManagedStorage` while the enum still owns the original. The nested reference consequently appears shared; this function allocates and copies a replacement. The following setter extracts the replacement again and its own COW check also copies. `fill` performs this for every scalar rather than filling in one mutable borrow. `DynamicVector(dimension:repeating:)`, zero/ones factories and resize use this fill path.

**Trigger:** any heap-backed Float `HybridStorage` / repeating `DynamicVector` with dimension >16. Constructing one d-element vector performs Θ(d) allocations and Θ(d²) copied/zeroed elements. This is consequential for embedding dimensions and repeated construction. No wall-clock speedup is claimed.

**Evidence:** optimized standalone compilation of the exact Managed/Hybrid/Dimension storage source with counters inserted only in allocation/copy constructors:

```text
managed 10 writes allocations/copies 0 0 unique true
dimension 10 writes allocations/copies 0 0 unique false
hybrid 10 writes allocations/copies 10 10 unique false
hybrid fill512 allocations/copies 513 512 last 42.0
hybrid repeating512 allocations/copies 514 512 last 3.0
dimension mutable buffer allocations/copies 0 0
```

Here `allocations` counts normal `StorageBuffer` allocations; `copies` counts its separately allocating copy constructor. Thus repeating512 executes 1,026 buffer allocations total. This is allocation evidence, not elapsed-time benchmarking. High confidence.

**Remedy/test:** remove the redundant outer COW layer or restructure enum mutation so the unique owner is not retained twice; fill in one mutable-buffer call. Add allocation-scaling regression checks in optimized builds for unique writes/fill/constructor and shared-first-mutation, across inline/heap transition and realistic dimensions. Validate value semantics before using an ownership-extracting optimization.

**Important rejected extrapolation:** the superficially similar `DimensionStorage` extraction is optimized away for ordinary managed scalar writes and buffer mutation in this probe. Do **not** claim generic fixed vectors necessarily have the same mutation-copy cost. Its uniqueness-reporting getter does still return false for a unique heap instance.

### MS-3 — P2 performance: each large triangular-matrix write clones the complete allocation

**Location:** `Utilities/TieredStorage.swift:581`–587 (`AlignedBuffer.isUniquelyReferenced`), called at `:181`–183. Public loops `Utilities/TriangularMatrix.swift:225`–228 and `:258`–260.

`AlignedBuffer.isUniquelyReferenced()` copies `self` into `mutableSelf` while the enclosing tier also owns the object. It tests this extra local reference, so the result is false even for an otherwise uniquely owned matrix. Every subscript or `setAt` mutation deep-copies all m=n(n−1)/2 elements. Filling a matrix element-by-element through `fromFullMatrix` or `buildParallel` consequently copies Θ(m²)=Θ(n⁴) elements, in addition to the actual computation.

**Trigger:** default public TriangularMatrix size >=65 (more than 2048 stored elements). The documented example size1000 lies on this route. Ten writes are enough to demonstrate it; no huge allocation or timing run was needed.

**Evidence:** exact source compiled with `-O`, with one integer counter added to `AlignedBuffer.copy()`, reported `unique size65 matrix 10 writes aligned copies 10`. All values remained correct. High confidence.

**Remedy/test:** check uniqueness at the owning stored-reference level as in MS-1, and allow bulk construction to fill one exclusive allocation. Add an optimized allocation regression proving a unique matrix can perform many scalar writes without copying; first mutation after a retained copy must detach exactly once. Because two independent COW failures occur in this component, class-wide ownership tests are warranted rather than patching each example alone.

### MS-4 — P2 memory lifetime: compact generic storage never releases initialized elements

**Location:** `Utilities/TieredStorage.swift:406`–418 and class `:380`–488 (no `deinit`). Public path `Utilities/TriangularMatrix.swift:44`–52.

`CompactBuffer<Element>` initializes the tail elements of a plain `ManagedBuffer`, but neither subclasses it with element destruction nor deinitializes those elements from the wrapper's `deinit`. Deallocating a ManagedBuffer allocation does not supply destruction of the manually initialized tail. Arbitrary `TriangularMatrix<T>` is public and unconstrained, so this retains class instances, Array/String storage, and other nontrivial values beyond matrix lifetime.

**Evidence:** create a `Token`, weakly observe it, create `TriangularMatrix<Token>(size:4, defaultValue:token)` within a scope, keep it alive to scope end, then leave scope. Optimized probe prints `compact reference leaked true`. The only intended strong owners were inside that scope. High confidence; runtime reproduction.

**Remedy/test:** deinitialize exactly `header.count` tail elements once at the actual owner destruction boundary. Test weak sentinels/deinit counts through compact creation, copies, replacement, and destruction; include all storage tiers so future threshold changes do not move reference-bearing elements into an untested allocator.

### MS-5 — P2 unsafe API correctness: alignment changes generic `alignedCopy` initialization semantics

**Location:** `Utilities/MemoryUtilities.swift:135`–141.

The aligned branch calls `destination.initialize(from:count:)`, while the unaligned branch assigns `destination[i] = source[i]`. For nontrivial generic T, initialization requires uninitialized storage whereas assignment requires initialized storage. No single destination-state contract is implemented by both branches; pointer alignment silently changes lifetime handling.

**Evidence:** exact public function, two distinct aligned allocations of one initialized Token each, and a weak reference to the previous destination. After copying at offset0 (64-byte aligned), the old destination remains leaked; at offset1 (still naturally aligned for Token but not 64-byte aligned), it is released. Optimized output:

```text
alignedCopy offset 0 old destination leaked true
alignedCopy offset 1 old destination leaked false
```

The other interpretation—destination is initially uninitialized—makes the assignment branch unsafe because it may release an uninitialized reference. This latter crash mode is source-derived and was not executed. No production callers were found, but this function is public and unconstrained in T.

**Remedy/test:** choose and document either initialize or update semantics, use the corresponding primitive on every route, and define overlap support separately. Or split into explicitly named initializing/updating APIs. Cover initialized and uninitialized destinations as appropriate, naturally aligned offsets, class/deinit sentinels, and nontrivial value types. Do not let tests exercise Float exclusively.

## Dormant internal defects / opportunities (separate from public-path findings)

### MS-6 — P3 latent correctness: internal bulk mutation loses writes, and slices escape owners

- `Utilities/TieredStorage.swift:224`–227 mutates a local `ContiguousArray` and never writes it back into `.standard`. Verified `TieredStorage` counts513/2048 remain unchanged after `$0[0]=19`; controls inline/aligned preserve the write. `TieredStorage.withUnsafeMutableBufferPointer` has no production callers in this checkout, so this is presently dormant. Add a `defer` to store the modified payload back and include normal/throwing-closure tests across every tier before introducing a bulk matrix route as suggested by MS-3.
- `Utilities/TieredStorage.swift:218`–220 and `:365`–371 only copy inline mutations back after the closure returns normally; a thrown error skips writeback. This differs from its pointer-backed tiers and from the `defer` used correctly in HybridStorage/DimensionStorage. Source-derived; not separately executed.
- `Storage/UnifiedStorage.swift:280`–285 and `Storage/DimensionStorage.swift:298`–305 return `StorageSlice` holding only a pointer from a scoped borrow. Heap slices do not retain the owner and can dangle when it dies; inline DimensionStorage slices immediately refer to the closure's temporary tuple. No production calls to these internal slice methods were found. Remove the unused API or retain a suitable owner / redesign as a scoped slice closure before routing any new code through it.
- `UnifiedStorage.swift:16`–70 experimental `MoveOnlyStorage.into()` returns its pointer while the consumed value's deinit still frees it. Gated off by `MOVE_ONLY_EXPERIMENTAL`, absent from package settings; this is a dormant ownership-transfer defect, not an active package failure. Do not enable that gate without a separate lifetime audit and death/use-after-free test.

## Existing documented limitations, not rediscovered findings

- SoA access after successful `consumeAllocation()` is invalid. Some accessors (`lanePointer`, `withUnsafeRawBuffer`) do not check ownership; docs explicitly impose the caller rule. PageAlignedBuffer does check ownership on its public checked accessors. Both return the original allocation at transfer and distinguish logical from allocated length. Callers still need synchronization; no GPU behavior was tested here.
- SoA, PageAlignedBuffer and public raw mutable class storage use `@unchecked Sendable`; there is no general data lock. Core's ownership docs already assign synchronization to callers. An immutability/owned-transfer API could make misuse harder, but these annotations alone are not a new proof of a race in a conforming caller.
- AlignedMemory count multiplication, page rounding, SoALayout products and type alignment are caller obligations documented in Memory_Alignment/SoA_Layout_Contract. Swift checked integer arithmetic generally traps, rather than silently wrapping, under supported build modes. Checked-size helpers would improve API robustness, but do not describe these as newly found representable-input overflows.
- AlignedDynamicArrayStorage falls back to natural alignment after an allocation error yet still reports the requested `guaranteedAlignment` (`:180`). This contradicts its nominal alignment guarantee, but no realistic fallback was induced. Prefer preserving alignment on fallback or honestly reporting it. Keep as an unverified failure-path contract concern rather than a reproduced primary defect.
- `SoACompatible` is public and externally conformable. Built-ins have full SIMD4 lanes; consistency of custom lane/dimension/storage properties is explicitly assigned to custom conformers. Do not infer arbitrary dimension coverage from four built-in conformances.

## Strengths and architecture observations

- Explicit page-owned allocation and scoped read-only buffer protocols are a useful CPU/GPU ownership boundary. `SoALayout` centralizes the frozen index/stride formulas, and tests exercise non-power-of-two candidate counts and logical-vs-rounded size.
- The dedicated `AlignedValueStorage` COW wrapper consistently detaches before both scalar and buffer mutation; its fixed-size wrappers all enforce matching initializer counts and expose logical counts. `COWDynamicStorage` similarly detaches, although its copy uses an avoidable temporary array and is no longer the DynamicVector backend.
- Float-only storage uses correct matching `free` versus Swift deallocation flags where two allocators are possible. Plain SoA zero-initializes logical storage; page paths also initialize padding. Existing page tests cover alignment, content, transfer identity and logical length.
- Multiple overlapping ownership layers (ManagedStorage, HybridStorage, DimensionStorage, AlignedValueStorage, old dynamic classes, COWDynamicStorage, TieredStorage) increase the chance of checking uniqueness at the wrong level and of documentation describing obsolete paths. Consolidation should begin with the concrete verified COW failures and consumer compatibility, not wholesale replacement.
- Existing tests are strong on layout/alignment and weaker on ownership equivalence across storage tiers, generic destruction, and allocation scaling. A small mechanical ownership suite would close several defect classes that numerical parity tests cannot reveal.

## Validation and limitations

No repository edits or shared `swift test`/`swift build` runs. Orchestrator owns full tests. All isolated fixtures are under `/private/tmp/vectorcore-review-2026-09-14/memory-probes/`:

- `TieredProbe.swift` / `tiered-probe`: concatenated original TieredStorage, TriangularMatrix, MemoryUtilities and AlignedMemory (excluding its unrelated AlignedValueStorage conformance), minimal VectorError stub; one optional aligned-copy call counter. Compiled using `swiftc -O -parse-as-library`.
- `COWProbe.swift` / `cow-probe`: original storage source, minimal protocol/error/SIMD provider stubs, integer allocation/copy counters added at constructors. Compiled `-O`; no ownership behavior altered. SIMD stubs are irrelevant to the pointer/copy observations and were not used to establish numerical behavior.

Hardware target reported by swiftc was arm64 macOS26. No wall-clock benchmarking, ASan, TSan, allocator-failure injection, or external GPU import was performed. The optimized probes are focused reproductions rather than the package's full test target, and should be converted into mechanical regressions before fixing. Formal Swift memory-binding questions in older tuple helpers were not promoted without a reachable failing path.

Cross-component observations sent to orchestrator: public DynamicVector.storage replacement can desynchronize immutable dimension; TriangularMatrix.fromFullMatrix([[value]]) indexes matrix[0][1] at line223. These belong in the vector/API review so they are not duplicated as primary findings here.
