# Component 10 — manifold learning / UMAP review

Revision: `8b02e75bca7055ae11170873d15c06f1538709b7`. Reviewed all of `Sources/VectorCore/ManifoldLearning/{KNNGraph,FuzzySimplicialSet,UMAP}.swift`, all of `Tests/ComprehensiveTests/UMAPTests.swift`, PCA flattening/initialization call sites, `Docs/{Package_Boundaries,API_Overview_Map,Numerical_Behavior,Linear_Algebra_and_Projection}.md`, and downstream `../VectorIndex/Sources/VectorIndex/HNSWKNNGraph.swift`. Searched VectorAccelerate sources for integration; its separate Metal UMAP implementation is not automatically invoked by Core.

Validation: standalone `umap_probe.swift` linked stable Debug objects via the orchestrator's `build_probe.py`; no repository changes or shared builds. Results: `umap_probe.log`, `umap_csr.log`, `umap_rate.log`, `umap_dimension.log` in this directory. Proofs below use the implemented equations and direct probes, requiring no external reference claim.

## Verified defects recommended for consideration

### 1. P2 correctness: legal output scales silently defeat the kernel curve fitter

Location: `Sources/VectorCore/ManifoldLearning/UMAP.swift:350–372` (scale-dependent normal equations and unconditional result), with unnormalized sample grid at 320–327.

Trigger: `UMAPConfig(minDist: 1e-6, spread: 1e-5)` passes validation and reaches `fitABParams`; this returns `(a: 1, b: 1)` without signaling failure. Probe output:

```
spread=1        a=1.5769436 b=0.8950607 kernel(spread)=0.3880566055
spread=0.001    a=369981.88 b=0.8950607 kernel(spread)=0.3880566018
spread=0.00001  a=1          b=1         kernel(spread)=0.9999999999
spread=1e10     a=1          b=1         kernel(spread)=1e-20
```

For fixed `minDist/spread=0.1`, set `x=spread*t`. The fitting problem becomes `1/(1+A*t^(2b))` with `A=a*spread^(2b)` and identical target/grid in `t`. Therefore the optimum `b` and kernel value at `x=spread` are scale-independent. The observed change is a failure to solve the same mathematical problem, not expected UMAP stochastic variation or Float rounding. The absolute determinant cutoff (`1e-30`) can reject a well-posed fit solely because coordinates are small; unsuccessful iterations also return the initial guess as if fitted.

Impact: valid public configuration silently produces a radically different output similarity kernel and layout. Existing `UMAPCurveFitSuite.fitQuality` only exercises spread 0.7–1.5; default constant coverage does not catch scaling.

Remedy/test: fit a dimensionless grid with `minDist/spread`, transform the fitted coefficient back using `a=A/spread^(2b)`, and verify convergence and representability. Use a log coefficient or normalized distance evaluation where Float `a` cannot represent the result. Add a scale-family test asserting the same `b` and dimensionless kernel over several orders of magnitude, plus explicit failure behavior outside supported output ranges. Confidence: high, probe plus exact change-of-variable derivation.

### 2. P2 correctness: smooth-kNN calibration loses scale invariance for valid finite graph distances

Location: `Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:65–87`.

Trigger: a valid 3-node graph has offsets `[0,2,2,2]`, neighbors `[1,2]`, and distances `[s,2*s]`. All stored distances remain finite and nonnegative for the tested scales. The defining equation is `1+exp(-s/sigma)=log2(3)`, so `sigma≈1.86495*s` and the nonnearest membership should be `log2(3)-1≈0.58496` at every scale. Probe:

```
s=1      sigma=1.8649902      symmetric weights=[1,0.58496934,1,0.58496934]
s=1e-30  sigma=5.421011e-20   symmetric weights=[1,1,1,1]
s=1e30   sigma=1.5000001e27   symmetric weights=[1,1]  // second pair lost entirely
```

Mechanism: starting at 1 and allowing only 64 halvings/doublings restricts the explored bandwidth independent of row scale. The absolute starting scale overwhelms tiny data; for large data even the bandwidth floor remains far below the true solution. The finite-input graph contract does not limit distances to the roughly explored dynamic range.

Impact: units alone can change graph topology and membership, before layout initialization or SGD. At large scales a valid neighbor pair underflows away. This concerns graph-only/ANN input, so avoiding the reference brute-force builder does not avoid it.

Remedy/test: normalize distances by a positive row scale or scale the initial bracket, then return bandwidth in original units. Add multiplicative scaling tests with isolated rows, duplicates and varying degrees; check both membership preservation and satisfaction of the equation where a solution exists. Confidence: high, direct probe and closed-form expected solution. Existing `smoothDistancesHandGraph` verifies the unit-scale case only.

### 3. P2 API correctness: config validation admits infinities and returns invalid layouts

Location: `Sources/VectorCore/ManifoldLearning/UMAP.swift:102–115`, used by both public overloads.

Trigger: valid 3-point graph, `UMAPConfig(epochs:5, learningRate:.infinity)`. The `learningRate > 0` guard accepts infinity. The public function succeeds with coordinates `[nan,nan,nan,inf,inf,inf]`. Likewise `spread:.infinity` passes the relational checks, reaches a grid with nonfinite values and returns a meaningless fallback fit; `minDist=spread=.infinity` also passes by code inspection.

Impact: the existing throwing config boundary gives no diagnostic for unusable tuning values; downstream rendering/geometry receives nonfinite coordinates. NaN config values generally fail the comparisons already, so this is specifically incomplete finiteness checks rather than an assertion that all NaNs bypass validation.

Remedy/test: require finite learningRate/spread/minDist and document any supported magnitude bounds. Extend `UMAPBehaviorSuite.errorContracts` with table-driven `nan`, positive infinity and negative infinity cases per floating field. A finite but huge learning rate can also overflow, so finiteness is necessary rather than a universal convergence guarantee. Confidence: high, public probe.

## Verified boundary failures / documented limitations (do not present as newly undisclosed ordinary-input defects)

### CSR throwing initializer can trap before detecting malformed offsets

`KNNGraph.swift:67–74`: `KNNGraph(pointCount:2,rowOffsets:[0,3,2],neighborIndices:[1,1],distances:[1,1])` terminates with `Index out of range` (probe subprocess exit -5). It checks monotonicity one row at a time and traverses row 0 before discovering row 1's decrease. All individual entries read before the out-of-bounds element avoid self-loop/index errors, exposing the defect. `KNNGraphSuite.csrValidation`'s decreasing fixture also has a mismatched final offset, so it never exercises this path.

**Qualification:** `Docs/Linear_Algebra_and_Projection.md:114` explicitly requires callers to supply structurally valid buffers and says this is not an arbitrary-data parser. Therefore this is an acknowledged validation boundary/hardening opportunity (P3), despite conflicting API wording “Throws invalidData for malformed structure” and downstream “single contract gatekeeper” comment. Resolve that mismatch by validating all offset bounds/monotonicity before dereferencing. A small table or generated malformed-offset matrix should close the class. I would not describe it as a new exploitable parser vulnerability.

### Extreme integer options trap instead of throwing

`UMAP.swift:450`: `negativeSampleRate:Int.max, epochs:2` on the 3-point graph terminates during conversion of a Double value to Int (exit -5); `Double(Int.max)` rounds above Int.max and the due-sample arithmetic reaches that value. `UMAP.swift:229/241` (also vectors-path multiplication) performs unchecked `n*dimensions`: `dimensions:Int.max` traps on the same tiny graph. These values are operationally impractical, but they are accepted by the positive/nonnegative config checks. P3 boundary hardening: checked size arithmetic and representable sampling counts, or documented supported upper limits. Probe logs record both failures. No ordinary-sized negative-sampling schedule failure was established.

### Gram cancellation affects the advertised exact reference graph builder

`KNNGraph.swift:166–167`: the 1D Float vectors `[10000,10002,10001]` with k=1 return neighbors `[1,0,0]`, all distances zero. Direct differences give row-0 nearest neighbor 2 at distance 1, not neighbor 1 at distance 2. This is the familiar Gram cancellation problem already documented for matrix Euclidean paths in `Docs/Numerical_Behavior.md`; it is not unique to UMAP or a new algorithmic complexity failure. The builder's “exact” wording should mean exhaustive search over computed scores, and its docs should explicitly point to the conditioning caveat. If it is intended as a numerical reference oracle, use stable direct differences or a validated refinement policy. Existing Gaussian fixture avoids large common offsets.

### Explicit initial-coordinate values are shape-checked only

`UMAP.swift:221–234` accepts NaN/infinite coordinate values. Probe with `[[nan,0],[1,0],[0,1]]` returns a NaN and other significantly shifted points; it does not throw. Finite coordinates of magnitude 1e30 also overflow squared differences and can remain effectively immobile. No all-finite/result-conditioning guarantee is documented, so treat this as an input-contract clarification opportunity. Validate finite coordinates or explicitly state caller responsibility; a robust general-purpose variant should normalize/rescale supplied initialization before SGD. Do not overstate this as a proven failure on normal PCA/random init.

## Performance and architectural opportunities (source-derived, not benchmarked regressions)

- **Graph representation amplifies memory at corpus scale.** `FuzzySimplicialSet.swift:105–154` materializes and sorts 16-byte canonical tuples for E directed input entries, reserves 16-byte fused tuples for up to 2E entries, then builds 12 bytes of head/tail/weight arrays per symmetric entry F. Sort scratch and array capacity growth add overhead. `UMAP.swift:412–431` then keeps original edges while creating 8 bytes of head/tail indices plus four Double arrays (32 bytes) per retained edge R. `nextSample=cadence` shares initially but detaches on its first mutation. Logical optimizer storage is approximately `12F+40R` bytes, plus input CSR, coordinates and array capacity slack. For n=5M,k=15,E=75M,F≈R≈150M this alone is about 7.8 GB decimal, before roughly 640 MB input CSR and coordinates. This is a sizing model, not a measured peak/RSS. F/R depend on reciprocity/pruning. Consider compact symmetric CSR, avoiding duplicate index arrays, releasing staging data earlier, and deriving negative cadence instead of storing it; profile Release peak allocations before promising savings.
- **Two graph-sized sorts and repeated full epoch scans.** Fuzzy construction is O(E log E + F log F), calibration up to 64E exponentials, and optimization scans R edges for every epoch even when edges fire sparsely. The current serial deterministic SGD and O(n²d+n²k) reference builder are expressly documented; neither is newly discovered missing ANN/GPU support. A bucketed edge schedule, radix sorting or linear CSR symmetrization is a measured-optimization candidate, with reproducibility/order tests needed for any schedule change.
- **No cancellation checkpoints.** Core's three manifold source files contain no cancellation checks or callback. Both public entry points are synchronous and may hold a cooperative executor thread for substantial time if called inside an async task. VectorIndex checks cancellation in graph workers, but cancellation stops being observed after entering Core layout. This is not a broken documented cancellation promise; a throwing cancellation-aware overload or epoch/progress callback is a useful composition option for large jobs. Do not add concurrency silently at the expense of the current deterministic contract.
- **Boundary is architecturally sound.** VectorIndex emits raw directed CSR in live-ID order and converts its L2-squared/cosine traversal distances to Euclidean/chord distances; Core owns rho/sigma and t-conorm fusion. Core does not depend on Index or auto-route UMAP into its compute-provider/GPU seam. Preserve the explicit row-ID ordering contract and let downstream packages own ANN construction and GPU execution.

## Strengths, coverage and rejected suspicions

- All graph fields are immutable arrays in a Sendable value; no mutable shared optimizer state or escaped pointer was found in this scope. Coordinate mutable-buffer lifetime is confined to synchronous optimization.
- Unsorted and uneven-degree graph rows are deliberately supported: rho scans the row minimum positive distance, and calibration uses actual row degree. Isolated rows produce no edges and are covered by `isolatedRow`.
- The self-free graph convention deliberately uses log2(k+1), with a clear explanation and a closed-form test. I did not label this as an off-by-one bug simply because umap-learn counts self differently.
- Duplicate-point zero distances, coincident initial points and degree-one rows do not automatically cause division by zero in the inspected paths. An unattainable smooth-kNN target for tied nearest neighbors is already handled by the bandwidth floor and explicitly tested; no claim that every graph has an exact calibration solution is warranted.
- T-conorm symmetrization preserves both directions, merges repeated directed entries by maximum before fusion, and sorts by head/tail. Existing hand-graph tests check exact expected weights. Duplicate entries still participate in calibration before merging; deduplication invariance is not promised and was not reported as a defect.
- Sampling uses separate seeded state, the attractive update moves both endpoints, the negative update moves the head, and zero negative samples are supported. Tiny epoch counts can make no/very few updates by design of the cadence; no independent schedule-math defect was proven.
- `determinism`, `graphInjectionEquivalence`, `graphEntryPoint`, `clusterSeparation` and error/accessor tests provide useful baseline coverage. They do not establish broad manifold quality or scale robustness. No cross-hardware bitwise guarantee is inferred; docs correctly qualify that claim.
- Scope limitations: no full corpus run, no memory/performance benchmark, no cross-backend UMAP parity study, no new shared test execution. External GPU UMAP math and downstream HNSW traversal correctness belong to their own review scopes.
