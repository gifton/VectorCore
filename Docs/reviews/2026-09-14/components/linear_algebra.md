# Component 09 review: dense linear algebra and PCA

Revision: `8b02e75bca7055ae11170873d15c06f1538709b7`. Review only; no repository edits or shared builds.

## Coverage and method

Read every line of:

- `Sources/VectorCore/LinearAlgebra/LinearAlgebraProvider.swift`
- `Sources/VectorCore/LinearAlgebra/LAPACKLinearAlgebraProvider.swift`
- `Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift`
- `Sources/VectorCore/LinearAlgebra/MatrixMultiply.swift`
- `Sources/VectorCore/LinearAlgebra/PCA.swift`
- `Sources/VectorCoreC/common/vc_lapack.c`
- `Sources/VectorCoreC/include/vc_lapack.h`
- `Tests/ComprehensiveTests/LinearAlgebraProviderTests.swift`
- `Tests/ComprehensiveTests/PCATests.swift`
- `Docs/Linear_Algebra_and_Projection.md`, `Docs/Numerical_Behavior.md`, `Docs/Package_Boundaries.md`, `Docs/API_Overview_Map.md`.

Traced PCA's UMAP initialization caller (`UMAP.swift:266–273`) and the shared flattening call from `KNNGraph.bruteForce`. Searched sibling VectorIndex/VectorAccelerate Swift sources for these model/provider names; no direct usages found. This is a name search, not a complete downstream compatibility review.

Checked column-major indexing, all GEMM transpose modes, thin factor dimensions, lower-triangle symmetrization, permutation/sign rules, rank bounds, scalar precision, task-local propagation, buffer lifetimes, model construction, iteration bounds, workspace allocation, and in-memory complexity. A standalone executable links existing Debug objects through the root-provided `build_probe.py`; its source is `/private/tmp/vectorcore-review-2026-09-14/la_probe.swift`. No external numerical/API uncertainty needed web research: the findings below follow from local arithmetic and independent closed-form answers. The root owns full Debug/Release test results.

## Prioritized verified candidate findings

### LA-1 — P2, numerical correctness: SVD reports convergence before orthogonalizing finite scaled inputs

**Location:** `Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:231–238`, especially line 238.

**Trigger:** Factor the full-rank 2-by-2 column-major buffer `[s, 0, s, s]` with `SwiftLinearAlgebraProvider.svdThin`, at `s = 1e-12` or `s = 1e15`. The matrix is well conditioned (condition number about 2.618), and both inputs and correct outputs are representable Floats. This does not involve zero singular values or the documented incomplete-null-space basis limitation.

**Mechanism:** `gamma * gamma <= eps * eps * alpha * beta` uses Float arithmetic on fourth-power quantities. For the small scale both sides underflow to zero. At the large scale both sides become infinity. The comparison succeeds in either case, so the implementation skips the required rotation, leaves `converged = true`, and returns norms of nonorthogonal columns as the singular spectrum.

**Observed Debug probe:**

| scale | Swift spectrum divided by scale | LAPACK / exact spectrum divided by scale | Swift U column dot |
|---|---|---|---|
| 1 | [1.6180341, 0.618034] | [1.618034, 0.618034] | 2.98e-8 |
| 1e-12 | [1.4142135, 1] | [1.618034, 0.618034] | 0.70710677 |
| 1e15 | [1.4142135, 1] | [1.618034, 0.618034] | 0.7071068 |

The exact spectrum follows from the eigenvalues of `[[1,1],[1,2]]`: singular values are `(sqrt(5)+1)/2` and `(sqrt(5)-1)/2`. Reconstruction alone can still pass: unrotated columns divided by their norms and multiplied back reproduce A, while the spectrum and orthonormality contracts fail.

**Impact/reachability:** Public provider calls and PCA fits under its supported task-local override can silently return incorrect spectra and principal axes. The Apple default LAPACK path did not exhibit this error in the same probe.

**Remedy/test:** Use range-safe correlations/norms (Double products and accumulators are sufficient for finite Float inputs at the validated dimensional limits, with appropriate conversion of output quantities), or scaled normalized comparisons. Add scale-equivariance tests covering QR/SVD/eigen with nonzero full-rank fixtures across exponent ranges. Assert normalized spectrum, factor orthogonality, normalized reconstruction residual, and output finiteness; reconstruction alone is insufficient. The recurring unscaled-norm arithmetic in LA-2 makes a mechanical class-wide exponent sweep appropriate.

**Confidence:** High; executed public-API reproduction plus analytical spectrum.

### LA-2 — P2, numerical correctness: Swift QR silently loses data or emits NaNs on representable finite vectors

**Location:** `Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:46–55`.

**Trigger:** Thin QR of `[s, s]`, rows 2, columns 1, at `s = 1e-25` or `s = 1e20`.

**Mechanism:** The unscaled Float sum of squares underflows or overflows although the norm itself is representable. Underflow takes the zero-column branch, leaving Q's column `[1,0]` and R `[s]`, losing the second input element. Overflow feeds infinite beta into `(beta-alpha)/beta`, producing NaN reflectors and Q. The method returns success in both cases.

**Observed Debug probe:**

- `s = 1e-25`: Swift Q `[1,0]`, R `[1e-25]`; reconstruction `[1e-25,0]` has relative 2-norm error `1/sqrt(2)`. LAPACK Q approximately `[-0.7071067,-0.7071067]`, R `[-1.4142136e-25]`.
- `s = 1e20`: Swift Q `[NaN,NaN]`, R `[-infinity]`. LAPACK Q approximately `[-0.7071068,-0.7071068]`, R `[-1.4142136e20]`.

The SVD final normalization has the same sum-of-squares range exposure at lines 268–274; fix/test the class, not just QR. Extremely small nonzero norms also make `1 / norm` vulnerable; merely widening the accumulation while immediately narrowing all reflector arithmetic is not a complete full-range repair.

**Impact:** Fallback factorization violates `A=QR` on nonzero finite input, and QR-stabilized PCA iteration can propagate corrupt panels. The source's introductory “unconditional numerical robustness” claim is contradicted by these probes; the public docs specify the SVD zero-basis exception, not a restriction to an implicit safe input exponent band.

**Remedy/test:** Use scaled norm/Householder construction or sufficiently wide intermediate arithmetic, with finite-result/error handling where the mathematical result cannot be represented. Include the QR fixtures above and single-column SVD in a common exponent-range test matrix. Verify relative residuals, orthogonality and finiteness explicitly.

**Confidence:** High; executed public-API reproduction, correct result is a closed-form two-element norm.

### LA-3 — P2, input validation: nonnegative PCA oversampling can trap instead of being clamped or rejected

**Location:** `Sources/VectorCore/LinearAlgebra/PCA.swift:173`; validation at lines 137–146 only rejects negative configuration values.

**Trigger:**

```swift
try PCAModel.fit(
    [DynamicVector([Float(1)]), DynamicVector([Float(2)])],
    components: 1,
    config: PCAConfig(oversampling: Int.max))
```

**Mechanism:** `k + config.oversampling` overflows before the enclosing `min` can limit it to `min(n,d)`. The input is two ordinary 1D vectors; no large matrix allocation is needed to trigger the failure. The accepted oversampling setting can arrive from an application's user/config input.

**Observed Debug probe:** `/private/tmp/vectorcore-review-2026-09-14/la_probe overflow` exits with signal-trap status 133. Static derivation identifies the exact addition; normal mode uses valid vectors of the same type successfully.

**Impact:** The throwing fit API terminates the process for an oversized configuration value. This is distinct from the docs' warning that representable matrix shapes may not be practical allocations.

**Remedy/test:** Bound oversampling before addition, e.g. `k + min(config.oversampling, min(n,d) - k)` after rank validation, or use checked addition and a typed error. Exercise `Int.max`, `Int.max-k+1`, zero and ordinary values; verify either documented rejection or equivalent exact-path output without a trap. Validate config/k before packing the entire sample as a related efficiency improvement.

**Confidence:** High; executed trap and direct arithmetic derivation.

### LA-4 — P2 candidate, centering accuracy: rounding the mean before subtraction can double centered variance

**Location:** `Sources/VectorCore/LinearAlgebra/PCA.swift:160–164`; model stores only Float mean at line 80; batch/single transforms subtract that rounded mean at lines 257–260 and 291.

**Trigger:** Default, LAPACK-backed, centered one-component PCA of 1D Float samples `[16777216]` and `[16777218]`. Both input values are exactly representable; their exact mean 16777217 is not representable as Float, while the correct centered residuals -1 and +1 and sample variance 2 are exactly representable.

**Mechanism:** Double summation computes the correct mean, but `Float(sum / Double(n))` rounds it to 16777216 before centering, making residuals `[0,2]`. The algorithm then fits the second moment about that rounded center rather than the true sample covariance. More generally the rounded-mean error adds a rank-one term proportional to `n/(n-1) * deltaMean * deltaMean^T`, so in multiple dimensions it can also perturb axes. This is an avoidable centering precision loss, not inaccurate input parsing.

**Observed Debug probe:** mean `[16777216]`, explained variance `[4]`, projected samples `[[0],[2]]`; exact variance is 2. Subtracting the common offset first gives samples `[0,2]` whose exact mean is 1 and variance is 2, demonstrating lost translation invariance.

**Impact:** Near-constant dimensions with a large common offset can have a large relative variance error even with the default backend. Typical normalized embedding workloads are less exposed, so prioritize below the malformed configuration trap and fallback decomposition failures. The docs acknowledge general conditioning concerns; they do not explicitly identify this rounded-centering behavior. This is a candidate numerical-quality fix rather than an assertion that all Float pipelines must promise relative accuracy arbitrarily near zero.

**Remedy/test:** Retain a higher-precision internal center or a Float anchor plus residual mean, and center using sufficient precision before converting residuals to Float. Fit and both transform forms must use the same improved center; changing only fit would make the stored model inconsistent. Add offset/translation invariance tests with exactly representable inputs and residuals, independently calculate covariance in Double, and check projected sample mean and centered variance.

**Confidence:** High on behavior and cause; priority is subject to the intended centering precision contract.

## Strengths and verified structural conclusions

- Column-major layouts and leading dimensions are internally consistent across the Swift/LAPACK seam. Wide SVD correctly factors the transpose and swaps/transposes factors. PCA's row-major component rows are correctly reinterpreted as column-major W-transpose for batch transform.
- QR extracts R before explicit Q generation overwrites reflectors. LAPACK workspace query/execute buffers remain scoped through calls; allocations are freed on the observed error paths. No pointer escape or Swift buffer lifetime defect found in this component.
- Both providers validate positive dimensions, shape and the Int32 dimension ceiling before narrowing. On this 64-bit target, products of two positive Int32 dimensions fit in Swift Int, so those shared shape-validation products are not themselves an Int-overflow defect.
- Eigen copies lower to upper triangle correctly, sorts eigenpairs together, and avoids allocating eigenvector storage when not requested. The fallback explicitly rejects overflow of its Frobenius-norm computation, which is safer than silently accepting an infinite convergence threshold.
- The randomized PCA path uses QR after both directions of subspace iteration, captures the task-local factorization provider once, and copies V-transpose axes with a deterministic sign rule. It has the intended O(n*d*ell) matrix-pass structure plus panel factorizations. Existing `swiftKernelMatchesPlatform` tests all four GEMM transpose combinations on rectangular shapes.
- Existing QR/SVD/eigen invariant and PCA covariance-reference tests cover ordinary finite inputs, rectangular shapes, sorted spectra, orthogonality, reconstruction, invalid dimensions and scoped provider overrides. These are meaningful tests, but their small, roughly unit-scale data does not close exponent-range defects.
- `PCAModel` has immutable public fields, no public initializer, and no Codable/Decodable conformance in this checkout. Malformed decoded-model invariants are therefore inapplicable. Public factorization result structs have unchecked initializers; custom providers are responsible for honoring the protocol's stated result shapes. Consumers treating a custom provider as trusted numerical implementation is reasonable; no separate malformed-provider finding asserted.

## Documented limitations, opportunities and rejected suspicions

1. **Incomplete null-space basis is already documented.** Swift SVD can leave zero U columns for tall/square input or zero V-transpose rows for wide input. PCA can consequently return zero axes on degenerate wide panels. Do not report this as a new bug. LA-1 deliberately uses full-rank matrices and positive singular values.
2. **PCA is in-memory, not online or corpus-wide streaming.** It packs the complete sample; batch transform also packs a batch and materializes its output. The public docs state this. For `n=500000,d=1536`, one Float sample matrix alone is 3.072 GB (decimal). Copy-on-write centering, the input vector collection, QR panels and factorization arrays add more. Exact fallback requests thin U even though PCA discards U. Source-derived opportunity: a right-factor-only SVD path or tall-QR reduction could reduce exact-fit memory; performance/peak-RSS benefits need coordinated Release measurements, not an unmeasured speed claim.
3. **Scalar transforms allocate per point and use Float accumulation.** Batch output also creates one small array per row. A caller-owned flat output / into-buffer transform could reduce allocation at serving scale, but this is an API/performance opportunity, not a correctness defect or measured regression.
4. **Cancellation is unsupported, not incorrectly propagated.** APIs here are synchronous and expose no cancellation contract. Fit and 60-sweep fallback factorizations have no cancellation checks; cancellation-aware chunking is a useful future concern for large interactive tasks. Arbitrarily large positive `powerIterations` is accepted and can request very long work; unlike oversampling overflow, this is explicit work requested by the caller.
5. **Factorization override does not replace GEMM.** Binding the Swift factorization provider on Apple still uses Accelerate for internal GEMM. This follows the narrower protocol and current documentation; it is not a complete pure-Swift execution switch or a task-local propagation defect.
6. **Scale claims in comments exceed evidence.** `PCA.swift:14–15` says no full n-by-d factorization is ever formed immediately before describing exact fallback; “100–500k ... fast,” and provider comments' “unconditional numerical robustness,” are not justified by current test coverage. `PCAConfig.seed` cross-provider determinism must also allow arbitrary basis changes in degenerate eigenspaces, as the public docs already do. Align source comments with the careful public docs and measured records.
7. **LAPACK dimension limit is not full workspace safety.** C expressions such as `8 * minmn` are evaluated in signed LAPACK integer arithmetic before size_t conversion; optimal workspace can exceed Int32 even when dimensions do not. The declared small-panel scope (roughly d<=2000 or sketch width<=100) makes extreme square cases outside the intended operating envelope; no practical-size C memory-corruption reproduction established, so do not elevate speculative extreme-dimension UB to a verified finding. A future promise of arbitrary shape support should add checked workspace sizing and re-audit backend index/workspace limits.
8. **Nonfinite values and allocation safety are not comprehensively validated by PCA.** The docs explicitly say so. LA-1/LA-2 use finite inputs with representable mathematical outputs; they are not requests to turn the numerical layer into a general nonfinite parser.
9. **Uncentered second moments use n-1.** Both the `explainedVariance` field documentation and code specify sigma-squared/(n-1). “Raw second moments” wording is loose for this normalization, but no new factor-of-n failure is asserted against the explicit formula.

## Validation limits

No benchmarks, stress allocations, shared swift builds, custom-provider abuse tests or non-Apple execution were performed. Probe outputs are from existing Debug package objects on this Apple arm64 environment; the root's full suite results are separate evidence. LAPACK supplied an independent comparator in probes, while exact 2-by-2 spectra/norms and sample covariance establish the mathematical expected results without relying on backend agreement alone.
