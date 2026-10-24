# L02 implementation — final v2

The approved PCA repair is implemented in the isolated archive based on `ed95f95133abcb5245e7dd1fc04e11ea3ceb96fd`. Production changes are confined to PCA.swift. The complete v2 patch is `L02-v2-diff.patch`, SHA256 `0f8a26ef7798fdf1e5ffcdc7bea0979359ae95bea7b1a33536c73052f1c08602`. Its source manifest covers 303 base and 305 final code files, with the two owned documentation files recorded separately. The reconstructed full archive inventory has 5,744 base and 5,746 final regular files, excluding generated Python caches, and exactly six changed/new paths.

## Behavior and migration

`PCAModel.mean` is now `[Double]`. Fitting computes a deterministic compensated Double sum of the original Float coordinates, divides once in Double, and stores that center. Fit, single transform, and batch transform use one `Float(Double(input) - mean)` helper. Components, panels, projections, and published variance statistics remain Float. PCATests' actual mean consumers now reconstruct and compute their independent covariance reference in Double; no other existing test or production caller was migrated.

All nonnegative oversampling values, including Int.max, clamp to the remaining sketch width before addition. The existing exact fallback, randomized algorithm, seed consumption, sign convention, and provider override remain. Pure local helpers validate LP64 dimensions, component/configuration bounds, flat-buffer count, element products and Float/Double byte counts before the touched allocations and BLAS calls. Public n < 2 remains invalidOperation; internal invalid shapes and representability failures are invalidDimension; valid-shape wrong flat counts and ragged rows are dimensionMismatch.

Structural checks precede finite scans. Nonfinite input is invalidData; unrepresentable residuals or nonfinite published statistics/components/projections are stage-specific operationFailed. Provider errors propagate unchanged, including errors caused by an overflowing intermediate Float panel. Shared flattening retains KNN's values, including nonfinite bit patterns, and applies only general row/count/byte validation, not PCA's finite-input or LP64 policy.

The caller inventory identifies KNNGraph's shared flatten call and UMAP's fit/transform use. Neither consumes mean or required an algorithm change. The public Swift 6 consumer deliberately reads `[Double]`, fits the translated pair with Int.max oversampling, and executes both transforms. Downstream users must reconstruct in Double or explicitly round mean where they choose; no downstream integration claim is made.

## Actual RED and version history

Unchanged base production was compiled with the initial new regression suite before production edits. Debug exited 1 with 22 issues across three test functions: 20 numerical assertions failed for the translated pair under default and Swift providers, and two parent assertions reported the actual oversampling child's SIGTRAP (status 5, reason 2). The unshifted pair passed. The parent-process childProbe entry was a no-op; one actual oversampling child ran and did not emit its completion marker.

A retained separately linked baseline probe measured both providers: `[0,2]` produced mean 1, variance approximately 2 and projections [-1,1]; `[16777216,16777218]` produced mean 16777216, variance 4 and projections [0,2], including Operations.pca. The explicit `[Double]` mean consumer separately failed baseline typechecking because the public property was `[Float]`. These distinguish the qualified NUM-10 accuracy opportunity from the actual API-20 integer trap and from the source migration.

V1 froze the complete implementation and tests. Its Debug build compiled production but failed test compilation because two inferred fixture expressions exceeded Swift's type-checking budget. Those failed artifacts remain untouched. V2 changes exactly two files from v1: typed test fixture loops plus a compensated-mean control, and the root-approved GaussianSource comment correction. The latter disclaims cross-platform libm bit identity; no production behavior changed between v1 and v2. All successful final runtime results below use v2 directly, not an inferred v1 mapping.

## Verification

| Check | Actual result |
| --- | --- |
| v2 Debug | Exit 0; 37 functions in 10 suites; two actual bounded children pass |
| v2 Release | Exit 0; same 37 functions in 10 suites; two actual bounded children pass |
| v2 ASan first selection | Exit 0; 13 functions in three suites: six direct new contract functions plus KNN/UMAP controls |
| v2 ASan existing PCA selection | Exit 0; nine functions in three suites |
| v2 direct ASan children | Oversampling and finite scenarios each exit 0 with completion marker |
| Public Swift 6 compile/link/run | Compile 0, run 0, `L02_PUBLIC_COMPLETE` |
| Child-process scanner | Seven Python tests pass, including repository scan |

Debug and Release each report one no-op parent `childProbe` function among the 37 functions. `oversamplingInBoundedChild` is one parameterized parent function executing two scenario children; the finite child checks both providers internally. The ASan parent filters exclude both this parent and the no-op childProbe. Its two direct children actually load the ASan test bundle, with the ASan runtime explicitly preloaded. These are separate executions, not inflated test-function counts. The initial ASan filter included several unmatched legacy-looking suite names; it nevertheless ran the six intended direct contract functions and seven KNN/UMAP controls. The exact existing PCA suite names were then run separately. The first existing-PCA invocation's log reports nine passes but its shell chained the direct driver; a second identical skip-build invocation records an independent exit 0. Both logs are retained.

Analytic controls include translated one-dimensional pairs, signed power-of-two translations of axes with distinct known variances, zero/constant and uncentered fixtures, and `[2^60,1,-2^60]` whose exact sum is 1 and expected Double center is 1/3. Independent Double covariance, Rayleigh variance, projected mean, shape-aware reconstruction, and nonfinite rejection controls supplement seeded randomized-route and ordinary offset cases. The finite-aware comparison allowance is 4e-6 times the residual/variance scale (about 34 Float epsilons at unit scale), intended for these small Float panels and reductions; pair projections use a tighter 1e-6 absolute allowance. These tolerances are not a universal relative error bound.

The forwarding provider records actual QR/SVD dimensions across oversampling boundaries and checks exact fallback for Int.max powerIterations without executing a huge loop. A sentinel error verifies identity on exact and randomized paths. Pure shape tests exercise Int32, Int product and element-byte limits without huge allocations or vendor calls. Bounded finite scenarios cover input NaN/infinities, residual overflow, variance conversion overflow, single/batch projection overflow, shape precedence, empty batch, and both providers. Ordinary controls preserve original input bit patterns.

## Retention and provenance

Canonical root: `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-09-24/L02`.

`base.tar`, `red/debug-build`, `build-v1-debug`, and the three `build-v2-*` trees retain actual versioned source/build bytes. No build tree was rebuilt against a later source version. Artifact inventories record real files and exclude symlink aliases; the older initial RED inventory counted symlink-visible aliases, so its count differs without losing underlying bytes. `red` additionally retains the original child capture, public typecheck failure, numeric probe source/executable/driver and logs. Versioned source snapshots reside under v1/source and v2/source.

`L02-command-results.json`, public command/object records, numeric RED command records, and artifact manifests associate commands, exits, logs and source versions. `L02-child-associations.json` explicitly distinguishes the SwiftPM helper executable from the loaded test bundle and maps original temporary paths to canonical retained bytes. `L02-v2-asan-direct-children.json` records exact argv, environment, helper, bundle, runtime hashes, markers and bounded termination. Effective Python drivers and actual host/runtime copies are retained. Process handlers are installed before launch; captures are file-backed; scenario deadlines are 15 seconds with at most two seconds of cleanup, and the Swift parent throws on any timeout before querying termination status/reason.

Original pre-RED caller/full-inventory reports were lost in the September 29 temporary-directory cleanup. Root confirmed no survivor copies. `L02-reconstructed-base-inventory.json` and `L02-reconstructed-callers.txt` are explicitly reconstructed from immutable base.tar with its SHA and the reconstruction command; they are not represented as recovered original reports. Prior reported counts are historical and agree with this reconstruction. Immutable v1/v2 source inventories and RED artifacts survived. New evidence is copied immediately to canonical storage.

The final worker-only index excludes root-owned and independent-review records. Root reconstruction and independent review have separate lifecycles and are not incorporated into this worker's immutable evidence identity.

## Limits

Widened compensated centering addresses the demonstrated early Float mean rounding, not arbitrary exact summation or universal translation invariance. Input translation may already lose differences. Float centered panels, GEMM and published results still round and can overflow; no new matmul or factorization precision mode is claimed. Single/batch reductions can differ numerically. The accepted Swift rank-deficient basis limitation remains. Representability checks do not promise allocation success or a corpus memory/performance envelope. No throughput, whole-corpus, GPU, downstream or cross-platform bit-parity claim is made.
