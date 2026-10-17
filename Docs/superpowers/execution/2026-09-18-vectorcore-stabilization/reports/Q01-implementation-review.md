# Q01 independent implementation review

## Revision 2 disposition

**CHANGES REQUIRED / acceptance pending.** The reviewed production arithmetic is consistent with the binding Q01 contract; no numerical correctness blocker was found in the complete frozen revision 2. The child runner has one bounded-cleanup defect, also independently identified by root and already assigned for revision 3. Final Debug/Release reruns and valid-path ASan evidence remain pending. This review does not close NUM-03/04/05, EX-01/04, or SER-08.

Reviewed POLICY.md, binding briefs/Q01.md, the complete historical Q01-design-review.md including its resolved blockers, Q01-implementation-review-v1.md, the v2 manifest and all 1,665 lines of Q01-review-v2.diff. Prior reports and revisions are unchanged. No builds, production/test edits, index changes, commits or delegation occurred.

### Frozen artifact verification

- Base: `0dbb18c818f26edd8e28bb303a6fccedbd063a6f`; implementation archive: `/private/tmp/vectorcore-stabilization-Q01-implementation`.
- All seven final file SHA256 values and byte counts matched Q01-review-v2-manifest.json. All three existing-file base SHA256 values and byte counts matched `git show` at the stated base.
- Complete diff SHA256 matched `24998c1fd21f7770f32afe1f7674d2d5d336ac13c69769f5befa2a8d8552cfe6`. Independently regenerating the full unified diff from base and archive produced exactly the frozen diff bytes, including every new file.
- An independent archive-versus-base blob comparison over Sources, Tests and Package.swift found precisely the seven manifested changed/new files and no other changes in those areas. Shape bodies, numerical producers outside scope, Package and optional backend gates remain untouched.
- Independently evaluated all 96 retained affine fixture inputs with Python Fraction and checked both JSON references and Swift fixture result bits. All matched. Independently evaluated the revised negative wide-dot integer `-(2^31+127)^2 * dimension`; its Float bits match 3808428033, 3812622337 and 3821010945 at widths 512, 768 and 1536.

## Required correction and pending evidence

**R2-1 — Child cleanup is not bounded.** `Tests/ComprehensiveTests/QuantizationRangeContractTests.swift:107-111` polls with a 15-second wall-clock deadline, then sends termination/signals, but calls `process.waitUntilExit()` without a bound. Thus a timeout can still hang the test runner during cleanup, violating the binding child-process requirement. Root has requested a pre-run termination handler/semaphore, monotonic deadline and bounded cleanup for revision 3. Review that changed file and rerun the actual diagnostic matrix in Debug/Release before accepting. Unique file capture, selected child-only filter and owned-process signaling are otherwise appropriate.

**Pending final evidence.** v1 Debug and historical RED logs are not a substitute for verification of the final diagnostic and child-runner revision. Require exact final source/test hashes, successful focused Debug/Release exit records and selected symbol counts, intended-cause diagnostics for every child, plus ASan on valid raw/SoA/mutation and S01 controls with deliberate traps excluded. I inspected the retained metric RED summary showing 33 assertion failures, including the cancellation and EX04 cases; did not rerun baseline or final tests.

## Arithmetic assessment

### Calibration, parameter validation and conversion

Shared calibration validates bounds before arithmetic, preserves original metadata, handles zero canonically, includes zero for asymmetric intervals, and computes its rounded-away zero point from the ideal Double scale. The Float lower clamp prevents tiny nonzero scales collapsing to zero. The finite-decode cap uses the worst centered byte extent and checks the candidate's exact Double product before a downward Float step. With the bounded extent, this implements the requested cap without overflow. New calibrated full-byte-domain tuples therefore have the intended finite projection bound; legacy tuples are separately validated and preserved.

Custom Codable validates finite positive scale, finite ordered metadata and symmetric zeroPoint zero, preserving the five keys and unrestricted asymmetric Int32 offset. It does not recalibrate decoded values or incorrectly reject legacy raw-code/metadata relationships. Scalar/SIMD decode both perform Double centered subtraction/product followed by one Float projection. Infinity from a valid legacy projection is retained rather than falsely rejected as invalid parameters.

Scalar quantization validates before Double divide/add/round/clamp/cast; AoS and SoA conversion delegate to that operation. Finite Float/positive finite Float ratios remain within Double range. All three throwing array constructors preserve shape-error precedence and report actual nonfinite indices. Nonthrowing routes use named guard/fatalError boundaries instead of generic conversion or try! failures. Repeating/from/SoA/calibrator validation occurs before deriving extrema. Explicit empty SoA parameters are validated and retained; padding uses quantize(0) and logical count controls consumption.

The percentile helper checks the domain, branches for 100%, and compares against a floating upper bound before converting rank to Int. Rounded Double(Int.max) cannot reach an unsafe cast through the guarded branch. The absolute-magnitude nearest-rank interpretation and empty/unsupported-mode order match the brief. QuantizationSchemes and ordinary calibrator zero defaults agree with their specified modes.

### Integer sums, dot and cosine

Every centered Int32-derived code fits Int64; its product also fits Int64. AffineWideInteger sign-extends each product into high and adds the low bit pattern with carry, exactly representing the bounded built-in sums. Negative conversion forms two-word unsigned magnitude first, including the low-word-zero carry, then applies sign after high/low Double conversion. This avoids cancellation from separately converting signed high and unsigned low. No universally correctly rounded integer-to-Double claim is made.

Symmetric integer loops remain selected only for valid symmetric zero-offset tuples. Their full-byte products fit Int16 individually and their widened sums fit the stated built-in bounds. Asymmetric centered dot and both norm squares use exact wide accumulation. Cosine cancels positive scales, preserves the zero-vector distance 1, and checks nonfinite similarity before finite clipping. Dot applies the exactly representable Float-scale product in Double only after centered summation; the required zero-dot fixture cannot become a false nonzero or infinity through premature coordinate/product rounding.

### Euclidean and mixed/SoA routes

Exact scale equality now gates the symmetric integer path. The newly reachable unequal/asymmetric route evaluates both affine products through high plus fused-product residual, rather than projecting either coordinate to Float or subtracting two separately rounded products. The four-term growing expansion uses full two-sum transforms, retaining both residuals and high-part subtraction error. Its final reduction rounds the expansion for subsequent Double difference-square accumulation; bounded test accuracy, not universal correctly rounded metrics, is the claimed contract.

Both mandatory difference -1/+1 fixtures and power-of-two families are reached in actual scalar/mixed calls. Their underlying products and residuals stay far from Double overflow/underflow. Nonfinite mixed Float coordinates branch before the finite-only expansion: infinities produce infinite squared differences and NaNs remain NaN. Final positive accumulation, sqrt and scale projection use Double; no Float squares or premature Float scale products remain in the changed routes.

Both SoA routes guard output capacity first and write only indices below vectorCount. The integer fallback keeps query and candidate original affine tuples, avoiding the previous toFP32 loss. Existing integer square accumulation remains safely widened before squaring. Empty operations write no output. Optional C alternatives remain unenabled and unverified; no conclusion is extended to custom conformers with untruthful lane/storage dimensions.

## Ownership, coverage and oracle assessment

All three mutable-borrow implementations leave self untouched until the body returns normally, validate temporary coordinates before construction/publication, and propagate thrown body errors without attempting quantization. The remaining try! follows the fixed-shape, finite validation boundary; it is not used to detect nonfinite output. The docs correctly distinguish throwing rollback from successful no-op recalibration and legacy overflow rejection.

The new matrix reaches all three widths through concrete dot/cosine/Euclidean wrappers and generic kernels, mixed argument orders, and both generic batch routes. Existing selected S01 postreplacement tests reach the two public batchEuclidean512 wrappers. Required symmetric/asymmetric/legacy offsets, tiny/huge/equal/neighboring scales, zero/orthogonal/opposite/cancellation inputs and finite/nonfinite mixed classifications appear. Candidate counts 0...5 cover padding and output sentinels, including rebased output pointers. The SoA raw-code-zero fixture limitation is explicitly documented; its supported conversion's code 2 and independently derived difference -16777221 are tested, while the mixed +1 fixture remains reachable as SoA code 0.

Calibration exponent/interval matrices check finite endpoint tolerances and all 256 code projections. Codable tests cover invalid fields/relationships and all-width valid legacy payloads. Conversion uses an independently written scalar Double formula as its ordinary oracle; it intentionally matches the specified observable operation, while the exact cancellation oracles use arbitrary integers/Fraction instead of production helpers. Half-tie/neighbor/saturation inputs and all-width route agreement are present. Near-zero metric fixtures use exact zero assertions or widened ULP allowances, and finite expected values receive finiteness checks. Wide accumulator fixtures cover sign extension, positive carry, negative magnitude conversion, low-word-zero and cancellation.

The two existing expectation migrations are confined to root-approved changes: zero-inclusive asymmetric negative-only scale and explicit rejection of NaN/Inf conversion. The wider behavior changes and generic-metric/optional-C exclusions are retained in the review package. No unapproved existing-test relaxation was found.

## Resource observation for root adjudication

The new finite difference helper creates growing Swift arrays per coordinate, and ordinary calibrate now materializes all coordinates using flatMap instead of reducing extrema while streaming. These are visible source-level work/allocation changes: the latter changes auxiliary storage from a scalar reduction to a full dataset of Float coordinates. I have not measured optimized allocations, throughput or RSS and make no timing/regression threshold claim. Root is obtaining actual Release allocation evidence before deciding whether a bounded implementation refinement is required. Such a refinement, if requested, must preserve the exact affine difference/validation contracts and re-freeze its production files; it is not justified to discard the residual arithmetic for speed.

Acceptance remains pending R2-1's disclosed correction, root's resource adjudication, and the final frozen Debug/Release/ASan evidence. No additional numerical or scope blocker was identified during this source review.

## Revision 3 follow-up — child cleanup source correction accepted

Verified Q01-review-v3.diff SHA256 `6736233f3ed9f373fce8d9626467df457d8dd9b10a6b13e8b2302eb7dfceb228` and all seven final archive file hashes against its manifest. Comparing the complete v2/v3 diff artifacts shows only the disclosed child-runner replacement in QuantizationRangeContractTests; production and numerical test bytes are unchanged.

**R2-1 is resolved at source level in v3.** A termination handler is installed before Process.run and signals a semaphore. The normal execution wait uses a monotonic DispatchTime deadline of 15 seconds. Timeout signals only the owned running process with SIGKILL, waits at most two further seconds for termination notification, closes the file, records a test failure with cleanup status and returns. There is no unbounded waitUntilExit or Date-based polling remaining. Normal diagnostic reads occur after termination notification. An unsuccessful cleanup is explicitly reported rather than silently accepted. Final child-matrix runtime evidence remains required for this revised test source.

Root has now supplied actual linked Release allocation evidence for the resource concern, retained in `/private/tmp/vectorcore-Q01-root-allocation-v2` and `/private/tmp/vectorcore-Q01-root-allocation-baseline-unsandboxed`; I read both phase-2 output tables. The v2 unequal/mixed scalar cases record 2,048 allocation events/81,920 requested bytes at width 512 and 6,144/245,760 at width 1536, versus zero in the baseline scalar cases. Five width-1536 SoA candidates record 30,720/1,228,800 in either fallback, and ordinary calibration of eight width-1536 vectors records 28/156,288 versus baseline zero. These are the probe's allocation events and requested bytes, not measured RSS or latency; baseline unequal-distance results themselves are numerically defective and are not a replacement algorithm recommendation.

Root has required v4 to retain the same numerical transforms using fixed SIMD4<Double> expansion storage/counts and to stream finite/extrema validation in ordinary calibration. **Final approval is still withheld** pending that disclosed production revision, its full freeze/review, allocation recheck and final Debug/Release/ASan evidence. The v2 finding and review above remain historical evidence.

## Revision 4 follow-up — allocation correction source review

Verified Q01-review-v4.diff SHA256 `03b1df34f94d20807683da228af2d3f382ee6bd389382d0e63b136bdc03a1117`, all seven final file hashes/byte counts, and the three existing-file base hashes/byte counts. Independently regenerating the complete diff from the stated base and archive again matched the frozen artifact exactly. The v3-to-v4 delta changes only AffineQuantizationCore.difference and ordinary QuantizationCalibrator.calibrate; tests and all other production bytes retain their v3 hashes.

**The v4 production correction is acceptable at source level.** The fixed SIMD4<Double> expansion preserves the precise v3 term order, two-sum operations, nonzero residual retention, carry insertion and low-to-high final reduction. It introduces no SIMD horizontal sum/reassociation. Its capacity argument is local: before processing term index i, count <= i; at most count residuals are emitted, then one carry, so the output uses at most i+1 <= 4 slots. Every read is below the current count, and the carry write index is at most 3. Zero-initialized unused lanes do not enter the reduction. The previously reviewed dyadic/cancellation tests are unchanged and will exercise this implementation in the final runs.

Ordinary calibration now reads existing storage lanes, rejects each nonfinite value before extrema arithmetic and reduces min/max without an accumulated coordinate array. hasValues ensures the first finite value initializes both extrema and preserves its signed-zero representation; subsequent strict comparisons retain an equal first value. Empty storage returns literal positive-zero metadata and the requested symmetric/asymmetric mode. Unsupported perChannel still rejects before iteration. The final shared calibration/scale cap is unchanged. Distribution calibration's sort/materialization is unchanged and is not represented as having this streaming property.

No new numerical, bounds, API or scope defect was found in this revision. The earlier v2 allocation observation is **source-corrected, awaiting the linked Release allocation recheck**; removing explicit arrays does not itself establish the optimized runtime allocation count. The v3 bounded child helper remains intact and still requires the final diagnostic runs. Final acceptance remains pending frozen Debug/Release/ASan results and root's allocation evidence/adjudication.

Root reported restoration of externally missing worktree files before this review. All files used for the v4 manifest/base/diff verification were present and verified successfully. The unrelated environmental deletion/restoration is not classified as Q01's implementation diff or as an intentional scope change.

## Final v4 evidence adjudication — ACCEPT

**ACCEPT frozen Q01 revision 4 for the bounded implementation contract.** The prior child-cleanup and allocation findings are resolved by the reviewed v3/v4 changes and final evidence. Root retains integration and historical claim-closure decisions; this verdict does not extend the exclusions documented above.

Read Q01-implementation-final.md and Q01-validation-commands.json. Verified Q01-evidence-sha256.json SHA256 `cc6e63d6396547fc93d742d88494d5008a873515ea3dc56acfa4af9e300cc695`; every one of its 845 retained artifacts matches its recorded byte count and SHA256. Rechecked all seven archive source/test hashes against the unchanged v4 manifest. The recorded commands identify the same archive and final source manifest, with separate Release/ASan scratches; no final source mutation was found.

The final logs agree with their recorded exit-zero results: Debug and Release each show 17 XCTest methods with zero failures and 50 Swift Testing functions passing. Both explicitly show finiteOutlierSaturates and namedRejectionMatrix passing, exercising the bounded child runner and intended rejection diagnostics. The function total includes the documented parent no-op child entry; it is not misrepresented as an additional validation case. Valid-path ASan shows 10 XCTest methods and seven Swift Testing functions passing, with no sanitizer finding; deliberate trap children are excluded by the retained filter. These are inspected worker execution records, not reviewer-rerun builds.

Independently checked the root v4 allocation provenance's 131 frozen compiled artifact hashes and all recorded probe-source hashes, with no mismatch. Its recorded build/link/probe runs exit zero and their log hashes match. Read the durable phase-2 tables: all 22 measured v4 rows contain zero allocation events and zero requested bytes, and every result Float bit pattern is identical to the corresponding v2 row. This resolves the demonstrated per-coordinate expansion/ordinary-calibration allocation regression for those bounded fixtures, without implying a latency, RSS or general scale-readiness result.

The full source review, independent exact-reference checks, complete-diff/scope verification, final diagnostic tests and valid-path sanitizer evidence now satisfy the requested review gates. Earlier failed attempts and findings remain preserved in this report and their frozen artifacts.
