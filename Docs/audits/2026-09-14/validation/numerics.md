> Independent reviewer working notes. The [final claim register](../CLAIMS.md) controls priorities and dispositions. Historical temporary paths in replay commands are retained for provenance; portable replay guidance and preserved artifacts are in [the evidence index](../evidence/README.md).

# Numerical claims — independent second-pass validation

Revision: `8b02e75bca7055ae11170873d15c06f1538709b7`. Review only; no repository code/tests edited and no shared build run by this reviewer. Machine-readable record: `validate_numerics.json`.

## Outcome

All assigned principal numerical failures have a valid mechanism; most now have fresh full-library Release reproductions. Keep PCA centering, safeLog option ambiguity, existing conditioning limits and extreme configuration boundaries qualified. The reference equations, rather than agreement between two implementations, establish expected outputs.

**Remediation corrections:** `vDSP_vsdiv` also fails for subnormal division on this host. Direct `inPlaceDivide` succeeds while `/=` fails in the independent control. Normalize a representable direction even when the full magnitude/reciprocal is not representable; fixing the clamp reconstruction alone is insufficient. Quantized cosine should cancel scales, not merely reorder one square root. Stable QR/SVD repair must cover final normalization and reflector arithmetic. UMAP curve calibration invariance does not imply invariance of the complete SGD layout. Diagnostics must classify rounded results: 65505 becomes finite65504, while current counters call it overflow.

## Evidence and provenance

Fresh `validate_numerics/NumericProbe.swift` links existing current Release objects without production-source extraction or stubs. `compile.log`, `results.log`, and `provenance.txt` retain exact commands/results. A separate `DivisionProbe.swift`, `division-compile.log`, and `division-control.log` isolate direct method versus operator routing. Original four numerical probe sources and the linking script were read before reuse of their ideas. No sanitizer, GPU/Intel runtime, stress allocation, or performance measurement was performed.

The fresh fixture assertions are printed comparisons for audit, not a newly installed test suite. Float inputs used for the main defects satisfy their public finite/shape conditions; pointer storage is explicitly16-byte aligned and adequately sized. UMAP internal calibration probes are accompanied by public entry-point reachability in source. Crash claims marked confirmed-source use exact arithmetic/bounds proofs and reviewed original probes; they were not independently rerun here.

## Ledger

### NUM-01 — Scalar division computes an overflowing reciprocal even when each quotient is finite.

- **References:** C3, vectors V2, simd_kernels finding 4
- **Verdict / priority:** confirmed-reproduced / P2
- **Exposure:** Public vector operators, public Swift array/static providers; Accelerate provider also fails on this host.
- **Source:** Sources/VectorCore/Protocols/VectorProtocol.swift:264; Sources/VectorCore/Platform/ArraySIMDProvider.swift:181; Sources/VectorCore/Platform/ArraySIMDProvider.swift:639; Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift:358; Sources/VectorCore/Platform/SIMDProvider.swift:210; Sources/VectorCore/Platform/AccelerateSIMDProvider.swift:95
- **Evidence:** Fresh Release NumericProbe results.log: DynamicVector [s,0]/s and AccelerateFloatProvider.divideByScalar return [inf,nan] at s=1e-40. Independent DivisionProbe gives direct inPlaceDivide=[1,0], /= operator=[inf,nan], scalar s/s=1. Original Double probe source reviewed; Double reciprocal-overflow mechanism source-confirmed.
- **Qualification:** Do not assume vDSP_vsdiv repairs the problem: it fails on this Apple host. Direct inPlaceDivide and /= are distinct observed routes.
- **Improvement:** Use actual per-element division or a tested safe fallback when reciprocal cannot represent the operation; preserve signed-zero and nonfinite policies.
- **Acceptance:** Provider/type/operator conformance matrix Float/Double, normal/subnormal divisors, zero numerators and tails; compare representable quotients to widened scalar reference; test /= separately from inPlaceDivide.

### NUM-02 — Normalization reconstructs a mismatched scale or relies on an unrepresentable norm/reciprocal.

- **References:** C3, vectors V2, simd_kernels finding 1
- **Verdict / priority:** confirmed-reproduced / P2
- **Exposure:** Public generic checked normalization and public pointer path; concrete optimized helpers have additional scale limits.
- **Source:** Sources/VectorCore/Protocols/VectorProtocol.swift:402; Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:453; Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:483; Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:276
- **Evidence:** Fresh Release NumericProbe results.log: 1e-38 repeated over 1/4/5/17/512 has output Double norm 1.1754944..1.1754947. greatestFinite repeated over 4/5/17/512 becomes all zero (dimension1 approximately unit). Generic checked DynamicVector [1e-40,0] reports success with inf/nan. For clamped denominator den, current reconstructed mag=trueNorm*maxAbs/den, so output norm=den/maxAbs.
- **Qualification:** Representability of the full norm or its reciprocal is unnecessary for a finite unit direction. Deep-subnormal unchanged behavior is currently an explicit implementation policy, not proof that normalization is mathematically impossible.
- **Improvement:** Normalize in a scaled domain and state failure/unchanged policy; do not just replace maxAbs with den and retain reciprocal dependence. Preserve sign/direction while supporting finite norms outside Float range.
- **Acceptance:** Across typed/generic/pointer paths and widths/tails, assert unit norm in Double, positive alignment with original direction, zero/nonfinite policy, and unchanged-on-failure if promised. Include least subnormal,1e-38,maxFinite.

### NUM-03 — QuantizationSchemes clamps its zero point so positive/negative-only and constant intervals cannot be represented.

- **References:** C6, reduced_precision finding 2
- **Verdict / priority:** confirmed-reproduced / P2
- **Exposure:** Public opt-in QuantizationSchemes calibration.
- **Source:** Sources/VectorCore/Quantization/QuantizationSchemes.swift:96; Sources/VectorCore/Quantization/QuantizationSchemes.swift:104; Sources/VectorCore/Quantization/QuantizationSchemes.swift:130
- **Evidence:** Fresh Release NumericProbe results.log: [1,2] reconstructs [1,1], [-10,-1] reconstructs [-9,-0.98823535], constant1.5 reconstructs ~3.57e-43 using the source documented decode (q-offset)*scale.
- **Qualification:** Independent decode is legitimate: same formula appears in analyzeQuantizationError. LinearQuantizationParams is a separate public surface and its existing tests do not establish this one.
- **Improvement:** Consolidate affine calibration, include zero before constraining zero point or retain adequate bias, and define constants. Merely widening zeroPoint to Int32 is not a universal finite-range solution.
- **Acceptance:** Both calibration APIs across signed/one-sided/cross-zero/constant intervals must cover endpoints within a defined quantization step; scalar/SIMD/decoded construction agree.

### NUM-04 — Clamping integer conversions trap before the clamp for finite large intermediates.

- **References:** C6, reduced_precision finding 3
- **Verdict / priority:** confirmed-source / P2
- **Exposure:** Public scalar quantizer and asymmetric parameter construction.
- **Source:** Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:46; Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:54
- **Evidence:** Original executable crash sources inspected: calibrated [-1,1].quantize(1e20) evaluates Int(scaledValue) outside Int range; asymmetric constant1e20 evaluates Int(-128-1e20). Int8/Int32(clamping:) is outermost and never reached. No fresh crash run by this reviewer.
- **Qualification:** Keep finite outliers distinct from NaN/Inf policy and from intentionally unsafe raw buffers. An Int32 clamp avoids traps but can still destroy representation for high offsets.
- **Improvement:** Clamp/validate floating intermediates before integer conversion; define nonfinite and unrepresentable-parameter policy.
- **Acceptance:** Mechanical Float-to-integer conversion matrix: finite extremes/constants/outliers, NaN/Inf, scalar versus SIMD, decoded params; no process termination for supported inputs and explicit error/fallback otherwise.

### NUM-05 — Quantized cosine overflows the product of finite squared norms.

- **References:** C7, reduced_precision finding 5
- **Verdict / priority:** confirmed-reproduced / P2
- **Exposure:** Public INT8 methods at widths512/768/1536, opt-in.
- **Source:** Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:754; Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:793; Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:825
- **Evidence:** Fresh Release NumericProbe results.log: all three widths at repeating1e10 return self-distance1; repeating1 and1e-10 return0. Finite nonzero decoded vector has self cosine distance0 by definition.
- **Qualification:** Not quantization noise. sqrt(qMagSq*cMagSq) overflows even with finite individual squared norms. Fixing only square-root order still leaves scale-squared/product underflow/overflow elsewhere.
- **Improvement:** Cancel positive symmetric scales and evaluate bounded integer cosine with sufficiently wide arithmetic; keep asymmetric fallback and zero policy explicit.
- **Acceptance:** Self/opposite/orthogonal scale-family matrix all widths, asymmetric and unequal-scale cases, zero/nonfinite policy; compare to Double decoded-vector reference.

### NUM-06 — Batch magnitude variance uses cancellation-prone raw moments above N=1000.

- **References:** C8, batch_matrix finding 2
- **Verdict / priority:** confirmed-reproduced / P2
- **Exposure:** Public BatchOperations.statistics; serial route is also inaccurate for large repeated values.
- **Source:** Sources/VectorCore/Operations/BatchOperations.swift:357; Sources/VectorCore/Operations/BatchOperations.swift:367; Sources/VectorCore/Operations/BatchOperations.swift:392; Sources/VectorCore/Operations/BatchOperations.swift:604
- **Evidence:** Fresh Release NumericProbe results.log: alternating10000/10001: std0.49999908 at999 and0 at1000/1024. Repeated Float(1e18): serial std6.25e12; parallel inf. For balanced samples independent population variance=1/4; for odd999, sqrt(499*500)/999.
- **Qualification:** Statistics concerns magnitudes, not component covariance. At1e18 adding1 rounds away, so those inputs really are identical. Merely replacing negative variance clamp does not repair cancellation or squared-sum overflow.
- **Improvement:** Use stable widened mean/M2 for both serial and parallel, with stable merges; define nonfinite magnitude behavior.
- **Acceptance:** Both sides of999/1000 and multiple chunk counts, shuffled order, constants and large offsets; compare Double population standard deviation and assert finite results where reference is representable.

### NUM-07 — Entropy cutoff and Float sum violate finite scale invariance.

- **References:** C9, batch_matrix finding 4
- **Verdict / priority:** confirmed-reproduced / P2
- **Exposure:** Public Vector entropy facade, small and large routes.
- **Source:** Sources/VectorCore/Operations/VectorEntropy.swift:42; Sources/VectorCore/Operations/VectorEntropy.swift:81; Sources/VectorCore/Operations/VectorMath.swift:488
- **Evidence:** Fresh Release NumericProbe results.log: uniform1e-10 returns0 at Dim4/128; unit inputs yield log4/log128; maxFinite yields0/NaN. Independent probabilities are1/d at every nonzero finite common scale.
- **Qualification:** Existing tiny-probability cutoff beside a dominant1 does not justify the absolute absSum cutoff on uniform vectors.
- **Improvement:** Scale by max absolute component or sum in Double; distinguish all-zero from small nonzero. Keep tiny normalized-probability policy separate.
- **Acceptance:** Scale sweeps over finite exponent range and zero-padding across64; compare normalized Double entropy, include all-zero/nonfinite and existing tiny-probability tests.

### NUM-08 — Swift SVD accepts nonorthogonal columns as converged at harmless input scales.

- **References:** C13, linear_algebra LA-1
- **Verdict / priority:** confirmed-reproduced / P2
- **Exposure:** Public SwiftLinearAlgebraProvider; PCA when override selects it, default on unsupported LAPACK platforms.
- **Source:** Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:231; Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:238
- **Evidence:** Fresh Release NumericProbe results.log: s*[1,0,1,1] returns normalized singular spectrum[sqrt2,1] and U correlation0.7071 at1e-12/1e15; unit scale returns golden-ratio spectrum. Exact A-transpose-A eigenvalues=(3±sqrt5)/2.
- **Qualification:** Full rank and condition number2.618: documented null-space-basis limitation is irrelevant. Reconstruction alone can pass with invalid SVD factors.
- **Improvement:** Widen or scale correlation tests and rotations; repair final column norm too. A wider comparison alone does not close normalization or reflector range failures.
- **Acceptance:** Scale-equivariant spectra, orthogonality and normalized residuals for full-rank rectangular matrices; output finiteness; isolate rank-deficiency exception.

### NUM-09 — Swift QR and SVD final normalization lose representable finite vectors through Float squared norms.

- **References:** C13, linear_algebra LA-2
- **Verdict / priority:** confirmed-reproduced / P2
- **Exposure:** Public Swift fallback and PCA using its QR.
- **Source:** Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:46; Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:53; Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:268
- **Evidence:** Fresh Release NumericProbe results.log: QR [s,s] at1e-25 returns Q[1,0],R[s]; at1e20 Q NaN,R-inf. Fresh single-column SVD at1e-25 gives singular0 and tiny unnormalized U; at1e20 singularinf and zero U. Exact norm=sqrt2*abs(s) is representable.
- **Qualification:** Widened norm accumulation immediately narrowed into unsafe reciprocal/alpha-beta arithmetic is insufficient for full exponent range.
- **Improvement:** Scaled/wide Householder and normalization, explicit handling when outputs genuinely cannot be represented.
- **Acceptance:** Shared exponent sweep QR/SVD/eigen with normalized residual, orthogonality, spectrum, finiteness; include single-column and near-extreme finite vectors.

### NUM-10 — Rounding PCA mean before centering doubles variance in a valid offset fixture.

- **References:** README lower-priority PCA centering, linear_algebra LA-4
- **Verdict / priority:** qualified / P3
- **Exposure:** Default LAPACK-backed centered PCA, large-offset near-constant data.
- **Source:** Sources/VectorCore/LinearAlgebra/PCA.swift:160; Sources/VectorCore/LinearAlgebra/PCA.swift:257; Sources/VectorCore/LinearAlgebra/PCA.swift:291
- **Evidence:** Fresh Release NumericProbe results.log: [0,2] produces variance~2, projections[-1,1]; translated exactly representable Float[16777216,16777218] produces mean16777216, variance4, projections[0,2]. Exact mean16777217 and sample variance2.
- **Qualification:** Numerical-quality improvement under broad documented conditioning limitations, not universal relative-accuracy violation. Error term is n/(n-1)*deltaMean*deltaMean^T before further rounding.
- **Improvement:** Retain Double center or Float anchor plus residual mean and apply same centering in fit and both transforms.
- **Acceptance:** Translation/offset invariance with representable input/residual fixtures; compare Double covariance and projected sample mean; no fit/transform center mismatch.

### NUM-11 — UMAP output curve fit depends on units and silently returns its initial guess.

- **References:** C14, umap finding 1
- **Verdict / priority:** confirmed-reproduced / P2
- **Exposure:** Public valid minDist/spread configurations through internal fitter.
- **Source:** Sources/VectorCore/ManifoldLearning/UMAP.swift:314; Sources/VectorCore/ManifoldLearning/UMAP.swift:350; Sources/VectorCore/ManifoldLearning/UMAP.swift:372
- **Evidence:** Fresh Release NumericProbe results.log: fixed minDist/spread0.1: kernel(spread)0.3880566 at1 and0.001; ~1 at1e-5 and1e-20 at1e10. Change variable x=spread*t gives same dimensionless fitting problem with A=a*spread^(2b).
- **Qualification:** Claim concerns kernel calibration, not stochastic layout-scale invariance: learning rate, gradient clipping and init need not rescale the whole layout.
- **Improvement:** Fit dimensionless parameters, detect failed fit, and handle coefficient representability via log/normalized evaluation where needed.
- **Acceptance:** Assert b and dimensionless fitted curve/SSE invariant over scale families and check convergence; explicit failure outside supported range. Do not assert entire SGD layout is scale-equivariant.

### NUM-12 — Smooth-kNN bandwidth search starts at absolute1 and fails outside its fixed64-step range.

- **References:** C14, umap finding 2
- **Verdict / priority:** confirmed-reproduced / P2
- **Exposure:** Public graph-only UMAP input with finite nonnegative distances.
- **Source:** Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:65; Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:87
- **Evidence:** Fresh Release NumericProbe results.log: valid row distances[s,2s] produce nonnearest weight0.584969 at1, weight1 at1e-30, and lost pair at1e30. Equation1+exp(-s/sigma)=log2(3) gives sigma=-s/log(log2(3)-1), ~1.86495s.
- **Qualification:** Use solvable rows for equation assertions; duplicates/tied minimum can make target unattainable and are already handled by floor. Float sigma can itself become unrepresentable at extreme inputs even after bracket fix.
- **Improvement:** Scale row distances or starting bracket; verify residual/convergence and preserve explicit floor behavior.
- **Acceptance:** Multiplicative-scale tests with unequal degrees/isolated/duplicate rows; assert weights/topology and equation residual where solvable, with supported-range policy for sigma overflow.

### NUM-13 — UMAP accepts infinite learning rate/spread at an existing throwing validation boundary.

- **References:** C14, umap finding 3
- **Verdict / priority:** confirmed-reproduced / P2
- **Exposure:** Public UMAPConfig used by both overloads.
- **Source:** Sources/VectorCore/ManifoldLearning/UMAP.swift:102; Sources/VectorCore/ManifoldLearning/UMAP.swift:113
- **Evidence:** Fresh Release NumericProbe results.log: valid3-point graph and learningRate infinity returns [nan,nan,nan,inf,inf,inf]. Comparisons admit positive infinite spread and minDist=spread=infinity by source.
- **Qualification:** NaN generally fails current comparisons. Finiteness checks are necessary but cannot promise every huge finite learning rate yields a finite converged layout.
- **Improvement:** Reject nonfinite tuning fields; document supported ranges or output validation for large finite options.
- **Acceptance:** Table of NaN/±Inf per floating option across both overloads must throw; normal configs remain deterministic; include finite extremes to characterize policy.

### NUM-14 — Approximate equality accepts NaN differences as equal.

- **References:** C16, vectors V7, reduced_precision internal validator
- **Verdict / priority:** confirmed-reproduced / P2
- **Exposure:** Public VectorProtocol approximate equality; PrecisionValidator is internal dormant helper.
- **Source:** Sources/VectorCore/Protocols/VectorProtocol.swift:550; Sources/VectorCore/Operations/Kernels/MixedPrecisionValidation.swift:262; Sources/VectorCore/Operations/Kernels/MixedPrecisionValidation.swift:278; Sources/VectorCore/Operations/Kernels/MixedPrecisionValidation.swift:292
- **Evidence:** Fresh Release NumericProbe results.log: NaN versus42 returns true; internal512 NaN-versus-zero validator true. Same-sign infinity self equality true because differenceNaN. Only diff>tolerance rejects.
- **Qualification:** Matching infinities could be an intentional equality policy; NaN-versus-finite acceptance cannot be justified by tolerance. Negative/NaN/infinite tolerances also need explicit policy.
- **Improvement:** Define shared finite/nonfinite and tolerance rules, then fail closed on undefined differences.
- **Acceptance:** Mechanical helpers matrix finite/NaN/±Inf/signedzero and invalid tolerances across dimensions; distances and vectors use consistent declared rules.

### NUM-15 — safeLog bypasses options for NaN and positive infinity.

- **References:** C16, simd_kernels finding 5
- **Verdict / priority:** qualified / P2
- **Exposure:** Public Vector safeLog; narrower method docs conflict with general options.
- **Source:** Sources/VectorCore/Operations/NaNInfinityHandling.swift:30; Sources/VectorCore/Operations/NaNInfinityHandling.swift:383
- **Evidence:** Fresh Release NumericProbe results.log: strict NaN/+Inf returns nonfinite vector; strict -Inf throws. value<=0 routes are the only ones consulting options.
- **Qualification:** Behavior reproduced, but method-specific options documentation says non-positive values. Classify as inconsistent option semantics requiring resolution, rather than an unqualified promise of finite logarithm for every call.
- **Improvement:** Apply selected nonfinite policy separately from log domain, or expose a distinctly named/domain-only policy and clarify docs.
- **Acceptance:** All presets on negative/zero/NaN/±Inf/mixed vectors; if strict keeps general meaning, strict success must imply finite result.

### NUM-16 — Adaptive distance ignores threshold and candidate range.

- **References:** C19, reduced_precision finding 6
- **Verdict / priority:** confirmed-reproduced / P2
- **Exposure:** Explicit public adaptive API; not default Operations routing.
- **Source:** Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:3388; Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:3403
- **Evidence:** Fresh Release NumericProbe results.log: threshold0 on ones vs1.0001 returns0 versus direct0.0022631169. Zero-query vs70000 yieldsinf while reversed operands yield1583918.9. Source never reads threshold.
- **Qualification:** Default docs disclaim universal FP16 score bounds, but ignored threshold is still ineffective API control. Representability checks on both operands alone do not enforce an arbitrary relative error bound.
- **Improvement:** Define absolute/relative policy and use FP32 fallback whenever requested tolerance cannot be established; or remove misleading parameter and name heuristic.
- **Acceptance:** Operand reversal, high candidate/low query, threshold sweeps and near ties; prove/test conservative selection bounds and compare with defined baseline.

### NUM-17 — Quantized tuner cache excludes accuracyRequirement, making selection call-history dependent.

- **References:** C19, reduced_precision finding 7
- **Verdict / priority:** confirmed-reproduced / P2
- **Exposure:** Public opt-in singleton tuner; no automatic production consumer established.
- **Source:** Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:1389; Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:1401
- **Evidence:** Fresh Release NumericProbe results.log: dot/512/2000 first accuracy0.02=>candidatesINT8, warm0.2=>candidatesINT8, fresh0.2=>fullFP32.
- **Qualification:** Counterintuitive heuristic direction is not independently a bug until accuracyRequirement units/meaning are specified. Quantized tuner lock protects dictionary access; no race alleged here.
- **Improvement:** Include every decision input in key; define accuracy semantics and validate its domain.
- **Acceptance:** Change one input at a time, compare warm/fresh answers, exercise call permutations and invalid accuracy values.

### NUM-18 — Diagnostics misclassifies underflows, rounded boundary values, and observed extrema.

- **References:** reduced_precision additional diagnostics classification
- **Verdict / priority:** confirmed-reproduced / P3
- **Exposure:** Opt-in diagnostics reached by public FP16 construction.
- **Source:** Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:26; Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:132; Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:173
- **Evidence:** Fresh Release NumericProbe results.log: 1e-12 decodes0 but reports512 subnormals/0 underflows;65505 decodes65504 but reports512 overflows; positive-only min remains0. Underflow branch is dominated by broader subnormal predicate.
- **Qualification:** Swapping branches or using65504 as an overflow cutoff is insufficient; classification must follow actual rounded conversion.
- **Improvement:** Classify input and converted bit pattern with explicit terminology; seed extrema from first finite value or empty sentinels.
- **Acceptance:** Scalar/batch classification same on rounding ties, adjacent FP16 normal/subnormal/zero/overflow boundaries, NaNs/infinities and positive/negative-only batches.

### NUM-19 — KernelAutoTuner accepts memory/latency constraints it never uses and persists profiles without provenance.

- **References:** reduced_precision tuner architecture
- **Verdict / priority:** confirmed-source / P3
- **Exposure:** Public opt-in tuner, not automatic Operations dispatch.
- **Source:** Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:41; Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:189; Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:224; Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:592; Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:774
- **Evidence:** Source search: memoryBudget/latencyTarget only stored in workload, key uses dimension/batchSize/precisionMode. Profile serialization contains profileCache only; override dictionary never serialized despite persistence comment. No revision/hardware expiry gate inspected in loadProfiles.
- **Qualification:** Missing hardware identity is stale-profile risk, not reproduced bad dispatch; memoryFootprint is an estimate, not allocation measurement. Tests explicitly bypass profile selection under detected CI/XCTest, so existing passes do not cover ordinary persistence path.
- **Improvement:** Make constraints operative or remove implied guarantees; version profile provenance and persist override if promised.
- **Acceptance:** Temporary-path persistence round trip and non-CI isolated selection test; vary constraints with same shape, require feasible or explicit no-feasible response; invalidate incompatible profiles.

### NUM-20 — Explicit selectMixedPrecisionStrategy maps full precision mode to nonzero tolerance.

- **References:** reduced_precision tuner architecture fp32Full
- **Verdict / priority:** qualified / P3
- **Exposure:** Public opt-in low-level tuner method.
- **Source:** Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:793; Sources/VectorCore/Operations/Kernels/KernelAutoTuner.swift:808
- **Evidence:** Source maps fp32Full to0.0001 then calls mixed tuner, which filters measured synthetic errors and can select reduced precision. Normal selectStrategy heuristic/profile availability respects fullFP32 mode.
- **Qualification:** Do not claim normal fp32Full dispatch automatically quantizes; contradictory behavior is limited to explicit mixed-selection entry point, whose docs expect cpuMixedPrecision workload.
- **Improvement:** Reject incompatible mode or return fullFP32 immediately; clarify method precondition.
- **Acceptance:** Explicit fp32Full call always returns fullFP32 or a documented error; no synthetic benchmark can weaken exact precision policy.

### NUM-21 — Mixed tuner calibrates all5 strategies synchronously on exact workload key and converts candidates inside each reduced execution.

- **References:** reduced_precision calibration cost
- **Verdict / priority:** confirmed-source / P3
- **Exposure:** First opt-in MixedPrecisionAutoTuner selection for a fresh dimension/count/accuracy key.
- **Source:** Sources/VectorCore/Operations/Kernels/MixedPrecisionAutoTuner.swift:155; Sources/VectorCore/Operations/Kernels/MixedPrecisionAutoTuner.swift:269; Sources/VectorCore/Operations/Kernels/MixedPrecisionAutoTuner.swift:419; Sources/VectorCore/Operations/Kernels/MixedPrecisionAutoTuner.swift:909
- **Evidence:** 5 strategies*(5 warmups+20 measurements)=125 timed/warmup executions, plus data generation and FP32 baseline; empty accepted set can add25 FP32 fallback executions. Calibration synchronous inside actor; executeStrategy constructs FP16 SoA per invocation.
- **Qualification:** 125 is normal-path count, not unconditional exact total. Profiles use synthetic data and end-to-end conversion cost; they cannot establish arbitrary input accuracy or preconverted serving latency. No wall-clock regression measured.
- **Improvement:** Bound/offline calibration, coarser workload families or explicit profiles; distinguish preconverted serving and conversion-inclusive costs.
- **Acceptance:** Count calibration work and conversion boundaries mechanically; measure coordinated Release first-call latency/peak allocations on representative workloads; preserve error semantics.

### NUM-22 — FP16 range wrappers map all candidates even for small or empty ranges.

- **References:** PERF4, reduced_precision range wrapper performance
- **Verdict / priority:** confirmed-source / P2
- **Exposure:** Nine public range FP16/mixed wrappers.
- **Source:** Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4666; Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4736; Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4802
- **Evidence:** All nine wrappers construct candidatesFP16.map{$0.storage} before forwarding range; K chunks over N inputs entails O(KN) metadata work in source and O(N) temporary handles per concurrent call.
- **Qualification:** No measured latency/allocation count; optimizer may influence realized overhead. Merely slicing without adjusting range/global output conventions can break indexing.
- **Improvement:** Pass typed candidates directly or process correctly indexed slice; preserve compact output convention.
- **Acceptance:** Empty/single range retained metadata/work independent of unrelated candidate count; all widths and nonzero range starts produce correct scores/sentinels.

### NUM-23 — Source numerical/performance assurances exceed tests or contradict implementation.

- **References:** reduced_precision misleading numerical/performance claims, simd_kernels architecture unsupported claims, linear_algebra limitations 6, batch_matrix opportunity 6
- **Verdict / priority:** confirmed-source / P3
- **Exposure:** Public API comments and internal optimization documentation.
- **Source:** Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:68; Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4968; Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:6; Sources/VectorCore/LinearAlgebra/PCA.swift:14; Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:115; Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift:1343
- **Evidence:** FP16 can merge distinct1/1.0001 values (fresh probe); FP32 accumulation cannot restore lost ordering. Pure Swift robustness disproved NUM08/09. FP16 pool acquire/release explicitly lock; source speed percentages lack named exercising benchmark. PCA exact fallback contradicts no-full-factorization wording.
- **Qualification:** Do not convert unsupported speed claims into claims of measured slowdown; claims of external GPU limitations remain unverified without GPU execution.
- **Improvement:** Remove absolute ranking/robustness claims; tie speedups to hardware/revision/workload and named artifacts; state documented numerical limitations.
- **Acceptance:** Mechanical claim inventory: each guarantee has nearby one-step proof, named test, or explicit unverified-contract wording; performance records carry reproducible provenance.

### NUM-24 — Immutable FP16 vector pool cannot refill storage for conversion of new values via a supported API.

- **References:** reduced_precision FP16VectorPool reuse
- **Verdict / priority:** confirmed-source / P3
- **Exposure:** Public opt-in pool.
- **Source:** Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:319; Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:4983; Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:5012
- **Evidence:** Acquire returns immutable FP16 value; release appends value. Supported construction creates storage, no refill operation. Source comments claim allocation elimination and GC pressure reduction.
- **Qualification:** It can reuse immutable values for repeated computation; claim is specifically unsupported reuse as mutable conversion destination. No allocation speed benefit measured.
- **Improvement:** Clarify purpose or provide explicit mutable owned scratch with scoped fill semantics; avoid changing ownership casually.
- **Acceptance:** A consumer converts distinct successive vectors while demonstrating actual allocation reuse and correct values; otherwise remove conversion-reuse claim.

### NUM-25 — CSR initializer may dereference malformed offsets before throwing validation catches them.

- **References:** umap CSR boundary, README KNNGraph ambiguity
- **Verdict / priority:** qualified / P3
- **Exposure:** Malformed public KNNGraph input outside documented structural caller obligation.
- **Source:** Sources/VectorCore/ManifoldLearning/KNNGraph.swift:42; Sources/VectorCore/ManifoldLearning/KNNGraph.swift:67; Docs/Linear_Algebra_and_Projection.md:113
- **Evidence:** Inspected original probe [0,3,2], neighbors[1,1],distances[1,1],pointCount2. End/count checks pass; first row iterates index2 out of bounds before second row monotonicity check. Prior trap is not freshly rerun here.
- **Qualification:** Not supported valid graph input; docs explicitly prohibit treating constructor as arbitrary parser, while source promises malformed structure throws. Frame as boundary/documentation inconsistency, not ordinary-use corruption.
- **Improvement:** Validate all offsets monotone and bounded before dereferencing; synchronize source/downstream documentation.
- **Acceptance:** Generated small malformed-offset matrix including valid final endpoint but interior overshoot/negative/decrease must throw under chosen checked contract; valid uneven/unsorted rows remain accepted.

### NUM-26 — Accepted extreme integer configuration can trap before throwing or clamping.

- **References:** C15, linear_algebra LA-3, umap extreme integer options
- **Verdict / priority:** qualified / P3
- **Exposure:** Explicit oversized positive PCA/UMAP options on tiny ordinary data.
- **Source:** Sources/VectorCore/LinearAlgebra/PCA.swift:137; Sources/VectorCore/LinearAlgebra/PCA.swift:173; Sources/VectorCore/ManifoldLearning/UMAP.swift:215; Sources/VectorCore/ManifoldLearning/UMAP.swift:225; Sources/VectorCore/ManifoldLearning/UMAP.swift:450
- **Evidence:** PCA k+Int.max overflows before min at k1. UMAP3*Int.max overflows before allocation; negativeSampleRate Int.max is converted to Double rounding to2^63, and due calculation can convert that value back to Int. Original crash sources inspected; no fresh crashes required.
- **Qualification:** Operationally impractical settings; not ordinary-sized failure or giant allocation test. General allocation-safety disclaimer does not explain away the easily checked preallocation addition, but priority is below valid ordinary-input defects. Original UMAP report line229/241 corrected to215/225 for n*dimensions.
- **Improvement:** Clamp oversampling before addition or typed rejection; checked dimensions products; bound/check sample counts before floating-to-Int conversion.
- **Acceptance:** Tiny fixtures atInt.max, safe bounds, zero/negative/normal values either throw documented error or produce bounded intended result without process termination.

### NUM-27 — Exact brute-force kNN builder is exhaustive over computed Gram scores, not a reliable numerical distance oracle.

- **References:** umap reference builder Gram cancellation
- **Verdict / priority:** qualified / P3
- **Exposure:** Public reference KNNGraph.bruteForce.
- **Source:** Sources/VectorCore/ManifoldLearning/KNNGraph.swift:135; Sources/VectorCore/ManifoldLearning/KNNGraph.swift:166; Docs/Numerical_Behavior.md:86
- **Evidence:** Fresh Release NumericProbe results.log: [10000,10002,10001],k1=>neighbors[1,0,0], all distances0. Direct Float-exact differences give row0 nearest2 at distance1, row1 nearest2.
- **Qualification:** Documented Gram cancellation elsewhere; do not present as novel UMAP algorithm defect. Broad numerical docs apply; exact label should not imply stable reference scores.
- **Improvement:** Link conditioning caveat; if advertised as numerical oracle, use direct differences or sufficiently comprehensive refinement.
- **Acceptance:** Near-offset fixtures compared with widened/direct reference; any refinement must not lose true nearest candidate before reranking.

### NUM-28 — UMAP initial coordinates are shape checked, not validated for finiteness or safe scale.

- **References:** umap explicit initial-coordinate values
- **Verdict / priority:** qualified / P3
- **Exposure:** Explicit caller-supplied initialization.
- **Source:** Sources/VectorCore/ManifoldLearning/UMAP.swift:211; Sources/VectorCore/ManifoldLearning/UMAP.swift:215; Sources/VectorCore/ManifoldLearning/UMAP.swift:471
- **Evidence:** Source copies coordinates verbatim; NaN differences fail positive squared guard and remain NaN; Float squared differences at1e30 overflow. Original NaN/huge probes inspected, no new initialization run.
- **Qualification:** No all-finite or conditioning guarantee; do not generalize to ordinary random/PCA initialization.
- **Improvement:** Specify responsibility, reject nonfinite values in checked entry point, optionally normalize supported initialization.
- **Acceptance:** NaN/±Inf coordinate tests and large finite scale fixtures with documented result/error policy; ordinary init deterministic.

### NUM-29 — UMAP and PCA expose material in-memory storage costs at large corpus/sample size.

- **References:** PERF8, umap graph memory, linear_algebra limitations 2
- **Verdict / priority:** confirmed-source / P3
- **Exposure:** Public in-memory UMAP/PCA; default sizes/workloads determine exposure.
- **Source:** Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:105; Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:126; Sources/VectorCore/ManifoldLearning/UMAP.swift:412; Sources/VectorCore/ManifoldLearning/UMAP.swift:425; Sources/VectorCore/LinearAlgebra/PCA.swift:148; Sources/VectorCore/LinearAlgebra/PCA.swift:179
- **Evidence:** UMAP original edges12F bytes plus retained optimizer8R indices+32R four Double arrays=>12F+40R logical payload. AtF=R=150M =>7.8GB decimal, before CSR/input/coordinates. PCA500000*1536*4=3.072GB sample payload; exact SVD also returns discarded U.
- **Qualification:** F/R depend on reciprocity/pruning; staging/sort slack and ARC lifetimes vary. This is a byte model, not measured RSS. In-memory design explicitly documented.
- **Improvement:** Budget/report logical bytes, reduce duplicate graph arrays, consider right-factor-only exact PCA and bounded transform APIs; profile before promising savings.
- **Acceptance:** Measured peak allocation/RSS alongside predicted payload for representative Release runs; verify identical results/order or explicitly documented numerical changes.

### NUM-30 — Graph sorts/epoch scans and PCA per-point transforms offer concrete allocation/work opportunities.

- **References:** umap sorts/epoch scans, linear_algebra limitations 3
- **Verdict / priority:** confirmed-source / P3
- **Exposure:** Public synchronous algorithms; no throughput regression measured.
- **Source:** Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:123; Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:146; Sources/VectorCore/ManifoldLearning/UMAP.swift:439; Sources/VectorCore/LinearAlgebra/PCA.swift:257; Sources/VectorCore/LinearAlgebra/PCA.swift:291
- **Evidence:** Canonical and fused sorts O(E logE+F logF);64-pass calibration O(64E), optimizer scansR every epoch. Scalar transforms allocate outputs and use Float accumulation; batch output builds per-row arrays.
- **Qualification:** Not missing ANN/GPU functionality; serial deterministic and in-memory contract is explicit.
- **Improvement:** Evaluate compact graphs/scheduled edges/flat output views from consumer needs; preserve ordering and seed behavior.
- **Acceptance:** Coordinated Release time/allocation profiles and result/seed reproducibility checks for any schedule or format change.

### NUM-31 — Long synchronous manifold/factorization work lacks cancellation checkpoints.

- **References:** umap no cancellation, linear_algebra limitations 4
- **Verdict / priority:** qualified / P3
- **Exposure:** Callers using synchronous APIs, including inside async tasks.
- **Source:** Sources/VectorCore/ManifoldLearning/UMAP.swift:439; Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift:46; Sources/VectorCore/LinearAlgebra/PCA.swift:189; Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift:30
- **Evidence:** Inspected loops have no cancellation hooks; synchronous actor/executor caller cannot interrupt their work via task cancellation alone.
- **Qualification:** No broken cancellation guarantee is documented. Large positive powerIterations requests more work intentionally.
- **Improvement:** Consider throwing cancellation/progress overload for long jobs, with bounded polling cost and deterministic unchanged no-cancel behavior.
- **Acceptance:** Cancel at meaningful phases and bound latency/resources; legacy synchronous success output remains consistent.

### NUM-32 — C INT8 accumulation can exceed signed32 range only beyond supported fixed-vector dimensions.

- **References:** simd_kernels C arithmetic policy, reduced_precision INT8 strengths
- **Verdict / priority:** qualified / P3
- **Exposure:** Generic exported C lanes API; optional Swift C INT8 call sites use fixed512/768/1536.
- **Source:** Sources/VectorCoreC/common/vc_scalar.c:43; Sources/VectorCoreC/arm64/vc_arm64.c:116; Sources/VectorCoreC/x86_64/vc_x86.c:47; Sources/VectorCoreC/common/vc_dispatch.c:78
- **Evidence:** Worst signed Int8 product16384;131072 terms sum2147483648 outside Int32. Maximum1536 terms=25165824 fits. Scalar C signed+= overflows for generic large input; Swift fallback wrapping arithmetic explicit. Tail bounds inspected: complete blocks then i<lanes remainder.
- **Qualification:** Do not allege active fixed-width overflow. C intrinsic lane behavior differs from scalar C signed-overflow language semantics; no Intel/NEON overflow execution or UBSan run.
- **Improvement:** Specify representable-sum bound or widen/define accumulator for generic lanes before broadening routing.
- **Acceptance:** All architectures/C gate configurations compare max-magnitude fixed widths and remainder counts to Int64 reference; arbitrary supported lanes must have explicit overflow behavior.

### NUM-33 — C header range indexing wording and proposed unimplemented declarations need precise public status.

- **References:** simd_kernels architecture C API hygiene
- **Verdict / priority:** qualified / P3
- **Exposure:** Public C/header and Swift wrapper interface; no current Swift link failure established.
- **Source:** Sources/VectorCoreC/include/VectorCoreC.h:1; Sources/VectorCoreC/common/vc_scalar.c:39; Sources/VectorCore/Interop/CKernels.swift:1
- **Evidence:** Inspected scalar/NEON/x86 range implementations output compact0..<end-start; Swift wrapper follows compact convention. Proposed symbols are marked proposed, and FP32 wrappers have no established production consumers.
- **Qualification:** Header declaration of proposed symbol is API-hygiene risk, not proof a normal build fails.
- **Improvement:** Align range docs and segregate proposal declarations; preserve compact convention.
- **Acceptance:** C consumer symbol/link and nonzero-start sentinel tests; supported public declarations all link.

### NUM-34 — Allocation-heavy vector helpers, unequal SIMD reduction structure and platform heuristics warrant measurement, not speed claims.

- **References:** simd_kernels architecture allocation/heuristics
- **Verdict / priority:** qualified / P3
- **Exposure:** Public VectorMath helpers; internal PlatformConfiguration and mostly unused optimization helpers.
- **Source:** Sources/VectorCore/Operations/VectorMath.swift:1; Sources/VectorCore/Platform/SwiftDoubleSIMDProvider.swift:1; Sources/VectorCore/Platform/PlatformConfiguration.swift:1; Sources/VectorCore/Optimization/OptimizationAttributes.swift:1
- **Evidence:** Original locations inspected: L1/variance create full arrays; Double reduces SIMD chunks horizontally while Float retains lanes. Compile-time architecture branches assume hardware traits; named likely/prefetch helpers do not establish compiler behavior.
- **Qualification:** No universal measured slowdown; no illegal instruction dispatch is proven from these width/cache hints.
- **Improvement:** Profile allocations/generated code before fusion/tuning changes; replace hardware facts with explicit heuristics/runtime facts where needed.
- **Acceptance:** Representative Release allocation/throughput and generated-code records, matched semantics, supported-platform conformance.

### NUM-35 — LAPACK dimension ceiling does not by itself establish extreme workspace arithmetic safety.

- **References:** linear_algebra limitations 7
- **Verdict / priority:** qualified / P3
- **Exposure:** Shapes far beyond documented small-panel operating envelope.
- **Source:** Sources/VectorCoreC/common/vc_lapack.c:99; Sources/VectorCore/LinearAlgebra/LinearAlgebraProvider.swift:1
- **Evidence:** C8*minmn occurs before size_t cast; optimal workspace may exceed32-bit with otherwise valid dimensions. No practical-size failure reproduced.
- **Qualification:** Do not elevate to verified practical memory-corruption defect. Two Int32 dimensions multiply safely in64-bit Swift Int; workspace arithmetic is distinct.
- **Improvement:** Checked workspace/index sizes if arbitrary-shape support expands; retain small-panel caveat.
- **Acceptance:** Boundary size arithmetic tested without giant allocations; supported envelope documented and clean error outside it.

### NUM-36 — Alignment, FP16 cache identity/capacity/retention and quantized storage shape/lifetime claims are assigned to ownership validation.

- **References:** C4, simd_kernels finding 2, A4, reduced_precision finding 1, reduced_precision cache capacity, reduced_precision ownership contract, reduced_precision invariant gap, PERF8 FP16 cache
- **Verdict / priority:** unverified / cross-reference
- **Exposure:** See ownership ledger.
- **Source:** Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift:419; Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift:26; Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift:171; Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift:344
- **Evidence:** Source read for numerical context; this row records complete crosswalk and delegates final evidence/verdict to validate_ownership.
- **Qualification:** Do not double-count this cross-reference as an unverified finding; replace with owning ledger adjudication.
- **Improvement:** Use owning reviewer improvements.
- **Acceptance:** Use owning reviewer acceptance criteria.

### NUM-37 — Diagnostics synchronization, loop widths, full-sort helper and metric factory claims are assigned to API/execution validation.

- **References:** C18, reduced_precision finding 4, simd_kernels finding 3, simd_kernels finding 6, simd_kernels smaller factory issue, PERF5 SIMD full sort
- **Verdict / priority:** unverified / cross-reference
- **Exposure:** See API/execution ledger.
- **Source:** Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift:103; Sources/VectorCore/Optimization/OptimizationAttributes.swift:91; Sources/VectorCore/Operations/VectorMath.swift:386; Sources/VectorCore/Operations/OptimizedDistanceMetrics.swift:402
- **Evidence:** Mechanisms source-read and routed to API/execution reviewer; numerical diagnostics labels separately validated NUM18.
- **Qualification:** Do not double-count delegated rows.
- **Improvement:** Use owning reviewer improvements.
- **Acceptance:** Use owning reviewer acceptance criteria.

## Exhaustive component subsection crosswalk

- **simd_kernels findings1–6:** NUM02, ownership cross-reference NUM36, execution cross-reference NUM37, NUM01, NUM15, NUM37 respectively. Smaller factory issue: NUM37. Architecture paragraphs: three numerical surfaces NUM01/02; VectorMath temporaries and Double lane reduction NUM34; hardware assumptions/helper names NUM34; C dispatch/tails/header NUM32/33; generic INT8 overflow NUM32. Strengths on bounds, fixed widths/tails, operand order and array initialization are local source observations, not a universal arithmetic guarantee; Accelerate division operand order is correct but its numerical subnormal result still fails NUM01. Rejected cases—API-specific cosine zero/NaN behavior, direct sum-square/GEMM limits, unsafe pointer counts, internal EdgeCaseHandler, and deep-subnormal finite-only normalization—remain qualified limitations (NUM02/27 and factual appendix below).
- **reduced_precision principal1–7:** NUM36,NUM03,NUM04,NUM37,NUM05,NUM16,NUM17. Additional bullets in original order: NUM18;NUM36;NUM22;NUM36;NUM36;NUM36;NUM14;NUM19/20;NUM21;NUM23/24. Strengths/rejections: FP16 immutable/length checks and scoped kernel borrows delegated ownership; rounding implementation and remainder loops have no newly established defect; supported INT8 widening bounds NUM32; asymmetric/differing-scale fallback source-confirmed; NormCache owned arrays and absent callers a local source observation; cache actor and tuner/pool lock statements concern storage synchronization only, not cache identity or diagnostics. Unsafe undersized outputs remain caller violations. FP16 representability is not a score/ranking guarantee (NUM16/23).
- **linear_algebra LA1–4:** NUM08,NUM09,NUM26,NUM10. Seven structural strength paragraphs: column-major/transposition and R-before-Q organization source-consistent; scoped C pointers and positive/Int32 shape checks have no newly established defect; eigen overflow guard throws; PCA QR/provider capture layout source-consistent; existing unit-scale invariant tests do not close NUM08/09; malformed Codable model concern inapplicable because no Codable/public initializer. Limitations1–9 map respectively to factual null-basis limitation;NUM29;NUM30;NUM31;factorization/GEMM factual boundary;NUM23;NUM35;documented nonfinite/allocation boundary;documented n-1 denominator. No claim of exhaustive fresh backend verification.
- **umap findings1–3:** NUM11/12/13. Boundary paragraphs CSR, integer, Gram reference, explicit init: NUM25/26/27/28. Performance/architecture bullets memory, sorts/scan, cancellation, package seam: NUM29/30/31 and factual seam appendix. Eight strengths/rejections remain local supported observations: immutable graph arrays/scoped mutation; unsorted/uneven rows; self-free log2(k+1); zero-distance floor; t-conorm max-merge; seeded scheduling and endpoint updates; existing basic determinism/quality tests; no new cross-backend/corpus guarantee.
- **batch_matrix assigned findings2 and4:** NUM06/07. All other numbered findings and batch performance observations stay with root/other validators; source performance claim discipline was included NUM23. This reviewer does not independently adjudicate those other batch findings.
- **vectors assigned V2 andV7:** NUM01/02 andNUM14. Other vector claims stay with ownership/API validators.

## Factual limitations and rejected overclaims

1. Neither finite input nor correct shape implies arbitrary numerical stability. Documented API-specific zero/nonfinite cosine and Gram cancellation remain existing limitations. They do not excuse a disabled tolerance argument or an incorrect scale reconstruction under an advertised normalization operation.
2. Swift SVD zero-singular-value basis completion is explicitly absent; do not combine that known rank-deficient limitation with NUM08 full-rank failure. PCA is in-memory and uncentered explained variance explicitly divides by n-1; no new factor-of-n defect established. Setting factorization provider does not override separate GEMM routing.
3. UMAP self-free graph target is log2(k+1); for the tested degree2 row this produces an independently solvable equation. Duplicate minima may make target unattainable, isolated rows have no edges, and duplicate directed entries affect calibration before max merge. No deduplication invariance is promised.
4. Core graph input is directed CSR; fuzzy construction owns rho/sigma and t-conorm, and downstream Index builds graph/distances. No Core-to-GPU automatic UMAP route was found in scoped search. External Metal/GPU numerical correctness and broad CPU/GPU parity are **unverified** by this pass; downstream StableNormalization documentation demonstrates deliberate policy divergence, not an executed GPU result.
5. C architecture source has complete-block/tail loops and runtime dispatch gates, but no Intel/NEON hardware parity campaign was run. Generic arbitrary-lane arithmetic policy needs NUM32 if exposure expands. No illegal-instruction claim follows merely from heuristic width constants.
6. No named benchmark evidence supports portable fixed speedup percentages; no claims of universal slowdown or speedup are made. Byte estimates count logical payload and must not be reported as measured RSS.
