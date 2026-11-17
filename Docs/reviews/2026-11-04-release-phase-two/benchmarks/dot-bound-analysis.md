# Dot policy analysis

Independent observations exposed three seed-0 failures against the existing empirical relative limit of `1e-6`: `dot.512.generic` (absolute error `3.659607699546541e-6`), `dot.1536.generic` (`2.2157493759777935e-6`) and `dot.1536.optimized` (`1.8075641463610737e-6`). These are observations, not evidence that the kernels violate a forward-error contract. Seed 1 was separately executed for transport inspection and passed six cases; it does not close the seed-0 gate.

For exact inputs `a_i`, `b_i`, put `t_i = a_i b_i`, `s = Σ t_i` and `A = Σ |t_i|`. The standard rounded-operation model gives `|c-s| ≤ γ32(k32) A`, with `γp(k)=ku_p/(1-ku_p)`. The independent Double product-and-sum reference gives `|r-s| ≤ γ64(k64) A`. Therefore `|c-r| ≤ (γ32(k32)+γ64(k64)) A`. Cancellation prevents replacing `A` by `|s|`. The existing policy supports the absolute rule with the fixture bound `A ≤ D B²`, where `B=1`.

The actual source routes determine the counts:

- `VectorProtocol.dotProduct` multiplies each pair and adds sequentially from exact zero: one product plus at most `D-1` rounded additions, so `k32=D`.
- `DotKernels.dot` puts `D/16` products in each scalar accumulator, then combines four accumulators with two additions and reduces four lanes with at most three additions. Thus `k32=1+(D/16-1)+2+3=D/16+5`. The stride of 16 is in SIMD4 lanes: four updates per accumulator per stride. This is not `D/64`.
- `DoubleRef.dot` is a product and sequential sum, conservatively `k64=D`. Products of two finite Float inputs are exact in Double here, so this overcounts the reference product rounding by one.

These are bounded-fixture rules: the LCG coordinates are multiples of `2^-23` in `[-1,1)`. Nonzero products and intermediate results cannot underflow binary32; their quantum is at least `2^-46`, and their magnitude is bounded by `D≤1536`. Zero results are exact. The bound assumes ordinary round-to-nearest arithmetic; fused multiplication/addition only reduces the number of rounding factors. It does not cover arbitrary externally supplied vector magnitudes.

| D | Route | k32 | k64 | Absolute bound including existing evaluation margin | Observed max absolute error |
|---|---|---:|---:|---:|---:|
| 512 | generic | 512 | 512 | 0.015625476880828604 | 3.659607699546541e-6 |
| 512 | optimized | 37 | 512 | 0.0011291529099318428 | 1.108763882484709e-6 |
| 768 | generic | 768 | 768 | 0.03515785946459697 | 1.355058586227642e-6 |
| 768 | optimized | 53 | 768 | 0.002426155190759389 | 5.522900465848579e-7 |
| 1536 | generic | 1536 | 1536 | 0.1406378760441469 | 2.2157493759777935e-6 |
| 1536 | optimized | 101 | 1536 | 0.00924688210068572 | 1.8075641463610737e-6 |

The dimension-only bound is conservative. Seed-0 sums of absolute products are 135.0342, 198.3812 and 371.9118 instead of the bounds 512, 768 and 1536. A fixture-specific bound would require carrying extra scale identity through the policy; this proposal uses the existing case-name-based model without adding a schema or run-seed-dependent rule.

Sensitivity was measured independently by replacing the ideal exact result with `Float(r-t_i)` (omission) or `Float(r-2t_i)` (sign error) for every coordinate, then checking the proposed absolute bound. This isolates the injected error; it is not a distribution of real implementation defects or a proof of arbitrary error detection.

| D | Route | Omission missed / D | Sign error missed / D | Largest absolute term omission detected |
|---|---|---:|---:|---|
| 512 | generic | 50/512 | 30/512 | yes |
| 512 | optimized | 7/512 | 6/512 | yes |
| 768 | generic | 113/768 | 69/768 | yes |
| 768 | optimized | 17/768 | 8/768 | yes |
| 1536 | generic | 654/1536 | 400/1536 | yes |
| 1536 | optimized | 71/1536 | 39/1536 | yes |

Raw analysis and script are retained as `benchmarks/dot-analysis.json`, `analyze-dot.py` and the bounded `dot-bound-analysis.*` command artifacts under `/private/tmp/vectorcore-release-phase-two-2026-11-04`. The original failing reports remain intact; the actual comparator refused them with exit 2 even for self-comparison.

The coordinator reviewed and authorized this policy repair. The six rules are implemented by source commit `73b7e7d0638dd0335dae963613a5a6ee1d090ec6`; the original failures remain retained. `testDotModelsUseSignedSumAbsoluteBoundsAtEveryWidth` pins counts/domains/bounds and rejects largest-product omissions and above-bound perturbations. `testEveryDotRouteRetainsEachSignedCoordinateExactly` pairs signed one-hot vectors at every index of each actual generic/optimized width and rejects deliberately omitted contributions with exact bit comparisons, independently of the random-fixture bound. Final seed-0 strict execution passes all six dot routes under these reviewed rules.
