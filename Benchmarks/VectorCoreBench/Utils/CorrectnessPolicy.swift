import Foundation
import VectorCoreBenchmarking

// MARK: - Rounding-error constants

/// Constants of the standard model of floating-point arithmetic: each basic operation and each square root returns
/// `fl(x op y) = (x op y)(1 + δ)` with `|δ| <= u` (round to nearest, no underflow or overflow). Notation and lemmas
/// follow Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., section 2.2, Lemmas 3.1 and 3.3.
enum RoundingBound {
    /// Unit roundoff of binary32 (`Float`), `u = 2^-24`.
    static let binary32 = 0x1p-24
    /// Unit roundoff of binary64 (`Double`), `u = 2^-53`.
    static let binary64 = 0x1p-53

    /// `γ_k = k·u / (1 − k·u)`. Lemma 3.1: if `|δ_j| <= u`, `ρ_j = ±1` and `k·u < 1`, then
    /// `∏_{j=1..k} (1 + δ_j)^ρ_j = 1 + θ_k` with `|θ_k| <= γ_k`. Lemma 3.3: `γ_a + γ_b + γ_a·γ_b <= γ_{a+b}`.
    static func gamma(_ k: Int, unitRoundoff u: Double) -> Double {
        precondition(k >= 0 && Double(k) * u < 1, "gamma requires 0 <= k and k*u < 1")
        return Double(k) * u / (1 - Double(k) * u)
    }

    /// Relative margin on every derived threshold. It exceeds the binary64 roundings of the gate's own error
    /// computation (a subtraction and a division) and of evaluating the threshold expression (fewer than ten
    /// operations): at most twelve factors `(1 + δ)`, `|δ| <= 2^-53`, so less than `1.4e-15` relative.
    static let evaluationMargin = 0x1p-40
}

// MARK: - Derived error models

/// The worst-case error of one benchmark route, derived from the route's actual summation.
///
/// `k32` is the largest number of binary32 rounding factors any term of the result passes through, so the computed
/// FP32 result is `ĉ = Σ t_i (1 + θ_i)` with `|θ_i| <= γ32(k32)` (Lemma 3.1); `k64` is the same count for the
/// benchmark's Double reference `r̂`. See `CorrectnessPolicy` for each route's count.
struct ErrorModel: Equatable {
    enum Kind: Equatable {
        /// Every term `t_i` is nonnegative, so `Σ|t_i| = r` and `|ĉ − r| <= γ32(k32)·r`, `|r̂ − r| <= γ64(k64)·r`.
        /// Then `|ĉ − r̂| / max(|r̂|, f) <= |ĉ − r̂| / r̂ <= (γ32(k32) + γ64(k64)) / (1 − γ64(k64))` for every floor `f`
        /// (and `ĉ = r̂ = 0` when `r = 0`, since every term is then exactly zero). The gate checks `maxRelError`.
        case relative
        /// The error scales with `scale >= Σ|t_i|` (and `scale >= r`), not with `r`:
        /// `|ĉ − r̂| <= (γ32(k32) + γ64(k64))·scale`. The gate checks `maxAbsError`; no relative error is consulted.
        case absolute(scale: Double)
    }

    /// Model id, printed with a violation.
    let route: String
    let kind: Kind
    let k32: Int
    let k64: Int

    /// The gate threshold implied by the derivation (see `Kind`), including `RoundingBound.evaluationMargin`.
    var threshold: Double {
        let g32 = RoundingBound.gamma(k32, unitRoundoff: RoundingBound.binary32)
        let g64 = RoundingBound.gamma(k64, unitRoundoff: RoundingBound.binary64)
        switch kind {
        case .relative: return (g32 + g64) / (1 - g64) * (1 + RoundingBound.evaluationMargin)
        case .absolute(let scale): return (g32 + g64) * scale * (1 + RoundingBound.evaluationMargin)
        }
    }
}

/// The threshold one validated case is held to (`CorrectnessPolicy.rule(forCaseName:)`), and where it comes from.
struct CorrectnessRule: Equatable {
    enum Basis: Equatable {
        case derived(ErrorModel)
        case empirical(reason: String)
    }
    let maxRel: Double?
    let maxAbs: Double?
    let basis: Basis

    var summary: String {
        switch basis {
        case .derived(let model): return "derived \(model.route) k32=\(model.k32) k64=\(model.k64)"
        case .empirical: return "empirical"
        }
    }
}

// MARK: - Policy

/// The benchmark correctness gate: which cases must carry a validation, and the threshold each validated case is held to.
///
/// **Derived thresholds (T11).** A route whose summation is read below gets the threshold its worst-case rounding bound
/// implies for the case's actual dimension `D` and candidate count `N` (both parsed from the case ID). Notation:
/// `d_i = a_i − b_i`, `r = Σ d_i^2`, `u32 = 2^-24`, `u64 = 2^-53`; each count below is the most rounding factors one
/// term `d_i^2` passes through.
///
/// - `generic-euclid2` (`batch.euclidean2.D.N.generic.*`): `Vector<D>.euclideanDistanceSquared(to:)` is the
///   `VectorProtocol` default `(self − other).magnitudeSquared`, where `magnitudeSquared = magnitude * magnitude` and
///   `magnitude = M · sqrt(Σ fl(d_i / M)^2)` with `M = max |d_i|`, summed left to right in FP32. Per term: the
///   difference (2, it is squared), the division by `M` (2), the product (1), the recursive sum (at most `D − 1`),
///   the square root (2, it is squared), the product with `M` (2) and the final square (1): `k32 = D + 9`. `M`
///   cancels exactly. Root cause of the pre-T11 failures: the old fixed `1e-6` (about `17·u32`) lies far below this
///   bound (`γ32(D + 9) ≈ 3.1e-5` at D = 512, `9.2e-5` at D = 1536), so rounding that this recursive FP32 sum can
///   legitimately produce exceeded it. The T11 evidence records per-distance errors of 1.04e-06 to 2.31e-06 on the CI
///   fixtures (an observation outside this repository; the test below prints the values and asserts only the bound).
/// - `vdsp-euclid` (`batch.euclidean.D.N.generic.*`): `EuclideanDistance().distance` is `sqrt(vDSP_distancesq)`. The
///   accumulation order inside Accelerate is not visible; the count assumes only that `vDSP_distancesq` evaluates
///   `Σ (a_i − b_i)^2` with binary32 operations in some order (an unverified contract about Accelerate): difference
///   (2), product (1), additions (at most `D − 1`), so `D + 2` for the square, and `sqrt(1 + θ) = 1 + θ'` with
///   `|θ'| <= |θ|` plus the square root's own rounding: `k32 = D + 3`.
/// - `optimized-euclid2` (`batch.euclidean2.D.N.optimized.*` and the boundary `aos-fp32-*` and batch
///   `consumer-only` entries): `BatchKernels.range_euclid2_D` (and the odd-tail `EuclideanKernels.squaredD`) keep 4
///   `SIMD4<Float>` accumulators, so each of the 16 scalar lanes sums `D/16` terms starting from zero (`D/16 − 1`
///   rounded additions), then `(acc0 + acc1) + (acc2 + acc3)` (2) and `sum()` of 4 lanes (at most 3 in any order),
///   plus the difference (2) and the product (1): `k32 = D/16 + 7`.
/// - `optimized-euclid` (`batch.euclidean.D.N.optimized.*`): `range_euclid_D` is `range_euclid2_D` followed by
///   `applySqrt`: `k32 = D/16 + 8`.
/// - `soa-euclid2` (`batch.euclidean2.D.N.optimized-soa`, boundary `soa-fp32-*`): `BatchKernels_SoA.euclid2_D` (2-way,
///   4-way and tiled kernels alike) keeps one `SIMD4<Float>` accumulator per candidate over `D/4` lanes with
///   `addProduct` (one rounding per step, fused or not the count is the same), then `sum()` (at most 3), plus the
///   difference (2): `k32 = D/4 + 5`.
/// - Provider routes (`.sequential`, `.parallel`, `.automatic`, `.injected`) also compare the provider's FP32 sum of
///   the `N` results with the sum of the references. A sum of `N` nonnegative terms in any order (`parallelReduce`
///   chunks combined from an exact `0`) adds at most `N − 1` factors to each term, so the observation's count is
///   `k32 + N − 1` (and `k64 + N − 1`). Where a case records both comparisons they are merged into one statistic
///   (`ProviderObservations.combining`), so its per-distance comparison is held to the larger, observation bound. The
///   512-wide `runBatchOptimizedEuclid` overload (`batch.euclidean.512.N.optimized.*`) records only the observation, so
///   the per-distance outputs of that route are not gated at all (a gap in a fingerprinted body, routed); the 768 and
///   1536 overloads merge both. The empirical cosine preNorm routes likewise record only the observation.
/// - References: `DoubleRef.euclid2` and the matrix suite's reference loop are a binary64 difference (2), product (1)
///   and recursive sum (`D − 1`): `k64 = D + 2`; `DoubleRef.euclid` adds a square root: `k64 = D + 3`.
/// - `matrix-norm-identity` (`matrix.euclidean2.D.N.optimized`, `…consumer-only`): `MatrixDistance` computes
///   `fl(fl(g + n_x) + n_y)` clamped at 0, with `g = −2·x·y` from `cblas_sgemm` and squared norms `n_x`, `n_y` from
///   `vDSP_svesq`, both assumed to be binary32 inner products of length `D` in some order (an unverified contract
///   about Accelerate; a fast-matrix-multiplication algorithm would void it). Each of the three terms passes through
///   at most `D + 2` factors, so `|ĉ − r| <= γ32(D + 2)·(n_x + n_y + 2·Σ|x_k·y_k|) <= γ32(D + 2)·4·D·B^2` termwise,
///   with `|x_k|, |y_k| <= B = fixtureCoordinateBound` (and `r <= 4·D·B^2` too); the clamp only moves a negative value
///   toward `r >= 0`. The error does not scale with `r`: a zero diagonal reference still carries rounding error of the
///   size of the norms, which is the zero-reference class (relative error against `|ref| = 0`), so this route is gated
///   on absolute error.
///
/// The counts are pinned by `BenchmarkCorrectnessPolicyTests.testDerivedThresholdsFollowTheRoundingCounts`; that the
/// bounds hold on the actual route outputs by `testGenericSquaredEuclideanBoundHoldsAndStillCatchesDroppedWork`,
/// `testOptimizedAndSoABoundsHoldOnFixtures`, `testMatrixZeroReferenceIsGatedOnAbsoluteError` and
/// `testCIRecipeObservationsPassAndPerturbationsFail`. What those tests show the thresholds still catch at the CI
/// recipe's sizes is narrow: the generic route with the LARGEST squared difference of one distance dropped, the summed
/// observation with its SMALLEST candidate dropped (`testGenericSquaredEuclideanBoundHoldsAndStillCatchesDroppedWork`),
/// and a matrix entry off by twice the bound or read from the wrong index. A small defect in a single term or a single
/// candidate can pass. The T11 review's single-host analysis of the run-seed-0 fixtures (`T11/review/tools/sensitivity.py`,
/// outside this repository; not verified by a test) found, at the CI sizes, a dropped term at a random coordinate of
/// one distance missed in 5-48% of placements and a sign error in one coordinate missed in 0.5-28%; a dropped term in
/// one matrix entry missed in 23-61%; and at N = 10000 a dropped smallest candidate of the summed observation missed.
/// The bounds are worst cases, linear in the summation length; that typical rounding errors grow only like its square
/// root is a heuristic (not verified here), which is why measured errors sit far below them.
///
/// **Empirical thresholds.** Every other validated route keeps the pre-T11 fixed thresholds (FP32 relative `1e-6`, FP16
/// relative `2e-3`, FP16 cosine `3e-3`). They are not derived from those routes' summations and are not verified to
/// bound their rounding error; `testEveryValidatedCatalogRouteIsDerivedOrListedEmpirical` pins the exact list.
///
/// **Validation requirement.** A case without a correctness record fails ("Missing validation") unless it is declared
/// not applicable here; `testValidationRequirementIsDeclaredPerCase` pins the declarations.
enum CorrectnessPolicy {
    /// The widths whose kernels the derivations above read.
    static let modeledDimensions: Set<Int> = [512, 768, 1536]

    /// Every benchmark fixture coordinate lies in [-1, 1): `InputFactory.randomArray` maps a 24-bit integer `x` to
    /// `-1 + 2·(x / 2^24)` by default. Pinned by `BenchmarkCorrectnessPolicyTests.testFixtureCoordinatesLieWithinTheStatedBound`.
    static let fixtureCoordinateBound: Double = 1

    /// Pre-T11 fixed thresholds, kept only for routes without a derived model.
    static let fp32EmpiricalRelative = 1e-6
    static let fp16EmpiricalRelative = 2e-3
    static let fp16CosineEmpiricalRelative = 3e-3

    enum ValidationRequirement: Equatable {
        case required
        case notApplicable(reason: String)
    }

    /// The derived model of a case, or nil when its route has none (its rule is then empirical).
    static func errorModel(forCaseName name: String) -> ErrorModel? {
        let p = CaseParsing.parse(name: name)
        guard let kind = p.kind, let metric = p.metric, let variant = p.variant, let dim = p.dim, let n = p.n, n > 0,
              modeledDimensions.contains(dim) else { return nil }
        let observed = p.provider != nil
        let observation = observed ? n - 1 : 0
        func relative(_ route: String, _ k32: Int, _ k64: Int) -> ErrorModel {
            ErrorModel(route: route, kind: .relative, k32: k32 + observation, k64: k64 + observation)
        }
        switch (kind, metric, variant, observed) {
        case ("batch", "euclidean2", "generic", true): return relative("generic-euclid2", dim + 9, dim + 2)
        case ("batch", "euclidean", "generic", true): return relative("vdsp-euclid", dim + 3, dim + 3)
        case ("batch", "euclidean2", "optimized", true): return relative("optimized-euclid2", dim / 16 + 7, dim + 2)
        case ("batch", "euclidean", "optimized", true): return relative("optimized-euclid", dim / 16 + 8, dim + 3)
        case ("batch", "euclidean2", "aos-fp32-reused", false), ("batch", "euclidean2", "aos-fp32-setup", false),
             ("batch", "euclidean2", "consumer-only", false):
            return relative("optimized-euclid2", dim / 16 + 7, dim + 2)
        case ("batch", "euclidean2", "optimized-soa", false), ("batch", "euclidean2", "soa-fp32-reused", false),
             ("batch", "euclidean2", "soa-fp32-setup", false):
            return relative("soa-euclid2", dim / 4 + 5, dim + 2)
        case ("matrix", "euclidean2", "optimized", false), ("matrix", "euclidean2", "consumer-only", false):
            let scale = 4 * Double(dim) * fixtureCoordinateBound * fixtureCoordinateBound
            return ErrorModel(route: "matrix-norm-identity", kind: .absolute(scale: scale), k32: dim + 2, k64: dim + 2)
        default:
            return nil
        }
    }

    static func rule(forCaseName name: String) -> CorrectnessRule {
        if let model = errorModel(forCaseName: name) {
            switch model.kind {
            case .relative: return CorrectnessRule(maxRel: model.threshold, maxAbs: nil, basis: .derived(model))
            case .absolute: return CorrectnessRule(maxRel: nil, maxAbs: model.threshold, basis: .derived(model))
            }
        }
        let lower = name.lowercased()
        let reason = "pre-T11 fixed threshold; not derived from this route's summation"
        if lower.contains("fp16") {
            let metric = CaseParsing.parse(name: name).metric
            return CorrectnessRule(maxRel: metric == "cosine" ? fp16CosineEmpiricalRelative : fp16EmpiricalRelative, maxAbs: nil,
                        basis: .empirical(reason: reason))
        }
        return CorrectnessRule(maxRel: fp32EmpiricalRelative, maxAbs: nil, basis: .empirical(reason: reason))
    }

    /// Whether a case must carry a correctness record. Not applicable only where there is no VectorCore output to compare
    /// with a reference.
    static func validation(forCaseName name: String) -> ValidationRequirement {
        if name == "noop.scalar-add" {
            return .notApplicable(reason: "harness reference loop; it calls no VectorCore API, so there is no library output")
        }
        let p = CaseParsing.parse(name: name)
        if p.kind == "mem" {
            switch p.metric {
            case "alloc", "pool":
                return .notApplicable(reason: "allocation timing; the output is an address, not a value, and a failed "
                    + "allocation or acquisition already ends the case as an execution failure")
            case "fill":
                return .notApplicable(reason: "times a benchmark-local store loop; it calls no VectorCore API")
            default:
                break
            }
        }
        return .required
    }

    /// The failure the gate records for one result, or nil. An explicit `--max-rel-error` replaces only the relative
    /// threshold and an explicit `--max-abs-error` only the absolute one, for every case; a kind without an explicit
    /// value keeps the rule's threshold of that kind (nil when the rule has none). The failure reason names the source
    /// of each threshold (`testOverrideReplacesOnlyItsOwnKindAndTheReasonSaysSo`).
    static func failure(for result: BenchResult, overrideRel: Double?, overrideAbs: Double?) -> BenchmarkFailure? {
        guard let c = result.correctness else {
            switch validation(forCaseName: result.name) {
            case .required: return BenchmarkFailure(name: result.name, reason: "Missing validation", boundary: result.boundary)
            case .notApplicable: return nil
            }
        }
        guard c.status == .valid && c.samples > 0 else {
            return BenchmarkFailure(name: result.name, reason: c.reason ?? "Invalid comparison", boundary: result.boundary)
        }
        let rule = rule(forCaseName: result.name)
        let relThr = overrideRel ?? rule.maxRel
        let absThr = overrideAbs ?? rule.maxAbs
        let relBad = relThr.map { !(c.maxRelError <= $0) } ?? false
        let absBad = absThr.map { !(c.maxAbsError <= $0) } ?? false
        guard relBad || absBad else { return nil }
        func label(_ threshold: Double?, overridden: Bool) -> String {
            guard let threshold else { return "nil" }
            return String(format: "%.3e", threshold) + " (" + (overridden ? "override" : rule.summary) + ")"
        }
        let reason = String(format: "%@  maxRel=%.3e maxAbs=%.3e  thr(rel=%@ abs=%@)", result.name, c.maxRelError, c.maxAbsError,
                            label(relThr, overridden: overrideRel != nil), label(absThr, overridden: overrideAbs != nil))
        return BenchmarkFailure(name: result.name, reason: reason, boundary: result.boundary)
    }
}
