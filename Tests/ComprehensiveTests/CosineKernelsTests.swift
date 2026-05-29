import Testing
@testable import VectorCore

@Suite("Cosine Kernels (Optimized)")
struct CosineKernelsTests {

    // MARK: - Fused vs two-pass reference

    @Test
    func testFusedMatchesTwoPass_512() {
        for seed in 0..<5 {
            var rng = SeededRandom(seed: UInt64(5120 + seed))
            let aArr = (0..<512).map { _ in rng.nextFloat(in: -1...1) }
            let bArr = (0..<512).map { _ in rng.nextFloat(in: -1...1) }
            let a = try! Vector512Optimized(aArr)
            let b = try! Vector512Optimized(bArr)

            let fused = CosineKernels.distance512_fused(a, b)
            let dot = DotKernels.dot512(a, b)
            let aa = DotKernels.dot512(a, a)
            let bb = DotKernels.dot512(b, b)
            let expected = CosineKernels.calculateCosineDistance(dot: dot, sumAA: aa, sumBB: bb)
            #expect(approxEqual(fused, expected, tol: 1e-5))
        }
    }

    @Test
    func testFusedMatchesTwoPass_768() {
        for seed in 0..<5 {
            var rng = SeededRandom(seed: UInt64(7680 + seed))
            let aArr = (0..<768).map { _ in rng.nextFloat(in: -1...1) }
            let bArr = (0..<768).map { _ in rng.nextFloat(in: -1...1) }
            let a = try! Vector768Optimized(aArr)
            let b = try! Vector768Optimized(bArr)

            let fused = CosineKernels.distance768_fused(a, b)
            let dot = DotKernels.dot768(a, b)
            let aa = DotKernels.dot768(a, a)
            let bb = DotKernels.dot768(b, b)
            let expected = CosineKernels.calculateCosineDistance(dot: dot, sumAA: aa, sumBB: bb)
            #expect(approxEqual(fused, expected, tol: 1e-5))
        }
    }

    @Test
    func testFusedMatchesTwoPass_1536() {
        for seed in 0..<5 {
            var rng = SeededRandom(seed: UInt64(15360 + seed))
            let aArr = (0..<1536).map { _ in rng.nextFloat(in: -1...1) }
            let bArr = (0..<1536).map { _ in rng.nextFloat(in: -1...1) }
            let a = try! Vector1536Optimized(aArr)
            let b = try! Vector1536Optimized(bArr)

            let fused = CosineKernels.distance1536_fused(a, b)
            let dot = DotKernels.dot1536(a, b)
            let aa = DotKernels.dot1536(a, a)
            let bb = DotKernels.dot1536(b, b)
            let expected = CosineKernels.calculateCosineDistance(dot: dot, sumAA: aa, sumBB: bb)
            #expect(approxEqual(fused, expected, tol: 1e-5))
        }
    }

    // MARK: - Pre-normalized path

    @Test
    func testPreNormalizedMatchesFused_512() {
        var rng = SeededRandom(seed: 9101)
        let a = try! Vector512Optimized((0..<512).map { _ in rng.nextFloat(in: -1...1) })
        let b = try! Vector512Optimized((0..<512).map { _ in rng.nextFloat(in: -1...1) })
        let aN = (try? a.normalized().get()) ?? a
        let bN = (try? b.normalized().get()) ?? b
        let pre = CosineKernels.distance512_preNormalized(aN, bN)
        let fused = CosineKernels.distance512_fused(aN, bN)
        #expect(approxEqual(pre, fused, tol: 1e-5))
    }

    @Test
    func testPreNormalizedMatchesFused_768() {
        var rng = SeededRandom(seed: 9102)
        let a = try! Vector768Optimized((0..<768).map { _ in rng.nextFloat(in: -1...1) })
        let b = try! Vector768Optimized((0..<768).map { _ in rng.nextFloat(in: -1...1) })
        let aN = (try? a.normalized().get()) ?? a
        let bN = (try? b.normalized().get()) ?? b
        let pre = CosineKernels.distance768_preNormalized(aN, bN)
        let fused = CosineKernels.distance768_fused(aN, bN)
        #expect(approxEqual(pre, fused, tol: 1e-5))
    }

    @Test
    func testPreNormalizedMatchesFused_1536() {
        var rng = SeededRandom(seed: 9103)
        let a = try! Vector1536Optimized((0..<1536).map { _ in rng.nextFloat(in: -1...1) })
        let b = try! Vector1536Optimized((0..<1536).map { _ in rng.nextFloat(in: -1...1) })
        let aN = (try? a.normalized().get()) ?? a
        let bN = (try? b.normalized().get()) ?? b
        let pre = CosineKernels.distance1536_preNormalized(aN, bN)
        let fused = CosineKernels.distance1536_fused(aN, bN)
        #expect(approxEqual(pre, fused, tol: 1e-5))
    }

    // MARK: - Zero-vector handling (optimized semantics)

    @Test
    func testZeroVectorHandling_Fused() {
        let z512 = try! Vector512Optimized(Array(repeating: 0, count: 512))
        let u512 = try! Vector512Optimized(Array(repeating: 1, count: 512))
        // One zero, one non-zero -> distance = 1
        #expect(approxEqual(CosineKernels.distance512_fused(z512, u512), 1, tol: 1e-6))
        // Both zero -> distance = 0 (CosineKernels behavior)
        #expect(approxEqual(CosineKernels.distance512_fused(z512, z512), 0, tol: 1e-6))
    }

    @Test
    func testDistanceRangeClamped() {
        var rng = SeededRandom(seed: 7777)
        let a = try! Vector768Optimized((0..<768).map { _ in rng.nextFloat(in: -2...2) })
        let b = try! Vector768Optimized((0..<768).map { _ in rng.nextFloat(in: -2...2) })
        let d = CosineKernels.distance768_fused(a, b)
        #expect(d >= 0 - 1e-6 && d <= 2 + 1e-6)
    }

    // MARK: - Regression: +Inf overflow in cosine denominator (Fix 4.2)

    /// For large unnormalized vectors, forming the product `sumAA * sumBB`
    /// overflows Float to +Inf (e.g. 9e38 * 9e38). The old code then computed
    /// `dot / sqrt(+Inf) = dot / +Inf = 0`, clamped to 0, and returned a spurious
    /// distance of 1.0 for vectors that are in fact identical in direction.
    /// The fix computes `sqrt(sumAA) * sqrt(sumBB)` so the denominator stays finite.
    @Test
    func testCosineDistanceNoOverflowForHugeIdenticalDirection() {
        // Two vectors of identical direction with huge magnitude.
        // ||v||² ≈ 9e38 for each; dot(a,b) ≈ 9e38 as well (parallel).
        // sumAA * sumBB ≈ 8.1e77 → +Inf in Float. sqrt path keeps it finite.
        let sumAA: Float = 9.0e38
        let sumBB: Float = 9.0e38
        let dot: Float = 9.0e38  // perfectly parallel: dot == sqrt(sumAA)*sqrt(sumBB)

        // Confirm the product genuinely overflows (the bug's precondition).
        #expect((sumAA * sumBB).isInfinite, "Precondition: product must overflow to +Inf")

        let d = CosineKernels.calculateCosineDistance(dot: dot, sumAA: sumAA, sumBB: sumBB)
        #expect(d.isFinite, "Distance must be finite, not derived from dot/Inf")
        #expect(!d.isNaN, "Distance must not be NaN")
        // Identical direction → cosine similarity ≈ 1 → distance ≈ 0, NOT 1.0.
        #expect(approxEqual(d, 0, tol: 1e-5), "Expected ≈0 for identical direction, got \(d)")
    }

    /// Huge orthogonal-ish case: dot = 0 with huge magnitudes should yield
    /// distance ≈ 1 (cosine similarity 0), and must remain finite.
    @Test
    func testCosineDistanceHugeMagnitudeOrthogonal() {
        let sumAA: Float = 9.0e38
        let sumBB: Float = 9.0e38
        let dot: Float = 0.0

        let d = CosineKernels.calculateCosineDistance(dot: dot, sumAA: sumAA, sumBB: sumBB)
        #expect(d.isFinite && !d.isNaN, "Distance must be finite")
        #expect(approxEqual(d, 1, tol: 1e-5), "Orthogonal huge vectors → distance ≈ 1, got \(d)")
    }

    /// Zero-vector semantics must be preserved exactly by the rewritten guard.
    /// "Zero" now means at/below Float.leastNormalMagnitude (squared magnitude),
    /// so we use true zero / sub-normal squared magnitudes here.
    @Test
    func testCosineDistanceZeroVectorSemanticsPreserved() {
        let z: Float = 0
        // Both zero → 0.
        #expect(approxEqual(CosineKernels.calculateCosineDistance(dot: 0, sumAA: z, sumBB: z), 0, tol: 1e-6))
        // One zero, one non-zero → 1.
        #expect(approxEqual(CosineKernels.calculateCosineDistance(dot: 0, sumAA: z, sumBB: 1.0), 1, tol: 1e-6))
        #expect(approxEqual(CosineKernels.calculateCosineDistance(dot: 0, sumAA: 1.0, sumBB: z), 1, tol: 1e-6))
    }

    /// Regression (Fix 4.5): valid micro-magnitude vectors must NOT be rejected.
    /// Two identical vectors of magnitude 1e-5 have sumAA = sumBB = dot = 1e-10.
    /// Under the old `epsilon = 1e-9` floor (a magnitude floor of ~3.16e-5) this
    /// was wrongly treated as a zero vector and returned distance 1.0. With the
    /// `Float.leastNormalMagnitude` floor it is accepted and the true distance
    /// (≈0 for identical direction) is computed.
    @Test
    func testCosineDistanceMicroMagnitudeNotRejected() {
        let sq: Float = 1e-10  // = (1e-5)^2, well above leastNormalMagnitude (~1.18e-38)
        // Identical direction → distance ≈ 0.
        let dParallel = CosineKernels.calculateCosineDistance(dot: sq, sumAA: sq, sumBB: sq)
        #expect(approxEqual(dParallel, 0, tol: 1e-5),
                "micro-magnitude parallel vectors must give distance ~0, got \(dParallel)")
        // Opposite direction → distance ≈ 2.
        let dAnti = CosineKernels.calculateCosineDistance(dot: -sq, sumAA: sq, sumBB: sq)
        #expect(approxEqual(dAnti, 2, tol: 1e-5),
                "micro-magnitude anti-parallel vectors must give distance ~2, got \(dAnti)")
    }
}

// Simple deterministic RNG for tests
private struct SeededRandom {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed &* 0x9E3779B97F4A7C15 }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    mutating func nextFloat(in range: ClosedRange<Float>) -> Float {
        let u = next()
        let x = Double(u) / Double(UInt64.max)
        let f = Float(x)
        return range.lowerBound + (range.upperBound - range.lowerBound) * f
    }
}
