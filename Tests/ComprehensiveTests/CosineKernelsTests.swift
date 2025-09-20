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
}

// Simple deterministic RNG for tests
fileprivate struct SeededRandom {
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

