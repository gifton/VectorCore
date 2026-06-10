import Testing
import Foundation
@testable import VectorCore

/// Numerical-parity suite for ``AccelerateArraySIMDProvider``.
///
/// The Accelerate provider routes its hot arithmetic and reductions through vDSP
/// (``AccelerateFloatProvider``) while ``DefaultArraySIMDProvider`` uses
/// hand-rolled Swift SIMD loops. The two are expected to agree on the same
/// inputs — but only to within floating-point rounding, because vDSP reductions
/// (`vDSP_sve`, `vDSP_dotpr`, `vDSP_svesq`, `vDSP_distancesq`) accumulate with a
/// pairwise/tree order while the Swift provider accumulates sequentially. So we
/// assert agreement under a tight mixed absolute/relative tolerance rather than
/// bit-for-bit equality.
///
/// Guarded behind `canImport(Accelerate)` to match the provider's own guard.
#if canImport(Accelerate)
@Suite("AccelerateArraySIMDProvider parity")
struct AccelerateArraySIMDProviderParityTests {

    /// Mixed absolute/relative closeness: `|x - y| <= absTol + relTol * max(|x|, |y|)`.
    ///
    /// Pure relative tolerance is unstable when the reference value is ~0 (e.g. a
    /// near-zero dot product), so an absolute floor is added. These bounds are
    /// deliberately tight (1e-5 relative / 1e-6 absolute) to catch a wrong vDSP
    /// routing — they are loose only relative to exact equality.
    private func isClose(
        _ x: Float, _ y: Float,
        relTol: Float = 1e-5, absTol: Float = 1e-6
    ) -> Bool {
        if x == y { return true } // exact (covers ±inf and identical bit patterns)
        let diff = Swift.abs(x - y)
        let scale = Swift.max(Swift.abs(x), Swift.abs(y))
        return diff <= absTol + relTol * scale
    }

    /// Element-wise closeness over two equal-length arrays.
    private func arraysClose(
        _ x: [Float], _ y: [Float],
        relTol: Float = 1e-5, absTol: Float = 1e-6
    ) -> Bool {
        guard x.count == y.count else { return false }
        for i in x.indices where !isClose(x[i], y[i], relTol: relTol, absTol: absTol) {
            return false
        }
        return true
    }

    /// Sizes spanning: scalar tail (1), sub-SIMD4 remainder (7), an exact
    /// SIMD8-aligned block (64), and a large odd size with a non-trivial
    /// remainder (513) so the vDSP and Swift accumulation orders diverge most.
    private static let sizes = [1, 7, 64, 513]

    @Test("Accelerate provider matches the default Swift provider", arguments: sizes)
    func parity(n: Int) {
        let accel = AccelerateArraySIMDProvider()
        let swift = DefaultArraySIMDProvider()

        // Fresh, per-size deterministic generator (documented race-free pattern).
        var rng = SeededGenerator(seed: 0xA22E_1D00 &+ UInt64(n))
        let a = (0..<n).map { _ in Float.random(in: -1...1, using: &rng) }
        // `b` is offset away from zero so cosineSimilarity / euclidean divisors
        // stay well-conditioned and the relative tolerance is meaningful.
        let b = (0..<n).map { _ in Float.random(in: 0.25...1.25, using: &rng) }

        // --- Reductions ---
        #expect(isClose(accel.dot(a, b), swift.dot(a, b)),
                "dot mismatch (n=\(n))")
        #expect(isClose(accel.sum(a), swift.sum(a)),
                "sum mismatch (n=\(n))")
        #expect(isClose(accel.magnitude(a), swift.magnitude(a)),
                "magnitude mismatch (n=\(n))")
        #expect(isClose(accel.euclideanDistanceSquared(a, b),
                        swift.euclideanDistanceSquared(a, b)),
                "euclideanDistanceSquared mismatch (n=\(n))")
        #expect(isClose(accel.cosineSimilarity(a, b), swift.cosineSimilarity(a, b)),
                "cosineSimilarity mismatch (n=\(n))")

        // --- Array-returning arithmetic ---
        #expect(arraysClose(accel.normalize(a), swift.normalize(a)),
                "normalize mismatch (n=\(n))")
        #expect(arraysClose(accel.add(a, b), swift.add(a, b)),
                "add mismatch (n=\(n))")
        #expect(arraysClose(accel.subtract(a, b), swift.subtract(a, b)),
                "subtract mismatch (n=\(n))")
        #expect(arraysClose(accel.multiply(a, by: 3.5), swift.multiply(a, by: 3.5)),
                "multiply(_:by:) mismatch (n=\(n))")
    }
}
#endif // canImport(Accelerate)
