import Foundation
import Testing
@testable import VectorCoreBenchSupport
import VectorCoreBenchmarking

@Test func BenchmarkContractSwiftTestingEntry() throws {
    let invalid = Correctness.vector(currents: [.nan], references: [1])
    let data = try JSONEncoder().encode(CorrectnessOut(from: invalid))
    let decoded = try JSONDecoder().decode(CorrectnessOut.self, from: data)
    #expect(decoded.status == .invalid)
    #expect(decoded.maxAbsError == nil)
}

// T15 ruling 10: the benchmark correctness oracle `DoubleRef.cosineDist` follows the library's cosine exceptional
// policy (registered in Tests/ComprehensiveTests/T15CosineRegistry.swift). Benchmark correctness compares finite
// fixtures only, so its finite results are unchanged; these rows pin the exceptional classes. This suite lives in
// this file because the BenchmarkContractTests source membership is a reviewed contract
// (Scripts/ci/check_ci_contracts.py BENCHMARK_TEST_SOURCES).

@Suite("T15 DoubleRef.cosineDist exceptional policy")
struct T15DoubleRefCosineTests {
    private static func distance(_ a: [Float], _ b: [Float]) -> Double {
        a.withUnsafeBufferPointer { pa in b.withUnsafeBufferPointer { pb in DoubleRef.cosineDist(pa, pb) } }
    }

    @Test func doubleRefCosineFollowsThePolicy() {
        let x: [Float] = [0.3, -0.2, 0.5, 0.1, -0.4, 0.25], y: [Float] = [-0.1, 0.4, 0.2, 0.3, 0.15, -0.35]
        let zero = [Float](repeating: 0, count: x.count)
        func with(_ v: [Float], _ index: Int, _ value: Float) -> [Float] { var w = v; w[index] = value; return w }
        // P1: a NaN or infinite coordinate on either side gives NaN, including against a zero vector.
        let exceptional: [(String, [Float], [Float])] = [
            ("nanA", with(x, 2, .nan), y), ("nanB", x, with(y, 2, .nan)), ("posInfA", with(x, 2, .infinity), y),
            ("negInfB", x, with(y, 2, -.infinity)), ("infMixB", x, with(with(y, 2, .infinity), 3, -.infinity)),
            ("nanA.zeroB", with(x, 2, .nan), zero), ("zeroA.infB", zero, with(y, 2, .infinity))
        ]
        for (name, a, b) in exceptional {
            let value = Self.distance(a, b)
            #expect(value.isNaN, "\(name): got \(value)")
        }
        // P3a / P3b: two zero vectors, or one, give 1 (the oracle's zero rule is sqrt(sa * sb) <= ulpOfOne, P3c).
        #expect(Self.distance(zero, zero) == 1)
        #expect(Self.distance(zero, y) == 1)
        #expect(Self.distance(x, zero) == 1)
        // Finite inputs: the [-1, 1] clamp keeps results in [0, 2]; parallel and anti-parallel pairs give 0 and 2.
        #expect(abs(Self.distance(x, x)) <= 1e-12)
        #expect(abs(Self.distance(x, x.map { -$0 }) - 2) <= 1e-12)
        let general = Self.distance(x, y)
        #expect(general.isFinite && general >= 0 && general <= 2)
    }
}
