import Foundation
import Testing
import VectorCore

@Suite
struct UMAPPublicCurveTests {
    private let nondefaultSpread: Float = 1.5
    private let nondefaultMinDist: Float = 0.5

    @Test func defaultCurveMatchesBaselineCoefficients() throws {
        let curve = try Operations.umapCurveParameters(spread: 1, minDist: 0.1)
        try UMAPReferenceAssertions.compare([curve.a, curve.b], words: [0x3fc9d94a, 0x3f6522b3],
                                             tolerance: 1e-6, caseID: "default-coefficients")
    }

    @Test func nondefaultCurveMatchesBaselineCoefficients() throws {
        let curve = try Operations.umapCurveParameters(spread: nondefaultSpread, minDist: nondefaultMinDist)
        #expect(abs(curve.a - Float(bitPattern: 0x3eb7_f9d2)) < 1e-6)
        #expect(abs(curve.b - Float(bitPattern: 0x3f93_0ecd)) < 1e-6)
    }

    @Test func invalidCurveInputsThrowInvalidOperation() {
        let invalid: [(spread: Float, minDist: Float)] = [
            (0, 0), (-1, 0), (1, -0.1), (1, 2),
            (.nan, 0), (.infinity, 0), (1, .nan), (1, .infinity)
        ]
        for (spread, minDist) in invalid {
            do {
                _ = try Operations.umapCurveParameters(spread: spread, minDist: minDist)
                Issue.record("Accepted spread=\(spread), minDist=\(minDist)")
            } catch let error as VectorError {
                #expect(error.kind == .invalidOperation)
            } catch {
                Issue.record("Expected VectorError.invalidOperation, got \(error)")
            }
        }
    }

    @Test(arguments: [2, 5], [true, false])
    func bothEntryPointsPreserveCurveLayout(dimensions: Int, usesDefaultCurve: Bool) throws {
        let graph = try KNNGraph(
            pointCount: 4,
            rowOffsets: [0, 2, 4, 6, 8],
            neighborIndices: [1, 2, 0, 2, 1, 3, 2, 1],
            distances: [1, 2, 1, 1, 1, 2, 2, 3]
        )
        let vectors = (0..<4).map { DynamicVector([Float($0)]) }
        var config = UMAPConfig(neighbors: 2, epochs: 10, negativeSampleRate: 2, initialization: .random, seed: 17)
        if !usesDefaultCurve {
            config.spread = nondefaultSpread
            config.minDist = nondefaultMinDist
        }
        let vectorResult = try Operations.umap(vectors, dimensions: dimensions, graph: graph, config: config)
        let graphResult = try Operations.umap(graph: graph, dimensions: dimensions, config: config)

        // Captured before the API change at 0e7fbf182d5d (alpha UMAP sources).
        // SwiftPM Debug and Release, Xcode 27.2 (27B52), Swift 6.4.2 (swiftlang-6.4.2.1.9),
        // arm64 macOS 27.0.1 (26A434). The supplied CSR and random init bypass linear-algebra providers.
        let expected: [UInt32]
        switch (dimensions, usesDefaultCurve) {
        case (2, true):
            expected = [
                0xc027_e22b, 0xc050_cb28,
                0xc018_3cc9, 0xc022_6d7a,
                0xc0d8_4c58, 0xc0a0_044e,
                0xc107_15e0, 0x3f46_f84d
            ]
        case (2, false):
            expected = [
                0xc0b4_8c32, 0xc09f_24a4,
                0xc0d4_3f7c, 0xc0a0_d53a,
                0xc100_7995, 0xc08a_8664,
                0xc107_ce7c, 0xbf19_24e7
            ]
        case (5, true):
            expected = [
                0xbef3_0d5d, 0xc029_de3e, 0xc046_103f, 0xc0b1_c2ab, 0xc109_e6d5,
                0xc0f1_06bd, 0xc109_ee0e, 0x4002_41b2, 0xc0fd_d672, 0x4067_ad28,
                0xc0ab_5e7e, 0xc0fc_6132, 0xbf96_a37a, 0xbfbd_5cf4, 0x409e_759d,
                0xc111_e781, 0x410f_750c, 0xbfb8_feb4, 0xc0d1_d336, 0xc104_8ee2
            ]
        case (5, false):
            expected = [
                0xbf23_8b78, 0xc033_6f6c, 0xc03f_560a, 0xc0b1_e9c6, 0xc104_9ce1,
                0xc0e8_6010, 0xc106_100e, 0x3fd8_74a6, 0xc0ec_b2a6, 0x405d_b17d,
                0xc0b0_2603, 0xc0f7_9213, 0xbf75_1aa3, 0xc003_2d5d, 0x4093_1e4d,
                0xc111_3626, 0x410b_900e, 0xbfb6_771d, 0xc0d0_8ffe, 0xc101_abc5
            ]
        default:
            Issue.record("Missing baseline for \(dimensions) dimensions")
            return
        }

        for (entryPoint, result) in [("vectors", vectorResult), ("graph", graphResult)] {
            try UMAPReferenceAssertions.compare(result.coordinates, words: expected, tolerance: 1e-4,
                                                 caseID: "curve-\(dimensions)-\(usesDefaultCurve)-\(entryPoint)")
        }
        #expect(vectorResult.coordinates.map(\.bitPattern) == graphResult.coordinates.map(\.bitPattern))
    }
}
