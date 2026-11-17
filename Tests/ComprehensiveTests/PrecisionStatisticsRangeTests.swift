import Testing
import VectorCore

@Suite struct PrecisionStatisticsRangeTests {
    private func expectZeroSentinel(_ profile: MixedPrecisionKernels.PrecisionProfile, dimension: Int) {
        let statistics = [profile.minValue, profile.maxValue, profile.meanValue,
                          profile.stdDev, profile.dynamicRange, profile.expectedError]
        #expect(statistics.allSatisfy { $0.isFinite && $0 == 0 })
        #expect(profile.recommendedPrecision == .fp32)
        #expect(profile.outlierCount == 0)
        #expect(profile.dimension == dimension)
    }

    private func noFiniteObservations<V: OptimizedVector & MixedPrecisionKernels.AnalyzableVector>(
        _ type: V.Type, width: Int
    ) throws {
        let mixedSpecials = (0..<width).map { index -> Float in
            switch index % 3 {
            case 0: return .nan
            case 1: return .infinity
            default: return -.infinity
            }
        }
        let fixtures: [[V]] = [
            [try V(Array(repeating: .nan, count: width))],
            [try V(Array(repeating: .infinity, count: width))],
            [try V(Array(repeating: -.infinity, count: width))],
            [try V(mixedSpecials)],
            [try V(Array(repeating: .nan, count: width)),
             try V(Array(repeating: .infinity, count: width)), try V(mixedSpecials)]
        ]
        for vectors in fixtures {
            expectZeroSentinel(MixedPrecisionKernels.analyzePrecision(vectors), dimension: width)
            #expect(MixedPrecisionKernels.selectOptimalPrecision(for: vectors) == .fp32)
            #expect(MixedPrecisionKernels.selectOptimalPrecision(for: vectors, errorTolerance: 1) == .fp32)
        }
    }

    private func sentinelControls<V: OptimizedVector & MixedPrecisionKernels.AnalyzableVector>(
        _ type: V.Type, width: Int
    ) throws {
        let empty: [V] = []
        expectZeroSentinel(MixedPrecisionKernels.analyzePrecision(empty), dimension: 0)
        #expect(MixedPrecisionKernels.selectOptimalPrecision(for: empty) == .fp32)
        #expect(MixedPrecisionKernels.selectOptimalPrecision(for: empty, errorTolerance: 1) == .fp32)

        let zeroVectors = [try V(Array(repeating: Float.zero, count: width))]
        let zero = MixedPrecisionKernels.analyzePrecision(zeroVectors)
        #expect([zero.minValue, zero.maxValue, zero.meanValue, zero.stdDev, zero.dynamicRange]
            .allSatisfy { $0.isFinite && $0 == 0 })
        #expect(zero.expectedError == 0.005)
        #expect(zero.recommendedPrecision == .int8)
        #expect(zero.outlierCount == 0 && zero.dimension == width)
        #expect(MixedPrecisionKernels.selectOptimalPrecision(for: zeroVectors) == .fp16)
        #expect(MixedPrecisionKernels.selectOptimalPrecision(for: zeroVectors, errorTolerance: 1) == .int8)

        let finiteValues = Array(repeating: Float(1), count: width / 2)
            + Array(repeating: Float(3), count: width / 2)
        let finite = MixedPrecisionKernels.analyzePrecision([
            try V(Array(repeating: .nan, count: width)), try V(finiteValues)
        ])
        #expect(finite.minValue == 1 && finite.maxValue == 3)
        #expect(finite.meanValue == 2 && finite.stdDev == 1 && finite.dynamicRange == 2)
        #expect(finite.expectedError.isFinite && finite.expectedError > 0)
        #expect(finite.recommendedPrecision == .mixed)
        #expect(finite.outlierCount == 0 && finite.dimension == width)
    }

    private func statistics<V: OptimizedVector & MixedPrecisionKernels.AnalyzableVector>(_ type: V.Type, width: Int) throws {
        let close = Array(repeating: Float(16_777_216), count: width / 2)
            + Array(repeating: Float(16_777_218), count: width / 2)
        let closeProfile = MixedPrecisionKernels.analyzePrecision([try V(close)])
        #expect(closeProfile.stdDev == 1)
        #expect(closeProfile.outlierCount == 0)

        let maximum = Float.greatestFiniteMagnitude
        var extreme = Array(repeating: maximum, count: width)
        extreme[0] = -maximum
        let profile = MixedPrecisionKernels.analyzePrecision([try V(extreme)])
        let expected = 2 * Double(maximum) * Double(width - 1).squareRoot() / Double(width)
        #expect(profile.stdDev.isFinite)
        #expect(abs(Double(profile.stdDev) - expected) / expected < 2e-6)
        #expect(profile.outlierCount == 1)
        #expect(profile.minValue == -maximum && profile.maxValue == maximum)
        #expect(profile.recommendedPrecision == .fp32)

        let constant = MixedPrecisionKernels.analyzePrecision([try V(Array(repeating: Float(7), count: width))])
        #expect(constant.meanValue == 7 && constant.stdDev == 0 && constant.outlierCount == 0)

        let filtered = (0..<width).map { index -> Float in
            switch index % 4 {
            case 0: return 1
            case 1: return 3
            case 2: return .nan
            default: return .infinity
            }
        }
        let finite = MixedPrecisionKernels.analyzePrecision([try V(filtered)])
        #expect(finite.meanValue == 2 && finite.stdDev == 1)
        #expect(finite.minValue == 1 && finite.maxValue == 3 && finite.outlierCount == 0)
    }

    @Test func statisticsAcrossSupportedWidths() throws {
        try statistics(Vector512Optimized.self, width: 512)
        try statistics(Vector768Optimized.self, width: 768)
        try statistics(Vector1536Optimized.self, width: 1536)
    }

    @Test func noFiniteObservationsAcrossSupportedWidths() throws {
        try noFiniteObservations(Vector512Optimized.self, width: 512)
        try noFiniteObservations(Vector768Optimized.self, width: 768)
        try noFiniteObservations(Vector1536Optimized.self, width: 1536)
    }

    @Test func sentinelControlsAcrossSupportedWidths() throws {
        try sentinelControls(Vector512Optimized.self, width: 512)
        try sentinelControls(Vector768Optimized.self, width: 768)
        try sentinelControls(Vector1536Optimized.self, width: 1536)
    }
}
