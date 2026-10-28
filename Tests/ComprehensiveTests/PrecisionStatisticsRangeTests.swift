import Testing
import VectorCore

@Suite struct PrecisionStatisticsRangeTests {
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
}
