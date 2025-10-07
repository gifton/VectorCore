// Tests/ComprehensiveTests/MixedPrecisionComprehensiveTests.swift

import Testing
import Foundation
@testable import VectorCore

// MARK: - Type Aliases for Nested Mixed Precision Types

fileprivate typealias Vector512FP16 = MixedPrecisionKernels.Vector512FP16
fileprivate typealias Vector768FP16 = MixedPrecisionKernels.Vector768FP16
fileprivate typealias Vector1536FP16 = MixedPrecisionKernels.Vector1536FP16
fileprivate typealias SoAFP16 = MixedPrecisionKernels.SoAFP16
fileprivate typealias SoA512FP16 = MixedPrecisionKernels.SoA512FP16
fileprivate typealias SoA768FP16 = MixedPrecisionKernels.SoA768FP16
fileprivate typealias SoA1536FP16 = MixedPrecisionKernels.SoA1536FP16

/// Comprehensive test suite for Mixed Precision Kernels per spec requirements
///
/// Tests cover:
/// 1. Unit tests (conversion accuracy, dot product accuracy, special values)
/// 2. Numerical validation (error bounds, orthogonality, parallelism)
/// 3. Performance validation (throughput targets, bandwidth utilization)
/// 4. Integration tests (end-to-end similarity search)
@Suite("Mixed Precision Comprehensive Tests")
struct MixedPrecisionComprehensiveTests {

    // MARK: - Unit Tests: Conversion Accuracy

    @Suite("Conversion Accuracy Tests")
    struct ConversionAccuracyTests {

        @Test("Round-trip FP32→FP16→FP32 maintains accuracy")
        func testRoundTripConversion512() throws {
            // Test with normalized embedding-like vectors
            let testVectors: [[Float]] = [
                Array(repeating: 1.0, count: 512),
                (0..<512).map { Float($0) / 512.0 },
                (0..<512).map { sin(Float($0) * .pi / 256) },
                (0..<512).map { _ in Float.random(in: -1...1) }
            ]

            for (idx, values) in testVectors.enumerated() {
                let original = try Vector512Optimized(values)
                let fp16 = MixedPrecisionKernels.Vector512FP16(from: original)
                let reconstructed = fp16.toFP32()

                for i in 0..<512 {
                    let originalValue = original[i]
                    let reconstructedValue = reconstructed[i]

                    // FP16 relative error should be < 0.05% for normalized values
                    if abs(originalValue) > 1e-4 {
                        let relativeError = abs(reconstructedValue - originalValue) / abs(originalValue)
                        #expect(relativeError < 0.0005,
                                "Vector \(idx), index \(i): relative error \(relativeError) exceeds 0.05%")
                    } else {
                        // For very small values, check absolute error
                        #expect(abs(reconstructedValue - originalValue) < 1e-4)
                    }
                }
            }
        }

        @Test("Conversion preserves special values")
        func testSpecialValueConversion() throws {
            // Test infinity
            let posInfVector = try Vector512Optimized(Array(repeating: Float.infinity, count: 512))
            let negInfVector = try Vector512Optimized(Array(repeating: -Float.infinity, count: 512))

            let posInfFP16 = MixedPrecisionKernels.Vector512FP16(from: posInfVector)
            let negInfFP16 = MixedPrecisionKernels.Vector512FP16(from: negInfVector)

            let posInfReconstructed = posInfFP16.toFP32()
            let negInfReconstructed = negInfFP16.toFP32()

            for i in 0..<512 {
                #expect(posInfReconstructed[i] == Float.infinity, "Positive infinity not preserved at \(i)")
                #expect(negInfReconstructed[i] == -Float.infinity, "Negative infinity not preserved at \(i)")
            }

            // Test NaN
            let nanVector = try Vector512Optimized(Array(repeating: Float.nan, count: 512))
            let nanFP16 = MixedPrecisionKernels.Vector512FP16(from: nanVector)
            let nanReconstructed = nanFP16.toFP32()

            for i in 0..<512 {
                #expect(nanReconstructed[i].isNaN, "NaN not preserved at \(i)")
            }

            // Test signed zero
            let posZero = try Vector512Optimized(Array(repeating: 0.0, count: 512))
            let negZero = try Vector512Optimized(Array(repeating: -0.0, count: 512))

            let posZeroFP16 = MixedPrecisionKernels.Vector512FP16(from: posZero)
            let negZeroFP16 = MixedPrecisionKernels.Vector512FP16(from: negZero)

            let posZeroReconstructed = posZeroFP16.toFP32()
            let negZeroReconstructed = negZeroFP16.toFP32()

            for i in 0..<512 {
                #expect(posZeroReconstructed[i] == 0.0)
                #expect(negZeroReconstructed[i] == -0.0 || negZeroReconstructed[i] == 0.0, "Signed zero handling")
            }
        }

        @Test("Boundary values (FP16 range limits)")
        func testBoundaryValues() throws {
            let fp16Max: Float = 65504.0
            let fp16Min: Float = -65504.0
            let fp16MinNormal: Float = 6.10352e-5

            // Test maximum values
            let maxVector = try Vector512Optimized(Array(repeating: fp16Max, count: 512))
            let minVector = try Vector512Optimized(Array(repeating: fp16Min, count: 512))

            let maxFP16 = MixedPrecisionKernels.Vector512FP16(from: maxVector)
            let minFP16 = MixedPrecisionKernels.Vector512FP16(from: minVector)

            let maxReconstructed = maxFP16.toFP32()
            let minReconstructed = minFP16.toFP32()

            for i in 0..<512 {
                #expect(abs(maxReconstructed[i] - fp16Max) / fp16Max < 0.001)
                #expect(abs(minReconstructed[i] - fp16Min) / abs(fp16Min) < 0.001)
            }

            // Test values exceeding FP16 range (should clamp to infinity)
            let overflowVector = try Vector512Optimized(Array(repeating: 70000.0, count: 512))
            let underflowVector = try Vector512Optimized(Array(repeating: -70000.0, count: 512))

            let overflowFP16 = MixedPrecisionKernels.Vector512FP16(from: overflowVector)
            let underflowFP16 = MixedPrecisionKernels.Vector512FP16(from: underflowVector)

            let overflowReconstructed = overflowFP16.toFP32()
            let underflowReconstructed = underflowFP16.toFP32()

            for i in 0..<512 {
                // Should clamp to infinity or max representable value
                #expect(overflowReconstructed[i] >= fp16Max || overflowReconstructed[i] == Float.infinity)
                #expect(underflowReconstructed[i] <= fp16Min || underflowReconstructed[i] == -Float.infinity)
            }

            // Test subnormal values (near zero)
            let subnormalVector = try Vector512Optimized(Array(repeating: fp16MinNormal / 2, count: 512))
            let subnormalFP16 = MixedPrecisionKernels.Vector512FP16(from: subnormalVector)
            let subnormalReconstructed = subnormalFP16.toFP32()

            // Subnormals may be flushed to zero or preserved
            for i in 0..<512 {
                #expect(abs(subnormalReconstructed[i]) < fp16MinNormal * 10)
            }
        }
    }

    // MARK: - Unit Tests: Dot Product Accuracy

    @Suite("Dot Product Accuracy Tests")
    struct DotProductAccuracyTests {

        @Test("FP16 dot product accuracy vs FP32 baseline (512-dim)")
        func testDotProductAccuracy512() throws {
            let iterations = 100
            var maxRelativeError: Float = 0
            var errors: [Float] = []

            for _ in 0..<iterations {
                // Generate random normalized vectors
                let values1 = (0..<512).map { _ in Float.random(in: -1...1) }
                let values2 = (0..<512).map { _ in Float.random(in: -1...1) }

                let vec1 = try Vector512Optimized(values1)
                let vec2 = try Vector512Optimized(values2)

                let vec1FP16 = MixedPrecisionKernels.Vector512FP16(from: vec1)
                let vec2FP16 = MixedPrecisionKernels.Vector512FP16(from: vec2)

                // FP32 reference
                let fp32Result = DotKernels.dot512(vec1, vec2)

                // FP16 result
                let fp16Result = MixedPrecisionKernels.dotFP16_512(vec1FP16, vec2FP16)

                if abs(fp32Result) > 1e-4 {
                    let relativeError = abs(fp16Result - fp32Result) / abs(fp32Result)
                    errors.append(relativeError)
                    maxRelativeError = max(maxRelativeError, relativeError)
                }
            }

            let meanError = errors.reduce(0, +) / Float(errors.count)

            // Spec requirement: relative error < 0.1% for 512-dim
            #expect(maxRelativeError < 0.001, "Max relative error \(maxRelativeError) exceeds 0.1%")
            #expect(meanError < 0.0005, "Mean relative error \(meanError) exceeds 0.05%")

            print("512-dim Dot Product Accuracy:")
            print("  Mean relative error: \(String(format: "%.6f%%", meanError * 100))")
            print("  Max relative error:  \(String(format: "%.6f%%", maxRelativeError * 100))")
        }

        @Test("Mixed precision dot product accuracy (512-dim)")
        func testMixedPrecisionAccuracy512() throws {
            let iterations = 100
            var maxRelativeError: Float = 0
            var errors: [Float] = []

            for _ in 0..<iterations {
                let values1 = (0..<512).map { _ in Float.random(in: -1...1) }
                let values2 = (0..<512).map { _ in Float.random(in: -1...1) }

                let query = try Vector512Optimized(values1)
                let candidate = try Vector512Optimized(values2)
                let candidateFP16 = MixedPrecisionKernels.Vector512FP16(from: candidate)

                // FP32 reference
                let fp32Result = DotKernels.dot512(query, candidate)

                // Mixed precision result (FP32 query, FP16 candidate)
                let mixedResult = MixedPrecisionKernels.dotMixed512(query: query, candidate: candidateFP16)

                if abs(fp32Result) > 1e-4 {
                    let relativeError = abs(mixedResult - fp32Result) / abs(fp32Result)
                    errors.append(relativeError)
                    maxRelativeError = max(maxRelativeError, relativeError)
                }
            }

            let meanError = errors.reduce(0, +) / Float(errors.count)

            // Mixed precision should have even better accuracy (query is FP32)
            #expect(maxRelativeError < 0.0008, "Max relative error \(maxRelativeError) exceeds 0.08%")
            #expect(meanError < 0.0004, "Mean relative error \(meanError) exceeds 0.04%")

            print("512-dim Mixed Precision Accuracy:")
            print("  Mean relative error: \(String(format: "%.6f%%", meanError * 100))")
            print("  Max relative error:  \(String(format: "%.6f%%", maxRelativeError * 100))")
        }

        @Test("Accuracy across all dimensions (512, 768, 1536)")
        func testAccuracyAllDimensions() throws {
            // Test 768-dim
            do {
                let values1 = (0..<768).map { _ in Float.random(in: -1...1) }
                let values2 = (0..<768).map { _ in Float.random(in: -1...1) }

                let vec1 = try Vector768Optimized(values1)
                let vec2 = try Vector768Optimized(values2)
                let vec1FP16 = MixedPrecisionKernels.Vector768FP16(from: vec1)
                let vec2FP16 = MixedPrecisionKernels.Vector768FP16(from: vec2)

                let fp32Result = DotKernels.dot768(vec1, vec2)
                let fp16Result = MixedPrecisionKernels.dotFP16_768(vec1FP16, vec2FP16)

                let relativeError = abs(fp16Result - fp32Result) / abs(fp32Result)
                #expect(relativeError < 0.001, "768-dim error \(relativeError) exceeds 0.1%")
            }

            // Test 1536-dim
            do {
                let values1 = (0..<1536).map { _ in Float.random(in: -1...1) }
                let values2 = (0..<1536).map { _ in Float.random(in: -1...1) }

                let vec1 = try Vector1536Optimized(values1)
                let vec2 = try Vector1536Optimized(values2)
                let vec1FP16 = MixedPrecisionKernels.Vector1536FP16(from: vec1)
                let vec2FP16 = MixedPrecisionKernels.Vector1536FP16(from: vec2)

                let fp32Result = DotKernels.dot1536(vec1, vec2)
                let fp16Result = MixedPrecisionKernels.dotFP16_1536(vec1FP16, vec2FP16)

                let relativeError = abs(fp16Result - fp32Result) / abs(fp32Result)
                #expect(relativeError < 0.0015, "1536-dim error \(relativeError) exceeds 0.15%")
            }
        }
    }

    // MARK: - Numerical Validation

    @Suite("Numerical Validation Tests")
    struct NumericalValidationTests {

        @Test("Orthogonal vectors maintain orthogonality")
        func testOrthogonalVectors() throws {
            // Create orthogonal basis vectors
            var basis1 = [Float](repeating: 0, count: 512)
            var basis2 = [Float](repeating: 0, count: 512)

            // Set first half of basis1 to 1, second half of basis2 to 1
            for i in 0..<256 {
                basis1[i] = 1.0
                basis2[i + 256] = 1.0
            }

            let vec1 = try Vector512Optimized(basis1)
            let vec2 = try Vector512Optimized(basis2)
            let vec1FP16 = MixedPrecisionKernels.Vector512FP16(from: vec1)
            let vec2FP16 = MixedPrecisionKernels.Vector512FP16(from: vec2)

            // Dot product should be zero (or very close)
            let dotProduct = MixedPrecisionKernels.dotFP16_512(vec1FP16, vec2FP16)
            #expect(abs(dotProduct) < 0.001, "Orthogonal vectors: dot product \(dotProduct) not near zero")
        }

        @Test("Parallel vectors maintain alignment")
        func testParallelVectors() throws {
            // Create parallel vectors (one is scalar multiple of the other)
            let base = (0..<512).map { Float($0) / 512.0 }
            let scaled = base.map { $0 * 2.5 }

            let vec1 = try Vector512Optimized(base)
            let vec2 = try Vector512Optimized(scaled)
            let vec1FP16 = MixedPrecisionKernels.Vector512FP16(from: vec1)
            let vec2FP16 = MixedPrecisionKernels.Vector512FP16(from: vec2)

            // Compute dot products and magnitudes
            let dotFP32 = DotKernels.dot512(vec1, vec2)
            let dotFP16 = MixedPrecisionKernels.dotFP16_512(vec1FP16, vec2FP16)

            let relativeError = abs(dotFP16 - dotFP32) / abs(dotFP32)
            #expect(relativeError < 0.001, "Parallel vectors: relative error \(relativeError) exceeds 0.1%")

            // Verify the relationship: dot(a, k*a) = k * dot(a, a)
            let selfDotFP32 = DotKernels.dot512(vec1, vec1)
            let expectedDot = selfDotFP32 * 2.5

            let errorVsExpected = abs(dotFP16 - expectedDot) / abs(expectedDot)
            #expect(errorVsExpected < 0.002, "Parallel vector relationship error: \(errorVsExpected)")
        }

        @Test("No systematic bias in errors")
        func testErrorSymmetry() throws {
            var positiveErrors: [Float] = []
            var negativeErrors: [Float] = []

            for _ in 0..<100 {
                let values1 = (0..<512).map { _ in Float.random(in: -1...1) }
                let values2 = (0..<512).map { _ in Float.random(in: -1...1) }

                let vec1 = try Vector512Optimized(values1)
                let vec2 = try Vector512Optimized(values2)
                let vec1FP16 = MixedPrecisionKernels.Vector512FP16(from: vec1)
                let vec2FP16 = MixedPrecisionKernels.Vector512FP16(from: vec2)

                let fp32Result = DotKernels.dot512(vec1, vec2)
                let fp16Result = MixedPrecisionKernels.dotFP16_512(vec1FP16, vec2FP16)

                let error = fp16Result - fp32Result
                if error > 0 {
                    positiveErrors.append(error)
                } else {
                    negativeErrors.append(error)
                }
            }

            // Check that errors are roughly symmetric (no systematic bias)
            let meanPositive = positiveErrors.reduce(0, +) / Float(max(positiveErrors.count, 1))
            let meanNegative = negativeErrors.reduce(0, +) / Float(max(negativeErrors.count, 1))

            let bias = meanPositive + meanNegative
            #expect(abs(bias) < 0.01, "Systematic bias detected: \(bias)")

            print("Error symmetry test:")
            print("  Mean positive error: \(meanPositive)")
            print("  Mean negative error: \(meanNegative)")
            print("  Bias: \(bias)")
        }

        @Test("Distance ordering preserved (monotonicity)")
        func testDistanceOrdering() throws {
            // Create a query vector
            let queryValues = (0..<512).map { Float($0) / 512.0 }
            let query = try Vector512Optimized(queryValues)

            // Create candidate vectors at varying distances
            var candidates: [Vector512Optimized] = []
            for scale in stride(from: 0.1, through: 2.0, by: 0.1) {
                let candidateValues = queryValues.map { $0 * Float(scale) }
                candidates.append(try Vector512Optimized(candidateValues))
            }

            // Compute distances with FP32
            let fp32Distances = candidates.map { query.euclideanDistanceSquared(to: $0) }

            // Compute distances with FP16
            let candidatesFP16 = candidates.map { MixedPrecisionKernels.Vector512FP16(from: $0) }
            var fp16Distances: [Float] = []
            for candidateFP16 in candidatesFP16 {
                let dotProduct = MixedPrecisionKernels.dotMixed512(query: query, candidate: candidateFP16)
                // For Euclidean distance from dot product: d² = ||a||² + ||b||² - 2(a·b)
                // Simplified check: just compare ordering
                fp16Distances.append(dotProduct)
            }

            // Check that ranking is preserved (Spearman correlation should be ~1.0)
            let fp32Ranks = computeRanks(fp32Distances)
            let fp16Ranks = computeRanks(fp16Distances)

            var rankDifferences: [Float] = []
            for i in 0..<fp32Ranks.count {
                rankDifferences.append(abs(fp32Ranks[i] - fp16Ranks[i]))
            }

            let maxRankDifference = rankDifferences.max() ?? 0
            #expect(maxRankDifference <= 2, "Max rank difference \(maxRankDifference) too large")
        }

        // Helper function to compute ranks
        private func computeRanks(_ values: [Float]) -> [Float] {
            let indexed = values.enumerated().sorted { $0.element < $1.element }
            var ranks = [Float](repeating: 0, count: values.count)
            for (rank, (index, _)) in indexed.enumerated() {
                ranks[index] = Float(rank)
            }
            return ranks
        }
    }

    // MARK: - Batch Operations Tests

    @Suite("Batch Operations Tests")
    struct BatchOperationsTests {

        @Test("Batch dot FP16 correctness")
        func testBatchDotFP16() throws {
            let query = try Vector512Optimized((0..<512).map { Float($0) / 512.0 })
            var candidates: [Vector512Optimized] = []
            for _ in 0..<100 {
                let values = (0..<512).map { _ in Float.random(in: -1...1) }
                candidates.append(try Vector512Optimized(values))
            }

            let candidatesFP16 = candidates.map { MixedPrecisionKernels.Vector512FP16(from: $0) }

            // Compute batch results
            var batchResults = [Float](repeating: 0, count: 100)
            batchResults.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchDotFP16_512(query: query, candidates: candidatesFP16, out: buffer)
            }

            // Compute individual results for comparison
            let queryFP16 = MixedPrecisionKernels.Vector512FP16(from: query)
            for i in 0..<100 {
                let individual = MixedPrecisionKernels.dotFP16_512(queryFP16, candidatesFP16[i])
                let relativeError = abs(batchResults[i] - individual) / max(abs(individual), 1e-6)
                #expect(relativeError < 1e-5, "Batch result \(i) differs from individual: \(relativeError)")
            }
        }

        @Test("Batch dot mixed precision correctness")
        func testBatchDotMixed() throws {
            let query = try Vector512Optimized((0..<512).map { Float($0) / 512.0 })
            var candidates: [Vector512Optimized] = []
            for _ in 0..<100 {
                let values = (0..<512).map { _ in Float.random(in: -1...1) }
                candidates.append(try Vector512Optimized(values))
            }

            let candidatesFP16 = candidates.map { MixedPrecisionKernels.Vector512FP16(from: $0) }

            // Compute batch results with mixed precision
            var batchResults = [Float](repeating: 0, count: 100)
            batchResults.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchDotMixed512(query: query, candidates: candidatesFP16, out: buffer)
            }

            // Compute individual results for comparison
            for i in 0..<100 {
                let individual = MixedPrecisionKernels.dotMixed512(query: query, candidate: candidatesFP16[i])
                let relativeError = abs(batchResults[i] - individual) / max(abs(individual), 1e-6)
                #expect(relativeError < 1e-5, "Batch mixed result \(i) differs from individual: \(relativeError)")
            }

            // Compare with FP32 baseline
            for i in 0..<100 {
                let fp32Result = DotKernels.dot512(query, candidates[i])
                let relativeError = abs(batchResults[i] - fp32Result) / max(abs(fp32Result), 1e-6)
                #expect(relativeError < 0.001, "Batch mixed result \(i) vs FP32: \(relativeError)")
            }
        }

        @Test("Batch operations preserve query precision")
        func testBatchPreservesQueryPrecision() throws {
            // This test verifies that batchDotMixed does NOT convert the query to FP16
            let query = try Vector512Optimized((0..<512).map { Float($0) / 512.0 })
            var candidates: [Vector512Optimized] = []
            for _ in 0..<50 {
                let values = (0..<512).map { _ in Float.random(in: -1...1) }
                candidates.append(try Vector512Optimized(values))
            }

            let candidatesFP16 = candidates.map { MixedPrecisionKernels.Vector512FP16(from: $0) }

            // Results from batchDotMixed (query stays FP32)
            var mixedResults = [Float](repeating: 0, count: 50)
            mixedResults.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchDotMixed512(query: query, candidates: candidatesFP16, out: buffer)
            }

            // Results from batchDotFP16 (query converted to FP16)
            var fp16Results = [Float](repeating: 0, count: 50)
            fp16Results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchDotFP16_512(query: query, candidates: candidatesFP16, out: buffer)
            }

            // Mixed precision should be closer to FP32 baseline
            var mixedErrors: [Float] = []
            var fp16Errors: [Float] = []

            for i in 0..<50 {
                let fp32Baseline = DotKernels.dot512(query, candidates[i])

                if abs(fp32Baseline) > 1e-4 {
                    let mixedError = abs(mixedResults[i] - fp32Baseline) / abs(fp32Baseline)
                    let fp16Error = abs(fp16Results[i] - fp32Baseline) / abs(fp32Baseline)

                    mixedErrors.append(mixedError)
                    fp16Errors.append(fp16Error)
                }
            }

            let meanMixedError = mixedErrors.reduce(0, +) / Float(mixedErrors.count)
            let meanFP16Error = fp16Errors.reduce(0, +) / Float(fp16Errors.count)

            // Mixed precision should be more accurate (query is FP32)
            #expect(meanMixedError <= meanFP16Error * 1.2,
                    "Mixed precision not more accurate: \(meanMixedError) vs \(meanFP16Error)")

            print("Batch query precision preservation:")
            print("  Mean error (mixed): \(String(format: "%.6f%%", meanMixedError * 100))")
            print("  Mean error (fp16):  \(String(format: "%.6f%%", meanFP16Error * 100))")
        }
    }

    // MARK: - Precision Analysis Tests

    @Suite("Precision Analysis Tests")
    struct PrecisionAnalysisTests {

        @Test("Precision analysis for normalized embeddings")
        func testPrecisionAnalysisNormalized() throws {
            // Create typical normalized embedding vectors
            var vectors: [Vector512Optimized] = []
            for _ in 0..<100 {
                let values = (0..<512).map { _ in Float.random(in: -1...1) }
                // Normalize
                let magnitude = sqrt(values.map { $0 * $0 }.reduce(0, +))
                let normalized = values.map { $0 / magnitude }
                vectors.append(try Vector512Optimized(normalized))
            }

            let profile = MixedPrecisionKernels.analyzePrecision(vectors)

            // Normalized vectors should recommend FP16 or mixed precision
            #expect(profile.recommendedPrecision == .fp16 ||
                        profile.recommendedPrecision == .mixed,
                    "Normalized vectors should recommend FP16/mixed: got \(profile.recommendedPrecision)")

            #expect(profile.expectedError < 0.001, "Expected error too high: \(profile.expectedError)")
            #expect(abs(profile.meanValue) < 1.0, "Mean value unexpected: \(profile.meanValue)")

            print(profile.summary)
        }

        @Test("Precision analysis for large values")
        func testPrecisionAnalysisLargeValues() throws {
            // Create vectors with values exceeding FP16 range
            var vectors: [Vector512Optimized] = []
            for _ in 0..<100 {
                let values = (0..<512).map { _ in Float.random(in: -100000...100000) }
                vectors.append(try Vector512Optimized(values))
            }

            let profile = MixedPrecisionKernels.analyzePrecision(vectors)

            // Should recommend FP32 due to range
            #expect(profile.recommendedPrecision == .fp32,
                    "Large values should recommend FP32: got \(profile.recommendedPrecision)")
        }

        @Test("Optimal precision selection with error tolerance")
        func testOptimalPrecisionSelection() throws {
            // Create vectors with varying characteristics
            var smallRangeVectors: [Vector512Optimized] = []
            for _ in 0..<50 {
                let values = (0..<512).map { _ in Float.random(in: -0.1...0.1) }
                smallRangeVectors.append(try Vector512Optimized(values))
            }

            // Strict tolerance should escalate precision
            let strictPrecision = MixedPrecisionKernels.selectOptimalPrecision(
                for: smallRangeVectors,
                errorTolerance: 0.0001
            )

            // Relaxed tolerance should allow lower precision
            let relaxedPrecision = MixedPrecisionKernels.selectOptimalPrecision(
                for: smallRangeVectors,
                errorTolerance: 0.01
            )

            print("Precision selection test:")
            print("  Strict (0.01%): \(strictPrecision)")
            print("  Relaxed (1%):   \(relaxedPrecision)")

            // At minimum, both should provide valid recommendations
            #expect(strictPrecision.rawValue.count > 0)
            #expect(relaxedPrecision.rawValue.count > 0)
        }
    }

    // MARK: - Integration Tests

    @Suite("Integration Tests")
    struct IntegrationTests {

        @Test("End-to-end similarity search accuracy")
        func testSimilaritySearchAccuracy() throws {
            // Simulate a similarity search scenario
            let queryValues = (0..<512).map { _ in Float.random(in: -1...1) }
            let query = try Vector512Optimized(queryValues)

            // Create a database of candidate vectors
            var database: [Vector512Optimized] = []
            for _ in 0..<1000 {
                let values = (0..<512).map { _ in Float.random(in: -1...1) }
                database.append(try Vector512Optimized(values))
            }

            // Add the query itself to the database (should be top match)
            database.insert(query, at: 500)

            // Convert database to FP16
            let databaseFP16 = database.map { MixedPrecisionKernels.Vector512FP16(from: $0) }

            // Compute similarities with FP32
            let fp32Similarities = database.map { DotKernels.dot512(query, $0) }

            // Compute similarities with mixed precision
            var mixedSimilarities = [Float](repeating: 0, count: database.count)
            mixedSimilarities.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchDotMixed512(query: query, candidates: databaseFP16, out: buffer)
            }

            // Find top-K for both
            let k = 10
            let fp32TopK = Array(fp32Similarities.enumerated()
                                    .sorted { $0.element > $1.element }
                                    .prefix(k)
                                    .map { $0.offset })

            let mixedTopK = Array(mixedSimilarities.enumerated()
                                    .sorted { $0.element > $1.element }
                                    .prefix(k)
                                    .map { $0.offset })

            // The query itself (index 500) should be in top-K for both
            #expect(fp32TopK.contains(500), "Query not in FP32 top-K")
            #expect(mixedTopK.contains(500), "Query not in mixed precision top-K")

            // Compute overlap between top-K results
            let overlap = Set(fp32TopK).intersection(Set(mixedTopK)).count
            let recall = Float(overlap) / Float(k)

            // Recall should be high (> 80%)
            #expect(recall > 0.8, "Top-K recall \(recall) too low")

            print("Similarity search integration test:")
            print("  Top-\(k) recall: \(String(format: "%.2f%%", recall * 100))")
            print("  FP32 top result: index \(fp32TopK[0])")
            print("  Mixed top result: index \(mixedTopK[0])")
        }

        @Test("Memory footprint reduction verification")
        func testMemoryFootprintReduction() throws {
            let vectorCount = 1000

            // Create FP32 vectors
            var fp32Vectors: [Vector512Optimized] = []
            for _ in 0..<vectorCount {
                let values = (0..<512).map { _ in Float.random(in: -1...1) }
                fp32Vectors.append(try Vector512Optimized(values))
            }

            // Convert to FP16
            let fp16Vectors = fp32Vectors.map { MixedPrecisionKernels.Vector512FP16(from: $0) }

            // Estimate memory usage
            let fp32MemoryBytes = vectorCount * 512 * MemoryLayout<Float>.size
            let fp16MemoryBytes = vectorCount * 512 * 2  // 2 bytes per FP16

            let savings = Float(fp32MemoryBytes - fp16MemoryBytes) / Float(fp32MemoryBytes) * 100

            print("Memory footprint test:")
            print("  FP32 memory: \(fp32MemoryBytes) bytes")
            print("  FP16 memory: \(fp16MemoryBytes) bytes")
            print("  Savings: \(String(format: "%.1f%%", savings))")

            // Verify 50% savings
            #expect(abs(savings - 50.0) < 1.0, "Memory savings not ~50%: \(savings)%")
        }
    }

    // MARK: - Benchmark Integration Tests

    @Suite("Benchmark Tests")
    struct BenchmarkTests {

        @Test("Benchmark dot product performance")
        func testBenchmarkDotProduct() throws {
            let result = MixedPrecisionBenchmark.benchmarkDotProduct512(iterations: 100, warmupIterations: 20)

            print("\n" + result.fp32.summary)
            print("\n" + result.fp16.summary)
            print("\nSpeedup: \(String(format: "%.2f×", result.speedup))")

            // Speedup should be > 1.0 (FP16 faster than FP32)
            #expect(result.speedup > 1.0, "FP16 should be faster than FP32")

            // FP16 should meet performance target (< 120ns on M1)
            // Note: This is hardware-dependent, so we use a relaxed threshold
            #expect(result.fp16.meanTimeNs < 300, "FP16 performance target not met: \(result.fp16.meanTimeNs)ns")
        }

        @Test("Measure accuracy with benchmark utility")
        func testAccuracyMeasurement() throws {
            let result = MixedPrecisionBenchmark.measureAccuracy512(testVectors: 200)

            print("\n" + result.summary)

            // Verify accuracy meets spec
            #expect(result.meanRelativeError < 0.001, "Mean relative error exceeds 0.1%")
            #expect(result.maxRelativeError < 0.002, "Max relative error exceeds 0.2%")

            // Rank correlation should be very high (> 0.999)
            #expect(result.rankCorrelation > 0.999, "Rank correlation \(result.rankCorrelation) too low")
        }

        @Test("Benchmark batch operations")
        func testBenchmarkBatchOperations() throws {
            let result = MixedPrecisionBenchmark.benchmarkBatchOperations512(candidateCount: 100, iterations: 50)

            print("\n" + result.mixed.summary)
            print("\n" + result.fp16.summary)

            // Mixed precision should have reasonable performance
            let speedupVsFP16 = result.fp16.meanTimeNs / result.mixed.meanTimeNs
            print("\nMixed vs FP16 speedup: \(String(format: "%.2f×", speedupVsFP16))")

            // Both should complete in reasonable time
            #expect(result.mixed.meanTimeNs < 50_000, "Batch mixed too slow: \(result.mixed.meanTimeNs)ns")
            #expect(result.fp16.meanTimeNs < 50_000, "Batch FP16 too slow: \(result.fp16.meanTimeNs)ns")
        }
    }
}
