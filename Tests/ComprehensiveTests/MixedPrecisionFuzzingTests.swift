// Tests/ComprehensiveTests/MixedPrecisionFuzzingTests.swift

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

/// Property-based fuzzing tests for Mixed Precision Kernels
///
/// These tests use randomized inputs to discover edge cases and verify
/// invariants hold across a wide range of inputs.
@Suite("Mixed Precision Fuzzing Tests")
struct MixedPrecisionFuzzingTests {

    // MARK: - Fuzzing Configuration

    private static let fuzzingIterations = 1000
    private static let extremeFuzzingIterations = 10000

    // MARK: - Property: Conversion Invertibility

    @Suite("Conversion Invertibility Properties")
    struct ConversionInvertibilityTests {

        @Test("FP32→FP16→FP32 round-trip bounded error", arguments: [512, 768, 1536])
        func testRoundTripBoundedError(dimension: Int) throws {
            for _ in 0..<fuzzingIterations {
                // Generate random values in FP16 safe range
                let values = (0..<dimension).map { _ in
                    Float.random(in: -65000...65000)
                }

                let original = try createVector(values, dimension: dimension)
                let fp16 = convertToFP16(original, dimension: dimension)
                let reconstructed = convertToFP32(fp16, dimension: dimension)

                // Get concrete Float arrays for comparison
                // Note: Explicitly type as [Float] to avoid existential type issues
                // All test vectors use Float as Scalar type, so this is safe
                let originalArray: [Float] = original.toArray() as! [Float]
                let reconstructedArray: [Float] = reconstructed.toArray() as! [Float]

                for i in 0..<dimension {
                    let originalValue = originalArray[i]
                    let reconstructedValue = reconstructedArray[i]

                    if abs(originalValue) > 0.001 {
                        let relativeError = abs(reconstructedValue - originalValue) / abs(originalValue)
                        #expect(relativeError < 0.001,
                                "Dimension \(dimension), index \(i): relative error \(relativeError) exceeds 0.1%")
                    }
                }
            }
        }

        @Test("Edge case values preserve through conversion")
        func testEdgeCasePreservation() throws {
            let edgeCases: [Float] = [
                0.0, -0.0,
                Float.infinity, -Float.infinity,
                Float.nan,
                65504.0, -65504.0,  // FP16 max
                6.10352e-5,         // FP16 min normal
                1.0, -1.0,
                Float.pi, -Float.pi,
                Float.ulpOfOne
            ]

            for value in edgeCases {
                let vec = try Vector512Optimized(Array(repeating: value, count: 512))
                let fp16 = MixedPrecisionKernels.Vector512FP16(from: vec)
                let reconstructed = fp16.toFP32()

                for i in 0..<512 {
                    let orig = vec[i]
                    let recon = reconstructed[i]

                    if orig.isNaN {
                        #expect(recon.isNaN, "NaN not preserved at \(i)")
                    } else if orig.isInfinite {
                        #expect(recon.isInfinite && (orig < 0) == (recon < 0),
                                "Infinity not preserved at \(i)")
                    } else if orig == 0 {
                        #expect(abs(recon) < 1e-6, "Zero not preserved at \(i)")
                    }
                }
            }
        }

        @Test("Subnormal handling consistency")
        func testSubnormalHandling() throws {
            let subnormalValues: [Float] = [
                1e-6, 1e-7, 1e-8, 1e-10, 1e-20,
                -1e-6, -1e-7, -1e-8, -1e-10, -1e-20
            ]

            for value in subnormalValues {
                let vec = try Vector512Optimized(Array(repeating: value, count: 512))
                let fp16 = MixedPrecisionKernels.Vector512FP16(from: vec)
                let reconstructed = fp16.toFP32()

                // Subnormals may flush to zero or be preserved
                for i in 0..<512 {
                    let recon = reconstructed[i]
                    #expect(abs(recon) <= abs(value) * 1.1 || abs(recon) < 1e-4,
                            "Subnormal \(value) handling inconsistent at \(i): got \(recon)")
                }
            }
        }
    }

    // MARK: - Property: Dot Product Invariants

    @Suite("Dot Product Invariants")
    struct DotProductInvariantTests {

        @Test("Dot product commutivity: dot(a,b) = dot(b,a)")
        func testCommutativity() throws {
            for _ in 0..<fuzzingIterations {
                let values1 = (0..<512).map { _ in Float.random(in: -10...10) }
                let values2 = (0..<512).map { _ in Float.random(in: -10...10) }

                let vec1 = try Vector512Optimized(values1)
                let vec2 = try Vector512Optimized(values2)

                let vec1_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec1)
                let vec2_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec2)

                let dotAB = MixedPrecisionKernels.dotFP16_512(vec1_fp16, vec2_fp16)
                let dotBA = MixedPrecisionKernels.dotFP16_512(vec2_fp16, vec1_fp16)

                let relativeError = abs(dotAB - dotBA) / max(abs(dotAB), abs(dotBA), 1e-6)
                #expect(relativeError < 1e-5,
                        "Commutativity violated: dot(a,b)=\(dotAB), dot(b,a)=\(dotBA)")
            }
        }

        @Test("Dot product linearity: dot(a, k*b) = k * dot(a, b)")
        func testLinearity() throws {
            for _ in 0..<fuzzingIterations {
                let values1 = (0..<512).map { _ in Float.random(in: -10...10) }
                let values2 = (0..<512).map { _ in Float.random(in: -10...10) }
                let scalar = Float.random(in: -5...5)

                let vec1 = try Vector512Optimized(values1)
                let vec2 = try Vector512Optimized(values2)
                let vec2_scaled = try Vector512Optimized(values2.map { $0 * scalar })

                let vec1_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec1)
                let vec2_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec2)
                let vec2_scaled_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec2_scaled)

                let dotAB = MixedPrecisionKernels.dotFP16_512(vec1_fp16, vec2_fp16)
                let dotAKB = MixedPrecisionKernels.dotFP16_512(vec1_fp16, vec2_scaled_fp16)

                let expected = dotAB * scalar
                let relativeError = abs(dotAKB - expected) / max(abs(expected), 1e-6)

                #expect(relativeError < 0.01,  // Allow 1% error due to FP16 precision
                        "Linearity violated: dot(a, k*b)=\(dotAKB), k*dot(a,b)=\(expected), k=\(scalar)")
            }
        }

        @Test("Dot product self-consistency: dot(a,a) >= 0")
        func testSelfConsistency() throws {
            for _ in 0..<fuzzingIterations {
                let values = (0..<512).map { _ in Float.random(in: -100...100) }
                let vec = try Vector512Optimized(values)
                let vec_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec)

                let dotAA = MixedPrecisionKernels.dotFP16_512(vec_fp16, vec_fp16)

                #expect(dotAA >= -1e-4,  // Allow small negative due to numerical error
                        "Self dot product negative: \(dotAA)")
            }
        }

        @Test("Cauchy-Schwarz inequality: |dot(a,b)| <= ||a|| * ||b||")
        func testCauchySchwarz() throws {
            for _ in 0..<fuzzingIterations {
                let values1 = (0..<512).map { _ in Float.random(in: -10...10) }
                let values2 = (0..<512).map { _ in Float.random(in: -10...10) }

                let vec1 = try Vector512Optimized(values1)
                let vec2 = try Vector512Optimized(values2)
                let vec1_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec1)
                let vec2_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec2)

                let dotAB = abs(MixedPrecisionKernels.dotFP16_512(vec1_fp16, vec2_fp16))
                let normA = sqrt(MixedPrecisionKernels.dotFP16_512(vec1_fp16, vec1_fp16))
                let normB = sqrt(MixedPrecisionKernels.dotFP16_512(vec2_fp16, vec2_fp16))

                let bound = normA * normB
                let margin = bound * 0.01  // 1% margin for FP16 errors

                #expect(dotAB <= bound + margin,
                        "Cauchy-Schwarz violated: |dot(a,b)|=\(dotAB) > ||a||*||b||=\(bound)")
            }
        }
    }

    // MARK: - Property: Accuracy Bounds

    @Suite("Accuracy Bounds Properties")
    struct AccuracyBoundsTests {

        @Test("FP16 vs FP32 relative error bounded for normalized vectors")
        func testNormalizedVectorAccuracy() throws {
            var errors: [Float] = []

            for _ in 0..<fuzzingIterations {
                // Generate random normalized vectors
                let values1 = (0..<512).map { _ in Float.random(in: -1...1) }
                let values2 = (0..<512).map { _ in Float.random(in: -1...1) }

                let mag1 = sqrt(values1.map { $0 * $0 }.reduce(0, +))
                let mag2 = sqrt(values2.map { $0 * $0 }.reduce(0, +))

                let normalized1 = values1.map { $0 / mag1 }
                let normalized2 = values2.map { $0 / mag2 }

                let vec1 = try Vector512Optimized(normalized1)
                let vec2 = try Vector512Optimized(normalized2)
                let vec1_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec1)
                let vec2_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec2)

                let fp32Result = DotKernels.dot512(vec1, vec2)
                let fp16Result = MixedPrecisionKernels.dotFP16_512(vec1_fp16, vec2_fp16)

                if abs(fp32Result) > 1e-6 {
                    let relativeError = abs(fp16Result - fp32Result) / abs(fp32Result)
                    errors.append(relativeError)
                }
            }

            let maxError = errors.max() ?? 0
            let meanError = errors.reduce(0, +) / Float(errors.count)

            #expect(maxError < 0.002, "Max relative error \(maxError) exceeds 0.2%")
            #expect(meanError < 0.0008, "Mean relative error \(meanError) exceeds 0.08%")

            print("Normalized vector accuracy fuzzing:")
            print("  Max error:  \(String(format: "%.6f%%", maxError * 100))")
            print("  Mean error: \(String(format: "%.6f%%", meanError * 100))")
        }

        @Test("Mixed precision accuracy bounded")
        func testMixedPrecisionAccuracy() throws {
            var errors: [Float] = []

            for _ in 0..<fuzzingIterations {
                let values1 = (0..<512).map { _ in Float.random(in: -10...10) }
                let values2 = (0..<512).map { _ in Float.random(in: -10...10) }

                let query = try Vector512Optimized(values1)
                let candidate = try Vector512Optimized(values2)
                let candidate_fp16 = MixedPrecisionKernels.Vector512FP16(from: candidate)

                let fp32Result = DotKernels.dot512(query, candidate)
                let mixedResult = MixedPrecisionKernels.dotMixed512(query: query, candidate: candidate_fp16)

                if abs(fp32Result) > 1e-6 {
                    let relativeError = abs(mixedResult - fp32Result) / abs(fp32Result)
                    errors.append(relativeError)
                }
            }

            let maxError = errors.max() ?? 0
            let meanError = errors.reduce(0, +) / Float(errors.count)

            // Mixed precision should be even more accurate (query is FP32)
            #expect(maxError < 0.0015, "Max relative error \(maxError) exceeds 0.15%")
            #expect(meanError < 0.0006, "Mean relative error \(meanError) exceeds 0.06%")
        }
    }

    // MARK: - Property: Batch Consistency

    @Suite("Batch Operation Consistency")
    struct BatchConsistencyTests {

        @Test("Batch results match individual computations")
        func testBatchIndividualEquivalence() throws {
            for _ in 0..<100 {  // 100 trials
                let candidateCount = Int.random(in: 10...100)

                let queryValues = (0..<512).map { _ in Float.random(in: -10...10) }
                let query = try Vector512Optimized(queryValues)

                var candidates: [MixedPrecisionKernels.Vector512FP16] = []
                for _ in 0..<candidateCount {
                    let values = (0..<512).map { _ in Float.random(in: -10...10) }
                    let vec = try Vector512Optimized(values)
                    candidates.append(MixedPrecisionKernels.Vector512FP16(from: vec))
                }

                // Batch computation
                var batchResults = [Float](repeating: 0, count: candidateCount)
                batchResults.withUnsafeMutableBufferPointer { buffer in
                    MixedPrecisionKernels.batchDotMixed512(query: query, candidates: candidates, out: buffer)
                }

                // Individual computations
                for i in 0..<candidateCount {
                    let individual = MixedPrecisionKernels.dotMixed512(query: query, candidate: candidates[i])
                    let batchResult = batchResults[i]

                    let error = abs(batchResult - individual) / max(abs(individual), 1e-6)
                    #expect(error < 1e-5,
                            "Batch result \(i) differs from individual: batch=\(batchResult), individual=\(individual)")
                }
            }
        }

        @Test("Batch operations preserve ordering")
        func testBatchOrderingPreservation() throws {
            for _ in 0..<100 {
                let queryValues = (0..<512).map { _ in Float.random(in: -1...1) }
                let query = try Vector512Optimized(queryValues)

                // Create candidates with known ordering (scaled versions of query)
                var candidates: [MixedPrecisionKernels.Vector512FP16] = []
                var expectedOrder: [Float] = []

                for scale in stride(from: 0.1, to: 2.0, by: 0.1) {
                    let scaleFloat = Float(scale)
                    let candidateValues = queryValues.map { $0 * scaleFloat }
                    let vec = try Vector512Optimized(candidateValues)
                    candidates.append(MixedPrecisionKernels.Vector512FP16(from: vec))
                    expectedOrder.append(scaleFloat)
                }

                var batchResults = [Float](repeating: 0, count: candidates.count)
                batchResults.withUnsafeMutableBufferPointer { buffer in
                    MixedPrecisionKernels.batchDotMixed512(query: query, candidates: candidates, out: buffer)
                }

                // Results should be in the same order as expectedOrder
                let sortedResults = batchResults.enumerated().sorted { $0.element > $1.element }
                let sortedExpected = expectedOrder.enumerated().sorted { $0.element > $1.element }

                for i in 0..<min(5, sortedResults.count) {
                    let resultIndex = sortedResults[i].offset
                    let expectedIndex = sortedExpected[i].offset

                    #expect(abs(resultIndex - expectedIndex) <= 1,
                            "Ordering not preserved: top \(i) result index \(resultIndex) vs expected \(expectedIndex)")
                }
            }
        }
    }

    // MARK: - Property: Extreme Values

    @Suite("Extreme Value Handling")
    struct ExtremeValueTests {

        @Test("Very large values don't cause overflow in accumulation")
        func testLargeValueAccumulation() throws {
            for _ in 0..<fuzzingIterations {
                // Values near FP16 max
                let values1 = (0..<512).map { _ in Float.random(in: 60000...65000) }
                let values2 = (0..<512).map { _ in Float.random(in: 60000...65000) }

                let vec1 = try Vector512Optimized(values1)
                let vec2 = try Vector512Optimized(values2)
                let vec1_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec1)
                let vec2_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec2)

                let result = MixedPrecisionKernels.dotFP16_512(vec1_fp16, vec2_fp16)

                #expect(result.isFinite, "Large value accumulation resulted in non-finite: \(result)")
                #expect(result > 0, "Large value dot product should be positive")
            }
        }

        @Test("Very small values don't underflow to zero incorrectly")
        func testSmallValuePrecision() throws {
            for _ in 0..<fuzzingIterations {
                // Values near FP16 min normal
                let values1 = (0..<512).map { _ in Float.random(in: 1e-4...1e-3) }
                let values2 = (0..<512).map { _ in Float.random(in: 1e-4...1e-3) }

                let vec1 = try Vector512Optimized(values1)
                let vec2 = try Vector512Optimized(values2)
                let vec1_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec1)
                let vec2_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec2)

                let fp32Result = DotKernels.dot512(vec1, vec2)
                let fp16Result = MixedPrecisionKernels.dotFP16_512(vec1_fp16, vec2_fp16)

                // Should not be exactly zero (statistical impossibility)
                #expect(abs(fp16Result) > 1e-6, "Small values incorrectly underflowed to zero")

                // Relative error may be larger for small values
                if abs(fp32Result) > 1e-6 {
                    let relativeError = abs(fp16Result - fp32Result) / abs(fp32Result)
                    #expect(relativeError < 0.1,  // 10% tolerance for very small values
                            "Small value relative error too large: \(relativeError)")
                }
            }
        }

        @Test("Mixed positive and negative values cancel correctly")
        func testCancellation() throws {
            for _ in 0..<fuzzingIterations {
                // Create vectors with balanced positive/negative values
                var values1: [Float] = []
                var values2: [Float] = []

                for i in 0..<512 {
                    let sign: Float = (i % 2 == 0) ? 1.0 : -1.0
                    values1.append(sign * Float.random(in: 1...10))
                    values2.append(sign * Float.random(in: 1...10))
                }

                let vec1 = try Vector512Optimized(values1)
                let vec2 = try Vector512Optimized(values2)
                let vec1_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec1)
                let vec2_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec2)

                let fp32Result = DotKernels.dot512(vec1, vec2)
                let fp16Result = MixedPrecisionKernels.dotFP16_512(vec1_fp16, vec2_fp16)

                // Cancellation should still produce reasonable results
                #expect(fp16Result.isFinite, "Cancellation produced non-finite result")

                if abs(fp32Result) > 1e-3 {
                    let relativeError = abs(fp16Result - fp32Result) / abs(fp32Result)
                    #expect(relativeError < 0.05,  // 5% tolerance for cancellation scenarios
                            "Cancellation error too large: \(relativeError)")
                }
            }
        }
    }

    // MARK: - Property: Overflow Detection

    @Suite("Overflow Detection Properties")
    struct OverflowDetectionTests {

        @Test("Diagnostic overflow tracking works correctly")
        func testOverflowTracking() throws {
            MixedPrecisionDiagnostics.shared.enable()
            MixedPrecisionDiagnostics.shared.reset()

            // Create vectors with values that will overflow FP16
            let overflowValues = Array(repeating: Float(100000.0), count: 512)
            let normalValues = (0..<512).map { _ in Float.random(in: -1...1) }

            let overflowVec = try Vector512Optimized(overflowValues)
            let normalVec = try Vector512Optimized(normalValues)

            // This should trigger overflow tracking
            _ = MixedPrecisionKernels.Vector512FP16(from: overflowVec)
            _ = MixedPrecisionKernels.Vector512FP16(from: normalVec)

            let stats = MixedPrecisionDiagnostics.shared.getStatistics()

            print("Overflow tracking test:")
            print(stats.summary)

            // Should have detected overflows
            #expect(stats.totalConversions > 0, "No conversions tracked")
            #expect(stats.overflowToInfinity > 0, "Overflows not detected")

            MixedPrecisionDiagnostics.shared.disable()
        }
    }

    // MARK: - Helper Functions

    private static func createVector(_ values: [Float], dimension: Int) throws -> any VectorProtocol {
        switch dimension {
        case 512:
            return try Vector512Optimized(values)
        case 768:
            return try Vector768Optimized(values)
        case 1536:
            return try Vector1536Optimized(values)
        default:
            fatalError("Unsupported dimension: \(dimension)")
        }
    }

    private static func convertToFP16(_ vector: any VectorProtocol, dimension: Int) -> Any {
        switch dimension {
        case 512:
            return MixedPrecisionKernels.Vector512FP16(from: vector as! Vector512Optimized)
        case 768:
            return MixedPrecisionKernels.Vector768FP16(from: vector as! Vector768Optimized)
        case 1536:
            return MixedPrecisionKernels.Vector1536FP16(from: vector as! Vector1536Optimized)
        default:
            fatalError("Unsupported dimension: \(dimension)")
        }
    }

    private static func convertToFP32(_ fp16Vector: Any, dimension: Int) -> any VectorProtocol {
        switch dimension {
        case 512:
            return (fp16Vector as! MixedPrecisionKernels.Vector512FP16).toFP32()
        case 768:
            return (fp16Vector as! MixedPrecisionKernels.Vector768FP16).toFP32()
        case 1536:
            return (fp16Vector as! MixedPrecisionKernels.Vector1536FP16).toFP32()
        default:
            fatalError("Unsupported dimension: \(dimension)")
        }
    }
}
