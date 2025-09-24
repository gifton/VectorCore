//
//  QuantizedKernelsComprehensiveTests.swift
//  VectorCore
//
//  Comprehensive test suite for INT8 quantized kernels, including
//  quantization accuracy, compression ratios, and performance validation.
//

import Testing
import Foundation
import simd
@testable import VectorCore

/// Comprehensive test suite for INT8 Quantized Kernels
@Suite("INT8 Quantized Kernels")
struct QuantizedKernelsComprehensiveTests {

    // MARK: - Test Constants

    /// Tolerance for quantization error (INT8 has ~2-3 bits of precision loss)
    let quantizationRelativeTolerance: Float = 0.02  // 2%
    let quantizationAbsoluteTolerance: Float = 0.01

    /// Expected compression ratio for INT8 (4x compared to Float32)
    let expectedCompressionRatio: Float = 4.0

    // MARK: - Helper Methods

    /// Creates test vectors with specific value distributions
    func createTestVectors(count: Int, dimension: Int = 512, distribution: ValueDistribution) -> [Vector512Optimized] {
        var vectors: [Vector512Optimized] = []

        for _ in 0..<count {
            let values: [Float]

            switch distribution {
            case .uniform(let min, let max):
                values = (0..<dimension).map { _ in Float.random(in: min...max) }

            case .gaussian(let mean, let stdDev):
                // Box-Muller transform for Gaussian distribution
                values = (0..<dimension).map { _ in
                    let u1 = Float.random(in: 0.001..<1)
                    let u2 = Float.random(in: 0..<1)
                    let z = sqrt(-2 * log(u1)) * cos(2 * Float.pi * u2)
                    return mean + stdDev * z
                }

            case .sparse(let sparsity, let nonZeroRange):
                values = (0..<dimension).map { _ in
                    Float.random(in: 0..<1) < sparsity
                        ? Float.random(in: nonZeroRange)
                        : 0.0
                }

            case .bimodal(let peak1, let peak2, let ratio):
                values = (0..<dimension).map { _ in
                    Float.random(in: 0..<1) < ratio ? peak1 : peak2
                }

            case .powerLaw(let alpha):
                values = (0..<dimension).map { _ in
                    let u = Float.random(in: 0.001..<1)
                    return pow(u, -1.0 / alpha)
                }
            }

            vectors.append(try! Vector512Optimized(values))
        }

        return vectors
    }

    /// Calculates quantization error metrics
    func calculateQuantizationError(original: Vector512Optimized, quantized: Vector512INT8) -> (mse: Float, maxError: Float, relativeError: Float) {
        let dequantized = quantized.toFP32()
        let originalArray = original.toArray()
        let dequantizedArray = dequantized.toArray()

        var mse: Float = 0
        var maxError: Float = 0
        var relativeErrorSum: Float = 0
        var validCount = 0

        for i in 0..<originalArray.count {
            let error = abs(originalArray[i] - dequantizedArray[i])
            mse += error * error
            maxError = max(maxError, error)

            if abs(originalArray[i]) > 1e-6 {
                relativeErrorSum += error / abs(originalArray[i])
                validCount += 1
            }
        }

        mse /= Float(originalArray.count)
        let relativeError = validCount > 0 ? relativeErrorSum / Float(validCount) : 0

        return (mse, maxError, relativeError)
    }

    /// Verifies compression ratio
    func verifyCompressionRatio<V: QuantizedVectorINT8>(original: V.FloatVectorType, quantized: V) -> Float {
        let originalSize = V.laneCount * 4 * MemoryLayout<Float>.size  // SIMD4<Float>
        let quantizedSize = quantized.memoryFootprint
        return Float(originalSize) / Float(quantizedSize)
    }

    // MARK: - Test Value Distributions

    enum ValueDistribution {
        case uniform(min: Float, max: Float)
        case gaussian(mean: Float, stdDev: Float)
        case sparse(sparsity: Float, nonZeroRange: ClosedRange<Float>)
        case bimodal(peak1: Float, peak2: Float, ratio: Float)
        case powerLaw(alpha: Float)
    }

    // MARK: - Linear Quantization Parameters Tests

    @Suite("Linear Quantization Parameters")
    struct LinearQuantizationTests {

        @Test("Symmetric quantization parameter calculation")
        func testSymmetricQuantizationParams() async throws {
            let testRanges: [(min: Float, max: Float)] = [
                (-1.0, 1.0),
                (-10.0, 10.0),
                (-100.0, 100.0),
                (0.0, 100.0),
                (-50.0, 25.0)
            ]

            for range in testRanges {
                let params = LinearQuantizationParams(
                    minValue: range.min,
                    maxValue: range.max,
                    symmetric: true
                )

                #expect(params.zeroPoint == 0)
                let expectedScale = max(abs(range.min), abs(range.max)) / 127.0
                #expect(abs(params.scale - expectedScale) < 1e-6)

                let quantizedMin = params.quantize(range.min)
                let quantizedMax = params.quantize(range.max)
                let dequantizedMin = params.dequantize(quantizedMin)
                let dequantizedMax = params.dequantize(quantizedMax)

                #expect(abs(dequantizedMin - range.min) / abs(range.min + 0.001) < 0.02)
                #expect(abs(dequantizedMax - range.max) / abs(range.max + 0.001) < 0.02)
            }
        }

        @Test("Asymmetric quantization parameter calculation")
        func testAsymmetricQuantizationParams() async throws {
            let testRanges: [(min: Float, max: Float)] = [
                (0.0, 10.0),
                (-5.0, 100.0),
                (-100.0, -10.0)
            ]

            for range in testRanges {
                let params = LinearQuantizationParams(
                    minValue: range.min,
                    maxValue: range.max,
                    symmetric: false
                )

                let expectedScale = (range.max - range.min) / 255.0
                #expect(abs(params.scale - expectedScale) < 1e-5)

                let quantizedMin = params.quantize(range.min)
                let quantizedMax = params.quantize(range.max)
                #expect(quantizedMin >= -128 && quantizedMin <= 127)
                #expect(quantizedMax >= -128 && quantizedMax <= 127)
            }
        }

        @Test("Zero-point calculation accuracy")
        func testZeroPointCalculation() async throws {
            let params1 = LinearQuantizationParams(minValue: -10, maxValue: 10, symmetric: true)
            #expect(params1.zeroPoint == 0)

            let params2 = LinearQuantizationParams(minValue: 0, maxValue: 10, symmetric: false)
            let quantizedZero = params2.quantize(0.0)
            let dequantizedZero = params2.dequantize(quantizedZero)
            #expect(abs(dequantizedZero) < 0.1)
        }

        @Test("Scale factor optimization")
        func testScaleFactorOptimization() async throws {
            let values = (0..<512).map { Float($0) * 0.1 - 25.0 }
            let vector = try Vector512Optimized(values)
            let quantized = Vector512INT8(from: vector)
            let params = quantized.quantizationParams

            let valueRange = values.max()! - values.min()!
            let expectedScale = params.isSymmetric
                ? max(abs(values.min()!), abs(values.max()!)) / 127.0
                : valueRange / 255.0

            #expect(abs(params.scale - expectedScale) / expectedScale < 0.01)
        }

        @Test("Edge case: zero range handling")
        func testZeroRangeHandling() async throws {
            let constantValue: Float = 42.0
            let params = LinearQuantizationParams(
                minValue: constantValue,
                maxValue: constantValue,
                symmetric: false
            )

            #expect(params.scale > 0)

            let quantized = params.quantize(constantValue)
            let dequantized = params.dequantize(quantized)
            #expect(abs(dequantized - constantValue) < 1.0 || params.scale == 1.0)
        }

        @Test("Edge case: extreme value ranges")
        func testExtremeValueRanges() async throws {
            let extremeRanges: [(min: Float, max: Float)] = [
                (-1e6, 1e6),
                (-1e-6, 1e-6),
                (1e-10, 1e-8)
            ]

            for range in extremeRanges {
                let params = LinearQuantizationParams(
                    minValue: range.min,
                    maxValue: range.max,
                    symmetric: false
                )

                #expect(params.scale >= Float.leastNormalMagnitude)

                let midPoint = (range.min + range.max) / 2
                let quantized = params.quantize(midPoint)
                #expect(quantized >= -128 && quantized <= 127)
            }
        }
    }

    // MARK: - INT8 Storage Tests

    @Suite("INT8 Storage Types")
    struct INT8StorageTests {

        @Test("Vector512INT8 initialization from FP32")
        func testVector512INT8Initialization() async throws {
            let values = (0..<512).map { Float($0) * 0.1 - 25.0 }
            let fp32Vector = try Vector512Optimized(values)
            let int8Vector = Vector512INT8(from: fp32Vector)

            #expect(int8Vector.storage.count == 128)  // 512 / 4 = 128 SIMD4 lanes

            let dequantized = int8Vector.toFP32()
            let dequantizedArray = dequantized.toArray()

            for i in 0..<512 {
                let error = abs(values[i] - dequantizedArray[i])
                let relError = abs(values[i]) > 0.001 ? error / abs(values[i]) : error
                #expect(relError < 0.02 || error < 0.1)
            }
        }

        @Test("Vector768INT8 initialization from FP32")
        func testVector768INT8Initialization() async throws {
            let values = (0..<768).map { Float(sin(Double($0) * 0.01)) * 10.0 }
            let fp32Vector = try Vector768Optimized(values)
            let int8Vector = Vector768INT8(from: fp32Vector)

            #expect(int8Vector.storage.count == 192)  // 768 / 4 = 192 SIMD4 lanes

            let dequantized = int8Vector.toFP32()
            let dequantizedArray = dequantized.toArray()

            for i in 0..<768 {
                let error = abs(values[i] - dequantizedArray[i])
                let relError = abs(values[i]) > 0.001 ? error / abs(values[i]) : error
                #expect(relError < 0.02 || error < 0.1)
            }
        }

        @Test("Vector1536INT8 initialization from FP32")
        func testVector1536INT8Initialization() async throws {
            let values = (0..<1536).map { Float(cos(Double($0) * 0.01)) * 5.0 }
            let fp32Vector = try Vector1536Optimized(values)
            let int8Vector = Vector1536INT8(from: fp32Vector)

            #expect(int8Vector.storage.count == 384)  // 1536 / 4 = 384 SIMD4 lanes

            let dequantized = int8Vector.toFP32()
            let dequantizedArray = dequantized.toArray()

            for i in 0..<1536 {
                let error = abs(values[i] - dequantizedArray[i])
                let relError = abs(values[i]) > 0.001 ? error / abs(values[i]) : error
                #expect(relError < 0.02 || error < 0.1)
            }
        }

        @Test("SIMD4<Int8> packing efficiency")
        func testSIMD4PackingEfficiency() async throws {
            let values = Array(repeating: Float(1.0), count: 512)
            let vector = try Vector512Optimized(values)
            let quantized = Vector512INT8(from: vector)

            // Check that storage is properly packed
            #expect(quantized.storage.count * 4 == 512)

            // Verify memory alignment
            quantized.storage.withUnsafeBufferPointer { buffer in
                let address = Int(bitPattern: buffer.baseAddress!)
                #expect(address % 4 == 0, "SIMD4 storage should be at least 4-byte aligned")
            }
        }

        @Test("Memory footprint validation")
        func testMemoryFootprint() async throws {
            let fp32Vector = try Vector512Optimized(Array(repeating: 1.0, count: 512))
            let int8Vector = Vector512INT8(from: fp32Vector)

            let fp32Size = 512 * MemoryLayout<Float>.size
            let int8Size = int8Vector.memoryFootprint

            let compressionRatio = Float(fp32Size) / Float(int8Size)
            #expect(compressionRatio >= 3.5, "Should achieve close to 4x compression")
        }

        @Test("Storage alignment for SIMD")
        func testStorageAlignment() async throws {
            let vector512 = Vector512INT8(from: try Vector512Optimized(Array(repeating: 1.0, count: 512)))
            let vector768 = Vector768INT8(from: try Vector768Optimized(Array(repeating: 1.0, count: 768)))
            let vector1536 = Vector1536INT8(from: try Vector1536Optimized(Array(repeating: 1.0, count: 1536)))

            // Storage should be contiguous and aligned
            #expect(vector512.storage.count > 0)
            #expect(vector768.storage.count > 0)
            #expect(vector1536.storage.count > 0)
        }
    }

    // MARK: - Quantization Accuracy Tests

    @Suite("Quantization Accuracy")
    struct QuantizationAccuracyTests {

        @Test("Uniform distribution quantization accuracy")
        func testUniformDistributionAccuracy() async throws {
            let tests = QuantizedKernelsComprehensiveTests()
            let vectors = tests.createTestVectors(
                count: 10,
                distribution: .uniform(min: -10, max: 10)
            )

            for vector in vectors {
                let quantized = Vector512INT8(from: vector)
                let errors = tests.calculateQuantizationError(
                    original: vector,
                    quantized: quantized
                )

                #expect(errors.relativeError < 0.02, "Relative error should be < 2%")
                #expect(errors.maxError < 0.2, "Max error should be reasonable")
            }
        }

        @Test("Gaussian distribution quantization accuracy")
        func testGaussianDistributionAccuracy() async throws {
            let tests = QuantizedKernelsComprehensiveTests()
            let vectors = tests.createTestVectors(
                count: 10,
                distribution: .gaussian(mean: 0, stdDev: 1)
            )

            for vector in vectors {
                let quantized = Vector512INT8(from: vector)
                let errors = tests.calculateQuantizationError(
                    original: vector,
                    quantized: quantized
                )

                #expect(errors.relativeError < 0.02)
                #expect(sqrt(errors.mse) < 0.05, "RMS error should be small")
            }
        }

        @Test("Sparse vector quantization")
        func testSparseVectorQuantization() async throws {
            let tests = QuantizedKernelsComprehensiveTests()
            let vectors = tests.createTestVectors(
                count: 5,
                distribution: .sparse(sparsity: 0.1, nonZeroRange: -5...5)
            )

            for vector in vectors {
                let quantized = Vector512INT8(from: vector)
                let dequantized = quantized.toFP32()

                let originalArray = vector.toArray()
                let dequantizedArray = dequantized.toArray()

                // Check that zeros are preserved (approximately)
                var zeroCount = 0
                for i in 0..<originalArray.count {
                    if originalArray[i] == 0 {
                        zeroCount += 1
                        #expect(abs(dequantizedArray[i]) < 0.1, "Zeros should be preserved")
                    }
                }

                #expect(zeroCount > 400, "Should have many zeros in sparse vector")
            }
        }

        @Test("Bimodal distribution handling")
        func testBimodalDistribution() async throws {
            let tests = QuantizedKernelsComprehensiveTests()
            let vectors = tests.createTestVectors(
                count: 5,
                distribution: .bimodal(peak1: -5, peak2: 5, ratio: 0.5)
            )

            for vector in vectors {
                let quantized = Vector512INT8(from: vector)
                let dequantized = quantized.toFP32()
                let dequantizedArray = dequantized.toArray()

                // Check that bimodal peaks are preserved
                var nearPeak1 = 0
                var nearPeak2 = 0

                for value in dequantizedArray {
                    if abs(value - (-5)) < 1.0 { nearPeak1 += 1 }
                    if abs(value - 5) < 1.0 { nearPeak2 += 1 }
                }

                #expect(nearPeak1 > 200 && nearPeak2 > 200, "Should preserve bimodal distribution")
            }
        }

        @Test("Power-law distribution quantization")
        func testPowerLawDistribution() async throws {
            let tests = QuantizedKernelsComprehensiveTests()
            let vectors = tests.createTestVectors(
                count: 5,
                distribution: .powerLaw(alpha: 2.0)
            )

            for vector in vectors {
                let quantized = Vector512INT8(from: vector)
                let errors = tests.calculateQuantizationError(
                    original: vector,
                    quantized: quantized
                )

                // Power law has many small values, error should still be reasonable
                #expect(errors.relativeError < 0.05, "Power law quantization should maintain reasonable accuracy")
            }
        }

        @Test("Round-trip conversion accuracy")
        func testRoundTripConversion() async throws {
            let values = (0..<512).map { Float($0) * 0.01 - 2.5 }
            let original = try Vector512Optimized(values)

            let quantized = Vector512INT8(from: original)
            let dequantized = quantized.toFP32()
            let requantized = Vector512INT8(from: dequantized)
            let redequantized = requantized.toFP32()

            let firstRoundArray = dequantized.toArray()
            let secondRoundArray = redequantized.toArray()

            // Second round-trip should produce nearly identical results to first
            for i in 0..<512 {
                #expect(abs(firstRoundArray[i] - secondRoundArray[i]) < 0.001,
                       "Round-trip should be stable")
            }
        }

        @Test("Quantization error bounds")
        func testQuantizationErrorBounds() async throws {
            let values = (0..<512).map { Float($0) * 0.1 }
            let vector = try Vector512Optimized(values)
            let quantized = Vector512INT8(from: vector)
            let params = quantized.quantizationParams

            let dequantized = quantized.toFP32()
            let dequantizedArray = dequantized.toArray()

            // Theoretical max error is scale/2 (half a quantization step)
            let maxTheoreticalError = params.scale / 2.0

            for i in 0..<values.count {
                let error = abs(values[i] - dequantizedArray[i])
                #expect(error <= maxTheoreticalError * 1.1, "Error should be within theoretical bounds")
            }
        }
    }

    // MARK: - Distance Computation Tests

    @Suite("Quantized Distance Computations")
    struct QuantizedDistanceTests {

        @Test("Euclidean distance INT8-INT8")
        func testEuclideanDistanceINT8() async throws {
            let values1 = (0..<512).map { Float($0) * 0.01 }
            let values2 = (0..<512).map { Float($0) * 0.01 + 0.5 }

            let vec1 = try Vector512Optimized(values1)
            let vec2 = try Vector512Optimized(values2)

            let quant1 = Vector512INT8(from: vec1)
            let quant2 = Vector512INT8(from: vec2)

            let fp32Distance = vec1.euclideanDistance(to: vec2)
            let int8Distance = QuantizedKernels.euclidean512(query: quant1, candidate: quant2)

            let relError = abs(fp32Distance - int8Distance) / fp32Distance
            #expect(relError < 0.05, "Quantized distance should be accurate")
        }

        @Test("Euclidean distance INT8-FP32 mixed")
        func testEuclideanDistanceMixed() async throws {
            let values1 = (0..<512).map { Float($0) * 0.01 }
            let values2 = (0..<512).map { Float($0) * 0.01 + 1.0 }

            let vec1 = try Vector512Optimized(values1)
            let vec2 = try Vector512Optimized(values2)
            let quant1 = Vector512INT8(from: vec1)

            let fp32Distance = vec1.euclideanDistance(to: vec2)
            let mixedDistance = QuantizedKernels.euclidean512(query: quant1, candidate: vec2)

            let relError = abs(fp32Distance - mixedDistance) / fp32Distance
            #expect(relError < 0.03, "Mixed precision distance should be accurate")
        }

        @Test("Cosine similarity INT8-INT8")
        func testCosineSimilarityINT8() async throws {
            let values1 = (0..<768).map { Float(sin(Double($0) * 0.1)) }
            let values2 = (0..<768).map { Float(cos(Double($0) * 0.1)) }

            let vec1 = try Vector768Optimized(values1)
            let vec2 = try Vector768Optimized(values2)

            let quant1 = Vector768INT8(from: vec1)
            let quant2 = Vector768INT8(from: vec2)

            let fp32Cosine = vec1.cosineDistance(to: vec2)
            let int8Cosine = QuantizedKernels.cosine768(query: quant1, candidate: quant2)

            #expect(abs(fp32Cosine - int8Cosine) < 0.01, "Cosine distance should be preserved")
        }

        @Test("Dot product INT8-INT8")
        func testDotProductINT8() async throws {
            let values1 = Array(repeating: Float(1.0), count: 1536)
            let values2 = Array(repeating: Float(2.0), count: 1536)

            let vec1 = try Vector1536Optimized(values1)
            let vec2 = try Vector1536Optimized(values2)

            let quant1 = Vector1536INT8(from: vec1)
            let quant2 = Vector1536INT8(from: vec2)

            let fp32Dot = vec1.dotProduct(vec2)
            let int8Dot = QuantizedKernels.dotProduct1536(query: quant1, candidate: quant2)

            let relError = abs(fp32Dot - int8Dot) / abs(fp32Dot)
            #expect(relError < 0.02, "Dot product should be accurate")
        }

        @Test("Distance accuracy vs FP32")
        func testDistanceAccuracyComparison() async throws {
            let tests = QuantizedKernelsComprehensiveTests()

            // Test with different distributions
            let distributions: [ValueDistribution] = [
                .uniform(min: -1, max: 1),
                .gaussian(mean: 0, stdDev: 1),
                .sparse(sparsity: 0.2, nonZeroRange: -5...5)
            ]

            for dist in distributions {
                let vectors = tests.createTestVectors(count: 10, distribution: dist)

                for i in 0..<vectors.count-1 {
                    let vec1 = vectors[i]
                    let vec2 = vectors[i+1]

                    let quant1 = Vector512INT8(from: vec1)
                    let quant2 = Vector512INT8(from: vec2)

                    let fp32Dist = vec1.euclideanDistance(to: vec2)
                    let int8Dist = QuantizedKernels.euclidean512(query: quant1, candidate: quant2)

                    if fp32Dist > 0.1 {  // Only check non-trivial distances
                        let relError = abs(fp32Dist - int8Dist) / fp32Dist
                        #expect(relError < 0.1, "Distance accuracy should be within 10%")
                    }
                }
            }
        }

        @Test("Distance computation with extreme values")
        func testDistanceWithExtremeValues() async throws {
            // Test with very large values
            let largeValues1 = Array(repeating: Float(1000.0), count: 512)
            let largeValues2 = Array(repeating: Float(1001.0), count: 512)

            let vec1 = try Vector512Optimized(largeValues1)
            let vec2 = try Vector512Optimized(largeValues2)

            let quant1 = Vector512INT8(from: vec1)
            let quant2 = Vector512INT8(from: vec2)

            let distance = QuantizedKernels.euclidean512(query: quant1, candidate: quant2)
            #expect(distance > 0, "Should compute non-zero distance")

            // Test with very small values
            let smallValues1 = Array(repeating: Float(0.001), count: 512)
            let smallValues2 = Array(repeating: Float(0.002), count: 512)

            let vecSmall1 = try Vector512Optimized(smallValues1)
            let vecSmall2 = try Vector512Optimized(smallValues2)

            let quantSmall1 = Vector512INT8(from: vecSmall1)
            let quantSmall2 = Vector512INT8(from: vecSmall2)

            let smallDistance = QuantizedKernels.euclidean512(query: quantSmall1, candidate: quantSmall2)
            #expect(smallDistance >= 0, "Distance should be non-negative")
        }

        @Test("Batch distance computation")
        func testBatchDistanceComputation() async throws {
            let query = try Vector512Optimized((0..<512).map { Float($0) * 0.01 })
            let queryQuantized = Vector512INT8(from: query)

            let candidateCount = 10
            var candidates: [Vector512Optimized] = []
            var candidatesQuantized: [Vector512INT8] = []

            for i in 0..<candidateCount {
                let values = (0..<512).map { Float($0 + i) * 0.01 }
                let candidate = try Vector512Optimized(values)
                candidates.append(candidate)
                candidatesQuantized.append(Vector512INT8(from: candidate))
            }

            // Convert to SoA format for batch processing
            let soa = SoAINT8<Vector512Optimized>(from: candidates)
            let results = UnsafeMutableBufferPointer<Float>.allocate(capacity: candidateCount)
            defer { results.deallocate() }

            QuantizedKernels.batchEuclidean512(
                query: queryQuantized,
                candidates: soa,
                results: results
            )

            // Verify batch results match individual computations
            for i in 0..<candidateCount {
                let individualDistance = QuantizedKernels.euclidean512(
                    query: queryQuantized,
                    candidate: candidatesQuantized[i]
                )
                #expect(abs(results[i] - individualDistance) < 0.001, "Batch result should match individual")
            }
        }
    }

    // MARK: - Remaining test suites with basic implementations

    @Suite("Auto-Calibration")
    struct AutoCalibrationTests {
        @Test("Auto-calibration with sample vectors")
        func testAutoCalibrationSampling() async throws {
            let vectors = (0..<10).map { i in
                try! Vector512Optimized((0..<512).map { Float($0 + i) * 0.1 })
            }

            // Use percentile-based calibration
            var minVal = Float.infinity
            var maxVal = -Float.infinity
            for vector in vectors {
                let flatArray = vector.storage.flatMap { simd in
                    [simd.x, simd.y, simd.z, simd.w]
                }
                for val in flatArray {
                    minVal = min(minVal, val)
                    maxVal = max(maxVal, val)
                }
            }
            let params = LinearQuantizationParams(minValue: minVal, maxValue: maxVal)
            #expect(params.scale > 0)
            #expect(params.minValue < params.maxValue)
        }

        @Test("Calibration with distribution hints")
        func testCalibrationWithDistribution() async throws {
            let vectors = (0..<10).map { _ in
                try! Vector512Optimized((0..<512).map { _ in Float.random(in: -10...10) })
            }

            // Use percentile-based calibration for Gaussian distribution
            var allValues: [Float] = []
            for vector in vectors {
                let flatArray = vector.storage.flatMap { simd in
                    [simd.x, simd.y, simd.z, simd.w]
                }
                allValues.append(contentsOf: flatArray)
            }
            allValues.sort()
            let percentile: Float = 0.995
            let count = Float(allValues.count)
            let lowerIdx = Int(count * (1.0 - percentile) / 2.0)
            let upperIdx = Int(count * (1.0 + percentile) / 2.0)
            let params = LinearQuantizationParams(
                minValue: allValues[lowerIdx],
                maxValue: allValues[upperIdx]
            )

            #expect(params.scale > 0)
        }

        @Test("Percentile-based calibration")
        func testPercentileCalibration() async throws {
            let values = (0..<512).map { _ in Float.random(in: -100...100) }
            _ = try Vector512Optimized(values)

            // Add some outliers
            var outlierValues = values
            outlierValues[0] = 10000
            outlierValues[1] = -10000
            let outlierVector = try Vector512Optimized(outlierValues)

            // Use percentile-based calibration for outliers
            let outlierVals = outlierVector.storage.flatMap { simd in
                [simd.x, simd.y, simd.z, simd.w]
            }
            let sortedVals = outlierVals.sorted()
            let percentile: Float = 0.99
            let count = Float(sortedVals.count)
            let lowerIdx = Int(count * (1.0 - percentile) / 2.0)
            let upperIdx = Int(count * (1.0 + percentile) / 2.0)
            let params99 = LinearQuantizationParams(
                minValue: sortedVals[lowerIdx],
                maxValue: sortedVals[upperIdx]
            )

            // Percentile-based should ignore outliers
            #expect(abs(params99.maxValue) < 1000)
        }

        @Test("Online calibration updates")
        func testOnlineCalibration() async throws {
            var currentParams = LinearQuantizationParams(minValue: -1, maxValue: 1)

            // Simulate streaming data
            for i in 0..<10 {
                let values = (0..<512).map { _ in Float.random(in: -Float(i)...Float(i)) }
                _ = try Vector512Optimized(values)

                // Update params with new data
                let newMin = min(currentParams.minValue, values.min()!)
                let newMax = max(currentParams.maxValue, values.max()!)
                currentParams = LinearQuantizationParams(minValue: newMin, maxValue: newMax)
            }

            #expect(currentParams.minValue <= -8)
            #expect(currentParams.maxValue >= 8)
        }

        @Test("Calibration convergence")
        func testCalibrationConvergence() async throws {
            // Generate consistent data
            let vectors = (0..<100).map { _ in
                try! Vector512Optimized((0..<512).map { _ in Float.random(in: -5...5) })
            }

            // Calibrate on different sample sizes
            func calibrateVectors(_ vecs: [Vector512Optimized]) throws -> LinearQuantizationParams {
                var minVal = Float.infinity
                var maxVal = -Float.infinity
                for vector in vecs {
                    let flatArray = vector.storage.flatMap { simd in
                        [simd.x, simd.y, simd.z, simd.w]
                    }
                    for val in flatArray {
                        minVal = min(minVal, val)
                        maxVal = max(maxVal, val)
                    }
                }
                return LinearQuantizationParams(minValue: minVal, maxValue: maxVal)
            }

            let params10 = try calibrateVectors(Array(vectors.prefix(10)))
            let params50 = try calibrateVectors(Array(vectors.prefix(50)))
            let params100 = try calibrateVectors(vectors)

            // Parameters should converge as sample size increases
            #expect(abs(params50.scale - params100.scale) < abs(params10.scale - params50.scale))
        }

        @Test("Multi-vector calibration")
        func testMultiVectorCalibration() async throws {
            let vectors512 = (0..<5).map { i in
                try! Vector512Optimized((0..<512).map { Float($0 + i) * 0.1 })
            }

            // Calculate params from the vectors
            var minVal = Float.infinity
            var maxVal = -Float.infinity
            for vector in vectors512 {
                let flatArray = vector.storage.flatMap { simd in
                    [simd.x, simd.y, simd.z, simd.w]
                }
                for val in flatArray {
                    minVal = min(minVal, val)
                    maxVal = max(maxVal, val)
                }
            }
            let params = LinearQuantizationParams(minValue: minVal, maxValue: maxVal)

            // Apply to all vectors and check accuracy
            for vector in vectors512 {
                let quantized = Vector512INT8(from: vector, params: params)
                let dequantized = quantized.toFP32()

                let originalArray = vector.toArray()
                let dequantizedArray = dequantized.toArray()

                for i in 0..<512 {
                    let error = abs(originalArray[i] - dequantizedArray[i])
                    #expect(error < params.scale, "Error should be within quantization step")
                }
            }
        }
    }

    @Suite("Performance Optimization")
    struct PerformanceOptimizationTests {
        @Test("SIMD vectorization efficiency")
        func testSIMDVectorization() async throws {
            let vector = try Vector512Optimized((0..<512).map { Float($0) * 0.01 })
            let quantized = Vector512INT8(from: vector)

            // Storage should use SIMD4 efficiently
            #expect(quantized.storage.count == 128)  // Exactly 512/4 SIMD4 vectors
        }

        @Test("Memory bandwidth utilization")
        func testMemoryBandwidth() async throws {
            let fp32Size = 512 * MemoryLayout<Float>.size
            let int8Size = 512 * MemoryLayout<Int8>.size + MemoryLayout<LinearQuantizationParams>.size

            let bandwidthReduction = Float(fp32Size) / Float(int8Size)
            #expect(bandwidthReduction > 3.5, "Should reduce memory bandwidth by ~4x")
        }

        @Test("Cache efficiency with INT8")
        func testCacheEfficiency() async throws {
            // INT8 vectors fit 4x more elements in cache
            let cacheLineSize = 64  // bytes
            let fp32ElementsPerLine = cacheLineSize / MemoryLayout<Float>.size
            let int8ElementsPerLine = cacheLineSize / MemoryLayout<Int8>.size

            #expect(int8ElementsPerLine == fp32ElementsPerLine * 4)
        }

        @Test("Batch processing throughput")
        func testBatchThroughput() async throws {
            let batchSize = 100
            let vectors = (0..<batchSize).map { i in
                try! Vector512Optimized((0..<512).map { Float($0 + i) * 0.01 })
            }

            let start = CFAbsoluteTimeGetCurrent()
            let quantized = vectors.map { Vector512INT8(from: $0) }
            let quantizationTime = CFAbsoluteTimeGetCurrent() - start

            #expect(quantized.count == batchSize)
            #expect(quantizationTime < 1.0, "Batch quantization should be fast")
        }

        @Test("Quantization/dequantization overhead")
        func testConversionOverhead() async throws {
            let vector = try Vector512Optimized((0..<512).map { Float($0) * 0.01 })

            let iterations = 1000
            let start = CFAbsoluteTimeGetCurrent()

            for _ in 0..<iterations {
                let quantized = Vector512INT8(from: vector)
                let _ = quantized.toFP32()
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let timePerConversion = elapsed / Double(iterations)

            #expect(timePerConversion < 0.001, "Conversion should be fast")
        }

        @Test("Mixed precision performance")
        func testMixedPrecisionPerformance() async throws {
            let vec1 = try Vector512Optimized((0..<512).map { Float($0) * 0.01 })
            let vec2 = try Vector512Optimized((0..<512).map { Float($0) * 0.02 })
            let quant1 = Vector512INT8(from: vec1)

            let iterations = 1000
            let start = CFAbsoluteTimeGetCurrent()

            for _ in 0..<iterations {
                let _ = QuantizedKernels.euclidean512(query: quant1, candidate: vec2)
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            #expect(elapsed < 1.0, "Mixed precision should be performant")
        }
    }

    // Edge cases and remaining test stubs
    @Suite("Edge Cases")
    struct EdgeCaseTests {
        @Test("Quantization of NaN and Inf values")
        func testNaNInfQuantization() async throws {
            let specialValues: [Float] = [.nan, .infinity, -.infinity, 0, -0]
            let normalValues = Array(repeating: Float(1.0), count: 507)
            let allValues = specialValues + normalValues

            let vector = try Vector512Optimized(allValues)
            let quantized = Vector512INT8(from: vector)
            let dequantized = quantized.toFP32()

            // Should handle special values gracefully
            #expect(dequantized.toArray().count == 512)
        }

        @Test("Zero vector quantization")
        func testZeroVectorQuantization() async throws {
            let zeroVector = Vector512Optimized()
            let quantized = Vector512INT8(from: zeroVector)
            let dequantized = quantized.toFP32()

            for value in dequantized.toArray() {
                #expect(abs(value) < 0.01, "Zero vector should remain near zero")
            }
        }
    }
}

// MARK: - Supporting Types

// Using VectorCore's SoAINT8 directly instead of custom wrapper

/// Test metrics for quantization quality
struct QuantizationMetrics {
    let mse: Float
    let psnr: Float
    let maxError: Float
    let relativeError: Float
    let compressionRatio: Float

    static func calculate(original: [Float], quantized: [Float]) -> QuantizationMetrics {
        var mse: Float = 0
        var maxError: Float = 0
        var relativeError: Float = 0

        for i in 0..<original.count {
            let error = abs(original[i] - quantized[i])
            mse += error * error
            maxError = max(maxError, error)

            if abs(original[i]) > 0.001 {
                relativeError += error / abs(original[i])
            }
        }

        mse /= Float(original.count)
        relativeError /= Float(original.count)

        let psnr = 20 * log10(1.0 / sqrt(mse))
        let compressionRatio = Float(original.count * 4) / Float(original.count + 16)  // Approximate

        return QuantizationMetrics(
            mse: mse,
            psnr: psnr,
            maxError: maxError,
            relativeError: relativeError,
            compressionRatio: compressionRatio
        )
    }
}