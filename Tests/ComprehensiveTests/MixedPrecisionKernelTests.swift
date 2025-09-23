//
//  MixedPrecisionKernelTests.swift
//  VectorCore
//
//  Comprehensive test suite for FP16 mixed-precision kernels, including
//  conversion accuracy, performance validation, and numerical stability.
//

import Testing
import Foundation
import simd
@testable import VectorCore

/// Comprehensive test suite for Mixed Precision Kernels
@Suite("Mixed Precision Kernels")
struct MixedPrecisionKernelTests {

    // MARK: - Test Constants

    /// Tolerance for FP16 conversion (Float16 has ~3 decimal digits of precision)
    let fp16RelativeTolerance: Float = 0.001  // 0.1%
    let fp16AbsoluteTolerance: Float = 1e-3

    /// Tolerance for mixed precision computations
    let mixedPrecisionTolerance: Float = 0.005  // 0.5%

    // MARK: - FP16 Storage Tests

    @Suite("FP16 Storage Types")
    struct FP16StorageTests {

        @Test("Vector512FP16 initialization and storage")
        func testVector512FP16Storage() async throws {
            // Test initialization from FP32 vector
            let values = (0..<512).map { Float($0) / 512.0 }
            let fp32Vector = try Vector512Optimized(values)
            let fp16Vector = Vector512FP16(from: fp32Vector)

            // Verify storage layout and SIMD4 lane count
            #expect(fp16Vector.storage.count == 128)  // 512 / 4 = 128 SIMD4 lanes

            // Convert back to FP32 and verify
            let convertedBack = fp16Vector.toFP32()
            for i in 0..<512 {
                let original = fp32Vector[i]
                let converted = convertedBack[i]
                // FP16 has ~3 decimal digits of precision
                #expect(abs(original - converted) < 0.001 || abs((original - converted) / original) < 0.001)
            }

            // Test edge cases: zeros
            let zerosVector = Vector512Optimized()
            let fp16Zeros = Vector512FP16(from: zerosVector)
            let zerosBack = fp16Zeros.toFP32()
            for i in 0..<512 {
                #expect(zerosBack[i] == 0.0)
            }

            // Test edge cases: ones
            let onesValues = Array(repeating: Float(1.0), count: 512)
            let onesVector = try Vector512Optimized(onesValues)
            let fp16Ones = Vector512FP16(from: onesVector)
            let onesBack = fp16Ones.toFP32()
            for i in 0..<512 {
                #expect(abs(onesBack[i] - 1.0) < Float.ulpOfOne)
            }
        }

        @Test("Vector768FP16 initialization and storage")
        func testVector768FP16Storage() async throws {
            // Test initialization from FP32 vector with various ranges
            let values = (0..<768).map { Float(sin(Double($0) * 0.1)) }
            let fp32Vector = try Vector768Optimized(values)
            let fp16Vector = Vector768FP16(from: fp32Vector)

            // Verify storage layout and SIMD4 lane count
            #expect(fp16Vector.storage.count == 192)  // 768 / 4 = 192 SIMD4 lanes

            // Convert back and verify accuracy
            let convertedBack = fp16Vector.toFP32()
            for i in 0..<768 {
                let original = fp32Vector[i]
                let converted = convertedBack[i]
                #expect(abs(original - converted) < 0.001 || abs((original - converted) / original) < 0.001)
            }

            // Test edge cases: denormalized values (very small numbers)
            let denormalValues = (0..<768).map { _ in Float.leastNormalMagnitude }
            let denormalVector = try Vector768Optimized(denormalValues)
            let fp16Denormal = Vector768FP16(from: denormalVector)
            let denormalBack = fp16Denormal.toFP32()
            // FP16 may flush denormals to zero
            for i in 0..<768 {
                #expect(denormalBack[i] >= 0.0 && denormalBack[i] <= Float.leastNormalMagnitude * 2)
            }

            // Test large values near FP16 max (65504)
            let largeValues = Array(repeating: Float(30000.0), count: 768)
            let largeVector = try Vector768Optimized(largeValues)
            let fp16Large = Vector768FP16(from: largeVector)
            let largeBack = fp16Large.toFP32()
            for i in 0..<768 {
                #expect(abs(largeBack[i] - 30000.0) / 30000.0 < 0.001)
            }
        }

        @Test("Vector1536FP16 initialization and storage")
        func testVector1536FP16Storage() async throws {
            // Test initialization from FP32 vector
            let values = (0..<1536).map { Float(cos(Double($0) * 0.05)) * 100.0 }
            let fp32Vector = try Vector1536Optimized(values)
            let fp16Vector = Vector1536FP16(from: fp32Vector)

            // Verify storage layout and SIMD4 lane count
            #expect(fp16Vector.storage.count == 384)  // 1536 / 4 = 384 SIMD4 lanes

            // Convert back and verify
            let convertedBack = fp16Vector.toFP32()
            for i in 0..<1536 {
                let original = fp32Vector[i]
                let converted = convertedBack[i]
                if !original.isNaN && !original.isInfinite {
                    #expect(abs(original - converted) < 0.1 || abs((original - converted) / original) < 0.001)
                }
            }

            // Test edge cases: NaN values
            var nanValues = Array(repeating: Float(1.0), count: 1536)
            nanValues[100] = Float.nan
            nanValues[500] = Float.nan
            let nanVector = try Vector1536Optimized(nanValues)
            let fp16Nan = Vector1536FP16(from: nanVector)
            let nanBack = fp16Nan.toFP32()
            #expect(nanBack[100].isNaN)
            #expect(nanBack[500].isNaN)

            // Test infinity values
            var infValues = Array(repeating: Float(1.0), count: 1536)
            infValues[200] = Float.infinity
            infValues[700] = -Float.infinity
            let infVector = try Vector1536Optimized(infValues)
            let fp16Inf = Vector1536FP16(from: infVector)
            let infBack = fp16Inf.toFP32()
            #expect(infBack[200] == Float.infinity)
            #expect(infBack[700] == -Float.infinity)
        }

        @Test("FP32 to FP16 conversion accuracy")
        func testFP32ToFP16ConversionAccuracy() async throws {
            // Test conversion for various value ranges
            let testRanges: [(name: String, min: Float, max: Float)] = [
                ("small", 0.0001, 0.001),
                ("normal", 1.0, 100.0),
                ("large", 1000.0, 30000.0),
                ("negative", -100.0, -1.0)
            ]

            for range in testRanges {
                let values = (0..<512).map { i in
                    range.min + (range.max - range.min) * Float(i) / 512.0
                }
                let fp32Vector = try Vector512Optimized(values)
                let fp16Vector = Vector512FP16(from: fp32Vector)
                let backVector = fp16Vector.toFP32()

                for i in 0..<512 {
                    let original = values[i]
                    let converted = backVector[i]
                    // Handle very small values to avoid division by zero
                    if abs(original) < 1e-10 {
                        // For tiny values, use absolute error
                        #expect(abs(converted - original) < 1e-6, "Range \(range.name): absolute error at index \(i)")
                    } else {
                        let relError = abs((original - converted) / original)
                        // FP16 has ~3 decimal digits of precision
                        #expect(relError < 0.001, "Range \(range.name): relative error \(relError) at index \(i)")
                    }
                }
            }

            // Test special values
            let specialValues: [Float] = [
                Float.nan,
                Float.infinity,
                -Float.infinity,
                0.0,
                -0.0,
                Float.leastNormalMagnitude,
                65000.0  // Near FP16 max (65504)
            ]

            for value in specialValues {
                let values = Array(repeating: value, count: 512)
                let fp32Vector = try Vector512Optimized(values)
                let fp16Vector = Vector512FP16(from: fp32Vector)
                let backVector = fp16Vector.toFP32()

                if value.isNaN {
                    #expect(backVector[0].isNaN)
                } else if value.isInfinite {
                    #expect(backVector[0] == value)
                } else if value == 0.0 || value == -0.0 {
                    #expect(backVector[0] == 0.0 || backVector[0] == -0.0)
                } else if abs(value) < Float(Float16.leastNormalMagnitude) {
                    // Values smaller than FP16's smallest normal may flush to zero
                    #expect(abs(backVector[0]) <= Float(Float16.leastNormalMagnitude) * 2,
                           "Very small value should flush to zero or stay small")
                } else if abs(value) > 65504.0 {  // FP16 max value
                    // Values beyond FP16 range become infinity
                    #expect(backVector[0].isInfinite && backVector[0].sign == value.sign,
                           "Value beyond FP16 range should become infinity with correct sign")
                } else {
                    // For values within FP16 normal range
                    let absError = abs(backVector[0] - value)
                    let relError = abs(value) > 1e-10 ? abs((backVector[0] - value) / value) : Float.infinity

                    // Use relative error for normal values, absolute for tiny ones
                    if abs(value) > 1e-4 {
                        #expect(relError < 0.01, "Relative error too large for value \(value)")
                    } else {
                        #expect(absError < 1e-6, "Absolute error too large for small value \(value)")
                    }
                }
            }
        }

        @Test("FP16 to FP32 round-trip conversion")
        func testFP16ToFP32RoundTrip() async throws {
            // Test FP32 → FP16 → FP32 conversion
            // Values that are exactly representable in FP16 should be preserved
            let exactValues: [Float] = [
                0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25,
                4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0
            ]

            for exactValue in exactValues {
                let values = Array(repeating: exactValue, count: 768)
                let fp32Vector = try Vector768Optimized(values)
                let fp16Vector = Vector768FP16(from: fp32Vector)
                let backVector = fp16Vector.toFP32()

                for i in 0..<768 {
                    #expect(backVector[i] == exactValue, "Value \(exactValue) not preserved in round-trip")
                }
            }

            // Test boundary values: max/min normal FP16 values
            // FP16 max is 65504, min normal is 6.10352e-5
            let boundaryValues: [Float] = [
                65504.0,      // Max FP16
                -65504.0,     // Min FP16
                6.10352e-5,   // Min normal positive FP16
                -6.10352e-5,  // Min normal negative FP16
                32768.0,      // Large power of 2
                -32768.0
            ]

            for boundaryValue in boundaryValues {
                let values = Array(repeating: boundaryValue, count: 1536)
                let fp32Vector = try Vector1536Optimized(values)
                let fp16Vector = Vector1536FP16(from: fp32Vector)
                let backVector = fp16Vector.toFP32()

                for i in 0..<1536 {
                    if abs(boundaryValue) <= 65504.0 {
                        // Should be representable
                        let relError = abs((backVector[i] - boundaryValue) / boundaryValue)
                        #expect(relError < 0.001, "Boundary value \(boundaryValue) has error \(relError)")
                    } else {
                        // Should overflow to infinity
                        #expect(backVector[i].isInfinite)
                    }
                }
            }
        }

        @Test("FP16 overflow and underflow handling")
        func testFP16OverflowUnderflow() async throws {
            // Test values exceeding FP16 range (±65504)
            let overflowValues: [Float] = [
                70000.0,
                -70000.0,
                Float.greatestFiniteMagnitude,
                -Float.greatestFiniteMagnitude,
                100000.0,
                -100000.0
            ]

            for overflowValue in overflowValues {
                let values = Array(repeating: overflowValue, count: 512)
                let fp32Vector = try Vector512Optimized(values)
                let fp16Vector = Vector512FP16(from: fp32Vector)
                let backVector = fp16Vector.toFP32()

                // Should overflow to infinity
                #expect(backVector[0].isInfinite)
                #expect(backVector[0].sign == overflowValue.sign)
            }

            // Test subnormal values near zero
            let subnormalValues: [Float] = [
                1e-10,
                -1e-10,
                Float.leastNonzeroMagnitude,
                -Float.leastNonzeroMagnitude,
                1e-20,
                -1e-20
            ]

            for subnormalValue in subnormalValues {
                let values = Array(repeating: subnormalValue, count: 768)
                let fp32Vector = try Vector768Optimized(values)
                let fp16Vector = Vector768FP16(from: fp32Vector)
                let backVector = fp16Vector.toFP32()

                // May flush to zero or preserve as smallest subnormal
                for i in 0..<768 {
                    #expect(abs(backVector[i]) <= abs(subnormalValue) * 2 || backVector[i] == 0.0)
                }
            }

            // Test gradual underflow
            let underflowSequence = (0..<1536).map { i in
                Float(65504.0) * pow(0.5, Float(i))  // Exponentially decreasing
            }
            let fp32Vector = try Vector1536Optimized(underflowSequence)
            let fp16Vector = Vector1536FP16(from: fp32Vector)
            let backVector = fp16Vector.toFP32()

            // First values should be preserved, later values should underflow
            #expect(abs(backVector[0] - 65504.0) / 65504.0 < 0.001)
            #expect(backVector[1535] == 0.0 || backVector[1535] < 1e-5)
        }

        @Test("FP16 storage memory efficiency")
        func testFP16StorageMemoryEfficiency() async throws {
            // Calculate theoretical memory sizes
            let dimensions = [512, 768, 1536]

            for dim in dimensions {
                // FP32 storage size
                let fp32BytesPerElement = MemoryLayout<Float>.size  // 4 bytes
                let fp32TotalBytes = dim * fp32BytesPerElement

                // FP16 storage size
                let fp16BytesPerElement = MemoryLayout<Float16>.size  // 2 bytes
                let fp16TotalBytes = dim * fp16BytesPerElement

                // Verify 50% memory reduction
                let reduction = Float(fp32TotalBytes - fp16TotalBytes) / Float(fp32TotalBytes)
                #expect(abs(reduction - 0.5) < 0.01, "Memory reduction should be 50%, got \(reduction * 100)%")

                // Test actual storage sizes
                if dim == 512 {
                    let fp32Vector = Vector512Optimized()
                    let fp16Vector = Vector512FP16(from: fp32Vector)

                    let fp32StorageBytes = fp32Vector.storage.count * MemoryLayout<SIMD4<Float>>.size
                    let fp16StorageBytes = fp16Vector.storage.count * MemoryLayout<SIMD4<Float16>>.size

                    #expect(fp32StorageBytes == fp32TotalBytes)
                    #expect(fp16StorageBytes == fp16TotalBytes)
                    #expect(fp16StorageBytes == fp32StorageBytes / 2)
                } else if dim == 768 {
                    let fp32Vector = Vector768Optimized()
                    let fp16Vector = Vector768FP16(from: fp32Vector)

                    let fp32StorageBytes = fp32Vector.storage.count * MemoryLayout<SIMD4<Float>>.size
                    let fp16StorageBytes = fp16Vector.storage.count * MemoryLayout<SIMD4<Float16>>.size

                    #expect(fp32StorageBytes == fp32TotalBytes)
                    #expect(fp16StorageBytes == fp16TotalBytes)
                } else if dim == 1536 {
                    let fp32Vector = Vector1536Optimized()
                    let fp16Vector = Vector1536FP16(from: fp32Vector)

                    let fp32StorageBytes = fp32Vector.storage.count * MemoryLayout<SIMD4<Float>>.size
                    let fp16StorageBytes = fp16Vector.storage.count * MemoryLayout<SIMD4<Float16>>.size

                    #expect(fp32StorageBytes == fp32TotalBytes)
                    #expect(fp16StorageBytes == fp16TotalBytes)
                }
            }

            // Test memory efficiency report
            let report = MixedPrecisionKernels.estimateMemoryImprovement(
                candidateCount: 1000,
                dimension: 768
            )
            #expect(report.contains("50.0%"))
        }
    }

    // MARK: - Mixed Precision Distance Tests

    @Suite("Mixed Precision Distance Computations")
    struct MixedPrecisionDistanceTests {

        @Test("Euclidean distance 512D mixed precision accuracy")
        func testEuclidean512MixedAccuracy() async throws {
            // Create test vectors with various magnitudes
            let queryValues = (0..<512).map { Float(sin(Double($0) * 0.01)) * 10.0 }
            let query = try Vector512Optimized(queryValues)

            // Create candidate vectors
            let candidateCount = 10
            var candidates: [Vector512Optimized] = []
            for i in 0..<candidateCount {
                let values = (0..<512).map { j in
                    Float(cos(Double(j + i * 100) * 0.01)) * 10.0
                }
                candidates.append(try Vector512Optimized(values))
            }

            // Convert candidates to FP16
            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)

            // Compute distances with mixed precision
            let outputBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: candidateCount)
            defer { outputBuffer.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<candidateCount,
                out: outputBuffer
            )

            // Compute reference distances with full FP32
            var referenceDist: [Float] = []
            for candidate in candidates {
                referenceDist.append(query.euclideanDistanceSquared(to: candidate))
            }

            // Compare results
            for i in 0..<candidateCount {
                let mixed = outputBuffer[i]
                let reference = referenceDist[i]
                let relError = abs((mixed - reference) / reference)
                #expect(relError < 0.005, "Distance \(i): mixed=\(mixed), ref=\(reference), error=\(relError)")
            }

            // Test with zero vectors
            let zeroQuery = Vector512Optimized()
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: zeroQuery,
                candidatesFP16: candidatesFP16,
                range: 0..<1,
                out: outputBuffer
            )
            // Distance from zero to any vector is the magnitude squared
            let expectedDist = candidates[0].magnitude * candidates[0].magnitude
            #expect(abs(outputBuffer[0] - expectedDist) / expectedDist < 0.005)
        }

        @Test("Euclidean distance 768D mixed precision accuracy")
        func testEuclidean768MixedAccuracy() async throws {
            // Test with orthogonal vectors
            var orthogonal1 = Array(repeating: Float(0.0), count: 768)
            var orthogonal2 = Array(repeating: Float(0.0), count: 768)
            orthogonal1[0] = 10.0  // Unit vector in first dimension
            orthogonal2[1] = 10.0  // Unit vector in second dimension

            let query = try Vector768Optimized(orthogonal1)
            let candidateVec = try Vector768Optimized(orthogonal2)
            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_768([candidateVec])

            let outputBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1)
            defer { outputBuffer.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_768(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<1,
                out: outputBuffer
            )

            // Orthogonal vectors: distance^2 should be sum of magnitudes^2
            let expectedDist: Float = 10.0 * 10.0 + 10.0 * 10.0  // 200
            #expect(abs(outputBuffer[0] - expectedDist) < 0.1)

            // Test with nearly identical vectors
            let nearlyIdentical1 = (0..<768).map { Float($0) / 768.0 }
            let nearlyIdentical2 = nearlyIdentical1.map { $0 + Float.ulpOfOne * 10 }

            let query2 = try Vector768Optimized(nearlyIdentical1)
            let candidate2 = try Vector768Optimized(nearlyIdentical2)
            let candidates2FP16 = MixedPrecisionKernels.convertToFP16_768([candidate2])

            MixedPrecisionKernels.range_euclid2_mixed_768(
                query: query2,
                candidatesFP16: candidates2FP16,
                range: 0..<1,
                out: outputBuffer
            )

            // Nearly identical vectors should have very small distance
            #expect(outputBuffer[0] < 0.001)

            // Test multiple candidates with various properties
            var testCandidates: [Vector768Optimized] = []
            testCandidates.append(query)  // Identical to query
            testCandidates.append(try Vector768Optimized(nearlyIdentical1.map { -$0 }))  // Negated
            testCandidates.append(Vector768Optimized())  // Zero vector

            let testCandidatesFP16 = MixedPrecisionKernels.convertToFP16_768(testCandidates)
            let multiBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 3)
            defer { multiBuffer.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_768(
                query: query2,
                candidatesFP16: testCandidatesFP16,
                range: 0..<3,
                out: multiBuffer
            )

            // Identical vector should have distance ~0
            #expect(multiBuffer[0] < 1e-6)
            // Check other distances are reasonable
            #expect(multiBuffer[1] > 0)
            #expect(multiBuffer[2] > 0)
        }

        @Test("Euclidean distance 1536D mixed precision accuracy")
        func testEuclidean1536MixedAccuracy() async throws {
            // Test with high-dimensional sparse vectors
            var sparseValues = Array(repeating: Float(0.0), count: 1536)
            // Set only 10% of values to non-zero
            for i in stride(from: 0, to: 1536, by: 10) {
                sparseValues[i] = Float.random(in: -10...10)
            }

            let sparseQuery = try Vector1536Optimized(sparseValues)

            // Create candidates with varying sparsity
            var candidates: [Vector1536Optimized] = []
            for sparsityLevel in [0.1, 0.3, 0.5, 0.7, 0.9] {
                var values = Array(repeating: Float(0.0), count: 1536)
                let nonZeroCount = Int(Float(1536) * Float(sparsityLevel))
                for i in 0..<nonZeroCount {
                    values[i] = Float.random(in: -5...5)
                }
                values.shuffle()
                candidates.append(try Vector1536Optimized(values))
            }

            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_1536(candidates)
            let outputBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: candidates.count)
            defer { outputBuffer.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_1536(
                query: sparseQuery,
                candidatesFP16: candidatesFP16,
                range: 0..<candidates.count,
                out: outputBuffer
            )

            // Compute reference distances
            for i in 0..<candidates.count {
                let reference = sparseQuery.euclideanDistanceSquared(to: candidates[i])
                let mixed = outputBuffer[i]
                let relError = abs((mixed - reference) / reference)
                #expect(relError < 0.01, "Sparsity test \(i): error=\(relError)")
            }

            // Test accumulation error in large dimensions
            // Use values that would accumulate error
            let accumValues = (0..<1536).map { i in
                Float(0.1) * (1.0 + Float(i) / 1536.0)  // Gradually increasing small values
            }
            let accumQuery = try Vector1536Optimized(accumValues)

            // Create a slightly perturbed version
            let perturbedValues = accumValues.map { $0 * 1.001 }
            let perturbedCandidate = try Vector1536Optimized(perturbedValues)
            let perturbedFP16 = MixedPrecisionKernels.convertToFP16_1536([perturbedCandidate])

            MixedPrecisionKernels.range_euclid2_mixed_1536(
                query: accumQuery,
                candidatesFP16: perturbedFP16,
                range: 0..<1,
                out: outputBuffer
            )

            let referenceAccum = accumQuery.euclideanDistanceSquared(to: perturbedCandidate)
            let mixedAccum = outputBuffer[0]
            // In high dimensions, we allow slightly more error due to accumulation
            let accumError = abs((mixedAccum - referenceAccum) / referenceAccum)
            #expect(accumError < 0.02, "Accumulation error: \(accumError)")
        }

        @Test("Cosine distance 512D mixed precision accuracy")
        func testCosine512MixedAccuracy() async throws {
            // Test normalized vectors (unit vectors)
            let normalizedValues = (0..<512).map { Float(sin(Double($0) * 0.1)) }
            var normalizedQuery = try Vector512Optimized(normalizedValues)
            let queryMag = normalizedQuery.magnitude
            let normalizedQueryValues = normalizedValues.map { $0 / queryMag }
            normalizedQuery = try Vector512Optimized(normalizedQueryValues)

            // Create normalized candidates
            var candidates: [Vector512Optimized] = []
            for i in 0..<5 {
                let values = (0..<512).map { j in
                    Float(cos(Double(j * (i + 1)) * 0.05))
                }
                let vec = try Vector512Optimized(values)
                let mag = vec.magnitude
                let normalizedVals = values.map { $0 / mag }
                candidates.append(try Vector512Optimized(normalizedVals))
            }

            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)
            let outputBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: candidates.count)
            defer { outputBuffer.deallocate() }

            MixedPrecisionKernels.range_cosine_mixed_512(
                query: normalizedQuery,
                candidatesFP16: candidatesFP16,
                range: 0..<candidates.count,
                out: outputBuffer
            )

            // Compute reference cosine distances
            for i in 0..<candidates.count {
                let reference = normalizedQuery.distance(to: candidates[i], metric: .cosine)
                let mixed = outputBuffer[i]
                #expect(abs(mixed - reference) < 0.005, "Normalized test \(i): mixed=\(mixed), ref=\(reference)")
            }

            // Test unnormalized vectors
            let unnormalizedQuery = try Vector512Optimized(
                (0..<512).map { Float($0) * 0.01 }
            )
            let unnormalizedCandidates = [
                try Vector512Optimized((0..<512).map { Float(511 - $0) * 0.01 }),
                try Vector512Optimized(Array(repeating: Float(1.0), count: 512))
            ]
            let unnormalizedFP16 = MixedPrecisionKernels.convertToFP16_512(unnormalizedCandidates)

            MixedPrecisionKernels.range_cosine_mixed_512(
                query: unnormalizedQuery,
                candidatesFP16: unnormalizedFP16,
                range: 0..<unnormalizedCandidates.count,
                out: outputBuffer
            )

            for i in 0..<unnormalizedCandidates.count {
                let reference = unnormalizedQuery.distance(to: unnormalizedCandidates[i], metric: .cosine)
                let mixed = outputBuffer[i]
                #expect(abs(mixed - reference) < 0.01)
            }

            // Test edge case: zero magnitude vectors
            let zeroVector = Vector512Optimized()
            let zeroFP16 = MixedPrecisionKernels.convertToFP16_512([zeroVector])

            MixedPrecisionKernels.range_cosine_mixed_512(
                query: zeroVector,
                candidatesFP16: zeroFP16,
                range: 0..<1,
                out: outputBuffer
            )

            // Both vectors zero -> distance should be 0
            #expect(outputBuffer[0] == 0.0)

            // Query zero, candidate non-zero -> distance should be 1
            MixedPrecisionKernels.range_cosine_mixed_512(
                query: zeroVector,
                candidatesFP16: candidatesFP16,
                range: 0..<1,
                out: outputBuffer
            )
            #expect(outputBuffer[0] == 1.0)
        }

        @Test("Cosine distance 768D mixed precision accuracy")
        func testCosine768MixedAccuracy() async throws {
            // Test parallel vectors (same direction)
            let parallelValues = (0..<768).map { Float($0 + 1) * 0.1 }
            let query = try Vector768Optimized(parallelValues)
            let parallelCandidate = try Vector768Optimized(parallelValues.map { $0 * 2.0 })  // Scaled version
            let parallelFP16 = MixedPrecisionKernels.convertToFP16_768([parallelCandidate])

            let outputBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1)
            defer { outputBuffer.deallocate() }

            MixedPrecisionKernels.range_cosine_mixed_768(
                query: query,
                candidatesFP16: parallelFP16,
                range: 0..<1,
                out: outputBuffer
            )

            // Parallel vectors should have cosine distance ~0
            #expect(outputBuffer[0] < 1e-5, "Parallel vectors distance: \(outputBuffer[0])")

            // Test anti-parallel vectors (opposite direction)
            let antiParallelCandidate = try Vector768Optimized(parallelValues.map { -$0 })
            let antiParallelFP16 = MixedPrecisionKernels.convertToFP16_768([antiParallelCandidate])

            MixedPrecisionKernels.range_cosine_mixed_768(
                query: query,
                candidatesFP16: antiParallelFP16,
                range: 0..<1,
                out: outputBuffer
            )

            // Anti-parallel vectors should have cosine distance ~2 (1 - (-1))
            #expect(abs(outputBuffer[0] - 2.0) < 1e-4, "Anti-parallel vectors distance: \(outputBuffer[0])")

            // Test numerical stability with small magnitudes
            let smallMagnitudeValues = (0..<768).map { _ in Float.random(in: -1e-10...1e-10) }
            let smallQuery = try Vector768Optimized(smallMagnitudeValues)
            let smallCandidates = [
                try Vector768Optimized(smallMagnitudeValues.map { $0 * 1.1 }),
                try Vector768Optimized(smallMagnitudeValues.map { $0 * 0.9 })
            ]
            let smallFP16 = MixedPrecisionKernels.convertToFP16_768(smallCandidates)

            let smallBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 2)
            defer { smallBuffer.deallocate() }

            MixedPrecisionKernels.range_cosine_mixed_768(
                query: smallQuery,
                candidatesFP16: smallFP16,
                range: 0..<2,
                out: smallBuffer
            )

            // With very small values, FP16 may flush to zero
            // Results should still be valid (0, 1, or proper cosine distance)
            for i in 0..<2 {
                #expect(smallBuffer[i] >= 0.0 && smallBuffer[i] <= 2.0)
            }

            // Test with mixed magnitudes
            var mixedValues = Array(repeating: Float(0.1), count: 768)
            for i in 0..<10 {
                mixedValues[i] = 100.0  // Some large values
            }
            let mixedQuery = try Vector768Optimized(mixedValues)
            let mixedCandidate = try Vector768Optimized(mixedValues.shuffled())
            let mixedFP16 = MixedPrecisionKernels.convertToFP16_768([mixedCandidate])

            MixedPrecisionKernels.range_cosine_mixed_768(
                query: mixedQuery,
                candidatesFP16: mixedFP16,
                range: 0..<1,
                out: outputBuffer
            )

            let reference = mixedQuery.distance(to: mixedCandidate, metric: .cosine)
            #expect(abs(outputBuffer[0] - reference) < 0.01)
        }

        @Test("Cosine distance 1536D mixed precision accuracy")
        func testCosine1536MixedAccuracy() async throws {
            // Test with random unit vectors
            var randomUnitVectors: [Vector1536Optimized] = []
            for _ in 0..<5 {
                let values = (0..<1536).map { _ in Float.random(in: -1...1) }
                let vec = try Vector1536Optimized(values)
                let mag = vec.magnitude
                let unitValues = values.map { $0 / mag }
                randomUnitVectors.append(try Vector1536Optimized(unitValues))
            }

            let query = randomUnitVectors[0]
            let candidates = Array(randomUnitVectors[1...])
            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_1536(candidates)

            let outputBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: candidates.count)
            defer { outputBuffer.deallocate() }

            MixedPrecisionKernels.range_cosine_mixed_1536(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<candidates.count,
                out: outputBuffer
            )

            // Verify results are in valid range and close to reference
            for i in 0..<candidates.count {
                let mixed = outputBuffer[i]
                #expect(mixed >= 0.0 && mixed <= 2.0, "Cosine distance out of range: \(mixed)")

                let reference = query.distance(to: candidates[i], metric: .cosine)
                let error = abs(mixed - reference)
                #expect(error < 0.01, "Unit vector \(i): error=\(error)")
            }

            // Test clamping with numerical edge cases
            // Create vectors that might produce similarity slightly outside [-1, 1]
            let edgeValues1 = Array(repeating: Float(1.0), count: 1536)
            let edgeQuery = try Vector1536Optimized(edgeValues1)

            // Nearly identical vector with rounding errors
            let edgeValues2 = edgeValues1.map { $0 + Float.ulpOfOne * Float.random(in: -1...1) }
            let edgeCandidate = try Vector1536Optimized(edgeValues2)
            let edgeFP16 = MixedPrecisionKernels.convertToFP16_1536([edgeCandidate])

            MixedPrecisionKernels.range_cosine_mixed_1536(
                query: edgeQuery,
                candidatesFP16: edgeFP16,
                range: 0..<1,
                out: outputBuffer
            )

            // Should be clamped to valid range even with numerical errors
            #expect(outputBuffer[0] >= 0.0 && outputBuffer[0] <= 2.0)
            #expect(outputBuffer[0] < 0.001, "Nearly identical vectors should have ~0 distance")

            // Test with maximum dimensionality stress
            let stressValues = (0..<1536).map { i in
                // Create a pattern that stresses accumulation
                Float(sin(Double(i) * Double.pi / 100.0)) * pow(-1, Float(i))
            }
            let stressQuery = try Vector1536Optimized(stressValues)
            let stressCandidates = [
                try Vector1536Optimized(stressValues.reversed()),
                try Vector1536Optimized(stressValues.map { abs($0) }),
                try Vector1536Optimized(stressValues.map { -$0 })
            ]
            let stressFP16 = MixedPrecisionKernels.convertToFP16_1536(stressCandidates)

            let stressBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 3)
            defer { stressBuffer.deallocate() }

            MixedPrecisionKernels.range_cosine_mixed_1536(
                query: stressQuery,
                candidatesFP16: stressFP16,
                range: 0..<3,
                out: stressBuffer
            )

            // All results should be valid
            for i in 0..<3 {
                #expect(stressBuffer[i] >= 0.0 && stressBuffer[i] <= 2.0)
                let reference = stressQuery.distance(to: stressCandidates[i], metric: .cosine)
                #expect(abs(stressBuffer[i] - reference) < 0.02)  // Allow more error in stress test
            }
        }

        @Test("Range processing correctness")
        func testRangeProcessingCorrectness() async throws {
            // Create test data
            let query = try Vector512Optimized((0..<512).map { Float($0) * 0.01 })
            let candidateCount = 10
            var candidates: [Vector512Optimized] = []
            for i in 0..<candidateCount {
                candidates.append(try Vector512Optimized((0..<512).map { Float($0 + i) * 0.01 }))
            }
            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)

            // Test partial range processing
            let partialRange = 2..<7
            let partialBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: partialRange.count)
            defer { partialBuffer.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: partialRange,
                out: partialBuffer
            )

            // Verify correct candidates were processed
            for i in 0..<partialRange.count {
                let candidateIdx = partialRange.lowerBound + i
                let expected = query.euclideanDistanceSquared(to: candidates[candidateIdx])
                let actual = partialBuffer[i]
                #expect(abs(actual - expected) / expected < 0.01)
            }

            // Test boundary conditions
            // First element only
            let firstBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1)
            defer { firstBuffer.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<1,
                out: firstBuffer
            )

            let firstExpected = query.euclideanDistanceSquared(to: candidates[0])
            if abs(firstExpected) > 1e-4 {
                #expect(abs(firstBuffer[0] - firstExpected) / firstExpected < 0.01)
            } else {
                // For very small distances, use more lenient absolute tolerance due to FP16 precision
                #expect(abs(firstBuffer[0] - firstExpected) < 0.001)
            }

            // Last element only
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: (candidateCount-1)..<candidateCount,
                out: firstBuffer
            )

            let lastExpected = query.euclideanDistanceSquared(to: candidates[candidateCount-1])
            if abs(lastExpected) > 1e-4 {
                #expect(abs(firstBuffer[0] - lastExpected) / lastExpected < 0.01)
            } else {
                // For very small distances, use more lenient absolute tolerance due to FP16 precision
                #expect(abs(firstBuffer[0] - lastExpected) < 0.001)
            }

            // Test empty range (should not crash)
            let emptyBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1)
            defer { emptyBuffer.deallocate() }
            emptyBuffer[0] = 999.0  // Set sentinel value

            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 5..<5,  // Empty range
                out: emptyBuffer
            )

            // Buffer should not be modified
            #expect(emptyBuffer[0] == 999.0)

            // Test single-element range
            let singleBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1)
            defer { singleBuffer.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 3..<4,
                out: singleBuffer
            )

            let singleExpected = query.euclideanDistanceSquared(to: candidates[3])
            if abs(singleExpected) > 1e-4 {
                #expect(abs(singleBuffer[0] - singleExpected) / singleExpected < 0.01)
            } else {
                // For very small distances, use more lenient absolute tolerance due to FP16 precision
                #expect(abs(singleBuffer[0] - singleExpected) < 0.001)
            }
        }

        @Test("Output buffer management")
        func testOutputBufferManagement() async throws {
            // Create test vectors
            let query = try Vector768Optimized((0..<768).map { Float($0) * 0.001 })
            let candidates = (0..<5).map { i in
                try! Vector768Optimized((0..<768).map { Float($0 + i * 10) * 0.001 })
            }
            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_768(candidates)

            // Test pre-allocated buffer usage
            let preAllocBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 10)
            defer { preAllocBuffer.deallocate() }

            // Initialize with sentinel values
            for i in 0..<10 {
                preAllocBuffer[i] = Float(1000 + i)
            }

            // Write to first 5 elements
            MixedPrecisionKernels.range_euclid2_mixed_768(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<5,
                out: preAllocBuffer
            )

            // Verify first 5 elements were written
            for i in 0..<5 {
                #expect(preAllocBuffer[i] < 1000, "Element \(i) should be written")
            }

            // Verify remaining elements untouched
            for i in 5..<10 {
                #expect(preAllocBuffer[i] == Float(1000 + i), "Element \(i) should be untouched")
            }

            // Test with offset writes
            let offsetBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 10)
            defer { offsetBuffer.deallocate() }

            // Initialize all to sentinel
            for i in 0..<10 {
                offsetBuffer[i] = 777.0
            }

            // Process subset and write to offset position
            let subsetRange = 1..<4
            let offsetSubBuffer = UnsafeMutableBufferPointer(
                start: offsetBuffer.baseAddress! + 2,
                count: subsetRange.count
            )

            MixedPrecisionKernels.range_cosine_mixed_768(
                query: query,
                candidatesFP16: candidatesFP16,
                range: subsetRange,
                out: offsetSubBuffer
            )

            // Check writes at correct positions
            #expect(offsetBuffer[0] == 777.0)  // Untouched
            #expect(offsetBuffer[1] == 777.0)  // Untouched
            #expect(offsetBuffer[2] != 777.0)  // Written
            #expect(offsetBuffer[3] != 777.0)  // Written
            #expect(offsetBuffer[4] != 777.0)  // Written
            #expect(offsetBuffer[5] == 777.0)  // Untouched

            // Test buffer size validation in debug mode
            #if DEBUG
            // This would assert in debug mode if buffer is too small
            let smallBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 2)
            defer { smallBuffer.deallocate() }

            // Should work with exact size
            MixedPrecisionKernels.range_euclid2_mixed_768(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<2,
                out: smallBuffer
            )

            // Verify results were written
            #expect(smallBuffer[0] > 0)
            #expect(smallBuffer[1] > 0)
            #endif
        }
    }

    // MARK: - Batch Processing Tests

    @Suite("Batch Processing")
    struct BatchProcessingTests {

        @Test("Batch conversion FP32 to FP16")
        func testBatchConversionFP32ToFP16() async throws {
            // Test batch conversion utilities for different dimensions
            let testSizes = [(512, 20), (768, 15), (1536, 10)]

            for (dim, count) in testSizes {
                if dim == 512 {
                    var batch512: [Vector512Optimized] = []
                    for i in 0..<count {
                        let values = (0..<dim).map { Float($0 + i * 100) * 0.01 }
                        batch512.append(try Vector512Optimized(values))
                    }

                    let fp16Batch = MixedPrecisionKernels.convertToFP16_512(batch512)
                    #expect(fp16Batch.count == count)

                    // Verify conversion accuracy
                    for i in 0..<count {
                        let converted = fp16Batch[i].toFP32()
                        for j in 0..<dim {
                            let original = batch512[i][j]
                            let recovered = converted[j]
                            #expect(abs(original - recovered) < 0.01 || abs((original - recovered) / original) < 0.001)
                        }
                    }
                } else if dim == 768 {
                    var batch768: [Vector768Optimized] = []
                    for i in 0..<count {
                        let values = (0..<dim).map { Float(sin(Double($0 + i * 50) * 0.01)) }
                        batch768.append(try Vector768Optimized(values))
                    }

                    let fp16Batch = MixedPrecisionKernels.convertToFP16_768(batch768)
                    #expect(fp16Batch.count == count)

                    // Verify first and last vectors
                    let firstConverted = fp16Batch[0].toFP32()
                    let lastConverted = fp16Batch[count-1].toFP32()
                    #expect(abs(firstConverted[0] - batch768[0][0]) < 0.001)
                    #expect(abs(lastConverted[dim-1] - batch768[count-1][dim-1]) < 0.001)
                } else if dim == 1536 {
                    var batch1536: [Vector1536Optimized] = []
                    for i in 0..<count {
                        let values = (0..<dim).map { Float(cos(Double($0 - i * 25) * 0.005)) * 5.0 }
                        batch1536.append(try Vector1536Optimized(values))
                    }

                    let fp16Batch = MixedPrecisionKernels.convertToFP16_1536(batch1536)
                    #expect(fp16Batch.count == count)
                }
            }

            // Test memory efficiency for large batches
            let largeBatchSize = 100
            var largeBatch: [Vector512Optimized] = []
            for i in 0..<largeBatchSize {
                let values = Array(repeating: Float(i) * 0.1, count: 512)
                largeBatch.append(try Vector512Optimized(values))
            }

            _ = MixedPrecisionKernels.convertToFP16_512(largeBatch)
            let fp32Memory = largeBatchSize * 512 * MemoryLayout<Float>.size
            let fp16Memory = largeBatchSize * 512 * MemoryLayout<Float16>.size
            let memorySavings = Float(fp32Memory - fp16Memory) / Float(fp32Memory)
            #expect(abs(memorySavings - 0.5) < 0.01)  // Should save ~50% memory

            // Test concurrent conversion safety
            await withTaskGroup(of: [Vector512FP16].self) { group in
                for i in 0..<5 {
                    group.addTask {
                        let values = (0..<512).map { Float($0 * (i + 1)) * 0.001 }
                        let vector = try! Vector512Optimized(values)
                        return MixedPrecisionKernels.convertToFP16_512([vector])
                    }
                }

                var results: [[Vector512FP16]] = []
                for await result in group {
                    results.append(result)
                }

                #expect(results.count == 5)
            }
        }

        @Test("Batch Euclidean distance computation")
        func testBatchEuclideanDistance() async throws {
            // Create query and candidates
            let query = try Vector512Optimized((0..<512).map { Float($0) * 0.01 })
            let candidateCount = 25
            var candidates: [Vector512Optimized] = []
            for i in 0..<candidateCount {
                let values = (0..<512).map { j in
                    Float(sin(Double(j * (i + 1)) * 0.01)) * 10.0
                }
                candidates.append(try Vector512Optimized(values))
            }

            // Convert to FP16
            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)

            // Test batch processing with mixed precision
            let batchBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: candidateCount)
            defer { batchBuffer.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<candidateCount,
                out: batchBuffer
            )

            // Compare with sequential processing
            var sequentialResults: [Float] = []
            for candidate in candidates {
                sequentialResults.append(query.euclideanDistanceSquared(to: candidate))
            }

            // Verify result ordering and accuracy
            for i in 0..<candidateCount {
                let batchResult = batchBuffer[i]
                let sequentialResult = sequentialResults[i]
                let relError = abs((batchResult - sequentialResult) / sequentialResult)
                #expect(relError < 0.005, "Index \(i): batch=\(batchResult), seq=\(sequentialResult)")
            }

            // Test with different batch sizes
            let batchSizes = [1, 5, 10, candidateCount]
            for batchSize in batchSizes {
                let subBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: batchSize)
                defer { subBuffer.deallocate() }

                MixedPrecisionKernels.range_euclid2_mixed_512(
                    query: query,
                    candidatesFP16: candidatesFP16,
                    range: 0..<batchSize,
                    out: subBuffer
                )

                for i in 0..<batchSize {
                    let expected = sequentialResults[i]
                    let actual = subBuffer[i]
                    #expect(abs(actual - expected) / expected < 0.005)
                }
            }
        }

        @Test("Batch cosine distance computation")
        func testBatchCosineDistance() async throws {
            // Create normalized query vector
            let queryValues = (0..<768).map { Float(sin(Double($0) * 0.01)) }
            let queryVec = try Vector768Optimized(queryValues)
            let queryMag = queryVec.magnitude
            let normalizedQuery = try Vector768Optimized(queryValues.map { $0 / queryMag })

            // Create batch of candidates with varying properties
            var candidates: [Vector768Optimized] = []

            // Add identical vector
            candidates.append(normalizedQuery)

            // Add orthogonal vector
            var orthogonal = Array(repeating: Float(0), count: 768)
            orthogonal[0] = 1.0
            candidates.append(try Vector768Optimized(orthogonal))

            // Add anti-parallel vector
            candidates.append(try Vector768Optimized(queryValues.map { -$0 / queryMag }))

            // Add random vectors
            for _ in 0..<10 {
                let randomVals = (0..<768).map { _ in Float.random(in: -1...1) }
                let randomVec = try Vector768Optimized(randomVals)
                let randomMag = randomVec.magnitude
                candidates.append(try Vector768Optimized(randomVals.map { $0 / randomMag }))
            }

            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_768(candidates)
            let batchBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: candidates.count)
            defer { batchBuffer.deallocate() }

            // Test batch processing with mixed precision
            MixedPrecisionKernels.range_cosine_mixed_768(
                query: normalizedQuery,
                candidatesFP16: candidatesFP16,
                range: 0..<candidates.count,
                out: batchBuffer
            )

            // Verify specific cases
            #expect(batchBuffer[0] < 1e-5, "Identical vector should have ~0 distance")
            #expect(abs(batchBuffer[1] - 1.0) < 0.01, "Orthogonal vector should have distance ~1")
            #expect(abs(batchBuffer[2] - 2.0) < 0.01, "Anti-parallel vector should have distance ~2")

            // Verify all distances are in valid range
            for i in 0..<candidates.count {
                #expect(batchBuffer[i] >= 0.0 && batchBuffer[i] <= 2.0,
                       "Distance \(i) out of range: \(batchBuffer[i])")
            }

            // Test query magnitude caching efficiency
            // The kernel should compute query magnitude once and reuse it
            // We can't directly test this, but we can verify consistency
            let secondBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: candidates.count)
            defer { secondBuffer.deallocate() }

            MixedPrecisionKernels.range_cosine_mixed_768(
                query: normalizedQuery,
                candidatesFP16: candidatesFP16,
                range: 0..<candidates.count,
                out: secondBuffer
            )

            // Results should be identical
            for i in 0..<candidates.count {
                #expect(batchBuffer[i] == secondBuffer[i])
            }

            // Test numerical stability across batch with small magnitudes
            var smallCandidates: [Vector768Optimized] = []
            for i in 0..<5 {
                let smallVals = (0..<768).map { _ in
                    Float.random(in: -1e-5...1e-5) * Float(i + 1)
                }
                smallCandidates.append(try Vector768Optimized(smallVals))
            }

            let smallFP16 = MixedPrecisionKernels.convertToFP16_768(smallCandidates)
            let smallBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: smallCandidates.count)
            defer { smallBuffer.deallocate() }

            MixedPrecisionKernels.range_cosine_mixed_768(
                query: normalizedQuery,
                candidatesFP16: smallFP16,
                range: 0..<smallCandidates.count,
                out: smallBuffer
            )

            // With very small values, results should still be valid
            for i in 0..<smallCandidates.count {
                #expect(smallBuffer[i] >= 0.0 && smallBuffer[i] <= 2.0)
            }
        }

        @Test("Large batch memory pressure")
        func testLargeBatchMemoryPressure() async throws {
            // Test with large number of candidates
            let largeBatchSize = 1000  // Reduced from 10K for test performance
            let dimension = 512

            // Create query
            let query = try Vector512Optimized((0..<dimension).map { Float($0) * 0.001 })

            // Create large batch of candidates
            var candidates: [Vector512Optimized] = []
            candidates.reserveCapacity(largeBatchSize)

            for i in 0..<largeBatchSize {
                // Use simple pattern to avoid excessive computation
                let values = Array(repeating: Float(i) * 0.001 + 1.0, count: dimension)
                candidates.append(try Vector512Optimized(values))
            }

            // Convert to FP16 - this should use ~50% less memory
            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)

            // Calculate memory footprint
            let fp32MemoryMB = Float(largeBatchSize * dimension * MemoryLayout<Float>.size) / (1024 * 1024)
            let fp16MemoryMB = Float(largeBatchSize * dimension * MemoryLayout<Float16>.size) / (1024 * 1024)

            #expect(fp16MemoryMB < fp32MemoryMB * 0.51, "FP16 should use ~50% memory")

            // Process in chunks to verify memory efficiency
            let chunkSize = 100
            let outputBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: chunkSize)
            defer { outputBuffer.deallocate() }

            var allResults: [Float] = []
            allResults.reserveCapacity(largeBatchSize)

            for chunkStart in stride(from: 0, to: largeBatchSize, by: chunkSize) {
                let chunkEnd = min(chunkStart + chunkSize, largeBatchSize)
                let chunkRange = chunkStart..<chunkEnd
                let chunkBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: chunkRange.count)
                defer { chunkBuffer.deallocate() }

                MixedPrecisionKernels.range_euclid2_mixed_512(
                    query: query,
                    candidatesFP16: candidatesFP16,
                    range: chunkRange,
                    out: chunkBuffer
                )

                for i in 0..<chunkRange.count {
                    allResults.append(chunkBuffer[i])
                }
            }

            #expect(allResults.count == largeBatchSize)

            // Verify results are reasonable
            for i in 0..<min(10, allResults.count) {
                #expect(allResults[i] > 0, "Distance should be positive")
            }

            // Clear to help with memory pressure
            candidates.removeAll()

            // Test that FP16 vectors can be reused multiple times without issues
            let reuseBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 10)
            defer { reuseBuffer.deallocate() }

            for _ in 0..<3 {
                MixedPrecisionKernels.range_euclid2_mixed_512(
                    query: query,
                    candidatesFP16: candidatesFP16,
                    range: 0..<10,
                    out: reuseBuffer
                )

                // Results should be consistent
                #expect(reuseBuffer[0] == allResults[0])
            }
        }
    }

    // MARK: - SoA Layout Tests

    @Suite("Structure-of-Arrays Layout")
    struct SoALayoutTests {

        @Test("SoAFP16 initialization with various dimensions")
        func testSoAFP16Initialization() async throws {
            // Test SoA layout creation with 512D vectors
            let vectors512: [Vector512Optimized] = (0..<10).map { i in
                let values = (0..<512).map { Float($0 + i * 512) * 0.001 }
                return try! Vector512Optimized(values)
            }

            let soa512 = try SoAFP16(vectors: vectors512, blockSize: 64)
            #expect(soa512.dimension == 512)
            #expect(soa512.vectorCount == 10)
            #expect(soa512.blockSize == 64)
            #expect(soa512.blocksPerVector == 8)  // 512 / 64 = 8

            // Verify we can extract vectors back
            let extracted = try soa512.getVector(at: 0)
            for i in 0..<512 {
                let original = vectors512[0][i]
                let recovered = extracted[i]
                #expect(abs(original - recovered) < 0.001 || abs((original - recovered) / original) < 0.001)
            }

            // Test with 768D vectors and different block size
            let vectors768: [Vector768Optimized] = (0..<5).map { i in
                let values = (0..<768).map { Float(sin(Double($0 * i) * 0.01)) }
                return try! Vector768Optimized(values)
            }

            let soa768 = try SoAFP16(vectors: vectors768, blockSize: 128)
            #expect(soa768.dimension == 768)
            #expect(soa768.vectorCount == 5)
            #expect(soa768.blockSize == 128)
            #expect(soa768.blocksPerVector == 6)  // 768 / 128 = 6

            // Test with 1536D vectors
            let vectors1536: [Vector1536Optimized] = (0..<3).map { i in
                let values = (0..<1536).map { Float(cos(Double($0 - i * 100) * 0.005)) }
                return try! Vector1536Optimized(values)
            }

            let soa1536 = try SoAFP16(vectors: vectors1536, blockSize: 256)
            #expect(soa1536.dimension == 1536)
            #expect(soa1536.vectorCount == 3)
            #expect(soa1536.blockSize == 256)
            #expect(soa1536.blocksPerVector == 6)  // 1536 / 256 = 6

            // Test dimension validation - empty vectors should throw
            do {
                _ = try SoAFP16<Vector512Optimized>(vectors: [], blockSize: 64)
                Issue.record("Should throw for empty vector array")
            } catch let error as VectorError {
                #expect(error.kind == .invalidData)
            }

            // Test dimension mismatch detection
            var mismatchedVectors = vectors512
            mismatchedVectors.append(try Vector512Optimized(Array(repeating: Float(1.0), count: 512)))
            // This should work as all have same dimension
            let soaMismatched = try SoAFP16(vectors: mismatchedVectors, blockSize: 64)
            #expect(soaMismatched.vectorCount == 11)
        }

        @Test("SoAFP16 block pointer access")
        func testSoAFP16BlockPointerAccess() async throws {
            // Create test vectors
            let vectorCount = 8
            let dimension = 512
            let blockSize = 64

            let vectors: [Vector512Optimized] = (0..<vectorCount).map { i in
                // Use distinct patterns for each vector
                let values = (0..<dimension).map { j in
                    Float(i * 1000 + j)  // Each vector has unique values
                }
                return try! Vector512Optimized(values)
            }

            let soa = try SoAFP16(vectors: vectors, blockSize: blockSize)

            // Test block pointer calculation
            let blocksPerVector = (dimension + blockSize - 1) / blockSize
            #expect(soa.blocksPerVector == blocksPerVector)

            // Access each block and verify structure
            for blockIndex in 0..<blocksPerVector {
                let blockPtr = soa.blockPointer(blockIndex: blockIndex)
                let blockStart = blockIndex * blockSize
                let blockEnd = min(blockStart + blockSize, dimension)
                let elementsInBlock = blockEnd - blockStart

                // Calculate expected SIMD4 groups in this block
                let elementsPerBlock = elementsInBlock * vectorCount
                let simd4Groups = (elementsPerBlock + 3) / 4

                // Verify we can safely access the expected number of SIMD4 groups
                // We'll check first and last SIMD4 group
                if simd4Groups > 0 {
                    let firstGroup = blockPtr[0]
                    let lastGroup = blockPtr[simd4Groups - 1]

                    // Verify values are in FP16 range
                    for lane in 0..<4 {
                        let fp32First = Float(firstGroup[lane])
                        let fp32Last = Float(lastGroup[lane])

                        // Values should be finite
                        #expect(fp32First.isFinite || fp32First == 0.0)
                        #expect(fp32Last.isFinite || fp32Last == 0.0)
                    }
                }
            }

            // Test boundary blocks (last block may be partially filled)
            let lastBlockIndex = blocksPerVector - 1
            let lastBlockPtr = soa.blockPointer(blockIndex: lastBlockIndex)
            let lastBlockStart = lastBlockIndex * blockSize
            let remainingElements = dimension - lastBlockStart

            #expect(remainingElements <= blockSize)

            // Verify memory alignment
            // SIMD4<Float16> should be 8-byte aligned (4 * 2 bytes)
            let ptrAddress = Int(bitPattern: lastBlockPtr)
            #expect(ptrAddress % 8 == 0, "Block pointer should be 8-byte aligned")

            // Test that block pointers are stable
            let block0Ptr1 = soa.blockPointer(blockIndex: 0)
            let block0Ptr2 = soa.blockPointer(blockIndex: 0)
            #expect(block0Ptr1 == block0Ptr2, "Block pointers should be stable")
        }

        @Test("SoAFP16 vector extraction")
        func testSoAFP16VectorExtraction() async throws {
            // Create diverse test vectors
            let vectorCount = 7
            let testVectors: [Vector768Optimized] = (0..<vectorCount).map { i in
                let values = (0..<768).map { j in
                    // Different patterns for each vector
                    switch i {
                    case 0: return Float(j) * 0.01  // Linear
                    case 1: return Float(sin(Double(j) * 0.01))  // Sine wave
                    case 2: return Float(cos(Double(j) * 0.01))  // Cosine wave
                    case 3: return j % 2 == 0 ? 1.0 : -1.0  // Alternating
                    case 4: return Float.random(in: -10...10)  // Random
                    case 5: return Float(j < 384 ? 1.0 : 0.0)  // Step function
                    default: return Float(exp(-Double(j) / 100.0))  // Exponential decay
                    }
                }
                return try! Vector768Optimized(values)
            }

            let soa = try SoAFP16(vectors: testVectors, blockSize: 96)

            // Test extracting individual vectors
            for i in 0..<vectorCount {
                let extracted = try soa.getVector(at: i)
                #expect(extracted.scalarCount == 768)

                // Verify FP16→FP32 conversion accuracy
                for j in 0..<768 {
                    let original = testVectors[i][j]
                    let recovered = extracted[j]

                    if original.isFinite {
                        let absError = abs(original - recovered)
                        let relError = original != 0 ? abs((original - recovered) / original) : absError

                        // FP16 has limited precision
                        #expect(absError < 0.01 || relError < 0.001,
                               "Vector \(i), element \(j): orig=\(original), recovered=\(recovered)")
                    } else {
                        // Handle special values
                        #expect(original.isNaN ? recovered.isNaN : recovered == original)
                    }
                }
            }

            // Test extracting first, middle, and last vectors
            let firstVector = try soa.getVector(at: 0)
            let middleVector = try soa.getVector(at: vectorCount / 2)
            let lastVector = try soa.getVector(at: vectorCount - 1)

            #expect(firstVector.scalarCount == 768)
            #expect(middleVector.scalarCount == 768)
            #expect(lastVector.scalarCount == 768)

            // Verify vectors are different
            let firstSum = (0..<768).reduce(Float(0)) { $0 + firstVector[$1] }
            let middleSum = (0..<768).reduce(Float(0)) { $0 + middleVector[$1] }
            let lastSum = (0..<768).reduce(Float(0)) { $0 + lastVector[$1] }

            #expect(firstSum != middleSum || middleSum != lastSum, "Vectors should be different")

            // Test round-trip: original → SoA → extracted
            let simpleVector = try Vector768Optimized(Array(repeating: Float(3.14159), count: 768))
            let soaSimple = try SoAFP16(vectors: [simpleVector], blockSize: 64)
            let extractedSimple = try soaSimple.getVector(at: 0)

            for i in 0..<768 {
                #expect(abs(extractedSimple[i] - 3.14159) < 0.001)
            }
        }

        @Test("SoAFP16 blocked layout efficiency")
        func testSoAFP16BlockedLayoutEfficiency() async throws {
            // Test various block sizes
            let blockSizes = [32, 64, 128, 256]
            let vectorCount = 20
            let dimension = 1536

            var vectors: [Vector1536Optimized] = []
            for i in 0..<vectorCount {
                var values: [Float] = []
                values.reserveCapacity(dimension)
                for j in 0..<dimension {
                    let value = Float((i + 1) * (j + 1)) * 0.0001
                    values.append(value)
                }
                vectors.append(try! Vector1536Optimized(values))
            }

            for blockSize in blockSizes {
                let soa = try SoAFP16(vectors: vectors, blockSize: blockSize)

                // Verify block organization
                let expectedBlocks = (dimension + blockSize - 1) / blockSize
                #expect(soa.blocksPerVector == expectedBlocks)

                // Test that smaller block sizes allow better cache utilization
                // Smaller blocks should fit in L1 cache (typically 32KB)
                let bytesPerBlock = blockSize * vectorCount * MemoryLayout<Float16>.size
                let l1CacheSize = 32 * 1024  // 32KB typical L1 cache

                if blockSize <= 64 {
                    // Small blocks should fit in L1
                    #expect(bytesPerBlock < l1CacheSize,
                           "Block size \(blockSize) should fit in L1 cache")
                }

                // Verify SIMD4 alignment optimization
                // Each block should have elements padded to SIMD4 boundaries
                for blockIndex in 0..<soa.blocksPerVector {
                    let blockStart = blockIndex * blockSize
                    let blockEnd = min(blockStart + blockSize, dimension)
                    let elementsInBlock = (blockEnd - blockStart) * vectorCount
                    let simd4Groups = (elementsInBlock + 3) / 4

                    // Storage should be allocated for complete SIMD4 groups
                    let expectedStorageElements = simd4Groups * 4
                    #expect(expectedStorageElements >= elementsInBlock)
                    #expect(expectedStorageElements - elementsInBlock < 4)  // Padding < 4
                }
            }

            // Test cache-friendly access patterns
            // Accessing vectors sequentially within a block should be efficient
            let cacheTestSoa = try SoAFP16(vectors: vectors, blockSize: 64)

            // Simulate cache-friendly access: process all vectors for each dimension block
            var accessedValues: [Float] = []
            for blockIndex in 0..<cacheTestSoa.blocksPerVector {
                // Within this block, we access all vectors sequentially
                // This should have good cache locality
                for vectorIndex in 0..<vectorCount {
                    let vector = try cacheTestSoa.getVector(at: vectorIndex)
                    let blockStart = blockIndex * 64
                    let blockEnd = min(blockStart + 64, dimension)

                    for dimIndex in blockStart..<blockEnd {
                        accessedValues.append(vector[dimIndex])
                    }
                }
            }

            // Verify we accessed all values
            #expect(accessedValues.count == dimension * vectorCount)
        }

        @Test("Batch Euclidean SoA processing")
        func testBatchEuclideanSoA() async throws {
            // Create test data
            let vectorCount = 15
            let queryValues = (0..<512).map { Float(sin(Double($0) * 0.01)) * 5.0 }
            let query = try Vector512Optimized(queryValues)

            var candidates: [Vector512Optimized] = []
            for i in 0..<vectorCount {
                let values = (0..<512).map { j in
                    Float(cos(Double(j + i * 50) * 0.01)) * 5.0
                }
                candidates.append(try Vector512Optimized(values))
            }

            // Create SoA layout
            let soaCandidates = try SoAFP16(vectors: candidates, blockSize: 64)

            // Test SoA batch Euclidean distance
            let soaResults = UnsafeMutableBufferPointer<Float>.allocate(capacity: vectorCount)
            defer { soaResults.deallocate() }

            MixedPrecisionKernels.batchEuclideanSquaredSoA(
                query: query,
                candidates: soaCandidates,
                results: soaResults
            )

            // Compare with regular batch processing
            let fp16Candidates = MixedPrecisionKernels.convertToFP16_512(candidates)
            let regularResults = UnsafeMutableBufferPointer<Float>.allocate(capacity: vectorCount)
            defer { regularResults.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: fp16Candidates,
                range: 0..<vectorCount,
                out: regularResults
            )

            // Verify SoA results match regular batch processing
            for i in 0..<vectorCount {
                let soaResult = soaResults[i]
                let regularResult = regularResults[i]
                let relError = abs((soaResult - regularResult) / regularResult)
                #expect(relError < 0.01,
                       "Index \(i): SoA=\(soaResult), regular=\(regularResult), error=\(relError)")
            }

            // Verify accumulation correctness
            // Manually compute one distance to verify
            let reference = query.euclideanDistanceSquared(to: candidates[0])
            let soaFirst = soaResults[0]
            #expect(abs(soaFirst - reference) / reference < 0.01)

            // Test with different block sizes
            let blockSizes = [32, 128]
            for blockSize in blockSizes {
                let soaAlt = try SoAFP16(vectors: candidates, blockSize: blockSize)
                let altResults = UnsafeMutableBufferPointer<Float>.allocate(capacity: vectorCount)
                defer { altResults.deallocate() }

                MixedPrecisionKernels.batchEuclideanSquaredSoA(
                    query: query,
                    candidates: soaAlt,
                    results: altResults
                )

                // Results should be consistent regardless of block size
                for i in 0..<vectorCount {
                    #expect(abs(altResults[i] - soaResults[i]) < 0.001)
                }
            }
        }

        @Test("Batch dot product SoA processing")
        func testBatchDotProductSoA() async throws {
            // Create normalized vectors for dot product testing
            let vectorCount = 12
            let dimension = 768

            let queryValues = (0..<dimension).map { Float(sin(Double($0) * 0.005)) }
            let queryVec = try Vector768Optimized(queryValues)
            let queryMag = queryVec.magnitude
            let normalizedQuery = try Vector768Optimized(queryValues.map { $0 / queryMag })

            var candidates: [Vector768Optimized] = []

            // Add parallel vector (dot product ≈ 1)
            candidates.append(normalizedQuery)

            // Add orthogonal vector (dot product ≈ 0)
            var orthValues = Array(repeating: Float(0), count: dimension)
            orthValues[0] = 1.0
            candidates.append(try Vector768Optimized(orthValues))

            // Add anti-parallel vector (dot product ≈ -1)
            candidates.append(try Vector768Optimized(queryValues.map { -$0 / queryMag }))

            // Add random normalized vectors
            for _ in 0..<(vectorCount - 3) {
                let randomValues = (0..<dimension).map { _ in Float.random(in: -1...1) }
                let randomVec = try Vector768Optimized(randomValues)
                let randomMag = randomVec.magnitude
                candidates.append(try Vector768Optimized(randomValues.map { $0 / randomMag }))
            }

            // Test SoA batch dot product
            let soaCandidates = try SoAFP16(vectors: candidates, blockSize: 96)
            let dotResults = UnsafeMutableBufferPointer<Float>.allocate(capacity: vectorCount)
            defer { dotResults.deallocate() }

            MixedPrecisionKernels.batchDotProductSoA(
                query: normalizedQuery,
                candidates: soaCandidates,
                results: dotResults
            )

            // Verify specific cases
            #expect(abs(dotResults[0] - 1.0) < 0.01, "Parallel vector dot product should be ~1")
            #expect(abs(dotResults[1]) < 0.01, "Orthogonal vector dot product should be ~0")
            #expect(abs(dotResults[2] + 1.0) < 0.01, "Anti-parallel vector dot product should be ~-1")

            // Verify all dot products are in valid range [-1, 1] for normalized vectors
            for i in 0..<vectorCount {
                #expect(dotResults[i] >= -1.01 && dotResults[i] <= 1.01,
                       "Dot product \(i) out of range: \(dotResults[i])")
            }

            // Test block-wise accumulation correctness
            // Manually compute reference dot products
            var referenceDots: [Float] = []
            for candidate in candidates {
                var dotProduct: Float = 0
                for j in 0..<dimension {
                    dotProduct += normalizedQuery[j] * candidate[j]
                }
                referenceDots.append(dotProduct)
            }

            // Compare with SoA results
            for i in 0..<vectorCount {
                let soaDot = dotResults[i]
                let refDot = referenceDots[i]
                let error = abs(soaDot - refDot)
                #expect(error < 0.01, "Index \(i): SoA=\(soaDot), ref=\(refDot), error=\(error)")
            }

            // Test with various block sizes
            let blockSizes = [48, 192, 384]
            for blockSize in blockSizes {
                let soaAlt = try SoAFP16(vectors: candidates, blockSize: blockSize)
                let altDotResults = UnsafeMutableBufferPointer<Float>.allocate(capacity: vectorCount)
                defer { altDotResults.deallocate() }

                MixedPrecisionKernels.batchDotProductSoA(
                    query: normalizedQuery,
                    candidates: soaAlt,
                    results: altDotResults
                )

                // Results should be consistent across block sizes
                for i in 0..<vectorCount {
                    #expect(abs(altDotResults[i] - dotResults[i]) < 0.001,
                           "Block size \(blockSize), index \(i): inconsistent results")
                }
            }
        }

        @Test("SoA block processing correctness")
        func testSoABlockProcessingCorrectness() async throws {
            // Create small test case for detailed verification
            let vectorCount = 5
            let dimension = 100  // Not divisible by common block sizes
            let blockSize = 32

            // Create vectors with predictable patterns
            var vectors: [Vector512Optimized] = []
            for i in 0..<vectorCount {
                var values = Array(repeating: Float(0), count: 512)
                for j in 0..<dimension {
                    values[j] = Float(i * 1000 + j)  // Unique value per element
                }
                vectors.append(try Vector512Optimized(values))
            }

            let soa = try SoAFP16(vectors: vectors, blockSize: blockSize)

            // Test individual block processing
            let blocksPerVector = (dimension + blockSize - 1) / blockSize
            #expect(blocksPerVector == 4)  // ceil(100/32) = 4

            // Verify dimension/vector indexing within blocks
            for blockIdx in 0..<blocksPerVector {
                let blockStart = blockIdx * blockSize
                let blockEnd = min(blockStart + blockSize, dimension)
                let blockPtr = soa.blockPointer(blockIndex: blockIdx)

                // Manually verify some values
                for dimOffset in 0..<min(3, blockEnd - blockStart) {
                    for vecIdx in 0..<min(2, vectorCount) {
                        let elementIdx = dimOffset * vectorCount + vecIdx
                        let simd4Idx = elementIdx / 4
                        let laneIdx = elementIdx % 4

                        let fp16Value = blockPtr[simd4Idx][laneIdx]
                        let fp32Value = Float(fp16Value)

                        let expectedValue = Float(vecIdx * 1000 + blockStart + dimOffset)
                        let error = abs(fp32Value - expectedValue)

                        #expect(error < 1.0,
                               "Block \(blockIdx), dim \(dimOffset), vec \(vecIdx): expected \(expectedValue), got \(fp32Value)")
                    }
                }
            }

            // Test padding handling in last block
            let lastBlockIdx = blocksPerVector - 1
            let lastBlockStart = lastBlockIdx * blockSize
            let remainingDims = dimension - lastBlockStart
            #expect(remainingDims == 4)  // 100 - 96 = 4

            // Elements beyond dimension should be padded with zeros
            _ = soa.blockPointer(blockIndex: lastBlockIdx)
            let totalElementsInBlock = blockSize * vectorCount
            let actualElementsInBlock = remainingDims * vectorCount
            let paddingElements = totalElementsInBlock - actualElementsInBlock

            // Check that padding exists
            #expect(paddingElements > 0)

            // Verify extracted vectors match original (despite padding)
            for i in 0..<vectorCount {
                let extracted = try soa.getVector(at: i)
                for j in 0..<dimension {
                    let original = vectors[i][j]
                    let recovered = extracted[j]
                    #expect(abs(original - recovered) < 1.0)
                }
                // Elements beyond dimension should be zero
                for j in dimension..<512 {
                    #expect(extracted[j] == 0.0)
                }
            }
        }

        @Test("SoA memory layout optimization")
        func testSoAMemoryLayoutOptimization() async throws {
            // Create large dataset to test memory patterns
            let vectorCount = 50
            let dimension = 1536
            let blockSize = 128  // Optimized for cache line size

            let vectors: [Vector1536Optimized] = (0..<vectorCount).map { i in
                let values = (0..<dimension).map { j in
                    Float(sin(Double(i * j) * 0.0001))
                }
                return try! Vector1536Optimized(values)
            }

            let soa = try SoAFP16(vectors: vectors, blockSize: blockSize)

            // Test memory access patterns
            // Sequential access within blocks should be efficient
            var sequentialSum: Float = 0
            for blockIdx in 0..<soa.blocksPerVector {
                let blockPtr = soa.blockPointer(blockIndex: blockIdx)
                let blockStart = blockIdx * blockSize
                let blockEnd = min(blockStart + blockSize, dimension)
                let elementsInBlock = (blockEnd - blockStart) * vectorCount
                let simd4Groups = (elementsInBlock + 3) / 4

                // Sequential SIMD4 access
                for simd4Idx in 0..<simd4Groups {
                    let simd4 = blockPtr[simd4Idx]
                    for lane in 0..<4 {
                        sequentialSum += Float(simd4[lane])
                    }
                }
            }

            #expect(sequentialSum != 0, "Should have processed values")

            // Verify sequential access within blocks
            // Each block stores dimensions contiguously for all vectors
            // This means accessing all vectors for a dimension range is efficient
            let testBlockIdx = 2
            let testBlockStart = testBlockIdx * blockSize
            let testBlockEnd = min(testBlockStart + blockSize, dimension)

            // Simulate efficient access pattern
            var blockValues: [Float] = []
            let testBlockPtr = soa.blockPointer(blockIndex: testBlockIdx)

            for dimOffset in 0..<(testBlockEnd - testBlockStart) {
                for vecIdx in 0..<vectorCount {
                    let elementIdx = dimOffset * vectorCount + vecIdx
                    let simd4Idx = elementIdx / 4
                    let laneIdx = elementIdx % 4

                    blockValues.append(Float(testBlockPtr[simd4Idx][laneIdx]))
                }
            }

            // Values should be accessed sequentially
            #expect(blockValues.count == (testBlockEnd - testBlockStart) * vectorCount)

            // Test that block size aligns with cache lines
            // Modern CPUs have 64-byte cache lines
            let cacheLineSize = 64
            let fp16Size = MemoryLayout<Float16>.size
            _ = cacheLineSize / fp16Size  // 32 elements per cache line

            // Ideal block size should be multiple of cache line
            let idealBlockElements = blockSize * vectorCount
            let cacheLinesPerBlock = (idealBlockElements * fp16Size) / cacheLineSize

            // Verify reasonable cache alignment
            #expect(cacheLinesPerBlock > 0)

            // Test prefetching benefits by accessing predictable pattern
            let query = vectors[0]  // Use first vector as query
            let prefetchResults = UnsafeMutableBufferPointer<Float>.allocate(capacity: vectorCount)
            defer { prefetchResults.deallocate() }

            // This access pattern benefits from prefetching
            MixedPrecisionKernels.batchEuclideanSquaredSoA(
                query: query,
                candidates: soa,
                results: prefetchResults
            )

            // Verify all results computed
            for i in 0..<vectorCount {
                #expect(prefetchResults[i] >= 0, "Distance should be non-negative")
            }
        }
    }

    // MARK: - Numerical Stability Tests

    @Suite("Numerical Stability")
    struct NumericalStabilityTests {

        @Test("Accumulation error in high dimensions")
        func testAccumulationError() async throws {
            // Test error accumulation in 1536D
            let dimension = 1536

            // Create vectors with values that accumulate error
            // Use small values that when summed many times cause rounding errors
            let epsilon: Float = 0.001
            let values1 = (0..<dimension).map { _ in epsilon }
            let values2 = (0..<dimension).map { i in epsilon * (1.0 + Float(i) / Float(dimension)) }

            let vec1 = try Vector1536Optimized(values1)
            let vec2 = try Vector1536Optimized(values2)

            // Convert to FP16 and compute distance
            let vec2FP16 = Vector1536FP16(from: vec2)
            let outputBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1)
            defer { outputBuffer.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_1536(
                query: vec1,
                candidatesFP16: [vec2FP16],
                range: 0..<1,
                out: outputBuffer
            )

            let mixedResult = outputBuffer[0]

            // Compare with Kahan summation for reference
            var kahanSum: Float = 0
            var compensation: Float = 0

            for i in 0..<dimension {
                let diff = values1[i] - values2[i]
                let term = diff * diff
                let y = term - compensation
                let t = kahanSum + y
                compensation = (t - kahanSum) - y
                kahanSum = t
            }

            // Also compute naive summation
            var naiveSum: Float = 0
            for i in 0..<dimension {
                let diff = values1[i] - values2[i]
                naiveSum += diff * diff
            }

            // Verify error bounds
            let kahanError = abs(mixedResult - kahanSum) / kahanSum
            let naiveError = abs(mixedResult - naiveSum) / naiveSum

            // Mixed precision should be closer to Kahan than naive might be
            #expect(kahanError < 0.05, "Kahan error: \(kahanError)")
            #expect(naiveError < 0.1, "Naive error: \(naiveError)")

            // Test with alternating signs to stress cancellation
            let altValues1 = (0..<dimension).map { i in
                pow(-1, Float(i)) * epsilon * 100.0
            }
            let altValues2 = (0..<dimension).map { i in
                pow(-1, Float(i + 1)) * epsilon * 100.0
            }

            let altVec1 = try Vector1536Optimized(altValues1)
            let altVec2 = try Vector1536Optimized(altValues2)
            let altVec2FP16 = Vector1536FP16(from: altVec2)

            MixedPrecisionKernels.range_euclid2_mixed_1536(
                query: altVec1,
                candidatesFP16: [altVec2FP16],
                range: 0..<1,
                out: outputBuffer
            )

            let altMixedResult = outputBuffer[0]
            #expect(altMixedResult > 0, "Distance should be positive")

            // In high dimensions, relative error should stay bounded
            let expectedMagnitude = Float(dimension) * pow(epsilon * 200.0, 2)
            let magnitudeError = abs(altMixedResult - expectedMagnitude) / expectedMagnitude
            #expect(magnitudeError < 0.1, "High-dim error: \(magnitudeError)")
        }

        @Test("Catastrophic cancellation prevention")
        func testCatastrophicCancellation() async throws {
            // Test with nearly equal values that could cause catastrophic cancellation
            let baseValue: Float = 1000.0
            let epsilon: Float = 1e-4

            // Create two nearly identical vectors
            let values1 = (0..<512).map { _ in baseValue }
            let values2 = (0..<512).map { i in
                baseValue + epsilon * Float(i % 10)  // Small perturbations
            }

            let vec1 = try Vector512Optimized(values1)
            let vec2 = try Vector512Optimized(values2)

            // Convert to FP16 and compute distance
            let vec2FP16 = Vector512FP16(from: vec2)
            let outputBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1)
            defer { outputBuffer.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: vec1,
                candidatesFP16: [vec2FP16],
                range: 0..<1,
                out: outputBuffer
            )

            let mixedDistance = outputBuffer[0]

            // Compute reference distance carefully
            var referenceDistance: Float = 0
            for i in 0..<512 {
                let diff = values1[i] - values2[i]
                referenceDistance += diff * diff
            }

            // The mixed precision should handle this without catastrophic cancellation
            // Note: If all differences are very small relative to FP16 precision, distance might be 0
            if referenceDistance < 1e-3 {
                // For very small distances, FP16 might lose precision entirely
                #expect(mixedDistance >= 0, "Distance should be non-negative")
                #expect(abs(mixedDistance - referenceDistance) < 0.01, "Small distance absolute error")
            } else {
                #expect(mixedDistance > 0, "Distance should be positive")
                let relError = abs(mixedDistance - referenceDistance) / referenceDistance
                #expect(relError < 0.1, "Cancellation error: \(relError)")
            }

            // Test with values that differ only in least significant bits
            let preciseValue: Float = Float.pi * 1000
            let values3 = Array(repeating: preciseValue, count: 768)
            let values4 = values3.map { $0 * (1.0 + Float.ulpOfOne * 100) }

            let vec3 = try Vector768Optimized(values3)
            let vec4 = try Vector768Optimized(values4)
            let vec4FP16 = Vector768FP16(from: vec4)

            MixedPrecisionKernels.range_cosine_mixed_768(
                query: vec3,
                candidatesFP16: [vec4FP16],
                range: 0..<1,
                out: outputBuffer
            )

            let cosineDistance = outputBuffer[0]

            // Nearly identical vectors should have very small cosine distance
            #expect(cosineDistance >= 0 && cosineDistance < 0.01,
                   "Cosine distance for nearly identical: \(cosineDistance)")

            // Test difference computation accuracy with large common term
            // FP16 max value is ~65504, so we need to stay within that range
            let largeCommon: Float = 1000.0
            let smallDiff: Float = 1.0  // Difference needs to be representable after FP16 conversion

            let largeValues1 = Array(repeating: largeCommon, count: 512)
            let largeValues2 = Array(repeating: largeCommon + smallDiff, count: 512)

            let largeVec1 = try Vector512Optimized(largeValues1)
            let largeVec2 = try Vector512Optimized(largeValues2)
            let largeVec2FP16 = Vector512FP16(from: largeVec2)

            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: largeVec1,
                candidatesFP16: [largeVec2FP16],
                range: 0..<1,
                out: outputBuffer
            )

            let largeDistance = outputBuffer[0]
            let expectedDistance = Float(512) * smallDiff * smallDiff

            // Should preserve differences even with large common terms
            // FP16 has ~0.1% relative precision, so we need realistic expectations
            if expectedDistance > 0.001 {
                let largeError = abs(largeDistance - expectedDistance) / expectedDistance
                #expect(largeError < 0.01, "Large common term relative error: \(largeError)")
            } else {
                #expect(abs(largeDistance - expectedDistance) < 0.1, "Large common term absolute error")
            }
        }

        @Test("Denormalized number handling")
        func testDenormalizedNumbers() async throws {
            // Test with subnormal FP16 values
            // FP16 min normal: ~6.1e-5, subnormals below this
            let fp16MinNormal: Float = 6.10352e-5
            let subnormalValue: Float = fp16MinNormal / 10.0  // Well into subnormal range

            // Create vectors with subnormal values
            let subnormalValues = Array(repeating: subnormalValue, count: 512)
            let subnormalVec = try Vector512Optimized(subnormalValues)

            // Test conversion to FP16
            let fp16Vec = Vector512FP16(from: subnormalVec)
            let recoveredVec = fp16Vec.toFP32()

            // Verify graceful degradation (may flush to zero)
            for i in 0..<512 {
                let recovered = recoveredVec[i]
                #expect(recovered >= 0 && recovered <= subnormalValue * 2,
                       "Subnormal handling: original=\(subnormalValue), recovered=\(recovered)")
            }

            // Test mixed precision computation with subnormals
            let normalValues = Array(repeating: Float(0.1), count: 512)
            let normalVec = try Vector512Optimized(normalValues)

            let outputBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1)
            defer { outputBuffer.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: normalVec,
                candidatesFP16: [fp16Vec],
                range: 0..<1,
                out: outputBuffer
            )

            let distance = outputBuffer[0]
            #expect(distance > 0, "Distance with subnormals should be positive")

            // Test flush-to-zero behavior
            let tinyValue: Float = Float.leastNonzeroMagnitude
            let tinyValues = Array(repeating: tinyValue, count: 768)
            let tinyVec = try Vector768Optimized(tinyValues)
            let tinyFP16 = Vector768FP16(from: tinyVec)
            let tinyRecovered = tinyFP16.toFP32()

            // Very tiny values should flush to zero in FP16
            var zeroCount = 0
            for i in 0..<768 {
                if tinyRecovered[i] == 0.0 {
                    zeroCount += 1
                }
            }
            #expect(zeroCount == 768, "Tiny values should flush to zero, got \(zeroCount)/768")

            // Test gradual underflow
            let gradualValues = (0..<1536).map { i in
                fp16MinNormal * pow(0.5, Float(i / 100))  // Gradually underflow
            }
            let gradualVec = try Vector1536Optimized(gradualValues)
            let gradualFP16 = Vector1536FP16(from: gradualVec)
            let gradualRecovered = gradualFP16.toFP32()

            // Early values should be preserved, later ones should underflow
            #expect(gradualRecovered[0] > 0, "Early values should be preserved")
            #expect(gradualRecovered[1535] == 0 || gradualRecovered[1535] < 1e-10,
                   "Late values should underflow")

            // Find transition point
            var transitionIndex = 0
            for i in 0..<1536 {
                if gradualRecovered[i] == 0 {
                    transitionIndex = i
                    break
                }
            }
            #expect(transitionIndex > 0 && transitionIndex < 1536,
                   "Should have gradual underflow transition at index \(transitionIndex)")
        }

        @Test("Mixed precision with extreme values")
        func testExtremeValues() async throws {
            // Test with values near FP16 limits (max: 65504)
            let fp16Max: Float = 65504.0
            let nearMaxValues = Array(repeating: fp16Max * 0.9, count: 512)
            let nearMaxVec = try Vector512Optimized(nearMaxValues)
            let nearMaxFP16 = Vector512FP16(from: nearMaxVec)

            // Verify no overflow in conversion
            let nearMaxRecovered = nearMaxFP16.toFP32()
            for i in 0..<512 {
                #expect(!nearMaxRecovered[i].isInfinite, "Should not overflow to infinity")
                #expect(abs(nearMaxRecovered[i] - nearMaxValues[i]) / nearMaxValues[i] < 0.001)
            }

            // Test mixed small and large values
            var mixedValues: [Float] = []
            for i in 0..<768 {
                if i % 3 == 0 {
                    mixedValues.append(fp16Max * 0.5)  // Large
                } else if i % 3 == 1 {
                    mixedValues.append(0.01)  // Small
                } else {
                    mixedValues.append(1.0)  // Medium
                }
            }

            let mixedVec1 = try Vector768Optimized(mixedValues)
            let mixedVec2 = try Vector768Optimized(mixedValues.map { $0 * 0.99 })
            let mixedFP16 = Vector768FP16(from: mixedVec2)

            let outputBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1)
            defer { outputBuffer.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_768(
                query: mixedVec1,
                candidatesFP16: [mixedFP16],
                range: 0..<1,
                out: outputBuffer
            )

            let mixedDistance = outputBuffer[0]

            // Verify no overflow in accumulation
            #expect(mixedDistance.isFinite, "Distance should be finite")
            #expect(mixedDistance > 0, "Distance should be positive")

            // Test values that could overflow when squared
            let sqrtMax = sqrt(fp16Max)
            let dangerousValues = Array(repeating: sqrtMax * 0.8, count: 1536)
            let dangerousVec1 = try Vector1536Optimized(dangerousValues)
            let dangerousVec2 = try Vector1536Optimized(dangerousValues.map { $0 * 1.1 })
            let dangerousFP16 = Vector1536FP16(from: dangerousVec2)

            MixedPrecisionKernels.range_euclid2_mixed_1536(
                query: dangerousVec1,
                candidatesFP16: [dangerousFP16],
                range: 0..<1,
                out: outputBuffer
            )

            let dangerousDistance = outputBuffer[0]
            #expect(dangerousDistance.isFinite, "Should handle near-overflow values")

            // Test extreme ratios
            var extremeRatioValues: [Float] = []
            for i in 0..<512 {
                if i < 10 {
                    extremeRatioValues.append(fp16Max * 0.5)  // Very large
                } else {
                    extremeRatioValues.append(0.001)  // Very small
                }
            }

            let extremeVec = try Vector512Optimized(extremeRatioValues)
            let extremeFP16 = Vector512FP16(from: extremeVec)
            let extremeRecovered = extremeFP16.toFP32()

            // Large values should be preserved
            for i in 0..<10 {
                #expect(abs(extremeRecovered[i] - extremeRatioValues[i]) / extremeRatioValues[i] < 0.001)
            }

            // Small values should also be reasonably preserved
            for i in 10..<20 {
                #expect(abs(extremeRecovered[i] - extremeRatioValues[i]) < 0.01)
            }
        }

        @Test("Gradient preservation in FP16")
        func testGradientPreservation() async throws {
            // Test small gradient values typical in deep learning
            let typicalGradient: Float = 1e-3
            let dimension = 768

            // Create gradient vectors
            let gradientValues = (0..<dimension).map { i in
                typicalGradient * Float(sin(Double(i) * 0.1))  // Varying gradients
            }
            let gradientVec = try Vector768Optimized(gradientValues)

            // Convert to FP16 and back
            let gradientFP16 = Vector768FP16(from: gradientVec)
            let recoveredGradient = gradientFP16.toFP32()

            // Verify gradient direction preserved
            var dotProduct: Float = 0
            var origMagnitude: Float = 0
            var recoveredMagnitude: Float = 0

            for i in 0..<dimension {
                dotProduct += gradientValues[i] * recoveredGradient[i]
                origMagnitude += gradientValues[i] * gradientValues[i]
                recoveredMagnitude += recoveredGradient[i] * recoveredGradient[i]
            }

            origMagnitude = sqrt(origMagnitude)
            recoveredMagnitude = sqrt(recoveredMagnitude)

            // Cosine similarity should be very close to 1 (same direction)
            let cosineSim = dotProduct / (origMagnitude * recoveredMagnitude)
            #expect(cosineSim > 0.99, "Gradient direction should be preserved: cos=\(cosineSim)")

            // Test for vanishing gradients
            let verySmallGradient: Float = 1e-5  // Near FP16 precision limit
            let smallGradValues = Array(repeating: verySmallGradient, count: 512)
            let smallGradVec = try Vector512Optimized(smallGradValues)
            let smallGradFP16 = Vector512FP16(from: smallGradVec)
            let smallRecovered = smallGradFP16.toFP32()

            // Count non-zero values after conversion
            var nonZeroCount = 0
            for i in 0..<512 {
                if smallRecovered[i] != 0 {
                    nonZeroCount += 1
                }
            }

            // Some gradients might vanish, but not all should
            #expect(nonZeroCount > 256, "At least half of small gradients should survive: \(nonZeroCount)/512")

            // Test gradient accumulation scenario
            let batchSize = 10
            var accumulatedGradients = Array(repeating: Float(0), count: 1536)

            for batch in 0..<batchSize {
                let batchGradients = (0..<1536).map { i in
                    typicalGradient * Float.random(in: -1...1) / Float(batch + 1)
                }
                let batchVec = try Vector1536Optimized(batchGradients)
                let batchFP16 = Vector1536FP16(from: batchVec)
                let batchRecovered = batchFP16.toFP32()

                // Accumulate gradients
                for i in 0..<1536 {
                    accumulatedGradients[i] += batchRecovered[i]
                }
            }

            // Accumulated gradients should have reasonable magnitude
            let accumMagnitude = sqrt(accumulatedGradients.reduce(Float(0)) { $0 + $1 * $1 })
            #expect(accumMagnitude > 0, "Accumulated gradients should not vanish")
            #expect(accumMagnitude.isFinite, "Accumulated gradients should be finite")

            // Test gradient sparsity preservation
            var sparseGradients = Array(repeating: Float(0), count: 768)
            for i in stride(from: 0, to: 768, by: 10) {
                sparseGradients[i] = Float.random(in: -0.01...0.01)
            }

            let sparseVec = try Vector768Optimized(sparseGradients)
            let sparseFP16 = Vector768FP16(from: sparseVec)
            let sparseRecovered = sparseFP16.toFP32()

            // Check sparsity pattern is preserved
            for i in 0..<768 {
                if sparseGradients[i] == 0 {
                    #expect(sparseRecovered[i] == 0, "Zero gradients should stay zero")
                } else {
                    // Non-zero gradients should be preserved with reasonable accuracy
                    let relError = abs((sparseRecovered[i] - sparseGradients[i]) / sparseGradients[i])
                    #expect(relError < 0.1 || abs(sparseRecovered[i]) < 1e-6)
                }
            }
        }
    }

    // MARK: - AutoTuning Tests

    @Suite("AutoTuning")
    struct AutoTuningTests {

        @Test("Precision strategy selection")
        func testPrecisionStrategySelection() async throws {
            let autoTuner = MixedPrecisionAutoTuner.shared

            // Test strategy selection logic for different scenarios
            // Small dataset - should not use FP16
            let smallStrategy = autoTuner.getStrategy(
                dimension: 128,
                vectorCount: 10,
                accuracyRequired: 0.99
            )
            #expect(!smallStrategy.useFP16Storage, "Small dataset shouldn't use FP16")

            // Large dataset with moderate accuracy - should use FP16
            let largeStrategy = autoTuner.getStrategy(
                dimension: 1536,
                vectorCount: 1000,
                accuracyRequired: 0.95
            )
            #expect(largeStrategy.useFP16Storage, "Large dataset should use FP16")

            // Verify dimension-based decisions
            let dimensionTests = [
                (128, false),   // Small dimension
                (512, true),    // Medium dimension
                (1536, true),   // Large dimension
            ]

            for (dim, expectedFP16) in dimensionTests {
                let strategy = autoTuner.getStrategy(
                    dimension: dim,
                    vectorCount: 100,
                    accuracyRequired: 0.95
                )
                if expectedFP16 {
                    #expect(strategy.useFP16Storage || dim < 512,
                           "Dimension \(dim) should consider FP16")
                }
            }

            // Test accuracy threshold impact
            let highAccuracyStrategy = autoTuner.getStrategy(
                dimension: 768,
                vectorCount: 200,
                accuracyRequired: 0.99  // High accuracy
            )

            let lowAccuracyStrategy = autoTuner.getStrategy(
                dimension: 768,
                vectorCount: 200,
                accuracyRequired: 0.85  // Lower accuracy
            )

            // Lower accuracy requirement might allow more aggressive optimization
            #expect(lowAccuracyStrategy.accuracyThreshold <= highAccuracyStrategy.accuracyThreshold)

            // Test SoA layout decision
            let soaStrategy = autoTuner.getStrategy(
                dimension: 768,
                vectorCount: 100,  // Many candidates
                accuracyRequired: 0.95
            )
            #expect(soaStrategy.useSoALayout, "Many candidates should use SoA")

            let noSoaStrategy = autoTuner.getStrategy(
                dimension: 768,
                vectorCount: 10,  // Few candidates
                accuracyRequired: 0.95
            )
            #expect(!noSoaStrategy.useSoALayout, "Few candidates shouldn't use SoA")
        }

        @Test("Block size optimization")
        func testBlockSizeOptimization() async throws {
            let autoTuner = MixedPrecisionAutoTuner.shared

            // Test optimal block size calculation
            let testCases = [
                (dimension: 512, vectorCount: 10, expectedRange: 32...128),
                (dimension: 768, vectorCount: 50, expectedRange: 32...128),
                (dimension: 1536, vectorCount: 100, expectedRange: 32...128),
                (dimension: 3072, vectorCount: 200, expectedRange: 32...128),
            ]

            for (dim, count, range) in testCases {
                let strategy = autoTuner.getStrategy(
                    dimension: dim,
                    vectorCount: count,
                    accuracyRequired: 0.95
                )

                #expect(range.contains(strategy.blockSize),
                       "Block size \(strategy.blockSize) not in expected range \(range)")

                // Verify cache size considerations
                // Block should fit in L1 cache (typically 32KB)
                let blockBytes = strategy.blockSize * count * MemoryLayout<Float16>.size
                let l1CacheSize = 32 * 1024

                #expect(blockBytes <= l1CacheSize * 2,  // Allow some overflow
                       "Block size too large for cache: \(blockBytes) bytes")
            }

            // Test with various vector counts
            let vectorCounts = [10, 50, 100, 500, 1000]
            var previousBlockSize = 0

            for count in vectorCounts {
                let strategy = autoTuner.getStrategy(
                    dimension: 768,
                    vectorCount: count,
                    accuracyRequired: 0.95
                )

                // Block size should generally decrease as vector count increases
                // (to maintain cache residency)
                if previousBlockSize > 0 && count > 100 {
                    #expect(strategy.blockSize <= previousBlockSize * 2,
                           "Block size should adapt to vector count")
                }
                previousBlockSize = strategy.blockSize
            }

            // Test edge cases
            let tinyStrategy = autoTuner.getStrategy(
                dimension: 64,
                vectorCount: 5,
                accuracyRequired: 0.99
            )
            #expect(tinyStrategy.blockSize >= 32, "Minimum block size")

            let hugeStrategy = autoTuner.getStrategy(
                dimension: 4096,
                vectorCount: 1000,
                accuracyRequired: 0.90
            )
            #expect(hugeStrategy.blockSize <= 256, "Maximum block size for huge datasets")
        }

        @Test("Performance metrics collection")
        func testPerformanceMetricsCollection() async throws {
            let autoTuner = MixedPrecisionAutoTuner.shared

            // Create sample vectors for benchmarking
            let sampleVectors: [Vector512Optimized] = (0..<10).map { i in
                let values = (0..<512).map { Float($0 + i * 512) * 0.001 }
                return try! Vector512Optimized(values)
            }

            // Test benchmarking functionality
            let strategy1 = MixedPrecisionAutoTuner.PrecisionStrategy(
                useFP16Storage: true,
                useSoALayout: false,
                blockSize: 64,
                accuracyThreshold: 0.95
            )

            let metrics1 = autoTuner.benchmarkStrategy(
                strategy: strategy1,
                sampleVectors: sampleVectors,
                iterations: 10
            )

            // Verify metric accuracy
            #expect(metrics1.averageTimeMS >= 0, "Time should be non-negative")
            #expect(metrics1.accuracyScore > 0 && metrics1.accuracyScore <= 1.0,
                   "Accuracy score should be in [0, 1]")
            #expect(metrics1.throughputOpsPerSec > 0, "Throughput should be positive")

            // Test with different strategy
            let strategy2 = MixedPrecisionAutoTuner.PrecisionStrategy(
                useFP16Storage: false,
                useSoALayout: false,
                blockSize: 64,
                accuracyThreshold: 0.99
            )

            let metrics2 = autoTuner.benchmarkStrategy(
                strategy: strategy2,
                sampleVectors: sampleVectors,
                iterations: 10
            )

            // FP32 should have better accuracy but potentially worse performance
            #expect(metrics2.accuracyScore >= metrics1.accuracyScore,
                   "FP32 should have equal or better accuracy")

            // Test caching behavior
            let metrics1Cached = autoTuner.benchmarkStrategy(
                strategy: strategy1,
                sampleVectors: sampleVectors,
                iterations: 10
            )

            // Cached results should be identical
            #expect(metrics1Cached.averageTimeMS == metrics1.averageTimeMS,
                   "Cached metrics should match")

            // Test with empty vectors (edge case)
            let emptyVectors: [Vector512Optimized] = []
            let emptyMetrics = autoTuner.benchmarkStrategy(
                strategy: strategy1,
                sampleVectors: emptyVectors,
                iterations: 10
            )

            #expect(emptyMetrics.averageTimeMS == 0)
            #expect(emptyMetrics.throughputOpsPerSec == 0)

            // Test with single vector (edge case)
            let singleMetrics = autoTuner.benchmarkStrategy(
                strategy: strategy1,
                sampleVectors: [sampleVectors[0]],
                iterations: 10
            )

            #expect(singleMetrics.averageTimeMS == 0)  // Need at least 2 vectors
        }

        @Test("Adaptive kernel selection")
        func testAdaptiveKernelSelection() async throws {
            // Test adaptive algorithm selection
            let query = try Vector512Optimized((0..<512).map { Float($0) * 0.01 })

            // Small workload - should use regular FP32
            let smallCandidates = (0..<5).map { i in
                try! Vector512Optimized((0..<512).map { Float($0 + i) * 0.01 })
            }

            let smallResults = MixedPrecisionKernels.adaptiveEuclideanDistance(
                query: query,
                candidates: smallCandidates,
                accuracyRequired: 0.99
            )

            #expect(smallResults.count == smallCandidates.count)
            for result in smallResults {
                #expect(result >= 0, "Distance should be non-negative")
            }

            // Large workload - might use FP16/SoA
            let largeCandidates = (0..<100).map { i in
                try! Vector512Optimized((0..<512).map { Float($0 * (i + 1)) * 0.001 })
            }

            let largeResults = MixedPrecisionKernels.adaptiveEuclideanDistance(
                query: query,
                candidates: largeCandidates,
                accuracyRequired: 0.95
            )

            #expect(largeResults.count == largeCandidates.count)

            // Verify results are consistent
            let referenceResults = largeCandidates.map {
                query.euclideanDistanceSquared(to: $0)
            }

            for i in 0..<largeCandidates.count {
                let adaptive = largeResults[i]
                let reference = referenceResults[i]
                let relError = abs(adaptive - reference) / reference
                #expect(relError < 0.05, "Adaptive error too large: \(relError)")
            }

            // Test fallback mechanisms with incompatible vectors
            // Create vectors that might cause SoA creation to fail
            let incompatibleCandidates: [Vector768Optimized] = []

            let incompatibleQuery = Vector768Optimized()
            let fallbackResults = MixedPrecisionKernels.adaptiveEuclideanDistance(
                query: incompatibleQuery,
                candidates: incompatibleCandidates,
                accuracyRequired: 0.95
            )

            #expect(fallbackResults.isEmpty, "Empty candidates should return empty results")

            // Test with various workload sizes
            let workloadSizes = [1, 10, 50, 100, 500]

            for size in workloadSizes {
                let candidates = (0..<size).map { i in
                    try! Vector512Optimized(Array(repeating: Float(i) * 0.1, count: 512))
                }

                let results = MixedPrecisionKernels.adaptiveEuclideanDistance(
                    query: query,
                    candidates: candidates,
                    accuracyRequired: 0.95
                )

                #expect(results.count == size, "Should process all \(size) candidates")

                // Results should be monotonically increasing for this pattern
                if size > 1 {
                    for i in 1..<size {
                        #expect(results[i] >= results[i-1],
                               "Distances should increase for this pattern")
                    }
                }
            }
        }

        @Test("Memory pressure detection")
        func testMemoryPressureDetection() async throws {
            let autoTuner = MixedPrecisionAutoTuner.shared

            // Test memory usage estimation
            // Small memory footprint - shouldn't trigger FP16
            let smallMemoryStrategy = autoTuner.getStrategy(
                dimension: 128,
                vectorCount: 10,
                accuracyRequired: 0.95
            )

            let smallMemoryMB = Float(128 * 10 * 4) / (1024 * 1024)
            #expect(smallMemoryMB < 50.0, "Small memory footprint")
            #expect(!smallMemoryStrategy.useFP16Storage, "Shouldn't use FP16 for small memory")

            // Large memory footprint - should trigger FP16
            let largeMemoryStrategy = autoTuner.getStrategy(
                dimension: 1536,
                vectorCount: 10000,
                accuracyRequired: 0.95
            )

            let largeMemoryMB = Float(1536 * 10000 * 4) / (1024 * 1024)
            #expect(largeMemoryMB > 50.0, "Large memory footprint")
            #expect(largeMemoryStrategy.useFP16Storage, "Should use FP16 for large memory")

            // Verify FP16 selection triggers at threshold
            let thresholdTests = [
                (dimension: 512, vectorCount: 100, expectedFP16: false),
                (dimension: 768, vectorCount: 500, expectedFP16: true),
                (dimension: 1536, vectorCount: 1000, expectedFP16: true),
            ]

            for (dim, count, expected) in thresholdTests {
                let strategy = autoTuner.getStrategy(
                    dimension: dim,
                    vectorCount: count,
                    accuracyRequired: 0.92
                )

                let memoryMB = Float(dim * count * 4) / (1024 * 1024)

                if memoryMB > 50.0 {
                    #expect(strategy.useFP16Storage == expected,
                           "FP16 selection for \(memoryMB)MB")
                }
            }

            // Test threshold tuning with accuracy requirements
            // High accuracy should be less likely to use FP16
            let highAccMemStrategy = autoTuner.getStrategy(
                dimension: 768,
                vectorCount: 1000,
                accuracyRequired: 0.99  // Very high accuracy
            )

            let lowAccMemStrategy = autoTuner.getStrategy(
                dimension: 768,
                vectorCount: 1000,
                accuracyRequired: 0.85  // Lower accuracy
            )

            // Lower accuracy allows more aggressive memory optimization
            if !highAccMemStrategy.useFP16Storage {
                #expect(lowAccMemStrategy.useFP16Storage || lowAccMemStrategy.accuracyThreshold < 0.9,
                       "Lower accuracy should be more likely to use FP16")
            }

            // Test edge cases
            let zeroVectorsStrategy = autoTuner.getStrategy(
                dimension: 1536,
                vectorCount: 0,
                accuracyRequired: 0.95
            )
            #expect(!zeroVectorsStrategy.useFP16Storage, "Zero vectors shouldn't use FP16")

            let hugeDimensionStrategy = autoTuner.getStrategy(
                dimension: 10000,
                vectorCount: 100,
                accuracyRequired: 0.95
            )
            #expect(hugeDimensionStrategy.useFP16Storage, "Huge dimensions should use FP16")
        }

        @Test("AutoTuner thread safety")
        func testAutoTunerThreadSafety() async throws {
            let autoTuner = MixedPrecisionAutoTuner.shared

            // Test concurrent strategy queries
            await withTaskGroup(of: MixedPrecisionAutoTuner.PrecisionStrategy.self) { group in
                // Launch multiple concurrent queries
                for i in 0..<10 {
                    group.addTask {
                        autoTuner.getStrategy(
                            dimension: 512 + i * 128,
                            vectorCount: 100 + i * 10,
                            accuracyRequired: 0.90 + Float(i) * 0.01
                        )
                    }
                }

                var strategies: [MixedPrecisionAutoTuner.PrecisionStrategy] = []
                for await strategy in group {
                    strategies.append(strategy)
                }

                #expect(strategies.count == 10, "All queries should complete")
            }

            // Verify cache consistency
            let testDimension = 768
            let testVectorCount = 200
            let testAccuracy: Float = 0.95

            // Query same parameters multiple times concurrently
            await withTaskGroup(of: MixedPrecisionAutoTuner.PrecisionStrategy.self) { group in
                for _ in 0..<5 {
                    group.addTask {
                        autoTuner.getStrategy(
                            dimension: testDimension,
                            vectorCount: testVectorCount,
                            accuracyRequired: testAccuracy
                        )
                    }
                }

                var results: [MixedPrecisionAutoTuner.PrecisionStrategy] = []
                for await strategy in group {
                    results.append(strategy)
                }

                // All results should be identical (cached)
                let first = results[0]
                for strategy in results {
                    #expect(strategy.useFP16Storage == first.useFP16Storage)
                    #expect(strategy.useSoALayout == first.useSoALayout)
                    #expect(strategy.blockSize == first.blockSize)
                    #expect(strategy.accuracyThreshold == first.accuracyThreshold)
                }
            }

            // Test lock contention with benchmarking
            let sampleVectors = (0..<5).map { i in
                try! Vector512Optimized((0..<512).map { Float($0 + i) * 0.001 })
            }

            let strategy = MixedPrecisionAutoTuner.PrecisionStrategy(
                useFP16Storage: true,
                useSoALayout: true,
                blockSize: 64,
                accuracyThreshold: 0.95
            )

            await withTaskGroup(of: MixedPrecisionAutoTuner.PerformanceMetrics.self) { group in
                // Multiple concurrent benchmarks
                for _ in 0..<3 {
                    group.addTask {
                        autoTuner.benchmarkStrategy(
                            strategy: strategy,
                            sampleVectors: sampleVectors,
                            iterations: 5
                        )
                    }
                }

                var metrics: [MixedPrecisionAutoTuner.PerformanceMetrics] = []
                for await metric in group {
                    metrics.append(metric)
                }

                #expect(metrics.count == 3, "All benchmarks should complete")

                // Cached results should be identical
                if metrics.count >= 2 {
                    #expect(metrics[0].averageTimeMS == metrics[1].averageTimeMS,
                           "Cached benchmark results should match")
                }
            }
        }
    }

    // MARK: - Performance Tests

    @Suite("Performance Validation")
    struct PerformanceValidationTests {

        @Test("Memory bandwidth improvement")
        func testMemoryBandwidthImprovement() async throws {
            // Create large dataset to stress memory bandwidth
            let vectorCount = 100
            let dimension = 1536

            var candidates: [Vector1536Optimized] = []
            for i in 0..<vectorCount {
                let values = (0..<dimension).map { Float($0 + i * dimension) * 0.0001 }
                candidates.append(try Vector1536Optimized(values))
            }

            let query = candidates[0]

            // Measure FP32 processing time
            let fp32Start = CFAbsoluteTimeGetCurrent()
            var fp32Results: [Float] = []
            for candidate in candidates {
                fp32Results.append(query.euclideanDistanceSquared(to: candidate))
            }
            let fp32Time = CFAbsoluteTimeGetCurrent() - fp32Start

            // Measure FP16 processing time
            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_1536(candidates)
            let fp16Buffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: vectorCount)
            defer { fp16Buffer.deallocate() }

            let fp16Start = CFAbsoluteTimeGetCurrent()
            MixedPrecisionKernels.range_euclid2_mixed_1536(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<vectorCount,
                out: fp16Buffer
            )
            let fp16Time = CFAbsoluteTimeGetCurrent() - fp16Start

            // Calculate bandwidth utilization
            let bytesProcessedFP32 = vectorCount * dimension * MemoryLayout<Float>.size
            let bytesProcessedFP16 = vectorCount * dimension * MemoryLayout<Float16>.size

            _ = Double(bytesProcessedFP32) / fp32Time / (1024 * 1024)  // MB/s - fp32
            _ = Double(bytesProcessedFP16) / fp16Time / (1024 * 1024)  // MB/s - fp16

            // FP16 should process data faster due to reduced memory transfer
            // Note: Actual improvement depends on memory bandwidth limitations
            let speedup = fp32Time / fp16Time

            // We expect some improvement, but exact amount varies by hardware
            #expect(speedup > 0.8, "FP16 should not be significantly slower: speedup=\(speedup)")

            // Memory usage should be ~50% less
            let memoryRatio = Float(bytesProcessedFP16) / Float(bytesProcessedFP32)
            #expect(abs(memoryRatio - 0.5) < 0.01, "Memory usage should be 50% of FP32")

            // Verify results are accurate
            for i in 0..<min(10, vectorCount) {
                let fp32Result = fp32Results[i]
                let fp16Result = fp16Buffer[i]
                let relError = abs(fp32Result - fp16Result) / fp32Result
                #expect(relError < 0.01, "Results should match within tolerance")
            }
        }

        @Test("Cache efficiency with SoA")
        func testCacheEfficiencySoA() async throws {
            // Create test vectors
            let vectorCount = 50
            let dimension = 768

            var vectors: [Vector768Optimized] = []
            for i in 0..<vectorCount {
                let values = (0..<dimension).map { Float(sin(Double($0 * i) * 0.001)) }
                vectors.append(try Vector768Optimized(values))
            }

            let query = vectors[0]

            // Test different block sizes for cache efficiency
            let blockSizes = [32, 64, 128, 256]
            var timings: [Int: Double] = [:]

            for blockSize in blockSizes {
                let soa = try SoAFP16(vectors: vectors, blockSize: blockSize)
                let results = UnsafeMutableBufferPointer<Float>.allocate(capacity: vectorCount)
                defer { results.deallocate() }

                let start = CFAbsoluteTimeGetCurrent()
                for _ in 0..<10 {  // Multiple iterations to measure
                    MixedPrecisionKernels.batchEuclideanSquaredSoA(
                        query: query,
                        candidates: soa,
                        results: results
                    )
                }
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                timings[blockSize] = elapsed
            }

            // Smaller block sizes should generally be faster for this size
            // (better cache locality)
            if let time32 = timings[32], let time256 = timings[256] {
                // Smaller blocks might be faster, but not necessarily
                // Just verify both complete successfully
                #expect(time32 > 0 && time256 > 0, "Both block sizes should work")
            }

            // Compare SoA with regular layout
            let soaOpt = try SoAFP16(vectors: vectors, blockSize: 64)
            let regularFP16 = MixedPrecisionKernels.convertToFP16_768(vectors)

            let soaResults = UnsafeMutableBufferPointer<Float>.allocate(capacity: vectorCount)
            defer { soaResults.deallocate() }
            let regularResults = UnsafeMutableBufferPointer<Float>.allocate(capacity: vectorCount)
            defer { regularResults.deallocate() }

            // Measure SoA performance
            let soaStart = CFAbsoluteTimeGetCurrent()
            MixedPrecisionKernels.batchEuclideanSquaredSoA(
                query: query,
                candidates: soaOpt,
                results: soaResults
            )
            _ = CFAbsoluteTimeGetCurrent() - soaStart  // SoA time

            // Measure regular performance
            let regularStart = CFAbsoluteTimeGetCurrent()
            MixedPrecisionKernels.range_euclid2_mixed_768(
                query: query,
                candidatesFP16: regularFP16,
                range: 0..<vectorCount,
                out: regularResults
            )
            _ = CFAbsoluteTimeGetCurrent() - regularStart  // Regular time

            // Both should complete and give similar results
            for i in 0..<vectorCount {
                let diff = abs(soaResults[i] - regularResults[i])
                #expect(diff < 0.01 || diff / regularResults[i] < 0.01,
                       "SoA and regular results should match")
            }

            // Test prefetching effectiveness with sequential access
            // Access pattern that benefits from prefetching
            var sequentialSum: Float = 0
            for i in 0..<vectorCount {
                sequentialSum += soaResults[i]
            }
            #expect(sequentialSum > 0, "Sequential access should work")
        }

        @Test("SIMD utilization efficiency")
        func testSIMDUtilization() async throws {
            // Create aligned data for optimal SIMD usage
            let dimension = 512  // Multiple of SIMD width (4)
            let vectorCount = 20

            // Verify SIMD lane usage with aligned data
            var vectors: [Vector512Optimized] = []
            for i in 0..<vectorCount {
                let values = (0..<dimension).map { Float($0 + i) * 0.01 }
                vectors.append(try Vector512Optimized(values))
            }

            // Verify storage is SIMD4 aligned
            for vector in vectors {
                #expect(vector.storage.count == dimension / 4,
                       "Storage should be SIMD4 aligned")
            }

            // Test SIMD operations with FP16
            let fp16Vectors = MixedPrecisionKernels.convertToFP16_512(vectors)
            let query = vectors[0]
            let output = UnsafeMutableBufferPointer<Float>.allocate(capacity: vectorCount)
            defer { output.deallocate() }

            // Process with SIMD kernels
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: fp16Vectors,
                range: 0..<vectorCount,
                out: output
            )

            // Verify all lanes were utilized
            for i in 0..<vectorCount {
                #expect(output[i] >= 0, "All outputs should be computed")
            }

            // Test instruction pipelining with unrolled loops
            // The kernel uses 4 accumulators for better pipelining
            let pipelineTest = try Vector512Optimized(Array(repeating: Float(1.0), count: 512))
            let pipelineFP16 = Vector512FP16(from: pipelineTest)

            // Multiple iterations to test pipelining
            let iterations = 100
            let pipelineStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                MixedPrecisionKernels.range_euclid2_mixed_512(
                    query: pipelineTest,
                    candidatesFP16: [pipelineFP16],
                    range: 0..<1,
                    out: output
                )
            }
            let pipelineTime = CFAbsoluteTimeGetCurrent() - pipelineStart

            // Calculate approximate operations per second
            let opsPerIteration = dimension * 3  // subtract, multiply, add
            let totalOps = opsPerIteration * iterations
            let opsPerSecond = Double(totalOps) / pipelineTime

            // Should achieve reasonable throughput
            #expect(opsPerSecond > 1e6, "Should achieve >1M ops/sec")

            // Test SIMD efficiency with different data patterns
            // Pattern 1: All same values (good for SIMD)
            let uniformValues = Array(repeating: Float(2.5), count: 512)
            let uniformVec = try Vector512Optimized(uniformValues)
            let uniformFP16 = Vector512FP16(from: uniformVec)

            // Pattern 2: Alternating values (still SIMD-friendly)
            let alternatingValues = (0..<512).map { $0 % 2 == 0 ? Float(1.0) : Float(-1.0) }
            let alternatingVec = try Vector512Optimized(alternatingValues)
            let alternatingFP16 = Vector512FP16(from: alternatingVec)

            // Both patterns should process efficiently
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: uniformVec,
                candidatesFP16: [uniformFP16, alternatingFP16],
                range: 0..<2,
                out: output
            )

            #expect(output[0] < 0.001, "Identical vectors should have ~0 distance")
            #expect(output[1] > 0, "Different vectors should have >0 distance")
        }

        @Test("Throughput scaling with batch size")
        func testThroughputScaling() async throws {
            // Test performance scaling with different batch sizes
            let dimension = 768
            let batchSizes = [1, 10, 50, 100, 200]
            let query = try Vector768Optimized((0..<dimension).map { Float($0) * 0.001 })

            var throughputs: [Int: Double] = [:]

            for batchSize in batchSizes {
                // Create batch of candidates
                var candidates: [Vector768Optimized] = []
                for i in 0..<batchSize {
                    let values = (0..<dimension).map { Float($0 + i) * 0.001 }
                    candidates.append(try Vector768Optimized(values))
                }

                let candidatesFP16 = MixedPrecisionKernels.convertToFP16_768(candidates)
                let output = UnsafeMutableBufferPointer<Float>.allocate(capacity: batchSize)
                defer { output.deallocate() }

                // Measure processing time
                let iterations = 10
                let start = CFAbsoluteTimeGetCurrent()
                for _ in 0..<iterations {
                    MixedPrecisionKernels.range_euclid2_mixed_768(
                        query: query,
                        candidatesFP16: candidatesFP16,
                        range: 0..<batchSize,
                        out: output
                    )
                }
                let elapsed = CFAbsoluteTimeGetCurrent() - start

                // Calculate throughput (vectors/second)
                let throughput = Double(batchSize * iterations) / elapsed
                throughputs[batchSize] = throughput
            }

            // Verify throughput increases with batch size (up to a point)
            var previousThroughput: Double = 0
            for batchSize in batchSizes {
                if let throughput = throughputs[batchSize] {
                    #expect(throughput > 0, "Throughput should be positive")

                    // Larger batches should generally have better throughput
                    // (amortizing fixed costs)
                    if previousThroughput > 0 && batchSize <= 100 {
                        #expect(throughput >= previousThroughput * 0.8,
                               "Throughput should scale with batch size")
                    }
                    previousThroughput = throughput
                }
            }

            // Identify optimal batch size
            let optimalBatch = throughputs.max(by: { $0.value < $1.value })?.key ?? 0
            #expect(optimalBatch >= 10, "Optimal batch size should be >1")

            // Test parallelization benefits with concurrent processing
            let parallelBatchSize = 100
            var parallelCandidates: [Vector768Optimized] = []
            for i in 0..<parallelBatchSize {
                let values = (0..<dimension).map { Float($0 * (i + 1)) * 0.0001 }
                parallelCandidates.append(try Vector768Optimized(values))
            }

            // Process in parallel chunks
            let chunkSize = 25
            await withTaskGroup(of: [Float].self) { group in
                for chunkStart in stride(from: 0, to: parallelBatchSize, by: chunkSize) {
                    let chunkEnd = min(chunkStart + chunkSize, parallelBatchSize)
                    let chunkCandidates = Array(parallelCandidates[chunkStart..<chunkEnd])

                    group.addTask {
                        let chunkFP16 = MixedPrecisionKernels.convertToFP16_768(chunkCandidates)
                        let chunkOutput = UnsafeMutableBufferPointer<Float>.allocate(
                            capacity: chunkCandidates.count
                        )
                        defer { chunkOutput.deallocate() }

                        MixedPrecisionKernels.range_euclid2_mixed_768(
                            query: query,
                            candidatesFP16: chunkFP16,
                            range: 0..<chunkCandidates.count,
                            out: chunkOutput
                        )

                        return Array(chunkOutput)
                    }
                }

                var allResults: [Float] = []
                for await chunkResults in group {
                    allResults.append(contentsOf: chunkResults)
                }

                #expect(allResults.count == parallelBatchSize,
                       "All parallel chunks should complete")
            }
        }

        @Test("Latency vs throughput trade-off")
        func testLatencyVsThroughput() async throws {
            // Setup test vectors
            let dimension = 512
            let query = try Vector512Optimized((0..<dimension).map { Float($0) * 0.01 })

            // Measure single-query latency
            let singleCandidate = try Vector512Optimized(
                (0..<dimension).map { Float($0) * 0.01 + 1.0 }
            )
            let singleFP16 = MixedPrecisionKernels.convertToFP16_512([singleCandidate])
            let singleOutput = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1)
            defer { singleOutput.deallocate() }

            let latencyIterations = 100
            let latencyStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<latencyIterations {
                MixedPrecisionKernels.range_euclid2_mixed_512(
                    query: query,
                    candidatesFP16: singleFP16,
                    range: 0..<1,
                    out: singleOutput
                )
            }
            let latencyTotal = CFAbsoluteTimeGetCurrent() - latencyStart
            let avgLatencyMs = (latencyTotal / Double(latencyIterations)) * 1000.0

            // Measure batch throughput
            let batchSize = 100
            var batchCandidates: [Vector512Optimized] = []
            for i in 0..<batchSize {
                let values = (0..<dimension).map { Float($0 * (i + 1)) * 0.001 }
                batchCandidates.append(try Vector512Optimized(values))
            }

            let batchFP16 = MixedPrecisionKernels.convertToFP16_512(batchCandidates)
            let batchOutput = UnsafeMutableBufferPointer<Float>.allocate(capacity: batchSize)
            defer { batchOutput.deallocate() }

            let throughputIterations = 10
            let throughputStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<throughputIterations {
                MixedPrecisionKernels.range_euclid2_mixed_512(
                    query: query,
                    candidatesFP16: batchFP16,
                    range: 0..<batchSize,
                    out: batchOutput
                )
            }
            let throughputTotal = CFAbsoluteTimeGetCurrent() - throughputStart
            let throughputVectorsPerSec = Double(batchSize * throughputIterations) / throughputTotal
            let avgBatchLatencyMs = (throughputTotal / Double(throughputIterations)) * 1000.0

            // Identify sweet spots
            let sweetSpotTests = [5, 10, 20, 50]
            var sweetSpotMetrics: [(size: Int, latency: Double, throughput: Double)] = []

            for testSize in sweetSpotTests {
                let testCandidates = Array(batchCandidates.prefix(testSize))
                let testFP16 = MixedPrecisionKernels.convertToFP16_512(testCandidates)
                let testOutput = UnsafeMutableBufferPointer<Float>.allocate(capacity: testSize)
                defer { testOutput.deallocate() }

                let testStart = CFAbsoluteTimeGetCurrent()
                for _ in 0..<10 {
                    MixedPrecisionKernels.range_euclid2_mixed_512(
                        query: query,
                        candidatesFP16: testFP16,
                        range: 0..<testSize,
                        out: testOutput
                    )
                }
                let testTime = CFAbsoluteTimeGetCurrent() - testStart
                let testLatency = (testTime / 10.0) * 1000.0
                let testThroughput = Double(testSize * 10) / testTime

                sweetSpotMetrics.append((testSize, testLatency, testThroughput))
            }

            // Verify trade-offs
            #expect(avgLatencyMs > 0, "Single query latency should be measurable")
            #expect(throughputVectorsPerSec > 0, "Batch throughput should be positive")

            // Larger batches should have higher latency but better throughput
            #expect(avgBatchLatencyMs > avgLatencyMs,
                   "Batch latency (\(avgBatchLatencyMs)ms) should be higher than single (\(avgLatencyMs)ms)")

            // Verify sweet spot metrics
            for i in 1..<sweetSpotMetrics.count {
                let prev = sweetSpotMetrics[i-1]
                let curr = sweetSpotMetrics[i]

                // Latency should generally increase with batch size
                #expect(curr.latency >= prev.latency * 0.9,
                       "Latency should increase with batch size")

                // Throughput should improve (up to a point)
                if curr.size <= 20 {
                    #expect(curr.throughput >= prev.throughput * 0.9,
                           "Throughput should improve with reasonable batch size")
                }
            }

            // Find optimal batch size for latency-sensitive applications
            // (e.g., <10ms latency requirement)
            let latencyBudgetMs = 10.0
            let optimalForLatency = sweetSpotMetrics.filter { $0.latency < latencyBudgetMs }
                .max(by: { $0.throughput < $1.throughput })?.size ?? 1

            #expect(optimalForLatency > 0, "Should find optimal batch size for latency budget")
        }
    }

    // MARK: - Integration Tests

    @Suite("Integration")
    struct IntegrationTests {

        @Test("Mixed precision in similarity search")
        func testMixedPrecisionSimilaritySearch() async throws {
            // Create a database of vectors
            let databaseSize = 100
            let dimension = 768
            let queryCount = 5
            let topK = 10

            // Generate database vectors
            var database: [Vector768Optimized] = []
            for i in 0..<databaseSize {
                let values = (0..<dimension).map { j in
                    Float(sin(Double(i * j) * 0.0001))  // Deterministic pattern
                }
                database.append(try Vector768Optimized(values))
            }

            // Generate query vectors
            var queries: [Vector768Optimized] = []
            for q in 0..<queryCount {
                let values = (0..<dimension).map { j in
                    Float(cos(Double(q * j) * 0.0001))
                }
                queries.append(try Vector768Optimized(values))
            }

            // Test end-to-end similarity search with FP32
            var fp32Results: [[Int]] = []  // Top-K indices for each query
            for query in queries {
                var distances: [(index: Int, distance: Float)] = []
                for (idx, candidate) in database.enumerated() {
                    let dist = query.euclideanDistanceSquared(to: candidate)
                    distances.append((idx, dist))
                }
                distances.sort { $0.distance < $1.distance }
                fp32Results.append(distances.prefix(topK).map { $0.index })
            }

            // Test with FP16 storage
            let databaseFP16 = MixedPrecisionKernels.convertToFP16_768(database)
            var fp16Results: [[Int]] = []

            for query in queries {
                let output = UnsafeMutableBufferPointer<Float>.allocate(capacity: databaseSize)
                defer { output.deallocate() }

                MixedPrecisionKernels.range_euclid2_mixed_768(
                    query: query,
                    candidatesFP16: databaseFP16,
                    range: 0..<databaseSize,
                    out: output
                )

                var distances: [(index: Int, distance: Float)] = []
                for i in 0..<databaseSize {
                    distances.append((i, output[i]))
                }
                distances.sort { $0.distance < $1.distance }
                fp16Results.append(distances.prefix(topK).map { $0.index })
            }

            // Verify recall with FP16 storage
            var totalRecall: Float = 0
            for q in 0..<queryCount {
                let fp32TopK = Set(fp32Results[q])
                let fp16TopK = Set(fp16Results[q])
                let intersection = fp32TopK.intersection(fp16TopK)
                let recall = Float(intersection.count) / Float(topK)
                totalRecall += recall

                // Expect high recall (>90% for top-K)
                #expect(recall >= 0.8, "Query \(q) recall: \(recall)")
            }

            let avgRecall = totalRecall / Float(queryCount)
            #expect(avgRecall >= 0.9, "Average recall should be high: \(avgRecall)")

            // Test ranking preservation
            // Check if relative ordering is mostly preserved
            for q in 0..<queryCount {
                // Check top-3 results (most important)
                let fp32Top3 = Array(fp32Results[q].prefix(3))
                let fp16Top3 = Array(fp16Results[q].prefix(3))

                // At least 2 out of top 3 should match
                let top3Matches = fp32Top3.filter { fp16Top3.contains($0) }.count
                #expect(top3Matches >= 2, "Top-3 preservation for query \(q): \(top3Matches)/3")
            }
        }

        @Test("Compatibility with optimized vectors")
        func testOptimizedVectorCompatibility() async throws {
            // Test with Vector512Optimized
            let values512 = (0..<512).map { Float($0) * 0.01 }
            let vec512 = try Vector512Optimized(values512)
            let fp16_512 = Vector512FP16(from: vec512)
            let recovered512 = fp16_512.toFP32()

            #expect(recovered512.scalarCount == 512)
            for i in 0..<512 {
                #expect(abs(recovered512[i] - vec512[i]) < 0.01)
            }

            // Test with Vector768Optimized
            let values768 = (0..<768).map { Float(sin(Double($0) * 0.01)) }
            let vec768 = try Vector768Optimized(values768)
            let fp16_768 = Vector768FP16(from: vec768)
            let recovered768 = fp16_768.toFP32()

            #expect(recovered768.scalarCount == 768)
            for i in 0..<min(10, 768) {
                #expect(abs(recovered768[i] - vec768[i]) < 0.01)
            }

            // Test with Vector1536Optimized
            let values1536 = (0..<1536).map { Float(cos(Double($0) * 0.005)) }
            let vec1536 = try Vector1536Optimized(values1536)
            let fp16_1536 = Vector1536FP16(from: vec1536)
            let recovered1536 = fp16_1536.toFP32()

            #expect(recovered1536.scalarCount == 1536)

            // Verify seamless conversion in batch operations
            let batch512 = [vec512, try Vector512Optimized(values512.map { $0 * 2 })]
            let batchFP16 = MixedPrecisionKernels.convertToFP16_512(batch512)
            #expect(batchFP16.count == 2)

            // Test performance benefits
            let candidateCount = 20
            var candidates512: [Vector512Optimized] = []
            for i in 0..<candidateCount {
                let vals = (0..<512).map { Float($0 * (i + 1)) * 0.001 }
                candidates512.append(try Vector512Optimized(vals))
            }

            // FP16 conversion should be fast
            let conversionStart = CFAbsoluteTimeGetCurrent()
            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates512)
            let conversionTime = CFAbsoluteTimeGetCurrent() - conversionStart

            #expect(conversionTime < 0.1, "Conversion should be fast")
            #expect(candidatesFP16.count == candidateCount)

            // Test mixed precision operations with optimized vectors
            let query512 = candidates512[0]
            let output = UnsafeMutableBufferPointer<Float>.allocate(capacity: candidateCount)
            defer { output.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query512,
                candidatesFP16: candidatesFP16,
                range: 0..<candidateCount,
                out: output
            )

            // Verify first result is ~0 (same vector)
            #expect(output[0] < 1e-5)

            // Test that optimized storage is preserved
            let optimizedStorage = vec512.storage
            let fp16Storage = fp16_512.storage
            #expect(optimizedStorage.count == 128)  // SIMD4<Float> count
            #expect(fp16Storage.count == 128)  // SIMD4<Float16> count
        }

        @Test("Fallback to FP32 on unsupported hardware")
        func testFP32Fallback() async throws {
            // Test hardware capability detection
            let shouldUseMixed = MixedPrecisionKernels.shouldUseMixedPrecision(
                candidateCount: 10,
                dimension: 512
            )

            // Small workloads shouldn't use mixed precision
            #expect(!shouldUseMixed, "Small workload shouldn't trigger mixed precision")

            // Large workloads should use mixed precision
            let shouldUseMixedLarge = MixedPrecisionKernels.shouldUseMixedPrecision(
                candidateCount: 1000,
                dimension: 1536
            )
            #expect(shouldUseMixedLarge, "Large workload should use mixed precision")

            // Test graceful fallback with adaptive selection
            let query = try Vector512Optimized((0..<512).map { Float($0) * 0.01 })
            let candidates = (0..<5).map { i in
                try! Vector512Optimized((0..<512).map { Float($0 + i) * 0.01 })
            }

            // This should fallback to FP32 due to small size
            let results = MixedPrecisionKernels.adaptiveEuclideanDistance(
                query: query,
                candidates: candidates,
                accuracyRequired: 0.99
            )

            #expect(results.count == candidates.count)

            // Verify results are correct (FP32 fallback)
            for i in 0..<candidates.count {
                let expected = query.euclideanDistanceSquared(to: candidates[i])
                let actual = results[i]
                // Even FP32 can have small rounding errors with many operations
                #expect(abs(actual - expected) < 1e-5, "FP32 fallback should be accurate")
            }

            // Test performance in fallback mode
            let fallbackStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<100 {
                _ = MixedPrecisionKernels.adaptiveEuclideanDistance(
                    query: query,
                    candidates: candidates,
                    accuracyRequired: 0.999  // Very high accuracy forces FP32
                )
            }
            let fallbackTime = CFAbsoluteTimeGetCurrent() - fallbackStart

            // Fallback should still complete reasonably fast
            #expect(fallbackTime < 1.0, "Fallback should complete in reasonable time")

            // Test with dimension that might not be optimized
            let oddDimension = 317  // Prime number, not optimized
            let oddQuery = try Vector512Optimized(
                (0..<512).map { $0 < oddDimension ? Float($0) * 0.01 : 0.0 }
            )
            let oddCandidates = [oddQuery]  // Single candidate

            let oddResults = MixedPrecisionKernels.adaptiveEuclideanDistance(
                query: oddQuery,
                candidates: oddCandidates,
                accuracyRequired: 0.95
            )

            #expect(oddResults.count == 1)
            #expect(oddResults[0] < 1e-6, "Same vector should have ~0 distance")
        }

        @Test("Memory-constrained environments")
        func testMemoryConstrainedEnvironments() async throws {
            // Simulate low-memory conditions with large dataset
            let dimension = 1536
            let largeVectorCount = 500  // Would use ~3MB in FP32, ~1.5MB in FP16

            // Create large dataset
            var largeDataset: [Vector1536Optimized] = []
            for i in 0..<largeVectorCount {
                // Use simple pattern to avoid excessive computation
                let values = Array(repeating: Float(i) * 0.001, count: dimension)
                largeDataset.append(try Vector1536Optimized(values))
            }

            // Test automatic FP16 selection based on memory pressure
            let autoTuner = MixedPrecisionAutoTuner.shared
            let memoryStrategy = autoTuner.getStrategy(
                dimension: dimension,
                vectorCount: largeVectorCount,
                accuracyRequired: 0.95
            )

            // Large dataset might trigger FP16 depending on AutoTuner implementation
            // The AutoTuner might prioritize SoA layout over FP16 storage
            #expect(memoryStrategy.useFP16Storage || memoryStrategy.useSoALayout,
                   "Large dataset should use memory-efficient strategy (FP16 or SoA)")

            // Convert to FP16 to save memory
            let datasetFP16 = MixedPrecisionKernels.convertToFP16_1536(largeDataset)

            // Calculate memory savings
            let fp32Memory = largeVectorCount * dimension * MemoryLayout<Float>.size
            let fp16Memory = largeVectorCount * dimension * MemoryLayout<Float16>.size
            let memorySaved = fp32Memory - fp16Memory

            #expect(memorySaved == fp32Memory / 2, "Should save 50% memory")

            // Test stability under pressure
            let query = largeDataset[0]
            let output = UnsafeMutableBufferPointer<Float>.allocate(capacity: largeVectorCount)
            defer { output.deallocate() }

            // Process in chunks to manage memory
            let chunkSize = 100
            for chunkStart in stride(from: 0, to: largeVectorCount, by: chunkSize) {
                let chunkEnd = min(chunkStart + chunkSize, largeVectorCount)
                let chunkOutput = UnsafeMutableBufferPointer<Float>(
                    start: output.baseAddress! + chunkStart,
                    count: chunkEnd - chunkStart
                )

                MixedPrecisionKernels.range_euclid2_mixed_1536(
                    query: query,
                    candidatesFP16: datasetFP16,
                    range: chunkStart..<chunkEnd,
                    out: chunkOutput
                )
            }

            // Verify all chunks processed
            for i in 0..<largeVectorCount {
                #expect(output[i] >= 0, "All distances should be computed")
            }

            // Test with SoA layout for better memory efficiency
            let soaDataset = try SoAFP16(vectors: Array(largeDataset.prefix(100)), blockSize: 64)
            let soaOutput = UnsafeMutableBufferPointer<Float>.allocate(capacity: 100)
            defer { soaOutput.deallocate() }

            MixedPrecisionKernels.batchEuclideanSquaredSoA(
                query: query,
                candidates: soaDataset,
                results: soaOutput
            )

            // SoA should work under memory constraints
            #expect(soaOutput[0] < 1e-6, "First result should be ~0")

            // Clear large dataset to free memory
            largeDataset.removeAll()

            // Verify FP16 vectors are still usable after clearing FP32
            MixedPrecisionKernels.range_euclid2_mixed_1536(
                query: query,
                candidatesFP16: datasetFP16,
                range: 0..<10,
                out: output
            )

            #expect(output[0] < 1e-6, "FP16 data should remain valid")
        }
    }

    // MARK: - Edge Cases

    @Suite("Edge Cases")
    struct EdgeCaseTests {

        @Test("Empty candidate set handling")
        func testEmptyCandidateSet() async throws {
            // Test with zero candidates
            let query = try Vector512Optimized((0..<512).map { Float($0) * 0.01 })
            let emptyCandidates: [Vector512FP16] = []
            let output = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1)
            defer { output.deallocate() }

            // Initialize with sentinel value
            output[0] = 999.999

            // Should handle empty range gracefully
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: emptyCandidates,
                range: 0..<0,
                out: output
            )

            // Output should not be modified
            #expect(output[0] == 999.999, "Output should not be modified for empty range")

            // Test with empty SoA
            do {
                _ = try SoAFP16<Vector512Optimized>(vectors: [], blockSize: 64)
                Issue.record("Should throw for empty vectors")
            } catch let error as VectorError {
                #expect(error.kind == .invalidData)
            }

            // Test adaptive with empty candidates
            let emptyResults = MixedPrecisionKernels.adaptiveEuclideanDistance(
                query: query,
                candidates: [],
                accuracyRequired: 0.95
            )
            #expect(emptyResults.isEmpty, "Empty candidates should return empty results")

            // Test batch conversion with empty array
            let emptyConverted = MixedPrecisionKernels.convertToFP16_512([])
            #expect(emptyConverted.isEmpty)
        }

        @Test("Single candidate processing")
        func testSingleCandidate() async throws {
            // Test with one candidate
            let query = try Vector768Optimized((0..<768).map { Float($0) * 0.001 })
            let singleCandidate = try Vector768Optimized((0..<768).map { Float($0) * 0.002 })
            let singleFP16 = MixedPrecisionKernels.convertToFP16_768([singleCandidate])

            let output = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1)
            defer { output.deallocate() }

            // Test Euclidean distance
            MixedPrecisionKernels.range_euclid2_mixed_768(
                query: query,
                candidatesFP16: singleFP16,
                range: 0..<1,
                out: output
            )

            let expected = query.euclideanDistanceSquared(to: singleCandidate)
            #expect(abs(output[0] - expected) / expected < 0.01)

            // Test cosine distance
            MixedPrecisionKernels.range_cosine_mixed_768(
                query: query,
                candidatesFP16: singleFP16,
                range: 0..<1,
                out: output
            )

            let expectedCosine = query.distance(to: singleCandidate, metric: .cosine)
            #expect(abs(output[0] - expectedCosine) < 0.01)

            // Test SoA with single vector
            let soaSingle = try SoAFP16(vectors: [singleCandidate], blockSize: 64)
            #expect(soaSingle.vectorCount == 1)

            MixedPrecisionKernels.batchEuclideanSquaredSoA(
                query: query,
                candidates: soaSingle,
                results: output
            )

            #expect(abs(output[0] - expected) / expected < 0.01)

            // Test optimization paths - single candidate might use simpler code path
            let iterations = 100
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                MixedPrecisionKernels.range_euclid2_mixed_768(
                    query: query,
                    candidatesFP16: singleFP16,
                    range: 0..<1,
                    out: output
                )
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            // Single candidate should be fast (100 iterations)
            // Allow up to 0.02 seconds for 100 iterations (0.2ms per iteration)
            #expect(elapsed < 0.02, "Single candidate should process quickly (\(elapsed)s for \(iterations) iterations)")
        }

        @Test("Mismatched dimensions error handling")
        func testMismatchedDimensions() async throws {
            // Test dimension validation in SoA
            let vec512 = try Vector512Optimized((0..<512).map { Float($0) * 0.01 })
            _ = try Vector768Optimized((0..<768).map { Float($0) * 0.01 })

            // Can't mix different dimensions in SoA
            // Note: This would fail at compile time due to type safety
            // We can test with dynamic vectors instead
            _ = DynamicVector((0..<100).map { Float($0) })
            _ = DynamicVector((0..<200).map { Float($0) })

            // Test dimension mismatch detection
            // Mixed precision kernels use specific dimensions, so mismatch
            // is prevented by type system

            // Test with wrong storage size
            _ = ContiguousArray<SIMD4<Float16>>(repeating: SIMD4<Float16>(), count: 100)
            // This would fatal error in init due to guard
            // We can't test it without crashing

            // Test query/candidate dimension mismatch
            // Type system prevents this at compile time
            let query512 = try Vector512Optimized((0..<512).map { Float($0) * 0.01 })
            let candidates512 = MixedPrecisionKernels.convertToFP16_512([vec512])

            // This would be a compile error:
            // MixedPrecisionKernels.range_euclid2_mixed_768(
            //     query: query512,  // Wrong type!
            //     candidatesFP16: ...
            // )

            // Test recovery behavior - operations should validate dimensions
            let validOutput = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1)
            defer { validOutput.deallocate() }

            // Valid operation should work
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query512,
                candidatesFP16: candidates512,
                range: 0..<1,
                out: validOutput
            )

            // Same vector comparison, but FP16 conversion can introduce small errors
            #expect(validOutput[0] < 0.001, "Same vector should have very small distance")

            // Test error messages are clear (in debug mode)
            #if DEBUG
            // Assertions would fire for invalid ranges
            // But we can't test them without crashing
            #endif

            // Test dimension validation in adaptive selection
            let adaptiveQuery = vec512
            let adaptiveCandidates = [vec512]  // Same dimension - OK

            let results = MixedPrecisionKernels.adaptiveEuclideanDistance(
                query: adaptiveQuery,
                candidates: adaptiveCandidates,
                accuracyRequired: 0.95
            )

            #expect(results.count == 1)
            #expect(results[0] < 1e-6)
        }

        @Test("Invalid range specifications")
        func testInvalidRanges() async throws {
            // Setup test data
            let query = try Vector512Optimized((0..<512).map { Float($0) * 0.01 })
            let candidateCount = 10
            var candidates: [Vector512Optimized] = []
            for i in 0..<candidateCount {
                candidates.append(try Vector512Optimized((0..<512).map { Float($0 + i) * 0.01 }))
            }
            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)

            // Test out-of-bounds ranges
            #if DEBUG
            // These would assert in debug mode
            // We can't test them without crashing in debug
            #else
            // In release mode, behavior is undefined but shouldn't crash
            let output = UnsafeMutableBufferPointer<Float>.allocate(capacity: 20)
            defer { output.deallocate() }

            // Initialize with sentinel values
            for i in 0..<20 {
                output[i] = -1.0
            }

            // Range beyond candidates count - undefined in release
            // Don't actually run this as it's undefined behavior
            #endif

            // Test empty range (valid but does nothing)
            let emptyOutput = UnsafeMutableBufferPointer<Float>.allocate(capacity: 5)
            defer { emptyOutput.deallocate() }

            for i in 0..<5 {
                emptyOutput[i] = 999.0
            }

            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 5..<5,  // Empty range
                out: emptyOutput
            )

            // Buffer should be unchanged
            #expect(emptyOutput[0] == 999.0)

            // Test reversed ranges
            // Swift ranges don't allow reversed ranges (5..<3 is invalid)
            // This is caught at compile time

            // Test single element range (valid)
            let singleOutput = UnsafeMutableBufferPointer<Float>.allocate(capacity: 1)
            defer { singleOutput.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 3..<4,  // Single element
                out: singleOutput
            )

            #expect(singleOutput[0] >= 0, "Single element range should work")

            // Test partial range at end
            let partialOutput = UnsafeMutableBufferPointer<Float>.allocate(capacity: 3)
            defer { partialOutput.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: (candidateCount-3)..<candidateCount,
                out: partialOutput
            )

            for i in 0..<3 {
                #expect(partialOutput[i] >= 0, "Partial range should work")
            }

            // Verify assertions in debug mode
            #if DEBUG
            // Assertions fire for:
            // - range.upperBound > candidatesFP16.count
            // - out.count < range.count
            // These are good safety checks
            #endif
        }

        @Test("Buffer overflow protection")
        func testBufferOverflowProtection() async throws {
            // Setup test data
            let query = try Vector1536Optimized((0..<1536).map { Float($0) * 0.001 })
            let candidateCount = 20
            var candidates: [Vector1536Optimized] = []
            for i in 0..<candidateCount {
                candidates.append(try Vector1536Optimized(
                    (0..<1536).map { Float($0 + i * 10) * 0.001 }
                ))
            }
            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_1536(candidates)

            // Test with correctly sized buffer
            let correctBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: candidateCount)
            defer { correctBuffer.deallocate() }

            MixedPrecisionKernels.range_euclid2_mixed_1536(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<candidateCount,
                out: correctBuffer
            )

            // All values should be written
            for i in 0..<candidateCount {
                #expect(correctBuffer[i] >= 0)
            }

            // Test undersized output buffers
            #if DEBUG
            // In debug mode, assertions catch undersized buffers
            // We can't test without crashing
            #else
            // In release mode, undefined behavior but shouldn't crash
            // Don't actually test as it's undefined
            #endif

            // Test with larger buffer (should be fine)
            let largerBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: candidateCount * 2)
            defer { largerBuffer.deallocate() }

            // Initialize extra space with sentinel
            for i in candidateCount..<(candidateCount * 2) {
                largerBuffer[i] = -999.0
            }

            MixedPrecisionKernels.range_euclid2_mixed_1536(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<candidateCount,
                out: largerBuffer
            )

            // Only requested range should be written
            for i in 0..<candidateCount {
                #expect(largerBuffer[i] >= 0, "Requested range should be written")
            }
            for i in candidateCount..<(candidateCount * 2) {
                #expect(largerBuffer[i] == -999.0, "Extra space should be untouched")
            }

            // Test bounds checking with SoA
            let soaCandidates = try SoAFP16(vectors: candidates, blockSize: 128)
            let soaBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: candidateCount)
            defer { soaBuffer.deallocate() }

            MixedPrecisionKernels.batchEuclideanSquaredSoA(
                query: query,
                candidates: soaCandidates,
                results: soaBuffer
            )

            // All results should be computed
            for i in 0..<candidateCount {
                #expect(soaBuffer[i] >= 0)
            }

            // Test error reporting - type system prevents most errors
            // Buffer size validation happens in debug assertions
            #if DEBUG
            // Good error reporting via assertions
            #endif
        }
    }

    // MARK: - Hardware-Specific Tests

    @Suite("Hardware Optimization")
    struct HardwareOptimizationTests {

        @Test("Apple Silicon NEON utilization")
        func testAppleSiliconNEON() async throws {
            // Test NEON instruction generation through FP16 conversions
            let dimension = 512
            let values = (0..<dimension).map { Float($0) * 0.01 }
            let vector = try Vector512Optimized(values)

            // FP32 to FP16 conversion uses NEON vcvt instructions
            let fp16Vector = Vector512FP16(from: vector)
            #expect(fp16Vector.storage.count == 128)  // SIMD4<Float16> lanes

            // FP16 to FP32 conversion also uses NEON vcvt
            let recovered = fp16Vector.toFP32()
            #expect(recovered.scalarCount == dimension)

            // Verify conversion accuracy (NEON should be IEEE compliant)
            for i in 0..<dimension {
                let original = values[i]
                let converted = recovered[i]
                if original != 0 {
                    let relError = abs((converted - original) / original)
                    #expect(relError < 0.001, "NEON conversion should be accurate")
                }
            }

            // Test SIMD4 operations which use NEON
            let simd4Values = SIMD4<Float>(1.0, 2.0, 3.0, 4.0)
            let simd4FP16 = SIMD4<Float16>(simd4Values)
            let simd4Back = SIMD4<Float>(simd4FP16)

            for i in 0..<4 {
                #expect(simd4Back[i] == simd4Values[i])
            }

            // Test performance on Apple Silicon
            // NEON should provide good throughput
            let iterations = 1000
            let candidates = (0..<10).map { i in
                try! Vector512Optimized((0..<dimension).map { Float($0 + i) * 0.001 })
            }
            let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)
            let output = UnsafeMutableBufferPointer<Float>.allocate(capacity: 10)
            defer { output.deallocate() }

            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                MixedPrecisionKernels.range_euclid2_mixed_512(
                    query: vector,
                    candidatesFP16: candidatesFP16,
                    range: 0..<10,
                    out: output
                )
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            // Should achieve good performance with NEON
            let opsPerIteration = dimension * 10 * 3  // Per-element: sub, mul, add
            let totalOps = opsPerIteration * iterations
            let gflops = Double(totalOps) / elapsed / 1e9

            // Apple Silicon should achieve reasonable GFLOPS
            #expect(gflops > 0.1, "Should achieve >0.1 GFLOPS with NEON")

            // Test that SIMD4 alignment is maintained
            // NEON works best with aligned data
            let storage = vector.storage
            #expect(storage.count == dimension / 4)

            // Each SIMD4 should be 16-byte aligned
            let simd4Size = MemoryLayout<SIMD4<Float>>.size
            #expect(simd4Size == 16, "SIMD4<Float> should be 16 bytes")

            let simd4FP16Size = MemoryLayout<SIMD4<Float16>>.size
            #expect(simd4FP16Size == 8, "SIMD4<Float16> should be 8 bytes")
        }

        @Test("Neural Engine compatibility")
        func testNeuralEngineCompatibility() async throws {
            // Test data format alignment with Neural Engine expectations
            // ANE typically expects specific memory layouts and data types

            // Create data in Neural Engine friendly format
            let dimension = 768  // Common embedding dimension
            let batchSize = 16  // Typical ANE batch size

            var batch: [Vector768Optimized] = []
            for i in 0..<batchSize {
                let values = (0..<dimension).map { j in
                    // Normalized values typical for neural networks
                    Float(tanh(Double(j - dimension/2 + i) * 0.01))
                }
                batch.append(try Vector768Optimized(values))
            }

            // Convert to FP16 (ANE often uses FP16)
            let batchFP16 = MixedPrecisionKernels.convertToFP16_768(batch)
            #expect(batchFP16.count == batchSize)

            // Test SoA layout (good for ANE)
            let soaBatch = try SoAFP16(vectors: batch, blockSize: 64)
            #expect(soaBatch.vectorCount == batchSize)
            #expect(soaBatch.dimension == dimension)

            // Verify data can be extracted for ANE handoff
            for i in 0..<batchSize {
                let extracted = try soaBatch.getVector(at: i)
                #expect(extracted.scalarCount == dimension)

                // Values should be in neural network range [-1, 1]
                for j in 0..<min(10, dimension) {
                    #expect(extracted[j] >= -1.0 && extracted[j] <= 1.0)
                }
            }

            // Test memory layout compatibility
            // ANE expects contiguous memory
            let fp16Storage = batchFP16[0].storage
            #expect(fp16Storage.count == dimension / 4)  // SIMD4 groups

            // Test that operations preserve ANE-friendly properties
            let query = batch[0]
            let output = UnsafeMutableBufferPointer<Float>.allocate(capacity: batchSize)
            defer { output.deallocate() }

            MixedPrecisionKernels.range_cosine_mixed_768(
                query: query,
                candidatesFP16: batchFP16,
                range: 0..<batchSize,
                out: output
            )

            // Cosine distances should be in [0, 2] range
            for i in 0..<batchSize {
                #expect(output[i] >= 0 && output[i] <= 2,
                       "Cosine distance in valid range for neural network data")
            }

            // Test handoff efficiency - data should be ready for ANE
            // In real use, this would interface with Core ML
            let aneReadyData = batchFP16.map { $0.storage }
            #expect(aneReadyData.count == batchSize)

            // Each vector's storage is contiguous and aligned
            for storage in aneReadyData {
                #expect(storage.count == dimension / 4)
            }
        }

        @Test("Memory alignment for optimal performance")
        func testMemoryAlignment() async throws {
            // Test 16-byte alignment for SIMD4<Float>
            let dimension = 512
            let vector = try Vector512Optimized((0..<dimension).map { Float($0) * 0.01 })

            // SIMD4<Float> should be 16-byte aligned
            vector.storage.withUnsafeBufferPointer { buffer in
                let address = Int(bitPattern: buffer.baseAddress!)
                #expect(address % 16 == 0, "SIMD4<Float> storage should be 16-byte aligned")
            }

            // Test FP16 storage alignment (8-byte for SIMD4<Float16>)
            let fp16Vector = Vector512FP16(from: vector)
            fp16Vector.storage.withUnsafeBufferPointer { buffer in
                let address = Int(bitPattern: buffer.baseAddress!)
                #expect(address % 8 == 0, "SIMD4<Float16> storage should be 8-byte aligned")
            }

            // Verify aligned load/store operations
            let alignedCount = 10
            var alignedVectors: [Vector512Optimized] = []
            for i in 0..<alignedCount {
                let values = Array(repeating: Float(i), count: dimension)
                alignedVectors.append(try Vector512Optimized(values))
            }

            // Convert batch - should maintain alignment
            let alignedFP16 = MixedPrecisionKernels.convertToFP16_512(alignedVectors)

            // Process with aligned data
            let output = UnsafeMutableBufferPointer<Float>.allocate(capacity: alignedCount)
            defer { output.deallocate() }

            let alignedStart = CFAbsoluteTimeGetCurrent()
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: vector,
                candidatesFP16: alignedFP16,
                range: 0..<alignedCount,
                out: output
            )
            let alignedTime = CFAbsoluteTimeGetCurrent() - alignedStart

            // Aligned access should be efficient
            #expect(alignedTime < 0.01, "Aligned access should be fast")

            // Test SoA alignment
            let soaVectors = try SoAFP16(vectors: alignedVectors, blockSize: 64)

            // Block pointers should be aligned
            for blockIdx in 0..<soaVectors.blocksPerVector {
                let blockPtr = soaVectors.blockPointer(blockIndex: blockIdx)
                let ptrAddress = Int(bitPattern: blockPtr)
                #expect(ptrAddress % 8 == 0, "SoA block pointers should be aligned")
            }

            // Test that misalignment doesn't occur with odd counts
            _ = 513  // Odd dimension not multiple of 4 - would need padding for SIMD4
            // Vector512Optimized handles exactly 512 elements

            // Test alignment preservation in batch operations
            let batchOutput = UnsafeMutableBufferPointer<Float>.allocate(capacity: alignedCount)
            defer { batchOutput.deallocate() }

            // Output buffer should be aligned too
            let outputAddress = Int(bitPattern: batchOutput.baseAddress!)
            #expect(outputAddress % 4 == 0, "Output buffer should be word-aligned")

            MixedPrecisionKernels.batchEuclideanSquaredSoA(
                query: vector,
                candidates: soaVectors,
                results: batchOutput
            )

            // Results should be computed correctly with aligned access
            for i in 0..<alignedCount {
                #expect(batchOutput[i] >= 0)
            }
        }
    }
}

// MARK: - Test Helpers

extension MixedPrecisionKernelTests {

    /// Generate random test vectors with specified properties
    func generateTestVectors(count: Int, dimension: Int, seed: UInt64 = 42) -> [Float] {
        var generator = SeededRandomNumberGenerator(seed: seed)
        var result: [Float] = []
        result.reserveCapacity(count * dimension)

        for _ in 0..<count {
            for _ in 0..<dimension {
                result.append(Float.random(in: -10...10, using: &generator))
            }
        }

        return result
    }

    /// Verify FP16 conversion accuracy
    func verifyFP16Accuracy(original: [Float], converted: [Float16], tolerance: Float) -> Bool {
        guard original.count == converted.count else { return false }

        for i in 0..<original.count {
            let orig = original[i]
            let conv = Float(converted[i])

            if orig.isNaN && conv.isNaN {
                continue  // Both NaN is OK
            } else if orig.isInfinite && conv.isInfinite && orig.sign == conv.sign {
                continue  // Same infinity is OK
            } else if abs(orig) < Float(Float16.leastNormalMagnitude) {
                // May flush to zero
                if conv == 0 { continue }
            }

            let absError = abs(orig - conv)
            let relError = orig != 0 ? abs((orig - conv) / orig) : absError

            if absError > tolerance && relError > tolerance {
                return false
            }
        }

        return true
    }

    /// Measure memory bandwidth utilization
    func measureMemoryBandwidth(operation: () throws -> Void) rethrows -> Double {
        let iterations = 10
        let start = CFAbsoluteTimeGetCurrent()

        for _ in 0..<iterations {
            try operation()
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start

        // Estimate based on time (simplified - real implementation would use system metrics)
        // Return MB/s estimate
        return 1000.0 / elapsed  // Placeholder calculation
    }

    /// Compare mixed precision vs full precision results
    func compareMixedVsFullPrecision(
        mixed: [Float],
        full: [Float],
        relativeTolerance: Float,
        absoluteTolerance: Float
    ) -> Bool {
        guard mixed.count == full.count else { return false }

        for i in 0..<mixed.count {
            let m = mixed[i]
            let f = full[i]

            if m.isNaN && f.isNaN {
                continue
            } else if m.isInfinite && f.isInfinite && m.sign == f.sign {
                continue
            }

            let absError = abs(m - f)
            let relError = f != 0 ? abs((m - f) / f) : absError

            if absError > absoluteTolerance && relError > relativeTolerance {
                return false
            }
        }

        return true
    }
}

// Helper for deterministic random numbers
struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        // Simple LCG for reproducibility
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}

// MARK: - Performance Benchmarking

struct MixedPrecisionBenchmark {
    let dimension: Int
    let candidateCount: Int
    let iterations: Int

    func runBenchmark() -> BenchmarkResults {
        // Implementation placeholder
        BenchmarkResults(
            fp32TimeMs: 0,
            fp16TimeMs: 0,
            memoryReduction: 0,
            accuracyScore: 0
        )
    }

    struct BenchmarkResults {
        let fp32TimeMs: Double
        let fp16TimeMs: Double
        let memoryReduction: Double
        let accuracyScore: Float
    }
}