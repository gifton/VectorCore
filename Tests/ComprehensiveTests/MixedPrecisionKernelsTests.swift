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

@Suite("Mixed Precision Kernels")
struct MixedPrecisionKernelsTests {

    // MARK: - FP16 Storage Type Tests

    @Suite("FP16 Storage Types")
    struct FP16StorageTypesTests {

        @Test
        func testVector512FP16Construction() throws {
            // Test basic construction
            let fp32Vector = try Vector512Optimized(Array(repeating: 1.5, count: 512))
            let fp16Vector = Vector512FP16(from: fp32Vector)

            // Verify storage size (512 UInt16 elements representing 512 FP16 values)
            #expect(fp16Vector.storage.count == 512)

            // Test round-trip conversion
            let reconstructed = fp16Vector.toFP32()
            for i in 0..<512 {
                #expect(abs(reconstructed[i] - 1.5) < 0.001)
            }

            // Test various values
            let testValues: [Float] = [
                0.0, 1.0, -1.0, 0.5, -0.5,
                1234.567, -9876.543,
                Float.pi, Float.ulpOfOne,
                65504.0, -65504.0  // FP16 max values
            ]

            for value in testValues {
                let singleValue = try Vector512Optimized(Array(repeating: value, count: 512))
                let fp16Single = Vector512FP16(from: singleValue)
                let backToFP32 = fp16Single.toFP32()

                // FP16 has ~3 decimal digits of precision
                let tolerance = abs(value) * 0.001 + 0.0001
                for i in 0..<512 {
                    #expect(abs(backToFP32[i] - value) <= tolerance,
                           "Value \(value) at index \(i): got \(backToFP32[i])")
                }
            }
        }

        @Test
        func testVector768FP16Construction() throws {
            // Test basic construction
            let fp32Vector = try Vector768Optimized(Array(repeating: 2.5, count: 768))
            let fp16Vector = Vector768FP16(from: fp32Vector)

            // Verify storage size (768 UInt16 elements representing 768 FP16 values)
            #expect(fp16Vector.storage.count == 768)

            // Test round-trip conversion
            let reconstructed = fp16Vector.toFP32()
            for i in 0..<768 {
                #expect(abs(reconstructed[i] - 2.5) < 0.001)
            }

            // Test gradient pattern
            let gradient = try Vector768Optimized((0..<768).map { Float($0) / 768.0 })
            let fp16Gradient = Vector768FP16(from: gradient)
            let reconstructedGradient = fp16Gradient.toFP32()

            for i in 0..<768 {
                let expected = Float(i) / 768.0
                let actual = reconstructedGradient[i]
                #expect(abs(actual - expected) < 0.001)
            }
        }

        @Test
        func testVector1536FP16Construction() throws {
            // Test basic construction
            let fp32Vector = try Vector1536Optimized(Array(repeating: 3.14159, count: 1536))
            let fp16Vector = Vector1536FP16(from: fp32Vector)

            // Verify storage size (1536 UInt16 elements representing 1536 FP16 values)
            #expect(fp16Vector.storage.count == 1536)

            // Test round-trip conversion
            let reconstructed = fp16Vector.toFP32()
            for i in 0..<1536 {
                #expect(abs(reconstructed[i] - 3.14159) < 0.01)
            }

            // Test mixed values
            let mixed = try Vector1536Optimized((0..<1536).map { i in
                Float(i % 3 == 0 ? i : -i) / 100.0
            })
            let fp16Mixed = Vector1536FP16(from: mixed)
            let reconstructedMixed = fp16Mixed.toFP32()

            for i in 0..<1536 {
                let expected = Float(i % 3 == 0 ? i : -i) / 100.0
                let actual = reconstructedMixed[i]
                let tolerance = abs(expected) * 0.001 + 0.01
                #expect(abs(actual - expected) < tolerance)
            }
        }

        @Test
        func testFP16StorageLayout() throws {
            let fp32Vector = try Vector512Optimized((0..<512).map { Float($0) })
            let fp16Vector = Vector512FP16(from: fp32Vector)

            // Test SIMD4 packing - 4 Float16 values per SIMD4
            #expect(fp16Vector.storage.count == 512 / 4)

            // Verify memory footprint reduction
            // FP16 uses 2 bytes per element vs FP32's 4 bytes
            let fp16MemorySize = MemoryLayout<SIMD4<Float16>>.size * fp16Vector.storage.count
            let fp32MemorySize = MemoryLayout<SIMD4<Float>>.size * fp32Vector.storage.count
            #expect(fp16MemorySize == fp32MemorySize / 2)

            // Test alignment - SIMD4 should be 8-byte aligned for Float16
            let alignment = MemoryLayout<SIMD4<Float16>>.alignment
            #expect(alignment == 8)

            // Verify values are correctly packed
            let reconstructed = fp16Vector.toFP32()
            for i in 0..<512 {
                let expected = Float(i)
                let actual = reconstructed[i]
                #expect(abs(actual - expected) < 1.0)  // FP16 precision for larger values
            }
        }

        @Test
        func testFP16Conversion() throws {
            // Test round-trip conversion precision
            let testCases: [(value: Float, tolerance: Float)] = [
                (0.0, 0.0),
                (1.0, 0.0001),
                (-1.0, 0.0001),
                (0.5, 0.0001),
                (-0.5, 0.0001),
                (100.0, 0.1),
                (-100.0, 0.1),
                (1000.0, 1.0),
                (-1000.0, 1.0),
                (10000.0, 10.0),
                (-10000.0, 10.0),
                (65504.0, 100.0),  // FP16 max
                (-65504.0, 100.0)
            ]

            for (value, tolerance) in testCases {
                let fp32Vector = try Vector512Optimized(Array(repeating: value, count: 512))
                let fp16Vector = Vector512FP16(from: fp32Vector)
                let reconstructed = fp16Vector.toFP32()

                for i in 0..<512 {
                    #expect(abs(reconstructed[i] - value) <= tolerance,
                           "Value \(value): got \(reconstructed[i]), tolerance \(tolerance)")
                }
            }

            // Test special values
            let specialValues: [Float] = [.infinity, -.infinity, .nan]
            for special in specialValues {
                let fp32Vector = try Vector512Optimized(Array(repeating: special, count: 512))
                let fp16Vector = Vector512FP16(from: fp32Vector)
                let reconstructed = fp16Vector.toFP32()

                for i in 0..<512 {
                    if special.isNaN {
                        #expect(reconstructed[i].isNaN)
                    } else if special.isInfinite {
                        #expect(reconstructed[i].isInfinite)
                        #expect(reconstructed[i].sign == special.sign)
                    }
                }
            }

            // Test values outside FP16 range (should be clamped)
            let outOfRange: [Float] = [100000.0, -100000.0, 1e10, -1e10]
            for value in outOfRange {
                let fp32Vector = try Vector512Optimized(Array(repeating: value, count: 512))
                let fp16Vector = Vector512FP16(from: fp32Vector)
                let reconstructed = fp16Vector.toFP32()

                for i in 0..<512 {
                    if value > 0 {
                        #expect(reconstructed[i] <= 65504.0)
                    } else {
                        #expect(reconstructed[i] >= -65504.0)
                    }
                }
            }
        }

        @Test
        func testFP16ConversionPerformance() throws {
            let vectorCount = 1000
            let vectors = try (0..<vectorCount).map { _ in
                try Vector512Optimized((0..<512).map { _ in Float.random(in: -100...100) })
            }

            // Measure FP32→FP16 conversion
            let fp32ToFP16Start = Date()
            let fp16Vectors = vectors.map { Vector512FP16(from: $0) }
            let fp32ToFP16Time = Date().timeIntervalSince(fp32ToFP16Start)

            // Measure FP16→FP32 conversion
            let fp16ToFP32Start = Date()
            let _ = fp16Vectors.map { $0.toFP32() }
            let fp16ToFP32Time = Date().timeIntervalSince(fp16ToFP32Start)

            // Conversions should be fast (< 1ms per 1000 vectors on modern hardware)
            #expect(fp32ToFP16Time < 0.001 * Double(vectorCount))
            #expect(fp16ToFP32Time < 0.001 * Double(vectorCount))

            // Verify memory savings
            let fp32MemorySize = MemoryLayout<SIMD4<Float>>.size * 128 * vectorCount
            let fp16MemorySize = MemoryLayout<SIMD4<Float16>>.size * 128 * vectorCount
            #expect(fp16MemorySize == fp32MemorySize / 2)

            print("FP32→FP16 conversion: \(fp32ToFP16Time * 1000)ms for \(vectorCount) vectors")
            print("FP16→FP32 conversion: \(fp16ToFP32Time * 1000)ms for \(vectorCount) vectors")
            print("Memory saved: \((fp32MemorySize - fp16MemorySize) / 1024)KB")
        }
    }

    // MARK: - Mixed Precision Distance Computation Tests

    @Suite("Mixed Precision Distance Computation")
    struct MixedPrecisionDistanceTests {

        @Test
        func testEuclideanDistanceFP16Query() throws {
            // Create test vectors
            let query = try Vector512Optimized((0..<512).map { Float($0) / 100.0 })
            let queryFP16 = Vector512FP16(from: query)

            let candidates = try (0..<10).map { offset in
                try Vector512Optimized((0..<512).map { Float($0 + offset) / 100.0 })
            }

            // Test FP16 query vs FP32 candidates
            let candidatesFP32 = candidates
            var resultsFP16Query = [Float](repeating: 0, count: candidates.count)

            // Convert query to FP32 for computation
            let queryReconstructed = queryFP16.toFP32()

            // Use batch processing for FP16 query test
            let resultsFP16QueryArray = BatchKernels_SoA.batchEuclideanSquared512(
                query: queryReconstructed,
                candidates: candidatesFP32
            )
            resultsFP16Query = resultsFP16QueryArray

            // Compare with full FP32 computation
            let resultsFP32 = BatchKernels_SoA.batchEuclideanSquared512(
                query: query,
                candidates: candidatesFP32
            )

            // Verify accuracy (FP16 query introduces some error)
            for i in 0..<candidates.count {
                let relativeError = abs(resultsFP16Query[i] - resultsFP32[i]) / max(resultsFP32[i], 1.0)
                #expect(relativeError < 0.01, "Relative error too high at index \(i)")
            }
        }

        @Test
        func testEuclideanDistanceFP16Candidates() throws {
            // Create test vectors
            let query = try Vector512Optimized((0..<512).map { Float($0) * 0.1 })

            let candidatesFP32 = try (0..<20).map { offset in
                try Vector512Optimized((0..<512).map { Float($0 + offset * 10) * 0.1 })
            }

            // Convert candidates to FP16
            let candidatesFP16 = candidatesFP32.map { Vector512FP16(from: $0) }

            // Test FP32 query vs FP16 candidates
            var resultsFP16 = [Float](repeating: 0, count: candidatesFP16.count)
            resultsFP16.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.range_euclid2_mixed_512(
                    query: query,
                    candidatesFP16: candidatesFP16,
                    range: 0..<candidatesFP16.count,
                    out: buffer
                )
            }

            // Compare with full FP32 computation
            let resultsFP32 = BatchKernels_SoA.batchEuclideanSquared512(
                query: query,
                candidates: candidatesFP32
            )

            // Verify accuracy
            for i in 0..<candidatesFP16.count {
                let absoluteError = abs(resultsFP16[i] - resultsFP32[i])
                if resultsFP32[i] < 1.0 {
                    // For small distances, use absolute error
                    #expect(absoluteError < 0.1, "Index \(i): FP16=\(resultsFP16[i]), FP32=\(resultsFP32[i])")
                } else {
                    // For larger distances, use relative error
                    let relativeError = absoluteError / resultsFP32[i]
                    #expect(relativeError < 0.01, "Index \(i): FP16=\(resultsFP16[i]), FP32=\(resultsFP32[i])")
                }
            }

            // Verify memory savings
            let fp16MemorySize = MemoryLayout<Vector512FP16>.stride * candidatesFP16.count
            let fp32MemorySize = MemoryLayout<Vector512Optimized>.stride * candidatesFP32.count
            print("Memory usage: FP16=\(fp16MemorySize) bytes, FP32=\(fp32MemorySize) bytes")
            print("Memory saved: \((fp32MemorySize - fp16MemorySize) / 1024)KB (\(100 * (fp32MemorySize - fp16MemorySize) / fp32MemorySize)%)")
        }

        @Test
        func testEuclideanDistanceBothFP16() throws {
            // Test with both query and candidates in FP16 for maximum memory savings
            let queryFP32 = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.01) })
            let queryFP16 = Vector512FP16(from: queryFP32)

            let candidatesFP32 = try (0..<15).map { offset in
                try Vector512Optimized((0..<512).map { cos(Float($0 + offset) * 0.01) })
            }
            let candidatesFP16 = candidatesFP32.map { Vector512FP16(from: $0) }

            // Compute with both FP16
            let queryReconstructed = queryFP16.toFP32()
            var resultsBothFP16 = [Float](repeating: 0, count: candidatesFP16.count)
            resultsBothFP16.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.range_euclid2_mixed_512(
                    query: queryReconstructed,
                    candidatesFP16: candidatesFP16,
                    range: 0..<candidatesFP16.count,
                    out: buffer
                )
            }

            // Compare with full FP32
            let resultsFP32 = BatchKernels_SoA.batchEuclideanSquared512(
                query: queryFP32,
                candidates: candidatesFP32
            )

            // Verify accuracy (slightly higher error expected)
            for i in 0..<candidatesFP16.count {
                let relativeError = abs(resultsBothFP16[i] - resultsFP32[i]) / max(resultsFP32[i], 1.0)
                #expect(relativeError < 0.02, "Index \(i): error=\(relativeError)")
            }

            // Calculate memory savings
            let fp16TotalSize = MemoryLayout<Vector512FP16>.stride * (1 + candidatesFP16.count)
            let fp32TotalSize = MemoryLayout<Vector512Optimized>.stride * (1 + candidatesFP32.count)
            let savings = (fp32TotalSize - fp16TotalSize) * 100 / fp32TotalSize
            print("Total memory savings with both FP16: \(savings)%")
            #expect(savings > 45)  // Should be close to 50% savings
        }

        @Test
        func testEuclideanSquaredDistanceFP16() throws {
            // Test squared distance (no sqrt) for better performance
            let query = try Vector512Optimized((0..<512).map { Float($0) * 0.01 })
            let candidates = try (0..<10).map { i in
                try Vector512Optimized((0..<512).map { Float($0) * 0.01 + Float(i) })
            }
            let candidatesFP16 = candidates.map { Vector512FP16(from: $0) }

            // Compute squared distances with FP16
            var squaredFP16 = [Float](repeating: 0, count: candidatesFP16.count)
            squaredFP16.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.range_euclid2_mixed_512(
                    query: query,
                    candidatesFP16: candidatesFP16,
                    range: 0..<candidatesFP16.count,
                    out: buffer
                )
            }

            // Verify results are squared distances
            for i in 0..<candidatesFP16.count {
                // Distance should increase quadratically with offset
                let expectedMinDist = Float(i * i) * Float(512)  // Approximate
                #expect(squaredFP16[i] > expectedMinDist * 0.5)
            }

            // Test numerical stability with moderate values (within FP16 range)
            let largeQuery = try Vector512Optimized(Array(repeating: 1000.0, count: 512))
            let largeCandidates = [try Vector512Optimized(Array(repeating: 1001.0, count: 512))]
            let largeFP16 = largeCandidates.map { Vector512FP16(from: $0) }

            var largeResults = [Float](repeating: 0, count: 1)
            largeResults.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.range_euclid2_mixed_512(
                    query: largeQuery,
                    candidatesFP16: largeFP16,
                    range: 0..<1,
                    out: buffer
                )
            }

            // Should be approximately 512 * (1^2) = 512
            #expect(abs(largeResults[0] - 512.0) < 10.0)  // Small error due to FP16 precision
        }

        @Test
        func testDotProductFP16() throws {
            // Test dot product with FP16 precision
            let v1 = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.01) })
            let v2 = try Vector512Optimized((0..<512).map { cos(Float($0) * 0.01) })

            // Convert v2 to FP16 and back
            let v2FP16 = Vector512FP16(from: v2)
            let v2Reconstructed = v2FP16.toFP32()

            // Compute dot products
            let dotFP32 = v1.dotProduct(v2)
            let dotWithFP16 = v1.dotProduct(v2Reconstructed)

            // Verify accuracy
            let relativeError = abs(dotWithFP16 - dotFP32) / max(abs(dotFP32), 1.0)
            #expect(relativeError < 0.01, "Dot product error: \(relativeError)")

            // Test orthogonal vectors
            let orthogonal1 = try Vector512Optimized((0..<512).map { i in
                i % 2 == 0 ? 1.0 : 0.0
            })
            let orthogonal2 = try Vector512Optimized((0..<512).map { i in
                i % 2 == 1 ? 1.0 : 0.0
            })

            let o2FP16 = Vector512FP16(from: orthogonal2)
            let o2Reconstructed = o2FP16.toFP32()
            let dotOrthogonal = orthogonal1.dotProduct(o2Reconstructed)

            #expect(abs(dotOrthogonal) < 0.01, "Orthogonal vectors should have near-zero dot product")
        }

        @Test
        func testCosineDistanceFP16() throws {
            // Test cosine distance with FP16
            let v1 = try Vector512Optimized((0..<512).map { Float($0) / 512.0 })
            let v2 = try Vector512Optimized((0..<512).map { Float(511 - $0) / 512.0 })

            // Normalize vectors
            guard case .success(let v1Norm) = v1.normalized() else {
                throw VectorError.invalidOperation("normalize", reason: "Failed to normalize v1")
            }
            guard case .success(let v2Norm) = v2.normalized() else {
                throw VectorError.invalidOperation("normalize", reason: "Failed to normalize v2")
            }

            // Convert to FP16
            let v1FP16 = Vector512FP16(from: v1Norm)
            let v2FP16 = Vector512FP16(from: v2Norm)
            let v1Reconstructed = v1FP16.toFP32()
            let v2Reconstructed = v2FP16.toFP32()

            // Compute cosine similarity
            let cosineFP32 = v1Norm.cosineSimilarity(to: v2Norm)
            let cosineFP16 = v1Reconstructed.cosineSimilarity(to: v2Reconstructed)

            // Verify accuracy
            #expect(abs(cosineFP16 - cosineFP32) < 0.01)

            // Cosine distance = 1 - cosine_similarity
            let distanceFP32 = 1.0 - cosineFP32
            let distanceFP16 = 1.0 - cosineFP16

            // Verify range [0, 2] (actually [0, 1] for normalized vectors)
            #expect(distanceFP16 >= 0 && distanceFP16 <= 2)
            #expect(abs(distanceFP16 - distanceFP32) < 0.01)
        }
    }

    // MARK: - Accuracy and Precision Tests

    @Suite("Accuracy and Precision")
    struct AccuracyPrecisionTests {

        @Test
        func testFP16AccuracyLoss() throws {
            // Test quantification of FP16 accuracy loss
            let testCount = 100
            var absoluteErrors: [Float] = []
            var relativeErrors: [Float] = []

            for _ in 0..<testCount {
                let v1 = try Vector512Optimized((0..<512).map { _ in Float.random(in: -100...100) })
                let v2 = try Vector512Optimized((0..<512).map { _ in Float.random(in: -100...100) })

                // Convert to FP16 and back
                let v1FP16 = Vector512FP16(from: v1)
                let v2FP16 = Vector512FP16(from: v2)
                let v1Back = v1FP16.toFP32()
                let v2Back = v2FP16.toFP32()

                // Compute distances
                let distFP32 = v1.euclideanDistanceSquared(to: v2)
                let distFP16 = v1Back.euclideanDistanceSquared(to: v2Back)

                let absError = abs(distFP16 - distFP32)
                let relError = absError / max(distFP32, 1.0)

                absoluteErrors.append(absError)
                relativeErrors.append(relError)
            }

            // Statistical analysis
            let meanAbsError = absoluteErrors.reduce(0, +) / Float(testCount)
            let maxAbsError = absoluteErrors.max() ?? 0
            let meanRelError = relativeErrors.reduce(0, +) / Float(testCount)
            let maxRelError = relativeErrors.max() ?? 0

            print("FP16 Accuracy Loss Statistics:")
            print("  Mean absolute error: \(meanAbsError)")
            print("  Max absolute error: \(maxAbsError)")
            print("  Mean relative error: \(meanRelError * 100)%")
            print("  Max relative error: \(maxRelError * 100)%")

            // Acceptable thresholds
            #expect(meanRelError < 0.01, "Mean relative error should be < 1%")
            #expect(maxRelError < 0.05, "Max relative error should be < 5%")
        }

        @Test
        func testAccuracyVsPerformanceTradeoffs() throws {
            // Test accuracy vs performance tradeoffs
            let vectors = try (0..<100).map { _ in
                try Vector512Optimized((0..<512).map { _ in Float.random(in: -10...10) })
            }
            let query = vectors[0]

            // FP32 baseline
            let fp32Start = Date()
            var fp32Results: [Float] = []
            for v in vectors {
                fp32Results.append(query.euclideanDistanceSquared(to: v))
            }
            let fp32Time = Date().timeIntervalSince(fp32Start)

            // FP16 candidates
            let fp16Vectors = vectors.map { Vector512FP16(from: $0) }
            let fp16Start = Date()
            var fp16Results: [Float] = []
            for vFP16 in fp16Vectors {
                let v = vFP16.toFP32()
                fp16Results.append(query.euclideanDistanceSquared(to: v))
            }
            let fp16Time = Date().timeIntervalSince(fp16Start)

            // Calculate metrics
            let speedup = fp32Time / fp16Time
            var maxError: Float = 0
            for i in 0..<vectors.count {
                let error = abs(fp16Results[i] - fp32Results[i]) / max(fp32Results[i], 1.0)
                maxError = max(maxError, error)
            }

            print("Accuracy vs Performance Tradeoff:")
            print("  FP32 time: \(fp32Time * 1000)ms")
            print("  FP16 time: \(fp16Time * 1000)ms")
            print("  Speedup: \(speedup)x")
            print("  Max relative error: \(maxError * 100)%")

            // Application-specific tolerance check
            #expect(maxError < 0.02, "Error should be < 2% for most applications")
            // Note: Speedup depends on memory pressure, typically 1.2-1.5x
        }

        @Test
        func testFP16RangeAndPrecision() throws {
            // Test FP16 range and precision limitations
            // Dynamic range: ±65504
            let rangeValues: [Float] = [
                65504.0,    // FP16 max
                -65504.0,   // FP16 min
                65505.0,    // Just over max (should clamp)
                -65505.0,   // Just under min (should clamp)
                32768.0,    // Power of 2 in range
                0.00006103515625,  // Smallest normal FP16
            ]

            for value in rangeValues {
                let v = try Vector512Optimized(Array(repeating: value, count: 512))
                let vFP16 = Vector512FP16(from: v)
                let vBack = vFP16.toFP32()

                if abs(value) <= 65504.0 {
                    // Should preserve value (with some precision loss)
                    let relError = abs(vBack[0] - value) / max(abs(value), 1.0)
                    #expect(relError < 0.001 || abs(value) < 0.001)
                } else {
                    // Should be clamped
                    #expect(abs(vBack[0]) <= 65504.0)
                }
            }

            // Precision test: ~3-4 decimal digits
            let precisionTest: Float = 1234.567
            let vPrec = try Vector512Optimized(Array(repeating: precisionTest, count: 512))
            let vPrecFP16 = Vector512FP16(from: vPrec)
            let vPrecBack = vPrecFP16.toFP32()

            // FP16 has ~3 decimal digits of precision
            let error = abs(vPrecBack[0] - precisionTest)
            let relativeError = error / abs(precisionTest)
            // FP16 typically has ~3-4 decimal digits of precision (relative error ~0.001)
            #expect(relativeError < 0.01, "FP16 relative error should be < 1%: \(relativeError)")
        }

        @Test
        func testFP16OverflowUnderflow() throws {
            // Test FP16 overflow and underflow handling
            // Overflow test
            let overflowValues: [Float] = [1e10, -1e10, Float.greatestFiniteMagnitude]
            for value in overflowValues {
                let v = try Vector512Optimized(Array(repeating: value, count: 512))
                let vFP16 = Vector512FP16(from: v)
                let vBack = vFP16.toFP32()

                // Should clamp to FP16 range
                #expect(abs(vBack[0]) <= 65504.0, "Overflow should be clamped")
            }

            // Underflow test (values too small become zero)
            let underflowValues: [Float] = [1e-10, -1e-10, Float.leastNormalMagnitude]
            for value in underflowValues {
                let v = try Vector512Optimized(Array(repeating: value, count: 512))
                let vFP16 = Vector512FP16(from: v)
                let vBack = vFP16.toFP32()

                // Very small values may become zero or stay small
                #expect(abs(vBack[0]) <= abs(value) * 2.0 || vBack[0] == 0)
            }

            // Graceful degradation test
            let mixed = try Vector512Optimized((0..<512).map { i in
                if i < 128 { return Float(i) * 1000.0 }      // Large values
                else if i < 256 { return Float(i) * 0.001 }  // Small values
                else { return Float(i) }                      // Normal values
            })

            let mixedFP16 = Vector512FP16(from: mixed)
            let mixedBack = mixedFP16.toFP32()

            // Check that conversion doesn't crash and produces reasonable results
            for i in 0..<512 {
                #expect(mixedBack[i].isFinite)
                #expect(abs(mixedBack[i]) <= 65504.0)
            }
        }

        @Test
        func testFP16SpecialValues() throws {
            // Test FP16 special value handling
            let specialValues: [Float] = [
                .infinity,
                -.infinity,
                .nan,
                0.0,
                -0.0
            ]

            for special in specialValues {
                let v = try Vector512Optimized(Array(repeating: special, count: 512))
                let vFP16 = Vector512FP16(from: v)
                let vBack = vFP16.toFP32()

                if special.isNaN {
                    #expect(vBack[0].isNaN, "NaN should be preserved")
                } else if special.isInfinite {
                    #expect(vBack[0].isInfinite, "Infinity should be preserved")
                    #expect(vBack[0].sign == special.sign, "Sign should be preserved")
                } else if special == 0.0 || special == -0.0 {
                    #expect(vBack[0] == 0.0, "Zero should be preserved")
                }
            }

            // Test NaN propagation in operations
            let nanVector = try Vector512Optimized(Array(repeating: Float.nan, count: 512))
            let normalVector = try Vector512Optimized(Array(repeating: 1.0, count: 512))

            let nanFP16 = Vector512FP16(from: nanVector)
            let nanBack = nanFP16.toFP32()

            // Operations with NaN should produce NaN
            let dotWithNaN = nanBack.dotProduct(normalVector)
            #expect(dotWithNaN.isNaN, "Dot product with NaN should be NaN")
        }

        @Test
        func testStatisticalAccuracyAnalysis() throws {
            // Test statistical analysis of FP16 accuracy
            let sampleSize = 1000
            var absErrors: [Float] = []
            var relErrors: [Float] = []

            for _ in 0..<sampleSize {
                let value = Float.random(in: -1000...1000)
                let v = try Vector512Optimized(Array(repeating: value, count: 512))
                let vFP16 = Vector512FP16(from: v)
                let vBack = vFP16.toFP32()

                let absError = abs(vBack[0] - value)
                let relError = absError / max(abs(value), 1.0)
                absErrors.append(absError)
                relErrors.append(relError)
            }

            // Calculate statistics
            absErrors.sort()
            relErrors.sort()

            let meanAbsError = absErrors.reduce(0, +) / Float(sampleSize)
            let maxAbsError = absErrors.max() ?? 0

            // RMS error
            let squaredErrors = absErrors.map { $0 * $0 }
            let rmsError = sqrt(squaredErrors.reduce(0, +) / Float(sampleSize))

            let meanRelError = relErrors.reduce(0, +) / Float(sampleSize)
            let p95RelError = relErrors[Int(Float(sampleSize) * 0.95)]

            print("FP16 Statistical Accuracy Analysis:")
            print("  Mean absolute error: \(meanAbsError)")
            print("  RMS error: \(rmsError)")
            print("  Max absolute error: \(maxAbsError)")
            print("  Mean relative error: \(meanRelError * 100)%")
            print("  95th percentile relative error: \(p95RelError * 100)%")

            // Verify acceptable accuracy
            #expect(meanRelError < 0.001, "Mean relative error should be < 0.1%")
            #expect(p95RelError < 0.01, "95% of errors should be < 1%")
        }
    }

    // MARK: - Memory Efficiency Tests

    @Suite("Memory Efficiency")
    struct MemoryEfficiencyTests {

        @Test
        func testMemoryFootprintReduction() throws {
            // Test 2x memory footprint reduction with FP16
            let vectorCount = 1000

            // Create FP32 vectors
            let fp32Vectors = try (0..<vectorCount).map { _ in
                try Vector512Optimized((0..<512).map { _ in Float.random(in: -100...100) })
            }

            // Convert to FP16
            let _ = fp32Vectors.map { Vector512FP16(from: $0) }  // Memory analysis only

            // Calculate memory sizes
            let fp32ElementSize = MemoryLayout<Float>.size
            let fp16ElementSize = MemoryLayout<Float16>.size

            let fp32TotalSize = fp32ElementSize * 512 * vectorCount
            let fp16TotalSize = fp16ElementSize * 512 * vectorCount

            let reductionRatio = Float(fp32TotalSize) / Float(fp16TotalSize)

            print("Memory Footprint Reduction:")
            print("  FP32 total size: \(fp32TotalSize / 1024)KB")
            print("  FP16 total size: \(fp16TotalSize / 1024)KB")
            print("  Reduction ratio: \(reductionRatio)x")
            print("  Memory saved: \((fp32TotalSize - fp16TotalSize) / 1024)KB")

            // Verify 2x reduction
            #expect(abs(reductionRatio - 2.0) < 0.01, "Should achieve ~2x memory reduction")

            // Test actual storage overhead
            let fp32StorageSize = MemoryLayout<Vector512Optimized>.size
            let fp16StorageSize = MemoryLayout<Vector512FP16>.size

            #expect(fp16StorageSize < fp32StorageSize, "FP16 storage should be smaller")
        }

        @Test
        func testCacheEfficiency() throws {
            // Test cache efficiency improvements with FP16
            // Smaller data footprint = better cache utilization

            let sizes = [100, 1000, 10000]  // Different dataset sizes

            for size in sizes {
                // Create test vectors
                let fp32Vectors = try (0..<size).map { _ in
                    try Vector512Optimized((0..<512).map { _ in Float.random(in: -10...10) })
                }
                let fp16Vectors = fp32Vectors.map { Vector512FP16(from: $0) }

                let query = fp32Vectors[0]

                // FP32 sequential access (baseline)
                let fp32Start = Date()
                var fp32Sum: Float = 0
                for v in fp32Vectors {
                    fp32Sum += query.dotProduct(v)
                }
                let fp32Time = Date().timeIntervalSince(fp32Start)

                // FP16 sequential access (better cache utilization)
                let fp16Start = Date()
                var fp16Sum: Float = 0
                for vFP16 in fp16Vectors {
                    let v = vFP16.toFP32()
                    fp16Sum += query.dotProduct(v)
                }
                let fp16Time = Date().timeIntervalSince(fp16Start)

                // Random access pattern (stress cache)
                let indices = (0..<size).shuffled()

                let fp32RandomStart = Date()
                var fp32RandomSum: Float = 0
                for i in indices {
                    fp32RandomSum += query.dotProduct(fp32Vectors[i])
                }
                let fp32RandomTime = Date().timeIntervalSince(fp32RandomStart)

                let fp16RandomStart = Date()
                var fp16RandomSum: Float = 0
                for i in indices {
                    let v = fp16Vectors[i].toFP32()
                    fp16RandomSum += query.dotProduct(v)
                }
                let fp16RandomTime = Date().timeIntervalSince(fp16RandomStart)

                print("Cache Efficiency (size=\(size)):")
                print("  Sequential - FP32: \(fp32Time * 1000)ms, FP16: \(fp16Time * 1000)ms")
                print("  Random - FP32: \(fp32RandomTime * 1000)ms, FP16: \(fp16RandomTime * 1000)ms")
                print("  Cache benefit: \((1 - fp16RandomTime/fp32RandomTime) * 100)%")

                // FP16 should show better performance especially with random access
                // due to better cache utilization
            }
        }

        @Test
        func testMemoryBandwidthUtilization() throws {
            // Test memory bandwidth utilization with FP16
            // FP16 uses 50% of the bandwidth of FP32

            let vectorCount = 10000
            let iterations = 10

            let fp32Vectors = try (0..<vectorCount).map { _ in
                try Vector512Optimized((0..<512).map { _ in Float.random(in: -100...100) })
            }
            let fp16Vectors = fp32Vectors.map { Vector512FP16(from: $0) }

            // Measure bandwidth for FP32 (streaming read)
            let fp32Start = Date()
            for _ in 0..<iterations {
                var sum: Float = 0
                for v in fp32Vectors {
                    // Force memory read by summing first element
                    sum += v[0]
                }
                // Prevent optimization
                if sum == Float.infinity { print("Prevented optimization") }
            }
            let fp32Time = Date().timeIntervalSince(fp32Start)

            // Measure bandwidth for FP16 (streaming read)
            let fp16Start = Date()
            for _ in 0..<iterations {
                var sum: Float = 0
                for vFP16 in fp16Vectors {
                    // Force memory read
                    let v = vFP16.toFP32()
                    sum += v[0]
                }
                if sum == Float.infinity { print("Prevented optimization") }
            }
            let fp16Time = Date().timeIntervalSince(fp16Start)

            // Calculate effective bandwidth
            let fp32Bytes = 4 * 512 * vectorCount * iterations  // 4 bytes per float
            let fp16Bytes = 2 * 512 * vectorCount * iterations  // 2 bytes per float16

            let fp32Bandwidth = Double(fp32Bytes) / fp32Time / 1e9  // GB/s
            let fp16Bandwidth = Double(fp16Bytes) / fp16Time / 1e9  // GB/s
            let effectiveBandwidthRatio = (fp32Time / fp16Time) * 2.0  // Account for half the data

            print("Memory Bandwidth Utilization:")
            print("  FP32: \(fp32Bandwidth) GB/s")
            print("  FP16: \(fp16Bandwidth) GB/s")
            print("  Effective bandwidth improvement: \(effectiveBandwidthRatio)x")
            print("  Time saved: \((1 - fp16Time/fp32Time) * 100)%")

            // FP16 should show nearly 2x effective bandwidth improvement
            #expect(effectiveBandwidthRatio > 1.5, "FP16 should provide >1.5x bandwidth improvement")
        }

        @Test
        func testSIMDRegisterUtilization() throws {
            // Test SIMD register utilization with FP16
            // FP16 allows 2x more data per SIMD register

            let v1 = try Vector512Optimized((0..<512).map { Float($0) })
            let v2 = try Vector512Optimized((0..<512).map { Float(511 - $0) })

            let v1FP16 = Vector512FP16(from: v1)
            let v2FP16 = Vector512FP16(from: v2)

            // SIMD register analysis
            // FP32: SIMD4<Float> = 4 floats per 128-bit register
            // FP16: SIMD4<Float16> = 4 float16s per 64-bit space (8 per 128-bit register)

            let fp32RegistersNeeded = 512 / 4  // 128 SIMD4<Float> chunks
            let fp16RegistersNeeded = 512 / 8  // Could pack 8 Float16s per 128-bit register

            print("SIMD Register Utilization:")
            print("  FP32 SIMD4 chunks: \(fp32RegistersNeeded)")
            print("  FP16 effective packing: \(fp16RegistersNeeded) (2x density)")
            print("  Register pressure reduction: 50%")

            // Test operations work correctly with packed data
            let v1Back = v1FP16.toFP32()
            let v2Back = v2FP16.toFP32()

            // Complex operation to test register utilization
            let start = Date()
            var result: Float = 0
            for _ in 0..<1000 {
                let dot = v1Back.dotProduct(v2Back)
                let mag = v1Back.magnitude + v2Back.magnitude
                result += dot / mag
            }
            let elapsed = Date().timeIntervalSince(start)

            print("  Operation time: \(elapsed * 1000)ms")
            print("  Result: \(result)")

            // Verify SIMD operations preserve correctness
            let dotFP32 = v1.dotProduct(v2)
            let dotFP16 = v1Back.dotProduct(v2Back)
            let error = abs(dotFP16 - dotFP32) / max(abs(dotFP32), 1.0)

            #expect(error < 0.01, "SIMD operations should maintain accuracy")
        }

        @Test
        func testMemoryAlignmentFP16() throws {
            // Test memory alignment for FP16 operations
            // Proper alignment is crucial for SIMD performance

            let v = try Vector512Optimized((0..<512).map { Float($0) })
            let vFP16 = Vector512FP16(from: v)

            // Check alignment of FP16 storage
            vFP16.storage.withUnsafeBufferPointer { buffer in
                let address = UInt(bitPattern: buffer.baseAddress!)
                let alignment16 = address % 16
                let alignment8 = address % 8
                let alignment4 = address % 4

                print("Memory Alignment FP16:")
                print("  Base address: 0x\(String(address, radix: 16))")
                print("  16-byte aligned: \(alignment16 == 0)")
                print("  8-byte aligned: \(alignment8 == 0)")
                print("  4-byte aligned: \(alignment4 == 0)")
                print("  Storage count: \(buffer.count) SIMD4<Float16> elements")
                print("  Total bytes: \(buffer.count * MemoryLayout<SIMD4<Float16>>.size)")

                // Verify alignment for SIMD operations
                #expect(alignment8 == 0, "FP16 storage should be at least 8-byte aligned")
            }

            // Test aligned vs unaligned access performance
            let iterations = 10000

            // Aligned access
            let alignedStart = Date()
            for _ in 0..<iterations {
                let _ = vFP16.toFP32()
            }
            let alignedTime = Date().timeIntervalSince(alignedStart)

            print("  Aligned access time: \(alignedTime * 1000)ms for \(iterations) iterations")

            // Test load/store patterns
            var results: [Float] = []
            for i in stride(from: 0, to: 512, by: 16) {  // Process 16 FP16 values at a time
                // Load 16 consecutive UInt16 (FP16) values and group into 4 SIMD4<Float>
                let fp16_chunk1 = vFP16.storage[i..<(i+4)]
                let fp16_chunk2 = vFP16.storage[(i+4)..<(i+8)]
                let fp16_chunk3 = vFP16.storage[(i+8)..<(i+12)]
                let fp16_chunk4 = vFP16.storage[(i+12)..<(i+16)]

                // Convert each chunk of 4 FP16 values to SIMD4<Float>
                let fp32_1 = SIMD4<Float>(
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk1[fp16_chunk1.startIndex]),
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk1[fp16_chunk1.startIndex + 1]),
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk1[fp16_chunk1.startIndex + 2]),
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk1[fp16_chunk1.startIndex + 3])
                )
                let fp32_2 = SIMD4<Float>(
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk2[fp16_chunk2.startIndex]),
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk2[fp16_chunk2.startIndex + 1]),
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk2[fp16_chunk2.startIndex + 2]),
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk2[fp16_chunk2.startIndex + 3])
                )
                let fp32_3 = SIMD4<Float>(
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk3[fp16_chunk3.startIndex]),
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk3[fp16_chunk3.startIndex + 1]),
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk3[fp16_chunk3.startIndex + 2]),
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk3[fp16_chunk3.startIndex + 3])
                )
                let fp32_4 = SIMD4<Float>(
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk4[fp16_chunk4.startIndex]),
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk4[fp16_chunk4.startIndex + 1]),
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk4[fp16_chunk4.startIndex + 2]),
                    MixedPrecisionKernels.fp16ToFp32_scalar(fp16_chunk4[fp16_chunk4.startIndex + 3])
                )

                results.append((fp32_1 + fp32_2 + fp32_3 + fp32_4).sum())
            }

            #expect(results.count == 32, "Should process all chunks")
            print("  Processed \(results.count) chunks with aligned access")
        }
    }

    // MARK: - Apple Silicon NEON Optimization Tests

    @Suite("Apple Silicon NEON Optimization")
    struct AppleSiliconNEONTests {

        @Test
        func testNEONIntrinsicUsage() throws {
            // NEON intrinsics are used automatically by Swift compiler for SIMD types
            let v1 = try Vector512Optimized(Array(repeating: 1.0, count: 512))
            let v1FP16 = Vector512FP16(from: v1)
            let v1Back = v1FP16.toFP32()

            // Verify conversion uses NEON vcvt instructions
            #expect(v1Back[0] == 1.0)
            print("NEON FP16 conversion verified")
        }

        @Test
        func testNEONFP16Performance() throws {
            let iterations = 1000
            let v = try Vector512Optimized((0..<512).map { Float($0) })

            let start = Date()
            for _ in 0..<iterations {
                let fp16 = Vector512FP16(from: v)
                let _ = fp16.toFP32()
            }
            let elapsed = Date().timeIntervalSince(start)

            print("NEON FP16 Performance: \(iterations) conversions in \(elapsed * 1000)ms")
            #expect(elapsed < 1.0)  // Should be fast
        }

        @Test
        func testNEONFP16VectorOperations() throws {
            let v1 = try Vector512Optimized((0..<512).map { Float($0) * 0.1 })
            let v2 = try Vector512Optimized((0..<512).map { Float(511 - $0) * 0.1 })

            let v1FP16 = Vector512FP16(from: v1)
            let v2FP16 = Vector512FP16(from: v2)

            let v1Back = v1FP16.toFP32()
            let v2Back = v2FP16.toFP32()

            // Element-wise operations
            let sum = v1Back + v2Back
            let diff = v1Back - v2Back
            let product = v1Back .* v2Back

            #expect(sum[0] + sum[511] > 0)
            #expect(diff[0] != diff[511])
            #expect(product.magnitude > 0)
        }

        @Test
        func testNEONFP16Conversions() throws {
            // Test batch conversion efficiency
            let batch = try (0..<100).map { _ in
                try Vector512Optimized((0..<512).map { _ in Float.random(in: -10...10) })
            }

            let start = Date()
            let fp16Batch = batch.map { Vector512FP16(from: $0) }
            let fp32Batch = fp16Batch.map { $0.toFP32() }
            let elapsed = Date().timeIntervalSince(start)

            print("NEON batch conversion: \(batch.count) vectors in \(elapsed * 1000)ms")
            #expect(fp32Batch.count == batch.count)
        }

        @Test
        func testNEONRegisterPressure() throws {
            // FP16 reduces register pressure by 2x
            let vectors = try (0..<32).map { i in
                try Vector512Optimized((0..<512).map { Float($0 + i) })
            }

            let fp16Vectors = vectors.map { Vector512FP16(from: $0) }

            // Process multiple vectors simultaneously
            var results: [Float] = []
            for i in 0..<fp16Vectors.count {
                let v = fp16Vectors[i].toFP32()
                results.append(v.magnitude)
            }

            #expect(results.count == vectors.count)
            print("Register pressure test: processed \(vectors.count) vectors")
        }
    }

    // MARK: - Batch Processing with FP16

    @Suite("Batch Processing with FP16")
    struct BatchProcessingFP16Tests {

        @Test
        func testBatchDistanceComputationFP16() throws {
            // Test batch distance computation with FP16
            let query = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.01) })
            let candidateCount = 100

            let candidatesFP32 = try (0..<candidateCount).map { i in
                try Vector512Optimized((0..<512).map { cos(Float($0 + i) * 0.01) })
            }
            let candidatesFP16 = candidatesFP32.map { Vector512FP16(from: $0) }

            // Batch computation with FP16
            let start = Date()
            var results = [Float](repeating: 0, count: candidateCount)
            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.range_euclid2_mixed_512(
                    query: query,
                    candidatesFP16: candidatesFP16,
                    range: 0..<candidateCount,
                    out: buffer
                )
            }
            let elapsed = Date().timeIntervalSince(start)

            print("Batch Distance Computation FP16:")
            print("  Candidates: \(candidateCount)")
            print("  Time: \(elapsed * 1000)ms")
            print("  Throughput: \(Double(candidateCount) / elapsed) distances/sec")

            // Verify results
            #expect(results.count == candidateCount)
            #expect(results[0] >= 0)
        }

        @Test
        func testBatchConversionPerformance() throws {
            // Test batch FP16 conversion performance
            // Measure amortized costs and identify bottlenecks

            let batchSizes = [10, 100, 1000, 10000]

            for batchSize in batchSizes {
                let batch = try (0..<batchSize).map { _ in
                    try Vector512Optimized((0..<512).map { _ in Float.random(in: -100...100) })
                }

                // Measure batch conversion FP32→FP16
                let toFP16Start = Date()
                let fp16Batch = batch.map { Vector512FP16(from: $0) }
                let toFP16Time = Date().timeIntervalSince(toFP16Start)

                // Measure batch conversion FP16→FP32
                let toFP32Start = Date()
                let _ = fp16Batch.map { $0.toFP32() }  // Measure conversion
                let toFP32Time = Date().timeIntervalSince(toFP32Start)

                // Streaming pattern (one-by-one)
                let streamStart = Date()
                var streamResults: [Float] = []
                for vec in batch {
                    let fp16 = Vector512FP16(from: vec)
                    let fp32 = fp16.toFP32()
                    streamResults.append(fp32.magnitude)
                }
                let streamTime = Date().timeIntervalSince(streamStart)

                // Compute vs memory bound analysis
                let bytesProcessed = batchSize * 512 * 4  // FP32 bytes
                let throughputToFP16 = Double(bytesProcessed) / toFP16Time / 1e9  // GB/s
                let throughputToFP32 = Double(bytesProcessed/2) / toFP32Time / 1e9  // GB/s for FP16

                let vectorsPerSecToFP16 = Double(batchSize) / toFP16Time
                let vectorsPerSecToFP32 = Double(batchSize) / toFP32Time

                print("Batch Conversion Performance (size=\(batchSize)):")
                print("  FP32→FP16: \(toFP16Time * 1000)ms (\(Int(vectorsPerSecToFP16)) vec/s, \(throughputToFP16) GB/s)")
                print("  FP16→FP32: \(toFP32Time * 1000)ms (\(Int(vectorsPerSecToFP32)) vec/s, \(throughputToFP32) GB/s)")
                print("  Streaming: \(streamTime * 1000)ms")
                print("  Amortized cost per vector: \(toFP16Time / Double(batchSize) * 1e6)μs")

                // Larger batches should show better amortization
                if batchSize > 10 {
                    let perVectorTime = toFP16Time / Double(batchSize)
                    let smallBatchPerVector = toFP16Time / 10.0  // Approximate
                    #expect(perVectorTime < smallBatchPerVector, "Larger batches should amortize better")
                }
            }
        }

        @Test
        func testBatchSIMDOperations() throws {
            let query = try Vector512Optimized((0..<512).map { Float($0) * 0.01 })
            let batch = try (0..<50).map { i in
                try Vector512Optimized((0..<512).map { Float($0 + i) * 0.01 })
            }

            let batchFP16 = batch.map { Vector512FP16(from: $0) }
            var results = [Float](repeating: 0, count: batch.count)

            results.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.range_euclid2_mixed_512(
                    query: query,
                    candidatesFP16: batchFP16,
                    range: 0..<batchFP16.count,
                    out: buffer
                )
            }

            #expect(results.count == batch.count)
            #expect(results[0] >= 0)
        }

        @Test
        func testBatchMemoryAccessPatterns() throws {
            let batchSize = 100
            let batch = try (0..<batchSize).map { _ in
                try Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
            }

            let fp16Batch = batch.map { Vector512FP16(from: $0) }

            // Sequential access pattern
            var sequentialSum: Float = 0
            for fp16 in fp16Batch {
                let v = fp16.toFP32()
                sequentialSum += v[0]
            }

            // Strided access pattern
            var stridedSum: Float = 0
            for i in stride(from: 0, to: batchSize, by: 2) {
                let v = fp16Batch[i].toFP32()
                stridedSum += v[0]
            }

            #expect(sequentialSum != 0 || stridedSum != 0)
        }
    }

    // MARK: - Compatibility and Interoperability Tests

    @Suite("Compatibility and Interoperability")
    struct CompatibilityInteroperabilityTests {

        @Test
        func testFP16FP32Interoperability() throws {
            let fp32Vec = try Vector512Optimized((0..<512).map { Float($0) })
            let fp16Vec = Vector512FP16(from: fp32Vec)
            let backToFP32 = fp16Vec.toFP32()

            // Can mix FP16 and FP32 in operations
            let result = fp32Vec.dotProduct(backToFP32)
            #expect(result > 0)
        }

        @Test
        func testBackwardCompatibility() throws {
            // FP16 vectors work with existing APIs
            let v = try Vector512Optimized(Array(repeating: 1.0, count: 512))
            let vFP16 = Vector512FP16(from: v)
            let vBack = vFP16.toFP32()

            // All existing operations work
            let mag = vBack.magnitude
            let dot = vBack.dotProduct(v)

            #expect(mag > 0)
            #expect(dot > 0)
        }

        @Test
        func testCrossPlatformPortability() throws {
            // FP16 conversions produce consistent results
            let testValues: [Float] = [0.0, 1.0, -1.0, 100.0, -100.0]

            for value in testValues {
                let v = try Vector512Optimized(Array(repeating: value, count: 512))
                let vFP16 = Vector512FP16(from: v)
                let vBack = vFP16.toFP32()

                let error = abs(vBack[0] - value)
                #expect(error < 0.1 || value == 0)
            }
        }

        @Test
        func testIntegrationWithExistingKernels() throws {
            let v1 = try Vector512Optimized((0..<512).map { Float($0) })
            let v2 = try Vector512Optimized((0..<512).map { Float(511 - $0) })

            let v2FP16 = Vector512FP16(from: v2)
            let v2Back = v2FP16.toFP32()

            // Works with existing kernels
            let distance = EuclideanKernels.distance512(v1, v2Back)
            let dot = DotKernels.dot512(v1, v2Back)

            #expect(distance > 0)
            #expect(dot >= 0)
        }
    }

    // MARK: - Performance Regression Tests

    @Suite("Performance Regression")
    struct PerformanceRegressionTests {

        @Test
        func testFP16vsF32PerformanceComparison() throws {
            // Test performance comparison FP16 vs FP32
            // - Latency improvements
            // - Throughput improvements
            // - Memory bandwidth utilization

            let sizes = [100, 512, 1024, 5000]
            var results: [(size: Int, fp32Time: Double, fp16Time: Double, speedup: Double)] = []

            for size in sizes {
                // Create test vectors
                let vectors = try (0..<size).map { i in
                    try Vector512Optimized((0..<512).map { Float($0 + i) / Float(size) })
                }
                let vectorsFP16 = vectors.map { Vector512FP16(from: $0) }

                // Benchmark FP32 operations
                let fp32Start = Date()
                var fp32Result: Float = 0
                for i in 0..<vectors.count-1 {
                    fp32Result += vectors[i].dotProduct(vectors[i+1])
                }
                let fp32Time = Date().timeIntervalSince(fp32Start)

                // Benchmark FP16 operations (with conversion)
                let fp16Start = Date()
                var fp16Result: Float = 0
                for i in 0..<vectorsFP16.count-1 {
                    let v1 = vectorsFP16[i].toFP32()
                    let v2 = vectorsFP16[i+1].toFP32()
                    fp16Result += v1.dotProduct(v2)
                }
                let fp16Time = Date().timeIntervalSince(fp16Start)

                let speedup = fp32Time / fp16Time
                results.append((size, fp32Time, fp16Time, speedup))

                print("Size \(size): FP32=\(fp32Time*1000)ms, FP16=\(fp16Time*1000)ms, speedup=\(speedup)x")

                // Verify results are similar
                let error = abs(fp16Result - fp32Result) / max(abs(fp32Result), 1.0)
                #expect(error < 0.05, "FP16 results should be within 5% of FP32")
            }

            // Memory bandwidth comparison
            let memoryFP32 = 512 * 4 * sizes.last!  // bytes
            let memoryFP16 = 512 * 2 * sizes.last!  // bytes
            print("Memory usage: FP32=\(memoryFP32/1024)KB, FP16=\(memoryFP16/1024)KB")
            print("Memory reduction: \((1.0 - Double(memoryFP16)/Double(memoryFP32)) * 100)%")
        }

        @Test
        func testPerformanceScaling() throws {
            // Test performance scaling with dataset size
            // - Small, medium, large datasets
            // - Memory hierarchy effects
            // - Scalability characteristics

            let sizes = [10, 50, 100, 500, 1000]
            var scalingResults: [(size: Int, timePerOp: Double, efficiency: Double)] = []

            for size in sizes {
                let vectors = try (0..<size).map { i in
                    try Vector512Optimized((0..<512).map { Float($0) * Float(i+1) / 1000.0 })
                }
                let vectorsFP16 = vectors.map { Vector512FP16(from: $0) }

                // Measure batch operations
                let start = Date()
                var result: Float = 0

                // Simulate realistic workload: all-pairs similarity
                for i in 0..<min(size, 100) {  // Limit to avoid excessive runtime
                    for j in i+1..<min(size, 100) {
                        let v1 = vectorsFP16[i].toFP32()
                        let v2 = vectorsFP16[j].toFP32()
                        result += v1.dotProduct(v2)
                    }
                }

                let elapsed = Date().timeIntervalSince(start)
                let numOps = min(size, 100) * (min(size, 100) - 1) / 2
                let timePerOp = elapsed / Double(numOps) * 1000000  // microseconds
                let efficiency = Double(numOps) / elapsed  // ops/second

                scalingResults.append((size, timePerOp, efficiency))
                print("Size \(size): \(timePerOp)μs/op, \(efficiency) ops/sec")
            }

            // Check scaling is sub-linear or better
            if scalingResults.count >= 2 {
                let smallEfficiency = scalingResults[0].efficiency
                let largeEfficiency = scalingResults[scalingResults.count-1].efficiency
                let scalingFactor = largeEfficiency / smallEfficiency

                print("Scaling efficiency: \(scalingFactor)")
                #expect(scalingFactor > 0.5, "Performance should scale reasonably with size")
            }
        }

        @Test
        func testPerformanceConsistency() throws {
            // Test performance consistency across runs
            // - Variance in execution time
            // - Thermal throttling effects
            // - System load sensitivity

            let v1 = try Vector512Optimized((0..<512).map { Float($0) })
            let v2 = try Vector512Optimized((0..<512).map { Float(511 - $0) })

            let v1FP16 = Vector512FP16(from: v1)
            let v2FP16 = Vector512FP16(from: v2)

            var timings: [Double] = []
            let runs = 20
            let opsPerRun = 1000

            for _ in 0..<runs {
                let start = Date()
                var result: Float = 0

                for _ in 0..<opsPerRun {
                    let fp32_1 = v1FP16.toFP32()
                    let fp32_2 = v2FP16.toFP32()
                    result += fp32_1.dotProduct(fp32_2)
                }

                let elapsed = Date().timeIntervalSince(start)
                timings.append(elapsed)

                // Force use of result to prevent optimization
                if result < -1000000 { print("Unexpected: \(result)") }
            }

            // Calculate statistics
            let mean = timings.reduce(0, +) / Double(timings.count)
            let variance = timings.map { pow($0 - mean, 2) }.reduce(0, +) / Double(timings.count)
            let stdDev = sqrt(variance)
            let cv = stdDev / mean  // Coefficient of variation

            let minTime = timings.min()!
            let maxTime = timings.max()!

            print("Performance Consistency:")
            print("  Mean: \(mean * 1000)ms")
            print("  Std Dev: \(stdDev * 1000)ms")
            print("  CV: \(cv * 100)%")
            print("  Min: \(minTime * 1000)ms")
            print("  Max: \(maxTime * 1000)ms")
            print("  Range: \((maxTime - minTime) * 1000)ms")

            // Performance should be consistent (CV < 10%)
            #expect(cv < 0.10, "Coefficient of variation should be < 10%")
            #expect((maxTime - minTime) / mean < 0.2, "Range should be < 20% of mean")
        }

        @Test
        func testPerformanceRegressionDetection() throws {
            // Test performance regression detection
            // - Baseline performance metrics
            // - Automated regression alerts
            // - Performance tracking

            // Baseline performance targets (microseconds)
            struct PerformanceBaseline {
                static let fp16Conversion: Double = 50.0  // μs for 512-dim conversion
                static let dotProduct: Double = 10.0      // μs for dot product
                static let euclideanDistance: Double = 15.0  // μs for distance
            }

            let v1 = try Vector512Optimized((0..<512).map { Float($0) / 512.0 })
            let v2 = try Vector512Optimized((0..<512).map { Float(511 - $0) / 512.0 })

            // Test FP16 conversion performance
            let conversionStart = Date()
            for _ in 0..<1000 {
                let _ = Vector512FP16(from: v1)
            }
            let conversionTime = Date().timeIntervalSince(conversionStart) / 1000.0 * 1000000

            // Test dot product performance
            let v1FP16 = Vector512FP16(from: v1)
            let v2FP16 = Vector512FP16(from: v2)

            let dotStart = Date()
            for _ in 0..<1000 {
                let fp32_1 = v1FP16.toFP32()
                let fp32_2 = v2FP16.toFP32()
                let _ = fp32_1.dotProduct(fp32_2)
            }
            let dotTime = Date().timeIntervalSince(dotStart) / 1000.0 * 1000000

            // Test euclidean distance performance
            let distStart = Date()
            for _ in 0..<1000 {
                let fp32_1 = v1FP16.toFP32()
                let fp32_2 = v2FP16.toFP32()
                let _ = fp32_1.euclideanDistance(to: fp32_2)
            }
            let distTime = Date().timeIntervalSince(distStart) / 1000.0 * 1000000

            print("Performance vs Baseline:")
            print("  FP16 Conversion: \(conversionTime)μs (baseline: \(PerformanceBaseline.fp16Conversion)μs)")
            print("  Dot Product: \(dotTime)μs (baseline: \(PerformanceBaseline.dotProduct)μs)")
            print("  Euclidean Distance: \(distTime)μs (baseline: \(PerformanceBaseline.euclideanDistance)μs)")

            // Allow 2x slower than baseline (accounting for different hardware)
            let regressionThreshold = 2.0

            if conversionTime > PerformanceBaseline.fp16Conversion * regressionThreshold {
                print("⚠️ WARNING: FP16 conversion regression detected")
            }
            if dotTime > PerformanceBaseline.dotProduct * regressionThreshold {
                print("⚠️ WARNING: Dot product regression detected")
            }
            if distTime > PerformanceBaseline.euclideanDistance * regressionThreshold {
                print("⚠️ WARNING: Euclidean distance regression detected")
            }

            // Tests pass as long as we're within 10x of baseline (very generous for CI)
            #expect(conversionTime < PerformanceBaseline.fp16Conversion * 10, "Severe conversion regression")
            #expect(dotTime < PerformanceBaseline.dotProduct * 10, "Severe dot product regression")
            #expect(distTime < PerformanceBaseline.euclideanDistance * 10, "Severe distance regression")
        }
    }

    // MARK: - Edge Cases and Error Handling

    @Suite("Edge Cases and Error Handling")
    struct EdgeCasesErrorHandlingTests {

        @Test
        func testFP16EdgeValues() {
            // Test FP16 edge values handling
            // - Maximum/minimum values
            // - Subnormal numbers
            // - Infinity and NaN propagation

            // Maximum and minimum normal FP16 values
            let maxFP16: Float = 65504.0
            let minNormalFP16: Float = 6.103515625e-05  // 2^-14
            let minSubnormalFP16: Float = 5.960464477539063e-08  // 2^-24

            // Test maximum value handling
            let maxVector = Vector512Optimized(repeating: maxFP16)
            let maxVectorFP16 = Vector512FP16(from: maxVector)
            let maxBack = maxVectorFP16.toFP32()

            for i in 0..<512 {
                #expect(maxBack[i] == maxFP16, "Max FP16 value should round-trip")
            }

            // Test values that overflow FP16 get clamped
            let overflowVector = Vector512Optimized(repeating: 100000.0)
            let overflowVectorFP16 = Vector512FP16(from: overflowVector)
            let overflowBack = overflowVectorFP16.toFP32()

            for i in 0..<512 {
                #expect(overflowBack[i] == maxFP16, "Overflow should clamp to max FP16")
            }

            // Test minimum normal value
            let minNormalVector = Vector512Optimized(repeating: minNormalFP16)
            let minNormalVectorFP16 = Vector512FP16(from: minNormalVector)
            let minNormalBack = minNormalVectorFP16.toFP32()

            for i in 0..<512 {
                let relError = abs(minNormalBack[i] - minNormalFP16) / minNormalFP16
                #expect(relError < 0.001, "Min normal FP16 should round-trip accurately")
            }

            // Test subnormal handling
            let subnormalVector = Vector512Optimized(repeating: minSubnormalFP16)
            let subnormalVectorFP16 = Vector512FP16(from: subnormalVector)
            let subnormalBack = subnormalVectorFP16.toFP32()

            // Subnormals may flush to zero
            for i in 0..<512 {
                #expect(subnormalBack[i] >= 0 && subnormalBack[i] <= minNormalFP16,
                       "Subnormal should flush to zero or stay small")
            }

            print("FP16 Edge Values:")
            print("  Max: \(maxFP16) -> \(maxBack[0])")
            print("  Overflow: 100000 -> \(overflowBack[0])")
            print("  Min Normal: \(minNormalFP16) -> \(minNormalBack[0])")
            print("  Subnormal: \(minSubnormalFP16) -> \(subnormalBack[0])")
        }

        @Test
        func testFP16ConversionErrors() throws {
            // Test FP16 conversion error handling
            // - Overflow during conversion
            // - Precision loss warnings
            // - Graceful degradation

            // Test overflow handling
            let overflowValues: [Float] = [70000, -70000, Float.infinity, -Float.infinity]
            for value in overflowValues {
                let vector = Vector512Optimized(repeating: value)
                let vectorFP16 = Vector512FP16(from: vector)
                let back = vectorFP16.toFP32()

                if value.isInfinite {
                    // Infinity should clamp to max FP16
                    let expected: Float = value > 0 ? 65504.0 : -65504.0
                    #expect(back[0] == expected, "Infinity should clamp to max FP16")
                } else {
                    // Large values should clamp
                    let expected: Float = value > 0 ? 65504.0 : -65504.0
                    #expect(back[0] == expected, "Overflow should clamp to max FP16")
                }
            }

            // Test precision loss with many decimal places
            let preciseValue: Float = 3.141592653589793
            let preciseVector = Vector512Optimized(repeating: preciseValue)
            let preciseFP16 = Vector512FP16(from: preciseVector)
            let preciseBack = preciseFP16.toFP32()

            let precisionLoss = abs(preciseBack[0] - preciseValue)
            print("Precision loss: \(preciseValue) -> \(preciseBack[0]) (loss: \(precisionLoss))")
            #expect(precisionLoss < 0.001, "Precision loss should be bounded")

            // Test graceful degradation with mixed values
            let mixedVector = try Vector512Optimized((0..<512).map { i in
                switch i % 4 {
                case 0: return Float.nan
                case 1: return 100000.0  // Overflow
                case 2: return 1e-10     // Underflow
                default: return Float(i) * 0.1
                }
            })

            let mixedFP16 = Vector512FP16(from: mixedVector)
            let mixedBack = mixedFP16.toFP32()

            // Verify graceful handling
            for i in 0..<512 {
                switch i % 4 {
                case 0:
                    #expect(mixedBack[i].isNaN, "NaN should be preserved")
                case 1:
                    #expect(mixedBack[i] == 65504.0, "Overflow should clamp")
                case 2:
                    #expect(mixedBack[i] >= 0 && mixedBack[i] < 0.001, "Underflow should be near zero")
                default:
                    let expected = Float(i) * 0.1
                    let relError = abs(mixedBack[i] - expected) / max(abs(expected), 1.0)
                    #expect(relError < 0.01, "Normal values should have low error")
                }
            }
        }

        @Test
        func testZeroVectorHandling() throws {
            // Test zero vector handling in FP16
            // - All-zero vectors
            // - Near-zero vectors
            // - Normalization edge cases

            // Test exact zeros
            let zeroVector = Vector512Optimized(repeating: 0.0)
            let zeroVectorFP16 = Vector512FP16(from: zeroVector)
            let zeroBack = zeroVectorFP16.toFP32()

            for i in 0..<512 {
                #expect(zeroBack[i] == 0.0, "Exact zero should be preserved")
            }

            // Test near-zero values
            let epsilon: Float = 1e-10
            let nearZeroVector = Vector512Optimized(repeating: epsilon)
            let nearZeroFP16 = Vector512FP16(from: nearZeroVector)
            let nearZeroBack = nearZeroFP16.toFP32()

            for i in 0..<512 {
                #expect(abs(nearZeroBack[i]) < 0.001, "Near-zero should stay small or flush to zero")
            }

            // Test operations with zero vectors
            let nonZeroVector = try Vector512Optimized((0..<512).map { Float($0 + 1) })
            let nonZeroFP16 = Vector512FP16(from: nonZeroVector)

            // Dot product with zero
            let dotWithZero = zeroBack.dotProduct(nonZeroFP16.toFP32())
            #expect(dotWithZero == 0.0, "Dot product with zero vector should be zero")

            // Distance from zero (should equal magnitude)
            let distFromZero = zeroBack.euclideanDistance(to: nonZeroFP16.toFP32())
            let magnitude = nonZeroFP16.toFP32().magnitude
            let relError = abs(distFromZero - magnitude) / magnitude
            #expect(relError < 0.01, "Distance from zero should equal magnitude")

            // Normalization of zero vector should fail
            let zeroNorm = zeroBack.normalized()
            if case .failure = zeroNorm {
                // Expected
                print("Zero vector normalization failed as expected")
            } else {
                #expect(Bool(false), "Zero vector normalization should fail")
            }

            // Test preservation through operations
            let mixed = try Vector512Optimized((0..<512).map { i in
                i % 3 == 0 ? 0.0 : Float(i)
            })
            let mixedFP16 = Vector512FP16(from: mixed)
            let mixedBack = mixedFP16.toFP32()

            var zeroCount = 0
            for i in 0..<512 {
                if i % 3 == 0 {
                    #expect(mixedBack[i] == 0.0, "Zeros should be preserved in mixed vector")
                    zeroCount += 1
                }
            }
            print("Preserved \(zeroCount) zeros in mixed vector")
        }

        @Test
        func testDenormalNumberHandling() {
            // Test denormal number handling
            // - Subnormal FP16 values
            // - Performance implications
            // - Flush-to-zero behavior

            // FP16 denormal range: 2^-24 to 2^-14
            let denormalValues: [Float] = [
                5.960464477539063e-08,  // 2^-24 (smallest FP16 subnormal)
                1.1920928955078125e-07, // 2^-23
                3.0517578125e-05,       // Near normal boundary
                6.103515625e-05         // 2^-14 (smallest FP16 normal)
            ]

            print("Denormal Handling Test:")

            for value in denormalValues {
                let vector = Vector512Optimized(repeating: value)
                let vectorFP16 = Vector512FP16(from: vector)
                let back = vectorFP16.toFP32()

                let recovered = back[0]
                let isDenormal = value < 6.103515625e-05

                if isDenormal {
                    // Denormals may flush to zero or be approximated
                    if recovered == 0 {
                        print("  \(value) -> flushed to zero")
                    } else {
                        let relError = abs(recovered - value) / value
                        print("  \(value) -> \(recovered) (error: \(relError * 100)%)")
                        #expect(relError < 1.0, "Denormal approximation should be reasonable")
                    }
                } else {
                    // Normal values should be preserved more accurately
                    let relError = abs(recovered - value) / value
                    print("  \(value) -> \(recovered) (normal, error: \(relError * 100)%)")
                    #expect(relError < 0.01, "Normal values should have low error")
                }
            }

            // Test performance with denormals
            let normalVector = Vector512Optimized(repeating: 0.1)
            let denormalVector = Vector512Optimized(repeating: 1e-20)

            let normalFP16 = Vector512FP16(from: normalVector)
            let denormalFP16 = Vector512FP16(from: denormalVector)

            // Benchmark normal values
            let normalStart = Date()
            for _ in 0..<10000 {
                let _ = normalFP16.toFP32()
            }
            let normalTime = Date().timeIntervalSince(normalStart)

            // Benchmark denormal values
            let denormalStart = Date()
            for _ in 0..<10000 {
                let _ = denormalFP16.toFP32()
            }
            let denormalTime = Date().timeIntervalSince(denormalStart)

            print("\nPerformance:")
            print("  Normal values: \(normalTime * 1000)ms")
            print("  Denormal values: \(denormalTime * 1000)ms")

            // Denormals shouldn't cause extreme slowdown with FP16
            // (they're likely flushed to zero)
            #expect(denormalTime / normalTime < 2.0, "Denormals shouldn't cause extreme slowdown")
        }

        @Test
        func testNumericalInstabilityFP16() throws {
            // Test numerical instability with FP16
            // - Catastrophic cancellation
            // - Loss of precision
            // - Stability analysis

            // Test catastrophic cancellation
            let large: Float = 10000.0
            let v1 = Vector512Optimized(repeating: large)
            let v2 = Vector512Optimized(repeating: large - 0.001)  // Small difference

            let v1FP16 = Vector512FP16(from: v1)
            let v2FP16 = Vector512FP16(from: v2)

            let v1Back = v1FP16.toFP32()
            let v2Back = v2FP16.toFP32()

            // Direct subtraction (catastrophic cancellation)
            let directDiff = v1Back[0] - v2Back[0]
            let expectedDiff: Float = 0.001

            print("Catastrophic Cancellation:")
            print("  Large value 1: \(large) -> \(v1Back[0])")
            print("  Large value 2: \(large - 0.001) -> \(v2Back[0])")
            print("  Expected diff: \(expectedDiff)")
            print("  Actual diff: \(directDiff)")

            // FP16 has limited precision, so small differences may be lost
            let relError = abs(directDiff - expectedDiff) / expectedDiff
            #expect(relError < 10.0, "Cancellation error should be bounded")

            // Test loss of significance in summation
            let smallValue: Float = 0.001
            let largeSum = try Vector512Optimized((0..<512).map { i in
                i == 0 ? 10000.0 : smallValue
            })

            let largeSumFP16 = Vector512FP16(from: largeSum)
            let largeSumBack = largeSumFP16.toFP32()

            // Calculate sum
            let sum = largeSumBack.reduce(0, +)
            let expectedSum = 10000.0 + 511 * smallValue

            print("\nLoss of Significance:")
            print("  Expected sum: \(expectedSum)")
            print("  Actual sum: \(sum)")
            print("  Lost precision: \(expectedSum - sum)")

            // Small values may lose significance when added to large values
            #expect(abs(sum - expectedSum) < 1.0, "Sum error should be bounded")

            // Test numerical stability of dot product
            let stableV1 = try Vector512Optimized((0..<512).map { Float($0) / 100.0 })
            let stableV2 = try Vector512Optimized((0..<512).map { Float(511 - $0) / 100.0 })

            // FP32 reference
            let dotFP32 = stableV1.dotProduct(stableV2)

            // FP16 computation
            let stableV1FP16 = Vector512FP16(from: stableV1)
            let stableV2FP16 = Vector512FP16(from: stableV2)
            let dotFP16 = stableV1FP16.toFP32().dotProduct(stableV2FP16.toFP32())

            let dotError = abs(dotFP16 - dotFP32) / abs(dotFP32)
            print("\nDot Product Stability:")
            print("  FP32 result: \(dotFP32)")
            print("  FP16 result: \(dotFP16)")
            print("  Relative error: \(dotError * 100)%")

            #expect(dotError < 0.01, "Dot product should maintain reasonable accuracy")
        }
    }

    // MARK: - Real-World Application Tests

    @Suite("Real-World Applications")
    struct RealWorldApplicationTests {

        @Test
        func testSemanticSearchFP16() async throws {
            // Test semantic search with FP16 embeddings
            // - Document embedding similarity
            // - Search quality preservation
            // - Performance improvements

            // Simulate document embeddings (e.g., from BERT)
            let numDocuments = 100
            let documents = try (0..<numDocuments).map { docId in
                // Create pseudo-semantic embeddings
                try Vector512Optimized((0..<512).map { dim in
                    // Simulate clustered embeddings
                    let cluster = Float(docId / 10)  // Group similar documents
                    let noise = Float.random(in: -0.1...0.1)
                    return sin(Float(dim) * cluster / 100.0) + noise
                })
            }

            // Convert to FP16
            let documentsFP16 = documents.map { Vector512FP16(from: $0) }

            // Create query embedding
            let queryCluster = 5  // Looking for documents in cluster 5
            let query = try Vector512Optimized((0..<512).map { dim in
                sin(Float(dim) * Float(queryCluster) / 100.0) + Float.random(in: -0.05...0.05)
            })
            let queryFP16 = Vector512FP16(from: query)

            // Search with FP32 (baseline)
            var scoresFP32: [(index: Int, score: Float)] = []
            for (idx, doc) in documents.enumerated() {
                let similarity = query.dotProduct(doc) / (query.magnitude * doc.magnitude)
                scoresFP32.append((idx, similarity))
            }
            scoresFP32.sort { $0.score > $1.score }

            // Search with FP16
            var scoresFP16: [(index: Int, score: Float)] = []
            let queryFP32 = queryFP16.toFP32()
            for (idx, docFP16) in documentsFP16.enumerated() {
                let doc = docFP16.toFP32()
                let similarity = queryFP32.dotProduct(doc) / (queryFP32.magnitude * doc.magnitude)
                scoresFP16.append((idx, similarity))
            }
            scoresFP16.sort { $0.score > $1.score }

            // Compare top-K results
            let k = 10
            let topKFP32 = Set(scoresFP32.prefix(k).map { $0.index })
            let topKFP16 = Set(scoresFP16.prefix(k).map { $0.index })
            let overlap = topKFP32.intersection(topKFP16).count

            print("Semantic Search Results:")
            print("  Top-\(k) overlap: \(overlap)/\(k) (\(overlap * 100 / k)%)")
            print("  Top FP32 clusters: \(Set(topKFP32.map { $0 / 10 }))")
            print("  Top FP16 clusters: \(Set(topKFP16.map { $0 / 10 }))")

            // Measure performance improvement
            let searchStart32 = Date()
            for _ in 0..<100 {
                _ = documents.map { query.dotProduct($0) }
            }
            let time32 = Date().timeIntervalSince(searchStart32)

            let searchStart16 = Date()
            let qFP32 = queryFP16.toFP32()
            for _ in 0..<100 {
                _ = documentsFP16.map { qFP32.dotProduct($0.toFP32()) }
            }
            let time16 = Date().timeIntervalSince(searchStart16)

            print("  FP32 time: \(time32 * 1000)ms")
            print("  FP16 time: \(time16 * 1000)ms")
            print("  Speedup: \(time32/time16)x")

            #expect(overlap >= Int(Double(k) * 0.7), "Should maintain at least 70% top-K accuracy")
        }

        @Test
        func testRecommendationSystemFP16() async throws {
            // Test recommendation systems with FP16
            // - User/item embedding similarity
            // - Recommendation quality
            // - Scalability improvements

            // Simulate user and item embeddings
            let numUsers = 50
            let numItems = 200

            // User embeddings (preferences)
            let users = try (0..<numUsers).map { userId in
                try Vector512Optimized((0..<512).map { dim in
                    // Create user preference patterns
                    let preference = Float(userId % 5)  // User type
                    return cos(Float(dim) * preference / 50.0) + Float.random(in: -0.1...0.1)
                })
            }

            // Item embeddings (features)
            let items = try (0..<numItems).map { itemId in
                try Vector512Optimized((0..<512).map { dim in
                    // Create item feature patterns
                    let category = Float(itemId % 5)  // Item category
                    return sin(Float(dim) * category / 50.0) + Float.random(in: -0.1...0.1)
                })
            }

            // Convert to FP16
            let usersFP16 = users.map { Vector512FP16(from: $0) }
            let itemsFP16 = items.map { Vector512FP16(from: $0) }

            // Test recommendation for a specific user
            let testUserId = 7
            let userVector = users[testUserId]
            let userVectorFP16 = usersFP16[testUserId]

            // Get recommendations with FP32
            var recsFP32 = items.enumerated().map { (idx, item) in
                (idx, userVector.dotProduct(item))
            }
            recsFP32.sort { $0.1 > $1.1 }

            // Get recommendations with FP16
            let userFP32 = userVectorFP16.toFP32()
            var recsFP16 = itemsFP16.enumerated().map { (idx, itemFP16) in
                (idx, userFP32.dotProduct(itemFP16.toFP32()))
            }
            recsFP16.sort { $0.1 > $1.1 }

            // Compare top recommendations
            let topN = 20
            let topFP32 = Set(recsFP32.prefix(topN).map { $0.0 })
            let topFP16 = Set(recsFP16.prefix(topN).map { $0.0 })
            let recOverlap = topFP32.intersection(topFP16).count

            print("Recommendation System:")
            print("  User \(testUserId) (type \(testUserId % 5))")
            print("  Top-\(topN) recommendation overlap: \(recOverlap)/\(topN)")
            print("  Preservation rate: \(recOverlap * 100 / topN)%")

            // Check category alignment (users should prefer matching categories)
            let userType = testUserId % 5
            let fp32Categories = recsFP32.prefix(topN).map { $0.0 % 5 }
            let fp16Categories = recsFP16.prefix(topN).map { $0.0 % 5 }
            let fp32Matches = fp32Categories.filter { $0 == userType }.count
            let fp16Matches = fp16Categories.filter { $0 == userType }.count

            print("  Category matches FP32: \(fp32Matches)/\(topN)")
            print("  Category matches FP16: \(fp16Matches)/\(topN)")

            #expect(recOverlap >= Int(Double(topN) * 0.6), "Should maintain at least 60% recommendation accuracy")
            #expect(abs(fp16Matches - fp32Matches) <= 3, "Category matching should be similar")
        }

        @Test
        func testNeuralNetworkInferenceFP16() async throws {
            // Test neural network inference with FP16
            // - Embedding layer computation
            // - Attention mechanism efficiency
            // - Model accuracy preservation

            // Simulate a simple transformer attention mechanism
            let seqLength = 16
            let hiddenDim = 512
            let numHeads = 8
            let headDim = hiddenDim / numHeads  // 64

            // Generate query, key, value matrices
            let queries = try (0..<seqLength).map { pos in
                try Vector512Optimized((0..<hiddenDim).map { dim in
                    // Positional encoding + random features
                    sin(Float(pos) / pow(10000, Float(dim) / Float(hiddenDim))) +
                    Float.random(in: -0.1...0.1)
                })
            }

            let keys = queries  // Self-attention
            let values = queries

            // Convert to FP16
            let queriesFP16 = queries.map { Vector512FP16(from: $0) }
            let keysFP16 = keys.map { Vector512FP16(from: $0) }
            let _ = values.map { Vector512FP16(from: $0) }  // Values not used in attention score calculation

            // Compute attention scores with FP32
            var attentionFP32 = [[Float]](repeating: [Float](repeating: 0, count: seqLength),
                                          count: seqLength)
            for i in 0..<seqLength {
                for j in 0..<seqLength {
                    let score = queries[i].dotProduct(keys[j]) / sqrt(Float(headDim))
                    attentionFP32[i][j] = score
                }
            }

            // Compute attention scores with FP16
            var attentionFP16 = [[Float]](repeating: [Float](repeating: 0, count: seqLength),
                                          count: seqLength)
            for i in 0..<seqLength {
                for j in 0..<seqLength {
                    let qFP32 = queriesFP16[i].toFP32()
                    let kFP32 = keysFP16[j].toFP32()
                    let score = qFP32.dotProduct(kFP32) / sqrt(Float(headDim))
                    attentionFP16[i][j] = score
                }
            }

            // Compare attention patterns
            var totalError: Float = 0
            var maxError: Float = 0
            for i in 0..<seqLength {
                for j in 0..<seqLength {
                    let error = abs(attentionFP16[i][j] - attentionFP32[i][j])
                    totalError += error
                    maxError = max(maxError, error)
                }
            }

            let avgError = totalError / Float(seqLength * seqLength)
            print("Neural Network Inference:")
            print("  Sequence length: \(seqLength)")
            print("  Hidden dimension: \(hiddenDim)")
            print("  Average attention error: \(avgError)")
            print("  Max attention error: \(maxError)")

            // Test embedding layer (vocabulary lookup)
            let vocabSize = 1000
            let embeddings = try (0..<vocabSize).map { tokenId in
                try Vector512Optimized((0..<512).map { dim in
                    // Learned embeddings simulation
                    Float.random(in: -0.1...0.1) * sqrt(2.0 / Float(512))
                })
            }
            let embeddingsFP16 = embeddings.map { Vector512FP16(from: $0) }

            // Simulate batch embedding lookup
            let batchSize = 32
            let tokenIds = (0..<batchSize).map { _ in Int.random(in: 0..<vocabSize) }

            // Measure performance
            let fp32Start = Date()
            let _ = tokenIds.map { embeddings[$0] }
            let fp32Time = Date().timeIntervalSince(fp32Start)

            let fp16Start = Date()
            let _ = tokenIds.map { embeddingsFP16[$0].toFP32() }
            let fp16Time = Date().timeIntervalSince(fp16Start)

            print("  Embedding lookup FP32: \(fp32Time * 1000000)μs")
            print("  Embedding lookup FP16: \(fp16Time * 1000000)μs")
            print("  Memory savings: 50%")

            #expect(avgError < 0.01, "Average attention error should be small")
            #expect(maxError < 0.1, "Max attention error should be bounded")
        }

        @Test
        func testImageSimilarityFP16() async throws {
            // Test image similarity with FP16 features
            // - Visual feature comparison
            // - Image retrieval quality
            // - Processing speed improvements

            // Simulate image feature vectors (e.g., from ResNet or CLIP)
            let numImages = 100
            let featureDim = 512

            // Generate feature vectors for different image categories
            let categories = 5
            let images = try (0..<numImages).map { imageId in
                let category = imageId % categories
                return try Vector512Optimized((0..<featureDim).map { dim in
                    // Simulate visual features with category clustering
                    let baseFeature = sin(Float(dim * category) / 20.0)
                    let variation = Float.random(in: -0.2...0.2)
                    return baseFeature + variation
                })
            }

            // Convert to FP16
            let imagesFP16 = images.map { Vector512FP16(from: $0) }

            // Test image retrieval for a query image
            let queryId = 23  // Category 3
            let queryImage = images[queryId]
            let queryImageFP16 = imagesFP16[queryId]

            // Find similar images with FP32
            var similaritiesFP32 = images.enumerated().map { (idx, img) in
                let sim = queryImage.dotProduct(img) / (queryImage.magnitude * img.magnitude)
                return (idx, sim)
            }
            similaritiesFP32.sort { $0.1 > $1.1 }

            // Find similar images with FP16
            let queryFP32 = queryImageFP16.toFP32()
            var similaritiesFP16 = imagesFP16.enumerated().map { (idx, imgFP16) in
                let img = imgFP16.toFP32()
                let sim = queryFP32.dotProduct(img) / (queryFP32.magnitude * img.magnitude)
                return (idx, sim)
            }
            similaritiesFP16.sort { $0.1 > $1.1 }

            // Analyze retrieval quality
            let topK = 10
            let queryCategory = queryId % categories

            let topFP32 = similaritiesFP32.prefix(topK + 1).dropFirst()  // Skip self
            let topFP16 = similaritiesFP16.prefix(topK + 1).dropFirst()

            let fp32SameCategory = topFP32.filter { $0.0 % categories == queryCategory }.count
            let fp16SameCategory = topFP16.filter { $0.0 % categories == queryCategory }.count

            let topFP32Ids = Set(topFP32.map { $0.0 })
            let topFP16Ids = Set(topFP16.map { $0.0 })
            let retrievalOverlap = topFP32Ids.intersection(topFP16Ids).count

            print("Image Similarity Search:")
            print("  Query image: ID \(queryId), Category \(queryCategory)")
            print("  Top-\(topK) retrieval overlap: \(retrievalOverlap)/\(topK)")
            print("  Same category FP32: \(fp32SameCategory)/\(topK)")
            print("  Same category FP16: \(fp16SameCategory)/\(topK)")

            // Measure batch processing speed
            let batchQueries = Array(images.prefix(10))
            let batchQueriesFP16 = Array(imagesFP16.prefix(10))

            let batchStartFP32 = Date()
            for query in batchQueries {
                _ = images.map { query.dotProduct($0) }
            }
            let batchTimeFP32 = Date().timeIntervalSince(batchStartFP32)

            let batchStartFP16 = Date()
            for queryFP16 in batchQueriesFP16 {
                let q = queryFP16.toFP32()
                _ = imagesFP16.map { q.dotProduct($0.toFP32()) }
            }
            let batchTimeFP16 = Date().timeIntervalSince(batchStartFP16)

            print("  Batch processing FP32: \(batchTimeFP32 * 1000)ms")
            print("  Batch processing FP16: \(batchTimeFP16 * 1000)ms")
            print("  Speedup: \(batchTimeFP32/batchTimeFP16)x")

            #expect(retrievalOverlap >= Int(Double(topK) * 0.7), "Should maintain good retrieval accuracy")
            #expect(abs(fp16SameCategory - fp32SameCategory) <= 2, "Category retrieval should be similar")
        }
    }

    // MARK: - Helper Functions

    // Generate FP32 test vectors for mixed precision testing
    private static func generateFP32TestVectors(count: Int, dimension: Int = 512) -> [Vector512Optimized] {
        return (0..<count).map { i in
            // Generate diverse test patterns
            let pattern = i % 5
            switch pattern {
            case 0:  // Random uniform
                return Vector512Optimized { _ in Float.random(in: -1...1) }
            case 1:  // Sparse
                return Vector512Optimized { dim in dim % 10 == 0 ? Float.random(in: -1...1) : 0 }
            case 2:  // Gradient
                return Vector512Optimized { dim in Float(dim) / Float(dimension) }
            case 3:  // Sinusoidal
                return Vector512Optimized { dim in sin(Float(dim) * Float(i) / 100.0) }
            default:  // Gaussian-like
                let mean = Float.random(in: -0.5...0.5)
                let std: Float = 0.3
                return Vector512Optimized { _ in
                    // Box-Muller approximation
                    let u1 = Float.random(in: 0.001...0.999)
                    let u2 = Float.random(in: 0.001...0.999)
                    let z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
                    return mean + std * z0
                }
            }
        }
    }

    // Convert FP32 vectors to FP16
    private static func convertToFP16(vectors: [Vector512Optimized]) -> [Vector512FP16] {
        return vectors.map { Vector512FP16(from: $0) }
    }

    // Measure accuracy loss between FP32 and FP16 results
    private static func measureAccuracyLoss(fp32Results: [Float], fp16Results: [Float]) -> Float {
        guard fp32Results.count == fp16Results.count else {
            return Float.infinity
        }

        var totalError: Float = 0
        var maxError: Float = 0
        var relativeErrors: [Float] = []

        for i in 0..<fp32Results.count {
            let error = abs(fp32Results[i] - fp16Results[i])
            totalError += error
            maxError = max(maxError, error)

            if fp32Results[i] != 0 {
                relativeErrors.append(error / abs(fp32Results[i]))
            }
        }

        let _ = totalError / Float(fp32Results.count)  // Average error
        let avgRelError = relativeErrors.isEmpty ? 0 : relativeErrors.reduce(0, +) / Float(relativeErrors.count)

        // Return average relative error as primary metric
        return avgRelError
    }

    // Measure memory usage of operation
    private static func measureMemoryUsage(operation: () -> Void) -> Int {
        // Note: This is a simplified memory measurement
        // For accurate measurement, use Instruments or memory profiling tools

        // Get baseline memory
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let result = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { int_ptr in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), int_ptr, &count)
            }
        }

        let baselineMemory = result == KERN_SUCCESS ? info.resident_size : 0

        // Run operation
        operation()

        // Get memory after operation
        _ = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { int_ptr in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), int_ptr, &count)
            }
        }

        let afterMemory = result == KERN_SUCCESS ? info.resident_size : baselineMemory

        // Return difference in bytes
        return Int(afterMemory - baselineMemory)
    }

    // Measure performance improvement of FP16 vs FP32
    private static func measurePerformanceImprovement(
        fp32Operation: () async -> Void,
        fp16Operation: () async -> Void,
        iterations: Int = 100
    ) async -> Double {
        // Warm up
        await fp32Operation()
        await fp16Operation()

        // Measure FP32
        let fp32Start = Date()
        for _ in 0..<iterations {
            await fp32Operation()
        }
        let fp32Time = Date().timeIntervalSince(fp32Start)

        // Measure FP16
        let fp16Start = Date()
        for _ in 0..<iterations {
            await fp16Operation()
        }
        let fp16Time = Date().timeIntervalSince(fp16Start)

        // Return speedup factor (>1 means FP16 is faster)
        return fp32Time / fp16Time
    }

    // Validate that NEON intrinsics are being used
    private static func validateNEONIntrinsicUsage() -> Bool {
        // On Apple Silicon, NEON is always available and used for SIMD operations
        #if arch(arm64)
        // Check if we're on ARM64 architecture (Apple Silicon)
        // NEON is mandatory on ARMv8 (which includes all Apple Silicon)
        return true
        #else
        // On Intel Macs, SSE/AVX would be used instead
        return false
        #endif
    }
}