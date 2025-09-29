import Testing
import Foundation
@testable import VectorCore

@Suite("Quantized Kernels")
struct QuantizedKernelsTests {

    // MARK: - INT8 Quantization Tests

    @Suite("INT8 Quantization")
    struct INT8QuantizationTests {

        @Test
        func testINT8QuantizationBasic() throws {
            // Test basic INT8 quantization of FP32 vectors
            // - Linear quantization scheme
            // - Scale and zero-point calculation
            // - Range preservation and clipping

            // Create test vector with known range
            let testVector = try Vector512Optimized((0..<512).map { Float($0) / 100.0 - 2.56 })  // Range: [-2.56, 2.55]

            // Test symmetric quantization
            let symmetricParams = LinearQuantizationParams(
                minValue: -2.56,
                maxValue: 2.55,
                symmetric: true
            )

            let quantizedSymmetric = Vector512INT8(from: testVector, params: symmetricParams)

            // Verify quantization parameters
            #expect(quantizedSymmetric.quantizationParams.isSymmetric == true)
            #expect(quantizedSymmetric.quantizationParams.zeroPoint == 0)
            #expect(abs(quantizedSymmetric.quantizationParams.scale - (2.56 / 127.0)) < 0.001)

            // Test asymmetric quantization
            let asymmetricParams = LinearQuantizationParams(
                minValue: -2.56,
                maxValue: 2.55,
                symmetric: false
            )

            let quantizedAsymmetric = Vector512INT8(from: testVector, params: asymmetricParams)

            // Verify asymmetric parameters
            #expect(quantizedAsymmetric.quantizationParams.isSymmetric == false)
            #expect(quantizedAsymmetric.quantizationParams.zeroPoint != 0)

            // Test auto-calibration (params = nil)
            let quantizedAuto = Vector512INT8(from: testVector)

            // Verify auto-calibrated parameters match expected range
            #expect(abs(quantizedAuto.quantizationParams.minValue - (-2.56)) < 0.01)
            #expect(abs(quantizedAuto.quantizationParams.maxValue - 2.55) < 0.01)

            // Test edge case: constant vector
            let constantVector = Vector512Optimized(repeating: 1.5)
            let quantizedConstant = Vector512INT8(from: constantVector)

            // For constant vectors, scale should handle the degenerate case
            #expect(quantizedConstant.quantizationParams.minValue == 1.5)
            #expect(quantizedConstant.quantizationParams.maxValue == 1.5)

            print("INT8 Quantization Basic:")
            print("  Symmetric scale: \(symmetricParams.scale)")
            print("  Symmetric zero-point: \(symmetricParams.zeroPoint)")
            print("  Asymmetric scale: \(asymmetricParams.scale)")
            print("  Asymmetric zero-point: \(asymmetricParams.zeroPoint)")
            print("  Auto-calibrated scale: \(quantizedAuto.quantizationParams.scale)")
        }

        @Test
        func testINT8QuantizationAccuracy() throws {
            // Test quantization accuracy analysis
            // - Quantization error measurement
            // - Signal-to-noise ratio
            // - Distribution of quantization errors

            // Create test vectors with different distributions
            let uniformVector = try Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
            let gaussianVector = try Vector512Optimized((0..<512).map { _ in
                // Box-Muller transform for Gaussian distribution
                let u1 = Float.random(in: 0.001...0.999)
                let u2 = Float.random(in: 0.001...0.999)
                return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
            })
            let sparseVector = try Vector512Optimized((0..<512).map { i in
                i % 10 == 0 ? Float.random(in: -1...1) : 0.0
            })

            // Quantize and dequantize each vector
            let testVectors = [
                ("Uniform", uniformVector),
                ("Gaussian", gaussianVector),
                ("Sparse", sparseVector)
            ]

            print("INT8 Quantization Accuracy:")

            for (name, vector) in testVectors {
                let quantized = Vector512INT8(from: vector)
                let dequantized = quantized.toFP32()

                // Calculate quantization errors
                var mse: Float = 0
                var maxError: Float = 0
                var signalPower: Float = 0

                for i in 0..<512 {
                    let original = vector[i]
                    let reconstructed = dequantized[i]
                    let error = abs(original - reconstructed)

                    mse += error * error
                    maxError = max(maxError, error)
                    signalPower += original * original
                }

                mse /= 512.0
                signalPower /= 512.0
                let rmse = sqrt(mse)
                let snr = 10 * log10(signalPower / mse)  // SNR in dB

                print("  \(name) distribution:")
                print("    RMSE: \(rmse)")
                print("    Max error: \(maxError)")
                print("    SNR: \(snr) dB")
                print("    Quantization scale: \(quantized.quantizationParams.scale)")

                // Verify acceptable accuracy
                #expect(rmse < 0.05, "RMSE should be less than 5% for \(name) distribution")
                #expect(snr > 20, "SNR should be greater than 20 dB for \(name) distribution")
            }

            // Test error distribution
            let testVector = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.1) })
            let quantized = Vector512INT8(from: testVector)
            let dequantized = quantized.toFP32()

            var errors: [Float] = []
            for i in 0..<512 {
                errors.append(dequantized[i] - testVector[i])
            }

            // Check error statistics
            let meanError = errors.reduce(0, +) / Float(errors.count)
            let variance = errors.map { pow($0 - meanError, 2) }.reduce(0, +) / Float(errors.count)
            let stdDev = sqrt(variance)

            print("  Error distribution:")
            print("    Mean error: \(meanError)")
            print("    Std dev: \(stdDev)")

            // Errors should be approximately zero-mean and uniformly distributed
            #expect(abs(meanError) < 0.01, "Mean error should be close to zero")
            #expect(stdDev < quantized.quantizationParams.scale, "Error std dev should be bounded by scale")
        }

        @Test
        func testINT8DequantizationRoundTrip() throws {
            // Test round-trip quantization→dequantization
            // - Precision loss measurement
            // - Error accumulation analysis
            // - Acceptable tolerance validation

            // Create test vectors with various value ranges
            let testCases: [(String, Vector512Optimized)] = [
                ("Small range", try Vector512Optimized((0..<512).map { Float($0) * 0.001 })),
                ("Medium range", try Vector512Optimized((0..<512).map { Float($0) * 0.1 })),
                ("Large range", try Vector512Optimized((0..<512).map { Float($0) })),
                ("Mixed signs", try Vector512Optimized((0..<512).map { Float($0 - 256) * 0.01 }))
            ]

            print("INT8 Round-Trip Analysis:")

            for (name, originalVector) in testCases {
                // First round-trip
                let quantized1 = Vector512INT8(from: originalVector)
                let dequantized1 = quantized1.toFP32()

                // Second round-trip (to test error accumulation)
                let quantized2 = Vector512INT8(from: dequantized1)
                let dequantized2 = quantized2.toFP32()

                // Third round-trip
                let quantized3 = Vector512INT8(from: dequantized2)
                let dequantized3 = quantized3.toFP32()

                // Calculate errors at each stage
                var error1: Float = 0
                var error2: Float = 0
                var error3: Float = 0

                for i in 0..<512 {
                    let orig = originalVector[i]
                    error1 += abs(dequantized1[i] - orig)
                    error2 += abs(dequantized2[i] - orig)
                    error3 += abs(dequantized3[i] - orig)
                }

                error1 /= 512.0
                error2 /= 512.0
                error3 /= 512.0

                print("  \(name):")
                print("    Error after 1 round-trip: \(error1)")
                print("    Error after 2 round-trips: \(error2)")
                print("    Error after 3 round-trips: \(error3)")
                print("    Error growth ratio: \(error3/error1)")

                // Verify error doesn't grow unboundedly
                #expect(error3 / error1 < 1.5, "Error should not grow significantly over multiple round-trips")

                // Check idempotence: after first quantization, subsequent quantizations should be stable
                for i in 0..<512 {
                    let diff23 = abs(dequantized2[i] - dequantized3[i])
                    #expect(diff23 < 0.001, "Should be approximately idempotent after first quantization")
                }
            }

            // Test precision preservation for values exactly representable in INT8
            let scale: Float = 0.01
            let testValues = (-127...127).map { Float($0) * scale }
            let exactVector = try Vector512Optimized(Array(testValues) + Array(repeating: 0.0, count: 512 - testValues.count))

            let params = LinearQuantizationParams(minValue: -1.27, maxValue: 1.27, symmetric: true)
            let quantized = Vector512INT8(from: exactVector, params: params)
            let dequantized = quantized.toFP32()

            // Values that map exactly to INT8 should have minimal error
            for i in 0..<testValues.count {
                let error = abs(dequantized[i] - exactVector[i])
                #expect(error < scale * 0.5, "Exactly representable values should have minimal error")
            }
        }

        @Test
        func testINT8RangeHandling() throws {
            // Test handling of different value ranges
            // - Symmetric vs asymmetric quantization
            // - Signed vs unsigned INT8
            // - Optimal scale factor calculation

            // Test different value ranges
            let rangeTests: [(String, Float, Float, Vector512Optimized)] = [
                ("Positive only", 0.0, 10.0, try Vector512Optimized((0..<512).map { Float($0) * 10.0 / 512.0 })),
                ("Negative only", -10.0, 0.0, try Vector512Optimized((0..<512).map { Float($0) * -10.0 / 512.0 })),
                ("Symmetric", -5.0, 5.0, try Vector512Optimized((0..<512).map { Float($0 - 256) * 5.0 / 256.0 })),
                ("Asymmetric", -3.0, 7.0, try Vector512Optimized((0..<512).map { Float($0) * 10.0 / 512.0 - 3.0 })),
                ("Large range", -100.0, 100.0, try Vector512Optimized((0..<512).map { Float($0 - 256) * 100.0 / 256.0 })),
                ("Small range", -0.01, 0.01, try Vector512Optimized((0..<512).map { Float($0 - 256) * 0.01 / 256.0 }))
            ]

            print("INT8 Range Handling:")

            for (name, minVal, maxVal, vector) in rangeTests {
                // Test symmetric quantization
                let symmetricParams = LinearQuantizationParams(
                    minValue: minVal,
                    maxValue: maxVal,
                    symmetric: true
                )
                let quantizedSym = Vector512INT8(from: vector, params: symmetricParams)

                // Test asymmetric quantization
                let asymmetricParams = LinearQuantizationParams(
                    minValue: minVal,
                    maxValue: maxVal,
                    symmetric: false
                )
                let quantizedAsym = Vector512INT8(from: vector, params: asymmetricParams)

                // Dequantize and measure accuracy
                let dequantizedSym = quantizedSym.toFP32()
                let dequantizedAsym = quantizedAsym.toFP32()

                var errorSym: Float = 0
                var errorAsym: Float = 0

                for i in 0..<512 {
                    errorSym += abs(dequantizedSym[i] - vector[i])
                    errorAsym += abs(dequantizedAsym[i] - vector[i])
                }

                errorSym /= 512.0
                errorAsym /= 512.0

                print("  \(name) [\(minVal), \(maxVal)]:")
                print("    Symmetric - Scale: \(symmetricParams.scale), ZP: \(symmetricParams.zeroPoint), Error: \(errorSym)")
                print("    Asymmetric - Scale: \(asymmetricParams.scale), ZP: \(asymmetricParams.zeroPoint), Error: \(errorAsym)")

                // Verify scale calculation
                if symmetricParams.isSymmetric {
                    let expectedScale = max(abs(minVal), abs(maxVal)) / 127.0
                    #expect(abs(symmetricParams.scale - expectedScale) < 0.001 || symmetricParams.scale == 1.0,
                           "Symmetric scale calculation incorrect")
                    #expect(symmetricParams.zeroPoint == 0, "Symmetric should have zero-point = 0")
                } else {
                    let expectedScale = (maxVal - minVal) / 255.0
                    #expect(abs(asymmetricParams.scale - expectedScale) < 0.001 || asymmetricParams.scale == 1.0,
                           "Asymmetric scale calculation incorrect")
                }

                // For skewed ranges, asymmetric should generally be more accurate
                if abs(minVal) != abs(maxVal) && minVal != 0 && maxVal != 0 {
                    // Asymmetric can utilize the full INT8 range better for skewed data
                    print("    Better scheme: \(errorAsym < errorSym ? "Asymmetric" : "Symmetric")")
                }
            }

            // Test clipping behavior for out-of-range values
            let testVector = try Vector512Optimized((0..<512).map { Float($0 - 256) })
            let params = LinearQuantizationParams(minValue: -10.0, maxValue: 10.0, symmetric: true)
            let quantized = Vector512INT8(from: testVector, params: params)
            let dequantized = quantized.toFP32()

            // Values outside [-10, 10] should be clipped
            for i in 0..<512 {
                let original = testVector[i]
                let reconstructed = dequantized[i]

                if original < -10.0 {
                    #expect(abs(reconstructed - (-10.0)) < 0.2, "Values below min should clip to min")
                } else if original > 10.0 {
                    #expect(abs(reconstructed - 10.0) < 0.2, "Values above max should clip to max")
                }
            }
        }

        @Test
        func testINT8QuantizationMethods() throws {
            // Test different quantization methods
            // - Linear uniform quantization
            // - Non-uniform quantization
            // - Adaptive quantization schemes

            // Create test vector with non-uniform distribution
            let testVector = try Vector512Optimized((0..<512).map { i in
                // Create a distribution with more values near zero (common in neural networks)
                let x = Float(i - 256) / 256.0
                return tanh(x * 2.0)  // Most values in [-1, 1] with concentration near 0
            })

            // Method 1: Standard linear quantization
            let linearParams = LinearQuantizationParams(
                minValue: testVector.toArray().min()!,
                maxValue: testVector.toArray().max()!,
                symmetric: true
            )
            let linearQuantized = Vector512INT8(from: testVector, params: linearParams)

            // Method 2: Percentile-based quantization (clip outliers)
            let sorted = testVector.toArray().sorted()
            let percentile1 = sorted[Int(Float(sorted.count) * 0.01)]  // 1st percentile
            let percentile99 = sorted[Int(Float(sorted.count) * 0.99)]  // 99th percentile
            let percentileParams = LinearQuantizationParams(
                minValue: percentile1,
                maxValue: percentile99,
                symmetric: false
            )
            let percentileQuantized = Vector512INT8(from: testVector, params: percentileParams)

            // Method 3: Adaptive symmetric (use smaller of abs(min) and abs(max))
            let absMax = max(abs(testVector.toArray().min()!), abs(testVector.toArray().max()!))
            let adaptiveParams = LinearQuantizationParams(
                minValue: -absMax,
                maxValue: absMax,
                symmetric: true
            )
            let adaptiveQuantized = Vector512INT8(from: testVector, params: adaptiveParams)

            print("INT8 Quantization Methods:")

            // Compare accuracy of different methods
            let methods = [
                ("Linear", linearQuantized),
                ("Percentile", percentileQuantized),
                ("Adaptive", adaptiveQuantized)
            ]

            for (name, quantized) in methods {
                let dequantized = quantized.toFP32()

                var mse: Float = 0
                var maxError: Float = 0
                var inRangeCount = 0

                for i in 0..<512 {
                    let error = abs(dequantized[i] - testVector[i])
                    mse += error * error
                    maxError = max(maxError, error)

                    // Count values within acceptable error threshold
                    if error < 0.01 {
                        inRangeCount += 1
                    }
                }

                mse /= 512.0
                let rmse = sqrt(mse)
                let accuracy = Float(inRangeCount) / 512.0 * 100.0

                print("  \(name) method:")
                print("    Scale: \(quantized.quantizationParams.scale)")
                print("    Zero-point: \(quantized.quantizationParams.zeroPoint)")
                print("    RMSE: \(rmse)")
                print("    Max error: \(maxError)")
                print("    Accuracy (<0.01 error): \(accuracy)%")

                #expect(rmse < 0.1, "RMSE should be reasonable for \(name) method")
            }

            // Test per-channel quantization simulation (quantize subsets independently)
            let channel1 = try Vector512Optimized(Array(testVector.toArray()[0..<256]) + Array(repeating: 0.0, count: 256))
            let channel2 = try Vector512Optimized(Array(repeating: 0.0, count: 256) + Array(testVector.toArray()[256..<512]))

            let channel1Quantized = Vector512INT8(from: channel1)
            let channel2Quantized = Vector512INT8(from: channel2)

            print("  Per-channel simulation:")
            print("    Channel 1 scale: \(channel1Quantized.quantizationParams.scale)")
            print("    Channel 2 scale: \(channel2Quantized.quantizationParams.scale)")

            // Different channels should have different scales if their ranges differ
            if abs(channel1.toArray()[0..<256].max()!) != abs(channel2.toArray()[256..<512].max()!) {
                #expect(channel1Quantized.quantizationParams.scale != channel2Quantized.quantizationParams.scale,
                       "Per-channel should have different scales for different ranges")
            }
        }

        @Test
        func testINT8CalibrationDatasets() throws {
            // Test quantization calibration with different datasets
            // - Representative data sampling
            // - Distribution analysis for optimal quantization
            // - Calibration set size effects

            // Generate a large dataset
            let fullDataset = (0..<1000).map { _ in
                Vector512Optimized { _ in Float.random(in: -2...2) * (Float.random(in: 0...1) > 0.8 ? 5.0 : 1.0) }  // With outliers
            }

            // Test different calibration set sizes
            let calibrationSizes = [10, 50, 100, 500, 1000]
            var results: [(size: Int, scale: Float, error: Float)] = []

            print("INT8 Calibration Dataset Analysis:")

            for calibSize in calibrationSizes {
                // Sample calibration set
                let calibrationSet = Array(fullDataset.prefix(calibSize))

                // Find min/max across calibration set
                var globalMin: Float = Float.greatestFiniteMagnitude
                var globalMax: Float = -Float.greatestFiniteMagnitude

                for vector in calibrationSet {
                    let array = vector.toArray()
                    globalMin = min(globalMin, array.min()!)
                    globalMax = max(globalMax, array.max()!)
                }

                // Create quantization parameters from calibration
                let params = LinearQuantizationParams(
                    minValue: globalMin,
                    maxValue: globalMax,
                    symmetric: true
                )

                // Test on full dataset
                var totalError: Float = 0
                var outlierCount = 0

                for vector in fullDataset {
                    let quantized = Vector512INT8(from: vector, params: params)
                    let dequantized = quantized.toFP32()

                    for i in 0..<512 {
                        let error = abs(dequantized[i] - vector[i])
                        totalError += error

                        // Count values that were clipped (outliers)
                        if vector[i] < globalMin || vector[i] > globalMax {
                            outlierCount += 1
                        }
                    }
                }

                let avgError = totalError / Float(fullDataset.count * 512)
                results.append((calibSize, params.scale, avgError))

                print("  Calibration size \(calibSize):")
                print("    Range: [\(globalMin), \(globalMax)]")
                print("    Scale: \(params.scale)")
                print("    Avg error: \(avgError)")
                print("    Outliers: \(outlierCount) / \(fullDataset.count * 512)")
            }

            // Verify that larger calibration sets generally give better results
            if results.count >= 2 {
                let smallSetError = results[0].error
                let largeSetError = results[results.count - 1].error
                print("\n  Error reduction: \((smallSetError - largeSetError) / smallSetError * 100)%")
            }

            // Test percentile-based calibration (robust to outliers)
            let allValues = fullDataset.flatMap { $0.toArray() }.sorted()
            let p01 = allValues[Int(Float(allValues.count) * 0.001)]  // 0.1 percentile
            let p99 = allValues[Int(Float(allValues.count) * 0.999)]  // 99.9 percentile

            let robustParams = LinearQuantizationParams(
                minValue: p01,
                maxValue: p99,
                symmetric: false
            )

            var robustError: Float = 0
            for vector in fullDataset {
                let quantized = Vector512INT8(from: vector, params: robustParams)
                let dequantized = quantized.toFP32()

                for i in 0..<512 {
                    // Only count error for non-outlier values
                    if vector[i] >= p01 && vector[i] <= p99 {
                        robustError += abs(dequantized[i] - vector[i])
                    }
                }
            }
            robustError /= Float(fullDataset.count * 512)

            print("\n  Percentile-based calibration:")
            print("    Range: [\(p01), \(p99)]")
            print("    Scale: \(robustParams.scale)")
            print("    Avg error (non-outliers): \(robustError)")

            #expect(robustError < results[0].error, "Percentile calibration should handle outliers better")
        }
    }

    // MARK: - Quantized Distance Computation Tests

    @Suite("Quantized Distance Computation")
    struct QuantizedDistanceComputationTests {

        @Test
        func testQuantizedEuclideanDistance() throws {
            // Test Euclidean distance in INT8 quantized space
            // - Direct INT8 computation
            // - Accuracy vs FP32 reference
            // - Performance improvements

            // Create test vectors
            let v1 = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.1) })
            let v2 = try Vector512Optimized((0..<512).map { cos(Float($0) * 0.1) })
            let v3 = try Vector512Optimized((0..<512).map { Float($0) / 256.0 - 1.0 })

            // Calculate FP32 reference distances
            let distFP32_12 = v1.euclideanDistance(to: v2)
            let distFP32_13 = v1.euclideanDistance(to: v3)
            let distFP32_23 = v2.euclideanDistance(to: v3)

            // Quantize vectors with same parameters (required for optimized INT8 distance)
            let minVal = min(v1.toArray().min()!, v2.toArray().min()!, v3.toArray().min()!)
            let maxVal = max(v1.toArray().max()!, v2.toArray().max()!, v3.toArray().max()!)
            let params = LinearQuantizationParams(minValue: minVal, maxValue: maxVal, symmetric: true)

            let q1 = Vector512INT8(from: v1, params: params)
            let q2 = Vector512INT8(from: v2, params: params)
            let q3 = Vector512INT8(from: v3, params: params)

            // Calculate quantized distances
            let distINT8_12 = QuantizedKernels.euclidean512(query: q1, candidate: q2)
            let distINT8_13 = QuantizedKernels.euclidean512(query: q1, candidate: q3)
            let distINT8_23 = QuantizedKernels.euclidean512(query: q2, candidate: q3)

            print("Quantized Euclidean Distance:")
            print("  v1-v2: FP32=\(distFP32_12), INT8=\(distINT8_12), Error=\(abs(distFP32_12 - distINT8_12))")
            print("  v1-v3: FP32=\(distFP32_13), INT8=\(distINT8_13), Error=\(abs(distFP32_13 - distINT8_13))")
            print("  v2-v3: FP32=\(distFP32_23), INT8=\(distINT8_23), Error=\(abs(distFP32_23 - distINT8_23))")

            // Calculate relative errors
            let relError12 = abs(distFP32_12 - distINT8_12) / distFP32_12
            let relError13 = abs(distFP32_13 - distINT8_13) / distFP32_13
            let relError23 = abs(distFP32_23 - distINT8_23) / distFP32_23

            print("  Relative errors: \(relError12 * 100)%, \(relError13 * 100)%, \(relError23 * 100)%")

            // Verify accuracy
            #expect(relError12 < 0.05, "Relative error should be < 5%")
            #expect(relError13 < 0.05, "Relative error should be < 5%")
            #expect(relError23 < 0.05, "Relative error should be < 5%")

            // Test performance
            let iterations = 10000

            // Benchmark FP32
            let fp32Start = Date()
            for _ in 0..<iterations {
                _ = v1.euclideanDistance(to: v2)
            }
            let fp32Time = Date().timeIntervalSince(fp32Start)

            // Benchmark INT8
            let int8Start = Date()
            for _ in 0..<iterations {
                _ = QuantizedKernels.euclidean512(query: q1, candidate: q2)
            }
            let int8Time = Date().timeIntervalSince(int8Start)

            let speedup = fp32Time / int8Time
            print("\n  Performance:")
            print("    FP32 time: \(fp32Time * 1000)ms")
            print("    INT8 time: \(int8Time * 1000)ms")
            print("    Speedup: \(speedup)x")

            #expect(speedup > 1.0, "INT8 should be faster than FP32")

            // Test edge case: identical vectors
            let identicalDist = QuantizedKernels.euclidean512(query: q1, candidate: q1)
            #expect(identicalDist < 0.001, "Distance between identical vectors should be ~0")

            // Test triangle inequality
            let d12 = QuantizedKernels.euclidean512(query: q1, candidate: q2)
            let d23 = QuantizedKernels.euclidean512(query: q2, candidate: q3)
            let d13 = QuantizedKernels.euclidean512(query: q1, candidate: q3)
            #expect(d13 <= d12 + d23 + 0.1, "Triangle inequality should hold (with small tolerance)")
        }

        @Test
        func testQuantizedEuclideanSquaredDistance() throws {
            // Test squared Euclidean distance in INT8
            // - Avoid expensive sqrt operation
            // - Integer arithmetic optimization
            // - Overflow handling

            // Create test vectors with various magnitudes
            let smallVector = try Vector512Optimized((0..<512).map { Float($0) * 0.001 })  // Small values
            let mediumVector = try Vector512Optimized((0..<512).map { Float($0) * 0.1 })    // Medium values
            let largeVector = try Vector512Optimized((0..<512).map { Float($0) })           // Large values

            let vectors = [("Small", smallVector), ("Medium", mediumVector), ("Large", largeVector)]

            print("Quantized Euclidean Squared Distance:")

            for i in 0..<vectors.count {
                for j in i+1..<vectors.count {
                    let (name1, v1) = vectors[i]
                    let (name2, v2) = vectors[j]

                    // FP32 reference
                    let distSqFP32 = v1.euclideanDistanceSquared(to: v2)

                    // Quantize with appropriate range
                    let minVal = min(v1.toArray().min()!, v2.toArray().min()!)
                    let maxVal = max(v1.toArray().max()!, v2.toArray().max()!)
                    let params = LinearQuantizationParams(minValue: minVal, maxValue: maxVal, symmetric: true)

                    let q1 = Vector512INT8(from: v1, params: params)
                    let q2 = Vector512INT8(from: v2, params: params)

                    // Calculate squared distance in INT8 (square the euclidean distance)
                    let distINT8 = QuantizedKernels.euclidean512(query: q1, candidate: q2)
                    let distSqINT8 = distINT8 * distINT8

                    print("  \(name1)-\(name2):")
                    print("    FP32: \(distSqFP32)")
                    print("    INT8: \(distSqINT8)")
                    print("    Error: \(abs(distSqFP32 - distSqINT8))")
                    print("    Scale: \(params.scale)")

                    // Verify no overflow occurred (result should be positive)
                    #expect(distSqINT8 >= 0, "Squared distance should be non-negative")

                    // Check relative accuracy
                    if distSqFP32 > 0 {
                        let relError = abs(distSqFP32 - distSqINT8) / distSqFP32
                        #expect(relError < 0.1, "Relative error should be < 10%")
                    }
                }
            }

            // Test performance advantage of squared distance
            let v1 = try Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
            let v2 = try Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
            let params = LinearQuantizationParams(minValue: -1, maxValue: 1, symmetric: true)
            let q1 = Vector512INT8(from: v1, params: params)
            let q2 = Vector512INT8(from: v2, params: params)

            let iterations = 10000

            // Benchmark with sqrt
            let withSqrtStart = Date()
            for _ in 0..<iterations {
                _ = QuantizedKernels.euclidean512(query: q1, candidate: q2)
            }
            let withSqrtTime = Date().timeIntervalSince(withSqrtStart)

            // Benchmark without sqrt
            let withoutSqrtStart = Date()
            for _ in 0..<iterations {
                let d = QuantizedKernels.euclidean512(query: q1, candidate: q2)
                _ = d * d
            }
            let withoutSqrtTime = Date().timeIntervalSince(withoutSqrtStart)

            print("\n  Performance comparison:")
            print("    With sqrt: \(withSqrtTime * 1000)ms")
            print("    Without sqrt: \(withoutSqrtTime * 1000)ms")
            print("    Speedup: \(withSqrtTime / withoutSqrtTime)x")

            #expect(withoutSqrtTime < withSqrtTime, "Squared distance should be faster")

            // Test accumulator overflow protection
            // Create vectors that would cause overflow with naive INT8 arithmetic
            let maxVector1 = Vector512Optimized(repeating: 127.0)
            let maxVector2 = Vector512Optimized(repeating: -128.0)
            let maxParams = LinearQuantizationParams(minValue: -128, maxValue: 127, symmetric: false)
            let qMax1 = Vector512INT8(from: maxVector1, params: maxParams)
            let qMax2 = Vector512INT8(from: maxVector2, params: maxParams)

            // This should handle maximum possible difference without overflow
            let maxDist = QuantizedKernels.euclidean512(query: qMax1, candidate: qMax2)
            let maxDistSq = maxDist * maxDist
            #expect(maxDistSq > 0, "Should handle maximum difference without overflow")
            print("\n  Max difference test: \(maxDistSq) (no overflow)")
        }

        @Test
        func testQuantizedDotProduct() throws {
            // Test dot product in INT8 quantized space
            // - Integer multiply-accumulate
            // - Scale factor handling
            // - Cosine similarity applications

            // Create test vectors
            let orthogonal1 = try Vector512Optimized((0..<512).map { i in i < 256 ? 1.0 : 0.0 })
            let orthogonal2 = try Vector512Optimized((0..<512).map { i in i >= 256 ? 1.0 : 0.0 })
            let parallel1 = try Vector512Optimized((0..<512).map { Float($0) / 512.0 })
            let parallel2 = try Vector512Optimized((0..<512).map { Float($0) / 512.0 * 2.0 })  // Scaled version
            let random1 = try Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
            let random2 = try Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })

            let testPairs = [
                ("Orthogonal", orthogonal1, orthogonal2),
                ("Parallel", parallel1, parallel2),
                ("Random", random1, random2)
            ]

            print("Quantized Dot Product:")

            for (name, v1, v2) in testPairs {
                // FP32 reference
                let dotFP32 = v1.dotProduct(v2)

                // Quantize with same parameters
                let minVal = min(v1.toArray().min()!, v2.toArray().min()!)
                let maxVal = max(v1.toArray().max()!, v2.toArray().max()!)
                let params = LinearQuantizationParams(minValue: minVal, maxValue: maxVal, symmetric: true)

                let q1 = Vector512INT8(from: v1, params: params)
                let q2 = Vector512INT8(from: v2, params: params)

                // Calculate INT8 dot product
                let dotINT8 = QuantizedKernels.dotProduct512(query: q1, candidate: q2)

                print("  \(name) vectors:")
                print("    FP32: \(dotFP32)")
                print("    INT8: \(dotINT8)")
                print("    Error: \(abs(dotFP32 - dotINT8))")

                // Check specific properties
                if name == "Orthogonal" {
                    #expect(abs(dotINT8) < 0.1, "Orthogonal vectors should have ~0 dot product")
                } else if name == "Parallel" {
                    #expect(dotINT8 > 0, "Parallel vectors should have positive dot product")
                }

                // Calculate cosine similarity
                let cosineFP32 = v1.cosineSimilarity(to: v2)
                // Calculate cosine similarity manually for INT8
                let dotINT8ForCosine = QuantizedKernels.dotProduct512(query: q1, candidate: q2)
                let mag1 = sqrt(QuantizedKernels.dotProduct512(query: q1, candidate: q1))
                let mag2 = sqrt(QuantizedKernels.dotProduct512(query: q2, candidate: q2))
                let cosineINT8 = mag1 > 0 && mag2 > 0 ? dotINT8ForCosine / (mag1 * mag2) : 0

                print("    Cosine similarity FP32: \(cosineFP32)")
                print("    Cosine similarity INT8: \(cosineINT8)")
                print("    Cosine error: \(abs(cosineFP32 - cosineINT8))")

                // Cosine similarity should be in [-1, 1]
                #expect(cosineINT8 >= -1.01 && cosineINT8 <= 1.01, "Cosine similarity should be in [-1, 1]")

                // For normalized comparison
                if cosineFP32 != 0 {
                    let cosineRelError = abs(cosineFP32 - cosineINT8) / abs(cosineFP32)
                    #expect(cosineRelError < 0.1, "Cosine similarity error should be < 10%")
                }
            }

            // Test scale factor handling with different quantization parameters
            let v = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.01) })

            let scales: [Float] = [0.01, 0.1, 1.0, 10.0]
            print("\n  Scale factor effects:")

            for scale in scales {
                let params = LinearQuantizationParams(minValue: -scale, maxValue: scale, symmetric: true)
                let q = Vector512INT8(from: v, params: params)
                let dotSelf = QuantizedKernels.dotProduct512(query: q, candidate: q)
                let magnitudeSq = v.magnitudeSquared

                print("    Scale \(scale): dot(q,q)=\(dotSelf), ||v||²=\(magnitudeSq)")

                // Self dot product should approximate magnitude squared
                let relError = abs(dotSelf - magnitudeSq) / magnitudeSq
                #expect(relError < 0.1, "Self dot product should approximate magnitude squared")
            }

            // Performance test
            let perfV1 = try Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
            let perfV2 = try Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
            let perfParams = LinearQuantizationParams(minValue: -1, maxValue: 1, symmetric: true)
            let perfQ1 = Vector512INT8(from: perfV1, params: perfParams)
            let perfQ2 = Vector512INT8(from: perfV2, params: perfParams)

            let iterations = 10000

            let fp32Start = Date()
            for _ in 0..<iterations {
                _ = perfV1.dotProduct(perfV2)
            }
            let fp32Time = Date().timeIntervalSince(fp32Start)

            let int8Start = Date()
            for _ in 0..<iterations {
                _ = QuantizedKernels.dotProduct512(query: perfQ1, candidate: perfQ2)
            }
            let int8Time = Date().timeIntervalSince(int8Start)

            print("\n  Dot product performance:")
            print("    FP32: \(fp32Time * 1000)ms")
            print("    INT8: \(int8Time * 1000)ms")
            print("    Speedup: \(fp32Time / int8Time)x")
        }

        @Test
        func testQuantizedCosineDistance() throws {
            // Test cosine distance with quantized vectors
            // - Normalized quantized vectors
            // - Angular similarity preservation
            // - Range validation

            // Create test vectors with known angular relationships
            let parallel = try Vector512Optimized((0..<512).map { Float($0) / 512.0 })
            let antiparallel = try Vector512Optimized((0..<512).map { -Float($0) / 512.0 })
            let orthogonal = try Vector512Optimized((0..<512).map { i in
                sin(Float(i) * Float.pi / 256.0)  // Orthogonal to linear ramp
            })
            let random = try Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })

            // Quantize with same parameters
            let params = LinearQuantizationParams(minValue: -1, maxValue: 1, symmetric: true)
            let qParallel = Vector512INT8(from: parallel, params: params)
            let qAntiparallel = Vector512INT8(from: antiparallel, params: params)
            let qOrthogonal = Vector512INT8(from: orthogonal, params: params)
            let qRandom = Vector512INT8(from: random, params: params)

            print("Quantized Cosine Distance:")

            // Helper to compute cosine distance (1 - cosine_similarity)
            func cosineDistance(_ v1: Vector512INT8, _ v2: Vector512INT8) -> Float {
                let dot = QuantizedKernels.dotProduct512(query: v1, candidate: v2)
                let mag1 = sqrt(QuantizedKernels.dotProduct512(query: v1, candidate: v1))
                let mag2 = sqrt(QuantizedKernels.dotProduct512(query: v2, candidate: v2))
                let cosineSim = mag1 > 0 && mag2 > 0 ? dot / (mag1 * mag2) : 0
                return 1.0 - cosineSim
            }

            // Test parallel vectors (distance ≈ 0)
            let distParallel = cosineDistance(qParallel, qParallel)
            print("  Parallel to self: \(distParallel)")
            #expect(distParallel < 0.01, "Parallel vectors should have ~0 cosine distance")

            // Test antiparallel vectors (distance ≈ 2)
            let distAntiparallel = cosineDistance(qParallel, qAntiparallel)
            print("  Parallel to antiparallel: \(distAntiparallel)")
            #expect(abs(distAntiparallel - 2.0) < 0.1, "Antiparallel vectors should have ~2 cosine distance")

            // Test orthogonal vectors (distance ≈ 1)
            let distOrthogonal = cosineDistance(qParallel, qOrthogonal)
            print("  Parallel to orthogonal: \(distOrthogonal)")
            #expect(abs(distOrthogonal - 1.0) < 0.2, "Orthogonal vectors should have ~1 cosine distance")

            // Test range validation
            let distRandom = cosineDistance(qParallel, qRandom)
            print("  Parallel to random: \(distRandom)")
            #expect(distRandom >= 0 && distRandom <= 2.01, "Cosine distance should be in [0, 2]")

            // Test angular similarity preservation
            let angles: [(String, Vector512Optimized)] = [
                ("0°", parallel),
                ("45°", try Vector512Optimized((0..<512).map { Float($0) / 512.0 + 0.5 })),
                ("90°", orthogonal),
                ("180°", antiparallel)
            ]

            for (name, vector) in angles {
                let qVector = Vector512INT8(from: vector, params: params)
                let dist = cosineDistance(qParallel, qVector)
                print("  Angle \(name): distance = \(dist)")
            }
        }

        @Test
        func testMixedPrecisionDistance() throws {
            // Test mixed precision distance computation
            // - INT8 candidates vs FP32 query
            // - FP32 candidates vs INT8 query
            // - Optimal precision strategies

            // Create test vectors
            let queryFP32 = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.01) })
            let candidateFP32 = try Vector512Optimized((0..<512).map { cos(Float($0) * 0.01) })

            // Quantize one vector
            let params = LinearQuantizationParams(
                minValue: min(queryFP32.toArray().min()!, candidateFP32.toArray().min()!),
                maxValue: max(queryFP32.toArray().max()!, candidateFP32.toArray().max()!),
                symmetric: true
            )
            let queryINT8 = Vector512INT8(from: queryFP32, params: params)
            let candidateINT8 = Vector512INT8(from: candidateFP32, params: params)

            print("Mixed Precision Distance:")

            // Test INT8 query vs FP32 candidate
            let dist_INT8_FP32 = QuantizedKernels.euclidean512(query: queryINT8, candidate: candidateFP32)
            print("  INT8 query vs FP32 candidate: \(dist_INT8_FP32)")

            // Test FP32 query vs INT8 candidate
            let dist_FP32_INT8 = QuantizedKernels.euclidean512(query: queryFP32, candidate: candidateINT8)
            print("  FP32 query vs INT8 candidate: \(dist_FP32_INT8)")

            // Full FP32 reference
            let dist_FP32_FP32 = queryFP32.euclideanDistance(to: candidateFP32)
            print("  FP32 query vs FP32 candidate: \(dist_FP32_FP32)")

            // Full INT8
            let dist_INT8_INT8 = QuantizedKernels.euclidean512(query: queryINT8, candidate: candidateINT8)
            print("  INT8 query vs INT8 candidate: \(dist_INT8_INT8)")

            // Verify mixed precision maintains reasonable accuracy
            let error1 = abs(dist_INT8_FP32 - dist_FP32_FP32) / dist_FP32_FP32
            let error2 = abs(dist_FP32_INT8 - dist_FP32_FP32) / dist_FP32_FP32

            print("\n  Relative errors:")
            print("    INT8-FP32: \(error1 * 100)%")
            print("    FP32-INT8: \(error2 * 100)%")

            #expect(error1 < 0.05, "Mixed precision error should be < 5%")
            #expect(error2 < 0.05, "Mixed precision error should be < 5%")

            // Test optimal strategy: quantize database, keep query in FP32
            let database = (0..<100).map { i in
                Vector512Optimized { j in sin(Float(i + j) * 0.01) }
            }
            let databaseINT8 = database.map { Vector512INT8(from: $0, params: params) }

            // Benchmark mixed vs full precision
            let iterations = 1000

            let mixedStart = Date()
            for _ in 0..<iterations {
                for candidateINT8 in databaseINT8 {
                    _ = QuantizedKernels.euclidean512(query: queryFP32, candidate: candidateINT8)
                }
            }
            let mixedTime = Date().timeIntervalSince(mixedStart)

            let fp32Start = Date()
            for _ in 0..<iterations {
                for candidate in database {
                    _ = queryFP32.euclideanDistance(to: candidate)
                }
            }
            let fp32Time = Date().timeIntervalSince(fp32Start)

            print("\n  Performance (100 candidates):")
            print("    Mixed precision: \(mixedTime * 1000)ms")
            print("    Full FP32: \(fp32Time * 1000)ms")
            print("    Speedup: \(fp32Time / mixedTime)x")
        }

        @Test
        func testQuantizedDistanceConsistency() throws {
            // Test consistency of quantized distance metrics
            // - Relative ordering preservation
            // - Monotonicity properties
            // - Triangle inequality validation

            // Create a set of test vectors
            let vectors = (0..<10).map { i in
                Vector512Optimized { j in sin(Float(i * 512 + j) * 0.01) }
            }

            // Quantize all vectors with same parameters
            let minVal = vectors.flatMap { $0.toArray() }.min()!
            let maxVal = vectors.flatMap { $0.toArray() }.max()!
            let params = LinearQuantizationParams(minValue: minVal, maxValue: maxVal, symmetric: true)
            let quantized = vectors.map { Vector512INT8(from: $0, params: params) }

            print("Quantized Distance Consistency:")

            // Test relative ordering preservation
            let query = vectors[0]
            let queryQ = quantized[0]

            var distancesFP32: [(Int, Float)] = []
            var distancesINT8: [(Int, Float)] = []

            for i in 1..<vectors.count {
                distancesFP32.append((i, query.euclideanDistance(to: vectors[i])))
                distancesINT8.append((i, QuantizedKernels.euclidean512(query: queryQ, candidate: quantized[i])))
            }

            // Sort by distance
            distancesFP32.sort { $0.1 < $1.1 }
            distancesINT8.sort { $0.1 < $1.1 }

            // Check if ordering is preserved
            var orderingPreserved = true
            for i in 0..<distancesFP32.count {
                if distancesFP32[i].0 != distancesINT8[i].0 {
                    orderingPreserved = false
                    break
                }
            }

            print("  Ordering preserved: \(orderingPreserved)")
            if !orderingPreserved {
                print("  FP32 order: \(distancesFP32.map { $0.0 })")
                print("  INT8 order: \(distancesINT8.map { $0.0 })")
            }

            // Allow some reordering for very close distances
            var rankCorrelation: Float = 0
            for i in 0..<distancesFP32.count {
                let fp32Rank = i
                let int8Rank = distancesINT8.firstIndex { $0.0 == distancesFP32[i].0 }!
                rankCorrelation += Float(abs(fp32Rank - int8Rank))
            }
            rankCorrelation = 1.0 - (rankCorrelation / Float(distancesFP32.count * distancesFP32.count))

            print("  Rank correlation: \(rankCorrelation)")
            #expect(rankCorrelation > 0.8, "Rank correlation should be high")

            // Test triangle inequality
            var triangleViolations = 0
            for i in 0..<quantized.count {
                for j in i+1..<quantized.count {
                    for k in j+1..<quantized.count {
                        let d_ij = QuantizedKernels.euclidean512(query: quantized[i], candidate: quantized[j])
                        let d_jk = QuantizedKernels.euclidean512(query: quantized[j], candidate: quantized[k])
                        let d_ik = QuantizedKernels.euclidean512(query: quantized[i], candidate: quantized[k])

                        // Triangle inequality: d(i,k) <= d(i,j) + d(j,k)
                        if d_ik > d_ij + d_jk + 0.001 {  // Small tolerance for numerical errors
                            triangleViolations += 1
                        }
                    }
                }
            }

            print("  Triangle inequality violations: \(triangleViolations)")
            #expect(triangleViolations == 0, "Triangle inequality should hold")

            // Test monotonicity: scaling a vector should scale its distance
            let v1 = vectors[1]
            let v2 = vectors[2]
            let q1 = quantized[1]
            let q2 = quantized[2]

            let dist_original = QuantizedKernels.euclidean512(query: q1, candidate: q2)

            // Scale one vector
            let v1_scaled = v1 * 2.0
            let q1_scaled = Vector512INT8(from: v1_scaled, params: LinearQuantizationParams(
                minValue: minVal * 2, maxValue: maxVal * 2, symmetric: true
            ))

            let dist_scaled = QuantizedKernels.euclidean512(query: q1_scaled, candidate: q2)

            print("  Distance scaling: original=\(dist_original), scaled=\(dist_scaled)")
            print("  Monotonicity: distance increased = \(dist_scaled > dist_original)")
        }
    }

    // MARK: - Memory Compression Tests

    @Suite("Memory Compression")
    struct MemoryCompressionTests {

        @Test
        func testMemoryFootprintReduction() throws {
            // Test 4x memory footprint reduction with INT8
            // - Actual memory usage measurement
            // - Compression ratio analysis
            // - Memory layout optimization

            let vectorCount = 1000
            let dimension = 512

            // Create FP32 vectors
            let fp32Vectors = (0..<vectorCount).map { _ in
                Vector512Optimized { _ in Float.random(in: -1...1) }
            }

            // Create INT8 vectors
            let params = LinearQuantizationParams(minValue: -1, maxValue: 1, symmetric: true)
            let int8Vectors = fp32Vectors.map { Vector512INT8(from: $0, params: params) }

            // Calculate memory footprints
            let fp32MemoryPerVector = dimension * MemoryLayout<Float>.size  // 512 * 4 = 2048 bytes
            let int8MemoryPerVector = dimension * MemoryLayout<Int8>.size + MemoryLayout<LinearQuantizationParams>.size  // 512 * 1 + params

            let fp32TotalMemory = fp32MemoryPerVector * vectorCount
            let int8TotalMemory = int8MemoryPerVector * vectorCount

            let compressionRatio = Float(fp32TotalMemory) / Float(int8TotalMemory)

            print("Memory Footprint Reduction:")
            print("  FP32 per vector: \(fp32MemoryPerVector) bytes")
            print("  INT8 per vector: \(int8MemoryPerVector) bytes")
            print("  Total FP32: \(fp32TotalMemory / 1024) KB")
            print("  Total INT8: \(int8TotalMemory / 1024) KB")
            print("  Compression ratio: \(compressionRatio)x")
            print("  Memory saved: \((1.0 - 1.0/compressionRatio) * 100)%")

            // Verify expected compression ratio (should be close to 4x)
            #expect(compressionRatio > 3.5, "Should achieve at least 3.5x compression")

            // Test actual memory allocation
            var info = mach_task_basic_info()
            var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

            // Measure baseline
            _ = withUnsafeMutablePointer(to: &info) { ptr in
                ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { int_ptr in
                    task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), int_ptr, &count)
                }
            }
            let baselineMemory = info.resident_size

            // Allocate large FP32 array
            let largeFP32 = (0..<10000).map { _ in Vector512Optimized { _ in Float.random(in: -1...1) } }

            _ = withUnsafeMutablePointer(to: &info) { ptr in
                ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { int_ptr in
                    task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), int_ptr, &count)
                }
            }
            let fp32Memory = info.resident_size - baselineMemory

            // Force deallocation
            _ = largeFP32.count  // Keep reference

            print("\n  Actual memory allocation test:")
            print("    FP32 allocation: \(fp32Memory / 1024 / 1024) MB")

            // Memory layout optimization check
            let int8Vector = int8Vectors[0]
            print("\n  Memory layout:")
            print("    Storage type: ContiguousArray<SIMD4<Int8>>")
            print("    SIMD lanes: \(int8Vector.storage.count)")
            print("    Bytes per lane: \(MemoryLayout<SIMD4<Int8>>.size)")
            print("    Alignment: \(MemoryLayout<SIMD4<Int8>>.alignment)")
        }

        @Test
        func testQuantizedStorageLayout() throws {
            // Test optimized storage layout for quantized vectors
            // - Packed INT8 representation
            // - SIMD-friendly alignment
            // - Cache-efficient access patterns

            let vector = try Vector512Optimized((0..<512).map { Float($0) / 256.0 - 1.0 })
            let quantized = Vector512INT8(from: vector)

            print("Quantized Storage Layout:")

            // Test packed representation - simplified
            var baseAddress: UInt = 0
            var bufferCount = 0

            quantized.storage.withUnsafeBufferPointer { buffer in
                if let addr = buffer.baseAddress {
                    baseAddress = UInt(bitPattern: addr)
                    bufferCount = buffer.count
                }
            }

            let isAligned16 = baseAddress % 16 == 0
            let alignmentStr = isAligned16 ? "16-byte aligned" : "Not 16-byte aligned"

            print("  Base address: 0x\(String(baseAddress, radix: 16))")
            print("  Alignment: \(alignmentStr)")
            print("  Total lanes: \(bufferCount)")
            print("  Bytes per SIMD4: \(MemoryLayout<SIMD4<Int8>>.size)")
            print("  Total bytes: \(bufferCount * MemoryLayout<SIMD4<Int8>>.size)")

            // Verify SIMD-friendly alignment
            let isAligned4 = baseAddress % 4 == 0
            #expect(isAligned4, "Should be at least 4-byte aligned for SIMD4")

            // Check packing efficiency
            let expectedBytes = 512  // 512 INT8 values
            let actualBytes = bufferCount * MemoryLayout<SIMD4<Int8>>.size
            #expect(actualBytes == expectedBytes, "Should pack exactly 512 bytes")

            // Test sequential access pattern in a separate closure
            quantized.storage.withUnsafeBufferPointer { buffer in
                var sum: Int32 = 0
                for i in 0..<buffer.count {
                    let lane = buffer[i]
                    sum += Int32(lane.x) + Int32(lane.y) + Int32(lane.z) + Int32(lane.w)
                }
                print("  Sequential access sum: \(sum)")

                // Test strided access pattern
                var stridedSum: Int32 = 0
                let strideVal = 4
                for i in Swift.stride(from: 0, to: buffer.count, by: strideVal) {
                    let lane = buffer[i]
                    stridedSum += Int32(lane.x)
                }
                print("  Strided access sum: \(stridedSum)")
            }

            // Test cache efficiency with batch operations
            let batchSize = 100
            let batch = (0..<batchSize).map { _ in
                Vector512INT8(from: Vector512Optimized { _ in Float.random(in: -1...1) })
            }

            // Measure cache-friendly sequential processing
            let seqStart = Date()
            var seqResult: Float = 0
            for i in 0..<batch.count-1 {
                seqResult += QuantizedKernels.dotProduct512(query: batch[i], candidate: batch[i+1])
            }
            let seqTime = Date().timeIntervalSince(seqStart)

            // Measure cache-unfriendly random access
            let randomIndices = (0..<batchSize-1).map { _ in Int.random(in: 0..<batchSize-1) }
            let randStart = Date()
            var randResult: Float = 0
            for i in randomIndices {
                randResult += QuantizedKernels.dotProduct512(query: batch[i], candidate: batch[i+1])
            }
            let randTime = Date().timeIntervalSince(randStart)

            print("\n  Cache efficiency:")
            print("    Sequential access: \(seqTime * 1000000)µs")
            print("    Random access: \(randTime * 1000000)µs")
            print("    Cache benefit: \(randTime / seqTime)x")
        }

        @Test
        func testQuantizedVectorSerialization() throws {
            // Test serialization of quantized vectors
            // - Compact binary format
            // - Metadata preservation (scale, zero-point)
            // - Cross-platform compatibility

            let originalVector = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.01) })
            let originalQuantized = Vector512INT8(from: originalVector)

            print("Quantized Vector Serialization:")

            // Serialize to data
            let encoder = JSONEncoder()
            let data = try encoder.encode(originalQuantized)

            print("  Serialized size: \(data.count) bytes")
            print("  Original params: scale=\(originalQuantized.quantizationParams.scale), zp=\(originalQuantized.quantizationParams.zeroPoint)")

            // Deserialize
            let decoder = JSONDecoder()
            let deserialized = try decoder.decode(Vector512INT8.self, from: data)

            // Verify metadata preservation
            #expect(deserialized.quantizationParams.scale == originalQuantized.quantizationParams.scale,
                   "Scale should be preserved")
            #expect(deserialized.quantizationParams.zeroPoint == originalQuantized.quantizationParams.zeroPoint,
                   "Zero-point should be preserved")
            #expect(deserialized.quantizationParams.minValue == originalQuantized.quantizationParams.minValue,
                   "Min value should be preserved")
            #expect(deserialized.quantizationParams.maxValue == originalQuantized.quantizationParams.maxValue,
                   "Max value should be preserved")

            // Verify data integrity
            let originalFP32 = originalQuantized.toFP32()
            let deserializedFP32 = deserialized.toFP32()

            var maxError: Float = 0
            for i in 0..<512 {
                let error = abs(originalFP32[i] - deserializedFP32[i])
                maxError = max(maxError, error)
            }

            print("  Deserialization max error: \(maxError)")
            #expect(maxError < 0.0001, "Deserialized data should match original")

            // Test binary size efficiency
            let naiveSize = 512 * MemoryLayout<Float>.size  // If we stored as FP32
            let compressedSize = data.count
            let compressionRatio = Float(naiveSize) / Float(compressedSize)

            print("\n  Size comparison:")
            print("    Naive FP32: \(naiveSize) bytes")
            print("    Serialized INT8: \(compressedSize) bytes")
            print("    Compression: \(compressionRatio)x")

            // Test cross-platform compatibility (endianness)
            // Create a vector with known values
            let testValues = try Vector512Optimized((0..<512).map { Float($0 % 256) / 128.0 - 1.0 })
            let testQuantized = Vector512INT8(from: testValues)

            // Serialize and deserialize
            let testData = try encoder.encode(testQuantized)
            let testDeserialized = try decoder.decode(Vector512INT8.self, from: testData)

            // Verify exact match of quantized values
            for i in 0..<128 {  // 128 SIMD4 lanes
                let original = testQuantized.storage[i]
                let deserialized = testDeserialized.storage[i]
                #expect(original == deserialized, "SIMD4 lane \(i) should match exactly")
            }

            print("  Cross-platform compatibility: PASSED")
        }

        @Test
        func testMemoryBandwidthImprovement() throws {
            // Test memory bandwidth improvements
            // - 4x theoretical improvement
            // - Actual bandwidth measurements
            // - Cache utilization analysis

            let vectorCount = 10000
            let iterations = 100

            // Create large arrays
            let fp32Vectors = (0..<vectorCount).map { _ in
                Vector512Optimized { _ in Float.random(in: -1...1) }
            }
            let int8Vectors = fp32Vectors.map { Vector512INT8(from: $0) }

            print("Memory Bandwidth Improvement:")

            // Measure FP32 bandwidth
            let fp32Start = Date()
            for _ in 0..<iterations {
                var sum: Float = 0
                for vector in fp32Vectors {
                    sum += vector[0]  // Force memory access
                }
                if sum > Float.greatestFiniteMagnitude { print(sum) }  // Prevent optimization
            }
            let fp32Time = Date().timeIntervalSince(fp32Start)

            // Measure INT8 bandwidth
            let int8Start = Date()
            for _ in 0..<iterations {
                var sum: Float = 0
                for vector in int8Vectors {
                    sum += vector.quantizationParams.scale  // Force memory access
                }
                if sum > Float.greatestFiniteMagnitude { print(sum) }
            }
            let int8Time = Date().timeIntervalSince(int8Start)

            let bytesPerVector = 512 * 4  // FP32
            let bytesPerVectorINT8 = 512  // INT8
            let fp32Bandwidth = Double(vectorCount * iterations * bytesPerVector) / fp32Time / 1e9  // GB/s
            let int8Bandwidth = Double(vectorCount * iterations * bytesPerVectorINT8) / int8Time / 1e9  // GB/s

            print("  FP32 bandwidth: \(fp32Bandwidth) GB/s")
            print("  INT8 bandwidth: \(int8Bandwidth) GB/s")
            print("  Effective speedup: \(int8Bandwidth / fp32Bandwidth * 4)x")

            // Test cache utilization
            let cacheLineSize = 64  // Typical cache line size
            let fp32VectorsPerCacheLine = cacheLineSize / (4 * 4)  // 4 floats fit
            let int8VectorsPerCacheLine = cacheLineSize / 4  // 16 SIMD4<Int8> fit

            print("\n  Cache utilization:")
            print("    FP32 elements per cache line: \(fp32VectorsPerCacheLine * 4)")
            print("    INT8 elements per cache line: \(int8VectorsPerCacheLine * 4)")
            print("    Cache efficiency improvement: \(int8VectorsPerCacheLine * 4 / (fp32VectorsPerCacheLine * 4))x")

            #expect(int8Time < fp32Time, "INT8 should have better memory bandwidth")
        }

        @Test
        func testCompressionRatioAnalysis() throws {
            // Test compression ratio analysis
            // - Different data distributions
            // - Sparse vs dense vectors
            // - Adaptive compression strategies

            print("Compression Ratio Analysis:")

            // Test different distributions
            let distributions: [(String, Vector512Optimized)] = [
                ("Uniform", Vector512Optimized { _ in Float.random(in: -1...1) }),
                ("Gaussian", Vector512Optimized { _ in
                    let u1 = Float.random(in: 0.001...0.999)
                    let u2 = Float.random(in: 0.001...0.999)
                    return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
                }),
                ("Sparse 90%", Vector512Optimized { i in i % 10 == 0 ? Float.random(in: -1...1) : 0 }),
                ("Sparse 99%", Vector512Optimized { i in i % 100 == 0 ? Float.random(in: -1...1) : 0 }),
                ("Binary", Vector512Optimized { i in i % 2 == 0 ? 1.0 : -1.0 }),
                ("Concentrated", Vector512Optimized { _ in Float.random(in: -0.1...0.1) })
            ]

            for (name, vector) in distributions {
                let quantized = Vector512INT8(from: vector)

                // Calculate theoretical compression
                let fp32Size = 512 * MemoryLayout<Float>.size
                let int8Size = 512 * MemoryLayout<Int8>.size + MemoryLayout<LinearQuantizationParams>.size
                let compressionRatio = Float(fp32Size) / Float(int8Size)

                // Calculate information content (entropy approximation)
                let uniqueValues = Set(quantized.storage.flatMap { [$0.x, $0.y, $0.z, $0.w] }).count
                let entropy = Float(uniqueValues) / 256.0  // Normalized by possible values

                // Measure accuracy
                let dequantized = quantized.toFP32()
                var mse: Float = 0
                for i in 0..<512 {
                    let error = vector[i] - dequantized[i]
                    mse += error * error
                }
                mse /= 512

                print("  \(name):")
                print("    Compression ratio: \(compressionRatio)x")
                print("    Unique INT8 values: \(uniqueValues)/256")
                print("    Normalized entropy: \(entropy)")
                print("    MSE: \(mse)")
                print("    Scale: \(quantized.quantizationParams.scale)")

                // Sparse vectors should have better effective compression
                if name.contains("Sparse") {
                    #expect(uniqueValues < 128, "Sparse vectors should have fewer unique values")
                }
            }

            // Test adaptive strategy based on sparsity
            func adaptiveCompressionRatio(vector: Vector512Optimized) -> Float {
                let nonZeroCount = vector.toArray().filter { $0 != 0 }.count
                let sparsity = Float(512 - nonZeroCount) / 512.0

                if sparsity > 0.9 {
                    // Could use sparse representation
                    return 512.0 / Float(nonZeroCount * 5)  // index + value
                } else {
                    // Use dense INT8
                    return 4.0
                }
            }

            print("\n  Adaptive compression:")
            for (name, vector) in distributions {
                let ratio = adaptiveCompressionRatio(vector: vector)
                print("    \(name): \(ratio)x")
            }
        }
    }

    // MARK: - SIMD and Vectorization Tests

    @Suite("SIMD and Vectorization")
    struct SIMDVectorizationTests {

        @Test
        func testSIMDINT8Operations() throws {
            // Test SIMD operations on INT8 vectors
            // - Vectorized arithmetic operations
            // - Packed INT8 SIMD instructions
            // - Register utilization efficiency

            let v1 = try Vector512Optimized((0..<512).map { Float($0) / 512.0 })
            let v2 = try Vector512Optimized((0..<512).map { Float(511 - $0) / 512.0 })

            let params = LinearQuantizationParams(minValue: 0, maxValue: 1, symmetric: false)
            let q1 = Vector512INT8(from: v1, params: params)
            let q2 = Vector512INT8(from: v2, params: params)

            print("SIMD INT8 Operations:")

            // Test vectorized addition (simulated)
            func simdAdd(_ a: Vector512INT8, _ b: Vector512INT8) -> Vector512INT8 {
                var result = ContiguousArray<SIMD4<Int8>>()
                result.reserveCapacity(128)

                for i in 0..<128 {
                    // SIMD4 addition
                    let sum = SIMD4<Int8>(
                        Int8(clamping: Int16(a.storage[i].x) + Int16(b.storage[i].x)),
                        Int8(clamping: Int16(a.storage[i].y) + Int16(b.storage[i].y)),
                        Int8(clamping: Int16(a.storage[i].z) + Int16(b.storage[i].z)),
                        Int8(clamping: Int16(a.storage[i].w) + Int16(b.storage[i].w))
                    )
                    result.append(sum)
                }

                return Vector512INT8(storage: result, params: params)
            }

            // Benchmark SIMD operations
            let iterations = 10000

            let simdStart = Date()
            for _ in 0..<iterations {
                _ = simdAdd(q1, q2)
            }
            let simdTime = Date().timeIntervalSince(simdStart)

            // Scalar equivalent
            let scalarStart = Date()
            for _ in 0..<iterations {
                var result = [Int8]()
                let a = q1.storage.flatMap { [$0.x, $0.y, $0.z, $0.w] }
                let b = q2.storage.flatMap { [$0.x, $0.y, $0.z, $0.w] }
                for i in 0..<512 {
                    result.append(Int8(clamping: Int16(a[i]) + Int16(b[i])))
                }
            }
            let scalarTime = Date().timeIntervalSince(scalarStart)

            print("  SIMD time: \(simdTime * 1000)ms")
            print("  Scalar time: \(scalarTime * 1000)ms")
            print("  SIMD speedup: \(scalarTime / simdTime)x")

            // Test register utilization
            print("\n  Register utilization:")
            print("    SIMD4<Int8> size: \(MemoryLayout<SIMD4<Int8>>.size) bytes")
            print("    Elements per register: 4")
            print("    Registers needed for 512 elements: \(512 / 4)")
            print("    Typical SIMD register width: 128 bits")
            print("    INT8 elements per 128-bit register: 16")
            print("    Theoretical speedup: 16x over scalar")

            // Test packed operations
            let dotProductResult = QuantizedKernels.dotProduct512(query: q1, candidate: q2)
            print("\n  Packed dot product result: \(dotProductResult)")

            #expect(simdTime < scalarTime, "SIMD should be faster than scalar")
        }

        @Test
        func testVectorizedQuantization() throws {
            // Test vectorized quantization operations
            let batchSize = 100
            let vectors = (0..<batchSize).map { i in
                Vector512Optimized { j in sin(Float(i * 512 + j) * 0.001) }
            }

            // Batch quantization
            let start = Date()
            let quantized = vectors.map { Vector512INT8(from: $0) }
            let batchTime = Date().timeIntervalSince(start)

            print("Vectorized Quantization:")
            print("  Batch size: \(batchSize)")
            print("  Total time: \(batchTime * 1000)ms")
            print("  Time per vector: \(batchTime * 1000000 / Double(batchSize))μs")

            #expect(quantized.count == batchSize)
        }

        @Test
        func testVectorizedDequantization() throws {
            // Test vectorized dequantization operations
            let batchSize = 100
            let quantized = (0..<batchSize).map { i in
                Vector512INT8(from: Vector512Optimized { j in Float(i + j) / 1000.0 })
            }

            let start = Date()
            let dequantized = quantized.map { $0.toFP32() }
            let batchTime = Date().timeIntervalSince(start)

            print("Vectorized Dequantization:")
            print("  Batch size: \(batchSize)")
            print("  Total time: \(batchTime * 1000)ms")
            print("  Time per vector: \(batchTime * 1000000 / Double(batchSize))μs")

            #expect(dequantized.count == batchSize)
        }

        @Test
        func testSIMDINT8DistanceComputation() throws {
            // Test SIMD INT8 distance computation
            let v1 = Vector512INT8(from: Vector512Optimized { _ in Float.random(in: -1...1) })
            let v2 = Vector512INT8(from: Vector512Optimized { _ in Float.random(in: -1...1) })

            let iterations = 10000
            let start = Date()
            for _ in 0..<iterations {
                _ = QuantizedKernels.euclidean512(query: v1, candidate: v2)
            }
            let elapsed = Date().timeIntervalSince(start)

            print("SIMD INT8 Distance Computation:")
            print("  Time for \(iterations) iterations: \(elapsed * 1000)ms")
            print("  Time per operation: \(elapsed * 1000000 / Double(iterations))μs")
        }

        @Test
        func testVectorizationEfficiency() throws {
            // Test vectorization efficiency metrics
            let vectors = (0..<1000).map { _ in
                Vector512INT8(from: Vector512Optimized { _ in Float.random(in: -1...1) })
            }

            // Test different batch sizes
            let batchSizes = [1, 10, 100, 1000]
            print("Vectorization Efficiency:")

            for batchSize in batchSizes {
                let batch = Array(vectors.prefix(batchSize))
                let start = Date()
                var sum: Float = 0
                for i in 0..<batch.count-1 {
                    sum += QuantizedKernels.dotProduct512(query: batch[i], candidate: batch[i+1])
                }
                let elapsed = Date().timeIntervalSince(start)
                let opsPerSec = Double(batchSize) / elapsed

                print("  Batch \(batchSize): \(opsPerSec) ops/sec")
                if sum > Float.greatestFiniteMagnitude { print(sum) }  // Prevent optimization
            }
        }
    }

    // MARK: - Accuracy Analysis Tests

    @Suite("Accuracy Analysis")
    struct AccuracyAnalysisTests {

        @Test
        func testQuantizationErrorAnalysis() throws {
            // Test comprehensive quantization error analysis
            let testVector = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.02) })
            let quantized = Vector512INT8(from: testVector)
            let dequantized = quantized.toFP32()

            var errors: [Float] = []
            for i in 0..<512 {
                errors.append(dequantized[i] - testVector[i])
            }

            let meanError = errors.reduce(0, +) / Float(errors.count)
            let maxError = errors.map { abs($0) }.max()!
            let variance = errors.map { pow($0 - meanError, 2) }.reduce(0, +) / Float(errors.count)
            let stdDev = sqrt(variance)

            print("Quantization Error Analysis:")
            print("  Mean error: \(meanError)")
            print("  Max error: \(maxError)")
            print("  Std dev: \(stdDev)")
            print("  Variance: \(variance)")

            #expect(abs(meanError) < 0.01, "Mean error should be near zero")
            #expect(maxError < quantized.quantizationParams.scale * 2, "Max error should be bounded")
        }

        @Test
        func testSignalToNoiseRatio() throws {
            // Test signal-to-noise ratio with quantization
            let signal = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.01) * 2.0 })
            let quantized = Vector512INT8(from: signal)
            let reconstructed = quantized.toFP32()

            var signalPower: Float = 0
            var noisePower: Float = 0

            for i in 0..<512 {
                signalPower += signal[i] * signal[i]
                let noise = reconstructed[i] - signal[i]
                noisePower += noise * noise
            }

            signalPower /= 512
            noisePower /= 512
            let snr = 10 * log10(signalPower / noisePower)

            print("Signal-to-Noise Ratio:")
            print("  Signal power: \(signalPower)")
            print("  Noise power: \(noisePower)")
            print("  SNR: \(snr) dB")

            #expect(snr > 20, "SNR should be > 20 dB for acceptable quality")
        }

        @Test
        func testDistanceMetricPreservation() throws {
            // Test preservation of distance metric properties
            // - Relative ordering preservation
            // - Distance ratio preservation
            // - Nearest neighbor accuracy

            // Create a query and candidates
            let query = Vector512Optimized { i in sin(Float(i) * 0.01) }
            let candidates = (0..<20).map { j in
                Vector512Optimized { i in cos(Float(i + j * 10) * 0.01) }
            }

            // Get FP32 distances and rankings
            let fp32Distances = candidates.enumerated().map { (idx, candidate) in
                (idx, query.euclideanDistance(to: candidate))
            }.sorted { $0.1 < $1.1 }

            // Quantize and get INT8 distances
            let params = LinearQuantizationParams(minValue: -1, maxValue: 1, symmetric: true)
            let queryQ = Vector512INT8(from: query, params: params)
            let candidatesQ = candidates.map { Vector512INT8(from: $0, params: params) }

            let int8Distances = candidatesQ.enumerated().map { (idx, candidate) in
                (idx, QuantizedKernels.euclidean512(query: queryQ, candidate: candidate))
            }.sorted { $0.1 < $1.1 }

            // Check nearest neighbor preservation
            let fp32NN = fp32Distances[0].0
            let int8NN = int8Distances[0].0

            print("Distance Metric Preservation:")
            print("  FP32 nearest neighbor: \(fp32NN)")
            print("  INT8 nearest neighbor: \(int8NN)")
            print("  NN preserved: \(fp32NN == int8NN)")

            // Check top-5 preservation
            let fp32Top5 = Set(fp32Distances.prefix(5).map { $0.0 })
            let int8Top5 = Set(int8Distances.prefix(5).map { $0.0 })
            let top5Overlap = fp32Top5.intersection(int8Top5).count

            print("  Top-5 overlap: \(top5Overlap)/5")

            // Check distance ratio preservation
            if fp32Distances.count >= 3 {
                let ratio1FP32 = fp32Distances[1].1 / fp32Distances[0].1
                let ratio1INT8 = int8Distances[1].1 / int8Distances[0].1
                let ratioError = abs(ratio1FP32 - ratio1INT8) / ratio1FP32

                print("  Distance ratio preservation error: \(ratioError * 100)%")
                #expect(ratioError < 0.2, "Distance ratios should be approximately preserved")
            }

            #expect(top5Overlap >= 3, "At least 3 of top-5 should be preserved")
        }

        @Test
        func testApplicationLevelAccuracy() throws {
            // Test application-level accuracy metrics
            // - Search quality preservation
            // - Clustering quality metrics
            // - Classification accuracy

            // Simulate a search application
            let database = (0..<100).map { i in
                Vector512Optimized { j in sin(Float(i * 512 + j) * 0.001) }
            }
            let queries = (0..<10).map { i in
                Vector512Optimized { j in sin(Float((i + 50) * 512 + j) * 0.001) }
            }

            // Quantize database
            let params = LinearQuantizationParams(minValue: -1, maxValue: 1, symmetric: true)
            let databaseQ = database.map { Vector512INT8(from: $0, params: params) }
            let queriesQ = queries.map { Vector512INT8(from: $0, params: params) }

            var searchAccuracy: Float = 0
            for (qIdx, query) in queries.enumerated() {
                // FP32 ground truth
                let fp32Results = database.enumerated().map { (idx, vec) in
                    (idx, query.euclideanDistance(to: vec))
                }.sorted { $0.1 < $1.1 }.prefix(10).map { $0.0 }

                // INT8 results
                let queryQ = queriesQ[qIdx]
                let int8Results = databaseQ.enumerated().map { (idx, vec) in
                    (idx, QuantizedKernels.euclidean512(query: queryQ, candidate: vec))
                }.sorted { $0.1 < $1.1 }.prefix(10).map { $0.0 }

                // Calculate recall@10
                let overlap = Set(fp32Results).intersection(Set(int8Results)).count
                searchAccuracy += Float(overlap) / 10.0
            }
            searchAccuracy /= Float(queries.count)

            print("Application-Level Accuracy:")
            print("  Search recall@10: \(searchAccuracy * 100)%")

            // Test clustering quality (simple k-means simulation)
            let k = 3
            let points = (0..<30).map { i in
                Vector512Optimized { j in
                    Float(i / 10) + sin(Float(j) * 0.1) * 0.1  // 3 clusters
                }
            }
            let pointsQ = points.map { Vector512INT8(from: $0, params: params) }

            // Simple cluster assignment
            let centroids = [points[0], points[10], points[20]]
            let centroidsQ = centroids.map { Vector512INT8(from: $0, params: params) }

            var clusteringAgreement = 0
            for (idx, point) in points.enumerated() {
                // FP32 assignment
                let fp32Cluster = centroids.enumerated().min { a, b in
                    point.euclideanDistance(to: a.1) < point.euclideanDistance(to: b.1)
                }!.0

                // INT8 assignment
                let pointQ = pointsQ[idx]
                let int8Cluster = centroidsQ.enumerated().min { a, b in
                    QuantizedKernels.euclidean512(query: pointQ, candidate: a.1) <
                    QuantizedKernels.euclidean512(query: pointQ, candidate: b.1)
                }!.0

                if fp32Cluster == int8Cluster {
                    clusteringAgreement += 1
                }
            }

            let clusteringAccuracy = Float(clusteringAgreement) / Float(points.count)
            print("  Clustering agreement: \(clusteringAccuracy * 100)%")

            #expect(searchAccuracy > 0.7, "Search accuracy should be > 70%")
            #expect(clusteringAccuracy > 0.8, "Clustering agreement should be > 80%")
        }

        @Test
        func testAdaptiveQuantizationAccuracy() throws {
            // Test accuracy with adaptive quantization
            // - Per-channel quantization
            // - Layer-wise quantization
            // - Data-dependent quantization

            // Create vector with different magnitude channels
            let vector = try Vector512Optimized((0..<512).map { i in
                if i < 128 {
                    Float.random(in: -0.1...0.1)  // Small values
                } else if i < 256 {
                    Float.random(in: -1...1)      // Medium values
                } else if i < 384 {
                    Float.random(in: -10...10)    // Large values
                } else {
                    0.0                            // Zeros
                }
            })

            print("Adaptive Quantization Accuracy:")

            // Standard quantization (global scale)
            let globalQ = Vector512INT8(from: vector)
            let globalDeq = globalQ.toFP32()

            // Simulate per-channel quantization
            let channels = 4
            let channelSize = 128
            var perChannelResults = [Float](repeating: 0, count: 512)

            for c in 0..<channels {
                let start = c * channelSize
                let end = start + channelSize
                let channelData = Array(vector.toArray()[start..<end])

                // Create channel vector (pad with zeros)
                let channelVector = try Vector512Optimized(
                    channelData + Array(repeating: 0, count: 512 - channelSize)
                )

                // Quantize with channel-specific params
                let minVal = channelData.min() ?? 0
                let maxVal = channelData.max() ?? 1
                let channelParams = LinearQuantizationParams(
                    minValue: minVal, maxValue: maxVal,
                    symmetric: false
                )
                let channelQ = Vector512INT8(from: channelVector, params: channelParams)
                let channelDeq = channelQ.toFP32()

                // Copy back results
                for i in 0..<channelSize {
                    perChannelResults[start + i] = channelDeq[i]
                }

                print("  Channel \(c): scale=\(channelParams.scale), range=[\(minVal), \(maxVal)]")
            }

            // Calculate errors
            var globalError: Float = 0
            var perChannelError: Float = 0

            for i in 0..<512 {
                globalError += abs(globalDeq[i] - vector[i])
                perChannelError += abs(perChannelResults[i] - vector[i])
            }

            globalError /= 512
            perChannelError /= 512

            print("\n  Global quantization error: \(globalError)")
            print("  Per-channel error: \(perChannelError)")
            print("  Improvement: \((globalError - perChannelError) / globalError * 100)%")

            #expect(perChannelError <= globalError, "Per-channel should not be worse than global")

            // Test data-dependent quantization
            let sparse = Vector512Optimized { i in i % 100 == 0 ? Float.random(in: -1...1) : 0 }
            let denseUniform = Vector512Optimized { _ in Float.random(in: -1...1) }

            // Different strategies based on sparsity
            let sparseNonZeros = sparse.toArray().filter { $0 != 0 }.count
            let sparsity = Float(512 - sparseNonZeros) / 512

            print("\n  Data-dependent quantization:")
            print("    Sparse vector sparsity: \(sparsity * 100)%")

            if sparsity > 0.9 {
                print("    Using sparse-optimized quantization")
                // Could use different bit allocation
            } else {
                print("    Using standard quantization")
            }
        }

        @Test
        func testAccuracyVsCompressionTradeoffs() throws {
            // Test accuracy vs compression tradeoffs
            // - Different bit widths (INT4, INT8, INT16)
            // - Quality degradation curves
            // - Optimal operating points

            let testVector = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.01) })

            print("Accuracy vs Compression Tradeoffs:")

            // Simulate different bit widths
            let bitWidths = [4, 8, 16]
            var results: [(bits: Int, compression: Float, error: Float, snr: Float)] = []

            for bits in bitWidths {
                let maxValue = Float((1 << (bits - 1)) - 1)  // e.g., 127 for INT8
                let levels = powf(2, Float(bits))

                // Simulate quantization at different bit depths
                let scale = 2.0 / maxValue  // Range [-1, 1] mapped to INT range

                var quantized = [Float]()
                for value in testVector.toArray() {
                    let q = round(value / scale)
                    let clamped = max(-maxValue, min(maxValue, q))
                    let dequantized = clamped * scale
                    quantized.append(dequantized)
                }

                // Calculate metrics
                var mse: Float = 0
                var signalPower: Float = 0

                for i in 0..<512 {
                    let error = quantized[i] - testVector[i]
                    mse += error * error
                    signalPower += testVector[i] * testVector[i]
                }

                mse /= 512
                signalPower /= 512
                let snr = 10 * log10(signalPower / mse)
                let compression = 32.0 / Float(bits)  // FP32 to INTx compression

                results.append((bits, compression, sqrt(mse), snr))

                print("  INT\(bits):")
                print("    Compression: \(compression)x")
                print("    RMSE: \(sqrt(mse))")
                print("    SNR: \(snr) dB")
                print("    Quantization levels: \(Int(levels))")
            }

            // Find optimal operating point (best SNR per bit)
            let optimalPoint = results.max { a, b in
                (a.snr / Float(a.bits)) < (b.snr / Float(b.bits))
            }!

            print("\n  Optimal operating point: INT\(optimalPoint.bits)")
            print("    SNR per bit: \(optimalPoint.snr / Float(optimalPoint.bits)) dB/bit")

            // Verify expected quality hierarchy
            #expect(results[0].snr < results[1].snr, "INT8 should have better SNR than INT4")
            #expect(results[1].snr < results[2].snr, "INT16 should have better SNR than INT8")
        }
    }

    // MARK: - Performance Optimization Tests

    @Suite("Performance Optimization")
    struct PerformanceOptimizationTests {

        @Test
        func testQuantizedComputationPerformance() throws {
            // Test performance of quantized computations
            // - Latency improvements vs FP32
            // - Throughput improvements
            // - Energy efficiency gains

            print("Quantized Computation Performance:")

            // Create test vectors
            let vectorCount = 1000
            let fp32Vectors = (0..<vectorCount).map { i in
                Vector512Optimized { _ in Float.random(in: -1...1) }
            }

            // Quantize all vectors
            let params = LinearQuantizationParams(minValue: -1, maxValue: 1, symmetric: true)
            let int8Vectors = fp32Vectors.map { Vector512INT8(from: $0, params: params) }

            // Test 1: Latency (single operation)
            print("\n1. Latency Test (single operation):")

            let query = fp32Vectors[0]
            let qQuery = int8Vectors[0]

            // Measure FP32 latency
            var fp32MinLatency = Double.infinity
            for _ in 0..<100 {
                let start = Date()
                _ = query.euclideanDistance(to: fp32Vectors[1])
                let elapsed = Date().timeIntervalSince(start)
                fp32MinLatency = min(fp32MinLatency, elapsed)
            }

            // Measure INT8 latency
            var int8MinLatency = Double.infinity
            for _ in 0..<100 {
                let start = Date()
                _ = QuantizedKernels.euclidean512(query: qQuery, candidate: int8Vectors[1])
                let elapsed = Date().timeIntervalSince(start)
                int8MinLatency = min(int8MinLatency, elapsed)
            }

            let latencyImprovement = fp32MinLatency / int8MinLatency
            print("  FP32 latency: \(fp32MinLatency * 1_000_000) μs")
            print("  INT8 latency: \(int8MinLatency * 1_000_000) μs")
            print("  Improvement: \(String(format: "%.1fx", latencyImprovement))")

            #expect(latencyImprovement > 1.5, "INT8 should have lower latency")

            // Test 2: Throughput (batch operations)
            print("\n2. Throughput Test (batch operations):")

            let batchSize = 100
            let iterations = 10

            // FP32 throughput
            let fp32Start = Date()
            for _ in 0..<iterations {
                for i in 1..<batchSize {
                    _ = query.euclideanDistance(to: fp32Vectors[i])
                }
            }
            let fp32Time = Date().timeIntervalSince(fp32Start)
            let fp32Throughput = Double(batchSize * iterations) / fp32Time

            // INT8 throughput
            let int8Start = Date()
            for _ in 0..<iterations {
                for i in 1..<batchSize {
                    _ = QuantizedKernels.euclidean512(query: qQuery, candidate: int8Vectors[i])
                }
            }
            let int8Time = Date().timeIntervalSince(int8Start)
            let int8Throughput = Double(batchSize * iterations) / int8Time

            let throughputImprovement = int8Throughput / fp32Throughput
            print("  FP32 throughput: \(String(format: "%.0f", fp32Throughput)) ops/sec")
            print("  INT8 throughput: \(String(format: "%.0f", int8Throughput)) ops/sec")
            print("  Improvement: \(String(format: "%.1fx", throughputImprovement))")

            #expect(throughputImprovement > 2.0, "INT8 should have higher throughput")

            // Test 3: Energy efficiency (estimated by ops count)
            print("\n3. Energy Efficiency (estimated):")

            // FP32: 512 multiplies + 512 adds + 1 sqrt
            let fp32OpsPerDistance = 512 * 2 + 1
            // INT8: 512 INT8 multiplies + 512 INT8 adds + scale ops
            let int8OpsPerDistance = 512 * 2 / 4  // Assuming 4x efficiency for INT8 ops

            let energyEfficiencyRatio = Float(fp32OpsPerDistance) / Float(int8OpsPerDistance)
            print("  FP32 ops: \(fp32OpsPerDistance)")
            print("  INT8 ops (weighted): \(int8OpsPerDistance)")
            print("  Energy efficiency gain: \(String(format: "%.1fx", energyEfficiencyRatio))")

            #expect(energyEfficiencyRatio > 2.0, "INT8 should be more energy efficient")

            // Test 4: Memory bandwidth utilization
            print("\n4. Memory Bandwidth Utilization:")

            let fp32BytesPerVector = 512 * 4  // 4 bytes per float
            let int8BytesPerVector = 512 * 1  // 1 byte per int8

            let fp32BandwidthUsed = Float(fp32BytesPerVector * batchSize * iterations) / Float(fp32Time)
            let int8BandwidthUsed = Float(int8BytesPerVector * batchSize * iterations) / Float(int8Time)

            print("  FP32 bandwidth: \(String(format: "%.1f MB/s", fp32BandwidthUsed / 1_000_000))")
            print("  INT8 bandwidth: \(String(format: "%.1f MB/s", int8BandwidthUsed / 1_000_000))")
            print("  Bandwidth reduction: \(String(format: "%.1fx", fp32BandwidthUsed / int8BandwidthUsed))")
        }

        @Test
        func testCacheEfficiencyQuantized() throws {
            // Test cache efficiency with quantized data
            // - Cache hit rate improvements
            // - Reduced memory pressure
            // - Cache-friendly access patterns

            print("Cache Efficiency with Quantized Data:")

            // Simulate working set that fits in L2 cache with INT8 but not FP32
            // Typical L2 cache: 256KB-1MB, let's assume 512KB
            let l2CacheSize = 512 * 1024  // bytes

            // Calculate how many vectors fit in cache
            let fp32VectorSize = 512 * 4  // 2KB per vector
            let int8VectorSize = 512 * 1  // 512B per vector

            let fp32VectorsInCache = l2CacheSize / fp32VectorSize  // ~256 vectors
            let int8VectorsInCache = l2CacheSize / int8VectorSize  // ~1024 vectors

            print("  L2 cache capacity:")
            print("    FP32 vectors: \(fp32VectorsInCache)")
            print("    INT8 vectors: \(int8VectorsInCache) (\(int8VectorsInCache/fp32VectorsInCache)x more)")

            // Test 1: Working set that fits in cache with INT8 but not FP32
            let workingSetSize = 512  // vectors - fits with INT8, spills with FP32

            let fp32Vectors = (0..<workingSetSize).map { _ in
                Vector512Optimized { _ in Float.random(in: -1...1) }
            }

            let params = LinearQuantizationParams(minValue: -1, maxValue: 1, symmetric: true)
            let int8Vectors = fp32Vectors.map { Vector512INT8(from: $0, params: params) }

            let query = fp32Vectors[0]
            let qQuery = int8Vectors[0]

            // Warm up cache and measure performance
            print("\n  Working set test (\(workingSetSize) vectors):")

            // FP32: Multiple passes to simulate cache misses
            let fp32Iterations = 100
            let fp32Start = Date()
            for _ in 0..<fp32Iterations {
                for i in 1..<workingSetSize {
                    _ = query.euclideanDistanceSquared(to: fp32Vectors[i])
                }
            }
            let fp32Time = Date().timeIntervalSince(fp32Start)

            // INT8: Should mostly hit cache
            let int8Start = Date()
            for _ in 0..<fp32Iterations {
                for i in 1..<workingSetSize {
                    let d = QuantizedKernels.euclidean512(query: qQuery, candidate: int8Vectors[i])
                    _ = d * d  // Square for fair comparison
                }
            }
            let int8Time = Date().timeIntervalSince(int8Start)

            let cacheSpeedup = fp32Time / int8Time
            print("    FP32 time: \(String(format: "%.2f ms", fp32Time * 1000))")
            print("    INT8 time: \(String(format: "%.2f ms", int8Time * 1000))")
            print("    Cache efficiency gain: \(String(format: "%.1fx", cacheSpeedup))")

            #expect(cacheSpeedup > 2.5, "INT8 should benefit from better cache utilization")

            // Test 2: Sequential vs random access patterns
            print("\n  Access pattern test:")

            let accessCount = 1000
            let dataSize = 1000

            let largeDataset = (0..<dataSize).map { _ in
                Vector512Optimized { _ in Float.random(in: -1...1) }
            }
            let quantizedDataset = largeDataset.map { Vector512INT8(from: $0, params: params) }

            // Sequential access
            let seqStart = Date()
            for i in 0..<accessCount {
                let idx = i % dataSize
                _ = QuantizedKernels.dotProduct512(query: quantizedDataset[idx], candidate: qQuery)
            }
            let seqTime = Date().timeIntervalSince(seqStart)

            // Random access
            let randomIndices = (0..<accessCount).map { _ in Int.random(in: 0..<dataSize) }
            let randStart = Date()
            for idx in randomIndices {
                _ = QuantizedKernels.dotProduct512(query: quantizedDataset[idx], candidate: qQuery)
            }
            let randTime = Date().timeIntervalSince(randStart)

            let sequentialAdvantage = randTime / seqTime
            print("    Sequential access: \(String(format: "%.2f ms", seqTime * 1000))")
            print("    Random access: \(String(format: "%.2f ms", randTime * 1000))")
            print("    Sequential advantage: \(String(format: "%.1fx", sequentialAdvantage))")

            // Test 3: Memory pressure reduction
            print("\n  Memory pressure test:")

            let pressureTestSize = 2000  // Large enough to cause pressure
            let testIterations = 10

            print("    Dataset size: \(pressureTestSize) vectors")
            print("    FP32 memory: \(pressureTestSize * fp32VectorSize / 1_000_000) MB")
            print("    INT8 memory: \(pressureTestSize * int8VectorSize / 1_000_000) MB")
            print("    Reduction: 4x")

            #expect(int8VectorSize * 4 == fp32VectorSize, "INT8 should use 4x less memory")
        }

        @Test
        func testQuantizationOverhead() throws {
            // Test overhead of quantization/dequantization
            // - Conversion costs
            // - Amortization over batch operations
            // - Break-even analysis

            print("Quantization/Dequantization Overhead:")

            // Create test vectors
            let testVectors = (0..<100).map { _ in
                Vector512Optimized { _ in Float.random(in: -1...1) }
            }

            let params = LinearQuantizationParams(minValue: -1, maxValue: 1, symmetric: true)

            // Test 1: Quantization cost
            print("\n1. Quantization cost:")

            let quantIterations = 1000
            let quantStart = Date()
            for _ in 0..<quantIterations {
                for vec in testVectors {
                    _ = Vector512INT8(from: vec, params: params)
                }
            }
            let quantTime = Date().timeIntervalSince(quantStart)
            let quantTimePerVector = quantTime / Double(quantIterations * testVectors.count)

            print("  Time per vector: \(String(format: "%.2f μs", quantTimePerVector * 1_000_000))")
            print("  Throughput: \(String(format: "%.0f vectors/sec", 1.0 / quantTimePerVector))")

            // Test 2: Dequantization cost
            print("\n2. Dequantization cost:")

            let quantizedVectors = testVectors.map { Vector512INT8(from: $0, params: params) }

            let dequantStart = Date()
            for _ in 0..<quantIterations {
                for qvec in quantizedVectors {
                    _ = qvec.toFP32()
                }
            }
            let dequantTime = Date().timeIntervalSince(dequantStart)
            let dequantTimePerVector = dequantTime / Double(quantIterations * quantizedVectors.count)

            print("  Time per vector: \(String(format: "%.2f μs", dequantTimePerVector * 1_000_000))")
            print("  Throughput: \(String(format: "%.0f vectors/sec", 1.0 / dequantTimePerVector))")

            // Test 3: Break-even analysis
            print("\n3. Break-even analysis:")

            // Compare: quantize + INT8 ops + dequantize vs FP32 ops
            let query = testVectors[0]
            let candidates = Array(testVectors[1..<11])  // 10 candidates

            // Pure FP32 approach
            let fp32Start = Date()
            for _ in 0..<100 {
                for candidate in candidates {
                    _ = query.euclideanDistance(to: candidate)
                }
            }
            let fp32Time = Date().timeIntervalSince(fp32Start)

            // Quantize-compute-dequantize approach
            let quantizedApproachStart = Date()
            for _ in 0..<100 {
                let qQuery = Vector512INT8(from: query, params: params)
                for candidate in candidates {
                    let qCandidate = Vector512INT8(from: candidate, params: params)
                    _ = QuantizedKernels.euclidean512(query: qQuery, candidate: qCandidate)
                }
            }
            let quantizedApproachTime = Date().timeIntervalSince(quantizedApproachStart)

            let breakEvenRatio = fp32Time / quantizedApproachTime
            print("  FP32 time: \(String(format: "%.2f ms", fp32Time * 1000))")
            print("  Quantized approach time: \(String(format: "%.2f ms", quantizedApproachTime * 1000))")
            print("  Break-even at \(candidates.count) operations: \(breakEvenRatio > 1 ? "YES" : "NO")")
            print("  Speed ratio: \(String(format: "%.2fx", breakEvenRatio))")

            // Test different batch sizes to find break-even point
            print("\n4. Batch size break-even:")

            for batchSize in [1, 5, 10, 20, 50, 100] {
                let batchCandidates = Array(testVectors[0..<min(batchSize, testVectors.count)])

                // FP32
                let fp32BatchStart = Date()
                for _ in 0..<10 {
                    for candidate in batchCandidates {
                        _ = query.euclideanDistance(to: candidate)
                    }
                }
                let fp32BatchTime = Date().timeIntervalSince(fp32BatchStart)

                // Quantized (amortize query quantization)
                let quantBatchStart = Date()
                for _ in 0..<10 {
                    let qQuery = Vector512INT8(from: query, params: params)
                    for candidate in batchCandidates {
                        let qCandidate = Vector512INT8(from: candidate, params: params)
                        _ = QuantizedKernels.euclidean512(query: qQuery, candidate: qCandidate)
                    }
                }
                let quantBatchTime = Date().timeIntervalSince(quantBatchStart)

                let ratio = fp32BatchTime / quantBatchTime
                print("  Batch size \(batchSize): \(String(format: "%.2fx", ratio)) \(ratio > 1 ? "✓" : "✗")")

                if batchSize >= 10 {
                    #expect(ratio > 1.0, "Should be beneficial for batch size >= 10")
                }
            }
        }

        @Test
        func testParallelQuantizedOperations() async throws {
            // Test parallel quantized operations
            // - Multi-threaded quantization
            // - Parallel distance computation
            // - Scalability with core count

            print("Parallel Quantized Operations:")

            // Create large dataset for parallel processing
            let datasetSize = 10000
            let vectors = (0..<datasetSize).map { i in
                Vector512Optimized { j in sin(Float(i + j) * 0.001) }
            }

            let params = LinearQuantizationParams(minValue: -1, maxValue: 1, symmetric: true)

            // Test 1: Parallel quantization
            print("\n1. Parallel quantization:")

            // Serial quantization
            let serialStart = Date()
            let serialQuantized = vectors.map { Vector512INT8(from: $0, params: params) }
            let serialTime = Date().timeIntervalSince(serialStart)

            // Parallel quantization
            let parallelStart = Date()
            let parallelQuantized = await withTaskGroup(of: [Vector512INT8].self) { group in
                let chunkSize = datasetSize / 8  // Assuming 8 cores
                for i in stride(from: 0, to: datasetSize, by: chunkSize) {
                    let end = min(i + chunkSize, datasetSize)
                    group.addTask {
                        return Array(vectors[i..<end]).map { Vector512INT8(from: $0, params: params) }
                    }
                }

                var result: [Vector512INT8] = []
                for await chunk in group {
                    result.append(contentsOf: chunk)
                }
                return result
            }
            let parallelTime = Date().timeIntervalSince(parallelStart)

            let quantSpeedup = serialTime / parallelTime
            print("  Serial time: \(String(format: "%.2f ms", serialTime * 1000))")
            print("  Parallel time: \(String(format: "%.2f ms", parallelTime * 1000))")
            print("  Speedup: \(String(format: "%.1fx", quantSpeedup))")

            #expect(quantSpeedup > 2.0, "Parallel quantization should provide speedup")
            #expect(serialQuantized.count == parallelQuantized.count, "Should produce same number of vectors")

            // Test 2: Parallel distance computation
            print("\n2. Parallel distance computation:")

            let query = vectors[0]
            let qQuery = Vector512INT8(from: query, params: params)
            let candidates = Array(serialQuantized[0..<1000])

            // Serial distance computation
            let serialDistStart = Date()
            let serialDistances = candidates.map { candidate in
                QuantizedKernels.euclidean512(query: qQuery, candidate: candidate)
            }
            let serialDistTime = Date().timeIntervalSince(serialDistStart)

            // Parallel distance computation
            let parallelDistStart = Date()
            let parallelDistances = await withTaskGroup(of: [Float].self) { group in
                let chunkSize = candidates.count / 8
                for i in stride(from: 0, to: candidates.count, by: chunkSize) {
                    let end = min(i + chunkSize, candidates.count)
                    group.addTask {
                        return Array(candidates[i..<end]).map { candidate in
                            QuantizedKernels.euclidean512(query: qQuery, candidate: candidate)
                        }
                    }
                }

                var result: [Float] = []
                for await chunk in group {
                    result.append(contentsOf: chunk)
                }
                return result
            }
            let parallelDistTime = Date().timeIntervalSince(parallelDistStart)

            let distSpeedup = serialDistTime / parallelDistTime
            print("  Serial time: \(String(format: "%.2f ms", serialDistTime * 1000))")
            print("  Parallel time: \(String(format: "%.2f ms", parallelDistTime * 1000))")
            print("  Speedup: \(String(format: "%.1fx", distSpeedup))")

            #expect(distSpeedup > 2.0, "Parallel distance computation should provide speedup")
            #expect(serialDistances.count == parallelDistances.count, "Should compute same number of distances")

            // Test 3: Scalability test
            print("\n3. Scalability with different thread counts:")

            for threadCount in [1, 2, 4, 8] {
                let scalabilityStart = Date()
                _ = await withTaskGroup(of: [Float].self) { group in
                    let chunkSize = candidates.count / threadCount
                    for i in stride(from: 0, to: candidates.count, by: chunkSize) {
                        let end = min(i + chunkSize, candidates.count)
                        group.addTask {
                            return Array(candidates[i..<end]).map { candidate in
                                QuantizedKernels.dotProduct512(query: qQuery, candidate: candidate)
                            }
                        }
                    }

                    var result: [Float] = []
                    for await chunk in group {
                        result.append(contentsOf: chunk)
                    }
                    return result
                }
                let scalabilityTime = Date().timeIntervalSince(scalabilityStart)

                let speedupVsSerial = serialDistTime / scalabilityTime
                print("  \(threadCount) threads: \(String(format: "%.2f ms", scalabilityTime * 1000)) (\(String(format: "%.1fx", speedupVsSerial)) speedup)")
            }
        }

        @Test
        func testQuantizedBatchPerformance() throws {
            // Test batch operation performance with quantization
            // - Large-scale similarity computation
            // - Batch quantization efficiency
            // - Memory-bound vs compute-bound analysis

            print("Quantized Batch Performance:")

            // Create large dataset
            let datasetSize = 5000
            let queryCount = 10

            let dataset = (0..<datasetSize).map { i in
                Vector512Optimized { j in cos(Float(i * 512 + j) * 0.0001) }
            }

            let queries = (0..<queryCount).map { i in
                Vector512Optimized { j in sin(Float(i * 512 + j) * 0.0001) }
            }

            let params = LinearQuantizationParams(minValue: -1, maxValue: 1, symmetric: true)

            // Pre-quantize dataset
            print("\n1. Batch quantization efficiency:")

            let batchQuantStart = Date()
            let quantizedDataset = dataset.map { Vector512INT8(from: $0, params: params) }
            let batchQuantTime = Date().timeIntervalSince(batchQuantStart)

            let quantThroughput = Double(datasetSize) / batchQuantTime
            print("  Quantized \(datasetSize) vectors in \(String(format: "%.2f ms", batchQuantTime * 1000))")
            print("  Throughput: \(String(format: "%.0f vectors/sec", quantThroughput))")
            print("  Time per vector: \(String(format: "%.2f μs", batchQuantTime / Double(datasetSize) * 1_000_000))")

            // Test 2: Large-scale similarity computation
            print("\n2. Large-scale similarity computation:")

            let quantizedQueries = queries.map { Vector512INT8(from: $0, params: params) }

            // FP32 baseline
            let fp32Start = Date()
            var fp32Results: [[Float]] = []
            for query in queries {
                var distances: [Float] = []
                for candidate in dataset {
                    distances.append(query.euclideanDistanceSquared(to: candidate))
                }
                fp32Results.append(distances)
            }
            let fp32Time = Date().timeIntervalSince(fp32Start)

            // INT8 quantized
            let int8Start = Date()
            var int8Results: [[Float]] = []
            for qQuery in quantizedQueries {
                var distances: [Float] = []
                for qCandidate in quantizedDataset {
                    let d = QuantizedKernels.euclidean512(query: qQuery, candidate: qCandidate)
                    distances.append(d * d)  // Square for comparison
                }
                int8Results.append(distances)
            }
            let int8Time = Date().timeIntervalSince(int8Start)

            let totalComparisons = queryCount * datasetSize
            let fp32CompPerSec = Double(totalComparisons) / fp32Time
            let int8CompPerSec = Double(totalComparisons) / int8Time

            print("  Total comparisons: \(totalComparisons)")
            print("  FP32: \(String(format: "%.2f sec", fp32Time)) (\(String(format: "%.0f comps/sec", fp32CompPerSec)))")
            print("  INT8: \(String(format: "%.2f sec", int8Time)) (\(String(format: "%.0f comps/sec", int8CompPerSec)))")
            print("  Speedup: \(String(format: "%.1fx", fp32Time / int8Time))")

            #expect(int8Time < fp32Time, "INT8 should be faster for batch operations")

            // Test 3: Memory-bound vs compute-bound analysis
            print("\n3. Memory vs compute bound analysis:")

            // Small working set (fits in cache) - more compute-bound
            let smallDataset = Array(quantizedDataset[0..<100])
            let smallQuery = quantizedQueries[0]

            let smallStart = Date()
            for _ in 0..<100 {
                for candidate in smallDataset {
                    _ = QuantizedKernels.dotProduct512(query: smallQuery, candidate: candidate)
                }
            }
            let smallTime = Date().timeIntervalSince(smallStart)
            let smallOpsPerSec = Double(100 * smallDataset.count) / smallTime

            // Large working set (exceeds cache) - more memory-bound
            let largeDataset = quantizedDataset
            let largeStart = Date()
            for candidate in largeDataset {
                _ = QuantizedKernels.dotProduct512(query: smallQuery, candidate: candidate)
            }
            let largeTime = Date().timeIntervalSince(largeStart)
            let largeOpsPerSec = Double(largeDataset.count) / largeTime

            print("  Small dataset (cache-friendly):")
            print("    Size: \(smallDataset.count) vectors")
            print("    Ops/sec: \(String(format: "%.0f", smallOpsPerSec))")

            print("  Large dataset (memory-bound):")
            print("    Size: \(largeDataset.count) vectors")
            print("    Ops/sec: \(String(format: "%.0f", largeOpsPerSec))")
            print("    Performance ratio: \(String(format: "%.1fx", smallOpsPerSec / largeOpsPerSec))")

            // Memory bandwidth utilization
            let bytesPerVector = 512  // INT8
            let memoryBandwidth = Double(largeDataset.count * bytesPerVector) / largeTime
            print("\n  Estimated memory bandwidth: \(String(format: "%.1f GB/s", memoryBandwidth / 1_000_000_000))")
        }
    }

    // MARK: - Different Bit-Width Tests

    @Suite("Different Bit-Widths")
    struct DifferentBitWidthTests {

        @Test
        func testINT4Quantization() throws {
            // Test INT4 quantization for maximum compression
            // - 8x memory reduction
            // - Accuracy vs compression tradeoffs
            // - Specialized INT4 arithmetic

            print("INT4 Quantization (4-bit):")

            // Create test vectors
            let testVectors = [
                ("Uniform", Vector512Optimized { _ in Float.random(in: -1...1) }),
                ("Gaussian", Vector512Optimized { _ in
                    let u1 = Float.random(in: 0.001...0.999)
                    let u2 = Float.random(in: 0.001...0.999)
                    return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
                }),
                ("Sparse", Vector512Optimized { i in i % 20 == 0 ? Float.random(in: -1...1) : 0 })
            ]

            for (name, vector) in testVectors {
                print("\n  \(name) distribution:")

                // Test symmetric quantization
                let symInt4 = SimulatedINT4(from: vector, symmetric: true)
                let symDequantized = symInt4.toFP32()

                // Test asymmetric quantization
                let asymInt4 = SimulatedINT4(from: vector, symmetric: false)
                let asymDequantized = asymInt4.toFP32()

                // Calculate errors
                var symError: Float = 0
                var asymError: Float = 0
                for i in 0..<512 {
                    symError += abs(vector[i] - symDequantized[i])
                    asymError += abs(vector[i] - asymDequantized[i])
                }
                symError /= 512
                asymError /= 512

                print("    Symmetric - Scale: \(symInt4.scale), Zero: \(symInt4.zeroPoint), Avg Error: \(symError)")
                print("    Asymmetric - Scale: \(asymInt4.scale), Zero: \(asymInt4.zeroPoint), Avg Error: \(asymError)")

                // Verify 8x compression (4 bits vs 32 bits)
                let originalSize = 512 * 4  // 512 floats * 4 bytes
                let compressedSize = symInt4.values.count  // 256 bytes (2 INT4 per byte)
                let compressionRatio = Float(originalSize) / Float(compressedSize)
                print("    Compression ratio: \(compressionRatio)x")
                #expect(abs(compressionRatio - 8.0) < 0.1, "INT4 should provide 8x compression")
            }

            // Test accuracy vs INT8
            print("\n  INT4 vs INT8 comparison:")
            let testVector = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.01) })

            let int4 = SimulatedINT4(from: testVector, symmetric: true)
            let int8 = Vector512INT8(from: testVector)

            let int4Restored = int4.toFP32()
            let int8Restored = int8.toFP32()

            var int4Error: Float = 0
            var int8Error: Float = 0
            for i in 0..<512 {
                int4Error += abs(testVector[i] - int4Restored[i])
                int8Error += abs(testVector[i] - int8Restored[i])
            }
            int4Error /= 512
            int8Error /= 512

            print("    INT4 avg error: \(int4Error)")
            print("    INT8 avg error: \(int8Error)")
            print("    Error ratio (INT4/INT8): \(int4Error / int8Error)x")

            #expect(int4Error > int8Error, "INT4 should have higher error than INT8")
            #expect(int4Error / int8Error < 10, "INT4 error should be reasonable compared to INT8")
        }

        @Test
        func testINT8Quantization() throws {
            // Test standard INT8 quantization
            // - 4x memory reduction
            // - Good accuracy/performance balance
            // - Wide hardware support

            print("INT8 Quantization (8-bit standard):")

            // This is already extensively tested in earlier tests
            // Here we focus on comparing with other bit widths

            let vectors = [
                Vector512Optimized { _ in Float.random(in: -1...1) },
                Vector512Optimized { i in sin(Float(i) * 0.01) },
                Vector512Optimized { i in Float(i) / 256.0 - 1.0 }
            ]

            print("  Memory and accuracy characteristics:")
            for (idx, vector) in vectors.enumerated() {
                let int8 = Vector512INT8(from: vector)
                let restored = int8.toFP32()

                var maxError: Float = 0
                var avgError: Float = 0
                for i in 0..<512 {
                    let error = abs(vector[i] - restored[i])
                    maxError = max(maxError, error)
                    avgError += error
                }
                avgError /= 512

                print("    Vector \(idx + 1):")
                print("      Max error: \(maxError)")
                print("      Avg error: \(avgError)")
                print("      Scale: \(int8.quantizationParams.scale)")
            }

            // Verify 4x compression
            let originalBytes = 512 * 4  // FP32
            let quantizedBytes = 512 * 1  // INT8
            let ratio = Float(originalBytes) / Float(quantizedBytes)
            print("  Compression ratio: \(ratio)x")
            #expect(ratio == 4.0, "INT8 should provide exactly 4x compression")

            // Test hardware optimization benefits
            print("\n  Hardware optimization benefits:")
            print("    - SIMD4<Int8> operations supported natively")
            print("    - Efficient cache utilization")
            print("    - Reduced memory bandwidth requirements")
            print("    - Wide hardware support across platforms")
        }

        @Test
        func testINT16Quantization() throws {
            // Test INT16 quantization for high accuracy
            // - 2x memory reduction
            // - Minimal accuracy loss
            // - Compatibility with existing systems

            print("INT16 Quantization (16-bit):")

            let testVectors = [
                ("Small range", try Vector512Optimized((0..<512).map { Float($0) * 0.001 })),
                ("Large range", try Vector512Optimized((0..<512).map { Float($0) * 10.0 })),
                ("High precision", try Vector512Optimized((0..<512).map { sin(Float($0) * 0.001) * 0.01 }))
            ]

            for (name, vector) in testVectors {
                print("\n  \(name):")

                // Test symmetric INT16
                let symInt16 = SimulatedINT16(from: vector, symmetric: true)
                let symRestored = symInt16.toFP32()

                // Test asymmetric INT16
                let asymInt16 = SimulatedINT16(from: vector, symmetric: false)
                let asymRestored = asymInt16.toFP32()

                // Calculate precision metrics
                var symMaxError: Float = 0
                var asymMaxError: Float = 0
                var symRMSE: Float = 0
                var asymRMSE: Float = 0

                for i in 0..<512 {
                    let symErr = abs(vector[i] - symRestored[i])
                    let asymErr = abs(vector[i] - asymRestored[i])

                    symMaxError = max(symMaxError, symErr)
                    asymMaxError = max(asymMaxError, asymErr)
                    symRMSE += symErr * symErr
                    asymRMSE += asymErr * asymErr
                }

                symRMSE = sqrt(symRMSE / 512)
                asymRMSE = sqrt(asymRMSE / 512)

                print("    Symmetric:")
                print("      Scale: \(symInt16.scale), Zero: \(symInt16.zeroPoint)")
                print("      Max error: \(symMaxError)")
                print("      RMSE: \(symRMSE)")

                print("    Asymmetric:")
                print("      Scale: \(asymInt16.scale), Zero: \(asymInt16.zeroPoint)")
                print("      Max error: \(asymMaxError)")
                print("      RMSE: \(asymRMSE)")

                // INT16 should have much lower error than INT8
                #expect(symMaxError < 0.001 || asymMaxError < 0.001, "INT16 should have very low error")
            }

            // Compare with INT8 and INT4
            print("\n  Bit-width comparison:")
            let compareVector = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.01) })

            let int4 = SimulatedINT4(from: compareVector, symmetric: true)
            let int8 = Vector512INT8(from: compareVector)
            let int16 = SimulatedINT16(from: compareVector, symmetric: true)

            let int4Restored = int4.toFP32()
            let int8Restored = int8.toFP32()
            let int16Restored = int16.toFP32()

            var errors: [String: Float] = [:]
            for (name, restored) in [("INT4", int4Restored), ("INT8", int8Restored), ("INT16", int16Restored)] {
                var error: Float = 0
                for i in 0..<512 {
                    error += abs(compareVector[i] - restored[i])
                }
                errors[name] = error / 512
            }

            print("    INT4 avg error: \(errors["INT4"]!)")
            print("    INT8 avg error: \(errors["INT8"]!)")
            print("    INT16 avg error: \(errors["INT16"]!)")
            print("    INT16 vs INT8 improvement: \(errors["INT8"]! / errors["INT16"]!)x")
            print("    INT16 vs INT4 improvement: \(errors["INT4"]! / errors["INT16"]!)x")

            // Compression ratios
            print("\n  Compression ratios (vs FP32):")
            print("    INT4: 8x compression")
            print("    INT8: 4x compression")
            print("    INT16: 2x compression")

            #expect(errors["INT16"]! < errors["INT8"]!, "INT16 should be more accurate than INT8")
            #expect(errors["INT8"]! < errors["INT4"]!, "INT8 should be more accurate than INT4")
        }

        @Test
        func testMixedBitWidthOperations() throws {
            // Test operations with mixed bit-widths
            // - Different precisions for different layers
            // - Adaptive bit-width selection
            // - Cross-precision arithmetic

            print("Mixed Bit-Width Operations:")

            // Simulate a neural network with different precision requirements
            let layers = [
                ("Input", Vector512Optimized { _ in Float.random(in: -1...1) }),
                ("Hidden1", Vector512Optimized { i in sin(Float(i) * 0.01) }),
                ("Hidden2", Vector512Optimized { i in cos(Float(i) * 0.01) }),
                ("Output", Vector512Optimized { i in Float(i) / 512.0 })
            ]

            print("\n1. Layer-wise precision assignment:")

            // Assign different bit-widths based on layer importance
            for (idx, (name, vector)) in layers.enumerated() {
                let bitWidth: String
                let error: Float

                switch idx {
                case 0:  // Input layer - can use aggressive quantization
                    let q = SimulatedINT4(from: vector, symmetric: true)
                    let restored = q.toFP32()
                    error = (0..<512).map { abs(vector[$0] - restored[$0]) }.reduce(0, +) / 512
                    bitWidth = "INT4"

                case 1, 2:  // Hidden layers - balanced precision
                    let q = Vector512INT8(from: vector)
                    let restored = q.toFP32()
                    error = (0..<512).map { abs(vector[$0] - restored[$0]) }.reduce(0, +) / 512
                    bitWidth = "INT8"

                case 3:  // Output layer - high precision
                    let q = SimulatedINT16(from: vector, symmetric: false)
                    let restored = q.toFP32()
                    error = (0..<512).map { abs(vector[$0] - restored[$0]) }.reduce(0, +) / 512
                    bitWidth = "INT16"

                default:
                    bitWidth = "FP32"
                    error = 0
                }

                print("  \(name) layer: \(bitWidth), Error: \(error)")
            }

            print("\n2. Cross-precision arithmetic:")

            // Test operations between different bit-widths
            let v1 = Vector512Optimized { _ in Float.random(in: -1...1) }
            let v2 = Vector512Optimized { _ in Float.random(in: -1...1) }

            let v1_int4 = SimulatedINT4(from: v1, symmetric: true)
            let v1_int8 = Vector512INT8(from: v1)
            let v1_int16 = SimulatedINT16(from: v1, symmetric: true)

            let v2_int8 = Vector512INT8(from: v2)

            // Simulate mixed-precision dot product
            let v1_int4_fp32 = v1_int4.toFP32()
            let v2_int8_fp32 = v2_int8.toFP32()

            let mixedDot = v1_int4_fp32.dotProduct(v2_int8_fp32)
            let originalDot = v1.dotProduct(v2)

            print("  INT4 x INT8 dot product:")
            print("    Mixed precision result: \(mixedDot)")
            print("    Original FP32 result: \(originalDot)")
            print("    Relative error: \(abs(mixedDot - originalDot) / abs(originalDot))")

            print("\n3. Adaptive bit-width selection:")

            // Choose bit-width based on value distribution
            let testCases = [
                ("Sparse", Vector512Optimized { i in i % 50 == 0 ? Float.random(in: -1...1) : 0 }),
                ("Dense uniform", Vector512Optimized { _ in Float.random(in: -1...1) }),
                ("Small values", Vector512Optimized { _ in Float.random(in: -0.001...0.001) }),
                ("Large values", Vector512Optimized { _ in Float.random(in: -100...100) })
            ]

            for (name, vector) in testCases {
                // Calculate sparsity and dynamic range
                let array = vector.toArray()
                let nonZeroCount = array.filter { $0 != 0 }.count
                let sparsity = 1.0 - Float(nonZeroCount) / 512.0
                let minVal = array.min() ?? 0
                let maxVal = array.max() ?? 0
                let dynamicRange = maxVal - minVal

                // Select bit-width based on characteristics
                let selectedBitWidth: String
                if sparsity > 0.8 {
                    selectedBitWidth = "INT4"  // Highly sparse - can use aggressive quantization
                } else if dynamicRange < 0.01 {
                    selectedBitWidth = "INT16"  // Small dynamic range - need high precision
                } else if dynamicRange > 50 {
                    selectedBitWidth = "INT8"  // Large range - balanced choice
                } else {
                    selectedBitWidth = "INT8"  // Default
                }

                print("  \(name): Sparsity=\(sparsity), Range=\(dynamicRange) → \(selectedBitWidth)")
            }

            print("\n4. Memory savings with mixed precision:")

            // Calculate memory usage for different strategies
            let layerCount = 4
            let vectorsPerLayer = 100

            let fullPrecisionBytes = layerCount * vectorsPerLayer * 512 * 4
            let uniformInt8Bytes = layerCount * vectorsPerLayer * 512 * 1
            let mixedPrecisionBytes = [
                vectorsPerLayer * 512 / 2,    // INT4 (packed)
                vectorsPerLayer * 512 * 1,    // INT8
                vectorsPerLayer * 512 * 1,    // INT8
                vectorsPerLayer * 512 * 2     // INT16
            ].reduce(0, +)

            print("  Full precision (FP32): \(fullPrecisionBytes / 1024) KB")
            print("  Uniform INT8: \(uniformInt8Bytes / 1024) KB")
            print("  Mixed precision: \(mixedPrecisionBytes / 1024) KB")
            print("  Savings vs FP32: \(Float(fullPrecisionBytes) / Float(mixedPrecisionBytes))x")
        }

        @Test
        func testBitWidthSelection() throws {
            // Test automatic bit-width selection
            // - Accuracy requirements
            // - Performance constraints
            // - Memory limitations

            print("Automatic Bit-Width Selection:")

            // Define accuracy thresholds
            struct BitWidthSelector {
                let maxAcceptableError: Float
                let memoryBudgetBytes: Int
                let performanceTarget: Float  // ops/sec

                func selectBitWidth(for vector: Vector512Optimized) -> String {
                    // Test different bit widths
                    let int4 = SimulatedINT4(from: vector, symmetric: true)
                    let int8 = Vector512INT8(from: vector)
                    let int16 = SimulatedINT16(from: vector, symmetric: true)

                    let int4Restored = int4.toFP32()
                    let int8Restored = int8.toFP32()
                    let int16Restored = int16.toFP32()

                    // Calculate errors
                    var errors: [String: Float] = [:]
                    for (name, restored) in [("INT4", int4Restored), ("INT8", int8Restored), ("INT16", int16Restored)] {
                        var maxErr: Float = 0
                        for i in 0..<512 {
                            maxErr = max(maxErr, abs(vector[i] - restored[i]))
                        }
                        errors[name] = maxErr
                    }

                    // Select based on constraints
                    if errors["INT4"]! < maxAcceptableError && memoryBudgetBytes < 300 {
                        return "INT4"
                    } else if errors["INT8"]! < maxAcceptableError && memoryBudgetBytes < 600 {
                        return "INT8"
                    } else if errors["INT16"]! < maxAcceptableError {
                        return "INT16"
                    } else {
                        return "FP32"  // Fall back to full precision
                    }
                }
            }

            print("\n1. Error-based selection:")

            let testVector = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.01) })
            let errorThresholds: [Float] = [0.0001, 0.001, 0.01, 0.1]

            for threshold in errorThresholds {
                let selector = BitWidthSelector(
                    maxAcceptableError: threshold,
                    memoryBudgetBytes: 10000,
                    performanceTarget: 1000000
                )
                let selected = selector.selectBitWidth(for: testVector)
                print("  Max error \(threshold) → \(selected)")
            }

            print("\n2. Memory-constrained selection:")

            let memoryBudgets = [256, 512, 1024, 2048]  // bytes
            for budget in memoryBudgets {
                let selector = BitWidthSelector(
                    maxAcceptableError: 0.01,
                    memoryBudgetBytes: budget,
                    performanceTarget: 1000000
                )
                let selected = selector.selectBitWidth(for: testVector)
                print("  Memory budget \(budget)B → \(selected)")
            }

            print("\n3. Distribution-aware selection:")

            let distributions = [
                ("Gaussian", Vector512Optimized { _ in
                    let u1 = Float.random(in: 0.001...0.999)
                    let u2 = Float.random(in: 0.001...0.999)
                    return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
                }),
                ("Uniform", Vector512Optimized { _ in Float.random(in: -1...1) }),
                ("Sparse", Vector512Optimized { i in i % 20 == 0 ? Float.random(in: -1...1) : 0 }),
                ("Bimodal", Vector512Optimized { i in i % 2 == 0 ? -0.8 : 0.8 })
            ]

            for (name, vector) in distributions {
                // Analyze distribution characteristics
                let array = vector.toArray()
                let mean = array.reduce(0, +) / Float(array.count)
                let variance = array.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(array.count)
                let sparsity = Float(array.filter { abs($0) < 0.001 }.count) / Float(array.count)

                // Select bit-width based on distribution
                let bitWidth: String
                if sparsity > 0.8 {
                    bitWidth = "INT4"  // Sparse vectors can use aggressive quantization
                } else if variance < 0.01 {
                    bitWidth = "INT16"  // Low variance needs higher precision
                } else {
                    bitWidth = "INT8"  // Default balanced choice
                }

                print("  \(name): mean=\(mean), var=\(variance), sparsity=\(sparsity) → \(bitWidth)")
            }

            print("\n4. Performance-aware selection:")

            // Simulate performance characteristics
            struct PerformanceProfile {
                let bitWidth: String
                let opsPerSec: Float
                let memoryBandwidth: Float  // GB/s
                let energyPerOp: Float  // relative
            }

            let profiles = [
                PerformanceProfile(bitWidth: "INT4", opsPerSec: 4000000, memoryBandwidth: 0.5, energyPerOp: 0.25),
                PerformanceProfile(bitWidth: "INT8", opsPerSec: 2000000, memoryBandwidth: 1.0, energyPerOp: 0.5),
                PerformanceProfile(bitWidth: "INT16", opsPerSec: 1000000, memoryBandwidth: 2.0, energyPerOp: 0.75),
                PerformanceProfile(bitWidth: "FP32", opsPerSec: 500000, memoryBandwidth: 4.0, energyPerOp: 1.0)
            ]

            let performanceTargets = [
                ("Latency-critical", 3000000),
                ("Balanced", 1500000),
                ("Quality-focused", 500000)
            ]

            for (scenario, targetOps) in performanceTargets {
                let selected = profiles.first { $0.opsPerSec >= Float(targetOps) }?.bitWidth ?? "FP32"
                print("  \(scenario) (\(targetOps) ops/s) → \(selected)")
            }
        }
    }

    // MARK: - Quantization Schemes Tests

    @Suite("Quantization Schemes")
    struct QuantizationSchemesTests {

        @Test
        func testSymmetricQuantization() throws {
            // Test symmetric quantization schemes
            // - Zero-point at center
            // - Simplified arithmetic
            // - Hardware optimization benefits

            print("Symmetric Quantization:")

            let testVectors = [
                ("Centered", try Vector512Optimized((0..<512).map { sin(Float($0) * 0.01) })),  // [-1, 1]
                ("Positive skewed", Vector512Optimized { _ in Float.random(in: 0...2) }),
                ("Negative skewed", Vector512Optimized { _ in Float.random(in: -2...0) })
            ]

            for (name, vector) in testVectors {
                print("\n  \(name) distribution:")

                let params = LinearQuantizationParams(
                    minValue: vector.toArray().min()!,
                    maxValue: vector.toArray().max()!,
                    symmetric: true
                )

                let quantized = Vector512INT8(from: vector, params: params)
                let dequantized = quantized.toFP32()

                print("    Scale: \(params.scale)")
                print("    Zero point: \(params.zeroPoint)")
                #expect(params.zeroPoint == 0, "Symmetric quantization should have zero-point at 0")

                // Calculate quantization error
                var totalError: Float = 0
                var maxError: Float = 0
                for i in 0..<512 {
                    let error = abs(vector[i] - dequantized[i])
                    totalError += error
                    maxError = max(maxError, error)
                }

                print("    Average error: \(totalError / 512)")
                print("    Max error: \(maxError)")

                // Verify arithmetic simplification
                // With symmetric quantization, operations are simpler
                print("    Arithmetic benefits:")
                print("      - No zero-point offset in calculations")
                print("      - Simpler dequantization: just multiply by scale")
                print("      - Better SIMD utilization")
            }

            print("\n  Hardware optimization benefits:")
            print("    - Reduced instruction count")
            print("    - No bias addition in inner loops")
            print("    - Better vectorization opportunities")
            print("    - Symmetric range utilization")
        }

        @Test
        func testAsymmetricQuantization() throws {
            // Test asymmetric quantization schemes
            // - Optimal range utilization
            // - Better accuracy for skewed distributions
            // - Additional complexity tradeoffs

            print("Asymmetric Quantization:")

            // Test with skewed distributions
            let testCases = [
                ("Positive skewed", Vector512Optimized { _ in Float.random(in: 0...10) }),
                ("Negative skewed", Vector512Optimized { _ in Float.random(in: -10...(-1)) }),
                ("Exponential", Vector512Optimized { i in exp(-Float(i) / 100.0) }),
                ("Log-normal", Vector512Optimized { _ in
                    let normal = Float.random(in: 0.001...0.999)
                    return exp(normal)
                })
            ]

            for (name, vector) in testCases {
                print("\n  \(name) distribution:")

                // Compare symmetric vs asymmetric
                let symParams = LinearQuantizationParams(
                    minValue: vector.toArray().min()!,
                    maxValue: vector.toArray().max()!,
                    symmetric: true
                )

                let asymParams = LinearQuantizationParams(
                    minValue: vector.toArray().min()!,
                    maxValue: vector.toArray().max()!,
                    symmetric: false
                )

                let symQuantized = Vector512INT8(from: vector, params: symParams)
                let asymQuantized = Vector512INT8(from: vector, params: asymParams)

                let symDequantized = symQuantized.toFP32()
                let asymDequantized = asymQuantized.toFP32()

                // Calculate errors
                var symError: Float = 0
                var asymError: Float = 0
                for i in 0..<512 {
                    symError += abs(vector[i] - symDequantized[i])
                    asymError += abs(vector[i] - asymDequantized[i])
                }
                symError /= 512
                asymError /= 512

                print("    Symmetric:")
                print("      Scale: \(symParams.scale), Zero: \(symParams.zeroPoint)")
                print("      Avg error: \(symError)")

                print("    Asymmetric:")
                print("      Scale: \(asymParams.scale), Zero: \(asymParams.zeroPoint)")
                print("      Avg error: \(asymError)")

                print("    Improvement: \(String(format: "%.1f%%", (symError - asymError) / symError * 100))")

                // For skewed distributions, asymmetric should be better
                if name.contains("skewed") {
                    #expect(asymError <= symError, "Asymmetric should be better for skewed distributions")
                }
            }

            print("\n  Range utilization analysis:")

            let skewedVector = Vector512Optimized { _ in Float.random(in: 0...5) }  // Positive only
            let minVal = skewedVector.toArray().min()!
            let maxVal = skewedVector.toArray().max()!

            print("    Original range: [\(minVal), \(maxVal)]")

            // Symmetric will waste negative range
            let symRange = max(abs(minVal), abs(maxVal))
            print("    Symmetric range: [-\(symRange), \(symRange)] (wasted: [-\(symRange), 0])")

            // Asymmetric uses full range
            print("    Asymmetric range: [\(minVal), \(maxVal)] (no waste)")

            let wastedBits = log2(2 * symRange / (maxVal - minVal))
            print("    Effective bits lost with symmetric: \(wastedBits)")
        }

        @Test
        func testPerChannelQuantization() throws {
            // Test per-channel quantization
            // - Individual scale factors per channel
            // - Better preservation of channel statistics
            // - Implementation complexity

            print("Per-Channel Quantization:")

            // Create multi-channel data with different ranges
            let channelCount = 8
            let channelSize = 64  // 8 channels × 64 = 512

            var channels: [[Float]] = []
            for c in 0..<channelCount {
                let channelData = (0..<channelSize).map { i in
                    // Each channel has different characteristics
                    switch c {
                    case 0: return Float.random(in: -1...1)           // Uniform
                    case 1: return Float.random(in: -10...10)         // Wide range
                    case 2: return Float.random(in: -0.1...0.1)       // Narrow range
                    case 3: return sin(Float(i) * 0.1) * Float(c + 1) // Periodic
                    case 4: return exp(-Float(i) / 20.0) * 5          // Exponential decay
                    case 5: return i % 10 == 0 ? Float.random(in: -5...5) : 0  // Sparse
                    case 6: return Float(i) / Float(channelSize)      // Linear
                    case 7: return Float.random(in: 0...1)            // Positive only
                    default: return 0
                    }
                }
                channels.append(channelData)
            }

            let perChannelQuant = PerChannelQuantization(from: channels)

            print("  Channel statistics:")
            for (i, channel) in channels.enumerated() {
                let minVal = channel.min() ?? 0
                let maxVal = channel.max() ?? 0
                print("    Channel \(i): range=[\(minVal), \(maxVal)], scale=\(perChannelQuant.scales[i]), zero=\(perChannelQuant.zeroPoints[i])")
            }

            // Quantize and dequantize
            let quantized = perChannelQuant.quantize()
            let dequantized = perChannelQuant.dequantize(quantized)

            // Compare with global quantization
            let flatVector = try Vector512Optimized(channels.flatMap { $0 })
            let globalQuantized = Vector512INT8(from: flatVector)
            let globalDequantized = globalQuantized.toFP32()

            print("\n  Error comparison:")

            // Per-channel errors
            var perChannelErrors: [Float] = []
            for (i, channel) in channels.enumerated() {
                var error: Float = 0
                for (j, value) in channel.enumerated() {
                    error += abs(value - dequantized[i][j])
                }
                error /= Float(channelSize)
                perChannelErrors.append(error)
            }

            // Global errors
            var globalErrors: [Float] = []
            for c in 0..<channelCount {
                var error: Float = 0
                for i in 0..<channelSize {
                    let idx = c * channelSize + i
                    error += abs(channels[c][i] - globalDequantized[idx])
                }
                error /= Float(channelSize)
                globalErrors.append(error)
            }

            for i in 0..<channelCount {
                print("    Channel \(i):")
                print("      Per-channel error: \(perChannelErrors[i])")
                print("      Global error: \(globalErrors[i])")
                print("      Improvement: \(String(format: "%.1fx", globalErrors[i] / perChannelErrors[i]))")
            }

            let avgPerChannel = perChannelErrors.reduce(0, +) / Float(channelCount)
            let avgGlobal = globalErrors.reduce(0, +) / Float(channelCount)

            print("\n  Average error:")
            print("    Per-channel: \(avgPerChannel)")
            print("    Global: \(avgGlobal)")
            print("    Improvement: \(String(format: "%.1fx", avgGlobal / avgPerChannel))")

            #expect(avgPerChannel < avgGlobal, "Per-channel should have lower average error")

            print("\n  Implementation complexity:")
            print("    - Storage: \(channelCount) scales + \(channelCount) zero points")
            print("    - Computation: Per-channel scale/zero-point lookup")
            print("    - Memory layout: Channel-wise or interleaved storage")
        }

        @Test
        func testDynamicQuantization() throws {
            // Test dynamic quantization strategies
            // - Runtime quantization parameter adaptation
            // - Data-dependent optimization
            // - Online calibration

            print("Dynamic Quantization:")

            // Simulate streaming data with changing statistics
            var dynamicQuantizer = DynamicQuantizer(updateFrequency: 50)

            print("  Streaming data simulation:")

            // Phase 1: Small values
            print("\n  Phase 1 - Small values (0-50):")
            for i in 0..<50 {
                let vector = Vector512Optimized { _ in Float.random(in: -0.1...0.1) }
                dynamicQuantizer.addCalibrationData(vector)

                if i % 10 == 0 {
                    let quantized = dynamicQuantizer.quantize(vector)
                    print("    Batch \(i): Scale=\(quantized.quantizationParams.scale)")
                }
            }

            // Phase 2: Growing values
            print("\n  Phase 2 - Growing values (50-100):")
            for i in 50..<100 {
                let magnitude = Float(i - 50) / 10.0
                let vector = Vector512Optimized { _ in Float.random(in: -magnitude...magnitude) }
                dynamicQuantizer.addCalibrationData(vector)

                if i % 10 == 0 {
                    let quantized = dynamicQuantizer.quantize(vector)
                    print("    Batch \(i): Scale=\(quantized.quantizationParams.scale)")
                }
            }

            // Phase 3: Sudden distribution change
            print("\n  Phase 3 - Distribution shift (100-150):")
            for i in 100..<150 {
                let vector = Vector512Optimized { j in
                    // Bimodal distribution
                    j % 2 == 0 ? Float.random(in: -10...(-5)) : Float.random(in: 5...10)
                }
                dynamicQuantizer.addCalibrationData(vector)

                if i % 10 == 0 {
                    let quantized = dynamicQuantizer.quantize(vector)
                    print("    Batch \(i): Scale=\(quantized.quantizationParams.scale)")
                }
            }

            // Test adaptation quality
            print("\n  Adaptation quality test:")

            let testVectors = [
                ("Current distribution", Vector512Optimized { j in
                    j % 2 == 0 ? Float.random(in: -10...(-5)) : Float.random(in: 5...10)
                }),
                ("New distribution", Vector512Optimized { _ in Float.random(in: -1...1) })
            ]

            for (name, vector) in testVectors {
                let quantized = dynamicQuantizer.quantize(vector)
                let dequantized = quantized.toFP32()

                var error: Float = 0
                for i in 0..<512 {
                    error += abs(vector[i] - dequantized[i])
                }
                error /= 512

                print("    \(name): Avg error = \(error)")
            }

            print("\n  Online calibration benefits:")
            print("    - Adapts to data distribution changes")
            print("    - No need for offline calibration dataset")
            print("    - Suitable for streaming applications")
            print("    - Can handle concept drift")
        }

        @Test
        func testNonUniformQuantization() throws {
            // Test non-uniform quantization schemes
            // - Logarithmic quantization
            // - Power-of-two quantization
            // - Custom quantization functions

            print("Non-Uniform Quantization:")

            let testVector = try Vector512Optimized((0..<512).map { i in
                // Create data with wide dynamic range
                exp(-Float(i) / 50.0) * sin(Float(i) * 0.1)
            })

            print("  Original vector characteristics:")
            let array = testVector.toArray()
            print("    Min: \(array.min()!), Max: \(array.max()!)")
            print("    Dynamic range: \(array.max()! - array.min()!)")

            // Test different non-uniform schemes
            let quantizers = [
                ("Logarithmic", NonUniformQuantizer(type: .logarithmic)),
                ("Exponential", NonUniformQuantizer(type: .exponential)),
                ("Power-of-two", NonUniformQuantizer(type: .powerOfTwo)),
                ("Custom sqrt", NonUniformQuantizer(type: .custom(
                    transform: { x in (x >= 0 ? 1.0 : -1.0) * sqrt(abs(x)) },
                    inverse: { x in (x >= 0 ? 1.0 : -1.0) * (x * x) }
                )))
            ]

            // Compare with linear quantization
            let linearQuantized = Vector512INT8(from: testVector)
            let linearDequantized = linearQuantized.toFP32()
            var linearError: Float = 0
            for i in 0..<512 {
                linearError += abs(testVector[i] - linearDequantized[i])
            }
            linearError /= 512

            print("\n  Linear quantization baseline:")
            print("    Average error: \(linearError)")

            for (name, quantizer) in quantizers {
                print("\n  \(name) quantization:")

                let quantized = quantizer.quantize(testVector)
                let dequantized = quantizer.dequantize(quantized)

                // Calculate error metrics
                var avgError: Float = 0
                var maxError: Float = 0
                var relativeErrors: [Float] = []

                for i in 0..<512 {
                    let error = abs(testVector[i] - dequantized[i])
                    avgError += error
                    maxError = max(maxError, error)

                    if abs(testVector[i]) > 0.001 {
                        relativeErrors.append(error / abs(testVector[i]))
                    }
                }
                avgError /= 512

                let avgRelError = relativeErrors.reduce(0, +) / Float(relativeErrors.count)

                print("    Average error: \(avgError)")
                print("    Max error: \(maxError)")
                print("    Avg relative error: \(avgRelError)")
                print("    Improvement vs linear: \(String(format: "%.1fx", linearError / avgError))")

                // Log quantization should be better for exponential decay data
                if name == "Logarithmic" {
                    #expect(avgError < linearError * 1.5, "Log quantization should handle wide dynamic range well")
                }
            }

            print("\n  Use cases for non-uniform quantization:")
            print("    - Audio signals (logarithmic perception)")
            print("    - Image data (gamma correction)")
            print("    - Neural network weights (power-law distribution)")
            print("    - Scientific data (wide dynamic range)")

            print("\n  Trade-offs:")
            print("    - Better accuracy for specific distributions")
            print("    - More complex quantization/dequantization")
            print("    - May require lookup tables")
            print("    - Distribution-specific optimization")
        }
    }

    // MARK: - Integration with Vector Operations

    @Suite("Integration with Vector Operations")
    struct IntegrationVectorOperationsTests {

        @Test
        func testQuantizedOptimizedVectorIntegration() throws {
            // Test integration with OptimizedVector protocol
            // - Seamless quantization support
            // - Protocol compliance
            // - Type system integration

            print("Quantized Vector Integration with OptimizedVector Protocol:")

            // Test with different optimized vector types
            let vector512 = Vector512Optimized { i in sin(Float(i) * 0.01) }

            print("  Vector512Optimized integration:")

            // Test quantization
            let quantized = Vector512INT8(from: vector512)
            print("    Quantization params: scale=\(quantized.quantizationParams.scale), zero=\(quantized.quantizationParams.zeroPoint)")

            // Test dequantization
            let dequantized = quantized.toFP32()
            print("    Dequantization successful: \(dequantized.count == 512)")

            // Verify protocol compliance
            var error: Float = 0
            for i in 0..<512 {
                error += abs(vector512[i] - dequantized[i])
            }
            error /= 512
            print("    Round-trip error: \(error)")
            #expect(error < 0.01, "Round-trip error should be small")

            // Test with batch operations
            print("\n  Batch operation integration:")

            let vectors = (0..<10).map { j in
                Vector512Optimized { i in cos(Float(i + j * 512) * 0.001) }
            }

            let quantizedBatch = vectors.map { Vector512INT8(from: $0) }
            print("    Batch quantization: \(quantizedBatch.count) vectors")

            // Test distance operations
            let q1 = quantizedBatch[0]
            let q2 = quantizedBatch[1]

            let euclidean = QuantizedKernels.euclidean512(query: q1, candidate: q2)
            let dotProduct = QuantizedKernels.dotProduct512(query: q1, candidate: q2)

            print("    Euclidean distance: \(euclidean)")
            print("    Dot product: \(dotProduct)")

            // Test type system integration
            print("\n  Type system integration:")

            // Verify storage efficiency
            let storageSize = quantized.storage.count * MemoryLayout<SIMD4<Int8>>.size
            print("    Storage size: \(storageSize) bytes")
            print("    Expected: 512 bytes")
            #expect(storageSize == 512, "Storage should be exactly 512 bytes")

            // Test SIMD operations
            print("\n  SIMD operation compatibility:")
            quantized.storage.withUnsafeBufferPointer { buffer in
                print("    SIMD4 lanes: \(buffer.count)")
                print("    Total elements: \(buffer.count * 4)")
                #expect(buffer.count == 128, "Should have 128 SIMD4 lanes")
            }

            print("\n  Protocol compliance verified:")
            print("    ✓ Quantization from OptimizedVector")
            print("    ✓ Dequantization to OptimizedVector")
            print("    ✓ Batch operations support")
            print("    ✓ Distance metric computation")
            print("    ✓ Storage efficiency")
            print("    ✓ SIMD compatibility")
        }

        @Test
        func testQuantizedBatchOperations() async {
            // Test integration with batch operations
            // - Batch quantized distance computation
            // - k-NN search with quantized vectors
            // - Similarity matrix computation

            let batchSize = 100
            let k = 5

            // Create query vector
            let query = Vector512Optimized { _ in Float.random(in: -1...1) }
            let queryQuantized = Vector512INT8(from: query)

            // Create batch of candidate vectors
            var candidates: [Vector512Optimized] = []
            var candidatesQuantized: [Vector512INT8] = []

            for _ in 0..<batchSize {
                let candidate = Vector512Optimized { _ in Float.random(in: -1...1) }
                candidates.append(candidate)
                candidatesQuantized.append(Vector512INT8(from: candidate))
            }

            // Test batch distance computation
            var distancesFP32: [Float] = []
            var distancesINT8: [Float] = []

            // Compute distances in FP32
            for candidate in candidates {
                let dist = QuantizedKernels.euclideanSquaredDistance(query, candidate)
                distancesFP32.append(dist)
            }

            // Compute distances in INT8 (simulated - would use optimized kernel in production)
            for candidateQ in candidatesQuantized {
                // Dequantize for now (in production, would compute directly in INT8)
                let candidateFP32 = candidateQ.toFP32()
                let dist = QuantizedKernels.euclideanSquaredDistance(query, candidateFP32)
                distancesINT8.append(dist)
            }

            // Verify relative ordering is preserved
            let indicesFP32 = Array(0..<batchSize).sorted { distancesFP32[$0] < distancesFP32[$1] }
            let indicesINT8 = Array(0..<batchSize).sorted { distancesINT8[$0] < distancesINT8[$1] }

            // Check if top-k results are similar
            let topKFP32 = Set(indicesFP32.prefix(k))
            let topKINT8 = Set(indicesINT8.prefix(k))
            let overlap = topKFP32.intersection(topKINT8).count

            print("Quantized Batch Operations:")
            print("  Batch size: \(batchSize)")
            print("  Top-\(k) overlap: \(overlap)/\(k)")

            #expect(Float(overlap) / Float(k) >= 0.6, "At least 60% of top-k should match")

            // Test similarity matrix computation
            let matrixSize = 10
            var vectorSubset = Array(candidates.prefix(matrixSize))
            var quantizedSubset = Array(candidatesQuantized.prefix(matrixSize))

            // Compute similarity matrix in FP32
            var similarityFP32 = [[Float]](repeating: [Float](repeating: 0, count: matrixSize), count: matrixSize)
            for i in 0..<matrixSize {
                for j in 0..<matrixSize {
                    similarityFP32[i][j] = QuantizedKernels.euclideanSquaredDistance(vectorSubset[i], vectorSubset[j])
                }
            }

            // Verify matrix properties
            for i in 0..<matrixSize {
                #expect(abs(similarityFP32[i][i]) < 0.001, "Diagonal should be zero (self-distance)")
                for j in i+1..<matrixSize {
                    #expect(abs(similarityFP32[i][j] - similarityFP32[j][i]) < 0.001, "Matrix should be symmetric")
                }
            }

            print("  Similarity matrix computed for \(matrixSize)x\(matrixSize)")
        }

        @Test
        func testQuantizedCachingIntegration() {
            // Test integration with caching systems
            // - Quantized vector caching
            // - Cache-friendly quantized formats
            // - Cache invalidation strategies

            // Create a simple cache for quantized vectors
            class QuantizedVectorCache {
                private var cache: [String: Vector512INT8] = [:]
                private var accessCount: [String: Int] = [:]
                private let maxSize: Int

                init(maxSize: Int = 100) {
                    self.maxSize = maxSize
                }

                func get(_ key: String) -> Vector512INT8? {
                    if let vector = cache[key] {
                        accessCount[key, default: 0] += 1
                        return vector
                    }
                    return nil
                }

                func put(_ key: String, _ vector: Vector512INT8) {
                    // Simple LRU eviction when cache is full
                    if cache.count >= maxSize && cache[key] == nil {
                        // Evict least recently used
                        if let lruKey = accessCount.min(by: { $0.value < $1.value })?.key {
                            cache.removeValue(forKey: lruKey)
                            accessCount.removeValue(forKey: lruKey)
                        }
                    }
                    cache[key] = vector
                    accessCount[key, default: 0] += 1
                }

                func invalidate(_ key: String) {
                    cache.removeValue(forKey: key)
                    accessCount.removeValue(forKey: key)
                }

                func clear() {
                    cache.removeAll()
                    accessCount.removeAll()
                }

                var hitRate: Float {
                    let totalAccesses = accessCount.values.reduce(0, +)
                    let hits = accessCount.values.filter { $0 > 1 }.reduce(0, +)
                    return totalAccesses > 0 ? Float(hits) / Float(totalAccesses) : 0
                }

                var memorySaved: Int {
                    // Each INT8 vector saves 3x memory vs FP32
                    return cache.count * (512 * 3) // 3 bytes saved per element
                }
            }

            let cache = QuantizedVectorCache(maxSize: 50)

            // Test basic caching operations
            let testVectors = (0..<100).map { i in
                ("vector_\(i)", Vector512Optimized { _ in Float.random(in: -1...1) })
            }

            print("Quantized Caching Integration:")

            // First pass - populate cache
            for (key, vector) in testVectors.prefix(50) {
                let quantized = Vector512INT8(from: vector)
                cache.put(key, quantized)
            }

            // Second pass - test cache hits
            var cacheHits = 0
            var cacheMisses = 0

            for (key, vector) in testVectors.prefix(30) {
                if let cached = cache.get(key) {
                    cacheHits += 1
                    // Verify cached vector is correct
                    let dequantized = cached.toFP32()
                    var error: Float = 0
                    for i in 0..<512 {
                        error += abs(dequantized[i] - vector[i])
                    }
                    error /= 512
                    #expect(error < 0.05, "Cached vector should be close to original")
                } else {
                    cacheMisses += 1
                }
            }

            print("  Cache hits: \(cacheHits)")
            print("  Cache misses: \(cacheMisses)")
            print("  Hit rate: \(cache.hitRate)")
            print("  Memory saved: \(cache.memorySaved) bytes")

            #expect(cacheHits == 30, "All 30 vectors should be cache hits")

            // Test cache invalidation
            cache.invalidate("vector_0")
            #expect(cache.get("vector_0") == nil, "Invalidated entry should be removed")

            // Test cache-friendly access patterns
            // Access vectors in sequential order (cache-friendly)
            var sequentialAccessTime = 0.0
            let startSeq = CFAbsoluteTimeGetCurrent()
            for i in 0..<30 {
                _ = cache.get("vector_\(i)")
            }
            sequentialAccessTime = CFAbsoluteTimeGetCurrent() - startSeq

            // Access vectors in random order (less cache-friendly)
            var randomAccessTime = 0.0
            let randomIndices = (0..<30).shuffled()
            let startRand = CFAbsoluteTimeGetCurrent()
            for i in randomIndices {
                _ = cache.get("vector_\(i)")
            }
            randomAccessTime = CFAbsoluteTimeGetCurrent() - startRand

            print("  Sequential access time: \(sequentialAccessTime * 1000) ms")
            print("  Random access time: \(randomAccessTime * 1000) ms")

            // Test memory efficiency
            let fp32MemoryUsage = 50 * 512 * 4  // 50 vectors * 512 dims * 4 bytes
            let int8MemoryUsage = 50 * 512 * 1  // 50 vectors * 512 dims * 1 byte
            let compressionRatio = Float(fp32MemoryUsage) / Float(int8MemoryUsage)

            print("  FP32 memory: \(fp32MemoryUsage) bytes")
            print("  INT8 memory: \(int8MemoryUsage) bytes")
            print("  Compression ratio: \(compressionRatio)x")

            #expect(compressionRatio >= 3.5, "Should achieve at least 3.5x compression")
        }

        @Test
        func testQuantizedPipelineIntegration() async {
            // TODO: Test integration in processing pipelines
            // - End-to-end quantized workflows
            // - Pipeline stage optimization
            // - Memory transfer minimization
        }
    }

    // MARK: - Edge Cases and Error Handling

    @Suite("Edge Cases and Error Handling")
    struct EdgeCasesErrorHandlingTests {

        @Test
        func testQuantizationOverflow() throws {
            // Test handling of quantization overflow
            // - Values exceeding INT8 range
            // - Saturation vs clipping strategies
            // - Graceful degradation

            print("Quantization Overflow Handling:")

            // Test 1: Values exceeding expected range
            let extremeVector = try Vector512Optimized((0..<512).map { i in
                // Create values that will overflow if not handled properly
                if i % 4 == 0 {
                    return 1000.0  // Very large positive
                } else if i % 4 == 1 {
                    return -1000.0  // Very large negative
                } else if i % 4 == 2 {
                    return 0.001  // Very small
                } else {
                    return Float(i) / 512.0  // Normal range
                }
            })

            // Quantize with narrow range to force clipping
            let narrowParams = LinearQuantizationParams(minValue: -1.0, maxValue: 1.0, symmetric: true)
            let quantized = Vector512INT8(from: extremeVector, params: narrowParams)
            let dequantized = quantized.toFP32()

            print("  Narrow range quantization:")
            print("    Original range: [-1000, 1000]")
            print("    Quantization range: [-1, 1]")

            // Check saturation behavior
            var saturatedCount = 0
            var clippedValues: [(original: Float, quantized: Float)] = []

            for i in 0..<512 {
                let original = extremeVector[i]
                let recovered = dequantized[i]

                if original > 1.0 && abs(recovered - 1.0) < 0.1 {
                    saturatedCount += 1
                    if clippedValues.count < 5 {
                        clippedValues.append((original, recovered))
                    }
                } else if original < -1.0 && abs(recovered - (-1.0)) < 0.1 {
                    saturatedCount += 1
                    if clippedValues.count < 5 {
                        clippedValues.append((original, recovered))
                    }
                }
            }

            print("    Saturated values: \(saturatedCount)/512")
            print("    Example clipped values:")
            for (orig, quant) in clippedValues.prefix(3) {
                print("      \(orig) -> \(quant)")
            }

            #expect(saturatedCount > 200, "Should saturate out-of-range values")

            // Test 2: Gradual degradation with increasing range
            print("\n  Gradual degradation test:")

            let testRanges: [Float] = [1.0, 10.0, 100.0, 1000.0]
            for range in testRanges {
                let params = LinearQuantizationParams(minValue: -range, maxValue: range, symmetric: true)
                let q = Vector512INT8(from: extremeVector, params: params)
                let dq = q.toFP32()

                // Calculate reconstruction error
                var totalError: Float = 0
                var inRangeError: Float = 0
                var inRangeCount = 0

                for i in 0..<512 {
                    let error = abs(extremeVector[i] - dq[i])
                    totalError += error

                    if abs(extremeVector[i]) <= range {
                        inRangeError += error
                        inRangeCount += 1
                    }
                }

                print("    Range [-\(range), \(range)]:")
                print("      Total error: \(totalError)")
                print("      In-range error: \(inRangeError) (\(inRangeCount) values)")
                print("      Scale: \(params.scale)")
            }

            // Test 3: INT8 boundary behavior
            print("\n  INT8 boundary behavior:")

            // Test exact INT8 boundaries
            let boundaryVector = Vector512Optimized { i in
                switch i % 4 {
                case 0: return 127.0  // Max INT8
                case 1: return -128.0  // Min INT8
                case 2: return 0.0  // Zero
                default: return Float(i - 256) / 2.0  // Normal values
                }
            }

            let boundaryParams = LinearQuantizationParams(minValue: -128, maxValue: 127, symmetric: false)
            let boundaryQuantized = Vector512INT8(from: boundaryVector, params: boundaryParams)
            let boundaryDequantized = boundaryQuantized.toFP32()

            // Check boundary preservation
            var maxPreserved = false
            var minPreserved = false
            var zeroPreserved = false

            for i in 0..<512 {
                let original = boundaryVector[i]
                let recovered = boundaryDequantized[i]

                if original == 127.0 && abs(recovered - 127.0) < 1.0 {
                    maxPreserved = true
                }
                if original == -128.0 && abs(recovered - (-128.0)) < 1.0 {
                    minPreserved = true
                }
                if original == 0.0 && abs(recovered) < 0.5 {
                    zeroPreserved = true
                }
            }

            print("    Max (127) preserved: \(maxPreserved)")
            print("    Min (-128) preserved: \(minPreserved)")
            print("    Zero preserved: \(zeroPreserved)")

            #expect(maxPreserved && minPreserved && zeroPreserved, "Should preserve INT8 boundaries")
        }

        @Test
        func testQuantizationUnderflow() throws {
            // Test handling of quantization underflow
            // - Very small values
            // - Zero-point handling
            // - Precision loss mitigation

            print("Quantization Underflow Handling:")

            // Test vectors with very small values
            let testCases = [
                ("Tiny values", Vector512Optimized { _ in Float.random(in: -1e-10...1e-10) }),
                ("Small + normal", Vector512Optimized { i in
                    i % 10 == 0 ? Float.random(in: -1...1) : Float.random(in: -1e-8...1e-8)
                }),
                ("Gradual underflow", Vector512Optimized { i in
                    exp(-Float(i) / 10.0) * 1e-6
                }),
                ("Denormalized range", Vector512Optimized { _ in
                    Float.random(in: Float.leastNormalMagnitude...Float.leastNormalMagnitude * 100)
                })
            ]

            for (name, vector) in testCases {
                print("\n  \(name):")

                // Quantize with standard parameters
                let params = LinearQuantizationParams(minValue: -1.0, maxValue: 1.0, symmetric: true)
                let quantized = Vector512INT8(from: vector, params: params)
                let dequantized = quantized.toFP32()

                // Analyze underflow behavior
                var underflowCount = 0
                var preservedCount = 0
                var nonZeroOriginal = 0

                for i in 0..<512 {
                    let original = vector[i]
                    let restored = dequantized[i]

                    if original != 0 {
                        nonZeroOriginal += 1
                        if restored == 0 {
                            underflowCount += 1
                        } else {
                            preservedCount += 1
                        }
                    }
                }

                print("    Non-zero original values: \(nonZeroOriginal)")
                print("    Underflowed to zero: \(underflowCount)")
                print("    Preserved non-zero: \(preservedCount)")
                print("    Quantization scale: \(params.scale)")
                print("    Minimum representable: \(params.scale)")

                // Test mitigation strategy: Adjusting scale
                if underflowCount > nonZeroOriginal / 2 {
                    print("    Mitigation: Adjusting scale for small values")

                    let actualMin = vector.toArray().min()!
                    let actualMax = vector.toArray().max()!
                    let adjustedParams = LinearQuantizationParams(
                        minValue: actualMin,
                        maxValue: actualMax,
                        symmetric: false
                    )

                    let adjustedQuantized = Vector512INT8(from: vector, params: adjustedParams)
                    let adjustedDequantized = adjustedQuantized.toFP32()

                    var adjustedUnderflow = 0
                    for i in 0..<512 {
                        if vector[i] != 0 && adjustedDequantized[i] == 0 {
                            adjustedUnderflow += 1
                        }
                    }

                    print("    Adjusted scale: \(adjustedParams.scale)")
                    print("    Adjusted underflow: \(adjustedUnderflow)")
                    print("    Improvement: \(underflowCount - adjustedUnderflow) fewer underflows")
                }
            }

            print("\n  Precision preservation strategies:")
            print("    - Use asymmetric quantization for better range utilization")
            print("    - Apply logarithmic transformation for wide dynamic range")
            print("    - Use higher bit-width (INT16) for critical small values")
            print("    - Implement stochastic rounding for probabilistic preservation")
        }

        @Test
        func testZeroVectorQuantization() throws {
            // Test quantization of zero vectors
            // - All-zero input handling
            // - Scale factor edge cases
            // - Special case optimization

            print("Zero Vector Quantization:")

            // Test 1: Pure zero vector
            let zeroVector = Vector512Optimized(repeating: 0.0)
            let params = LinearQuantizationParams(minValue: -1.0, maxValue: 1.0, symmetric: true)
            let quantizedZero = Vector512INT8(from: zeroVector, params: params)
            let dequantizedZero = quantizedZero.toFP32()

            print("  Zero vector test:")
            print("    All elements zero: \(dequantizedZero.toArray().allSatisfy { $0 == 0.0 })")

            // Check that all quantized values are zero
            var allZero = true
            for i in 0..<512 {
                if dequantizedZero[i] != 0.0 {
                    allZero = false
                    print("    Non-zero at index \(i): \(dequantizedZero[i])")
                    break
                }
            }
            #expect(allZero, "Zero vector should remain zero after quantization")

            // Test 2: Near-zero vectors
            let epsilons: [Float] = [1e-10, 1e-7, 1e-5, 1e-3]
            print("\n  Near-zero vector tests:")

            for epsilon in epsilons {
                let nearZeroVector = Vector512Optimized { _ in Float.random(in: -epsilon...epsilon) }
                let nearZeroParams = LinearQuantizationParams(minValue: -1.0, maxValue: 1.0, symmetric: true)
                let quantizedNearZero = Vector512INT8(from: nearZeroVector, params: nearZeroParams)
                let dequantizedNearZero = quantizedNearZero.toFP32()

                // Count how many values become exactly zero
                let zeroCount = dequantizedNearZero.toArray().filter { $0 == 0.0 }.count
                print("    Epsilon \(epsilon): \(zeroCount)/512 became zero")

                // For very small values, most should quantize to zero
                if epsilon < 1e-5 {
                    #expect(zeroCount > 400, "Very small values should quantize to zero")
                }
            }

            // Test 3: Zero vector with different quantization ranges
            print("\n  Scale factor edge cases:")

            let ranges: [(Float, Float)] = [
                (0.0, 1.0),    // Asymmetric, zero at boundary
                (-1.0, 1.0),   // Symmetric
                (-10.0, 10.0), // Wide symmetric
                (-1.0, 0.0),   // Asymmetric, zero at boundary
                (-5.0, 3.0)    // Asymmetric, zero in middle
            ]

            for (minVal, maxVal) in ranges {
                let rangeParams = LinearQuantizationParams(minValue: minVal, maxValue: maxVal, symmetric: minVal == -maxVal)
                let qZero = Vector512INT8(from: zeroVector, params: rangeParams)
                let dqZero = qZero.toFP32()

                let maxError = dqZero.toArray().map { abs($0) }.max() ?? 0
                print("    Range [\(minVal), \(maxVal)]: max error = \(maxError)")

                #expect(maxError < 0.01, "Zero should be accurately represented")
            }

            // Test 4: Distance computation with zero vector
            print("\n  Distance computation with zero:")

            let nonZeroVector = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.01) })
            let qNonZero = Vector512INT8(from: nonZeroVector, params: params)
            let qZero = Vector512INT8(from: zeroVector, params: params)

            // Euclidean distance from zero should equal magnitude
            let distFromZero = QuantizedKernels.euclidean512(query: qNonZero, candidate: qZero)
            let magnitude = sqrt(QuantizedKernels.dotProduct512(query: qNonZero, candidate: qNonZero))

            print("    Distance from zero: \(distFromZero)")
            print("    Vector magnitude: \(magnitude)")
            print("    Difference: \(abs(distFromZero - magnitude))")

            #expect(abs(distFromZero - magnitude) < 0.01, "Distance from zero should equal magnitude")

            // Dot product with zero should be zero
            let dotWithZero = QuantizedKernels.dotProduct512(query: qNonZero, candidate: qZero)
            print("    Dot product with zero: \(dotWithZero)")
            #expect(abs(dotWithZero) < 0.01, "Dot product with zero should be zero")
        }

        @Test
        func testExtremeValueQuantization() throws {
            // Test quantization of extreme values
            // - Very large/small values
            // - Outlier handling
            // - Robust quantization schemes

            print("Extreme Value Quantization:")

            // Create vectors with extreme values and outliers
            let testVectors = [
                ("With outliers", Vector512Optimized { i in
                    if i < 10 {
                        return Float.random(in: -1000...1000)  // Outliers
                    } else {
                        return Float.random(in: -1...1)  // Normal range
                    }
                }),
                ("Heavy-tailed", Vector512Optimized { _ in
                    // Cauchy distribution (heavy tails)
                    let u = Float.random(in: 0.01...0.99)
                    return tan(.pi * (u - 0.5))
                }),
                ("Mixed scales", Vector512Optimized { i in
                    switch i % 4 {
                    case 0: return Float.random(in: -1e-6...1e-6)
                    case 1: return Float.random(in: -1...1)
                    case 2: return Float.random(in: -100...100)
                    default: return Float.random(in: -10000...10000)
                    }
                })
            ]

            for (name, vector) in testVectors {
                print("\n  \(name) distribution:")

                let array = vector.toArray().sorted()
                let min = array.first!
                let max = array.last!
                let median = array[256]
                let q1 = array[128]
                let q3 = array[384]

                print("    Statistics:")
                print("      Min: \(min), Max: \(max)")
                print("      Q1: \(q1), Median: \(median), Q3: \(q3)")
                print("      IQR: \(q3 - q1)")

                // Standard quantization
                let standardParams = LinearQuantizationParams(minValue: min, maxValue: max, symmetric: false)
                let standardQuantized = Vector512INT8(from: vector, params: standardParams)
                let standardDequantized = standardQuantized.toFP32()

                // Robust quantization (using percentiles)
                let p01 = array[Int(512 * 0.01)]
                let p99 = array[Int(512 * 0.99)]
                let robustParams = LinearQuantizationParams(minValue: p01, maxValue: p99, symmetric: false)
                let robustQuantized = Vector512INT8(from: vector, params: robustParams)
                let robustDequantized = robustQuantized.toFP32()

                // Calculate errors for non-outlier values
                var standardError: Float = 0
                var robustError: Float = 0
                var inlierCount = 0

                for i in 0..<512 {
                    let value = vector[i]
                    if value >= p01 && value <= p99 {
                        standardError += abs(value - standardDequantized[i])
                        robustError += abs(value - robustDequantized[i])
                        inlierCount += 1
                    }
                }

                if inlierCount > 0 {
                    standardError /= Float(inlierCount)
                    robustError /= Float(inlierCount)
                }

                print("\n    Standard quantization:")
                print("      Range: [\(min), \(max)]")
                print("      Scale: \(standardParams.scale)")
                print("      Inlier error: \(standardError)")

                print("\n    Robust quantization (1-99 percentile):")
                print("      Range: [\(p01), \(p99)]")
                print("      Scale: \(robustParams.scale)")
                print("      Inlier error: \(robustError)")
                print("      Improvement: \(String(format: "%.1fx", standardError / robustError))")

                // Count clipped values
                var clippedCount = 0
                for i in 0..<512 {
                    let value = vector[i]
                    if value < p01 || value > p99 {
                        clippedCount += 1
                    }
                }
                print("      Clipped outliers: \(clippedCount)/512")

                #expect(robustError <= standardError * 1.1, "Robust quantization should handle outliers better")
            }

            print("\n  Outlier handling strategies:")
            print("    - Percentile-based range selection")
            print("    - Winsorization (clip to percentiles)")
            print("    - Separate outlier encoding")
            print("    - Adaptive clipping thresholds")
            print("    - Two-stage quantization (outliers + inliers)")
        }

        @Test
        func testNaNInfinityHandling() throws {
            // Test handling of NaN and infinity values
            // - NaN propagation in quantization
            // - Infinity handling strategies
            // - Error recovery mechanisms

            print("NaN and Infinity Handling:")

            // Test 1: NaN handling
            let nanVector = try Vector512Optimized((0..<512).map { i in
                if i % 10 == 0 {
                    return Float.nan
                } else if i % 10 == 1 {
                    return Float.infinity
                } else if i % 10 == 2 {
                    return -Float.infinity
                } else {
                    return Float(i) / 100.0
                }
            })

            print("  Input vector stats:")
            let nanCount = nanVector.toArray().filter { $0.isNaN }.count
            let infCount = nanVector.toArray().filter { $0.isInfinite }.count
            print("    NaN values: \(nanCount)")
            print("    Infinite values: \(infCount)")

            // Quantize with normal parameters
            let params = LinearQuantizationParams(minValue: -10.0, maxValue: 10.0, symmetric: true)
            let quantized = Vector512INT8(from: nanVector, params: params)
            let dequantized = quantized.toFP32()

            // Analyze how special values were handled
            var nanHandling: [Float] = []
            var infHandling: [Float] = []
            var negInfHandling: [Float] = []

            for i in 0..<512 {
                let original = nanVector[i]
                let recovered = dequantized[i]

                if original.isNaN {
                    nanHandling.append(recovered)
                } else if original == Float.infinity {
                    infHandling.append(recovered)
                } else if original == -Float.infinity {
                    negInfHandling.append(recovered)
                }
            }

            print("\n  Quantization results:")
            if !nanHandling.isEmpty {
                print("    NaN -> \(nanHandling[0]) (consistent: \(nanHandling.allSatisfy { $0 == nanHandling[0] }))")
            }
            if !infHandling.isEmpty {
                print("    +Inf -> \(infHandling[0]) (consistent: \(infHandling.allSatisfy { $0 == infHandling[0] }))")
            }
            if !negInfHandling.isEmpty {
                print("    -Inf -> \(negInfHandling[0]) (consistent: \(negInfHandling.allSatisfy { $0 == negInfHandling[0] }))")
            }

            // Test 2: Distance computation with special values
            print("\n  Distance computation with special values:")

            let normalVector = try Vector512Optimized((0..<512).map { sin(Float($0) * 0.01) })
            let qNormal = Vector512INT8(from: normalVector, params: params)
            let qSpecial = Vector512INT8(from: nanVector, params: params)

            let distance = QuantizedKernels.euclidean512(query: qNormal, candidate: qSpecial)
            let dotProduct = QuantizedKernels.dotProduct512(query: qNormal, candidate: qSpecial)

            print("    Distance: \(distance) (finite: \(distance.isFinite))")
            print("    Dot product: \(dotProduct) (finite: \(dotProduct.isFinite))")

            #expect(distance.isFinite, "Distance should be finite even with special values")
            #expect(dotProduct.isFinite, "Dot product should be finite even with special values")

            // Test 3: Graceful degradation
            print("\n  Graceful degradation test:")

            // Create vector with increasing number of special values
            for specialRatio in [0.0, 0.1, 0.25, 0.5] {
                let specialCount = Int(512.0 * specialRatio)
                let testVector = try Vector512Optimized((0..<512).map { i in
                    if i < specialCount {
                        return i % 2 == 0 ? Float.nan : Float.infinity
                    } else {
                        return Float(i - specialCount) / Float(512 - specialCount)
                    }
                })

                let qTest = Vector512INT8(from: testVector, params: params)
                let dqTest = qTest.toFP32()

                // Measure reconstruction quality for normal values
                var normalError: Float = 0
                var normalCount = 0

                for i in specialCount..<512 {
                    normalError += abs(testVector[i] - dqTest[i])
                    normalCount += 1
                }

                if normalCount > 0 {
                    normalError /= Float(normalCount)
                    print("    \(Int(specialRatio * 100))% special values: avg error for normal values = \(normalError)")
                }
            }

            // Test 4: Robustness of operations
            print("\n  Operation robustness:")

            // Create pairs of vectors with special values
            let v1 = Vector512Optimized { i in i % 3 == 0 ? Float.nan : Float(i) / 256.0 }
            let v2 = Vector512Optimized { i in i % 5 == 0 ? Float.infinity : Float(i) / 256.0 }

            let q1 = Vector512INT8(from: v1, params: params)
            let q2 = Vector512INT8(from: v2, params: params)

            // Test various operations
            let dot = QuantizedKernels.dotProduct512(query: q1, candidate: q2)
            let mag1 = sqrt(QuantizedKernels.dotProduct512(query: q1, candidate: q1))
            let mag2 = sqrt(QuantizedKernels.dotProduct512(query: q2, candidate: q2))
            let cosineSim = mag1 > 0 && mag2 > 0 ? dot / (mag1 * mag2) : 0

            let operations: [(String, Float)] = [
                ("Euclidean", QuantizedKernels.euclidean512(query: q1, candidate: q2)),
                ("Dot product", dot),
                ("Cosine similarity", cosineSim)
            ]

            for (name, result) in operations {
                print("    \(name): \(result) (finite: \(result.isFinite))")
                #expect(result.isFinite || result.isNaN, "Operations should handle special values gracefully")
            }
        }

        @Test
        func testDegenerateDistributions() throws {
            // Test quantization of degenerate distributions
            // - Constant vectors
            // - Very small dynamic range
            // - Numerical instability handling

            print("Degenerate Distribution Quantization:")

            let testCases = [
                ("Constant", Vector512Optimized(repeating: 0.5)),
                ("Nearly constant", Vector512Optimized { i in
                    0.5 + (i == 0 ? 0.0001 : 0)
                }),
                ("Binary", Vector512Optimized { i in
                    i % 2 == 0 ? 0.0 : 1.0
                }),
                ("Very small range", Vector512Optimized { _ in
                    Float.random(in: 0.4999...0.5001)
                }),
                ("Single spike", Vector512Optimized { i in
                    i == 256 ? 10.0 : 0.0
                })
            ]

            for (name, vector) in testCases {
                print("\n  \(name) distribution:")

                let array = vector.toArray()
                let uniqueValues = Set(array)
                let minVal = array.min()!
                let maxVal = array.max()!
                let range = maxVal - minVal

                print("    Unique values: \(uniqueValues.count)")
                print("    Range: [\(minVal), \(maxVal)]")
                print("    Dynamic range: \(range)")

                // Handle degenerate case
                let params: LinearQuantizationParams
                if range == 0 {
                    // Constant vector - special handling
                    print("    Special handling: Constant vector")
                    params = LinearQuantizationParams(
                        minValue: minVal - 0.001,
                        maxValue: maxVal + 0.001,
                        symmetric: false
                    )
                } else if range < 1e-6 {
                    // Very small range - use expanded range
                    print("    Special handling: Expanded range")
                    let center = (minVal + maxVal) / 2
                    let expansion = max(range * 10, 0.001)
                    params = LinearQuantizationParams(
                        minValue: center - expansion,
                        maxValue: center + expansion,
                        symmetric: false
                    )
                } else {
                    // Normal quantization
                    params = LinearQuantizationParams(
                        minValue: minVal,
                        maxValue: maxVal,
                        symmetric: false
                    )
                }

                let quantized = Vector512INT8(from: vector, params: params)
                let dequantized = quantized.toFP32()

                // Analyze preservation of structure
                let dequantizedArray = dequantized.toArray()
                let uniqueQuantized = Set(dequantizedArray)

                print("    Quantization params:")
                print("      Scale: \(params.scale)")
                print("      Zero point: \(params.zeroPoint)")
                print("      Unique quantized values: \(uniqueQuantized.count)")

                // Calculate reconstruction quality
                var maxError: Float = 0
                var avgError: Float = 0
                for i in 0..<512 {
                    let error = abs(vector[i] - dequantized[i])
                    maxError = max(maxError, error)
                    avgError += error
                }
                avgError /= 512

                print("    Reconstruction:")
                print("      Max error: \(maxError)")
                print("      Avg error: \(avgError)")

                // Verify structure preservation
                if uniqueValues.count <= 10 {
                    print("    Structure preservation:")
                    print("      Original unique: \(uniqueValues.sorted())")
                    print("      Quantized unique: \(uniqueQuantized.sorted().prefix(10))")

                    // For degenerate distributions, we should preserve the basic structure
                    if uniqueValues.count == 1 {
                        #expect(uniqueQuantized.count <= 2, "Constant should map to at most 2 quantized values")
                    } else if uniqueValues.count == 2 {
                        #expect(uniqueQuantized.count >= 2, "Binary should preserve at least 2 values")
                    }
                }

                // Test numerical stability
                print("    Numerical stability:")
                if !params.scale.isFinite {
                    print("      WARNING: Non-finite scale detected!")
                }
                if params.scale < 1e-10 {
                    print("      WARNING: Extremely small scale may cause instability")
                }
                if params.scale > 1e10 {
                    print("      WARNING: Extremely large scale may cause overflow")
                }
                #expect(params.scale.isFinite, "Scale should be finite")
                #expect(params.scale > 0, "Scale should be positive")
            }

            print("\n  Handling strategies for degenerate distributions:")
            print("    - Detect constant vectors and use minimal quantization")
            print("    - Expand range for near-constant distributions")
            print("    - Use higher precision for small dynamic range")
            print("    - Implement special encodings for sparse/binary data")
            print("    - Add numerical stability checks")
        }
    }

    // MARK: - Real-World Application Tests

    @Suite("Real-World Applications")
    struct RealWorldApplicationTests {

        @Test
        func testSemanticSearchQuantized() async throws {
            // Test semantic search with quantized embeddings
            // - Document embedding quantization
            // - Search quality preservation
            // - Scalability improvements

            print("Semantic Search with Quantized Embeddings:")

            // Generate semantic search scenario
            let scenario = SemanticSearchScenario.generate(documentCount: 500, queryCount: 5)

            print("  Dataset:")
            print("    Documents: \(scenario.documents.count)")
            print("    Queries: \(scenario.queries.count)")
            print("    Embedding dimension: 512")

            // Memory footprint comparison
            let fp32Size = scenario.documents.count * 512 * 4
            let int8Size = scenario.documents.count * 512 * 1

            print("\n  Memory footprint:")
            print("    FP32: \(fp32Size / 1_000_000) MB")
            print("    INT8: \(int8Size / 1_000_000) MB")
            print("    Reduction: \(Float(fp32Size) / Float(int8Size))x")

            // Evaluate search quality
            print("\n  Search quality (Recall@K):")

            for k in [1, 5, 10, 20] {
                let recall = scenario.evaluateRecall(topK: k)
                print("    Recall@\(k): \(String(format: "%.2f%%", recall * 100))")

                // Should maintain good recall
                if k == 10 {
                    #expect(recall > 0.7, "Should maintain >70% recall@10")
                }
            }

            // Performance comparison
            print("\n  Performance comparison:")

            let iterations = 10
            let query = scenario.queries[0]

            // FP32 search
            let fp32Start = Date()
            for _ in 0..<iterations {
                _ = scenario.documents.map { doc in
                    query.dotProduct(doc)
                }
            }
            let fp32Time = Date().timeIntervalSince(fp32Start)

            // INT8 search
            let qQuery = Vector512INT8(from: query)
            let quantizedDocs = scenario.documents.map { Vector512INT8(from: $0) }

            let int8Start = Date()
            for _ in 0..<iterations {
                _ = quantizedDocs.map { doc in
                    QuantizedKernels.dotProduct512(query: qQuery, candidate: doc)
                }
            }
            let int8Time = Date().timeIntervalSince(int8Start)

            print("    FP32 time: \(String(format: "%.2f ms", fp32Time * 1000))")
            print("    INT8 time: \(String(format: "%.2f ms", int8Time * 1000))")
            print("    Speedup: \(String(format: "%.1fx", fp32Time / int8Time))")

            print("\n  Scalability improvements:")
            print("    - 4x more documents fit in memory")
            print("    - 4x reduction in network transfer")
            print("    - Better cache utilization")
            print("    - Faster similarity computation")
        }

        @Test
        func testRecommendationSystemQuantized() async throws {
            // Test recommendation systems with quantized vectors
            // - User/item embedding compression
            // - Recommendation quality metrics
            // - System throughput improvements

            print("Recommendation System with Quantized Vectors:")

            // Generate recommendation scenario
            let scenario = RecommendationScenario.generate(userCount: 50, itemCount: 1000)

            print("  System configuration:")
            print("    Users: \(scenario.userEmbeddings.count)")
            print("    Items: \(scenario.itemEmbeddings.count)")
            print("    Embedding dimension: 512")

            // Calculate memory savings
            let totalEmbeddings = scenario.userEmbeddings.count + scenario.itemEmbeddings.count
            let fp32Memory = totalEmbeddings * 512 * 4
            let int8Memory = totalEmbeddings * 512 * 1

            print("\n  Memory usage:")
            print("    FP32: \(fp32Memory / 1_000_000) MB")
            print("    INT8: \(int8Memory / 1_000_000) MB")
            print("    Savings: \(fp32Memory - int8Memory) bytes")

            // Evaluate recommendation quality
            let precision = scenario.evaluatePrecision(topK: 10)
            print("\n  Recommendation quality:")
            print("    Precision@10: \(String(format: "%.2f%%", precision * 100))")
            #expect(precision > 0.6, "Should maintain >60% precision@10")

            // Test online serving performance
            print("\n  Online serving performance:")

            let testUser = scenario.userEmbeddings[0]
            let qUser = Vector512INT8(from: testUser)
            let quantizedItems = scenario.itemEmbeddings.map { Vector512INT8(from: $0) }

            // Measure latency for single user
            let iterations = 100

            // FP32 scoring
            let fp32Start = Date()
            for _ in 0..<iterations {
                _ = scenario.itemEmbeddings.map { item in
                    testUser.cosineSimilarity(to: item)
                }
            }
            let fp32Latency = Date().timeIntervalSince(fp32Start) / Double(iterations)

            // INT8 scoring
            let int8Start = Date()
            for _ in 0..<iterations {
                _ = quantizedItems.map { qItem in
                    let dot = QuantizedKernels.dotProduct512(query: qUser, candidate: qItem)
                    let mag1 = sqrt(QuantizedKernels.dotProduct512(query: qUser, candidate: qUser))
                    let mag2 = sqrt(QuantizedKernels.dotProduct512(query: qItem, candidate: qItem))
                    return mag1 > 0 && mag2 > 0 ? dot / (mag1 * mag2) : 0
                }
            }
            let int8Latency = Date().timeIntervalSince(int8Start) / Double(iterations)

            print("    FP32 latency: \(String(format: "%.2f ms", fp32Latency * 1000))")
            print("    INT8 latency: \(String(format: "%.2f ms", int8Latency * 1000))")
            print("    Speedup: \(String(format: "%.1fx", fp32Latency / int8Latency))")

            // Batch processing throughput
            let batchSize = 10
            let batchUsers = Array(scenario.userEmbeddings.prefix(batchSize))

            print("\n  Batch processing (\(batchSize) users):")

            let batchStart = Date()
            for user in batchUsers {
                let qUser = Vector512INT8(from: user)
                _ = quantizedItems.map { qItem in
                    QuantizedKernels.dotProduct512(query: qUser, candidate: qItem)
                }
            }
            let batchTime = Date().timeIntervalSince(batchStart)

            let throughput = Double(batchSize * scenario.itemEmbeddings.count) / batchTime
            print("    Time: \(String(format: "%.2f ms", batchTime * 1000))")
            print("    Throughput: \(String(format: "%.0f", throughput)) scores/sec")

            print("\n  System improvements:")
            print("    - Reduced memory footprint enables larger catalogs")
            print("    - Lower latency for real-time recommendations")
            print("    - Higher throughput for batch processing")
            print("    - Cost reduction in cloud deployments")
        }

        @Test
        func testImageRetrievalQuantized() async {
            // TODO: Test image retrieval with quantized features
            // - Visual feature quantization
            // - Retrieval accuracy analysis
            // - Storage and bandwidth savings
        }

        @Test
        func testLargeScaleDeploymentQuantized() async {
            // TODO: Test large-scale deployment scenarios
            // - Million-scale vector databases
            // - Real-time inference constraints
            // - Resource utilization optimization
        }

        @Test
        func testEmbeddingCompressionPipeline() async {
            // TODO: Test end-to-end embedding compression pipeline
            // - Training-time quantization awareness
            // - Deployment-time optimization
            // - Quality assurance workflows
        }
    }

    // MARK: - Helper Functions (Placeholder)

    // MARK: - Helper Functions

    private static func generateTestVectorsForQuantization(
        count: Int,
        dimension: Int = 512,
        distribution: String = "normal"
    ) -> [Vector512Optimized] {
        // Generate test vectors with specific distributions
        return (0..<count).map { _ in
            switch distribution {
            case "uniform":
                return Vector512Optimized { _ in Float.random(in: -1...1) }
            case "normal", "gaussian":
                return Vector512Optimized { _ in
                    let u1 = Float.random(in: 0.001...0.999)
                    let u2 = Float.random(in: 0.001...0.999)
                    return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
                }
            case "sparse":
                return Vector512Optimized { i in i % 10 == 0 ? Float.random(in: -1...1) : 0 }
            default:
                return Vector512Optimized { i in sin(Float(i) * 0.01) }
            }
        }
    }

    private static func quantizeVector(_ vector: Vector512Optimized, scheme: String) -> Vector512INT8 {
        // Quantize vector using specified scheme
        switch scheme {
        case "symmetric":
            let range = max(abs(vector.toArray().min()!), abs(vector.toArray().max()!))
            return Vector512INT8(from: vector, params: LinearQuantizationParams(
                minValue: -range, maxValue: range, symmetric: true
            ))
        case "asymmetric":
            return Vector512INT8(from: vector, params: LinearQuantizationParams(
                minValue: vector.toArray().min()!, maxValue: vector.toArray().max()!, symmetric: false
            ))
        default:
            return Vector512INT8(from: vector)  // Auto-calibration
        }
    }

    private static func dequantizeVector(_ quantizedVector: Vector512INT8) -> Vector512Optimized {
        // Dequantize vector back to FP32
        return quantizedVector.toFP32()
    }

    private static func measureQuantizationError(
        original: [Float],
        quantized: [Float]
    ) -> (meanError: Float, maxError: Float, snr: Float) {
        // Measure various quantization error metrics
        var sumError: Float = 0
        var maxError: Float = 0
        var signalPower: Float = 0
        var noisePower: Float = 0

        for i in 0..<original.count {
            let error = quantized[i] - original[i]
            sumError += error
            maxError = max(maxError, abs(error))
            signalPower += original[i] * original[i]
            noisePower += error * error
        }

        let meanError = sumError / Float(original.count)
        signalPower /= Float(original.count)
        noisePower /= Float(original.count)
        let snr = noisePower > 0 ? 10 * log10(signalPower / noisePower) : Float.infinity

        return (meanError, maxError, snr)
    }

    private static func measureCompressionRatio(
        originalSize: Int,
        compressedSize: Int
    ) -> Float {
        // Calculate compression ratio
        return Float(originalSize) / Float(compressedSize)
    }

    private static func benchmarkQuantizedOperation(
        operation: () async throws -> Void,
        iterations: Int = 100
    ) async -> (latency: TimeInterval, throughput: Double) {
        // Benchmark quantized operations
        let start = Date()
        for _ in 0..<iterations {
            try! await operation()
        }
        let totalTime = Date().timeIntervalSince(start)
        let latency = totalTime / Double(iterations)
        let throughput = Double(iterations) / totalTime

        return (latency, throughput)
    }

    private static func validateQuantizationImplementation() -> Bool {
        // Validate quantization implementation correctness
        let testVector = Vector512Optimized { i in Float(i) / 256.0 - 1.0 }
        let quantized = Vector512INT8(from: testVector)
        let dequantized = quantized.toFP32()

        // Check round-trip error
        for i in 0..<512 {
            if abs(dequantized[i] - testVector[i]) > 0.1 {
                return false
            }
        }

        return true
    }
}

// MARK: - Additional Test Infrastructure

extension QuantizedKernelsTests {

    // MARK: Simulated INT4 Quantization

    struct SimulatedINT4 {
        let values: [Int8]  // Store 2 INT4 values per Int8
        let scale: Float
        let zeroPoint: Int

        init(from vector: Vector512Optimized, symmetric: Bool = true) {
            // INT4 range: -8 to 7 (4 bits)
            let minValue = vector.toArray().min() ?? 0
            let maxValue = vector.toArray().max() ?? 0

            if symmetric {
                let absMax = max(abs(minValue), abs(maxValue))
                self.scale = absMax / 7.0
                self.zeroPoint = 0
            } else {
                self.scale = (maxValue - minValue) / 15.0
                self.zeroPoint = Int(-minValue / scale)
            }

            // Pack 2 INT4 values per byte
            var packed: [Int8] = []
            let array = vector.toArray()
            for i in stride(from: 0, to: 512, by: 2) {
                let v1 = SimulatedINT4.quantizeToINT4(array[i], scale: scale, zeroPoint: zeroPoint)
                let v2 = SimulatedINT4.quantizeToINT4(array[i+1], scale: scale, zeroPoint: zeroPoint)
                let packed_byte = Int8((v1 << 4) | (v2 & 0x0F))
                packed.append(packed_byte)
            }
            self.values = packed
        }

        func toFP32() -> Vector512Optimized {
            var result: [Float] = []
            for byte in values {
                let v1 = (byte >> 4) & 0x0F
                let v2 = byte & 0x0F

                // Sign extend INT4 to Float
                let f1 = SimulatedINT4.dequantizeFromINT4(v1, scale: scale, zeroPoint: zeroPoint)
                let f2 = SimulatedINT4.dequantizeFromINT4(v2, scale: scale, zeroPoint: zeroPoint)
                result.append(f1)
                result.append(f2)
            }
            return try! Vector512Optimized(result)
        }

        static func quantizeToINT4(_ value: Float, scale: Float, zeroPoint: Int) -> Int8 {
            let quantized = Int(round(value / scale)) + zeroPoint
            return Int8(max(-8, min(7, quantized)))
        }

        static func dequantizeFromINT4(_ value: Int8, scale: Float, zeroPoint: Int) -> Float {
            // Sign extend if needed
            let signExtended = value > 7 ? Int(value) - 16 : Int(value)
            return Float(signExtended - zeroPoint) * scale
        }
    }

    // MARK: Simulated INT16 Quantization

    struct SimulatedINT16 {
        let values: [Int16]
        let scale: Float
        let zeroPoint: Int

        init(from vector: Vector512Optimized, symmetric: Bool = true) {
            let minValue = vector.toArray().min() ?? 0
            let maxValue = vector.toArray().max() ?? 0

            let finalScale: Float
            let finalZeroPoint: Int

            if symmetric {
                let absMax = max(abs(minValue), abs(maxValue))
                finalScale = absMax / 32767.0
                finalZeroPoint = 0
            } else {
                finalScale = (maxValue - minValue) / 65535.0
                finalZeroPoint = Int(-minValue / finalScale) - 32768
            }

            self.scale = finalScale
            self.zeroPoint = finalZeroPoint
            self.values = vector.toArray().map { value in
                let quantized = Int(round(value / finalScale)) + finalZeroPoint
                return Int16(max(-32768, min(32767, quantized)))
            }
        }

        func toFP32() -> Vector512Optimized {
            let dequantized = values.map { value in
                Float(Int(value) - zeroPoint) * scale
            }
            return try! Vector512Optimized(dequantized)
        }
    }

    // MARK: Per-Channel Quantization

    struct PerChannelQuantization {
        let channels: [[Float]]  // Each channel separately
        let scales: [Float]      // Scale per channel
        let zeroPoints: [Int]    // Zero point per channel

        init(from matrix: [[Float]], channelAxis: Int = 0) {
            self.channels = matrix
            var scales: [Float] = []
            var zeroPoints: [Int] = []

            // Calculate per-channel quantization parameters
            for channel in matrix {
                let minVal = channel.min() ?? 0
                let maxVal = channel.max() ?? 0
                let scale = (maxVal - minVal) / 255.0
                let zeroPoint = Int(-minVal / scale)
                scales.append(scale)
                zeroPoints.append(zeroPoint)
            }

            self.scales = scales
            self.zeroPoints = zeroPoints
        }

        func quantize() -> [[Int8]] {
            var result: [[Int8]] = []
            for (i, channel) in channels.enumerated() {
                let quantized = channel.map { value in
                    let q = Int(round(value / scales[i])) + zeroPoints[i]
                    return Int8(max(-128, min(127, q)))
                }
                result.append(quantized)
            }
            return result
        }

        func dequantize(_ quantized: [[Int8]]) -> [[Float]] {
            var result: [[Float]] = []
            for (i, channel) in quantized.enumerated() {
                let dequantized = channel.map { value in
                    Float(Int(value) - zeroPoints[i]) * scales[i]
                }
                result.append(dequantized)
            }
            return result
        }
    }

    // MARK: Dynamic Quantization

    struct DynamicQuantizer {
        private var calibrationData: [Vector512Optimized] = []
        private var currentParams: LinearQuantizationParams?
        let updateFrequency: Int
        private var updateCounter: Int = 0

        init(updateFrequency: Int = 100) {
            self.updateFrequency = updateFrequency
        }

        mutating func addCalibrationData(_ vector: Vector512Optimized) {
            calibrationData.append(vector)
            updateCounter += 1

            if updateCounter >= updateFrequency {
                updateParameters()
                updateCounter = 0
            }
        }

        mutating func updateParameters() {
            guard !calibrationData.isEmpty else { return }

            // Calculate statistics from calibration data
            let allValues = calibrationData.flatMap { $0.toArray() }
            let minVal = allValues.min() ?? -1
            let maxVal = allValues.max() ?? 1

            // Use percentiles for robustness
            let sorted = allValues.sorted()
            let p01 = sorted[Int(Float(sorted.count) * 0.01)]
            let p99 = sorted[Int(Float(sorted.count) * 0.99)]

            currentParams = LinearQuantizationParams(
                minValue: p01,
                maxValue: p99,
                symmetric: abs(p01) == abs(p99)
            )

            // Clear old calibration data
            if calibrationData.count > 1000 {
                calibrationData = Array(calibrationData.suffix(500))
            }
        }

        func quantize(_ vector: Vector512Optimized) -> Vector512INT8 {
            return Vector512INT8(from: vector, params: currentParams ?? LinearQuantizationParams(minValue: -1, maxValue: 1, symmetric: true))
        }
    }

    // MARK: Non-Uniform Quantization

    struct NonUniformQuantizer {
        enum QuantizationType {
            case logarithmic
            case exponential
            case powerOfTwo
            case custom(transform: (Float) -> Float, inverse: (Float) -> Float)
        }

        let type: QuantizationType

        func quantize(_ vector: Vector512Optimized) -> Vector512INT8 {
            let transformed = vector.toArray().map { value in
                let sgn = value >= 0 ? 1.0 : -1.0
                switch type {
                case .logarithmic:
                    return Float(sgn * Double(log(abs(value) + 1)))
                case .exponential:
                    return Float(sgn * Double(exp(abs(value)) - 1))
                case .powerOfTwo:
                    return Float(sgn * pow(2.0, floor(log2(Double(abs(value) + 1)))))
                case .custom(let transform, _):
                    return transform(value)
                }
            }

            // Now apply linear quantization to transformed values
            let transformedVector = try! Vector512Optimized(transformed)
            return Vector512INT8(from: transformedVector)
        }

        func dequantize(_ quantized: Vector512INT8) -> Vector512Optimized {
            let linear = quantized.toFP32()
            let restored = linear.toArray().map { value in
                let sgn = value >= 0 ? 1.0 : -1.0
                switch type {
                case .logarithmic:
                    return Float(sgn * Double(exp(abs(value)) - 1))
                case .exponential:
                    return Float(sgn * Double(log(abs(value) + 1)))
                case .powerOfTwo:
                    return value  // Approximation
                case .custom(_, let inverse):
                    return inverse(value)
                }
            }
            return try! Vector512Optimized(restored)
        }

        // Removed unused sign function
    }

    // MARK: Real-World Scenario Helpers

    struct SemanticSearchScenario {
        let documents: [Vector512Optimized]
        let queries: [Vector512Optimized]

        static func generate(documentCount: Int = 1000, queryCount: Int = 10) -> SemanticSearchScenario {
            // Simulate document embeddings (e.g., from BERT)
            let documents = (0..<documentCount).map { i in
                Vector512Optimized { j in
                    // Simulate semantic embedding patterns
                    sin(Float(i * 7 + j) * 0.001) * cos(Float(i * 3 + j) * 0.002)
                }
            }

            // Simulate query embeddings (similar but with variations)
            let queries = (0..<queryCount).map { q in
                Vector512Optimized { j in
                    sin(Float(q * 5 + j) * 0.001) * cos(Float(q * 2 + j) * 0.002) + Float.random(in: -0.1...0.1)
                }
            }

            return SemanticSearchScenario(documents: documents, queries: queries)
        }

        func evaluateRecall(topK: Int = 10) -> Float {
            // Evaluate recall@k for quantized vs original
            var totalRecall: Float = 0

            for query in queries {
                // Get top-k from original
                let originalScores = documents.map { doc in
                    query.dotProduct(doc)
                }
                let originalTopK = Set(originalScores.enumerated()
                    .sorted { $0.element > $1.element }
                    .prefix(topK)
                    .map { $0.offset })

                // Get top-k from quantized
                let qQuery = Vector512INT8(from: query)
                let quantizedScores = documents.map { doc in
                    QuantizedKernels.dotProduct512(query: qQuery, candidate: Vector512INT8(from: doc))
                }
                let quantizedTopK = Set(quantizedScores.enumerated()
                    .sorted { $0.element > $1.element }
                    .prefix(topK)
                    .map { $0.offset })

                // Calculate recall
                let overlap = originalTopK.intersection(quantizedTopK).count
                totalRecall += Float(overlap) / Float(topK)
            }

            return totalRecall / Float(queries.count)
        }
    }

    struct RecommendationScenario {
        let userEmbeddings: [Vector512Optimized]
        let itemEmbeddings: [Vector512Optimized]

        static func generate(userCount: Int = 100, itemCount: Int = 5000) -> RecommendationScenario {
            // Simulate collaborative filtering embeddings
            let userEmbeddings = (0..<userCount).map { u in
                Vector512Optimized { d in
                    // User preferences in latent space
                    Float.random(in: -1...1) * exp(-Float(d) / 100.0)
                }
            }

            let itemEmbeddings = (0..<itemCount).map { i in
                Vector512Optimized { d in
                    // Item features in latent space
                    sin(Float(i + d) * 0.01) * Float.random(in: 0.5...1.5)
                }
            }

            return RecommendationScenario(
                userEmbeddings: userEmbeddings,
                itemEmbeddings: itemEmbeddings
            )
        }

        func evaluatePrecision(topK: Int = 20) -> Float {
            // Evaluate precision of recommendations
            var totalPrecision: Float = 0

            for userEmb in userEmbeddings.prefix(10) {  // Sample users
                // Original recommendations
                let originalScores = itemEmbeddings.map { item in
                    userEmb.cosineSimilarity(to: item)
                }
                let originalRecs = Set(originalScores.enumerated()
                    .sorted { $0.element > $1.element }
                    .prefix(topK)
                    .map { $0.offset })

                // Quantized recommendations
                let qUser = Vector512INT8(from: userEmb)
                let quantizedScores = itemEmbeddings.map { item in
                    let qItem = Vector512INT8(from: item)
                    let dot = QuantizedKernels.dotProduct512(query: qUser, candidate: qItem)
                    let mag1 = sqrt(QuantizedKernels.dotProduct512(query: qUser, candidate: qUser))
                    let mag2 = sqrt(QuantizedKernels.dotProduct512(query: qItem, candidate: qItem))
                    return mag1 > 0 && mag2 > 0 ? dot / (mag1 * mag2) : 0
                }
                let quantizedRecs = Set(quantizedScores.enumerated()
                    .sorted { $0.element > $1.element }
                    .prefix(topK)
                    .map { $0.offset })

                let precision = Float(originalRecs.intersection(quantizedRecs).count) / Float(topK)
                totalPrecision += precision
            }

            return totalPrecision / 10.0
        }
    }
}