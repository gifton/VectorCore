//
//  QuantizationTests.swift
//  VectorCore
//
//  Tests for quantization kernels and schemes
//

import XCTest
@testable import VectorCore

final class QuantizationTests: XCTestCase {

    func testQuantizationParams() {
        let minValue: Float = 1.0
        let maxValue: Float = 5.0
        let params = LinearQuantizationParams(minValue: minValue, maxValue: maxValue, symmetric: true)

        XCTAssertEqual(params.isSymmetric, true)
        XCTAssertGreaterThan(params.scale, 0)
        XCTAssertEqual(params.zeroPoint, 0) // Symmetric quantization has zero point = 0
        XCTAssertEqual(params.minValue, minValue)
        XCTAssertEqual(params.maxValue, maxValue)
    }

    func testVector512Quantization() {
        let original = try! Vector512Optimized([Float](repeating: 1.0, count: 512))
        let quantized = Vector512INT8(from: original)

        XCTAssertEqual(quantized.storage.count, 128) // 512 / 4 (SIMD4<Int8>)
        XCTAssertEqual(quantized.quantizationParams.isSymmetric, true)

        let dequantized = quantized.toFP32()
        XCTAssertEqual(dequantized.storage.count, 128) // 512 / 4
    }

    func testVector768Quantization() {
        let original = try! Vector768Optimized([Float](repeating: 1.5, count: 768))
        let quantized = Vector768INT8(from: original)

        XCTAssertEqual(quantized.storage.count, 192) // 768 / 4 (SIMD4<Int8>)
        XCTAssertEqual(quantized.quantizationParams.isSymmetric, true)

        let dequantized = quantized.toFP32()
        XCTAssertEqual(dequantized.storage.count, 192) // 768 / 4
    }

    func testVector1536Quantization() {
        let original = try! Vector1536Optimized([Float](repeating: 2.0, count: 1536))
        let quantized = Vector1536INT8(from: original)

        XCTAssertEqual(quantized.storage.count, 384) // 1536 / 4 (SIMD4<Int8>)
        XCTAssertEqual(quantized.quantizationParams.isSymmetric, true)

        let dequantized = quantized.toFP32()
        XCTAssertEqual(dequantized.storage.count, 384) // 1536 / 4
    }

    func testQuantizationAccuracy() {
        let values: [Float] = Array(0..<512).map { Float($0) / 100.0 }
        let original = try! Vector512Optimized(values)
        let quantized = Vector512INT8(from: original)
        let dequantized = quantized.toFP32()

        let originalArray = original.toArray()
        let dequantizedArray = dequantized.toArray()

        // Check that quantization preserves approximate values
        for i in 0..<512 {
            let error = abs(originalArray[i] - dequantizedArray[i])
            XCTAssertLessThan(error, 0.1, "Quantization error too large at index \(i)")
        }
    }

    func testQuantizationErrorAnalysis() {
        let original: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let params = LinearQuantizationParams(minValue: 1.0, maxValue: 5.0, symmetric: true)

        // Manually quantize using the LinearQuantizationParams
        let quantized: [Int8] = original.map { value in
            params.quantize(value)
        }

        // Manually dequantize
        let dequantized: [Float] = quantized.map { quantizedValue in
            params.dequantize(quantizedValue)
        }

        // Basic error analysis
        var maxError: Float = 0
        var mse: Float = 0
        for i in 0..<original.count {
            let error = abs(original[i] - dequantized[i])
            maxError = max(maxError, error)
            mse += error * error
        }
        mse /= Float(original.count)

        XCTAssertGreaterThanOrEqual(maxError, 0)
        XCTAssertGreaterThanOrEqual(mse, 0)
        XCTAssertLessThan(maxError, 1.0) // Reasonable error bounds
    }

    func testQuantizedVectorBasicOperations() {
        let query = try! Vector512Optimized([Float](repeating: 1.0, count: 512))
        let candidate = try! Vector512Optimized([Float](repeating: 1.1, count: 512))

        // Test basic quantization workflow
        let quantizedQuery = Vector512INT8(from: query)
        let quantizedCandidate = Vector512INT8(from: candidate)

        XCTAssertEqual(quantizedQuery.storage.count, 128) // 512 / 4
        XCTAssertEqual(quantizedCandidate.storage.count, 128) // 512 / 4

        // Test round-trip: quantize then dequantize
        let dequantizedQuery = quantizedQuery.toFP32()
        let dequantizedCandidate = quantizedCandidate.toFP32()

        XCTAssertEqual(dequantizedQuery.storage.count, 128)
        XCTAssertEqual(dequantizedCandidate.storage.count, 128)

        // Verify basic properties are preserved
        let originalQueryArray = query.toArray()
        let dequantizedQueryArray = dequantizedQuery.toArray()

        // Check that most values are reasonably close after quantization roundtrip
        var errorCount = 0
        for i in 0..<512 {
            let error = abs(originalQueryArray[i] - dequantizedQueryArray[i])
            if error > 0.2 { // Allow some quantization error
                errorCount += 1
            }
        }

        // Most values should be close (allow some error due to quantization)
        XCTAssertLessThan(errorCount, 50, "Too many values have large quantization errors")
    }
}