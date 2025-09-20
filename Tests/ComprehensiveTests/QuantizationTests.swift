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
        let values: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let params = QuantizationSchemes.computeQuantizationParams(values: values, strategy: .perVector)

        XCTAssertEqual(params.strategy, .perVector)
        XCTAssertEqual(params.scales.count, 1)
        XCTAssertEqual(params.offsets.count, 1)
        XCTAssertGreaterThan(params.scales[0], 0)
    }

    func testVector512Quantization() {
        let original = try! Vector512Optimized([Float](repeating: 1.0, count: 512))
        let quantized = Vector512INT8(from: original, strategy: .perVector)

        XCTAssertEqual(quantized.storage.count, 32) // 512 / 16
        XCTAssertEqual(quantized.params.strategy, .perVector)

        let dequantized = quantized.dequantize()
        XCTAssertEqual(dequantized.storage.count, 128) // 512 / 4
    }

    func testVector768Quantization() {
        let original = try! Vector768Optimized([Float](repeating: 1.5, count: 768))
        let quantized = Vector768INT8(from: original, strategy: .perVector)

        XCTAssertEqual(quantized.storage.count, 48) // 768 / 16
        XCTAssertEqual(quantized.params.strategy, .perVector)

        let dequantized = quantized.dequantize()
        XCTAssertEqual(dequantized.storage.count, 192) // 768 / 4
    }

    func testVector1536Quantization() {
        let original = try! Vector1536Optimized([Float](repeating: 2.0, count: 1536))
        let quantized = Vector1536INT8(from: original, strategy: .perVector)

        XCTAssertEqual(quantized.storage.count, 96) // 1536 / 16
        XCTAssertEqual(quantized.params.strategy, .perVector)

        let dequantized = quantized.dequantize()
        XCTAssertEqual(dequantized.storage.count, 384) // 1536 / 4
    }

    func testQuantizationAccuracy() {
        let values: [Float] = Array(0..<512).map { Float($0) / 100.0 }
        let original = try! Vector512Optimized(values)
        let quantized = Vector512INT8(from: original, strategy: .perVector)
        let dequantized = quantized.dequantize()

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
        let params = QuantizationSchemes.computeQuantizationParams(values: original, strategy: .perVector)

        // Manually quantize
        let quantized: [Int8] = original.map { value in
            let q = (value / params.scales[0]) + params.offsets[0]
            return Int8(max(-128, min(127, q.rounded(.toNearestOrAwayFromZero))))
        }

        let stats = QuantizationSchemes.analyzeQuantizationError(
            original: original,
            quantized: quantized,
            params: params
        )

        XCTAssertGreaterThanOrEqual(stats.maxAbsoluteError, 0)
        XCTAssertGreaterThanOrEqual(stats.meanSquaredError, 0)
        XCTAssertGreaterThan(stats.signalToNoiseRatio, 0)
    }

    func testQuantizedKernelAPI() {
        let query = try! Vector512Optimized([Float](repeating: 1.0, count: 512))
        let candidates = [
            try! Vector512Optimized([Float](repeating: 1.1, count: 512)),
            try! Vector512Optimized([Float](repeating: 0.9, count: 512))
        ]

        let quantizedCandidates = QuantizedKernels.quantizeVectors_512(candidates, strategy: .perVector)
        XCTAssertEqual(quantizedCandidates.count, 2)

        var results = [Float](repeating: 0, count: 2)
        results.withUnsafeMutableBufferPointer { buffer in
            QuantizedKernels.range_euclid2_quantized_512(
                query: query,
                candidatesINT8: quantizedCandidates,
                range: 0..<2,
                out: buffer
            )
        }

        XCTAssertGreaterThan(results[0], 0)
        XCTAssertGreaterThan(results[1], 0)
    }
}