//
//  EdgeCaseHandlerTests.swift
//  VectorCoreTests
//
//  Comprehensive tests for edge case handling
//

import XCTest
@testable import VectorCore

final class EdgeCaseHandlerTests: XCTestCase {

    // MARK: - Array Size Edge Cases

    func testEmptyArrayHandling() throws {
        let emptyArray: [Float] = []

        // Should throw on requireNonEmpty
        XCTAssertThrowsError(try EdgeCaseHandler.requireNonEmpty(emptyArray)) { error in
            XCTAssert(error is VectorError)
        }

        // Should return default value for handleEmpty
        let result = EdgeCaseHandler.handleEmpty(emptyArray, defaultValue: 42.0) { arr in
            arr.reduce(0, +)
        }
        XCTAssertEqual(result, 42.0)
    }

    func testMinimumSizeValidation() throws {
        let smallArray = [1.0, 2.0]

        // Should pass for min size 2
        XCTAssertNoThrow(try EdgeCaseHandler.requireMinimumSize(smallArray, minSize: 2))

        // Should fail for min size 3
        XCTAssertThrowsError(try EdgeCaseHandler.requireMinimumSize(smallArray, minSize: 3))
    }

    func testSingleElementHandling() throws {
        let singleVector = [try Vector512Optimized(Array(repeating: 1.0, count: 512))]

        let result = try EdgeCaseHandler.handleSingleElement(singleVector, operation: "test")
        XCTAssertTrue(result.isSingle)

        // Empty should throw
        let emptyVectors: [Vector512Optimized] = []
        XCTAssertThrowsError(try EdgeCaseHandler.handleSingleElement(emptyVectors, operation: "test"))
    }

    // MARK: - Zero Vector Handling

    func testZeroVectorDetection() throws {
        let zeroVector = try Vector512Optimized(Array(repeating: 0.0, count: 512))
        let nonZeroVector = try Vector512Optimized(Array(repeating: 1.0, count: 512))

        XCTAssertTrue(EdgeCaseHandler.isZeroVector(zeroVector))
        XCTAssertFalse(EdgeCaseHandler.isZeroVector(nonZeroVector))

        // Test with very small values
        let tinyVector = try Vector512Optimized(Array(repeating: 1e-11, count: 512))
        XCTAssertTrue(EdgeCaseHandler.isZeroVector(tinyVector, epsilon: 1e-9))
        XCTAssertFalse(EdgeCaseHandler.isZeroVector(tinyVector, epsilon: 1e-12))
    }

    func testZeroVectorNormalization() throws {
        let zeroVector = try Vector512Optimized(Array(repeating: 0.0, count: 512))
        let nonZeroVector = try Vector512Optimized(Array(repeating: 1.0, count: 512))

        // Should fail for zero vector using the built-in normalized method
        let zeroResult = zeroVector.normalized()
        if case .failure = zeroResult {
            // Expected failure for zero vector
        } else {
            XCTFail("Expected normalization to fail for zero vector")
        }

        // Non-zero vector should normalize successfully
        let normalResult = nonZeroVector.normalized()
        switch normalResult {
        case .success(let normalized):
            // Magnitude should be approximately 1.0 after normalization
            XCTAssertEqual(normalized.magnitude, 1.0, accuracy: 0.001)
        case .failure:
            XCTFail("Expected normalization to succeed for non-zero vector")
        }
    }

    // MARK: - Division by Zero Protection

    func testSafeDivision() {
        // Regular division
        XCTAssertEqual(EdgeCaseHandler.safeDivide(10.0, by: 2.0), 5.0)

        // Division by zero
        XCTAssertEqual(EdgeCaseHandler.safeDivide(10.0, by: 0.0), 0.0)
        XCTAssertEqual(EdgeCaseHandler.safeDivide(10.0, by: 0.0, defaultValue: -1.0), -1.0)

        // Division by NaN
        XCTAssertEqual(EdgeCaseHandler.safeDivide(10.0, by: Float.nan), 0.0)
    }

    func testSIMDSafeDivision() {
        let numerator = SIMD4<Float>(1.0, 2.0, 3.0, 4.0)
        let denominator = SIMD4<Float>(2.0, 0.0, Float.nan, 2.0)

        let result = EdgeCaseHandler.safeDivide(numerator, by: denominator, defaultValue: -1.0)

        XCTAssertEqual(result[0], 0.5)      // 1.0/2.0
        XCTAssertEqual(result[1], -1.0)     // 2.0/0.0 -> default
        XCTAssertEqual(result[2], -1.0)     // 3.0/NaN -> default
        XCTAssertEqual(result[3], 2.0)      // 4.0/2.0
    }

    // MARK: - NaN and Infinity Handling

    func testFiniteValidation() throws {
        // Should pass for finite values
        XCTAssertNoThrow(try EdgeCaseHandler.requireFinite(1.0))
        XCTAssertNoThrow(try EdgeCaseHandler.requireFinite(-1000.0))

        // Should throw for NaN
        XCTAssertThrowsError(try EdgeCaseHandler.requireFinite(Float.nan))

        // Should throw for infinity
        XCTAssertThrowsError(try EdgeCaseHandler.requireFinite(Float.infinity))
        XCTAssertThrowsError(try EdgeCaseHandler.requireFinite(-Float.infinity))
    }

    func testInfinityClamping() {
        XCTAssertEqual(EdgeCaseHandler.clampInfinities(Float.infinity), Float.greatestFiniteMagnitude)
        XCTAssertEqual(EdgeCaseHandler.clampInfinities(-Float.infinity), -Float.greatestFiniteMagnitude)
        XCTAssertEqual(EdgeCaseHandler.clampInfinities(100.0), 100.0)

        // Custom bounds
        XCTAssertEqual(EdgeCaseHandler.clampInfinities(Float.infinity, min: -1000, max: 1000), 1000)
    }

    func testNaNReplacement() {
        XCTAssertEqual(EdgeCaseHandler.replaceNaN(Float.nan, with: 42.0), 42.0)
        XCTAssertEqual(EdgeCaseHandler.replaceNaN(10.0, with: 42.0), 10.0)
    }

    func testSIMD4Sanitization() {
        let dirty = SIMD4<Float>(Float.nan, Float.infinity, -Float.infinity, 10.0)
        let clean = EdgeCaseHandler.sanitizeSIMD4(dirty, nanDefault: 0.0, infDefault: 1000.0)

        XCTAssertEqual(clean[0], 0.0)      // NaN -> 0
        XCTAssertEqual(clean[1], 1000.0)   // +Inf -> 1000
        XCTAssertEqual(clean[2], -1000.0)  // -Inf -> -1000
        XCTAssertEqual(clean[3], 10.0)     // Normal value unchanged
    }

    // MARK: - Bounds Checking

    func testIndexValidation() throws {
        let count = 10

        // Valid indices
        XCTAssertNoThrow(try EdgeCaseHandler.requireValidIndex(0, count: count))
        XCTAssertNoThrow(try EdgeCaseHandler.requireValidIndex(9, count: count))

        // Invalid indices
        XCTAssertThrowsError(try EdgeCaseHandler.requireValidIndex(-1, count: count))
        XCTAssertThrowsError(try EdgeCaseHandler.requireValidIndex(10, count: count))
    }

    func testRangeValidation() throws {
        let count = 10

        // Valid ranges
        XCTAssertNoThrow(try EdgeCaseHandler.requireValidRange(0..<10, count: count))
        XCTAssertNoThrow(try EdgeCaseHandler.requireValidRange(5..<8, count: count))

        // Invalid ranges
        XCTAssertThrowsError(try EdgeCaseHandler.requireValidRange(-1..<5, count: count))
        XCTAssertThrowsError(try EdgeCaseHandler.requireValidRange(5..<15, count: count))
    }

    func testIndexClamping() {
        let count = 10

        XCTAssertEqual(EdgeCaseHandler.clampIndex(-5, count: count), 0)
        XCTAssertEqual(EdgeCaseHandler.clampIndex(5, count: count), 5)
        XCTAssertEqual(EdgeCaseHandler.clampIndex(15, count: count), 9)
    }

    // MARK: - Dimension Validation

    func testDimensionMatching() throws {
        let v1 = try Vector512Optimized(Array(repeating: 1.0, count: 512))
        let v2 = try Vector512Optimized(Array(repeating: 2.0, count: 512))

        // Same dimensions should pass
        XCTAssertNoThrow(try EdgeCaseHandler.requireMatchingDimensions(v1, v2))

        // Different dimensions would fail (can't easily test without different vector types)
    }

    func testPositiveDimension() throws {
        XCTAssertNoThrow(try EdgeCaseHandler.requirePositiveDimension(512))
        XCTAssertThrowsError(try EdgeCaseHandler.requirePositiveDimension(0))
        XCTAssertThrowsError(try EdgeCaseHandler.requirePositiveDimension(-1))
    }

    func testSIMDAlignment() throws {
        XCTAssertNoThrow(try EdgeCaseHandler.requireSIMDAlignedDimension(512))
        XCTAssertNoThrow(try EdgeCaseHandler.requireSIMDAlignedDimension(768))
        XCTAssertThrowsError(try EdgeCaseHandler.requireSIMDAlignedDimension(513))
        XCTAssertThrowsError(try EdgeCaseHandler.requireSIMDAlignedDimension(7))
    }

    // MARK: - Batch Operations Edge Cases

    func testBatchDistanceEdgeCases() throws {
        let query = try Vector512Optimized(Array(repeating: 1.0, count: 512))

        // Empty candidates
        let emptyResult = try EdgeCaseHandler.handleBatchDistanceEdgeCases(query: query, candidates: [])
        if case .empty = emptyResult {
            // Success
        } else {
            XCTFail("Expected empty result")
        }

        // Single candidate
        let single = try Vector512Optimized(Array(repeating: 2.0, count: 512))
        let singleResult = try EdgeCaseHandler.handleBatchDistanceEdgeCases(query: query, candidates: [single])
        if case .single(let distance) = singleResult {
            XCTAssertEqual(distance, 512.0)  // (2-1)^2 * 512 = 512
        } else {
            XCTFail("Expected single result")
        }

        // Zero query with single candidate - should still return .single
        let zeroQuery = try Vector512Optimized(Array(repeating: 0.0, count: 512))
        let singleCandidateResult = try EdgeCaseHandler.handleBatchDistanceEdgeCases(query: zeroQuery, candidates: [single])
        if case .single(let distance) = singleCandidateResult {
            XCTAssertEqual(distance, 512.0 * 4.0)  // 2^2 * 512 = 2048
        } else {
            XCTFail("Expected single result")
        }

        // Zero query with multiple candidates - should return .computed
        let multiple = [
            single,  // All 2.0s
            try Vector512Optimized(Array(repeating: 1.0, count: 512))  // All 1.0s
        ]
        let zeroMultipleResult = try EdgeCaseHandler.handleBatchDistanceEdgeCases(query: zeroQuery, candidates: multiple)
        if case .computed(let distances) = zeroMultipleResult {
            XCTAssertEqual(distances[0], 512.0 * 4.0)  // 2^2 * 512
            XCTAssertEqual(distances[1], 512.0 * 1.0)  // 1^2 * 512
        } else {
            XCTFail("Expected computed result")
        }
    }

    // MARK: - Clustering Edge Cases

    func testClusteringEdgeCases() throws {
        // Empty vectors
        let empty: [Vector512Optimized] = []
        XCTAssertThrowsError(try EdgeCaseHandler.handleClusteringEdgeCases(vectors: empty))

        // Single vector
        let single = [try Vector512Optimized(Array(repeating: 1.0, count: 512))]
        let singleAction = try EdgeCaseHandler.handleClusteringEdgeCases(vectors: single)
        if case .singleCluster(let index) = singleAction {
            XCTAssertEqual(index, 0)
        } else {
            XCTFail("Expected single cluster")
        }

        // Two vectors
        let pair = [
            try Vector512Optimized(Array(repeating: 1.0, count: 512)),
            try Vector512Optimized(Array(repeating: 2.0, count: 512))
        ]
        let pairAction = try EdgeCaseHandler.handleClusteringEdgeCases(vectors: pair)
        if case .pairCluster(let i1, let i2) = pairAction {
            XCTAssertEqual(i1, 0)
            XCTAssertEqual(i2, 1)
        } else {
            XCTFail("Expected pair cluster")
        }

        // Identical vectors
        let identical = Array(repeating: try Vector512Optimized(Array(repeating: 1.0, count: 512)), count: 5)
        let identicalAction = try EdgeCaseHandler.handleClusteringEdgeCases(vectors: identical)
        if case .allIdentical(let count) = identicalAction {
            XCTAssertEqual(count, 5)
        } else {
            XCTFail("Expected all identical")
        }
    }

    // MARK: - Graph Edge Cases

    func testGraphEdgeCases() throws {
        // Empty graph
        let emptyGraph = try EdgeCaseHandler.handleGraphEdgeCases(nodeCount: 0, edgeCount: 0)
        if case .emptyGraph = emptyGraph {
            // Success
        } else {
            XCTFail("Expected empty graph")
        }

        // Single node
        let singleNode = try EdgeCaseHandler.handleGraphEdgeCases(nodeCount: 1, edgeCount: 0)
        if case .singleNode = singleNode {
            // Success
        } else {
            XCTFail("Expected single node")
        }

        // Disconnected graph
        let disconnected = try EdgeCaseHandler.handleGraphEdgeCases(nodeCount: 5, edgeCount: 0)
        if case .disconnected(let count) = disconnected {
            XCTAssertEqual(count, 5)
        } else {
            XCTFail("Expected disconnected graph")
        }

        // Invalid edge count
        XCTAssertThrowsError(try EdgeCaseHandler.handleGraphEdgeCases(nodeCount: 5, edgeCount: 100))

        // Negative counts
        XCTAssertThrowsError(try EdgeCaseHandler.handleGraphEdgeCases(nodeCount: -1, edgeCount: 0))
        XCTAssertThrowsError(try EdgeCaseHandler.handleGraphEdgeCases(nodeCount: 5, edgeCount: -1))
    }

    // MARK: - Integration Tests

    func testSafeBatchEuclidean() throws {
        // Test empty candidates
        let query = try Vector512Optimized(Array(repeating: 1.0, count: 512))
        let emptyResult = try BatchKernels_SoA.safeBatchEuclidean512(query: query, candidates: [])
        XCTAssertTrue(emptyResult.isEmpty)

        // Test single candidate
        let single = try Vector512Optimized(Array(repeating: 2.0, count: 512))
        let singleResult = try BatchKernels_SoA.safeBatchEuclidean512(query: query, candidates: [single])
        XCTAssertEqual(singleResult.count, 1)
        XCTAssertEqual(singleResult[0], 512.0, accuracy: 0.001)
    }

    func testSafeAgglomerativeClustering() throws {
        // Test single vector clustering
        let single = [try Vector512Optimized(Array(repeating: 1.0, count: 512))]
        let singleTree = try HierarchicalClusteringKernels.safeAgglomerativeClustering(vectors: single)
        XCTAssertEqual(singleTree.nodeCount, 1)
        XCTAssertEqual(singleTree.rootNodeId, 0)

        // Test pair clustering
        let pair = [
            try Vector512Optimized(Array(repeating: 1.0, count: 512)),
            try Vector512Optimized(Array(repeating: 2.0, count: 512))
        ]
        let pairTree = try HierarchicalClusteringKernels.safeAgglomerativeClustering(vectors: pair)
        XCTAssertEqual(pairTree.nodeCount, 3)  // 2 leaves + 1 root

        // Test identical vectors clustering
        let identical = Array(repeating: try Vector512Optimized(Array(repeating: 1.0, count: 512)), count: 4)
        let identicalTree = try HierarchicalClusteringKernels.safeAgglomerativeClustering(vectors: identical)
        XCTAssertEqual(identicalTree.nodeCount, 5)  // 4 leaves + 1 root
        XCTAssertEqual(identicalTree.rootNode?.mergeDistance, 0.0)  // All identical, distance is 0
    }

    // MARK: - Performance Tests

    func testEdgeCaseHandlingPerformance() throws {
        let vectors = try (0..<1000).map { i in
            try Vector512Optimized(Array(repeating: Float(i), count: 512))
        }

        measure {
            for vector in vectors {
                _ = EdgeCaseHandler.isZeroVector(vector)
                _ = EdgeCaseHandler.clampIndex(Int.random(in: -100...1100), count: vectors.count)
                _ = EdgeCaseHandler.replaceNaN(Float.random(in: -100...100), with: 0)
            }
        }
    }
}