//
//  RobustnessQuickTests.swift
//  VectorCore
//
//  Quick tests for robustness improvements
//

import XCTest
@testable import VectorCore

final class RobustnessQuickTests: XCTestCase {

    func testManhattanDistanceHandlesNaN() {
        let a = ContiguousArray<Float>([1.0, Float.nan, 3.0])
        let b = ContiguousArray<Float>([2.0, 3.0, 4.0])

        let distance = GraphConstructionKernels.computeDistance(a, b, metric: .manhattan)
        XCTAssertEqual(distance, Float.infinity, "NaN should result in infinity distance")
    }

    func testCosineDistanceHandlesZeroVector() {
        let a = ContiguousArray<Float>([0.0, 0.0, 0.0])
        let b = ContiguousArray<Float>([1.0, 2.0, 3.0])

        let distance = GraphConstructionKernels.computeDistance(a, b, metric: .cosine)
        XCTAssertEqual(distance, 2.0, "Zero vector should give maximum cosine distance")
    }

    func testGraphValidation() {
        // Valid graph
        let validGraph = SparseMatrix(
            rows: 3,
            cols: 3,
            rowPointers: [0, 2, 4, 6],
            columnIndices: [1, 2, 0, 2, 0, 1],
            values: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

        XCTAssertTrue(validGraph.hasValidIndices())

        // Invalid graph with out-of-bounds index
        let invalidGraph = SparseMatrix(
            rows: 2,
            cols: 2,
            rowPointers: [0, 2, 3],
            columnIndices: [0, 5, 1], // 5 is out of bounds
            values: [1.0, 1.0, 1.0]
        )

        XCTAssertFalse(invalidGraph.hasValidIndices())
    }

    func testErrorHandling() {
        // k not even or too large
        XCTAssertThrowsError(try GraphConstructionKernels.generateSmallWorldGraph(
            n: 5,
            k: 10, // Too large for 5 nodes
            p: 0.1
        ))

        // p out of range
        XCTAssertThrowsError(try GraphConstructionKernels.generateSmallWorldGraph(
            n: 10,
            k: 4,
            p: 2.0 // Invalid probability
        ))

        // m too large for scale-free graph
        XCTAssertThrowsError(try GraphConstructionKernels.generateScaleFreeGraph(
            n: 3,
            m: 5 // m should be less than n
        ))
    }
}