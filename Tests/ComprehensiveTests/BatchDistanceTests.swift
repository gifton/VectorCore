//
//  BatchDistanceTests.swift
//  VectorCore
//
//  Tests for batch distance computation functionality
//

import XCTest
@testable import VectorCore

final class BatchDistanceTests: XCTestCase {

    // MARK: - Test Data Generation

    func generateRandomVectors(count: Int, dimension: Int) -> [ContiguousArray<Float>] {
        return (0..<count).map { _ in
            ContiguousArray((0..<dimension).map { _ in Float.random(in: -1...1) })
        }
    }

    // MARK: - Correctness Tests

    func testBatchEuclideanDistance() {
        let source = ContiguousArray<Float>([1, 2, 3, 4])
        let targets = [
            ContiguousArray<Float>([1, 2, 3, 4]), // Same as source, distance = 0
            ContiguousArray<Float>([2, 3, 4, 5]), // Distance = 2
            ContiguousArray<Float>([5, 6, 7, 8])  // Distance = 8
        ]

        let distances = GraphConstructionKernels.computeDistancesBatch(
            from: source,
            to: targets,
            metric: .euclidean
        )

        XCTAssertEqual(distances.count, 3)
        XCTAssertEqual(distances[0], 0, accuracy: 1e-6)
        XCTAssertEqual(distances[1], 2.0, accuracy: 1e-6)
        XCTAssertEqual(distances[2], 8.0, accuracy: 1e-6)
    }

    func testBatchCosineDistance() {
        let source = ContiguousArray<Float>([1, 0, 0])
        let targets = [
            ContiguousArray<Float>([1, 0, 0]),    // Same direction, distance = 0
            ContiguousArray<Float>([0, 1, 0]),    // Orthogonal, distance = 1
            ContiguousArray<Float>([-1, 0, 0])    // Opposite direction, distance = 2
        ]

        let distances = GraphConstructionKernels.computeDistancesBatch(
            from: source,
            to: targets,
            metric: .cosine
        )

        XCTAssertEqual(distances.count, 3)
        XCTAssertEqual(distances[0], 0, accuracy: 1e-6)
        XCTAssertEqual(distances[1], 1.0, accuracy: 1e-6)
        XCTAssertEqual(distances[2], 2.0, accuracy: 1e-6)
    }

    func testBatchManhattanDistance() {
        let source = ContiguousArray<Float>([1, 2, 3])
        let targets = [
            ContiguousArray<Float>([1, 2, 3]), // Same as source, distance = 0
            ContiguousArray<Float>([2, 3, 4]), // Distance = 3
            ContiguousArray<Float>([4, 5, 6])  // Distance = 9
        ]

        let distances = GraphConstructionKernels.computeDistancesBatch(
            from: source,
            to: targets,
            metric: .manhattan
        )

        XCTAssertEqual(distances.count, 3)
        XCTAssertEqual(distances[0], 0, accuracy: 1e-6)
        XCTAssertEqual(distances[1], 3.0, accuracy: 1e-6)
        XCTAssertEqual(distances[2], 9.0, accuracy: 1e-6)
    }

    func testBatchVsSingleDistanceConsistency() {
        let dimension = 128
        let source = ContiguousArray((0..<dimension).map { _ in Float.random(in: -1...1) })
        let targets = generateRandomVectors(count: 10, dimension: dimension)

        // Test for each metric
        let metrics: [(GraphConstructionKernels.DistanceMetric, String, Float)] = [
            (.euclidean, "euclidean", 1e-5),
            (.cosine, "cosine", 1e-5),
            (.manhattan, "manhattan", 1e-4)  // Manhattan can have more floating point error
        ]

        for (metric, name, accuracy) in metrics {
            // Compute batch distances
            let batchDistances = GraphConstructionKernels.computeDistancesBatch(
                from: source,
                to: targets,
                metric: metric
            )

            // Compute individual distances for comparison
            var individualDistances = ContiguousArray<Float>()
            for target in targets {
                let dist = GraphConstructionKernels.computeDistance(
                    source,
                    target,
                    metric: metric
                )
                individualDistances.append(dist)
            }

            // Compare results
            XCTAssertEqual(batchDistances.count, individualDistances.count)
            for i in 0..<batchDistances.count {
                XCTAssertEqual(batchDistances[i], individualDistances[i], accuracy: accuracy,
                               "Mismatch at index \(i) for metric \(name)")
            }
        }
    }

    // MARK: - Performance Tests

    func testBatchPerformanceImprovement() {
        let dimension = 512
        let numVectors = 100
        let source = ContiguousArray((0..<dimension).map { _ in Float.random(in: -1...1) })
        let targets = generateRandomVectors(count: numVectors, dimension: dimension)

        // Measure batch computation time
        let batchStart = CFAbsoluteTimeGetCurrent()
        _ = GraphConstructionKernels.computeDistancesBatch(
            from: source,
            to: targets,
            metric: .euclidean
        )
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStart

        // Measure individual computation time
        let individualStart = CFAbsoluteTimeGetCurrent()
        for target in targets {
            _ = GraphConstructionKernels.computeDistance(
                source,
                target,
                metric: .euclidean
            )
        }
        let individualTime = CFAbsoluteTimeGetCurrent() - individualStart

        let speedup = individualTime / batchTime
        print("Batch distance computation speedup: \(String(format: "%.2f", speedup))x")
        print("Batch time: \(String(format: "%.6f", batchTime))s")
        print("Individual time: \(String(format: "%.6f", individualTime))s")

        // Batch should be faster (at least 1.5x speedup expected due to reduced overhead)
        XCTAssertGreaterThan(speedup, 1.5, "Batch computation should be significantly faster")
    }

    func testKNNGraphWithBatchDistances() async {
        // Create test vectors
        let vectors = ContiguousArray((0..<20).map { _ in
            ContiguousArray((0..<64).map { _ in Float.random(in: -1...1) })
        })

        let options = GraphConstructionKernels.KNNGraphOptions(
            k: 5,
            metric: .euclidean,
            symmetric: true
        )

        // This will use the batch distance computation internally
        let graph = await GraphConstructionKernels.buildKNNGraph(
            vectors: vectors,
            options: options
        )

        // Verify graph structure
        XCTAssertEqual(graph.rows, 20)
        XCTAssertEqual(graph.cols, 20)

        // Each node should have at most k outgoing edges (more with symmetry)
        for i in 0..<graph.rows {
            let rowStart = Int(graph.rowPointers[i])
            let rowEnd = Int(graph.rowPointers[i + 1])
            let degree = rowEnd - rowStart

            // With symmetry, nodes can have more than k connections
            XCTAssertGreaterThan(degree, 0, "Node \(i) should have at least one connection")
        }
    }

    // MARK: - Edge Cases

    func testEmptyTargets() {
        let source = ContiguousArray<Float>([1, 2, 3])
        let targets: [ContiguousArray<Float>] = []

        let distances = GraphConstructionKernels.computeDistancesBatch(
            from: source,
            to: targets,
            metric: .euclidean
        )

        XCTAssertEqual(distances.count, 0)
    }

    func testZeroVectorCosineDistance() {
        let source = ContiguousArray<Float>([0, 0, 0])
        let targets = [
            ContiguousArray<Float>([1, 2, 3]),
            ContiguousArray<Float>([0, 0, 0])
        ]

        let distances = GraphConstructionKernels.computeDistancesBatch(
            from: source,
            to: targets,
            metric: .cosine
        )

        // Zero vector cosine distance is handled as 1.0 (max distance) in our implementation
        XCTAssertEqual(distances[0], 1.0, accuracy: 1e-6)
        // Zero to zero is handled as 1.0 in our implementation (both have zero magnitude)
        XCTAssertEqual(distances[1], 1.0, accuracy: 1e-6)
    }

    func testLargeScale() {
        // Test with larger vectors to ensure stability
        let dimension = 2048
        let numVectors = 50

        let source = ContiguousArray((0..<dimension).map { _ in Float.random(in: -0.1...0.1) })
        let targets = generateRandomVectors(count: numVectors, dimension: dimension)

        let distances = GraphConstructionKernels.computeDistancesBatch(
            from: source,
            to: targets,
            metric: .euclidean
        )

        XCTAssertEqual(distances.count, numVectors)

        // All distances should be finite and non-negative
        for dist in distances {
            XCTAssertTrue(dist.isFinite, "Distance should be finite")
            XCTAssertGreaterThanOrEqual(dist, 0, "Euclidean distance should be non-negative")
        }
    }
}
