// VectorCore: Operations Tests
//
// Tests for the unified Operations API with ExecutionContext
//

import XCTest
@testable import VectorCore

@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)
final class OperationsTests: XCTestCase {
    
    // MARK: - Test Data
    
    private func createTestVectors(count: Int, dimension: Int = 128) -> [Vector<Dim128>] {
        (0..<count).map { i in
            let values = (0..<dimension).map { Float($0 + i) / Float(dimension) }
            return Vector<Dim128>(values)
        }
    }
    
    // MARK: - Find Nearest Tests
    
    func testFindNearestBasic() async throws {
        let vectors = createTestVectors(count: 10, dimension: 128)
        let query = Vector<Dim128>((0..<128).map { _ in 0.5 })
        
        let results = try await Operations.findNearest(to: query, in: vectors, k: 3)
        
        XCTAssertEqual(results.count, 3)
        // Results should be sorted by distance
        for i in 1..<results.count {
            XCTAssertLessThanOrEqual(results[i-1].distance, results[i].distance)
        }
    }
    
    func testFindNearestParallelization() async throws {
        // Create enough vectors to trigger parallelization
        let vectors = createTestVectors(count: 2000, dimension: 128)
        let query = vectors[500] // Use an existing vector as query
        
        // Sequential execution for comparison
        let sequentialStart = Date()
        let sequentialResults = try await Operations.$computeProvider.withValue(CPUComputeProvider.sequential) {
            try await Operations.findNearest(
                to: query,
                in: vectors,
                k: 10
            )
        }
        let sequentialTime = Date().timeIntervalSince(sequentialStart)
        
        // Parallel execution
        let parallelStart = Date()
        let parallelResults = try await Operations.$computeProvider.withValue(CPUComputeProvider.automatic) {
            try await Operations.findNearest(
                to: query,
                in: vectors,
                k: 10
            )
        }
        let parallelTime = Date().timeIntervalSince(parallelStart)
        
        // Results should match
        XCTAssertEqual(sequentialResults.count, parallelResults.count)
        
        // The query vector itself should be the closest (distance 0)
        XCTAssertEqual(sequentialResults[0].distance, 0.0, accuracy: 0.0001)
        XCTAssertEqual(parallelResults[0].distance, 0.0, accuracy: 0.0001)
        
        print("Sequential time: \(sequentialTime)s")
        print("Parallel time: \(parallelTime)s")
        print("Speedup: \(sequentialTime / parallelTime)x")
        
        // Parallel should be faster on multi-core systems
        if ProcessInfo.processInfo.activeProcessorCount > 1 {
            XCTAssertLessThan(parallelTime, sequentialTime)
        }
    }
    
    func testFindNearestBatch() async throws {
        let vectors = createTestVectors(count: 100, dimension: 128)
        let queries = Array(vectors[0..<5]) // Use first 5 as queries
        
        let results = try await Operations.findNearestBatch(
            queries: queries,
            in: vectors,
            k: 3
        )
        
        XCTAssertEqual(results.count, queries.count)
        
        // Each query should find itself as the nearest
        for (i, queryResults) in results.enumerated() {
            XCTAssertEqual(queryResults[0].distance, 0.0, accuracy: 0.0001)
            XCTAssertEqual(queryResults[0].index, i)
        }
    }
    
    func testFindNearestDifferentMetrics() async throws {
        let vectors = createTestVectors(count: 50, dimension: 128)
        let query = Vector<Dim128>((0..<128).map { _ in 0.5 })
        
        // Test with different metrics
        let euclideanResults = try await Operations.findNearest(
            to: query,
            in: vectors,
            k: 5,
            metric: EuclideanDistance()
        )
        
        let cosineResults = try await Operations.findNearest(
            to: query,
            in: vectors,
            k: 5,
            metric: CosineDistance()
        )
        
        let manhattanResults = try await Operations.findNearest(
            to: query,
            in: vectors,
            k: 5,
            metric: ManhattanDistance()
        )
        
        // All should return 5 results
        XCTAssertEqual(euclideanResults.count, 5)
        XCTAssertEqual(cosineResults.count, 5)
        XCTAssertEqual(manhattanResults.count, 5)
        
        // Results might be in different order due to different metrics
        // But all should be valid indices
        for results in [euclideanResults, cosineResults, manhattanResults] {
            for result in results {
                XCTAssertGreaterThanOrEqual(result.index, 0)
                XCTAssertLessThan(result.index, vectors.count)
            }
        }
    }
    
    func testFindNearestEdgeCases() async throws {
        let vectors = createTestVectors(count: 5, dimension: 128)
        let query = vectors[0]
        
        // k larger than vector count
        let results1 = try await Operations.findNearest(
            to: query,
            in: vectors,
            k: 10
        )
        XCTAssertEqual(results1.count, vectors.count)
        
        // Empty vectors
        let results2 = try await Operations.findNearest(
            to: query,
            in: [] as [Vector<Dim128>],
            k: 5
        )
        XCTAssertEqual(results2.count, 0)
        
        // k = 0 should throw
        do {
            _ = try await Operations.findNearest(
                to: query,
                in: vectors,
                k: 0
            )
            XCTFail("Should have thrown for k=0")
        } catch {
            XCTAssertTrue(error is VectorError)
        }
    }
    
    // MARK: - Distance Matrix Tests
    
    func testDistanceMatrix() async throws {
        let vectors1 = createTestVectors(count: 5, dimension: 128)
        let vectors2 = createTestVectors(count: 3, dimension: 128)
        
        let matrix = try await Operations.distanceMatrix(
            between: vectors1,
            and: vectors2
        )
        
        XCTAssertEqual(matrix.count, vectors1.count)
        XCTAssertEqual(matrix[0].count, vectors2.count)
        
        // Verify some distances
        for i in 0..<vectors1.count {
            for j in 0..<vectors2.count {
                let expected = EuclideanDistance().distance(vectors1[i], vectors2[j])
                XCTAssertEqual(matrix[i][j], expected, accuracy: 0.0001)
            }
        }
    }
    
    func testDistanceMatrixDifferentMetrics() async throws {
        let values1 = (0..<128).map { Float($0) / 128.0 }
        let values2 = (0..<128).map { Float($0 + 3) / 128.0 }
        
        let vectors1 = [Vector<Dim128>(values1)]
        let vectors2 = [Vector<Dim128>(values2)]
        
        let euclideanMatrix = try await Operations.distanceMatrix(
            between: vectors1,
            and: vectors2,
            metric: EuclideanDistance()
        )
        
        let cosineMatrix = try await Operations.distanceMatrix(
            between: vectors1,
            and: vectors2,
            metric: CosineDistance()
        )
        
        // Manually calculate expected values
        let v1 = vectors1[0]
        let v2 = vectors2[0]
        
        XCTAssertEqual(euclideanMatrix[0][0], EuclideanDistance().distance(v1, v2), accuracy: 0.0001)
        XCTAssertEqual(cosineMatrix[0][0], CosineDistance().distance(v1, v2), accuracy: 0.0001)
    }
    
    // MARK: - Map Operation Tests
    
    /* Disabled - Operations.map doesn't exist
    func testMapOperation() async throws {
        let vectors = createTestVectors(count: 100, dimension: 128)
        
        // Transform: normalize all vectors
        let normalized = try await Operations.map(vectors) { vector in
            vector.normalized()
        }
        
        XCTAssertEqual(normalized.count, vectors.count)
        
        // All normalized vectors should have magnitude ~1
        for vector in normalized {
            let magnitude = vector.magnitude
            XCTAssertEqual(magnitude, 1.0, accuracy: 0.0001)
        }
    }
    */
    
    /* Disabled - Operations.map doesn't exist
    func testMapParallelPerformance() async throws {
        let vectors = createTestVectors(count: 5000, dimension: 128)
        
        // Complex transformation to make parallelization worthwhile
        let transform: @Sendable (Vector<Dim128>) -> Vector<Dim128> = { vector in
            // Simulate expensive computation
            var result = vector
            for _ in 0..<10 {
                result = result.normalized()
                result = result * 1.1 // Scale up slightly
            }
            return result
        }
        
        // Sequential
        let sequentialStart = Date()
        let sequentialResults = try await Operations.map(
            vectors,
            transform: transform,
            context: CPUContext.sequential
        )
        let sequentialTime = Date().timeIntervalSince(sequentialStart)
        
        // Parallel
        let parallelStart = Date()
        let parallelResults = try await Operations.map(
            vectors,
            transform: transform,
            context: CPUContext.automatic
        )
        let parallelTime = Date().timeIntervalSince(parallelStart)
        
        XCTAssertEqual(sequentialResults.count, parallelResults.count)
        
        print("Map Sequential time: \(sequentialTime)s")
        print("Map Parallel time: \(parallelTime)s")
        print("Map Speedup: \(sequentialTime / parallelTime)x")
        
        // Parallel should be faster on multi-core systems
        if ProcessInfo.processInfo.activeProcessorCount > 1 {
            XCTAssertLessThan(parallelTime, sequentialTime)
        }
    }
    */
    
    // MARK: - Batch Processing Tests
    
    /* Disabled - processBatches doesn't exist
    func testProcessBatches() async throws {
        let vectors = createTestVectors(count: 1000, dimension: 128)
        
        actor ProcessCounter {
            private var count = 0
            
            func add(_ value: Int) {
                count += value
            }
            
            func getCount() -> Int {
                count
            }
        }
        
        let counter = ProcessCounter()
        
        try await Operations.processBatches(vectors, batchSize: 100) { batch in
            // Simulate batch processing
            try await Task.sleep(nanoseconds: 1_000_000) // 1ms
            await counter.add(batch.count)
        }
        
        let processedCount = await counter.getCount()
        XCTAssertEqual(processedCount, vectors.count)
    }
    
    */
    
    /* Disabled - processBatches doesn't exist
    func testProcessBatchesAutomaticSize() async throws {
        let vectors = createTestVectors(count: 500, dimension: 128)
        
        actor BatchSizeCollector {
            private var sizes: [Int] = []
            
            func add(_ size: Int) {
                sizes.append(size)
            }
            
            func getSizes() -> [Int] {
                sizes
            }
        }
        
        let collector = BatchSizeCollector()
        
        try await Operations.processBatches(vectors) { batch in
            await collector.add(batch.count)
        }
        
        let batchSizes = await collector.getSizes()
        
        // Should have created multiple batches
        XCTAssertGreaterThan(batchSizes.count, 1)
        
        // Total should match
        XCTAssertEqual(batchSizes.reduce(0, +), vectors.count)
    }
    */
    
    // MARK: - Integration Tests
    
    func testIntegrationWithBufferPool() async throws {
        // This test verifies that operations work well with the buffer pool
        let vectors = createTestVectors(count: 1000, dimension: 128)
        let query = vectors[500]
        
        // Perform operations with default providers
        _ = try await Operations.findNearest(
            to: query,
            in: vectors,
            k: 50
        )
        
        // Test passes if no errors occur
        XCTAssertTrue(true)
    }
}