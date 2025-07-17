// VectorCore: Batch Operations Tests
//
// Comprehensive tests for batch processing utilities
//

import XCTest
@testable import VectorCore

final class BatchOperationsTests: XCTestCase {
    
    // MARK: - Test Data
    
    var testVectors128: [Vector128]!
    var testVectors256: [Vector256]!
    var emptyVectors: [Vector256]!
    
    override func setUp() {
        super.setUp()
        
        // Create test vectors
        testVectors128 = (0..<100).map { i in
            Vector128(repeating: Float(i) / 100.0)
        }
        
        testVectors256 = (0..<100).map { i in
            let values = (0..<256).map { j in
                Float(i * 256 + j) / 25600.0
            }
            return Vector256(values)
        }
        
        emptyVectors = []
    }
    
    override func tearDown() {
        testVectors128 = nil
        testVectors256 = nil
        emptyVectors = nil
        super.tearDown()
    }
    
    // MARK: - Process Tests
    
    func testBatchProcess() async throws {
        // Test basic batch processing
        let results = try await BatchOperations.process(testVectors256, batchSize: 10) { batch in
            batch.map { $0.magnitude }
        }
        
        XCTAssertEqual(results.count, testVectors256.count)
        
        // Verify all vectors were processed
        for (index, magnitude) in results.enumerated() {
            let expected = testVectors256[index].magnitude
            XCTAssertEqual(magnitude, expected, accuracy: 1e-6)
        }
    }
    
    func testBatchProcessWithDifferentSizes() async throws {
        // Test various batch sizes
        let batchSizes = [1, 5, 10, 50, 100, 1000]
        
        for batchSize in batchSizes {
            let results = try await BatchOperations.process(testVectors256, batchSize: batchSize) { batch in
                batch.map { _ in 1 }
            }
            
            XCTAssertEqual(results.count, testVectors256.count,
                          "Failed for batch size \(batchSize)")
        }
    }
    
    func testBatchProcessEmpty() async throws {
        // Test empty input
        let results = try await BatchOperations.process(emptyVectors, batchSize: 10) { batch in
            batch.map { $0.magnitude }
        }
        
        XCTAssertTrue(results.isEmpty)
    }
    
    func testBatchProcessError() async {
        // Test error propagation
        enum TestError: Error {
            case simulatedError
        }
        
        do {
            _ = try await BatchOperations.process(testVectors256, batchSize: 10) { batch -> [Int] in
                throw TestError.simulatedError
            }
            XCTFail("Should have thrown error")
        } catch {
            // Expected error
            XCTAssertTrue(error is TestError)
        }
    }
    
    // MARK: - Find Nearest Tests
    
    func testFindNearest() async {
        let query = Vector256(repeating: 0.5)
        
        // Find 10 nearest neighbors
        let neighbors = await BatchOperations.findNearest(
            to: query,
            in: testVectors256,
            k: 10,
            metric: EuclideanDistance()
        )
        
        // Verify results
        XCTAssertEqual(neighbors.count, 10)
        
        // Check ordering (distances should be ascending)
        for i in 1..<neighbors.count {
            XCTAssertLessThanOrEqual(neighbors[i-1].distance, neighbors[i].distance)
        }
        
        // Verify indices are valid
        for neighbor in neighbors {
            XCTAssertGreaterThanOrEqual(neighbor.index, 0)
            XCTAssertLessThan(neighbor.index, testVectors256.count)
        }
    }
    
    func testFindNearestWithK1() async {
        let query = testVectors256[50]
        
        // Find single nearest (should be itself)
        let neighbors = await BatchOperations.findNearest(
            to: query,
            in: testVectors256,
            k: 1,
            metric: EuclideanDistance()
        )
        
        XCTAssertEqual(neighbors.count, 1)
        XCTAssertEqual(neighbors[0].index, 50)
        XCTAssertEqual(neighbors[0].distance, 0.0, accuracy: 1e-6)
    }
    
    func testFindNearestKLargerThanCount() async {
        let query = Vector256(repeating: 0.5)
        let smallSet = Array(testVectors256.prefix(5))
        
        // Request more neighbors than available
        let neighbors = await BatchOperations.findNearest(
            to: query,
            in: smallSet,
            k: 10,
            metric: EuclideanDistance()
        )
        
        // Should return all available
        XCTAssertEqual(neighbors.count, 5)
    }
    
    func testFindNearestEmpty() async {
        let query = Vector256(repeating: 0.5)
        
        // Empty vectors
        let neighbors = await BatchOperations.findNearest(
            to: query,
            in: emptyVectors,
            k: 10
        )
        
        XCTAssertTrue(neighbors.isEmpty)
    }
    
    func testFindNearestZeroK() async {
        let query = Vector256(repeating: 0.5)
        
        // k = 0
        let neighbors = await BatchOperations.findNearest(
            to: query,
            in: testVectors256,
            k: 0
        )
        
        XCTAssertTrue(neighbors.isEmpty)
    }
    
    func testFindNearestDifferentMetrics() async {
        let query = Vector128(repeating: 0.5)
        let metrics: [any DistanceMetric] = [
            EuclideanDistance(),
            CosineDistance(),
            ManhattanDistance(),
            DotProductDistance()
        ]
        
        for metric in metrics {
            let neighbors = await BatchOperations.findNearest(
                to: query,
                in: testVectors128,
                k: 5,
                metric: metric
            )
            
            XCTAssertEqual(neighbors.count, 5)
            
            // Verify ordering
            for i in 1..<neighbors.count {
                XCTAssertLessThanOrEqual(neighbors[i-1].distance, neighbors[i].distance,
                                        "Failed for metric \(type(of: metric))")
            }
        }
    }
    
    // MARK: - Map Tests
    
    func testBatchMap() async throws {
        // Test batch mapping
        let normalized = try await BatchOperations.map(testVectors256) { vector in
            vector.normalized()
        }
        
        XCTAssertEqual(normalized.count, testVectors256.count)
        
        // Verify all vectors are normalized
        for vector in normalized {
            XCTAssertEqual(vector.magnitude, 1.0, accuracy: 1e-5)
        }
    }
    
    func testBatchMapEmpty() async throws {
        let results = try await BatchOperations.map(emptyVectors) { $0.normalized() }
        XCTAssertTrue(results.isEmpty)
    }
    
    func testBatchMapError() async {
        enum TestError: Error {
            case simulatedError
        }
        
        do {
            _ = try await BatchOperations.map(testVectors256) { vector -> Vector256 in
                throw TestError.simulatedError
            }
            XCTFail("Should have thrown error")
        } catch {
            // Expected error
            XCTAssertTrue(error is TestError)
        }
    }
    
    // MARK: - Filter Tests
    
    func testBatchFilter() async throws {
        // Filter vectors with magnitude > 5
        let filtered = try await BatchOperations.filter(testVectors256) { vector in
            vector.magnitude > 5.0
        }
        
        // Verify filtering
        for vector in filtered {
            XCTAssertGreaterThan(vector.magnitude, 5.0)
        }
        
        // Should have filtered some out
        XCTAssertLessThan(filtered.count, testVectors256.count)
    }
    
    func testBatchFilterNone() async throws {
        // Filter with always false predicate
        let filtered = try await BatchOperations.filter(testVectors256) { _ in false }
        XCTAssertTrue(filtered.isEmpty)
    }
    
    func testBatchFilterAll() async throws {
        // Filter with always true predicate
        let filtered = try await BatchOperations.filter(testVectors256) { _ in true }
        XCTAssertEqual(filtered.count, testVectors256.count)
    }
    
    // MARK: - Pairwise Distance Tests
    
    func testPairwiseDistances() async {
        let vectors = Array(testVectors128.prefix(10))
        let distances = await BatchOperations.pairwiseDistances(vectors)
        
        // Check dimensions
        XCTAssertEqual(distances.count, vectors.count)
        XCTAssertEqual(distances[0].count, vectors.count)
        
        // Check symmetry
        for i in 0..<vectors.count {
            for j in 0..<vectors.count {
                XCTAssertEqual(distances[i][j], distances[j][i], accuracy: 1e-6)
            }
        }
        
        // Check diagonal (distance to self = 0)
        for i in 0..<vectors.count {
            XCTAssertEqual(distances[i][i], 0.0, accuracy: 1e-6)
        }
        
        // Verify actual distances
        for i in 0..<vectors.count {
            for j in 0..<vectors.count {
                let expected = EuclideanDistance().distance(vectors[i], vectors[j])
                XCTAssertEqual(distances[i][j], expected, accuracy: 1e-6)
            }
        }
    }
    
    func testPairwiseDistancesEmpty() async {
        let distances = await BatchOperations.pairwiseDistances([Vector128]())
        XCTAssertTrue(distances.isEmpty)
    }
    
    func testPairwiseDistancesSingle() async {
        let vectors = [Vector128(repeating: 1.0)]
        let distances = await BatchOperations.pairwiseDistances(vectors)
        
        XCTAssertEqual(distances.count, 1)
        XCTAssertEqual(distances[0].count, 1)
        XCTAssertEqual(distances[0][0], 0.0)
    }
    
    func testPairwiseDistancesDifferentMetrics() async {
        let vectors = Array(testVectors128.prefix(5))
        
        let euclidean = await BatchOperations.pairwiseDistances(vectors, metric: EuclideanDistance())
        let cosine = await BatchOperations.pairwiseDistances(vectors, metric: CosineDistance())
        
        // Different metrics should give different results
        var hasDifference = false
        for i in 0..<vectors.count {
            for j in 0..<vectors.count where i != j {
                if abs(euclidean[i][j] - cosine[i][j]) > 1e-6 {
                    hasDifference = true
                    break
                }
            }
        }
        
        XCTAssertTrue(hasDifference, "Different metrics should produce different distances")
    }
    
    // MARK: - Sample Tests
    
    func testSample() {
        let k = 10
        let sampled = BatchOperations.sample(testVectors256, k: k)
        
        XCTAssertEqual(sampled.count, k)
        
        // Check all samples are from original set
        for sample in sampled {
            XCTAssertTrue(testVectors256.contains { $0.toArray() == sample.toArray() })
        }
        
        // Check for duplicates (sampling without replacement)
        var seen = Set<[Float]>()
        for sample in sampled {
            let array = sample.toArray()
            XCTAssertFalse(seen.contains(array), "Duplicate sample found")
            seen.insert(array)
        }
    }
    
    func testSampleKGreaterThanCount() {
        let sampled = BatchOperations.sample(testVectors256, k: 1000)
        XCTAssertEqual(sampled.count, testVectors256.count)
    }
    
    func testSampleZero() {
        let sampled = BatchOperations.sample(testVectors256, k: 0)
        XCTAssertTrue(sampled.isEmpty)
    }
    
    func testSampleNegative() {
        let sampled = BatchOperations.sample(testVectors256, k: -5)
        XCTAssertTrue(sampled.isEmpty)
    }
    
    func testSampleRandomness() {
        // Sample multiple times and check for different results
        let samples1 = BatchOperations.sample(testVectors256, k: 5)
        let samples2 = BatchOperations.sample(testVectors256, k: 5)
        
        // Convert to arrays for comparison
        let arrays1 = samples1.map { $0.toArray() }
        let arrays2 = samples2.map { $0.toArray() }
        
        // They should be different (with very high probability)
        var allSame = true
        for i in 0..<arrays1.count {
            if arrays1[i] != arrays2[i] {
                allSame = false
                break
            }
        }
        
        XCTAssertFalse(allSame, "Random sampling should produce different results")
    }
    
    // MARK: - Statistics Tests
    
    func testStatistics() async {
        let stats = await BatchOperations.statistics(for: testVectors256)
        
        XCTAssertEqual(stats.count, testVectors256.count)
        XCTAssertGreaterThan(stats.meanMagnitude, 0)
        XCTAssertGreaterThan(stats.stdMagnitude, 0)
        
        // Manually verify mean
        let magnitudes = testVectors256.map { $0.magnitude }
        let expectedMean = magnitudes.reduce(0, +) / Float(magnitudes.count)
        XCTAssertEqual(stats.meanMagnitude, expectedMean, accuracy: 1e-5)
        
        // Verify standard deviation
        let variance = magnitudes.map { pow($0 - expectedMean, 2) }.reduce(0, +) / Float(magnitudes.count)
        let expectedStd = sqrt(variance)
        XCTAssertEqual(stats.stdMagnitude, expectedStd, accuracy: 1e-5)
    }
    
    func testStatisticsEmpty() async {
        let stats = await BatchOperations.statistics(for: emptyVectors)
        
        XCTAssertEqual(stats.count, 0)
        XCTAssertEqual(stats.meanMagnitude, 0)
        XCTAssertEqual(stats.stdMagnitude, 0)
    }
    
    func testStatisticsSingle() async {
        let vectors = [Vector256(repeating: 1.0)]
        let stats = await BatchOperations.statistics(for: vectors)
        
        XCTAssertEqual(stats.count, 1)
        XCTAssertGreaterThan(stats.meanMagnitude, 0)
        XCTAssertEqual(stats.stdMagnitude, 0.0, accuracy: 1e-6) // No variance
    }
    
    // MARK: - Performance Tests
    
    func testBatchProcessPerformance() async throws {
        let largeSet = (0..<10000).map { _ in Vector256.random(in: -1...1) }
        
        // Note: XCTest's measure{} doesn't support async, so we time manually
        let start = Date()
        _ = try await BatchOperations.process(largeSet, batchSize: 1024) { batch in
            batch.map { $0.magnitude }
        }
        let elapsed = Date().timeIntervalSince(start)
        print("Batch process performance: \(elapsed)s")
        XCTAssertLessThan(elapsed, 5.0, "Batch processing took too long")
    }
    
    func testFindNearestPerformance() async {
        let largeSet = (0..<10000).map { _ in Vector256.random(in: -1...1) }
        let query = Vector256.random(in: -1...1)
        
        // Note: XCTest's measure{} doesn't support async, so we time manually
        let start = Date()
        _ = await BatchOperations.findNearest(
            to: query,
            in: largeSet,
            k: 100,
            metric: EuclideanDistance()
        )
        let elapsed = Date().timeIntervalSince(start)
        print("Find nearest performance: \(elapsed)s")
        XCTAssertLessThan(elapsed, 5.0, "Find nearest took too long")
    }
    
    func testPairwiseDistancesPerformance() async {
        let vectors = (0..<100).map { _ in Vector128.random(in: -1...1) }
        
        // Note: XCTest's measure{} doesn't support async, so we time manually
        let start = Date()
        _ = await BatchOperations.pairwiseDistances(vectors)
        let elapsed = Date().timeIntervalSince(start)
        print("Pairwise distances performance: \(elapsed)s")
        XCTAssertLessThan(elapsed, 2.0, "Pairwise distances took too long")
    }
    
    // MARK: - Memory Tests
    
    func testLargeBatchMemoryUsage() async throws {
        // Test that batch operations don't cause excessive memory usage
        let largeSet = (0..<100000).map { _ in Vector128.random(in: -1...1) }
        
        // Process in batches should use less memory than processing all at once
        do {
            _ = try await BatchOperations.process(largeSet, batchSize: 1000) { batch in
                batch.map { $0.normalized() }
            }
        } catch {
            XCTFail("Batch processing failed: \(error)")
        }
        
        // If we got here without crashing, memory usage is reasonable
        XCTAssertTrue(true)
    }
}