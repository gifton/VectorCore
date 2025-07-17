// VectorCore: Synchronous Batch Operations Tests
//
// Tests for synchronous batch processing operations
//

import XCTest
@testable import VectorCore

final class SyncBatchOperationsTests: XCTestCase {
    
    // MARK: - Test Data
    
    let testVectors: [Vector<Dim32>] = [
        Vector<Dim32>(Array(repeating: 1.0, count: 32)),
        Vector<Dim32>(Array(repeating: 2.0, count: 32)),
        Vector<Dim32>(Array(repeating: 3.0, count: 32)),
        Vector<Dim32>(Array(repeating: 4.0, count: 32)),
        Vector<Dim32>(Array(repeating: 5.0, count: 32))
    ]
    
    // MARK: - Nearest Neighbor Tests
    
    func testFindNearest() {
        let query = Vector<Dim32>(Array(repeating: 2.5, count: 32))
        let neighbors = SyncBatchOperations.findNearest(
            to: query,
            in: testVectors,
            k: 3
        )
        
        XCTAssertEqual(neighbors.count, 3)
        
        // Should find vectors 2.0, 3.0, and 1.0 as nearest (in that order)
        XCTAssertEqual(neighbors[0].index, 1) // Vector with 2.0
        XCTAssertEqual(neighbors[1].index, 2) // Vector with 3.0
        XCTAssertEqual(neighbors[2].index, 0) // Vector with 1.0
        
        // Check distances are in ascending order
        for i in 1..<neighbors.count {
            XCTAssertGreaterThanOrEqual(neighbors[i].distance, neighbors[i-1].distance)
        }
    }
    
    func testFindNearestWithCustomMetric() {
        let query = Vector<Dim32>.basis(axis: 0)
        let vectors = [
            Vector<Dim32>.basis(axis: 0),  // Should have cosine distance 0
            Vector<Dim32>.basis(axis: 1),  // Should have cosine distance 1
            Vector<Dim32>.ones()          // Should have intermediate distance
        ]
        
        let neighbors = SyncBatchOperations.findNearest(
            to: query,
            in: vectors,
            k: 2,
            metric: CosineDistance()
        )
        
        XCTAssertEqual(neighbors.count, 2)
        XCTAssertEqual(neighbors[0].index, 0) // Exact match
        XCTAssertEqual(neighbors[0].distance, 0, accuracy: 1e-6)
    }
    
    func testFindWithinRadius() {
        let query = Vector<Dim32>(Array(repeating: 2.5, count: 32))
        
        // Calculate expected distances: sqrt(32) * |2.5 - x|
        let scale = sqrt(Float(32))
        let expectedDistances = testVectors.map { vector in
            scale * abs(2.5 - vector[0])
        }
        
        // Use a radius that should capture vectors 2.0 and 3.0
        let radius = scale * 0.6  // This should include distances of 0.5*scale
        
        let withinRadius = SyncBatchOperations.findWithinRadius(
            of: query,
            in: testVectors,
            radius: radius
        )
        
        // Should find vectors 2.0 and 3.0 (distances of 0.5 * scale each)
        XCTAssertEqual(withinRadius.count, 2)
        XCTAssertTrue(withinRadius.contains { $0.index == 1 }) // Vector 2.0
        XCTAssertTrue(withinRadius.contains { $0.index == 2 }) // Vector 3.0
        
        // All distances should be within radius
        for result in withinRadius {
            XCTAssertLessThanOrEqual(result.distance, radius)
        }
    }
    
    func testFindNearestEdgeCases() {
        let query = Vector<Dim32>.random(in: 0...1)
        
        // Empty vectors
        let emptyResult = SyncBatchOperations.findNearest(to: query, in: [], k: 5)
        XCTAssertTrue(emptyResult.isEmpty)
        
        // k = 0
        let zeroK = SyncBatchOperations.findNearest(to: query, in: testVectors, k: 0)
        XCTAssertTrue(zeroK.isEmpty)
        
        // k > vector count
        let largeK = SyncBatchOperations.findNearest(to: query, in: testVectors, k: 100)
        XCTAssertEqual(largeK.count, testVectors.count)
    }
    
    // MARK: - Transformation Tests
    
    func testMap() {
        let normalized = SyncBatchOperations.map(testVectors) { $0.normalized() }
        
        XCTAssertEqual(normalized.count, testVectors.count)
        
        // All normalized vectors should have magnitude 1
        for vector in normalized {
            XCTAssertEqual(vector.magnitude, 1.0, accuracy: 1e-6)
        }
    }
    
    func testMapInPlace() {
        var vectors = testVectors
        
        SyncBatchOperations.mapInPlace(&vectors) { vector in
            vector = vector * 2.0
        }
        
        // Check all vectors were doubled
        for (i, vector) in vectors.enumerated() {
            let original = testVectors[i]
            for j in 0..<32 {
                XCTAssertEqual(vector[j], original[j] * 2.0)
            }
        }
    }
    
    func testFilter() {
        let filtered = SyncBatchOperations.filter(testVectors) { vector in
            vector[0] > 2.5
        }
        
        XCTAssertEqual(filtered.count, 3) // Vectors 3.0, 4.0, 5.0
        
        for vector in filtered {
            XCTAssertGreaterThan(vector[0], 2.5)
        }
    }
    
    func testPartition() {
        let (small, large) = SyncBatchOperations.partition(testVectors) { vector in
            vector[0] <= 3.0
        }
        
        XCTAssertEqual(small.count, 3) // Vectors 1.0, 2.0, 3.0
        XCTAssertEqual(large.count, 2) // Vectors 4.0, 5.0
        
        for vector in small {
            XCTAssertLessThanOrEqual(vector[0], 3.0)
        }
        
        for vector in large {
            XCTAssertGreaterThan(vector[0], 3.0)
        }
    }
    
    // MARK: - Aggregation Tests
    
    func testCentroid() {
        let centroid = SyncBatchOperations.centroid(of: testVectors)
        
        XCTAssertNotNil(centroid)
        
        // Centroid should be (1+2+3+4+5)/5 = 3.0 for all elements
        for i in 0..<32 {
            XCTAssertEqual(centroid![i], 3.0, accuracy: 1e-6)
        }
        
        // Test empty case
        let emptyCentroid = SyncBatchOperations.centroid(of: [Vector<Dim32>]())
        XCTAssertNil(emptyCentroid)
        
        // Test single vector
        let singleCentroid = SyncBatchOperations.centroid(of: [testVectors[0]])
        XCTAssertEqual(singleCentroid, testVectors[0])
    }
    
    func testWeightedCentroid() {
        let weights: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let weightedCentroid = SyncBatchOperations.weightedCentroid(
            of: testVectors,
            weights: weights
        )
        
        XCTAssertNotNil(weightedCentroid)
        
        // Weighted centroid should be (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / 15 = 55/15 ≈ 3.667
        let expectedValue: Float = 55.0 / 15.0
        for i in 0..<32 {
            XCTAssertEqual(weightedCentroid![i], expectedValue, accuracy: 1e-5)
        }
        
        // Test zero weights
        let zeroWeights = Array(repeating: Float(0), count: 5)
        let zeroWeightedCentroid = SyncBatchOperations.weightedCentroid(
            of: testVectors,
            weights: zeroWeights
        )
        XCTAssertNil(zeroWeightedCentroid)
    }
    
    func testSum() {
        let sum = SyncBatchOperations.sum(testVectors)
        
        XCTAssertNotNil(sum)
        
        // Sum should be 1+2+3+4+5 = 15 for all elements
        for i in 0..<32 {
            XCTAssertEqual(sum![i], 15.0, accuracy: 1e-6)
        }
    }
    
    func testMean() {
        let mean = SyncBatchOperations.mean(testVectors)
        
        XCTAssertNotNil(mean)
        
        // Mean should be same as centroid
        let centroid = SyncBatchOperations.centroid(of: testVectors)
        XCTAssertEqual(mean, centroid)
    }
    
    // MARK: - Statistical Tests
    
    func testStatistics() {
        let stats = SyncBatchOperations.statistics(for: testVectors)
        
        XCTAssertEqual(stats.count, 5)
        
        // Mean magnitude: sqrt(32) * (1+2+3+4+5)/5 = sqrt(32) * 3
        let expectedMean = sqrt(Float(32)) * 3.0
        XCTAssertEqual(stats.meanMagnitude, expectedMean, accuracy: 1e-5)
        
        // Standard deviation
        XCTAssertGreaterThan(stats.stdMagnitude, 0)
        
        // Test empty case
        let emptyStats = SyncBatchOperations.statistics(for: [Vector<Dim32>]())
        XCTAssertEqual(emptyStats.count, 0)
        XCTAssertEqual(emptyStats.meanMagnitude, 0)
        XCTAssertEqual(emptyStats.stdMagnitude, 0)
    }
    
    func testFindOutliers() {
        var vectors = testVectors
        
        // Add an outlier
        let outlier = Vector<Dim32>(Array(repeating: 100.0, count: 32))
        vectors.append(outlier)
        
        let outlierIndices = SyncBatchOperations.findOutliers(in: vectors, zscoreThreshold: 2)
        
        // Should find the outlier at index 5
        XCTAssertTrue(outlierIndices.contains(5))
    }
    
    // MARK: - Distance Matrix Tests
    
    func testPairwiseDistances() {
        let distances = SyncBatchOperations.pairwiseDistances(testVectors)
        
        XCTAssertEqual(distances.count, 5)
        XCTAssertEqual(distances[0].count, 5)
        
        // Check diagonal is zero
        for i in 0..<5 {
            XCTAssertEqual(distances[i][i], 0.0, accuracy: 1e-6)
        }
        
        // Check symmetry
        for i in 0..<5 {
            for j in 0..<5 {
                XCTAssertEqual(distances[i][j], distances[j][i], accuracy: 1e-6)
            }
        }
        
        // Check specific distances
        // Distance between vector[i] and vector[j] should be sqrt(32) * |i-j|
        let scale = sqrt(Float(32))
        XCTAssertEqual(distances[0][1], scale * 1.0, accuracy: 1e-5)
        XCTAssertEqual(distances[0][2], scale * 2.0, accuracy: 1e-5)
        XCTAssertEqual(distances[1][3], scale * 2.0, accuracy: 1e-5)
    }
    
    func testBatchDistances() {
        let queries = [testVectors[0], testVectors[1]]
        let candidates = [testVectors[2], testVectors[3], testVectors[4]]
        
        let distances = SyncBatchOperations.batchDistances(
            from: queries,
            to: candidates
        )
        
        XCTAssertEqual(distances.count, 2)
        XCTAssertEqual(distances[0].count, 3)
        
        // Check specific distances
        let scale = sqrt(Float(32))
        XCTAssertEqual(distances[0][0], scale * 2.0, accuracy: 1e-5) // 1.0 to 3.0
        XCTAssertEqual(distances[0][1], scale * 3.0, accuracy: 1e-5) // 1.0 to 4.0
        XCTAssertEqual(distances[1][0], scale * 1.0, accuracy: 1e-5) // 2.0 to 3.0
    }
    
    // MARK: - Clustering Support Tests
    
    func testAssignToCentroids() {
        let centroids = [testVectors[0], testVectors[4]] // 1.0 and 5.0
        
        let assignments = SyncBatchOperations.assignToCentroids(
            testVectors,
            centroids: centroids
        )
        
        XCTAssertEqual(assignments.count, 5)
        
        // Vectors 1.0, 2.0 should be assigned to centroid 0
        XCTAssertEqual(assignments[0], 0)
        XCTAssertEqual(assignments[1], 0)
        
        // Vectors 4.0, 5.0 should be assigned to centroid 1
        XCTAssertEqual(assignments[3], 1)
        XCTAssertEqual(assignments[4], 1)
        
        // Vector 3.0 is equidistant but should be assigned to one of them
        XCTAssertTrue(assignments[2] == 0 || assignments[2] == 1)
    }
    
    func testUpdateCentroids() {
        let assignments = [0, 0, 0, 1, 1] // First 3 in cluster 0, last 2 in cluster 1
        
        let newCentroids = SyncBatchOperations.updateCentroids(
            vectors: testVectors,
            assignments: assignments,
            k: 2
        )
        
        XCTAssertEqual(newCentroids.count, 2)
        
        // Cluster 0 centroid: (1+2+3)/3 = 2.0
        for i in 0..<32 {
            XCTAssertEqual(newCentroids[0][i], 2.0, accuracy: 1e-6)
        }
        
        // Cluster 1 centroid: (4+5)/2 = 4.5
        for i in 0..<32 {
            XCTAssertEqual(newCentroids[1][i], 4.5, accuracy: 1e-6)
        }
    }
    
    // MARK: - Sampling Tests
    
    func testRandomSample() {
        let k = 3
        let sample = SyncBatchOperations.randomSample(from: testVectors, k: k)
        
        XCTAssertEqual(sample.count, k)
        
        // Check all samples are from original set
        for vector in sample {
            XCTAssertTrue(testVectors.contains(vector))
        }
        
        // Test edge cases
        let emptySample = SyncBatchOperations.randomSample(from: testVectors, k: 0)
        XCTAssertTrue(emptySample.isEmpty)
        
        let fullSample = SyncBatchOperations.randomSample(from: testVectors, k: 10)
        XCTAssertEqual(fullSample.count, testVectors.count)
    }
    
    func testStratifiedSample() {
        // Create vectors with different magnitudes
        let vectors = [
            Vector<Dim32>.ones() * 1,
            Vector<Dim32>.ones() * 2,
            Vector<Dim32>.ones() * 3,
            Vector<Dim32>.ones() * 4,
            Vector<Dim32>.ones() * 5,
            Vector<Dim32>.ones() * 6,
            Vector<Dim32>.ones() * 7,
            Vector<Dim32>.ones() * 8,
            Vector<Dim32>.ones() * 9,
            Vector<Dim32>.ones() * 10,
        ]
        
        let sample = SyncBatchOperations.stratifiedSample(
            from: vectors,
            k: 5,
            strata: 5
        )
        
        XCTAssertEqual(sample.count, 5)
        
        // Check that samples come from different magnitude ranges
        let magnitudes = sample.map { $0.magnitude }
        let sortedMagnitudes = magnitudes.sorted()
        
        // Should have good spread across magnitude range
        XCTAssertGreaterThan(sortedMagnitudes.last! - sortedMagnitudes.first!, 5.0)
    }
    
    // MARK: - Array Extension Tests
    
    func testArrayExtensions() {
        // Test findNearest extension
        let query = Vector<Dim32>(Array(repeating: 2.5, count: 32))
        let neighbors = testVectors.findNearest(to: query, k: 2)
        XCTAssertEqual(neighbors.count, 2)
        
        // Test centroid extension
        let centroid = testVectors.centroid()
        XCTAssertNotNil(centroid)
        
        // Test pairwiseDistances extension
        let distances = testVectors.pairwiseDistances()
        XCTAssertEqual(distances.count, testVectors.count)
        
        // Test batchStatistics extension
        let stats = testVectors.batchStatistics
        XCTAssertEqual(stats.count, testVectors.count)
    }
    
    // MARK: - Performance Tests
    
    func testFindNearestPerformance() {
        let vectors = (0..<1000).map { _ in Vector<Dim128>.random(in: -1...1) }
        let query = Vector<Dim128>.random(in: -1...1)
        
        measure {
            _ = SyncBatchOperations.findNearest(to: query, in: vectors, k: 10)
        }
    }
    
    func testCentroidPerformance() {
        let vectors = (0..<1000).map { _ in Vector<Dim256>.random(in: -1...1) }
        
        measure {
            _ = SyncBatchOperations.centroid(of: vectors)
        }
    }
    
    func testPairwiseDistancesPerformance() {
        let vectors = (0..<100).map { _ in Vector<Dim128>.random(in: -1...1) }
        
        measure {
            _ = SyncBatchOperations.pairwiseDistances(vectors)
        }
    }
}