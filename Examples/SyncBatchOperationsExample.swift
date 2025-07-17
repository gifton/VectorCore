#!/usr/bin/swift

// VectorCore: Synchronous Batch Operations Example
//
// Demonstrates high-performance synchronous batch processing
//

import VectorCore
import Foundation

// MARK: - Example 1: k-NN Search

print("=== k-NN Search Example ===\n")

// Create a database of vectors
let database = (0..<1000).map { i in
    // Create vectors in different clusters
    let cluster = i / 250  // 4 clusters
    let baseValue = Float(cluster) * 2.0
    return Vector<Dim128>.random(in: baseValue...(baseValue + 1.0))
}

// Query vector
let query = Vector<Dim128>.random(in: 1.5...2.5)

// Find 10 nearest neighbors
let startTime = CFAbsoluteTimeGetCurrent()
let neighbors = SyncBatchOperations.findNearest(
    to: query,
    in: database,
    k: 10
)
let elapsed = CFAbsoluteTimeGetCurrent() - startTime

print("Found \(neighbors.count) nearest neighbors in \(String(format: "%.3f", elapsed * 1000))ms")
print("Nearest neighbor distance: \(neighbors[0].distance)")
print("Farthest neighbor distance: \(neighbors.last!.distance)")

// Find vectors within radius
let withinRadius = SyncBatchOperations.findWithinRadius(
    of: query,
    in: database,
    radius: 5.0
)
print("\nFound \(withinRadius.count) vectors within radius 5.0")

// MARK: - Example 2: Clustering Support

print("\n\n=== Clustering Example ===\n")

// Create sample data
let clusterData = [
    // Cluster 1: around [1, 1, 1, ...]
    (0..<100).map { _ in Vector<Dim64>.random(in: 0.5...1.5) },
    // Cluster 2: around [5, 5, 5, ...]
    (0..<100).map { _ in Vector<Dim64>.random(in: 4.5...5.5) },
    // Cluster 3: around [10, 10, 10, ...]
    (0..<100).map { _ in Vector<Dim64>.random(in: 9.5...10.5) }
].flatMap { $0 }.shuffled()

// Initialize centroids
var centroids = [
    Vector<Dim64>.ones() * 2,
    Vector<Dim64>.ones() * 6,
    Vector<Dim64>.ones() * 9
]

// Run k-means iterations
for iteration in 1...5 {
    // Assign to clusters
    let assignments = SyncBatchOperations.assignToCentroids(
        clusterData,
        centroids: centroids
    )
    
    // Update centroids
    centroids = SyncBatchOperations.updateCentroids(
        vectors: clusterData,
        assignments: assignments,
        k: 3
    )
    
    // Count cluster sizes
    var clusterSizes = [0, 0, 0]
    for assignment in assignments {
        clusterSizes[assignment] += 1
    }
    
    print("Iteration \(iteration): Cluster sizes = \(clusterSizes)")
}

print("\nFinal centroids:")
for (i, centroid) in centroids.enumerated() {
    print("  Cluster \(i): mean value = \(centroid[0])")
}

// MARK: - Example 3: Batch Statistics

print("\n\n=== Batch Statistics Example ===\n")

// Create vectors with varying magnitudes
let statVectors = [
    // Normal vectors
    (0..<90).map { _ in Vector<Dim32>.random(in: -1...1) },
    // Outliers
    (0..<10).map { _ in Vector<Dim32>.random(in: -10...10) }
].flatMap { $0 }

// Compute statistics
let stats = SyncBatchOperations.statistics(for: statVectors)
print("Vector count: \(stats.count)")
print("Mean magnitude: \(String(format: "%.3f", stats.meanMagnitude))")
print("Std deviation: \(String(format: "%.3f", stats.stdMagnitude))")

// Find outliers
let outliers = SyncBatchOperations.findOutliers(in: statVectors, zscoreThreshold: 2.5)
print("\nFound \(outliers.count) outliers (z-score > 2.5)")

// MARK: - Example 4: Batch Transformations

print("\n\n=== Batch Transformation Example ===\n")

// Create test vectors
let transformVectors = (0..<100).map { _ in 
    Vector<Dim128>.random(in: -2...2)
}

// Normalize all vectors
let normalized = SyncBatchOperations.map(transformVectors) { $0.normalized() }

// Filter vectors by magnitude
let largeMagnitude = SyncBatchOperations.filter(transformVectors) { vector in
    vector.magnitude > 10.0
}

// Partition by criterion
let (positive, negative) = SyncBatchOperations.partition(transformVectors) { vector in
    vector.mean > 0
}

print("Normalized \(normalized.count) vectors")
print("Found \(largeMagnitude.count) vectors with magnitude > 10")
print("Partitioned into \(positive.count) positive and \(negative.count) negative mean vectors")

// MARK: - Example 5: Aggregation Operations

print("\n\n=== Aggregation Example ===\n")

// Create direction vectors
let directions = [
    Vector<Dim32>.basis(axis: 0),   // X direction
    Vector<Dim32>.basis(axis: 1),   // Y direction
    Vector<Dim32>.basis(axis: 2),   // Z direction
    Vector<Dim32>.basis(axis: 0) + Vector<Dim32>.basis(axis: 1)  // XY diagonal
]

// Compute centroid
if let centroid = SyncBatchOperations.centroid(of: directions) {
    print("Centroid of directions: [\(centroid[0]), \(centroid[1]), \(centroid[2]), ...]")
}

// Weighted centroid (emphasize certain directions)
let weights: [Float] = [1.0, 1.0, 1.0, 3.0]  // Emphasize diagonal
if let weightedCentroid = SyncBatchOperations.weightedCentroid(of: directions, weights: weights) {
    print("Weighted centroid: [\(weightedCentroid[0]), \(weightedCentroid[1]), \(weightedCentroid[2]), ...]")
}

// MARK: - Example 6: Distance Matrices

print("\n\n=== Distance Matrix Example ===\n")

// Create a small set of vectors
let points = [
    Vector<Dim32>.zeros(),
    Vector<Dim32>.ones(),
    Vector<Dim32>.ones() * 2,
    Vector<Dim32>.ones() * 3
]

// Compute pairwise distances
let distanceMatrix = SyncBatchOperations.pairwiseDistances(points)

print("Pairwise distance matrix:")
for (i, row) in distanceMatrix.enumerated() {
    let rowStr = row.map { String(format: "%.2f", $0) }.joined(separator: " ")
    print("  [\(rowStr)]")
}

// MARK: - Example 7: Sampling

print("\n\n=== Sampling Example ===\n")

// Create large dataset
let largeDataset = (0..<10000).map { i in
    // Create vectors with varying magnitudes
    let magnitude = Float(i) / 1000.0
    return Vector<Dim128>.ones() * magnitude
}

// Random sample
let randomSample = SyncBatchOperations.randomSample(from: largeDataset, k: 10)
print("Random sample of 10 vectors (by magnitude):")
for vector in randomSample {
    print("  Magnitude: \(String(format: "%.3f", vector.magnitude))")
}

// Stratified sample
let stratifiedSample = SyncBatchOperations.stratifiedSample(
    from: largeDataset,
    k: 10,
    strata: 5
)
print("\nStratified sample of 10 vectors (5 strata):")
let sortedByMag = stratifiedSample.sorted { $0.magnitude < $1.magnitude }
for vector in sortedByMag {
    print("  Magnitude: \(String(format: "%.3f", vector.magnitude))")
}

// MARK: - Performance Comparison

print("\n\n=== Performance Comparison ===\n")

// Compare synchronous vs async for small dataset
let smallDataset = (0..<100).map { _ in Vector<Dim256>.random(in: -1...1) }
let query256 = Vector<Dim256>.random(in: -1...1)

// Synchronous timing
let syncStart = CFAbsoluteTimeGetCurrent()
for _ in 0..<100 {
    _ = SyncBatchOperations.findNearest(to: query256, in: smallDataset, k: 10)
}
let syncTime = CFAbsoluteTimeGetCurrent() - syncStart

print("Synchronous 100 iterations: \(String(format: "%.3f", syncTime * 1000))ms")
print("Average per operation: \(String(format: "%.3f", syncTime * 10))ms")

// Array extension convenience
print("\n\n=== Array Extension Usage ===\n")

let vectors = (0..<50).map { _ in Vector<Dim64>.random(in: -1...1) }

// Direct array methods
let nearest = vectors.findNearest(to: vectors[0], k: 5)
let centroid = vectors.centroid()
let stats2 = vectors.batchStatistics

print("Found \(nearest.count) nearest neighbors using array extension")
print("Centroid magnitude: \(centroid?.magnitude ?? 0)")
print("Batch statistics: mean=\(stats2.meanMagnitude), std=\(stats2.stdMagnitude)")

print("\n✅ All examples completed!")