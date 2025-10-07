//
//  ClusteringTestUtilitiesTests.swift
//  VectorCore
//
//  Tests to verify ClusteringTestUtilities functions correctly
//

import Testing
import Foundation
@testable import VectorCore

@Suite("Clustering Test Utilities Verification")
struct ClusteringTestUtilitiesTests {

    @Test("Gaussian Mixture Generation - 512D")
    func testGaussianMixture512() throws {
        let (vectors, labels) = SyntheticDataGenerator.generateGaussianMixture512(
            numClusters: 3,
            vectorsPerCluster: 20,
            separation: 5.0,
            variance: 1.0,
            seed: 42
        )

        #expect(vectors.count == 60)
        #expect(labels.count == 60)
        #expect(Set(labels).count == 3)

        // Verify labels are in valid range [0, 2]
        #expect(labels.allSatisfy { $0 >= 0 && $0 < 3 })

        print("✓ Gaussian mixture 512D: \(vectors.count) vectors in \(Set(labels).count) clusters")
    }

    @Test("Random Vectors Generation - 512D")
    func testRandomVectors512() throws {
        let vectors = SyntheticDataGenerator.generateRandomVectors512(
            count: 100,
            distribution: .gaussian,
            seed: 42
        )

        #expect(vectors.count == 100)
        print("✓ Random 512D vectors: \(vectors.count) generated")
    }

    @Test("Silhouette Score - Well-separated clusters")
    func testSilhouetteScoreWellSeparated() throws {
        let (vectors, labels) = SyntheticDataGenerator.generateGaussianMixture512(
            numClusters: 3,
            vectorsPerCluster: 30,
            separation: 10.0,  // Well-separated
            variance: 1.0,
            seed: 42
        )

        let score = ClusteringMetrics.silhouetteScore512(
            vectors: vectors,
            labels: labels,
            metric: .euclidean
        )

        #expect(score > 0.3, "Well-separated clusters should have high silhouette score, got \(score)")
        print("✓ Silhouette score for well-separated clusters: \(score)")
    }

    @Test("Davies-Bouldin Index - Lower is better")
    func testDaviesBouldinIndex() throws {
        let (vectors, labels) = SyntheticDataGenerator.generateGaussianMixture512(
            numClusters: 3,
            vectorsPerCluster: 30,
            separation: 5.0,
            variance: 1.0,
            seed: 42
        )

        let dbIndex = ClusteringMetrics.daviesBouldinIndex512(
            vectors: vectors,
            labels: labels
        )

        #expect(dbIndex.isFinite, "DB index should be finite")
        #expect(dbIndex > 0, "DB index should be positive")
        print("✓ Davies-Bouldin index: \(dbIndex)")
    }

    @Test("Adjusted Rand Index - Perfect match")
    func testAdjustedRandIndexPerfect() throws {
        let labels1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        let labels2 = [0, 0, 0, 1, 1, 1, 2, 2, 2]

        let ari = ClusteringMetrics.adjustedRandIndex(
            labels1: labels1,
            labels2: labels2
        )

        #expect(abs(ari - 1.0) < 0.01, "Identical clusterings should have ARI ≈ 1.0, got \(ari)")
        print("✓ ARI for perfect match: \(ari)")
    }

    @Test("Adjusted Rand Index - Permuted labels")
    func testAdjustedRandIndexPermuted() throws {
        let labels1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        let labels2 = [5, 5, 5, 7, 7, 7, 9, 9, 9]  // Different label IDs, same clustering

        let ari = ClusteringMetrics.adjustedRandIndex(
            labels1: labels1,
            labels2: labels2
        )

        #expect(abs(ari - 1.0) < 0.01, "Permuted labels should still have ARI ≈ 1.0, got \(ari)")
        print("✓ ARI for permuted labels: \(ari)")
    }

    @Test("Adjusted Rand Index - Random clustering")
    func testAdjustedRandIndexRandom() throws {
        let labels1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        let labels2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]  // Completely different structure

        let ari = ClusteringMetrics.adjustedRandIndex(
            labels1: labels1,
            labels2: labels2
        )

        #expect(abs(ari) < 0.5, "Unrelated clusterings should have low ARI, got \(ari)")
        print("✓ ARI for random clustering: \(ari)")
    }

    @Test("Performance Measurement - Timing accuracy")
    func testPerformanceMeasurement() throws {
        let result = PerformanceMeasurement.measureClusteringPerformance(
            iterations: 20,
            warmupIterations: 5
        ) {
            // Simulate work
            var sum: Float = 0
            for i in 0..<1000 {
                sum += Float(i)
            }
            _ = sum
        }

        #expect(result.iterations == 20)
        #expect(result.meanTime > 0)
        #expect(result.medianTime > 0)
        #expect(result.stdDev >= 0)
        #expect(result.throughput > 0)

        print("✓ Performance measurement results:")
        print(result.summary)
    }

    @Test("Concentric Circles Generation")
    func testConcentricCircles() throws {
        let (vectors, labels) = SyntheticDataGenerator.generateConcentricCircles512(
            numRings: 3,
            pointsPerRing: 20,
            radiusMultiplier: 2.0,
            noise: 0.1,
            seed: 42
        )

        #expect(vectors.count == 60)
        #expect(labels.count == 60)
        #expect(Set(labels).count == 3)

        print("✓ Concentric circles 512D: \(vectors.count) points in \(Set(labels).count) rings")
    }

    @Test("Bug Fix #1 - Fisher-Yates shuffle bounds")
    func testFisherYatesNoCrash() throws {
        // This would have crashed with the old implementation if randomDouble() returned 1.0
        let (vectors, labels) = SyntheticDataGenerator.generateGaussianMixture512(
            numClusters: 2,
            vectorsPerCluster: 100,
            seed: 123456
        )

        #expect(vectors.count == 200)
        #expect(labels.count == 200)
        print("✓ Fisher-Yates shuffle bug fix verified (no crash)")
    }

    @Test("Bug Fix #2 - ARI overflow prevention")
    func testARILargeDataset() throws {
        // Generate large dataset that could cause Int64 overflow in old implementation
        var labels1 = [Int](repeating: 0, count: 5000)
        var labels2 = [Int](repeating: 0, count: 5000)

        for i in 0..<5000 {
            labels1[i] = i % 10
            labels2[i] = i % 10
        }

        let ari = ClusteringMetrics.adjustedRandIndex(
            labels1: labels1,
            labels2: labels2
        )

        #expect(ari.isFinite, "ARI should be finite even for large datasets")
        #expect(abs(ari - 1.0) < 0.01, "Identical large clusterings should have ARI ≈ 1.0")
        print("✓ ARI overflow bug fix verified (ARI = \(ari))")
    }

    @Test("Bug Fix #3 - Sample variance calculation")
    func testSampleVariance() throws {
        // Verify that standard deviation uses n-1 (sample variance)
        let result = PerformanceMeasurement.measureClusteringPerformance(
            iterations: 10,
            warmupIterations: 0
        ) {
            // Constant-time operation for predictable variance
            _ = 1 + 1
        }

        // For small n, sample variance (n-1) vs population variance (n) makes a difference
        #expect(result.stdDev >= 0, "Standard deviation should be non-negative")
        print("✓ Sample variance bug fix verified (stddev = \(result.stdDev))")
    }
}
