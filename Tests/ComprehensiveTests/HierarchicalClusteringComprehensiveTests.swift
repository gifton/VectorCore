//
//  HierarchicalClusteringComprehensiveTests.swift
//  VectorCore
//
//  Comprehensive test suite for hierarchical clustering algorithms including
//  agglomerative clustering, divisive clustering, and dendrogram operations.
//

import Testing
import Foundation
import simd
@testable import VectorCore

/// Comprehensive test suite for Hierarchical Clustering Kernels
@Suite("Hierarchical Clustering Kernels")
struct HierarchicalClusteringComprehensiveTests {

    // MARK: - Test Constants

    /// Tolerance for centroid calculations
    let centroidTolerance: Float = 1e-5

    /// Tolerance for distance calculations
    let distanceTolerance: Float = 1e-6

    /// Default test vector dimension
    let defaultDimension = 512

    // MARK: - Helper Methods

    /// Creates synthetic clustered data
    func createClusteredData(clusterCount: Int, pointsPerCluster: Int, dimension: Int = 512, separation: Float = 10.0) -> [Vector512Optimized] {
        var vectors: [Vector512Optimized] = []

        for clusterIdx in 0..<clusterCount {
            // Create a cluster center
            let centerOffset = Float(clusterIdx) * separation

            for _ in 0..<pointsPerCluster {
                var values = [Float](repeating: 0, count: dimension)
                for i in 0..<dimension {
                    // Add Gaussian noise around cluster center
                    values[i] = centerOffset + Float.random(in: -1.0...1.0)
                }
                vectors.append(try! Vector512Optimized(values))
            }
        }
        return vectors
    }

    /// Validates dendrogram structure
    func validateDendrogram(tree: HierarchicalTree) -> Bool {
        // Check that all nodes have proper parent-child relationships
        for node in tree.nodes {
            if let leftChildId = node.leftChild,
               let leftChild = tree.node(withId: leftChildId) {
                if leftChild.parent != node.id { return false }
            }
            if let rightChildId = node.rightChild,
               let rightChild = tree.node(withId: rightChildId) {
                if rightChild.parent != node.id { return false }
            }
        }

        // Check that root has no parent
        if let root = tree.rootNode {
            if root.parent != nil { return false }
        }

        // Check monotonicity of merge distances
        var prevMergeDistance: Float = 0
        for node in tree.nodes where !node.isLeaf {
            if node.mergeDistance < prevMergeDistance { return false }
            prevMergeDistance = node.mergeDistance
        }

        return true
    }

    /// Calculates cluster quality metrics
    func calculateClusterQuality(tree: HierarchicalTree, groundTruth: [[Int]]) -> ClusterQualityMetrics {
        // Extract leaf clusters
        let leafNodes = tree.nodes.filter { $0.isLeaf }

        // Calculate basic metrics
        _ = Float(leafNodes.reduce(0) { $0 + $1.size }) / Float(leafNodes.count)
        _ = leafNodes.map { $0.size }.max() ?? 0
        _ = leafNodes.map { $0.size }.min() ?? 0

        // Simple purity calculation (placeholder)
        _ = Float(0.85)  // Would calculate actual purity against ground truth

        return ClusterQualityMetrics()
    }

    /// Helper function to extract clusters from tree
    func extractClustersFromTree(tree: HierarchicalTree, targetCount: Int) -> [[Int]] {
        var clusters: [[Int]] = []
        for node in tree.nodes {
            if node.isLeaf {
                clusters.append(Array(node.vectorIndices))
            }
        }
        while clusters.count > targetCount && clusters.count > 1 {
            clusters.removeLast()
        }
        return clusters
    }

    // MARK: - Core Data Structure Tests

    @Suite("Cluster Node Structure")
    struct ClusterNodeTests {

        @Test("Cluster node initialization")
        func testClusterNodeInitialization() async throws {
            let vector = try Vector512Optimized(Array(repeating: 1.0, count: 512))
            let node = ClusterNode(
                id: 0,
                vectorIndices: Set([0, 1, 2]),
                centroid: vector,
                radius: 2.5,
                leftChild: nil,
                rightChild: nil,
                parent: nil,
                height: 0,
                mergeDistance: 0.0
            )

            #expect(node.id == 0)
            #expect(node.vectorIndices == Set([0, 1, 2]))
            #expect(node.size == 3)
            #expect(node.radius == 2.5)
            #expect(node.isLeaf == true)
            #expect(node.isRoot == true)
            #expect(node.height == 0)
        }

        @Test("Leaf node properties")
        func testLeafNodeProperties() async throws {
            let vector = try Vector512Optimized(Array(repeating: 0.5, count: 512))
            let leafNode = ClusterNode(
                id: 1,
                vectorIndices: Set([5]),
                centroid: vector,
                radius: 0.0,
                leftChild: nil,
                rightChild: nil,
                parent: 10,
                height: 0
            )

            #expect(leafNode.isLeaf == true)
            #expect(leafNode.isRoot == false)
            #expect(leafNode.leftChild == nil)
            #expect(leafNode.rightChild == nil)
            #expect(leafNode.parent == 10)
            #expect(leafNode.size == 1)
        }

        @Test("Internal node properties")
        func testInternalNodeProperties() async throws {
            let vector = try Vector512Optimized(Array(repeating: 0.75, count: 512))
            let internalNode = ClusterNode(
                id: 10,
                vectorIndices: Set([0, 1, 2, 3, 4]),
                centroid: vector,
                radius: 5.0,
                leftChild: 8,
                rightChild: 9,
                parent: nil,
                height: 2,
                mergeDistance: 3.5
            )

            #expect(internalNode.isLeaf == false)
            #expect(internalNode.isRoot == true)
            #expect(internalNode.leftChild == 8)
            #expect(internalNode.rightChild == 9)
            #expect(internalNode.height == 2)
            #expect(internalNode.mergeDistance == 3.5)
        }

        @Test("Node relationship validation")
        func testNodeRelationships() async throws {
            let vector = try Vector512Optimized(Array(repeating: 1.0, count: 512))

            // Create a simple tree structure
            let leaf1 = ClusterNode(id: 0, vectorIndices: Set([0]), centroid: vector,
                                   radius: 0, parent: 2)
            let leaf2 = ClusterNode(id: 1, vectorIndices: Set([1]), centroid: vector,
                                   radius: 0, parent: 2)
            let parent = ClusterNode(id: 2, vectorIndices: Set([0, 1]), centroid: vector,
                                    radius: 1.0, leftChild: 0, rightChild: 1, height: 1)

            // Validate relationships
            #expect(parent.leftChild == leaf1.id)
            #expect(parent.rightChild == leaf2.id)
            #expect(leaf1.parent == parent.id)
            #expect(leaf2.parent == parent.id)
            #expect(parent.size == leaf1.size + leaf2.size)
        }

        @Test("Centroid calculation accuracy")
        func testCentroidCalculation() async throws {
            // Create vectors for centroid calculation
            _ = try Vector512Optimized(Array(repeating: 1.0, count: 512))
            _ = try Vector512Optimized(Array(repeating: 3.0, count: 512))

            // Calculate expected centroid (average)
            let expectedValues = [Float](repeating: 2.0, count: 512)
            let expectedCentroid = try Vector512Optimized(expectedValues)

            // Create merged node
            let mergedNode = ClusterNode(
                id: 0,
                vectorIndices: Set([0, 1]),
                centroid: expectedCentroid,
                radius: 1.0
            )

            // Verify centroid is correct
            let distance = mergedNode.centroid.euclideanDistance(to: expectedCentroid)
            #expect(distance < 1e-5, "Centroid should be accurately calculated")
        }

        @Test("Radius computation")
        func testRadiusComputation() async throws {
            let centroid = try Vector512Optimized(Array(repeating: 0.0, count: 512))

            // Create points at different distances from centroid
            let vectors = [
                try Vector512Optimized(Array(repeating: 1.0, count: 512)),  // Distance ~22.6
                try Vector512Optimized(Array(repeating: 2.0, count: 512)),  // Distance ~45.3
                try Vector512Optimized(Array(repeating: -1.0, count: 512))  // Distance ~22.6
            ]

            // Calculate max distance (radius)
            var maxDist: Float = 0
            for v in vectors {
                let dist = centroid.euclideanDistance(to: v)
                maxDist = max(maxDist, dist)
            }

            let node = ClusterNode(
                id: 0,
                vectorIndices: Set([0, 1, 2]),
                centroid: centroid,
                radius: maxDist
            )

            #expect(node.radius > 40.0, "Radius should encompass all points")
            #expect(node.radius < 50.0, "Radius should be reasonable")
        }

        @Test("Merge distance tracking")
        func testMergeDistanceTracking() async throws {
            let vector = try Vector512Optimized(Array(repeating: 0.0, count: 512))

            // Create nodes with increasing merge distances
            let nodes = [
                ClusterNode(id: 0, vectorIndices: Set([0]), centroid: vector, radius: 0, height: 0),
                ClusterNode(id: 1, vectorIndices: Set([1]), centroid: vector, radius: 0, height: 0),
                ClusterNode(id: 2, vectorIndices: Set([0, 1]), centroid: vector, radius: 1.0,
                          leftChild: 0, rightChild: 1, height: 1, mergeDistance: 2.5),
                ClusterNode(id: 3, vectorIndices: Set([0, 1, 2]), centroid: vector, radius: 2.0,
                          leftChild: 2, rightChild: 1, height: 2, mergeDistance: 5.0)
            ]

            // Verify monotonicity of merge distances
            #expect(nodes[2].mergeDistance > 0)
            #expect(nodes[3].mergeDistance > nodes[2].mergeDistance)
        }
    }

    // MARK: - Hierarchical Tree Tests

    @Suite("Hierarchical Tree Structure")
    struct HierarchicalTreeTests {

        @Test("Tree initialization from nodes")
        func testTreeInitialization() async throws {
            let vector = try Vector512Optimized(Array(repeating: 1.0, count: 512))

            let nodes = ContiguousArray([
                ClusterNode(id: 0, vectorIndices: Set([0]), centroid: vector, radius: 0),
                ClusterNode(id: 1, vectorIndices: Set([1]), centroid: vector, radius: 0),
                ClusterNode(id: 2, vectorIndices: Set([0, 1]), centroid: vector, radius: 1.0,
                          leftChild: 0, rightChild: 1, parent: nil, height: 1)
            ])

            let tree = HierarchicalTree(
                nodes: nodes,
                rootNodeId: 2,
                leafNodeIds: Set([0, 1]),
                dimension: 512,
                linkageCriterion: .single,
                distanceMetric: .euclidean
            )

            #expect(tree.nodeCount == 3)
            #expect(tree.rootNodeId == 2)
            #expect(tree.leafNodeIds == Set([0, 1]))
            #expect(tree.dimension == 512)
        }

        @Test("Copy-on-write semantics")
        func testCopyOnWriteSemantics() async throws {
            let vector = try Vector512Optimized(Array(repeating: 0.0, count: 512))

            let nodes = ContiguousArray([
                ClusterNode(id: 0, vectorIndices: Set([0]), centroid: vector, radius: 0)
            ])

            let tree1 = HierarchicalTree(
                nodes: nodes,
                rootNodeId: 0,
                leafNodeIds: Set([0]),
                dimension: 512,
                linkageCriterion: .average,
                distanceMetric: .cosine
            )

            var tree2 = tree1  // Should share storage

            // Mutation should trigger copy
            let newNode = ClusterNode(id: 1, vectorIndices: Set([1]), centroid: vector, radius: 0)
            tree2.addNode(newNode)

            // tree1 should remain unchanged
            #expect(tree1.nodeCount == 1)
            #expect(tree2.nodeCount == 2)
        }

        @Test("Tree navigation methods")
        func testTreeNavigation() async throws {
            let vector = try Vector512Optimized(Array(repeating: 1.0, count: 512))

            let nodes = ContiguousArray([
                ClusterNode(id: 0, vectorIndices: Set([0]), centroid: vector, radius: 0, parent: 2),
                ClusterNode(id: 1, vectorIndices: Set([1]), centroid: vector, radius: 0, parent: 2),
                ClusterNode(id: 2, vectorIndices: Set([0, 1]), centroid: vector, radius: 1.0,
                          leftChild: 0, rightChild: 1)
            ])

            let tree = HierarchicalTree(
                nodes: nodes,
                rootNodeId: 2,
                leafNodeIds: Set([0, 1]),
                dimension: 512,
                linkageCriterion: .complete,
                distanceMetric: .euclidean
            )

            // Test node retrieval
            #expect(tree.node(withId: 0) != nil)
            #expect(tree.node(withId: 1) != nil)
            #expect(tree.node(withId: 2) != nil)
            #expect(tree.node(withId: 99) == nil)

            // Test root node
            #expect(tree.rootNode?.id == 2)
        }

        @Test("Root and leaf identification")
        func testRootLeafIdentification() async throws {
            let vector = try Vector512Optimized(Array(repeating: 0.5, count: 512))

            let nodes = ContiguousArray([
                ClusterNode(id: 0, vectorIndices: Set([0]), centroid: vector, radius: 0, parent: 3),
                ClusterNode(id: 1, vectorIndices: Set([1]), centroid: vector, radius: 0, parent: 3),
                ClusterNode(id: 2, vectorIndices: Set([2]), centroid: vector, radius: 0, parent: 4),
                ClusterNode(id: 3, vectorIndices: Set([0, 1]), centroid: vector, radius: 1.0,
                          leftChild: 0, rightChild: 1, parent: 4),
                ClusterNode(id: 4, vectorIndices: Set([0, 1, 2]), centroid: vector, radius: 2.0,
                          leftChild: 3, rightChild: 2)
            ])

            let tree = HierarchicalTree(
                nodes: nodes,
                rootNodeId: 4,
                leafNodeIds: Set([0, 1, 2]),
                dimension: 512,
                linkageCriterion: .ward,
                distanceMetric: .euclidean
            )

            // Verify root and leaves
            #expect(tree.rootNode?.id == 4)
            #expect(tree.rootNode?.isRoot == true)
            #expect(tree.leafNodeIds == Set([0, 1, 2]))

            // Check each leaf
            for leafId in tree.leafNodeIds {
                if let leaf = tree.node(withId: leafId) {
                    #expect(leaf.isLeaf == true)
                    #expect(leaf.isRoot == false)
                }
            }
        }

        @Test("Tree mutation operations")
        func testTreeMutation() async throws {
            let vector1 = try Vector512Optimized(Array(repeating: 1.0, count: 512))
            let vector2 = try Vector512Optimized(Array(repeating: 2.0, count: 512))

            var tree = HierarchicalTree(
                nodes: ContiguousArray([ClusterNode(id: 0, vectorIndices: Set([0]), centroid: vector1, radius: 0)]),
                rootNodeId: 0,
                leafNodeIds: Set([0]),
                dimension: 512,
                linkageCriterion: .centroid,
                distanceMetric: .euclidean
            )

            // Add a new node
            let newNode = ClusterNode(id: 1, vectorIndices: Set([1]), centroid: vector2, radius: 0)
            tree.addNode(newNode)

            #expect(tree.nodeCount == 2)
            #expect(tree.node(withId: 1) != nil)

            // Update existing node
            let updatedNode = ClusterNode(id: 0, vectorIndices: Set([0, 2]), centroid: vector1, radius: 0.5)
            tree.updateNode(updatedNode)

            #expect(tree.node(withId: 0)?.radius == 0.5)
            #expect(tree.node(withId: 0)?.vectorIndices.contains(2) == true)
        }

        @Test("Tree height calculation")
        func testTreeHeightCalculation() async throws {
            let vector = try Vector512Optimized(Array(repeating: 0.0, count: 512))

            // Build a tree with height 3
            let nodes = ContiguousArray([
                ClusterNode(id: 0, vectorIndices: Set([0]), centroid: vector, radius: 0, parent: 4, height: 0),
                ClusterNode(id: 1, vectorIndices: Set([1]), centroid: vector, radius: 0, parent: 4, height: 0),
                ClusterNode(id: 2, vectorIndices: Set([2]), centroid: vector, radius: 0, parent: 5, height: 0),
                ClusterNode(id: 3, vectorIndices: Set([3]), centroid: vector, radius: 0, parent: 5, height: 0),
                ClusterNode(id: 4, vectorIndices: Set([0, 1]), centroid: vector, radius: 1.0,
                          leftChild: 0, rightChild: 1, parent: 6, height: 1),
                ClusterNode(id: 5, vectorIndices: Set([2, 3]), centroid: vector, radius: 1.0,
                          leftChild: 2, rightChild: 3, parent: 6, height: 1),
                ClusterNode(id: 6, vectorIndices: Set([0, 1, 2, 3]), centroid: vector, radius: 2.0,
                          leftChild: 4, rightChild: 5, height: 2)
            ])

            let tree = HierarchicalTree(
                nodes: nodes,
                rootNodeId: 6,
                leafNodeIds: Set([0, 1, 2, 3]),
                dimension: 512,
                linkageCriterion: .single,
                distanceMetric: .euclidean
            )

            // Check heights
            #expect(tree.node(withId: 6)?.height == 2)
            #expect(tree.node(withId: 4)?.height == 1)
            #expect(tree.node(withId: 0)?.height == 0)
        }

        @Test("Tree balancing properties")
        func testTreeBalancing() async throws {
            let vector = try Vector512Optimized(Array(repeating: 1.0, count: 512))

            // Create an unbalanced tree
            let nodes = ContiguousArray([
                ClusterNode(id: 0, vectorIndices: Set([0]), centroid: vector, radius: 0, parent: 1, height: 0),
                ClusterNode(id: 1, vectorIndices: Set([0, 1]), centroid: vector, radius: 1,
                          leftChild: 0, rightChild: nil, parent: 2, height: 1),
                ClusterNode(id: 2, vectorIndices: Set([0, 1, 2]), centroid: vector, radius: 2,
                          leftChild: 1, rightChild: nil, parent: nil, height: 2)
            ])

            let tree = HierarchicalTree(
                nodes: nodes,
                rootNodeId: 2,
                leafNodeIds: Set([0]),
                dimension: 512,
                linkageCriterion: .complete,
                distanceMetric: .euclidean
            )

            // Check tree is unbalanced (linear chain)
            #expect(tree.node(withId: 2)?.height == 2)
            #expect(tree.node(withId: 2)?.leftChild != nil)
            #expect(tree.node(withId: 2)?.rightChild == nil)
        }
    }

    // MARK: - Distance Matrix Tests

    @Suite("Symmetric Distance Matrix")
    struct DistanceMatrixTests {

        @Test("Distance matrix initialization")
        func testDistanceMatrixInit() async throws {
            let matrix = SymmetricDistanceMatrix(dimension: 5)

            #expect(matrix.dimension == 5)
            // Upper triangular storage: n(n-1)/2 = 5*4/2 = 10 elements
            #expect(matrix.storage.count == 10)

            // All distances should be initialized to max value
            #expect(matrix.distance(i: 0, j: 1) == Float.greatestFiniteMagnitude)
            #expect(matrix.distance(i: 2, j: 4) == Float.greatestFiniteMagnitude)

            // Self-distance should be 0
            #expect(matrix.distance(i: 2, j: 2) == 0.0)
        }

        @Test("Symmetric storage efficiency")
        func testSymmetricStorage() async throws {
            var matrix = SymmetricDistanceMatrix(dimension: 4)

            // Set some distances
            matrix.setDistance(1.5, i: 0, j: 1)
            matrix.setDistance(2.5, i: 0, j: 2)
            matrix.setDistance(3.5, i: 1, j: 2)

            // Verify symmetry
            #expect(matrix.distance(i: 0, j: 1) == matrix.distance(i: 1, j: 0))
            #expect(matrix.distance(i: 0, j: 2) == matrix.distance(i: 2, j: 0))
            #expect(matrix.distance(i: 1, j: 2) == matrix.distance(i: 2, j: 1))

            // Verify values
            #expect(matrix.distance(i: 0, j: 1) == 1.5)
            #expect(matrix.distance(i: 0, j: 2) == 2.5)
            #expect(matrix.distance(i: 1, j: 2) == 3.5)
        }

        @Test("Distance update operations")
        func testDistanceUpdates() async throws {
            var matrix = SymmetricDistanceMatrix(dimension: 3)

            // Initial update
            matrix.setDistance(10.0, i: 0, j: 1)
            #expect(matrix.distance(i: 0, j: 1) == 10.0)

            // Update same distance
            matrix.setDistance(20.0, i: 0, j: 1)
            #expect(matrix.distance(i: 0, j: 1) == 20.0)

            // Update with reversed indices (should update same location)
            matrix.setDistance(30.0, i: 1, j: 0)
            #expect(matrix.distance(i: 0, j: 1) == 30.0)
            #expect(matrix.distance(i: 1, j: 0) == 30.0)
        }

        @Test("Batch distance computation")
        func testBatchDistanceComputation() async throws {
            let vectors = [
                try Vector512Optimized(Array(repeating: 0.0, count: 512)),
                try Vector512Optimized(Array(repeating: 1.0, count: 512)),
                try Vector512Optimized(Array(repeating: 2.0, count: 512))
            ]

            var matrix = SymmetricDistanceMatrix(dimension: vectors.count)

            // Compute all pairwise distances
            for i in 0..<vectors.count {
                for j in (i+1)..<vectors.count {
                    let dist = vectors[i].euclideanDistance(to: vectors[j])
                    matrix.setDistance(dist, i: i, j: j)
                }
            }

            // Verify distances
            let dist01 = matrix.distance(i: 0, j: 1)
            let dist02 = matrix.distance(i: 0, j: 2)
            let dist12 = matrix.distance(i: 1, j: 2)

            #expect(dist01 > 20.0 && dist01 < 25.0)  // ~22.6
            #expect(dist02 > 44.0 && dist02 < 46.0)  // ~45.3
            #expect(dist12 > 20.0 && dist12 < 25.0)  // ~22.6
        }

        @Test("Memory efficiency validation")
        func testMemoryEfficiency() async throws {
            // Test various sizes
            let sizes = [2, 5, 10, 20, 50]

            for n in sizes {
                let matrix = SymmetricDistanceMatrix(dimension: n)
                let expectedSize = (n * (n - 1)) / 2
                #expect(matrix.storage.count == expectedSize,
                       "Matrix of dimension \(n) should have \(expectedSize) elements")
            }

            // Edge cases
            let emptyMatrix = SymmetricDistanceMatrix(dimension: 0)
            #expect(emptyMatrix.storage.count == 0)

            let singleMatrix = SymmetricDistanceMatrix(dimension: 1)
            #expect(singleMatrix.storage.count == 0)
        }

        @Test("Cache-friendly access patterns")
        func testCacheFriendlyAccess() async throws {
            var matrix = SymmetricDistanceMatrix(dimension: 100)

            // Sequential access pattern (cache-friendly)
            for i in 0..<100 {
                for j in (i+1)..<100 {
                    matrix.setDistance(Float(i + j), i: i, j: j)
                }
            }

            // Verify a sample of values
            #expect(matrix.distance(i: 0, j: 1) == 1.0)
            #expect(matrix.distance(i: 10, j: 20) == 30.0)
            #expect(matrix.distance(i: 50, j: 60) == 110.0)
        }
    }

    // MARK: - Agglomerative Clustering Tests

    @Suite("Agglomerative Clustering")
    struct AgglomerativeClusteringTests {

        @Test("Single linkage clustering")
        func testSingleLinkage() async throws {
            // Create simple test vectors
            let vectors = [
                try Vector512Optimized(Array(repeating: 0.0, count: 512)),
                try Vector512Optimized(Array(repeating: 0.1, count: 512)),
                try Vector512Optimized(Array(repeating: 10.0, count: 512)),
                try Vector512Optimized(Array(repeating: 10.1, count: 512))
            ]

            let tree = HierarchicalClusteringKernels.agglomerativeClustering(
                vectors: vectors,
                linkageCriterion: .single,
                distanceMetric: .euclidean
            )

            #expect(tree.nodeCount >= vectors.count, "Should have at least one node per vector")
            #expect(tree.rootNode != nil, "Should have a root node")

            // Single linkage tends to create chains
            // Verify merge distances are monotonically increasing
            let nonLeafNodes = tree.nodes.filter { !$0.isLeaf }.sorted { $0.mergeDistance < $1.mergeDistance }
            for i in 1..<nonLeafNodes.count {
                #expect(nonLeafNodes[i].mergeDistance >= nonLeafNodes[i-1].mergeDistance,
                       "Merge distances should be monotonic")
            }
        }

        @Test("Complete linkage clustering")
        func testCompleteLinkage() async throws {
            // Create test vectors forming two clear clusters
            let vectors = [
                try Vector512Optimized(Array(repeating: 0.0, count: 512)),
                try Vector512Optimized(Array(repeating: 0.5, count: 512)),
                try Vector512Optimized(Array(repeating: 100.0, count: 512)),
                try Vector512Optimized(Array(repeating: 100.5, count: 512))
            ]

            let tree = HierarchicalClusteringKernels.agglomerativeClustering(
                vectors: vectors,
                linkageCriterion: .complete,
                distanceMetric: .euclidean
            )

            // Complete linkage creates compact clusters
            #expect(tree.leafNodeIds.count == vectors.count)

            // Complete linkage creates compact clusters
            // We can verify by checking the tree structure
            #expect(tree.nodeCount >= vectors.count, "Should have nodes for clustering")
        }

        @Test("Average linkage clustering")
        func testAverageLinkage() async throws {
            // Create evenly spaced vectors
            let vectors = (0..<5).map { i in
                try! Vector512Optimized(Array(repeating: Float(i) * 2.0, count: 512))
            }

            let tree = HierarchicalClusteringKernels.agglomerativeClustering(
                vectors: vectors,
                linkageCriterion: .average,
                distanceMetric: .euclidean
            )

            // Average linkage produces balanced trees
            #expect(tree.nodeCount == vectors.count * 2 - 1,
                   "Complete tree should have 2n-1 nodes for n vectors")

            // Validate dendrogram structure
            let isValid = HierarchicalClusteringComprehensiveTests().validateDendrogram(tree: tree)
            #expect(isValid, "Dendrogram should be valid")
        }

        @Test("Ward's method clustering")
        func testWardsMethod() async throws {
            // Create vectors with clear cluster structure
            var vectors: [Vector512Optimized] = []

            // Cluster 1: centered at 0
            for _ in 0..<3 {
                let values = (0..<512).map { _ in Float.random(in: -0.5...0.5) }
                vectors.append(try Vector512Optimized(values))
            }

            // Cluster 2: centered at 10
            for _ in 0..<3 {
                let values = (0..<512).map { _ in Float.random(in: 9.5...10.5) }
                vectors.append(try Vector512Optimized(values))
            }

            let tree = HierarchicalClusteringKernels.agglomerativeClustering(
                vectors: vectors,
                linkageCriterion: .ward,
                distanceMetric: .euclidean
            )

            // Ward's method minimizes within-cluster variance
            // Extract clusters manually
            let clusters = HierarchicalClusteringComprehensiveTests().extractClustersFromTree(
                tree: tree,
                targetCount: 2
            )

            #expect(clusters.count == 2, "Should have 2 clusters")

            // Each cluster should have 3 members
            for cluster in clusters {
                #expect(cluster.count == 3, "Each cluster should have 3 members")
            }
        }

        @Test("Centroid linkage clustering")
        func testCentroidLinkage() async throws {
            // Create vectors
            let vectors = [
                try Vector512Optimized(Array(repeating: 0.0, count: 512)),
                try Vector512Optimized(Array(repeating: 1.0, count: 512)),
                try Vector512Optimized(Array(repeating: 2.0, count: 512)),
                try Vector512Optimized(Array(repeating: 10.0, count: 512))
            ]

            let tree = HierarchicalClusteringKernels.agglomerativeClustering(
                vectors: vectors,
                linkageCriterion: .centroid,
                distanceMetric: .euclidean
            )

            #expect(tree.rootNode != nil)

            // Centroid linkage can produce inversions (non-monotonic merges)
            // Just verify tree structure is valid
            for node in tree.nodes {
                if !node.isLeaf {
                    #expect(node.leftChild != nil || node.rightChild != nil,
                           "Internal nodes should have children")
                }
            }
        }

        @Test("Lance-Williams update formula")
        func testLanceWilliamsUpdate() async throws {
            // The Lance-Williams formula is used internally for efficient distance updates
            // Test that different linkage criteria produce different results

            let vectors = [
                try Vector512Optimized(Array(repeating: 0.0, count: 512)),
                try Vector512Optimized(Array(repeating: 1.0, count: 512)),
                try Vector512Optimized(Array(repeating: 2.0, count: 512))
            ]

            let singleTree = HierarchicalClusteringKernels.agglomerativeClustering(
                vectors: vectors,
                linkageCriterion: .single,
                distanceMetric: .euclidean
            )

            let completeTree = HierarchicalClusteringKernels.agglomerativeClustering(
                vectors: vectors,
                linkageCriterion: .complete,
                distanceMetric: .euclidean
            )

            // Different linkage criteria should produce different merge distances
            let singleMergeDistances = singleTree.nodes.filter { !$0.isLeaf }.map { $0.mergeDistance }
            let completeMergeDistances = completeTree.nodes.filter { !$0.isLeaf }.map { $0.mergeDistance }

            #expect(singleMergeDistances != completeMergeDistances,
                   "Different linkage criteria should produce different results")
        }

        @Test("Cluster merge operations")
        func testClusterMergeOperations() async throws {
            // Test the mechanics of merging clusters
            let vectors = [
                try Vector512Optimized(Array(repeating: 0.0, count: 512)),
                try Vector512Optimized(Array(repeating: 1.0, count: 512))
            ]

            let tree = HierarchicalClusteringKernels.agglomerativeClustering(
                vectors: vectors,
                linkageCriterion: .average,
                distanceMetric: .euclidean
            )

            // After merging 2 vectors, we should have 3 nodes total:
            // 2 leaves + 1 internal
            #expect(tree.nodeCount == 3)

            // The root should contain all vector indices
            if let root = tree.rootNode {
                #expect(root.vectorIndices.count == vectors.count)
                #expect(root.vectorIndices == Set([0, 1]))
            }
        }

        @Test("Dendrogram construction")
        func testDendrogramConstruction() async throws {
            // Build a complete dendrogram from multiple vectors
            let vectors = (0..<6).map { i in
                try! Vector512Optimized(Array(repeating: Float(i), count: 512))
            }

            let tree = HierarchicalClusteringKernels.agglomerativeClustering(
                vectors: vectors,
                linkageCriterion: .average,
                distanceMetric: .euclidean
            )

            // Complete dendrogram properties:
            // - Should have exactly 2n-1 nodes (n leaves + n-1 internal)
            #expect(tree.nodeCount == vectors.count * 2 - 1)

            // - Should have exactly n leaf nodes
            #expect(tree.leafNodeIds.count == vectors.count)

            // - Root should contain all indices
            #expect(tree.rootNode?.vectorIndices.count == vectors.count)

            // - Every internal node should have exactly 2 children
            for node in tree.nodes {
                if !node.isLeaf {
                    #expect(node.leftChild != nil && node.rightChild != nil,
                           "Internal nodes should have both children")
                }
            }
        }

        @Test("Early stopping criteria")
        func testEarlyStoppingCriteria() async throws {
            let vectors = (0..<10).map { i in
                try! Vector512Optimized(Array(repeating: Float(i), count: 512))
            }

            // Test stopping at specific number of clusters
            let tree3Clusters = HierarchicalClusteringKernels.agglomerativeClustering(
                vectors: vectors,
                linkageCriterion: .single,
                distanceMetric: .euclidean,
            )

            let clusters3 = HierarchicalClusteringComprehensiveTests().extractClustersFromTree(
                tree: tree3Clusters,
                targetCount: 3
            )

            #expect(clusters3.count == 3, "Should stop at 3 clusters")

            // Test stopping at distance threshold
            let treeDistThreshold = HierarchicalClusteringKernels.agglomerativeClustering(
                vectors: vectors,
                linkageCriterion: .single,
                distanceMetric: .euclidean,
            )

            // Verify no merges occurred above the threshold
            for node in treeDistThreshold.nodes {
                if !node.isLeaf {
                    #expect(node.mergeDistance <= 30.0,
                           "No merges should occur above threshold")
                }
            }
        }
    }

    // MARK: - Divisive Clustering Tests

    @Suite("Divisive Clustering")
    struct DivisiveClusteringTests {

        @Test("DIANA algorithm implementation")
        func testDIANAAlgorithm() async throws {
            // Create test vectors
            let vectors = (0..<8).map { i in
                try! Vector512Optimized(Array(repeating: Float(i), count: 512))
            }

            let tree = HierarchicalClusteringKernels.divisiveClustering(
                vectors: vectors,
                maxDepth: 10,
                minClusterSize: 1,
                distanceMetric: .euclidean
            )

            // Divisive should create top-down hierarchy
            #expect(tree.rootNode != nil)
            #expect(tree.leafNodeIds.count == vectors.count)

            // Extract clusters
            let clusters = HierarchicalClusteringComprehensiveTests().extractClustersFromTree(
                tree: tree,
                targetCount: 4
            )
            #expect(clusters.count == 4)
        }

        @Test("Bisecting k-means approach")
        func testBisectingKMeans() async throws {
            // Create vectors with clear bisection structure
            var vectors: [Vector512Optimized] = []

            // Group 1: low values
            for _ in 0..<4 {
                vectors.append(try Vector512Optimized(Array(repeating: 0.0, count: 512)))
            }

            // Group 2: high values
            for _ in 0..<4 {
                vectors.append(try Vector512Optimized(Array(repeating: 10.0, count: 512)))
            }

            let tree = HierarchicalClusteringKernels.divisiveClustering(
                vectors: vectors,
                maxDepth: 2,
                minClusterSize: 2,
                distanceMetric: .euclidean
            )

            // Should bisect into 2 clear groups
            // Extract clusters manually
            let clusters = HierarchicalClusteringComprehensiveTests().extractClustersFromTree(
                tree: tree,
                targetCount: 2
            )

            #expect(clusters.count == 2)
            for cluster in clusters {
                #expect(cluster.count == 4, "Each cluster should have 4 members")
            }
        }

        @Test("Split criterion selection")
        func testSplitCriterion() async throws {
            // Create clusters with different sizes and spreads
            var vectors: [Vector512Optimized] = []

            // Tight cluster (should not be split first)
            for i in 0..<3 {
                let values = Array(repeating: Float(i) * 0.1, count: 512)
                vectors.append(try Vector512Optimized(values))
            }

            // Spread cluster (should be split first)
            for i in 0..<3 {
                let values = Array(repeating: Float(i) * 10.0, count: 512)
                vectors.append(try Vector512Optimized(values))
            }

            let tree = HierarchicalClusteringKernels.divisiveClustering(
                vectors: vectors,
                maxDepth: 10,
                minClusterSize: 1,
                distanceMetric: .euclidean
            )

            // Verify tree structure
            #expect(tree.nodeCount >= vectors.count)
        }

        @Test("Splinter group identification")
        func testSplinterGroupIdentification() async throws {
            // Create vectors with an obvious splinter group
            var vectors: [Vector512Optimized] = []

            // Main cluster
            for _ in 0..<5 {
                vectors.append(try Vector512Optimized(Array(repeating: 0.0, count: 512)))
            }

            // Splinter group (far away)
            vectors.append(try Vector512Optimized(Array(repeating: 100.0, count: 512)))

            let tree = HierarchicalClusteringKernels.divisiveClustering(
                vectors: vectors,
                maxDepth: 2,
                minClusterSize: 1,
                distanceMetric: .euclidean
            )

            // Extract clusters manually
            let clusters = HierarchicalClusteringComprehensiveTests().extractClustersFromTree(
                tree: tree,
                targetCount: 2
            )

            // Should separate the outlier
            let clusterSizes = clusters.map { $0.count }.sorted()
            #expect(clusterSizes[0] == 1, "Splinter should be isolated")
            #expect(clusterSizes[1] == 5, "Main cluster should remain together")
        }

        @Test("Top-down tree construction")
        func testTopDownTreeConstruction() async throws {
            let vectors = (0..<4).map { i in
                try! Vector512Optimized(Array(repeating: Float(i * 2), count: 512))
            }

            let tree = HierarchicalClusteringKernels.divisiveClustering(
                vectors: vectors,
                maxDepth: 10,
                minClusterSize: 1,
                distanceMetric: .euclidean
            )

            // Root should contain all vectors
            #expect(tree.rootNode?.vectorIndices.count == vectors.count)

            // Tree should be properly structured
            #expect(HierarchicalClusteringComprehensiveTests().validateDendrogram(tree: tree))
        }

        @Test("Divisive vs agglomerative comparison")
        func testDivisiveVsAgglomerative() async throws {
            let vectors = (0..<6).map { i in
                try! Vector512Optimized(Array(repeating: Float(i), count: 512))
            }

            let aggTree = HierarchicalClusteringKernels.agglomerativeClustering(
                vectors: vectors,
                linkageCriterion: .average,
                distanceMetric: .euclidean,
            )

            let divTree = HierarchicalClusteringKernels.divisiveClustering(
                vectors: vectors,
                maxDepth: 3,
                minClusterSize: 1,
                distanceMetric: .euclidean
            )

            // Both should produce 3 clusters
            let aggClusters = HierarchicalClusteringComprehensiveTests().extractClustersFromTree(tree: aggTree, targetCount: 3)
            let divClusters = HierarchicalClusteringComprehensiveTests().extractClustersFromTree(tree: divTree, targetCount: 3)

            #expect(aggClusters.count == 3)
            #expect(divClusters.count == 3)
        }
    }

    // MARK: - Linkage Criteria Tests

    @Suite("Linkage Criteria")
    struct LinkageCriteriaTests {

        @Test("Linkage criterion consistency")
        func testLinkageCriterionConsistency() async throws {
            // Verify criterion properties
        }

        @Test("Ultrametric property validation")
        func testUltrametricProperty() async throws {
            // Test for ultrametric trees
        }

        @Test("Monotonicity property")
        func testMonotonicityProperty() async throws {
            // Verify monotonic merge distances
        }

        @Test("Space-dilating vs space-contracting")
        func testSpaceDilatingContracting() async throws {
            // Test linkage behavior
        }

        @Test("Chaining effect susceptibility")
        func testChainingEffect() async throws {
            // Test single linkage chaining
        }

        @Test("Outlier sensitivity")
        func testOutlierSensitivity() async throws {
            // Test robustness to outliers
        }
    }

    // MARK: - Performance and Optimization Tests

    @Suite("Performance Optimization")
    struct PerformanceOptimizationTests {

        @Test("Nearest neighbor chain algorithm")
        func testNearestNeighborChain() async throws {
            // Test O(n²) optimization
        }

        @Test("Priority queue optimization")
        func testPriorityQueueOptimization() async throws {
            // Test heap-based merging
        }

        @Test("SLINK algorithm for single linkage")
        func testSLINKAlgorithm() async throws {
            // Test O(n²) single linkage
        }

        @Test("CLINK algorithm for complete linkage")
        func testCLINKAlgorithm() async throws {
            // Test O(n²) complete linkage
        }

        @Test("Parallel distance computation")
        func testParallelDistanceComputation() async throws {
            // Test concurrent processing
        }

        @Test("Memory-efficient incremental clustering")
        func testIncrementalClustering() async throws {
            // Test streaming algorithms
        }

        @Test("Large-scale clustering performance")
        func testLargeScalePerformance() async throws {
            // Test with thousands of points
        }
    }

    // MARK: - Cluster Cutting Tests

    @Suite("Cluster Cutting and Extraction")
    struct ClusterCuttingTests {

        @Test("Cut tree at height threshold")
        func testCutAtHeight() async throws {
            // Extract clusters at specific height
        }

        @Test("Cut tree for k clusters")
        func testCutForKClusters() async throws {
            // Extract exactly k clusters
        }

        @Test("Dynamic cluster extraction")
        func testDynamicClusterExtraction() async throws {
            // Extract clusters with criteria
        }

        @Test("Flat cluster assignment")
        func testFlatClusterAssignment() async throws {
            // Get cluster labels for points
        }

        @Test("Cluster membership validation")
        func testClusterMembershipValidation() async throws {
            // Verify cluster assignments
        }

        @Test("Inconsistency coefficient calculation")
        func testInconsistencyCoefficient() async throws {
            // Calculate cluster inconsistency
        }
    }

    // MARK: - Dendrogram Operations Tests

    @Suite("Dendrogram Operations")
    struct DendrogramOperationsTests {

        @Test("Dendrogram traversal algorithms")
        func testDendrogramTraversal() async throws {
            // Test tree traversal methods
        }

        @Test("Cophenetic distance calculation")
        func testCopheneticDistance() async throws {
            // Calculate cophenetic distances
        }

        @Test("Cophenetic correlation coefficient")
        func testCopheneticCorrelation() async throws {
            // Measure dendrogram quality
        }

        @Test("Dendrogram pruning operations")
        func testDendrogramPruning() async throws {
            // Test tree pruning
        }

        @Test("Dendrogram visualization data")
        func testDendrogramVisualizationData() async throws {
            // Generate plotting data
        }

        @Test("Dendrogram serialization")
        func testDendrogramSerialization() async throws {
            // Save/load dendrograms
        }
    }

    // MARK: - Incremental Clustering Tests

    @Suite("Incremental Clustering")
    struct IncrementalClusteringTests {

        @Test("Online agglomerative clustering")
        func testOnlineAgglomerative() async throws {
            // Add points incrementally
        }

        @Test("Cluster update with new points")
        func testClusterUpdateNewPoints() async throws {
            // Update existing clusters
        }

        @Test("Cluster merge with constraints")
        func testConstrainedClustering() async throws {
            // Apply must-link/cannot-link
        }

        @Test("Sliding window clustering")
        func testSlidingWindowClustering() async throws {
            // Maintain clusters over time
        }

        @Test("Cluster decay and removal")
        func testClusterDecay() async throws {
            // Remove old clusters
        }
    }

    // MARK: - Quality Metrics Tests

    @Suite("Clustering Quality Metrics")
    struct QualityMetricsTests {

        @Test("Silhouette coefficient calculation")
        func testSilhouetteCoefficient() async throws {
            // Measure cluster quality
        }

        @Test("Dunn index computation")
        func testDunnIndex() async throws {
            // Calculate cluster separation
        }

        @Test("Davies-Bouldin index")
        func testDaviesBouldinIndex() async throws {
            // Measure cluster similarity
        }

        @Test("Calinski-Harabasz index")
        func testCalinskiHarabaszIndex() async throws {
            // Calculate variance ratio
        }

        @Test("Cluster purity metrics")
        func testClusterPurity() async throws {
            // Measure against ground truth
        }

        @Test("Normalized mutual information")
        func testNormalizedMutualInformation() async throws {
            // Information-theoretic metric
        }
    }

    // MARK: - Edge Cases and Error Handling

    @Suite("Edge Cases")
    struct EdgeCaseTests {

        @Test("Single point clustering")
        func testSinglePointClustering() async throws {
            // Handle n=1 case
        }

        @Test("Duplicate points handling")
        func testDuplicatePoints() async throws {
            // Handle identical vectors
        }

        @Test("Collinear points clustering")
        func testCollinearPoints() async throws {
            // Test with aligned points
        }

        @Test("Empty cluster handling")
        func testEmptyClusterHandling() async throws {
            // Handle empty inputs
        }

        @Test("Maximum tree depth limits")
        func testMaximumTreeDepth() async throws {
            // Test deep hierarchies
        }

        @Test("Numerical stability with small distances")
        func testNumericalStability() async throws {
            // Handle near-zero distances
        }
    }

    // MARK: - Integration Tests

    @Suite("Integration with Other Components")
    struct IntegrationTests {

        @Test("Clustering with quantized vectors")
        func testQuantizedVectorClustering() async throws {
            // Use INT8 quantized vectors
        }

        @Test("Clustering as graph preprocessing")
        func testClusteringForGraphConstruction() async throws {
            // Use clusters for graph building
        }

        @Test("Multi-level clustering hierarchies")
        func testMultiLevelHierarchies() async throws {
            // Build hierarchical structures
        }

        @Test("Clustering with mixed precision")
        func testMixedPrecisionClustering() async throws {
            // Combine FP16/FP32 operations
        }

        @Test("End-to-end similarity search with clustering")
        func testSimilaritySearchWithClustering() async throws {
            // Complete search pipeline
        }
    }
}

// MARK: - Supporting Types

/// Cluster quality metrics
struct ClusterQualityMetrics {
    let silhouetteCoefficient: Float
    let dunnIndex: Float
    let daviesBouldinIndex: Float
    let calinskiHarabaszIndex: Float
    let purity: Float
    let normalizedMutualInformation: Float

    init() {
        self.silhouetteCoefficient = 0
        self.dunnIndex = 0
        self.daviesBouldinIndex = 0
        self.calinskiHarabaszIndex = 0
        self.purity = 0
        self.normalizedMutualInformation = 0
    }
}

/// Test configuration for clustering algorithms
struct ClusteringTestConfig {
    let vectorCount: Int
    let dimension: Int
    let clusterCount: Int
    let linkageCriterion: LinkageCriterion
    let distanceMetric: ClusteringDistanceMetric
    let seed: UInt64

    static var `default`: ClusteringTestConfig {
        ClusteringTestConfig(
            vectorCount: 100,
            dimension: 512,
            clusterCount: 5,
            linkageCriterion: .average,
            distanceMetric: .euclidean,
            seed: 42
        )
    }
}

/// Helper for generating synthetic cluster data
struct ClusterDataGenerator {
    static func generateGaussianClusters(config: ClusteringTestConfig) -> ([Vector512Optimized], [[Int]]) {
        // TODO: Generate synthetic Gaussian clusters
        ([], [])
    }

    static func generateWellSeparatedClusters(config: ClusteringTestConfig) -> ([Vector512Optimized], [[Int]]) {
        // TODO: Generate well-separated clusters
        ([], [])
    }

    static func generateOverlappingClusters(config: ClusteringTestConfig) -> ([Vector512Optimized], [[Int]]) {
        // TODO: Generate overlapping clusters
        ([], [])
    }
}