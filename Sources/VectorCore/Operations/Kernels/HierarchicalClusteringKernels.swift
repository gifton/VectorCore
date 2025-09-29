//
//  HierarchicalClusteringKernels.swift
//  VectorCore
//
//  Comprehensive hierarchical clustering implementation with agglomerative and divisive algorithms
//  Optimized for Vector512Optimized with SIMD acceleration
//

import Foundation
import simd

// MARK: - Core Data Structures

/// A node in the hierarchical clustering tree
public struct ClusterNode: Sendable, Hashable {
    public let id: Int
    public let vectorIndices: Set<Int>
    public let centroid: Vector512Optimized
    public let radius: Float
    public let leftChild: Int?
    public let rightChild: Int?
    public let parent: Int?
    /// Height in the tree (leaves = 0)
    public let height: Int
    public let mergeDistance: Float

    public init(
        id: Int, vectorIndices: Set<Int>, centroid: Vector512Optimized, radius: Float,
        leftChild: Int? = nil, rightChild: Int? = nil, parent: Int? = nil,
        height: Int = 0, mergeDistance: Float = 0.0
    ) {
        self.id = id
        self.vectorIndices = vectorIndices
        self.centroid = centroid
        self.radius = radius
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.parent = parent
        self.height = height
        self.mergeDistance = mergeDistance
    }

    public var isLeaf: Bool { leftChild == nil && rightChild == nil }
    public var isRoot: Bool { parent == nil }
    public var size: Int { vectorIndices.count }
}

/// Hierarchical clustering tree with Copy-on-Write semantics
public struct HierarchicalTree: Sendable {

    // Internal mutable storage class for CoW
    private final class Storage: @unchecked Sendable {
        var nodes: ContiguousArray<ClusterNode>
        var rootNodeId: Int
        var leafNodeIds: Set<Int>

        init(nodes: ContiguousArray<ClusterNode>, rootNodeId: Int, leafNodeIds: Set<Int>) {
            self.nodes = nodes
            self.rootNodeId = rootNodeId
            self.leafNodeIds = leafNodeIds
        }

        func copy() -> Storage {
            return Storage(nodes: nodes, rootNodeId: rootNodeId, leafNodeIds: leafNodeIds)
        }
    }

    private var storage: Storage

    public let dimension: Int
    public let linkageCriterion: LinkageCriterion
    public let distanceMetric: ClusteringDistanceMetric

    public init(
        nodes: ContiguousArray<ClusterNode>, rootNodeId: Int, leafNodeIds: Set<Int>,
        dimension: Int, linkageCriterion: LinkageCriterion, distanceMetric: ClusteringDistanceMetric
    ) {
        self.storage = Storage(nodes: nodes, rootNodeId: rootNodeId, leafNodeIds: leafNodeIds)
        self.dimension = dimension
        self.linkageCriterion = linkageCriterion
        self.distanceMetric = distanceMetric
    }

    // Public accessors
    public var nodes: ContiguousArray<ClusterNode> { storage.nodes }
    public var rootNodeId: Int { storage.rootNodeId }
    public var leafNodeIds: Set<Int> { storage.leafNodeIds }

    public func node(withId id: Int) -> ClusterNode? {
        // Assuming IDs correspond to array indices for O(1) access
        guard id >= 0 && id < storage.nodes.count && storage.nodes[id].id == id else { return nil }
        return storage.nodes[id]
    }

    public var rootNode: ClusterNode? { node(withId: rootNodeId) }
    public var nodeCount: Int { storage.nodes.count }

    // MARK: Mutation Helpers (for Incremental Clustering)

    private mutating func ensureUniqueStorage() {
        if !isKnownUniquelyReferenced(&storage) {
            storage = storage.copy()
        }
    }

    internal mutating func updateNode(_ node: ClusterNode) {
        ensureUniqueStorage()
        guard node.id < storage.nodes.count else { return }
        storage.nodes[node.id] = node
        // Update leaf status tracking
        if node.isLeaf {
            storage.leafNodeIds.insert(node.id)
        } else {
            storage.leafNodeIds.remove(node.id)
        }
    }

    internal mutating func addNode(_ node: ClusterNode) {
        ensureUniqueStorage()
        // Assuming sequential ID addition
        assert(node.id == storage.nodes.count)
        storage.nodes.append(node)
        if node.isLeaf {
             storage.leafNodeIds.insert(node.id)
        }
    }

    internal mutating func setRoot(id: Int) {
        ensureUniqueStorage()
        storage.rootNodeId = id
    }
}

/// Symmetric distance matrix for efficient pairwise distance storage
public struct SymmetricDistanceMatrix: Sendable {
    @usableFromInline
    internal var storage: ContiguousArray<Float>
    public let dimension: Int

    public init(dimension: Int) {
        self.dimension = dimension
        // Upper triangle: n(n-1)/2 elements
        let size = dimension > 0 ? (dimension * (dimension - 1)) / 2 : 0
        self.storage = ContiguousArray<Float>(repeating: Float.greatestFiniteMagnitude, count: size)
    }

    @inline(__always)
    @usableFromInline
    internal func index(i: Int, j: Int) -> Int {
        // Ensure i < j
        let (row, col) = i < j ? (i, j) : (j, i)
        // Optimized indexing formula for upper triangle storage
        return row * dimension - row * (row + 1) / 2 + col - row - 1
    }

    @inlinable
    public func distance(i: Int, j: Int) -> Float {
        if i == j { return 0.0 }
        return storage[index(i: i, j: j)]
    }

    @inlinable
    public mutating func setDistance(_ distance: Float, i: Int, j: Int) {
        if i == j { return }
        storage[index(i: i, j: j)] = distance
    }
}

// MARK: - Distance Metrics and Linkage Criteria

public enum ClusteringDistanceMetric: Sendable {
    case euclidean, cosine, dotProduct
}

public enum LinkageCriterion: Sendable, CaseIterable {
    case single, complete, average, ward, centroid, median
}

// MARK: - Distance Computation

/// Helper function to dispatch distance computation
@inline(__always)
@usableFromInline
internal func computeDistance(
    _ vector1: Vector512Optimized,
    _ vector2: Vector512Optimized,
    metric: ClusteringDistanceMetric
) -> Float {
    switch metric {
    case .euclidean:
        return vector1.euclideanDistance(to: vector2)
    case .cosine:
        return 1.0 - vector1.cosineSimilarity(to: vector2)
    case .dotProduct:
        // Negative dot product used as distance (larger product = smaller distance)
        return -vector1.dotProduct(vector2)
    }
}

// MARK: - Linkage Kernels

public enum LinkageKernels {

    /// Single linkage: minimum distance between any two points in different clusters
    @inlinable
    public static func singleLinkage(
        cluster1: ClusterNode, cluster2: ClusterNode, vectors: [Vector512Optimized], distanceMetric: ClusteringDistanceMetric
    ) -> Float {
        var minDistance: Float = .greatestFiniteMagnitude
        for i in cluster1.vectorIndices {
            for j in cluster2.vectorIndices {
                let distance = computeDistance(vectors[i], vectors[j], metric: distanceMetric)
                minDistance = min(minDistance, distance)
            }
        }
        return minDistance
    }

    /// Complete linkage: maximum distance between any two points in different clusters
    @inlinable
    public static func completeLinkage(
        cluster1: ClusterNode, cluster2: ClusterNode, vectors: [Vector512Optimized], distanceMetric: ClusteringDistanceMetric
    ) -> Float {
        var maxDistance: Float = 0.0
        for i in cluster1.vectorIndices {
            for j in cluster2.vectorIndices {
                let distance = computeDistance(vectors[i], vectors[j], metric: distanceMetric)
                maxDistance = max(maxDistance, distance)
            }
        }
        return maxDistance
    }

    /// Average linkage: average distance between all pairs of points in different clusters
    @inlinable
    public static func averageLinkage(
        cluster1: ClusterNode, cluster2: ClusterNode, vectors: [Vector512Optimized], distanceMetric: ClusteringDistanceMetric
    ) -> Float {
        var totalDistance: Double = 0.0 // Use Double for accumulation
        let pairCount = cluster1.size * cluster2.size

        for i in cluster1.vectorIndices {
            for j in cluster2.vectorIndices {
                let distance = computeDistance(vectors[i], vectors[j], metric: distanceMetric)
                totalDistance += Double(distance)
            }
        }
        return pairCount > 0 ? Float(totalDistance / Double(pairCount)) : 0.0
    }

    /// Ward linkage: minimizes within-cluster variance
    @inlinable
    public static func wardLinkage(
        cluster1: ClusterNode, cluster2: ClusterNode
    ) -> Float {
        // Ward's criterion minimizes the increase in within-cluster variance
        let n1 = Float(cluster1.size)
        let n2 = Float(cluster2.size)
        let n = n1 + n2

        // Use squared Euclidean distance for efficiency/stability
        let centroidDistanceSq = EuclideanKernels.squared512(cluster1.centroid, cluster2.centroid)

        // Ward's formula: sqrt( (n1*n2/n) * ||c1 - c2||^2 )
        return sqrt((n1 * n2 / n) * centroidDistanceSq)
    }

    /// Centroid linkage: distance between cluster centroids
    @inlinable
    public static func centroidLinkage(
        cluster1: ClusterNode, cluster2: ClusterNode, distanceMetric: ClusteringDistanceMetric
    ) -> Float {
        return computeDistance(cluster1.centroid, cluster2.centroid, metric: distanceMetric)
    }
}

// MARK: - Core Clustering Algorithms

public enum HierarchicalClusteringKernels {

    // MARK: Utility Functions

    /// Compute centroid of a set of vectors
    @inline(__always)
    internal static func computeCentroid(
        vectorIndices: Set<Int>, vectors: [Vector512Optimized]
    ) -> Vector512Optimized {
        guard !vectorIndices.isEmpty else { return Vector512Optimized() }

        let count = Float(vectorIndices.count)
        var centroidStorage = ContiguousArray<SIMD4<Float>>(repeating: .zero, count: 128)

        // Accumulate
        for index in vectorIndices {
            let vector = vectors[index]
            for i in 0..<128 {
                centroidStorage[i] += vector.storage[i]
            }
        }

        // Average
        for i in 0..<128 {
            centroidStorage[i] /= count
        }

        var result = Vector512Optimized()
        result.storage = centroidStorage
        return result
    }

    /// Compute radius (maximum distance from centroid to any vector in the cluster)
    @inline(__always)
    internal static func computeRadius(
        centroid: Vector512Optimized, vectorIndices: Set<Int>, vectors: [Vector512Optimized]
    ) -> Float {
        var maxDistanceSq: Float = 0.0
        for index in vectorIndices {
            // Radius is typically calculated using Euclidean distance
            let distanceSq = EuclideanKernels.squared512(centroid, vectors[index])
            maxDistanceSq = max(maxDistanceSq, distanceSq)
        }
        return sqrt(maxDistanceSq)
    }

    // MARK: Agglomerative Clustering (Bottom-Up)

    /// Agglomerative hierarchical clustering using the specified linkage criterion
    public static func agglomerativeClustering(
        vectors: [Vector512Optimized],
        linkageCriterion: LinkageCriterion = .ward,
        distanceMetric: ClusteringDistanceMetric = .euclidean
    ) -> HierarchicalTree {
        let n = vectors.count
        guard n > 1 else {
            // Handle 0 or 1 vector case
             if n == 1 {
                 let node = ClusterNode(id: 0, vectorIndices: [0], centroid: vectors[0], radius: 0.0)
                 return HierarchicalTree(nodes: [node], rootNodeId: 0, leafNodeIds: [0], dimension: 512, linkageCriterion: linkageCriterion, distanceMetric: distanceMetric)
            }
            return HierarchicalTree(nodes: [], rootNodeId: -1, leafNodeIds: [], dimension: 512, linkageCriterion: linkageCriterion, distanceMetric: distanceMetric)
        }

        // 1. Initialize leaf nodes
        var nodes = ContiguousArray<ClusterNode>()
        nodes.reserveCapacity(2 * n - 1)

        for i in 0..<n {
            let leafNode = ClusterNode(id: i, vectorIndices: [i], centroid: vectors[i], radius: 0.0)
            nodes.append(leafNode)
        }

        // 2. Agglomerative merging loop
        var activeClusters = Set(0..<n)
        var nextNodeId = n

        while activeClusters.count > 1 {
            // Find closest pair
            let (cluster1Id, cluster2Id, distance) = findClosestClusters(
                activeClusters: activeClusters, nodes: nodes, vectors: vectors,
                linkageCriterion: linkageCriterion, distanceMetric: distanceMetric
            )

            guard cluster1Id != -1 && cluster2Id != -1 else { break }

            // Merge clusters
            let newNode = mergeClusters(
                cluster1Id: cluster1Id, cluster2Id: cluster2Id, mergeDistance: distance,
                newNodeId: nextNodeId, nodes: &nodes, vectors: vectors
            )

            nodes.append(newNode)

            // Update active clusters
            activeClusters.remove(cluster1Id)
            activeClusters.remove(cluster2Id)
            activeClusters.insert(nextNodeId)

            nextNodeId += 1
        }

        let rootNodeId = nextNodeId - 1

        return HierarchicalTree(
            nodes: nodes, rootNodeId: rootNodeId, leafNodeIds: Set(0..<n), dimension: 512,
            linkageCriterion: linkageCriterion, distanceMetric: distanceMetric
        )
    }

    /// Find the closest pair of clusters among active clusters
    private static func findClosestClusters(
        activeClusters: Set<Int>, nodes: ContiguousArray<ClusterNode>, vectors: [Vector512Optimized],
        linkageCriterion: LinkageCriterion, distanceMetric: ClusteringDistanceMetric
    ) -> (Int, Int, Float) {
        var minDistance: Float = .greatestFiniteMagnitude
        var closestPair: (Int, Int) = (-1, -1)

        let clusterArray = Array(activeClusters)

        for i in 0..<clusterArray.count {
            for j in (i + 1)..<clusterArray.count {
                let c1Id = clusterArray[i]
                let c2Id = clusterArray[j]

                let distance = computeLinkageDistance(
                    cluster1: nodes[c1Id], cluster2: nodes[c2Id], vectors: vectors,
                    linkageCriterion: linkageCriterion, distanceMetric: distanceMetric
                )

                if distance < minDistance {
                    minDistance = distance
                    closestPair = (c1Id, c2Id)
                }
            }
        }
        return (closestPair.0, closestPair.1, minDistance)
    }

    /// Dispatch to the appropriate linkage kernel
    private static func computeLinkageDistance(
        cluster1: ClusterNode, cluster2: ClusterNode, vectors: [Vector512Optimized],
        linkageCriterion: LinkageCriterion, distanceMetric: ClusteringDistanceMetric
    ) -> Float {
        switch linkageCriterion {
        case .single:
            return LinkageKernels.singleLinkage(cluster1: cluster1, cluster2: cluster2, vectors: vectors, distanceMetric: distanceMetric)
        case .complete:
            return LinkageKernels.completeLinkage(cluster1: cluster1, cluster2: cluster2, vectors: vectors, distanceMetric: distanceMetric)
        case .average:
            return LinkageKernels.averageLinkage(cluster1: cluster1, cluster2: cluster2, vectors: vectors, distanceMetric: distanceMetric)
        case .ward:
            return LinkageKernels.wardLinkage(cluster1: cluster1, cluster2: cluster2)
        case .centroid, .median:
            // Median is often approximated by Centroid in this context
            return LinkageKernels.centroidLinkage(cluster1: cluster1, cluster2: cluster2, distanceMetric: distanceMetric)
        }
    }

    /// Merge two clusters and update parent pointers
    private static func mergeClusters(
        cluster1Id: Int, cluster2Id: Int, mergeDistance: Float, newNodeId: Int,
        nodes: inout ContiguousArray<ClusterNode>, vectors: [Vector512Optimized]
    ) -> ClusterNode {
        let cluster1 = nodes[cluster1Id]
        let cluster2 = nodes[cluster2Id]

        let mergedIndices = cluster1.vectorIndices.union(cluster2.vectorIndices)
        let newCentroid = computeCentroid(vectorIndices: mergedIndices, vectors: vectors)
        let newRadius = computeRadius(centroid: newCentroid, vectorIndices: mergedIndices, vectors: vectors)

        // Update parent pointers of children
        nodes[cluster1Id] = ClusterNode(
            id: cluster1.id, vectorIndices: cluster1.vectorIndices, centroid: cluster1.centroid, radius: cluster1.radius,
            leftChild: cluster1.leftChild, rightChild: cluster1.rightChild, parent: newNodeId,
            height: cluster1.height, mergeDistance: cluster1.mergeDistance
        )
        nodes[cluster2Id] = ClusterNode(
            id: cluster2.id, vectorIndices: cluster2.vectorIndices, centroid: cluster2.centroid, radius: cluster2.radius,
            leftChild: cluster2.leftChild, rightChild: cluster2.rightChild, parent: newNodeId,
            height: cluster2.height, mergeDistance: cluster2.mergeDistance
        )

        // Create the new parent node
        return ClusterNode(
            id: newNodeId,
            vectorIndices: mergedIndices,
            centroid: newCentroid,
            radius: newRadius,
            leftChild: cluster1Id,
            rightChild: cluster2Id,
            parent: nil,
            height: max(cluster1.height, cluster2.height) + 1,
            mergeDistance: mergeDistance
        )
    }

    // MARK: Divisive Clustering (Top-Down)

    /// K-Means splitting for divisive clustering (k=2)
    @inline(__always)
    internal static func kMeansSplit(
        vectorIndices: Set<Int>, vectors: [Vector512Optimized], distanceMetric: ClusteringDistanceMetric, maxIterations: Int = 10
    ) -> (Set<Int>, Set<Int>) {

        let indices = Array(vectorIndices)
        guard indices.count >= 2 else {
            return (Set(indices.prefix(1)), Set(indices.dropFirst(1)))
        }

        // Initialize centroids (first and last element)
        var centroid1 = vectors[indices[0]]
        var centroid2 = vectors[indices[indices.count - 1]]

        var cluster1: Set<Int> = []
        var cluster2: Set<Int> = []

        for _ in 0..<maxIterations {
            cluster1.removeAll(keepingCapacity: true)
            cluster2.removeAll(keepingCapacity: true)

            // Assignment step
            for index in vectorIndices {
                let vector = vectors[index]
                let dist1 = computeDistance(vector, centroid1, metric: distanceMetric)
                let dist2 = computeDistance(vector, centroid2, metric: distanceMetric)

                if dist1 <= dist2 {
                    cluster1.insert(index)
                } else {
                    cluster2.insert(index)
                }
            }

            // Handle empty clusters
            if cluster1.isEmpty && !cluster2.isEmpty {
                 let furthest = cluster2.max { a, b in computeDistance(vectors[a], centroid2, metric: distanceMetric) < computeDistance(vectors[b], centroid2, metric: distanceMetric) }!
                 cluster2.remove(furthest)
                 cluster1.insert(furthest)
            } else if cluster2.isEmpty && !cluster1.isEmpty {
                let furthest = cluster1.max { a, b in computeDistance(vectors[a], centroid1, metric: distanceMetric) < computeDistance(vectors[b], centroid1, metric: distanceMetric) }!
                cluster1.remove(furthest)
                cluster2.insert(furthest)
            }

            // Update step
            let newCentroid1 = computeCentroid(vectorIndices: cluster1, vectors: vectors)
            let newCentroid2 = computeCentroid(vectorIndices: cluster2, vectors: vectors)

            // Check for convergence
            let change1 = EuclideanKernels.distance512(centroid1, newCentroid1)
            let change2 = EuclideanKernels.distance512(centroid2, newCentroid2)

            centroid1 = newCentroid1
            centroid2 = newCentroid2

            if change1 < 1e-6 && change2 < 1e-6 { break }
        }
        return (cluster1, cluster2)
    }

    /// Divisive hierarchical clustering using top-down approach
    public static func divisiveClustering(
        vectors: [Vector512Optimized],
        maxDepth: Int = 10,
        minClusterSize: Int = 2,
        distanceMetric: ClusteringDistanceMetric = .euclidean
    ) -> HierarchicalTree {
        let n = vectors.count

        var nodesDict: [Int: ClusterNode] = [:]
        var leafNodeIds = Set<Int>()
        var nextNodeId = 0

        if n == 0 {
             return HierarchicalTree(nodes: [], rootNodeId: -1, leafNodeIds: [], dimension: 512, linkageCriterion: .centroid, distanceMetric: distanceMetric)
        }

        // Initialize root
        let rootIndices = Set(0..<n)
        let rootCentroid = computeCentroid(vectorIndices: rootIndices, vectors: vectors)
        let rootRadius = computeRadius(centroid: rootCentroid, vectorIndices: rootIndices, vectors: vectors)
        let rootCluster = ClusterNode(id: nextNodeId, vectorIndices: rootIndices, centroid: rootCentroid, radius: rootRadius)
        nextNodeId += 1

        // Queue for splitting: (Node, Depth)
        var splittingQueue: [(ClusterNode, Int)] = [(rootCluster, 0)]

        while !splittingQueue.isEmpty {
            let (currentCluster, depth) = splittingQueue.removeFirst()

            // Check termination conditions
            if depth >= maxDepth || currentCluster.size <= minClusterSize {
                // Finalize as leaf
                let leafNode = ClusterNode(
                    id: currentCluster.id, vectorIndices: currentCluster.vectorIndices, centroid: currentCluster.centroid, radius: currentCluster.radius,
                    parent: currentCluster.parent, height: 0
                )
                nodesDict[currentCluster.id] = leafNode
                leafNodeIds.insert(currentCluster.id)
            } else {
                // Split cluster
                let (leftIndices, rightIndices) = kMeansSplit(vectorIndices: currentCluster.vectorIndices, vectors: vectors, distanceMetric: distanceMetric)

                if leftIndices.isEmpty || rightIndices.isEmpty {
                    // Treat as leaf if split fails
                    let leafNode = ClusterNode(id: currentCluster.id, vectorIndices: currentCluster.vectorIndices, centroid: currentCluster.centroid, radius: currentCluster.radius, parent: currentCluster.parent, height: 0)
                    nodesDict[currentCluster.id] = leafNode
                    leafNodeIds.insert(currentCluster.id)
                    continue
                }

                // Create child nodes
                let leftId = nextNodeId
                let rightId = nextNodeId + 1
                nextNodeId += 2

                let leftCentroid = computeCentroid(vectorIndices: leftIndices, vectors: vectors)
                let rightCentroid = computeCentroid(vectorIndices: rightIndices, vectors: vectors)

                let leftRadius = computeRadius(centroid: leftCentroid, vectorIndices: leftIndices, vectors: vectors)
                let rightRadius = computeRadius(centroid: rightCentroid, vectorIndices: rightIndices, vectors: vectors)

                let leftCluster = ClusterNode(id: leftId, vectorIndices: leftIndices, centroid: leftCentroid, radius: leftRadius, parent: currentCluster.id)
                let rightCluster = ClusterNode(id: rightId, vectorIndices: rightIndices, centroid: rightCentroid, radius: rightRadius, parent: currentCluster.id)

                // Initially add child nodes to the dictionary (they may be updated later when processed)
                nodesDict[leftId] = leftCluster
                nodesDict[rightId] = rightCluster

                // Update current cluster to be internal node
                let mergeDistance = computeDistance(leftCentroid, rightCentroid, metric: distanceMetric)
                let internalNode = ClusterNode(
                    id: currentCluster.id, vectorIndices: currentCluster.vectorIndices, centroid: currentCluster.centroid, radius: currentCluster.radius,
                    leftChild: leftId, rightChild: rightId, parent: currentCluster.parent, height: 0, mergeDistance: mergeDistance
                )
                nodesDict[internalNode.id] = internalNode

                // Add children to queue for further processing
                splittingQueue.append((leftCluster, depth + 1))
                splittingQueue.append((rightCluster, depth + 1))
            }
        }

        // Convert dictionary to sorted array and calculate heights
        var finalNodes = nodesDict.values.sorted { $0.id < $1.id }

        // Calculate heights recursively
        @discardableResult
        func calculateHeight(nodeId: Int, nodes: inout [ClusterNode]) -> Int {
             guard nodeId < nodes.count else { return 0 }
             let node = nodes[nodeId]

             if node.isLeaf { return 0 }

             let leftHeight = node.leftChild.map { calculateHeight(nodeId: $0, nodes: &nodes) } ?? 0
             let rightHeight = node.rightChild.map { calculateHeight(nodeId: $0, nodes: &nodes) } ?? 0
             let height = max(leftHeight, rightHeight) + 1

             nodes[nodeId] = ClusterNode(
                 id: node.id, vectorIndices: node.vectorIndices, centroid: node.centroid, radius: node.radius,
                 leftChild: node.leftChild, rightChild: node.rightChild, parent: node.parent,
                 height: height, mergeDistance: node.mergeDistance
             )
             return height
        }

        _ = calculateHeight(nodeId: 0, nodes: &finalNodes)

        return HierarchicalTree(
            nodes: ContiguousArray(finalNodes), rootNodeId: 0, leafNodeIds: leafNodeIds, dimension: 512,
            linkageCriterion: .centroid, distanceMetric: distanceMetric
        )
    }
}

// MARK: - Tree Navigation

extension HierarchicalTree {
    /// Get children of a node
    public func children(of nodeId: Int) -> [ClusterNode] {
        guard let parentNode = self.node(withId: nodeId) else { return [] }
        var children: [ClusterNode] = []
        if let leftId = parentNode.leftChild, let leftNode = self.node(withId: leftId) {
            children.append(leftNode)
        }
        if let rightId = parentNode.rightChild, let rightNode = self.node(withId: rightId) {
            children.append(rightNode)
        }
        return children
    }

    /// Get parent of a node
    public func parent(of nodeId: Int) -> ClusterNode? {
        guard let childNode = self.node(withId: nodeId), let parentId = childNode.parent else { return nil }
        return self.node(withId: parentId)
    }

    /// Get all ancestors of a node
    public func ancestors(of nodeId: Int) -> [ClusterNode] {
        var ancestors: [ClusterNode] = []
        var currentId: Int? = nodeId

        while let id = currentId, let currentNode = self.node(withId: id), let parentId = currentNode.parent {
            if let parentNode = self.node(withId: parentId) {
                ancestors.append(parentNode)
                currentId = parentId
            } else {
                break
            }
        }

        return ancestors
    }

    /// Get all leaf descendants of a node
    public func leafDescendants(of nodeId: Int) -> [ClusterNode] {
        guard let node = node(withId: nodeId) else { return [] }

        if node.isLeaf {
            return [node]
        }

        var leaves: [ClusterNode] = []
        var stack = [nodeId]

        while let currentId = stack.popLast() {
            guard let currentNode = self.node(withId: currentId) else { continue }

            if currentNode.isLeaf {
                leaves.append(currentNode)
            } else {
                if let leftId = currentNode.leftChild {
                    stack.append(leftId)
                }
                if let rightId = currentNode.rightChild {
                    stack.append(rightId)
                }
            }
        }

        return leaves
    }
}

// MARK: - Priority Queue for Search Algorithms

/// Generic priority queue implementation
public struct PriorityQueue<Element> {
    private var elements: [Element]
    private let capacity: Int?
    private let hasHigherPriority: (Element, Element) -> Bool

    public init(capacity: Int? = nil, sort: @escaping (Element, Element) -> Bool) {
        self.capacity = capacity
        self.elements = []
        self.hasHigherPriority = sort
        if let cap = capacity {
            self.elements.reserveCapacity(cap)
        }
    }

    public var count: Int { elements.count }
    public var isEmpty: Bool { elements.isEmpty }

    public func peek() -> Element? {
        return elements.first
    }

    public mutating func insert(_ element: Element) {
        if let cap = capacity, elements.count >= cap {
            if let top = elements.first, !hasHigherPriority(element, top) {
                elements[0] = element
                siftDown(0)
            }
        } else {
            elements.append(element)
            siftUp(elements.count - 1)
        }
    }

    public mutating func extractTop() -> Element? {
        guard !elements.isEmpty else { return nil }
        if elements.count == 1 { return elements.removeLast() }
        let top = elements[0]
        elements[0] = elements.removeLast()
        siftDown(0)
        return top
    }

    public mutating func extractSorted() -> [Element] {
        var sorted: [Element] = []
        while !isEmpty {
            sorted.append(extractTop()!)
        }
        return sorted.reversed()
    }

    private mutating func siftUp(_ index: Int) {
        var child = index
        let childElement = elements[child]
        var parent = (child - 1) / 2

        while child > 0 && hasHigherPriority(childElement, elements[parent]) {
            elements[child] = elements[parent]
            child = parent
            parent = (child - 1) / 2
        }
        elements[child] = childElement
    }

    private mutating func siftDown(_ index: Int) {
        var parent = index
        let count = elements.count
        let element = elements[parent]

        while true {
            let left = 2 * parent + 1
            if left >= count { break }
            var candidate = left
            let right = left + 1

            if right < count && hasHigherPriority(elements[right], elements[left]) {
                candidate = right
            }

            if !hasHigherPriority(elements[candidate], element) { break }

            elements[parent] = elements[candidate]
            parent = candidate
        }
        elements[parent] = element
    }
}

// MARK: - Search Result Types

public struct SearchCandidate: Comparable, Sendable {
    public let vectorIndex: Int
    public let distance: Float
    public let nodeId: Int

    public static func < (lhs: SearchCandidate, rhs: SearchCandidate) -> Bool {
        return lhs.distance < rhs.distance
    }

    public static func > (lhs: SearchCandidate, rhs: SearchCandidate) -> Bool {
        return lhs.distance > rhs.distance
    }
}

public struct HierarchicalSearchResult: Sendable {
    public let vectorIndex: Int
    public let distance: Float
    public let confidence: Float
}

public struct VectorInRangeResult: Sendable {
    public let vectorIndex: Int
    public let distance: Float
    public let nodeId: Int
}

// MARK: - Tree-Based Search Algorithms

extension HierarchicalTree {

    /// Find k nearest neighbors using best-first search
    public func nearestNeighbors(
        to query: Vector512Optimized,
        k: Int,
        vectors: [Vector512Optimized],
        searchRadius: Float? = nil
    ) -> [HierarchicalSearchResult] {
        guard k > 0, let root = rootNode else { return [] }

        // Max-Heap for Top-K results
        var topKHeap = PriorityQueue<SearchCandidate>(capacity: k, sort: >)

        // Min-Heap for search frontier
        var frontier = PriorityQueue<(Float, Int)>(sort: { $0.0 < $1.0 })

        // Initialize frontier
        let rootDist = computeDistance(query, root.centroid, metric: distanceMetric)
        let rootBound = max(0, rootDist - root.radius)
        frontier.insert((rootBound, root.id))

        var worstDistInTopK: Float = searchRadius ?? .greatestFiniteMagnitude

        // Best-First Search Traversal
        while let (bound, nodeId) = frontier.extractTop() {

            // Pruning: If the optimistic bound is worse than the K-th best distance, stop
            if bound >= worstDistInTopK {
                break
            }

            guard let node = node(withId: nodeId) else { continue }

            if node.isLeaf {
                // Process leaf vectors
                for vectorIndex in node.vectorIndices {
                    let distance = computeDistance(query, vectors[vectorIndex], metric: distanceMetric)

                    if distance < worstDistInTopK {
                         let candidate = SearchCandidate(vectorIndex: vectorIndex, distance: distance, nodeId: node.id)
                         topKHeap.insert(candidate)

                         // Update the threshold if heap is full
                         if topKHeap.count == k, let top = topKHeap.peek() {
                             worstDistInTopK = top.distance
                         }
                    }
                }
            } else {
                // Explore children
                let children = self.children(of: nodeId)
                for child in children {
                    let childDist = computeDistance(query, child.centroid, metric: distanceMetric)
                    let childBound = max(0, childDist - child.radius)

                    // Add to frontier if promising
                    if childBound < worstDistInTopK {
                         frontier.insert((childBound, child.id))
                    }
                }
            }
        }

        // Extract results
        let sortedCandidates = topKHeap.extractSorted()

        // Format results (smallest distance first)
        let results = sortedCandidates.reversed().enumerated().map { index, candidate in
             let confidence: Float = 1.0
             return HierarchicalSearchResult(
                vectorIndex: candidate.vectorIndex,
                distance: candidate.distance,
                confidence: confidence
            )
        }

        return results
    }

    /// Find all vectors within a given radius using range search
    public func vectorsInRange(
        of query: Vector512Optimized,
        radius: Float,
        vectors: [Vector512Optimized],
        maxResults: Int? = nil
    ) -> [VectorInRangeResult] {
        var results: [VectorInRangeResult] = []
        guard let root = rootNode else { return [] }

        // Depth-First Traversal
        var stack: [Int] = [root.id]

        while let nodeId = stack.popLast() {
            guard let node = node(withId: nodeId) else { continue }

            // Check limit
            if let max = maxResults, results.count >= max { break }

            let centroidDistance = computeDistance(query, node.centroid, metric: distanceMetric)

            // Pruning: If node bounding sphere does not intersect the search radius
            if centroidDistance > radius + node.radius {
                continue
            }

            if node.isLeaf {
                // Check vectors in leaf
                for vectorIndex in node.vectorIndices {
                    let distance = computeDistance(query, vectors[vectorIndex], metric: distanceMetric)
                    if distance <= radius {
                        results.append(VectorInRangeResult(vectorIndex: vectorIndex, distance: distance, nodeId: node.id))
                        if let max = maxResults, results.count >= max { break }
                    }
                }
            } else {
                // Add children to stack
                if let left = node.leftChild { stack.append(left) }
                if let right = node.rightChild { stack.append(right) }
            }
        }

        return results.sorted { $0.distance < $1.distance }
    }
}

// MARK: - Incremental Clustering

/// Update type for incremental clustering operations
public struct ClusteringUpdate {
    public enum UpdateType { case insertion, restructure, error }
    public let type: UpdateType
    public let affectedNodes: Set<Int>
}

/// Manages online updates to hierarchical clustering trees
public struct IncrementalHierarchicalClustering {
    private var tree: HierarchicalTree
    private var vectors: [Vector512Optimized]
    private var nextNodeId: Int

    public init(initialTree: HierarchicalTree, vectors: [Vector512Optimized]) {
        self.tree = initialTree
        self.vectors = vectors
        self.nextNodeId = initialTree.nodeCount
    }

    public func getTree() -> HierarchicalTree { return tree }

    /// Add a new vector to the clustering tree
    public mutating func addVector(
        _ vector: Vector512Optimized,
        updateThreshold: Float = 0.1
    ) -> ClusteringUpdate {
        let vectorIndex = vectors.count
        vectors.append(vector)

        if tree.rootNodeId == -1 {
            // Initialize tree if empty
            let newNode = ClusterNode(id: 0, vectorIndices: [vectorIndex], centroid: vector, radius: 0.0)
            tree.addNode(newNode)
            tree.setRoot(id: 0)
            nextNodeId = 1
            return ClusteringUpdate(type: .insertion, affectedNodes: [0])
        }

        // Find best insertion point
        let insertionPoint = findBestInsertionPoint(for: vector)

        // Strategy: If close enough to the centroid, update statistics. Otherwise, restructure.
        if insertionPoint.distance <= updateThreshold {
            return updateStatistics(vectorIndex: vectorIndex, startNodeId: insertionPoint.nodeId)
        } else {
            return restructureTree(vectorIndex: vectorIndex, targetNodeId: insertionPoint.nodeId)
        }
    }

    private struct InsertionPoint {
        let nodeId: Int
        let distance: Float
    }

    /// Find the closest node using greedy traversal
    private func findBestInsertionPoint(for vector: Vector512Optimized) -> InsertionPoint {
        var currentNode = tree.rootNode!

        // Greedy traversal towards the closest centroid
        while !currentNode.isLeaf {
            let children = tree.children(of: currentNode.id)

            let bestChild = children.min { child1, child2 in
                let dist1 = computeDistance(vector, child1.centroid, metric: tree.distanceMetric)
                let dist2 = computeDistance(vector, child2.centroid, metric: tree.distanceMetric)
                return dist1 < dist2
            }

            if let bestChild = bestChild {
                currentNode = bestChild
            } else {
                break
            }
        }

        let distance = computeDistance(vector, currentNode.centroid, metric: tree.distanceMetric)
        return InsertionPoint(nodeId: currentNode.id, distance: distance)
    }

    /// Update statistics up the hierarchy
    private mutating func updateStatistics(vectorIndex: Int, startNodeId: Int) -> ClusteringUpdate {
        var currentNodeId: Int? = startNodeId
        var affectedNodes = Set<Int>()

        while let nodeId = currentNodeId {
            guard let node = tree.node(withId: nodeId) else { break }
            affectedNodes.insert(nodeId)

            // Update statistics
            var newIndices = node.vectorIndices
            newIndices.insert(vectorIndex)

            let newCentroid = HierarchicalClusteringKernels.computeCentroid(vectorIndices: newIndices, vectors: vectors)
            let newRadius = HierarchicalClusteringKernels.computeRadius(centroid: newCentroid, vectorIndices: newIndices, vectors: vectors)

            let updatedNode = ClusterNode(
                id: node.id, vectorIndices: newIndices, centroid: newCentroid, radius: newRadius,
                leftChild: node.leftChild, rightChild: node.rightChild, parent: node.parent,
                height: node.height, mergeDistance: node.mergeDistance
            )

            tree.updateNode(updatedNode)

            // Move up to the parent
            currentNodeId = node.parent
        }

        return ClusteringUpdate(type: .insertion, affectedNodes: affectedNodes)
    }

    /// Restructure the tree by merging the new vector with the target node
    private mutating func restructureTree(vectorIndex: Int, targetNodeId: Int) -> ClusteringUpdate {
        guard let targetNode = tree.node(withId: targetNodeId) else {
             return ClusteringUpdate(type: .error, affectedNodes: [])
        }

        // 1. Create a new leaf node for the incoming vector
        let newVectorNodeId = nextNodeId
        let newVectorNode = ClusterNode(id: newVectorNodeId, vectorIndices: [vectorIndex], centroid: vectors[vectorIndex], radius: 0.0)
        nextNodeId += 1

        // 2. Create a new parent node merging the target and the new leaf
        let newParentId = nextNodeId
        nextNodeId += 1

        let mergedIndices = targetNode.vectorIndices.union([vectorIndex])
        let newParentCentroid = HierarchicalClusteringKernels.computeCentroid(vectorIndices: mergedIndices, vectors: vectors)
        let newParentRadius = HierarchicalClusteringKernels.computeRadius(centroid: newParentCentroid, vectorIndices: mergedIndices, vectors: vectors)
        let mergeDistance = computeDistance(targetNode.centroid, newVectorNode.centroid, metric: tree.distanceMetric)

        let newParentNode = ClusterNode(
            id: newParentId, vectorIndices: mergedIndices, centroid: newParentCentroid, radius: newParentRadius,
            leftChild: targetNodeId, rightChild: newVectorNodeId, parent: targetNode.parent,
            height: max(targetNode.height, 0) + 1, mergeDistance: mergeDistance
        )

        // 3. Update parent pointers of the children
        let updatedTargetNode = ClusterNode(
            id: targetNode.id, vectorIndices: targetNode.vectorIndices, centroid: targetNode.centroid, radius: targetNode.radius,
            leftChild: targetNode.leftChild, rightChild: targetNode.rightChild, parent: newParentId,
            height: targetNode.height, mergeDistance: targetNode.mergeDistance
        )

        let updatedNewVectorNode = ClusterNode(
             id: newVectorNode.id, vectorIndices: newVectorNode.vectorIndices, centroid: newVectorNode.centroid, radius: newVectorNode.radius,
             parent: newParentId
        )

        // 4. Update the tree structure
        tree.updateNode(updatedTargetNode)
        tree.addNode(updatedNewVectorNode)
        tree.addNode(newParentNode)

        // 5. Update the original parent of the target node to point to the new parent
        if let grandParentId = targetNode.parent, let grandParent = tree.node(withId: grandParentId) {
            let updatedGrandParent = ClusterNode(
                id: grandParent.id, vectorIndices: grandParent.vectorIndices, centroid: grandParent.centroid, radius: grandParent.radius,
                leftChild: grandParent.leftChild == targetNodeId ? newParentId : grandParent.leftChild,
                rightChild: grandParent.rightChild == targetNodeId ? newParentId : grandParent.rightChild,
                parent: grandParent.parent, height: grandParent.height, mergeDistance: grandParent.mergeDistance
            )
            tree.updateNode(updatedGrandParent)
        } else {
            // If the target node was the root, the new parent becomes the new root
            tree.setRoot(id: newParentId)
        }

        // 6. Propagate statistics update upwards from the new parent
        if let parentId = newParentNode.parent {
            _ = updateStatistics(vectorIndex: vectorIndex, startNodeId: parentId)
        }

        return ClusteringUpdate(type: .restructure, affectedNodes: [targetNodeId, newVectorNodeId, newParentId])
    }
}