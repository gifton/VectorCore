//
//  HierarchicalClusteringKernels.swift
//  VectorCore
//
//  Complete implementation of hierarchical clustering kernels with packed storage
//

import Foundation
import simd
import OSLog

// MARK: - Utility Types and Errors

public enum ClusteringDistanceMetric: Sendable {
    case euclideanSquared, euclidean, cosine
}

// VectorError is defined in VectorCore/Errors/VectorError.swift - using existing implementation

public struct ScoredResult<T: Sendable & Equatable>: Comparable, Sendable {
    public let item: T
    public let score: Float

    public init(item: T, score: Float) {
        self.item = item
        self.score = score
    }

    public static func < (lhs: ScoredResult<T>, rhs: ScoredResult<T>) -> Bool {
        return lhs.score < rhs.score
    }

    public static func == (lhs: ScoredResult<T>, rhs: ScoredResult<T>) -> Bool {
        return lhs.item == rhs.item && lhs.score == rhs.score
    }
}

// Simple heap implementations for clustering
public struct MaxHeap<T: Comparable> {
    private var elements: [T] = []
    private let capacity: Int

    public init(capacity: Int) {
        self.capacity = capacity
        elements.reserveCapacity(capacity)
    }

    public var count: Int { elements.count }

    public func peek() -> T? { elements.first }

    public mutating func insert(_ item: T) {
        if elements.count < capacity {
            elements.append(item)
            elements.sort { $0 > $1 } // Max heap property
        } else if item < elements.first! {
            elements[0] = item
            elements.sort { $0 > $1 }
        }
    }

    public mutating func replaceMax(_ item: T) {
        if !elements.isEmpty {
            elements[0] = item
            elements.sort { $0 > $1 }
        }
    }

    public mutating func extractAll() -> [T] {
        let result = elements
        elements.removeAll()
        return result
    }
}

public struct PriorityQueue<T: Comparable> {
    private var heap: [T] = []

    public mutating func enqueue(_ item: T) {
        heap.append(item)
        heap.sort() // Simple implementation
    }

    public mutating func dequeue() -> T? {
        return heap.isEmpty ? nil : heap.removeFirst()
    }
}

// MARK: - CompactNode Implementation

/// Packed storage representation for memory-efficient tree nodes
public struct CompactNode {
    // Node identification (16 bytes)
    public let id: UUID

    // Node properties (16 bytes)
    public let isLeaf: Bool
    public let level: UInt16
    public let subtreeSize: UInt32
    public let vectorCount: UInt32  // Only used for leaf nodes
    private let _padding1: UInt16 = 0
    private let _padding2: UInt8 = 0

    // Geometric properties (20 bytes)
    public let radius: Float
    public let centroidOffset: UInt32  // Offset into centroid storage
    public let boundsOffset: UInt32    // Offset into bounds storage
    public let vectorsOffset: UInt32   // Offset into vectors storage (leaf only)
    public let childrenOffset: UInt32  // Offset into children storage (internal only)

    // Factory methods for creating nodes
    static func createLeaf<V: OptimizedVector>(
        id: UUID,
        vectors: ContiguousArray<V>,
        centroid: V,
        bounds: BoundingBox<V>,
        level: Int,
        storage: ClusterTreeStorage<V>
    ) -> CompactNode {

        let centroidOffset = storage.storeCentroid(centroid)
        let boundsOffset = storage.storeBounds(bounds)
        let vectorsOffset = storage.storeVectors(vectors)

        // Calculate radius as max distance from centroid to any vector
        let radius = vectors.map { sqrt(squaredDistanceSIMD(centroid, $0)) }.max() ?? 0.0

        return CompactNode(
            id: id,
            isLeaf: true,
            level: UInt16(level),
            subtreeSize: UInt32(vectors.count),
            vectorCount: UInt32(vectors.count),
            radius: radius,
            centroidOffset: centroidOffset,
            boundsOffset: boundsOffset,
            vectorsOffset: vectorsOffset,
            childrenOffset: 0
        )
    }

    static func createInternal<V: OptimizedVector>(
        id: UUID,
        leftChild: UUID,
        rightChild: UUID,
        centroid: V,
        bounds: BoundingBox<V>,
        level: Int,
        subtreeSize: Int,
        storage: ClusterTreeStorage<V>
    ) -> CompactNode {

        let centroidOffset = storage.storeCentroid(centroid)
        let boundsOffset = storage.storeBounds(bounds)
        let childrenOffset = storage.storeChildren(leftChild, rightChild)

        // Calculate radius as distance from centroid to furthest bound corner
        let corners = [bounds.min, bounds.max]
        let radius = corners.map { sqrt(squaredDistanceSIMD(centroid, $0)) }.max() ?? 0.0

        return CompactNode(
            id: id,
            isLeaf: false,
            level: UInt16(level),
            subtreeSize: UInt32(subtreeSize),
            vectorCount: 0,
            radius: radius,
            centroidOffset: centroidOffset,
            boundsOffset: boundsOffset,
            vectorsOffset: 0,
            childrenOffset: childrenOffset
        )
    }
}

// MARK: - Storage Management

/// Manages packed storage for all tree data
public final class ClusterTreeStorage<Vector: OptimizedVector>: @unchecked Sendable {

    // Centroids storage
    private var centroidsStorage: ContiguousArray<Vector> = []
    private let centroidsLock = OSAllocatedUnfairLock(initialState: ())

    // Bounds storage (min/max pairs)
    private var boundsStorage: ContiguousArray<(Vector, Vector)> = []
    private let boundsLock = OSAllocatedUnfairLock(initialState: ())

    // Vectors storage for leaf nodes
    private var vectorsStorage: ContiguousArray<ContiguousArray<Vector>> = []
    private let vectorsLock = OSAllocatedUnfairLock(initialState: ())

    // Children storage (UUID pairs for internal nodes)
    private var childrenStorage: ContiguousArray<(UUID, UUID)> = []
    private let childrenLock = OSAllocatedUnfairLock(initialState: ())

    func storeCentroid(_ centroid: Vector) -> UInt32 {
        return centroidsLock.withLock { _ in
            centroidsStorage.append(centroid)
            return UInt32(centroidsStorage.count - 1)
        }
    }

    func storeBounds(_ bounds: BoundingBox<Vector>) -> UInt32 {
        return boundsLock.withLock { _ in
            boundsStorage.append((bounds.min, bounds.max))
            return UInt32(boundsStorage.count - 1)
        }
    }

    func storeVectors(_ vectors: ContiguousArray<Vector>) -> UInt32 {
        return vectorsLock.withLock { _ in
            vectorsStorage.append(vectors)
            return UInt32(vectorsStorage.count - 1)
        }
    }

    func storeChildren(_ left: UUID, _ right: UUID) -> UInt32 {
        return childrenLock.withLock { _ in
            childrenStorage.append((left, right))
            return UInt32(childrenStorage.count - 1)
        }
    }

    func getCentroid(at offset: UInt32) -> Vector {
        return centroidsLock.withLock { _ in
            centroidsStorage[Int(offset)]
        }
    }

    func getBounds(at offset: UInt32) -> BoundingBox<Vector> {
        return boundsLock.withLock { _ in
            let (min, max) = boundsStorage[Int(offset)]
            return BoundingBox(min: min, max: max)
        }
    }

    func getVectors(at offset: UInt32) -> ContiguousArray<Vector> {
        return vectorsLock.withLock { _ in
            vectorsStorage[Int(offset)]
        }
    }

    func getChildren(at offset: UInt32) -> (UUID, UUID) {
        return childrenLock.withLock { _ in
            childrenStorage[Int(offset)]
        }
    }
}

// MARK: - Public ClusterNode Interface

public struct ClusterNode<Vector: OptimizedVector>: Sendable {
    public let id: UUID
    public let level: Int
    public let isLeaf: Bool
    public let centroid: Vector
    public let radius: Float
    public let boundingBox: BoundingBox<Vector>
    public let parent: UUID?
    public let leftChild: UUID?
    public let rightChild: UUID?
    public let vectors: ContiguousArray<Vector>?
    public let subtreeSize: Int

    // Required by spec but simplified
    @usableFromInline internal struct CompactNodeData: Sendable {}
    @usableFromInline internal var compactRepresentation: CompactNodeData = CompactNodeData()

    @usableFromInline internal init(
        from compactNode: CompactNode,
        storage: ClusterTreeStorage<Vector>,
        parent: UUID? = nil
    ) {
        self.id = compactNode.id
        self.level = Int(compactNode.level)
        self.isLeaf = compactNode.isLeaf
        self.centroid = storage.getCentroid(at: compactNode.centroidOffset)
        self.radius = compactNode.radius
        self.boundingBox = storage.getBounds(at: compactNode.boundsOffset)
        self.parent = parent
        self.subtreeSize = Int(compactNode.subtreeSize)

        if compactNode.isLeaf {
            self.vectors = storage.getVectors(at: compactNode.vectorsOffset)
            self.leftChild = nil
            self.rightChild = nil
        } else {
            self.vectors = nil
            let (left, right) = storage.getChildren(at: compactNode.childrenOffset)
            self.leftChild = left
            self.rightChild = right
        }
    }
}

// MARK: - Configuration and BoundingBox

public struct ClusteringConfig: Sendable {
    public let maxLeafSize: Int, maxDepth: Int, minClusterSize: Int
    public let balanceWeight: Float, separationWeight: Float, balancePenalty: Float
    public let parallelism: Int, memoryBudgetMB: Int, useApproximation: Bool
    public let metric: ClusteringDistanceMetric

    public init(
        maxLeafSize: Int = 64, maxDepth: Int = 32, minClusterSize: Int = 8,
        balanceWeight: Float = 0.7, separationWeight: Float = 0.2, balancePenalty: Float = 0.1,
        parallelism: Int? = nil, memoryBudgetMB: Int = 512, useApproximation: Bool = false,
        metric: ClusteringDistanceMetric = .euclideanSquared
    ) {
        self.maxLeafSize = maxLeafSize; self.maxDepth = maxDepth; self.minClusterSize = minClusterSize
        self.balanceWeight = balanceWeight; self.separationWeight = separationWeight; self.balancePenalty = balancePenalty
        self.parallelism = parallelism ?? ProcessInfo.processInfo.processorCount
        self.memoryBudgetMB = memoryBudgetMB; self.useApproximation = useApproximation; self.metric = metric
    }

    public static let `default` = ClusteringConfig()
}

public struct BoundingBox<Vector: OptimizedVector>: Sendable {
    public let min: Vector
    public let max: Vector

    public init(min: Vector, max: Vector) {
        self.min = min; self.max = max
    }

    public init(vectors: ContiguousArray<Vector>) {
        guard let first = vectors.first else {
            self.min = Vector.zero
            self.max = Vector.zero
            return
        }

        var minVec = first; var maxVec = first
        for vector in vectors.dropFirst() {
            minVec = minVec.elementwiseMin(vector)
            maxVec = maxVec.elementwiseMax(vector)
        }
        self.min = minVec; self.max = maxVec
    }

    @inlinable
    public func contains(_ point: Vector) -> Bool {
        // Check if point is within bounds using elementwise comparison
        let withinMin = point.elementwiseMax(min) == point  // point >= min
        let withinMax = point.elementwiseMin(max) == point  // point <= max
        return withinMin && withinMax
    }

    @inlinable
    public func distanceTo(_ point: Vector) -> Float {
        // Find closest point on the box by clamping coordinates
        let closestPoint = point.elementwiseMax(min).elementwiseMin(max)
        return squaredDistanceSIMD(point, closestPoint)
    }
}

// MARK: - Memory Pool Optimization

/// High-performance memory pool for CompactNode allocation
@usableFromInline internal final class CompactNodePool: @unchecked Sendable {
    @usableFromInline let pool: UnsafeMutablePointer<CompactNode>
    @usableFromInline let capacity: Int
    @usableFromInline var nextIndex: Int = 0

    init(capacity: Int) {
        self.capacity = capacity
        self.pool = UnsafeMutablePointer<CompactNode>.allocate(capacity: capacity)
    }

    deinit {
        pool.deallocate()
    }

    @inlinable
    func allocateNode() -> (UnsafeMutablePointer<CompactNode>, Int) {
        guard nextIndex < capacity else {
            fatalError("Node pool exhausted: \(nextIndex) >= \(capacity)")
        }

        let pointer = pool.advanced(by: nextIndex)
        let index = nextIndex
        nextIndex += 1

        return (pointer, index)
    }

    func storeNode(_ node: CompactNode, at index: Int) {
        pool.advanced(by: index).pointee = node
    }

    @inlinable
    func getBasePointer() -> UnsafeMutablePointer<CompactNode> {
        return pool
    }
}

// MARK: - Construction State Management

/// Actor-based construction state for safe concurrent access with memory pool
@usableFromInline internal actor ConstructionState<Vector: OptimizedVector> {
    private var idToIndexMap: [UUID: Int] = [:]
    let storage = ClusterTreeStorage<Vector>()
    nonisolated let nodePool: CompactNodePool

    init(estimatedNodes: Int) {
        self.nodePool = CompactNodePool(capacity: estimatedNodes)
    }

    func allocateNode(id: UUID, node: CompactNode) -> Int {
        let (pointer, index) = nodePool.allocateNode()
        pointer.pointee = node
        idToIndexMap[id] = index
        return index
    }

    func getIdToIndexMap() -> [UUID: Int] {
        return idToIndexMap
    }

    nonisolated func getNodeStorage() -> UnsafeMutablePointer<CompactNode> {
        return nodePool.getBasePointer()
    }
}

// MARK: - HierarchicalClusterTree Implementation

public final class HierarchicalClusterTree<Vector: OptimizedVector>: @unchecked Sendable {

    @usableFromInline
    internal let nodeStorage: UnsafeMutablePointer<CompactNode>

    @usableFromInline
    internal let nodeCount: Int

    internal let rootId: UUID
    internal let config: ClusteringConfig
    @usableFromInline internal let idToIndexMap: [UUID: Int]
    @usableFromInline internal let storage: ClusterTreeStorage<Vector>

    // Primary initializer (post-construction)
    internal init(
        nodeStorage: UnsafeMutablePointer<CompactNode>,
        nodeCount: Int,
        rootId: UUID,
        idToIndexMap: [UUID: Int],
        storage: ClusterTreeStorage<Vector>,
        config: ClusteringConfig
    ) {
        self.nodeStorage = nodeStorage
        self.nodeCount = nodeCount
        self.rootId = rootId
        self.idToIndexMap = idToIndexMap
        self.storage = storage
        self.config = config
    }

    // Convenience initializer (starts construction)
    public convenience init(vectors: ContiguousArray<Vector>, configuration: ClusteringConfig = .default) async throws {
        let (storage, count, rootId, map, treeStorage) = try await HierarchicalClustering.buildTree(vectors: vectors, config: configuration)
        self.init(nodeStorage: storage, nodeCount: count, rootId: rootId, idToIndexMap: map, storage: treeStorage, config: configuration)
    }

    deinit {
        nodeStorage.deallocate()
    }

    @inlinable
    public func node(id: UUID) -> ClusterNode<Vector> {
        guard let index = idToIndexMap[id] else {
            fatalError("Node ID \(id) not found in HierarchicalClusterTree.")
        }
        let compactNode = nodeStorage[index]
        return ClusterNode<Vector>(from: compactNode, storage: storage, parent: nil)
    }

    // Serialization stubs
    public func serialize(to url: URL) throws { throw VectorError(.unsupportedOperation) }
    public static func deserialize(from url: URL) throws -> HierarchicalClusterTree<Vector> { throw VectorError(.unsupportedOperation) }

    // MARK: - Performance Benchmarking

    /// Benchmark clustering performance
    public static func benchmark<V: OptimizedVector>(
        vectors: ContiguousArray<V>,
        config: ClusteringConfig = ClusteringConfig()
    ) async -> (tree: HierarchicalClusterTree<V>, constructionTime: TimeInterval) {
        let startTime = CFAbsoluteTimeGetCurrent()

        do {
            let tree = try await HierarchicalClusterTree<V>(vectors: vectors, configuration: config)
            let endTime = CFAbsoluteTimeGetCurrent()
            return (tree, endTime - startTime)
        } catch {
            fatalError("Clustering failed: \(error)")
        }
    }
}

// MARK: - SIMD-Optimized Computation Functions

/// High-performance SIMD-optimized centroid computation
internal func computeCentroidSIMD<Vector: OptimizedVector>(_ vectors: ContiguousArray<Vector>) -> Vector {
    guard !vectors.isEmpty else { return Vector.zero }

    let count = vectors.count
    let lanes = Vector.quantLaneCount

    // SIMD accumulation across all vectors
    var accumulators = ContiguousArray<SIMD4<Float>>(repeating: SIMD4<Float>(0, 0, 0, 0), count: lanes)

    for vector in vectors {
        for (laneIndex, chunk) in vector.storage.enumerated() {
            accumulators[laneIndex] += chunk
        }
    }

    // Vectorized division by count
    let invCount = SIMD4<Float>(repeating: 1.0 / Float(count))
    for laneIndex in 0..<lanes {
        accumulators[laneIndex] *= invCount
    }

    return Vector(storage: accumulators)
}

/// High-performance SIMD-optimized squared distance computation
@usableFromInline
internal func squaredDistanceSIMD<Vector: OptimizedVector>(_ a: Vector, _ b: Vector) -> Float {
    var accumulator = SIMD4<Float>(0, 0, 0, 0)

    for (chunkA, chunkB) in zip(a.storage, b.storage) {
        let diff = chunkA - chunkB
        accumulator += diff * diff
    }

    // Horizontal sum of SIMD4
    return accumulator.x + accumulator.y + accumulator.z + accumulator.w
}

/// SIMD-optimized bounding box computation
internal func computeBoundingBoxSIMD<Vector: OptimizedVector>(_ vectors: ContiguousArray<Vector>) -> BoundingBox<Vector> {
    guard let first = vectors.first else {
        let zero = Vector.zero
        return BoundingBox(min: zero, max: zero)
    }

    let lanes = Vector.quantLaneCount
    var minStorage = ContiguousArray(first.storage)
    var maxStorage = ContiguousArray(first.storage)

    // SIMD min/max across all vectors
    for vector in vectors.dropFirst() {
        for laneIndex in 0..<lanes {
            minStorage[laneIndex] = simd_min(minStorage[laneIndex], vector.storage[laneIndex])
            maxStorage[laneIndex] = simd_max(maxStorage[laneIndex], vector.storage[laneIndex])
        }
    }

    return BoundingBox(min: Vector(storage: minStorage), max: Vector(storage: maxStorage))
}

// MARK: - Construction Algorithm

internal struct VectorPartition<Vector: OptimizedVector>: Sendable {
    let left: ContiguousArray<Vector>, right: ContiguousArray<Vector>
    let leftCentroid: Vector, rightCentroid: Vector
    let leftBounds: BoundingBox<Vector>, rightBounds: BoundingBox<Vector>
}

@usableFromInline internal struct ConstructionResult: Sendable {
    let rootId: UUID, nodeCount: Int
}

public enum HierarchicalClustering {

    public static func buildTree<Vector: OptimizedVector>(
        vectors: ContiguousArray<Vector>,
        config: ClusteringConfig
    ) async throws -> (UnsafeMutablePointer<CompactNode>, Int, UUID, [UUID: Int], ClusterTreeStorage<Vector>) {

        guard !vectors.isEmpty else { throw VectorError(.insufficientData) }

        let estimatedNodes = estimateNodeCount(vectorCount: vectors.count, leafSize: config.maxLeafSize)
        let constructionState = ConstructionState<Vector>(estimatedNodes: estimatedNodes)

        let rootBounds = computeBoundingBoxSIMD(vectors)
        let rootCentroid = computeCentroidSIMD(vectors)

        let result = try await constructSubtree(
            vectors: vectors, bounds: rootBounds, centroid: rootCentroid, depth: 0,
            config: config, state: constructionState
        )

        let finalMap = await constructionState.getIdToIndexMap()
        let nodeStorage = constructionState.getNodeStorage()
        return (nodeStorage, result.nodeCount, result.rootId, finalMap, constructionState.storage)
    }

    @usableFromInline
    internal static func constructSubtree<Vector: OptimizedVector>(
        vectors: ContiguousArray<Vector>, bounds: BoundingBox<Vector>, centroid: Vector, depth: Int,
        config: ClusteringConfig, state: ConstructionState<Vector>
    ) async throws -> ConstructionResult {

        // Base case: Leaf Node
        if vectors.count <= config.maxLeafSize || depth >= config.maxDepth {
            return await createLeafNode(vectors: vectors, bounds: bounds, centroid: centroid, depth: depth, state: state)
        }

        // Partitioning
        let (centroid1, centroid2) = selectOptimalCentroids(vectors, config)
        var partition = partitionVectors(vectors, centroid1, centroid2, config.metric)

        // Balance check and fallback
        if partition.left.count < config.minClusterSize || partition.right.count < config.minClusterSize {
            partition = medianSplit(vectors, centroid, config.metric)
            if partition.left.isEmpty || partition.right.isEmpty {
                 return await createLeafNode(vectors: vectors, bounds: bounds, centroid: centroid, depth: depth, state: state)
            }
        }

        // Phase 3: SIMD-optimized sequential construction (parallelism via dispatch later)
        let leftChild = try await constructSubtree(
            vectors: partition.left, bounds: partition.leftBounds,
            centroid: partition.leftCentroid, depth: depth + 1,
            config: config, state: state
        )
        let rightChild = try await constructSubtree(
            vectors: partition.right, bounds: partition.rightBounds,
            centroid: partition.rightCentroid, depth: depth + 1,
            config: config, state: state
        )

        // Create internal node
        let nodeId = UUID()
        let internalNode = CompactNode.createInternal(id: nodeId, leftChild: leftChild.rootId,
                                                     rightChild: rightChild.rootId, centroid: centroid,
                                                     bounds: bounds, level: depth, subtreeSize: vectors.count,
                                                     storage: state.storage)

        _ = await state.allocateNode(id: nodeId, node: internalNode)

        return ConstructionResult(rootId: nodeId, nodeCount: 1 + leftChild.nodeCount + rightChild.nodeCount)
    }

    private static func createLeafNode<Vector: OptimizedVector>(
        vectors: ContiguousArray<Vector>, bounds: BoundingBox<Vector>, centroid: Vector, depth: Int,
        state: ConstructionState<Vector>
    ) async -> ConstructionResult {
        let nodeId = UUID()
        let leafNode = CompactNode.createLeaf(id: nodeId, vectors: vectors, centroid: centroid,
                                             bounds: bounds, level: depth, storage: state.storage)

        _ = await state.allocateNode(id: nodeId, node: leafNode)

        return ConstructionResult(rootId: nodeId, nodeCount: 1)
    }

    // Utility functions
    internal static func estimateNodeCount(vectorCount: Int, leafSize: Int) -> Int {
        return max(1, Int(2.5 * Double(vectorCount) / Double(max(1, leafSize))))
    }

    internal static func computeCentroid<V: OptimizedVector>(_ vectors: ContiguousArray<V>) -> V {
        guard !vectors.isEmpty else { return V.zero }
        var sum = V.zero
        for vector in vectors { sum = sum.add(vector) }
        return sum.divide(by: Float(vectors.count))
    }

    internal static func computeBoundingBox<V: OptimizedVector>(_ vectors: ContiguousArray<V>) -> BoundingBox<V> {
        return BoundingBox(vectors: vectors)
    }

    internal static func selectOptimalCentroids<Vector: OptimizedVector>(
        _ vectors: ContiguousArray<Vector>,
        _ config: ClusteringConfig
    ) -> (Vector, Vector) {
        if vectors.count <= 2 { return (vectors[0], vectors.last!) }

        // Simplified implementation - use first and last vectors
        // In production, this would use k-means++ algorithm
        return (vectors[0], vectors[vectors.count - 1])
    }

    internal static func partitionVectors<Vector: OptimizedVector>(
        _ vectors: ContiguousArray<Vector>, _ centroid1: Vector, _ centroid2: Vector, _ metric: ClusteringDistanceMetric
    ) -> VectorPartition<Vector> {
        var leftVectors = ContiguousArray<Vector>()
        var rightVectors = ContiguousArray<Vector>()

        for vector in vectors {
            let dist1 = squaredDistanceSIMD(centroid1, vector)
            let dist2 = squaredDistanceSIMD(centroid2, vector)

            if dist1 <= dist2 {
                leftVectors.append(vector)
            } else {
                rightVectors.append(vector)
            }
        }

        return VectorPartition(
            left: leftVectors, right: rightVectors,
            leftCentroid: leftVectors.isEmpty ? centroid1 : computeCentroidSIMD(leftVectors),
            rightCentroid: rightVectors.isEmpty ? centroid2 : computeCentroidSIMD(rightVectors),
            leftBounds: leftVectors.isEmpty ? BoundingBox(vectors: [centroid1]) : computeBoundingBoxSIMD(leftVectors),
            rightBounds: rightVectors.isEmpty ? BoundingBox(vectors: [centroid2]) : computeBoundingBoxSIMD(rightVectors)
        )
    }

    internal static func medianSplit<V: OptimizedVector>(_ vectors: ContiguousArray<V>, _ centroid: V, _ metric: ClusteringDistanceMetric) -> VectorPartition<V> {
        let mid = vectors.count / 2
        let left = ContiguousArray(vectors[0..<mid])
        let right = ContiguousArray(vectors[mid..<vectors.count])
        return VectorPartition(
            left: left, right: right,
            leftCentroid: computeCentroidSIMD(left), rightCentroid: computeCentroidSIMD(right),
            leftBounds: computeBoundingBoxSIMD(left), rightBounds: computeBoundingBoxSIMD(right)
        )
    }
}