//
//  EdgeCaseHandler.swift
//  VectorCore
//
//  Comprehensive edge case handling for kernel operations
//  Provides safety guards for empty arrays, single elements, zero vectors, and numerical edge cases
//

import Foundation
import simd

/// Central edge case handler for kernel operations
public enum EdgeCaseHandler {

    // MARK: - Array Size Edge Cases

    /// Validates that an array is not empty
    @inlinable
    public static func requireNonEmpty<T>(_ array: [T], operation: String = "operation") throws {
        guard !array.isEmpty else {
            throw VectorError.invalidData("Cannot perform \(operation) on empty array")
        }
    }

    /// Validates minimum array size
    @inlinable
    public static func requireMinimumSize<T>(_ array: [T], minSize: Int, operation: String = "operation") throws {
        guard array.count >= minSize else {
            throw VectorError.invalidData("Array size \(array.count) is below minimum \(minSize) for \(operation)")
        }
    }

    /// Handles empty array case with default value
    @inlinable
    public static func handleEmpty<T, R>(_ array: [T], defaultValue: R, operation: (([T]) throws -> R)) rethrows -> R {
        guard !array.isEmpty else {
            return defaultValue
        }
        return try operation(array)
    }

    // MARK: - Single Element Edge Cases

    /// Handles single element arrays in batch operations
    @inlinable
    public static func handleSingleElement<V: VectorProtocol>(_ vectors: [V], operation: String) throws -> SingleElementResult<V> {
        switch vectors.count {
        case 0:
            throw VectorError.invalidData("Cannot perform \(operation) on empty vector array")
        case 1:
            return .single(vectors[0])
        default:
            return .multiple(vectors)
        }
    }

    public enum SingleElementResult<V: VectorProtocol> {
        case single(V)
        case multiple([V])

        public var isSingle: Bool {
            switch self {
            case .single: return true
            case .multiple: return false
            }
        }
    }

    // MARK: - Zero Vector Handling

    /// Checks if a vector is effectively zero (all components below epsilon)
    @inlinable
    public static func isZeroVector<V: VectorProtocol>(_ vector: V, epsilon: Float = 1e-10) -> Bool
    where V.Scalar == Float {
        let magnitude = vector.magnitude
        return magnitude < epsilon
    }

    /// Checks if normalization will fail due to zero magnitude
    @inlinable
    public static func canNormalize<V: VectorProtocol>(_ vector: V, epsilon: Float = 1e-10) -> Bool
    where V.Scalar == Float {
        return vector.magnitude >= epsilon
    }

    /// Safe normalization helper for vectors that have Result-based normalized() method
    /// Returns the normalized vector if successful, or nil if it's a zero vector
    @inlinable
    public static func tryNormalize(_ vector: Vector512Optimized) -> Vector512Optimized? {
        if case .success(let normalized) = vector.normalized() {
            return normalized
        }
        return nil
    }

    /// Safe normalization helper for Vector768Optimized
    @inlinable
    public static func tryNormalize(_ vector: Vector768Optimized) -> Vector768Optimized? {
        if case .success(let normalized) = vector.normalized() {
            return normalized
        }
        return nil
    }

    /// Safe normalization helper for Vector1536Optimized
    @inlinable
    public static func tryNormalize(_ vector: Vector1536Optimized) -> Vector1536Optimized? {
        if case .success(let normalized) = vector.normalized() {
            return normalized
        }
        return nil
    }

    // MARK: - Division by Zero Protection

    /// Safe division with default value for division by zero
    @inlinable
    public static func safeDivide(_ numerator: Float, by denominator: Float, defaultValue: Float = 0.0) -> Float {
        guard denominator != 0 && !denominator.isNaN else {
            return defaultValue
        }
        return numerator / denominator
    }

    /// Safe division for SIMD vectors
    @inlinable
    public static func safeDivide(_ numerator: SIMD4<Float>, by denominator: SIMD4<Float>, defaultValue: Float = 0.0) -> SIMD4<Float> {
        var result = SIMD4<Float>()
        for i in 0..<4 {
            if denominator[i] != 0 && !denominator[i].isNaN {
                result[i] = numerator[i] / denominator[i]
            } else {
                result[i] = defaultValue
            }
        }
        return result
    }

    // MARK: - NaN and Infinity Handling

    /// Validates that a value is finite (not NaN or Infinity)
    @inlinable
    public static func requireFinite(_ value: Float, name: String = "value") throws {
        guard value.isFinite else {
            if value.isNaN {
                throw VectorError.invalidData("\(name) is NaN")
            } else if value.isInfinite {
                throw VectorError.invalidData("\(name) is infinite")
            } else {
                throw VectorError.invalidData("\(name) is not finite")
            }
        }
    }

    /// Clamps infinite values to finite bounds
    @inlinable
    public static func clampInfinities(_ value: Float, min: Float = -Float.greatestFiniteMagnitude,
                                       max: Float = Float.greatestFiniteMagnitude) -> Float {
        if value.isInfinite {
            return value.sign == .plus ? max : min
        }
        return value
    }

    /// Replaces NaN values with a default
    @inlinable
    public static func replaceNaN(_ value: Float, with defaultValue: Float = 0.0) -> Float {
        return value.isNaN ? defaultValue : value
    }

    /// Sanitizes a SIMD4 vector by replacing NaN/Inf with defaults
    @inlinable
    public static func sanitizeSIMD4(_ value: SIMD4<Float>, nanDefault: Float = 0.0,
                                     infDefault: Float = Float.greatestFiniteMagnitude) -> SIMD4<Float> {
        var result = value
        for i in 0..<4 {
            if result[i].isNaN {
                result[i] = nanDefault
            } else if result[i].isInfinite {
                result[i] = result[i].sign == .plus ? infDefault : -infDefault
            }
        }
        return result
    }

    // MARK: - Bounds Checking

    /// Validates array index is in bounds
    @inlinable
    public static func requireValidIndex(_ index: Int, count: Int, name: String = "index") throws {
        guard index >= 0 && index < count else {
            throw VectorError.indexOutOfBounds(
                index: index,
                dimension: count
            )
        }
    }

    /// Validates range is valid for array
    @inlinable
    public static func requireValidRange(_ range: Range<Int>, count: Int) throws {
        guard range.lowerBound >= 0 && range.upperBound <= count else {
            throw VectorError(.invalidRange, message: "Range \(range) invalid for array of size \(count)")
        }
    }

    /// Clamps an index to valid bounds
    @inlinable
    public static func clampIndex(_ index: Int, count: Int) -> Int {
        return max(0, min(index, count - 1))
    }

    // MARK: - Dimension Validation

    /// Validates that dimensions match
    @inlinable
    public static func requireMatchingDimensions<V: VectorProtocol>(_ v1: V, _ v2: V) throws {
        guard v1.count == v2.count else {
            throw VectorError.dimensionMismatch(
                expected: v1.count,
                actual: v2.count
            )
        }
    }

    /// Validates dimension is positive
    @inlinable
    public static func requirePositiveDimension(_ dimension: Int) throws {
        guard dimension > 0 else {
            throw VectorError.invalidDimension(
                dimension,
                reason: "Dimension must be positive"
            )
        }
    }

    /// Validates dimension is a multiple of SIMD lane width
    @inlinable
    public static func requireSIMDAlignedDimension(_ dimension: Int, laneWidth: Int = 4) throws {
        guard dimension % laneWidth == 0 else {
            throw VectorError.invalidDimension(
                dimension,
                reason: "Must be multiple of \(laneWidth) for SIMD operations"
            )
        }
    }

    // MARK: - Batch Operation Edge Cases

    /// Handles edge cases for batch euclidean distance
    public static func handleBatchDistanceEdgeCases<V: VectorProtocol>(
        query: V,
        candidates: [V]
    ) throws -> BatchDistanceResult where V.Scalar == Float {
        // Check for empty candidates
        guard !candidates.isEmpty else {
            return .empty
        }

        // Single candidate optimization
        if candidates.count == 1 {
            let distance = Float(query.euclideanDistanceSquared(to: candidates[0]))
            return .single(distance)
        }

        // Check for zero query vector
        if isZeroVector(query) {
            // All distances are just the magnitudes of candidates
            let distances = candidates.map { candidate in
                Float(candidate.magnitudeSquared)
            }
            return .computed(distances)
        }

        return .needsComputation
    }

    public enum BatchDistanceResult {
        case empty
        case single(Float)
        case computed([Float])
        case needsComputation
    }

    // MARK: - Clustering Edge Cases

    /// Handles edge cases for hierarchical clustering
    public static func handleClusteringEdgeCases<V: VectorProtocol>(
        vectors: [V]
    ) throws -> ClusteringAction where V.Scalar == Float {
        switch vectors.count {
        case 0:
            throw VectorError.invalidData("Cannot cluster empty vector set")
        case 1:
            return .singleCluster(0)
        case 2:
            return .pairCluster(0, 1)
        default:
            // Check if all vectors are identical
            let first = vectors[0]
            let allIdentical = vectors.allSatisfy { vector in
                // Use euclidean distance to check if vectors are identical
                Float(first.euclideanDistanceSquared(to: vector)) < 1e-10
            }

            if allIdentical {
                return .allIdentical(count: vectors.count)
            }

            return .proceed
        }
    }

    public enum ClusteringAction {
        case singleCluster(Int)
        case pairCluster(Int, Int)
        case allIdentical(count: Int)
        case proceed
    }

    // MARK: - SoA Edge Cases

    /// Validates SoA structure before operations
    public static func validateSoA<V: SoACompatible>(soa: SoA<V>) throws {
        guard soa.count > 0 else {
            throw VectorError.invalidData("SoA structure is empty")
        }

        guard soa.lanes > 0 else {
            throw VectorError.invalidData("SoA structure has zero lanes")
        }

        // Validate buffer size
        let expectedSize = soa.lanes * soa.count
        let actualSize = soa.memoryFootprint / MemoryLayout<SIMD4<Float>>.size
        guard actualSize >= expectedSize else {
            throw VectorError.allocationFailed(size: expectedSize)
        }
    }

    // MARK: - Graph Operation Edge Cases

    /// Handles edge cases for graph operations
    public static func handleGraphEdgeCases(nodeCount: Int, edgeCount: Int) throws -> GraphStatus {
        guard nodeCount >= 0 else {
            throw VectorError.invalidData("Node count cannot be negative: \(nodeCount)")
        }

        guard edgeCount >= 0 else {
            throw VectorError.invalidData("Edge count cannot be negative: \(edgeCount)")
        }

        if nodeCount == 0 {
            return .emptyGraph
        }

        if nodeCount == 1 {
            return .singleNode
        }

        if edgeCount == 0 {
            return .disconnected(nodeCount: nodeCount)
        }

        // Check for impossible edge count (more than complete graph)
        let maxEdges = nodeCount * (nodeCount - 1) / 2  // For undirected graph
        if edgeCount > maxEdges {
            throw VectorError.invalidData(
                "Edge count \(edgeCount) exceeds maximum \(maxEdges) for \(nodeCount) nodes"
            )
        }

        return .normal
    }

    public enum GraphStatus {
        case emptyGraph
        case singleNode
        case disconnected(nodeCount: Int)
        case normal
    }
}

// MARK: - Integration with Existing Kernels

extension EuclideanKernels {
    /// Safe euclidean distance with edge case handling
    public static func safeEuclideanDistance512(_ v1: Vector512Optimized, _ v2: Vector512Optimized) throws -> Float {
        try EdgeCaseHandler.requireMatchingDimensions(v1, v2)

        // Check for identical vectors
        let distance = squared512(v1, v2)

        // Handle numerical precision issues
        if distance < 0 {
            // Can happen due to floating point errors
            return 0
        }

        return sqrt(distance)
    }
}

extension BatchKernels_SoA {
    /// Safe batch euclidean with comprehensive edge case handling
    public static func safeBatchEuclidean512(
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) throws -> [Float] {
        let edgeCase = try EdgeCaseHandler.handleBatchDistanceEdgeCases(query: query, candidates: candidates)

        switch edgeCase {
        case .empty:
            return []
        case .single(let distance):
            return [distance]
        case .computed(let distances):
            return distances
        case .needsComputation:
            // Proceed with normal computation
            return batchEuclideanSquared512(query: query, candidates: candidates)
        }
    }
}

extension HierarchicalClusteringKernels {
    /// Safe agglomerative clustering with edge case handling
    public static func safeAgglomerativeClustering(
        vectors: [Vector512Optimized],
        linkageCriterion: LinkageCriterion = .ward,
        distanceMetric: ClusteringDistanceMetric = .euclidean
    ) throws -> HierarchicalTree {
        let action = try EdgeCaseHandler.handleClusteringEdgeCases(vectors: vectors)

        switch action {
        case .singleCluster(let index):
            // Create trivial tree with single node
            let node = ClusterNode(
                id: 0,
                vectorIndices: [index],
                centroid: vectors[0],
                radius: 0.0
            )
            return HierarchicalTree(
                nodes: [node],
                rootNodeId: 0,
                leafNodeIds: [0],
                dimension: 512,
                linkageCriterion: linkageCriterion,
                distanceMetric: distanceMetric
            )

        case .pairCluster(let i1, let i2):
            // Create simple tree with two leaves and one root
            let node1 = ClusterNode(id: 0, vectorIndices: [i1], centroid: vectors[i1], radius: 0.0, parent: 2)
            let node2 = ClusterNode(id: 1, vectorIndices: [i2], centroid: vectors[i2], radius: 0.0, parent: 2)
            let distance = EuclideanKernels.distance512(vectors[i1], vectors[i2])
            let rootCentroid = try Vector512Optimized((0..<512).map { i in
                (vectors[i1][i] + vectors[i2][i]) / 2.0
            })
            let root = ClusterNode(
                id: 2,
                vectorIndices: [i1, i2],
                centroid: rootCentroid,
                radius: distance / 2.0,
                leftChild: 0,
                rightChild: 1,
                height: 1,
                mergeDistance: distance
            )

            return HierarchicalTree(
                nodes: [node1, node2, root],
                rootNodeId: 2,
                leafNodeIds: [0, 1],
                dimension: 512,
                linkageCriterion: linkageCriterion,
                distanceMetric: distanceMetric
            )

        case .allIdentical(let count):
            // All vectors are identical - create star topology
            var nodes: [ClusterNode] = []
            let centroid = vectors[0]

            // Create leaf nodes
            for i in 0..<count {
                nodes.append(ClusterNode(
                    id: i,
                    vectorIndices: [i],
                    centroid: centroid,
                    radius: 0.0,
                    parent: count  // Root will have id = count
                ))
            }

            // Create root node
            nodes.append(ClusterNode(
                id: count,
                vectorIndices: Set(0..<count),
                centroid: centroid,
                radius: 0.0,
                height: 1,
                mergeDistance: 0.0
            ))

            return HierarchicalTree(
                nodes: ContiguousArray(nodes),
                rootNodeId: count,
                leafNodeIds: Set(0..<count),
                dimension: 512,
                linkageCriterion: linkageCriterion,
                distanceMetric: distanceMetric
            )

        case .proceed:
            // Use normal clustering algorithm
            return agglomerativeClustering(
                vectors: vectors,
                linkageCriterion: linkageCriterion,
                distanceMetric: distanceMetric
            )
        }
    }
}