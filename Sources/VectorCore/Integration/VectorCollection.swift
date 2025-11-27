//
//  VectorCollection.swift
//  VectorCore
//
//  Protocol for collections of vectors that support search operations.
//  Defines the contract between VectorCore and VectorIndex.
//

import Foundation

// MARK: - VectorCollection Protocol

/// Protocol for collections of vectors that support batch operations and search.
///
/// This protocol defines the interface that VectorIndex will implement,
/// allowing VectorCore to provide default brute-force implementations
/// that VectorIndex can override with optimized index-based algorithms.
///
/// ## Design Philosophy
///
/// - **VectorCore provides**: Default brute-force implementations
/// - **VectorIndex provides**: Optimized approximate nearest neighbor (ANN) algorithms
/// - **Clear ownership**: VectorCore owns vector operations, VectorIndex owns index structures
///
/// ## Example Usage
/// ```swift
/// // VectorIndex implements VectorCollection
/// let index: some VectorCollection = HNSWIndex(dimension: 512)
///
/// // Add vectors
/// for (id, vector) in vectors {
///     try index.add(id: id, vector: vector)
/// }
///
/// // Search uses index-optimized algorithm
/// let results = index.search(query: queryVector, k: 10)
/// ```
public protocol VectorCollection: Sendable {
    /// The type of vectors stored in this collection
    associatedtype Vector: IndexableVector

    /// The type used to identify vectors
    associatedtype ID: Hashable & Sendable = Int

    // MARK: - Collection Properties

    /// Number of vectors in the collection
    var count: Int { get }

    /// Dimension of vectors in this collection
    var dimension: Int { get }

    /// Whether the collection is empty
    var isEmpty: Bool { get }

    // MARK: - Vector Access

    /// Retrieve a vector by its identifier.
    ///
    /// - Parameter id: The vector's identifier
    /// - Returns: The vector, or nil if not found
    func vector(for id: ID) -> Vector?

    /// Retrieve multiple vectors by identifiers.
    ///
    /// - Parameter ids: Array of identifiers to retrieve
    /// - Returns: Dictionary mapping found IDs to vectors
    func vectors(for ids: [ID]) -> [ID: Vector]

    /// All identifiers in the collection.
    var allIDs: [ID] { get }

    // MARK: - Search Operations

    /// Find k nearest neighbors to a query vector.
    ///
    /// Default implementation performs brute-force search.
    /// VectorIndex implementations will override with optimized algorithms.
    ///
    /// - Parameters:
    ///   - query: The query vector
    ///   - k: Number of nearest neighbors to find
    ///   - metric: Distance metric to use
    /// - Returns: SearchResults containing k nearest neighbors
    func search(
        query: Vector,
        k: Int,
        metric: any DistanceMetric
    ) -> SearchResults<ID>
}

// MARK: - Default Implementations

extension VectorCollection {
    /// Default isEmpty based on count
    public var isEmpty: Bool { count == 0 }

    /// Default batch vector retrieval
    public func vectors(for ids: [ID]) -> [ID: Vector] {
        var result: [ID: Vector] = [:]
        result.reserveCapacity(ids.count)
        for id in ids {
            if let v = vector(for: id) {
                result[id] = v
            }
        }
        return result
    }

    /// Default brute-force search implementation using Euclidean distance.
    ///
    /// VectorIndex will override this with optimized ANN algorithms.
    /// This implementation has O(n * d) complexity where n is vector count
    /// and d is dimension.
    public func search(
        query: Vector,
        k: Int,
        metric: any DistanceMetric
    ) -> SearchResults<ID> {
        // Use specific metric implementations to avoid existential limitations
        if metric is EuclideanDistance {
            return searchWithMetric(query: query, k: k, metric: EuclideanDistance())
        } else if metric is CosineDistance {
            return searchWithMetric(query: query, k: k, metric: CosineDistance())
        } else if metric is ManhattanDistance {
            return searchWithMetric(query: query, k: k, metric: ManhattanDistance())
        } else if metric is DotProductDistance {
            return searchWithMetric(query: query, k: k, metric: DotProductDistance())
        } else {
            // Fallback to Euclidean for unknown metrics
            return searchWithMetric(query: query, k: k, metric: EuclideanDistance())
        }
    }

    /// Internal generic search implementation
    private func searchWithMetric<M: DistanceMetric>(
        query: Vector,
        k: Int,
        metric: M
    ) -> SearchResults<ID> where M.Scalar == Float {
        let startTime = DispatchTime.now().uptimeNanoseconds

        // Collect all vectors with their IDs
        let allVectors: [(ID, Vector)] = allIDs.compactMap { id in
            guard let v = vector(for: id) else { return nil }
            return (id, v)
        }

        guard !allVectors.isEmpty else {
            return SearchResults()
        }

        // Compute distances
        var distances: [Float] = []
        distances.reserveCapacity(allVectors.count)

        for (_, candidate) in allVectors {
            let distance = metric.distance(query, candidate)
            distances.append(distance)
        }

        // Select top-k
        let topK = TopKSelection.select(k: k, from: distances)

        // Build results with original IDs
        let results: [SearchResult<ID>] = topK.indices.enumerated().map { offset, index in
            let (id, _) = allVectors[index]
            return SearchResult(id: id, distance: topK.distances[offset])
        }

        let endTime = DispatchTime.now().uptimeNanoseconds

        return SearchResults(
            results: results,
            candidatesSearched: allVectors.count,
            searchTimeNanos: endTime - startTime,
            isExhaustive: true
        )
    }
}

// MARK: - Euclidean Search Convenience

extension VectorCollection {
    /// Search using Euclidean distance (most common case).
    public func searchEuclidean(query: Vector, k: Int) -> SearchResults<ID> {
        search(query: query, k: k, metric: EuclideanDistance())
    }

    /// Search using Cosine distance.
    public func searchCosine(query: Vector, k: Int) -> SearchResults<ID> {
        search(query: query, k: k, metric: CosineDistance())
    }

    /// Search using Dot Product (for maximum inner product search).
    public func searchDotProduct(query: Vector, k: Int) -> SearchResults<ID> {
        search(query: query, k: k, metric: DotProductDistance())
    }
}

// MARK: - Mutable Collection Protocol

/// Protocol for vector collections that support adding and removing vectors.
///
/// Extends VectorCollection with mutation operations. VectorIndex will
/// implement this for mutable index structures.
public protocol MutableVectorCollection: VectorCollection {
    /// Add a vector to the collection.
    ///
    /// - Parameters:
    ///   - id: Identifier for the vector
    ///   - vector: The vector to add
    /// - Throws: If the vector dimension doesn't match or ID already exists
    mutating func add(id: ID, vector: Vector) throws

    /// Remove a vector from the collection.
    ///
    /// - Parameter id: Identifier of vector to remove
    /// - Returns: The removed vector, or nil if not found
    @discardableResult
    mutating func remove(id: ID) -> Vector?

    /// Remove all vectors from the collection.
    mutating func removeAll()

    /// Update a vector in the collection.
    ///
    /// - Parameters:
    ///   - id: Identifier of vector to update
    ///   - vector: New vector value
    /// - Throws: If the vector dimension doesn't match or ID doesn't exist
    mutating func update(id: ID, vector: Vector) throws
}

// MARK: - Default Update Implementation

extension MutableVectorCollection {
    /// Default update implementation using remove + add
    public mutating func update(id: ID, vector: Vector) throws {
        guard self.vector(for: id) != nil else {
            throw VectorError.indexOutOfBounds(index: 0, dimension: dimension)
        }
        remove(id: id)
        try add(id: id, vector: vector)
    }
}

// MARK: - Simple In-Memory Collection

/// A simple in-memory vector collection for testing and small datasets.
///
/// This is a reference implementation of MutableVectorCollection.
/// For production use with large datasets, use VectorIndex instead.
public final class SimpleVectorCollection<V: IndexableVector>: MutableVectorCollection, @unchecked Sendable {
    public typealias Vector = V
    public typealias ID = Int

    private var vectors: [Int: V] = [:]
    private var nextID: Int = 0
    private let lock = NSLock()

    public let dimension: Int

    public init(dimension: Int) {
        self.dimension = dimension
    }

    public var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return vectors.count
    }

    public var allIDs: [Int] {
        lock.lock()
        defer { lock.unlock() }
        return Array(vectors.keys)
    }

    public func vector(for id: Int) -> V? {
        lock.lock()
        defer { lock.unlock() }
        return vectors[id]
    }

    public func add(id: Int, vector: V) throws {
        guard vector.scalarCount == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: vector.scalarCount)
        }
        lock.lock()
        defer { lock.unlock() }
        vectors[id] = vector
    }

    /// Add a vector with auto-generated ID
    @discardableResult
    public func add(vector: V) throws -> Int {
        guard vector.scalarCount == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: vector.scalarCount)
        }
        lock.lock()
        defer { lock.unlock() }
        let id = nextID
        nextID += 1
        vectors[id] = vector
        return id
    }

    @discardableResult
    public func remove(id: Int) -> V? {
        lock.lock()
        defer { lock.unlock() }
        return vectors.removeValue(forKey: id)
    }

    public func removeAll() {
        lock.lock()
        defer { lock.unlock() }
        vectors.removeAll()
    }
}
