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
        metric: any DistanceMetric<Float>
    ) -> SearchResults<ID>
}

// MARK: - Default Implementations

extension VectorCollection {
    /// Default isEmpty based on count
    public var isEmpty: Bool { count == 0 }

    /// Default batch retrieval composes individual reads, not an atomic snapshot.
    /// Concurrent snapshot consistency is the conformer's responsibility.
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

    /// Default brute-force search using the supplied metric, in O(n * d) work.
    /// Composing allIDs and vector(for:) does not provide an atomic snapshot;
    /// concurrent snapshot consistency is the conformer's responsibility.
    public func search(
        query: Vector,
        k: Int,
        metric: any DistanceMetric<Float>
    ) -> SearchResults<ID> {
        let startTime = DispatchTime.now().uptimeNanoseconds
        let snapshot: [(ID, Vector)] = allIDs.compactMap { id in
            guard let value = vector(for: id) else { return nil }
            return (id, value)
        }
        return searchCapturedVectors(snapshot, query: query, k: k, metric: metric, startTime: startTime)
    }
}

/// Rank captured bindings by the actual distance witness, preserving TopK policy.
/// Interpret only selected distances, once each; SearchResultScoreContractTests
/// checks both producer routes and reentrant score callbacks outside Simple's lock.
private func searchCapturedVectors<ID: Hashable & Sendable, V: IndexableVector, M: DistanceMetric>(
    _ snapshot: [(ID, V)], query: V, k: Int, metric: M, startTime: UInt64
) -> SearchResults<ID> where M.Scalar == Float {
    guard !snapshot.isEmpty else { return SearchResults() }
    var distances: [Float] = []
    distances.reserveCapacity(snapshot.count)
    for (_, candidate) in snapshot {
        distances.append(metric.distance(query, candidate))
    }
    let topK = TopKSelection.select(k: k, from: distances)
    let results: [SearchResult<ID>] = topK.indices.enumerated().map { offset, index in
        let distance = topK.distances[offset]
        return SearchResult(id: snapshot[index].0, distance: distance, score: metric.relevanceScore(for: distance))
    }
    return SearchResults(
        results: results,
        candidatesSearched: snapshot.count,
        searchTimeNanos: DispatchTime.now().uptimeNanoseconds - startTime,
        isExhaustive: true
    )
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

    /// Transactionally replace an existing binding.
    /// Every thrown update must preserve the previous binding and membership.
    /// Conformers must validate and commit within their own synchronization or
    /// transaction boundary; no generic remove/add rollback is provided.
    ///
    /// - Parameters:
    ///   - id: Identifier of vector to update
    ///   - vector: New vector value
    /// - Throws: If the vector dimension doesn't match or ID doesn't exist
    mutating func update(id: ID, vector: Vector) throws
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
    // nil denotes exhaustion after allocating Int.max. Errors leave this unchanged.
    private var nextID: Int? = 0
    private let lock = NSLock()

    public let dimension: Int

    public init(dimension: Int) {
        self.dimension = dimension
    }

    /// Internal boundary seam for bounded exhaustion tests; not public configuration.
    internal init(dimension: Int, nextGeneratedID: Int) {
        precondition(nextGeneratedID >= 0, "Generated IDs must be nonnegative")
        self.dimension = dimension
        self.nextID = nextGeneratedID
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

    /// Capture all requested bindings under one lock. V's value semantics remain
    /// the conformer's responsibility; this does not freeze external reference aliases.
    public func vectors(for ids: [Int]) -> [Int: V] {
        lock.lock()
        defer { lock.unlock() }
        var result: [Int: V] = [:]
        result.reserveCapacity(ids.count)
        for id in ids {
            if let value = vectors[id] { result[id] = value }
        }
        return result
    }

    /// Capture bindings once under the lock, then invoke the metric outside it.
    /// VectorCollectionTransactionTests exercises reentrant metric mutations.
    /// Captured V bindings are not universal deep copies of custom reference storage.
    public func search(query: V, k: Int, metric: any DistanceMetric<Float>) -> SearchResults<Int> {
        let startTime = DispatchTime.now().uptimeNanoseconds
        lock.lock()
        let snapshot: [(Int, V)] = vectors.map { ($0.key, $0.value) }
        lock.unlock()
        return searchCapturedVectors(snapshot, query: query, k: k, metric: metric, startTime: startTime)
    }

    /// Validate before locking; replace only after confirming membership.
    public func update(id: Int, vector: V) throws {
        guard vector.scalarCount == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: vector.scalarCount)
        }
        lock.lock()
        defer { lock.unlock() }
        guard vectors[id] != nil else {
            throw VectorError.indexOutOfBounds(index: 0, dimension: dimension)
        }
        vectors[id] = vector
    }

    public func add(id: Int, vector: V) throws {
        guard vector.scalarCount == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: vector.scalarCount)
        }
        lock.lock()
        defer { lock.unlock() }
        guard vectors[id] == nil else {
            throw VectorError(.invalidConfiguration, message: "Duplicate vector ID: \(id)")
        }
        vectors[id] = vector
    }

    /// Add with a nonnegative monotonic ID, skipping occupied bindings.
    /// Removal and removeAll do not rewind the cursor; exhaustion throws unchanged.
    @discardableResult
    public func add(vector: V) throws -> Int {
        guard vector.scalarCount == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: vector.scalarCount)
        }
        lock.lock()
        defer { lock.unlock() }
        guard var id = nextID else {
            throw VectorError(.resourceExhausted, message: "Generated vector IDs exhausted")
        }
        while vectors[id] != nil {
            let (successor, overflow) = id.addingReportingOverflow(1)
            guard !overflow else {
                throw VectorError(.resourceExhausted, message: "Generated vector IDs exhausted")
            }
            id = successor
        }
        vectors[id] = vector
        let (successor, overflow) = id.addingReportingOverflow(1)
        nextID = overflow ? nil : successor
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
