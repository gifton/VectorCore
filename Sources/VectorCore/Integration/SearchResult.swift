//
//  SearchResult.swift
//  VectorCore
//
//  Standardized result types for vector search operations.
//  Used by VectorIndex and other downstream packages.
//

import Foundation

// MARK: - Single Search Result

/// A single search result with distance and optional metadata.
///
/// Represents one match from a k-nearest neighbor search, containing
/// the identifier of the matched vector and its distance from the query.
public struct SearchResult<ID: Hashable & Sendable>: Sendable, Equatable {
    /// Identifier of the matched vector
    public let id: ID

    /// Distance from query (lower is more similar for distance metrics)
    public let distance: Float

    /// Optional relevance score (higher is better, typically 1/(1+distance) or similar)
    public let score: Float?

    /// Initialize with required fields
    public init(id: ID, distance: Float, score: Float? = nil) {
        self.id = id
        self.distance = distance
        self.score = score
    }

    /// Compute relevance score from distance using inverse transform.
    /// Returns a value in (0, 1] where 1 is perfect match (distance = 0).
    public var computedScore: Float {
        score ?? (1.0 / (1.0 + distance))
    }
}

// MARK: - Search Results Collection

/// Collection of search results with metadata about the search operation.
///
/// Provides structured access to k-nearest neighbor results along with
/// statistics about the search process.
public struct SearchResults<ID: Hashable & Sendable>: Sendable {
    /// Ordered results (closest first)
    public let results: [SearchResult<ID>]

    /// Total number of candidates searched
    public let candidatesSearched: Int

    /// Time taken for search in nanoseconds (optional)
    public let searchTimeNanos: UInt64?

    /// Whether the search was exhaustive (vs approximate)
    public let isExhaustive: Bool

    /// Number of results returned
    public var count: Int { results.count }

    /// Whether the result set is empty
    public var isEmpty: Bool { results.isEmpty }

    /// The best (closest) result, if any
    public var best: SearchResult<ID>? { results.first }

    /// Initialize with all fields
    public init(
        results: [SearchResult<ID>],
        candidatesSearched: Int,
        searchTimeNanos: UInt64? = nil,
        isExhaustive: Bool = true
    ) {
        self.results = results
        self.candidatesSearched = candidatesSearched
        self.searchTimeNanos = searchTimeNanos
        self.isExhaustive = isExhaustive
    }

    /// Initialize empty results
    public init() {
        self.results = []
        self.candidatesSearched = 0
        self.searchTimeNanos = nil
        self.isExhaustive = true
    }

    /// Search time in milliseconds (if available)
    public var searchTimeMs: Double? {
        searchTimeNanos.map { Double($0) / 1_000_000.0 }
    }

    /// IDs of all matched vectors
    public var ids: [ID] {
        results.map { $0.id }
    }

    /// Distances of all matched vectors
    public var distances: [Float] {
        results.map { $0.distance }
    }

    /// Map results to a different ID type
    public func mapIDs<NewID: Hashable & Sendable>(_ transform: (ID) -> NewID) -> SearchResults<NewID> {
        SearchResults<NewID>(
            results: results.map { SearchResult(id: transform($0.id), distance: $0.distance, score: $0.score) },
            candidatesSearched: candidatesSearched,
            searchTimeNanos: searchTimeNanos,
            isExhaustive: isExhaustive
        )
    }
}

// MARK: - Collection Conformance

extension SearchResults: Collection {
    public typealias Index = Int
    public typealias Element = SearchResult<ID>

    public var startIndex: Int { 0 }
    public var endIndex: Int { count }

    public subscript(position: Int) -> SearchResult<ID> {
        results[position]
    }

    public func index(after i: Int) -> Int { i + 1 }
}

// MARK: - Codable Conformance

extension SearchResult: Codable where ID: Codable {}

extension SearchResults: Codable where ID: Codable {}

// MARK: - Convenience Type Aliases

/// Search result with integer IDs (most common case)
public typealias IntSearchResult = SearchResult<Int>

/// Search results with integer IDs
public typealias IntSearchResults = SearchResults<Int>

/// Search result with string IDs (for named vectors)
public typealias StringSearchResult = SearchResult<String>

/// Search results with string IDs
public typealias StringSearchResults = SearchResults<String>

// MARK: - Conversion from TopKResult

extension SearchResults where ID == Int {
    /// Create SearchResults from a TopKResult.
    ///
    /// - Parameters:
    ///   - topK: The TopKResult to convert
    ///   - candidatesSearched: Total candidates that were searched
    ///   - searchTimeNanos: Optional search duration
    ///   - isExhaustive: Whether search was exhaustive
    public init(
        from topK: TopKResult,
        candidatesSearched: Int,
        searchTimeNanos: UInt64? = nil,
        isExhaustive: Bool = true
    ) {
        let results = zip(topK.indices, topK.distances).map { index, distance in
            SearchResult(id: index, distance: distance)
        }
        self.init(
            results: results,
            candidatesSearched: candidatesSearched,
            searchTimeNanos: searchTimeNanos,
            isExhaustive: isExhaustive
        )
    }
}

// MARK: - Hashable Conformance

extension SearchResult: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(distance)
    }
}

// MARK: - Comparable by Distance

extension SearchResult: Comparable {
    public static func < (lhs: SearchResult<ID>, rhs: SearchResult<ID>) -> Bool {
        lhs.distance < rhs.distance
    }
}
