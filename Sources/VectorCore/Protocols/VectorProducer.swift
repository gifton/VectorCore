// VectorCore - VectorProducer Protocol
// Defines the interface for embedding producers

import Foundation

/// Protocol for types that produce vector embeddings from text
/// Enables cross-package interoperability between EmbedKit and VectorIndex
public protocol VectorProducer: Sendable {
    /// The dimensionality of produced embeddings
    var dimensions: Int { get }

    /// Whether the producer normalizes output vectors
    var producesNormalizedVectors: Bool { get }

    /// Produce embeddings for a batch of texts
    func produce(_ texts: [String]) async throws -> [[Float]]

    /// Produce embedding for a single text
    func produce(_ text: String) async throws -> [Float]
}

// MARK: - Default Implementations

public extension VectorProducer {
    /// Default single-text implementation delegates to batch
    func produce(_ text: String) async throws -> [Float] {
        let results = try await produce([text])
        guard let first = results.first else {
            throw VectorError.invalidOperation(
                "produce(_:)",
                reason: "Producer returned empty results"
            )
        }
        return first
    }
}

/// Hints for vector consumers about producer characteristics
public struct VectorProducerHints: Sendable, Equatable {
    public let dimensions: Int
    public let isNormalized: Bool
    public let optimalBatchSize: Int
    public let maxBatchSize: Int

    public init(
        dimensions: Int,
        isNormalized: Bool = true,
        optimalBatchSize: Int = 32,
        maxBatchSize: Int = 128
    ) {
        self.dimensions = dimensions
        self.isNormalized = isNormalized
        self.optimalBatchSize = optimalBatchSize
        self.maxBatchSize = maxBatchSize
    }
}
