//
//  TransformOperations.swift
//  VectorCore
//
//  Additional vector transformation operations using VectorFactory
//

import Foundation
import Accelerate

// MARK: - Transform Operations

extension ExecutionOperations {

    /// Element-wise addition with automatic type preservation
    @inlinable
    internal static func add<V: VectorProtocol & VectorFactory>(
        _ vectors: [V],
        _ scalar: V.Scalar,
        context: any ExecutionContext = CPUContext.automatic
    ) async throws -> [V] where V.Scalar == Float {
        try await map(vectors, transform: { $0 + scalar }, context: context)
    }

    /// Element-wise multiplication with automatic type preservation
    @inlinable
    internal static func multiply<V: VectorProtocol & VectorFactory>(
        _ vectors: [V],
        by scalar: V.Scalar,
        context: any ExecutionContext = CPUContext.automatic
    ) async throws -> [V] where V.Scalar == Float {
        try await map(vectors, transform: { $0 * scalar }, context: context)
    }

    /// Normalize finite nonzero rows; preserve empty/zero/nonfinite rows through the factory.
    internal static func normalize<V: VectorProtocol & VectorFactory>(
        _ vectors: [V],
        context: any ExecutionContext = CPUContext.automatic
    ) async throws -> [V] where V.Scalar == Float {
        try await BoundedExecution.map(0..<vectors.count, limit: context.maxThreadCount,
            parallel: vectors.count >= parallelThreshold, minimum: context.preferredChunkSize,
            phase: "transform.normalize") { index in
            let output = NormalizationCore.permissive(vectors[index].toArray())
            return try V.create(from: output)
        }
    }

}

// MARK: - Convenience Extensions

public extension Array where Element: VectorProtocol & VectorFactory, Element.Scalar == Float {

    /// Transform all vectors in the array
    func mapElements(_ transform: @Sendable @escaping (Float) -> Float) async throws -> [Element] {
        try await ExecutionOperations.map(self, transform: transform)
    }

    /// Normalize finite nonzero rows; empty/zero/nonfinite rows remain unchanged.
    func normalized() async throws -> [Element] {
        try await ExecutionOperations.normalize(self)
    }

    /// Add scalar to all elements
    func adding(_ scalar: Float) async throws -> [Element] {
        try await ExecutionOperations.add(self, scalar)
    }

    /// Multiply all elements by scalar
    func multiplying(by scalar: Float) async throws -> [Element] {
        try await ExecutionOperations.multiply(self, by: scalar)
    }
}

// MARK: - Static Vector Specific Operations

public extension Array where Element == Vector<Dim768> {
    /// Specialized operations for 768-dimensional vectors (common in embeddings)

    /// Compute cosine similarities, with zero similarity for pairs involving a zero row.
    /// Finite rounding excursions are clamped to [-1, 1]; a non-finite score (a row with a NaN or
    /// infinite coordinate) gives NaN (T15CosineExceptionalTableTests.transformSimilarityMatrix).
    func cosineSimilarityMatrix() async throws -> [[Float]] {
        let normalized = try await self.normalized()
        let scores = try await ExecutionOperations.distanceMatrix(
            normalized,
            metric: DotProductDistance()
        )
        return scores.map { row in
            row.map { score in
                let similarity = -score
                return similarity.isFinite ? Swift.max(-1, Swift.min(1, similarity)) : .nan
            }
        }
    }
}
