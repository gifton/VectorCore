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

    /// Normalize vectors to unit length
    internal static func normalize<V: VectorProtocol & VectorFactory>(
        _ vectors: [V],
        context: any ExecutionContext = CPUContext.automatic
    ) async throws -> [V] where V.Scalar == Float {

        if vectors.count < parallelThreshold {
            // Sequential normalization
            return try await context.execute {
                try vectors.map { vector in
                    let magnitude = vector.magnitude
                    guard magnitude > Float.ulpOfOne else {
                        // Return zero vector for zero magnitude
                        return try V.create(from: Array(repeating: 0, count: vector.scalarCount))
                    }
                    return try V.createByTransforming(vector) { $0 / magnitude }
                }
            }
        }

        // Parallel normalization
        return try await withThrowingTaskGroup(of: (Int, V).self) { group in
            for (index, vector) in vectors.enumerated() {
                group.addTask {
                    let magnitude = vector.magnitude
                    let normalized: V
                    if magnitude > Float.ulpOfOne {
                        normalized = try V.createByTransforming(vector) { $0 / magnitude }
                    } else {
                        normalized = try V.create(from: Array(repeating: 0, count: vector.scalarCount))
                    }
                    return (index, normalized)
                }
            }

            var results = [V?](repeating: nil, count: vectors.count)
            for try await (index, result) in group {
                results[index] = result
            }
            return results.compactMap { $0 }
        }
    }

    /// Apply element-wise function to vector pairs
    internal static func combine<V: VectorProtocol & VectorFactory>(
        _ vectors1: [V],
        _ vectors2: [V],
        _ operation: @Sendable @escaping (Float, Float) -> Float,
        context: any ExecutionContext = CPUContext.automatic
    ) async throws -> [V] where V.Scalar == Float {

        guard vectors1.count == vectors2.count else {
            throw VectorError.invalidDimension(
                vectors2.count,
                reason: "Vector arrays must have same count: \(vectors1.count) != \(vectors2.count)"
            )
        }

        // Validate dimensions match
        if let first1 = vectors1.first, let first2 = vectors2.first {
            guard first1.scalarCount == first2.scalarCount else {
                throw VectorError.dimensionMismatch(
                    expected: first1.scalarCount,
                    actual: first2.scalarCount
                )
            }
        }

        if vectors1.count < parallelThreshold {
            // Sequential combination
            return try await context.execute {
                try zip(vectors1, vectors2).map { v1, v2 in
                    try V.createByCombining(v1, v2, operation)
                }
            }
        }

        // Parallel combination
        return try await withThrowingTaskGroup(of: (Int, V).self) { group in
            for (index, (v1, v2)) in zip(vectors1, vectors2).enumerated() {
                group.addTask {
                    let result = try V.createByCombining(v1, v2, operation)
                    return (index, result)
                }
            }

            var results = [V?](repeating: nil, count: vectors1.count)
            for try await (index, result) in group {
                results[index] = result
            }
            return results.compactMap { $0 }
        }
    }
}

// MARK: - Convenience Extensions

public extension Array where Element: VectorProtocol & VectorFactory, Element.Scalar == Float {

    /// Transform all vectors in the array
    func mapElements(_ transform: @Sendable @escaping (Float) -> Float) async throws -> [Element] {
        try await ExecutionOperations.map(self, transform: transform)
    }

    /// Normalize all vectors to unit length
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

    /// Compute cosine similarity matrix
    func cosineSimilarityMatrix() async throws -> [[Float]] {
        let normalized = try await self.normalized()
        return try await ExecutionOperations.distanceMatrix(
            normalized,
            metric: DotProductDistance() // Negative dot product on normalized vectors = cosine distance
        )
    }
}

// MARK: - Example Usage

/*
 // Works with both static and dynamic vectors:

 let staticVectors: [Vector<Dim128>] = [...]
 let dynamicVectors: [DynamicVector] = [...]

 // Transform operations preserve type
 let normalizedStatic = try await staticVectors.normalized()  // Returns [Vector<Dim128>]
 let normalizedDynamic = try await dynamicVectors.normalized() // Returns [DynamicVector]

 // Element-wise operations
 let scaled = try await staticVectors.multiplying(by: 2.0)
 let shifted = try await dynamicVectors.adding(1.0)

 // Combine vectors
 let sum = try await ExecutionOperations.combine(vectors1, vectors2, +)
 let diff = try await ExecutionOperations.combine(vectors1, vectors2, -)
 */
