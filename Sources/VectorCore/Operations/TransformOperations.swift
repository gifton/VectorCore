//
//  TransformOperations.swift
//  VectorCore
//
//  Additional vector transformation operations using VectorFactory
//

import Foundation
import Accelerate

// MARK: - Transform Operations

// Intentionally omit re-declaring Operations.* here to avoid duplication.

// MARK: - Convenience Extensions

public extension Array where Element: VectorProtocol & VectorFactory, Element.Scalar == Float {
    
    /// Transform all vectors in the array
    func mapElements(_ transform: @Sendable @escaping (Float) -> Float) async throws -> [Element] {
        try await Operations.map(self, transform: transform)
    }
    
    /// Normalize all vectors to unit length
    func normalized() async throws -> [Element] {
        try await Operations.normalize(self)
    }
    
    /// Add scalar to all elements
    func adding(_ scalar: Float) async throws -> [Element] {
        try await Operations.add(self, scalar)
    }
    
    /// Multiply all elements by scalar
    func multiplying(by scalar: Float) async throws -> [Element] {
        try await Operations.multiply(self, by: scalar)
    }
}

// MARK: - Static Vector Specific Operations

public extension Array where Element == Vector<Dim768> {
    /// Specialized operations for 768-dimensional vectors (common in embeddings)
    
    /// Compute cosine similarity matrix
    func cosineSimilarityMatrix() async throws -> [[Float]] {
        let normalized = try await self.normalized()
        return try await Operations.distanceMatrix(
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
