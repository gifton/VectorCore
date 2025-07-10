// VectorCore: Core Vector Data Types
//
// Fundamental types for vector operations, designed to be 
// dependency-free and usable across all platforms

import Foundation
import simd

// MARK: - Vector Data

/// Core vector data structure without storage-specific concerns
/// 
/// This type represents the essential vector data and metadata,
/// excluding storage-specific fields that were previously in VectorEntry.
/// Storage-related metadata is now handled by VectorDB.StoredVector.
public struct VectorData<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: Codable, Sendable 
where Vector.Scalar: BinaryFloatingPoint {
    
    // MARK: - Core Properties
    
    /// Unique identifier for this vector
    public let id: VectorID
    
    /// The vector data with SIMD optimization
    public let vector: Vector
    
    /// Associated metadata with type safety
    public let metadata: Metadata
    
    /// Creation timestamp
    public let timestamp: Timestamp
    
    // MARK: - Initialization
    
    public init(
        id: VectorID,
        vector: Vector,
        metadata: Metadata,
        timestamp: Timestamp = Timestamp(Date().timeIntervalSince1970 * 1_000_000_000)
    ) {
        self.id = id
        self.vector = vector
        self.metadata = metadata
        self.timestamp = timestamp
    }
}

// MARK: - Vector Quality

/// Basic vector quality metrics for validation and optimization
public struct VectorQualityMetrics: Codable, Sendable {
    /// Magnitude of the vector
    public let magnitude: Float
    
    /// Sparsity ratio (0.0 = dense, 1.0 = all zeros)
    public let sparsity: Float
    
    /// Entropy measure for information content
    public let entropy: Float
    
    /// Statistical variance
    public let variance: Float
    
    public init(magnitude: Float, sparsity: Float, entropy: Float, variance: Float) {
        self.magnitude = magnitude
        self.sparsity = sparsity
        self.entropy = entropy
        self.variance = variance
    }
}

// MARK: - Common Vector Type Aliases

/// Standard SIMD vector types
public typealias Vector32 = SIMD32<Float>
public typealias Vector64 = SIMD64<Float>

// Note: Swift's SIMD types only go up to SIMD64
// Custom implementations are provided in Math/VectorTypes.swift
// and Math/Vector512.swift for larger dimensions