// VectorCore: Core Vector Data Types
//
// Fundamental types for vector operations, designed to be 
// dependency-free and usable across all platforms

import Foundation
import simd

// MARK: - Vector Data

/// Core vector data structure that combines a vector with its metadata.
///
/// `VectorData` provides a type-safe container for vectors with associated metadata,
/// optimized for SIMD operations. This structure represents the essential vector
/// information without storage-specific concerns, making it suitable for use
/// across different storage backends.
///
/// ## Type Parameters
/// - `Vector`: A SIMD-optimized vector type conforming to `SIMD` and `Sendable`
/// - `Metadata`: Custom metadata type that must be `Codable` and `Sendable`
///
/// ## Example Usage
/// ```swift
/// struct ImageMetadata: Codable, Sendable {
///     let filename: String
///     let dimensions: (width: Int, height: Int)
/// }
///
/// let vectorData = VectorData(
///     id: VectorID("img-001"),
///     vector: SIMD32<Float>(repeating: 0.5),
///     metadata: ImageMetadata(filename: "cat.jpg", dimensions: (224, 224))
/// )
/// ```
///
/// - Note: Storage-related metadata is handled separately by storage implementations
///   (e.g., `VectorDB.StoredVector`), keeping this type focused on core vector data.
public struct VectorData<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: Codable, Sendable 
where Vector.Scalar: BinaryFloatingPoint {
    
    // MARK: - Core Properties
    
    /// Unique identifier for this vector.
    ///
    /// Used to reference and retrieve specific vectors from storage.
    public let id: VectorID
    
    /// The vector data with SIMD optimization.
    ///
    /// Contains the actual numerical data optimized for high-performance
    /// mathematical operations using SIMD instructions.
    public let vector: Vector
    
    /// Associated metadata with type safety.
    ///
    /// Can contain any application-specific information related to the vector,
    /// such as source data, labels, or processing parameters.
    public let metadata: Metadata
    
    /// Creation timestamp in nanoseconds since epoch.
    ///
    /// Automatically set to the current time if not specified during initialization.
    /// Useful for temporal queries and data versioning.
    public let timestamp: Timestamp
    
    // MARK: - Initialization
    
    /// Creates a new vector data instance with the specified components.
    ///
    /// - Parameters:
    ///   - id: Unique identifier for the vector
    ///   - vector: The SIMD-optimized vector data
    ///   - metadata: Associated metadata for the vector
    ///   - timestamp: Creation timestamp (defaults to current time in nanoseconds)
    ///
    /// - Note: The timestamp parameter defaults to nanosecond precision for
    ///   compatibility with high-frequency data collection scenarios.
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

/// Metrics for assessing vector quality and characteristics.
///
/// `VectorQualityMetrics` provides a comprehensive set of measurements
/// to evaluate vector quality, useful for validation, optimization,
/// and understanding vector properties in machine learning applications.
///
/// ## Metrics Explained
/// - **Magnitude**: L2 norm of the vector, indicates overall scale
/// - **Sparsity**: Proportion of zero or near-zero elements (0.0 = dense, 1.0 = all zeros)
/// - **Entropy**: Information content measure, higher values indicate more randomness
/// - **Variance**: Statistical spread of values, indicates diversity of components
///
/// ## Example Usage
/// ```swift
/// let metrics = VectorQualityMetrics(
///     magnitude: 1.0,      // Unit vector
///     sparsity: 0.1,       // 10% sparse
///     entropy: 0.95,       // High information content
///     variance: 0.25       // Moderate spread
/// )
///
/// // Use for quality checks
/// if metrics.sparsity > 0.9 {
///     print("Warning: Vector is highly sparse")
/// }
/// ```
public struct VectorQualityMetrics: Codable, Sendable {
    /// Magnitude (L2 norm) of the vector.
    ///
    /// Indicates the overall scale of the vector. Useful for:
    /// - Detecting zero or near-zero vectors
    /// - Normalization requirements
    /// - Scale consistency checks
    public let magnitude: Float
    
    /// Sparsity ratio indicating the proportion of zero elements.
    ///
    /// - Range: [0.0, 1.0]
    /// - 0.0 = fully dense (no zeros)
    /// - 1.0 = all zeros
    /// - Typical embeddings: < 0.1
    ///
    /// High sparsity may indicate:
    /// - Inefficient representation
    /// - Potential for compression
    /// - Initialization issues
    public let sparsity: Float
    
    /// Shannon entropy measure for information content.
    ///
    /// Higher values indicate more randomness/information:
    /// - Low entropy: Vector has redundant or predictable patterns
    /// - High entropy: Vector components are diverse
    ///
    /// Useful for detecting:
    /// - Degenerate embeddings
    /// - Over-regularization effects
    /// - Information loss
    public let entropy: Float
    
    /// Statistical variance of vector components.
    ///
    /// Measures the spread of values:
    /// - Low variance: Components are similar
    /// - High variance: Components are diverse
    ///
    /// Applications:
    /// - Detecting collapsed representations
    /// - Assessing feature diversity
    /// - Identifying outlier components
    public let variance: Float
    
    /// Creates a new set of vector quality metrics.
    ///
    /// - Parameters:
    ///   - magnitude: L2 norm of the vector
    ///   - sparsity: Ratio of zero elements (0.0 to 1.0)
    ///   - entropy: Information content measure
    ///   - variance: Statistical variance of components
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