// VectorCore: Core Protocol Definitions
//
// Essential protocols for the vector store ecosystem
//

import Foundation

// MARK: - Distance Metric Protocol

/// Protocol for distance and similarity metrics between vectors.
///
/// `DistanceMetric` defines the interface for computing distances or
/// similarities between vectors. Implementations can provide various
/// metrics such as Euclidean, Cosine, Manhattan, etc.
///
/// ## Conforming Types
/// - `EuclideanDistance`: L2 norm distance
/// - `CosineDistance`: Angular distance (1 - cosine similarity)
/// - `ManhattanDistance`: L1 norm distance
/// - `DotProductDistance`: Negative dot product
///
/// ## Implementation Notes
/// - Distances should be non-negative (except DotProductDistance)
/// - Smaller values indicate more similar vectors
/// - Implementations should be optimized for performance
///
/// ## Example Usage
/// ```swift
/// let metric = EuclideanDistance()
/// let distance = metric.distance(vector1, vector2)
/// 
/// // Batch computation for efficiency
/// let distances = metric.batchDistance(query: query, candidates: vectors)
/// ```
public protocol DistanceMetric: Sendable {
    /// Unique identifier for this metric.
    ///
    /// Used for serialization, logging, and metric selection.
    /// Examples: "euclidean", "cosine", "manhattan"
    var identifier: String { get }
    
    /// Compute distance between two vectors.
    ///
    /// - Parameters:
    ///   - a: First vector
    ///   - b: Second vector
    /// - Returns: Distance score (typically non-negative)
    /// - Complexity: Usually O(n) where n is vector dimension
    func distance<Vector: ExtendedVectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore
    
    /// Efficiently compute distances from query to multiple candidates.
    ///
    /// Default implementation calls distance() for each candidate.
    /// Override for optimized batch processing.
    ///
    /// - Parameters:
    ///   - query: Query vector to compare against
    ///   - candidates: Array of vectors to compute distances to
    /// - Returns: Array of distances corresponding to each candidate
    /// - Complexity: O(n×m) where n is dimension, m is candidate count
    func batchDistance<Vector: ExtendedVectorProtocol>(query: Vector, candidates: [Vector]) -> [DistanceScore]
}

extension DistanceMetric {
    /// Default batch distance implementation
    public func batchDistance<Vector: ExtendedVectorProtocol>(query: Vector, candidates: [Vector]) -> [DistanceScore] {
        candidates.map { distance(query, $0) }
    }
}

// MARK: - Acceleration Provider Protocol

/// Protocol for hardware acceleration providers.
///
/// `AccelerationProvider` defines the interface for leveraging hardware
/// acceleration such as SIMD instructions, GPU compute, or neural engines
/// to optimize vector operations.
///
/// ## Implementation Examples
/// - Metal-based GPU acceleration
/// - Accelerate framework SIMD operations
/// - Core ML neural engine integration
/// - Custom hardware accelerators
///
/// ## Usage Pattern
/// ```swift
/// let accelerator = MetalAccelerator(configuration: .default)
/// if accelerator.isSupported(for: .matrixMultiplication) {
///     let result = try await accelerator.accelerate(.matrixMultiplication, input: matrices)
/// }
/// ```
public protocol AccelerationProvider: Sendable {
    /// Configuration type for the acceleration provider.
    ///
    /// Defines provider-specific settings such as device selection,
    /// memory limits, or optimization preferences.
    associatedtype Config
    
    /// Initialize acceleration provider with configuration.
    ///
    /// - Parameter configuration: Provider-specific configuration
    /// - Throws: If initialization fails (e.g., hardware unavailable)
    init(configuration: Config) async throws
    
    /// Check if a specific operation can be accelerated.
    ///
    /// - Parameter operation: The operation to check support for
    /// - Returns: true if the operation can be accelerated
    ///
    /// ## Example
    /// ```swift
    /// if provider.isSupported(for: .distanceComputation) {
    ///     // Use accelerated path
    /// } else {
    ///     // Fall back to CPU implementation
    /// }
    /// ```
    func isSupported(for operation: AcceleratedOperation) -> Bool
    
    /// Perform hardware-accelerated computation.
    ///
    /// - Parameters:
    ///   - operation: The operation to accelerate
    ///   - input: Operation-specific input data
    /// - Returns: Computed result of the same type as input
    /// - Throws: If acceleration fails or operation unsupported
    func accelerate<T>(_ operation: AcceleratedOperation, input: T) async throws -> T
}

/// Operations that can be accelerated
/// Types of operations that can be hardware-accelerated.
///
/// `AcceleratedOperation` identifies specific vector operations that can
/// benefit from hardware acceleration such as SIMD instructions, GPU compute,
/// or specialized neural engines. Used by `AccelerationProvider` to query
/// and optimize performance-critical operations.
///
/// ## Example Usage
/// ```swift
/// if accelerator.isOperationSupported(.distanceComputation) {
///     // Use accelerated path
/// } else {
///     // Fall back to standard implementation
/// }
/// ```
public enum AcceleratedOperation: String, Sendable {
    /// High-performance distance calculations between vectors.
    case distanceComputation
    
    /// Accelerated matrix-vector and matrix-matrix multiplications.
    case matrixMultiplication
    
    /// Fast vector normalization using SIMD square root operations.
    case vectorNormalization
    
    /// Batch processing of multiple vectors in parallel.
    case batchedOperations
}

// MARK: - Vector Serializable Protocol

/// Protocol for vector serialization to various formats.
///
/// `VectorSerializable` enables vectors to be converted to and from
/// different serialization formats for storage, transmission, or
/// interoperability with other systems.
///
/// ## Common Serialized Forms
/// - `Data`: Binary representation
/// - `String`: Text formats (JSON, Base64)
/// - Custom formats for specific use cases
///
/// ## Example Implementation
/// ```swift
/// extension Vector: VectorSerializable {
///     typealias SerializedForm = Data
///     
///     func serialize() throws -> Data {
///         return try encodeBinary()
///     }
///     
///     static func deserialize(from data: Data) throws -> Self {
///         return try decodeBinary(from: data)
///     }
/// }
/// ```
public protocol VectorSerializable {
    /// The type used for serialized representation.
    ///
    /// Common choices include Data (binary), String (text), or custom types.
    associatedtype SerializedForm
    
    /// Serialize the vector to its serialized form.
    ///
    /// - Returns: Serialized representation of the vector
    /// - Throws: If serialization fails
    ///
    /// ## Example
    /// ```swift
    /// let data = try vector.serialize()
    /// saveToFile(data)
    /// ```
    func serialize() throws -> SerializedForm
    
    /// Deserialize a vector from its serialized form.
    ///
    /// - Parameter from: Serialized representation to decode
    /// - Returns: Reconstructed vector instance
    /// - Throws: If deserialization fails or data is invalid
    ///
    /// ## Example
    /// ```swift
    /// let data = loadFromFile()
    /// let vector = try Vector.deserialize(from: data)
    /// ```
    static func deserialize(from: SerializedForm) throws -> Self
}

// MARK: - Binary Serialization Extensions

/// Default implementation for vectors that are binary encodable
extension VectorType where Self: BinaryEncodable {
    public func encodeBinary() throws -> Data {
        // Validate dimension
        guard self.scalarCount > 0 && self.scalarCount <= Int(BinaryFormat.maxDimension) else {
            throw VectorError.invalidDimension(
                self.scalarCount,
                reason: "Must be between 1 and \(BinaryFormat.maxDimension)"
            )
        }
        
        // Pre-allocate exact size needed
        let totalSize = BinaryFormat.expectedDataSize(for: self.scalarCount)
        var data = Data()
        data.reserveCapacity(totalSize)
        
        // Create header
        let header = BinaryHeader(dimension: self.scalarCount)
        
        // Write header in little-endian format
        BinaryFormat.writeUInt32(header.magic, to: &data)
        BinaryFormat.writeUInt16(header.version, to: &data)
        BinaryFormat.writeUInt32(header.dimension, to: &data)
        BinaryFormat.writeUInt16(header.flags, to: &data)
        
        // Write vector data in little-endian format
        let values = self.toArray()
        BinaryFormat.writeFloatArray(values, to: &data)
        
        // Calculate and append checksum
        let checksum = data.crc32()
        BinaryFormat.writeUInt32(checksum, to: &data)
        
        return data
    }
}