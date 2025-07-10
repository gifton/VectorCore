// VectorCore: Core Protocol Definitions
//
// Essential protocols for the vector store ecosystem
//

import Foundation

// MARK: - Distance Metric Protocol

/// Protocol for distance/similarity metrics
public protocol DistanceMetric: Sendable {
    /// Unique identifier for this metric
    var identifier: String { get }
    
    /// Compute distance between two vectors
    func distance<Vector: ExtendedVectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore
    
    /// Batch compute distances (default implementation provided)
    func batchDistance<Vector: ExtendedVectorProtocol>(query: Vector, candidates: [Vector]) -> [DistanceScore]
}

extension DistanceMetric {
    /// Default batch distance implementation
    public func batchDistance<Vector: ExtendedVectorProtocol>(query: Vector, candidates: [Vector]) -> [DistanceScore] {
        candidates.map { distance(query, $0) }
    }
}

// MARK: - Acceleration Provider Protocol

/// Protocol for hardware acceleration providers
public protocol AccelerationProvider: Sendable {
    associatedtype Config
    
    /// Initialize with configuration
    init(configuration: Config) async throws
    
    /// Check if operation is supported
    func isSupported(for operation: AcceleratedOperation) -> Bool
    
    /// Accelerate an operation
    func accelerate<T>(_ operation: AcceleratedOperation, input: T) async throws -> T
}

/// Operations that can be accelerated
public enum AcceleratedOperation: String, Sendable {
    case distanceComputation
    case matrixMultiplication
    case vectorNormalization
    case batchedOperations
}

// MARK: - Vector Serializable Protocol

/// Protocol for vector serialization
public protocol VectorSerializable {
    associatedtype SerializedForm
    
    /// Serialize the vector
    func serialize() throws -> SerializedForm
    
    /// Deserialize from serialized form
    static func deserialize(from: SerializedForm) throws -> Self
}