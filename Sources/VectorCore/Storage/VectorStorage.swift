// VectorCore: Vector Storage Protocol
//
// Defines the storage abstraction for vectors
//

import Foundation

/// Protocol defining the storage interface for vectors
public protocol VectorStorage: Sendable {
    /// Initialize with zeros
    init()
    
    /// Initialize with a repeating value
    init(repeating value: Float)
    
    /// Initialize from an array of values
    init(from values: [Float])
    
    /// Access elements by index
    subscript(index: Int) -> Float { get set }
    
    /// Number of elements in the storage
    var count: Int { get }
    
    /// Access the underlying buffer for optimized operations
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R
    
    /// Mutate the underlying buffer
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R
}

/// Protocol for storage types that support optimized operations
public protocol VectorStorageOperations: VectorStorage {
    /// Compute dot product with another storage
    func dotProduct(_ other: Self) -> Float
}

// MARK: - Storage Operations Extension

extension VectorStorageOperations {
    /// Default dot product implementation using SwiftSIMDProvider
    public func dotProduct(_ other: Self) -> Float {
        // Convert to arrays and use SIMD provider
        let a = self.withUnsafeBufferPointer { Array($0) }
        let b = other.withUnsafeBufferPointer { Array($0) }
        return Operations.simdProvider.dot(a, b)
    }
}