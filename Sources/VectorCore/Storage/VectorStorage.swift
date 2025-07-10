// VectorCore: Vector Storage Protocol
//
// Defines the storage abstraction for vectors
//

import Foundation
import Accelerate

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
    /// Default dot product implementation using vDSP
    public func dotProduct(_ other: Self) -> Float {
        var result: Float = 0
        self.withUnsafeBufferPointer { a in
            other.withUnsafeBufferPointer { b in
                vDSP_dotpr(a.baseAddress!, 1, b.baseAddress!, 1, &result, vDSP_Length(count))
            }
        }
        return result
    }
}