// VectorCore: Vector Storage Protocol
//
// Defines the storage abstraction for vectors
//

import Foundation

/// Protocol defining the storage interface for vectors
public protocol VectorStorage: Sendable {
    associatedtype Scalar: BinaryFloatingPoint

    /// Initialize with zeros
    init()

    /// Initialize with a repeating value
    init(repeating value: Scalar)

    /// Initialize from an array of values
    init(from values: [Scalar])

    /// Access elements by index
    subscript(index: Int) -> Scalar { get set }

    /// Number of elements in the storage
    var count: Int { get }

    /// Access the underlying buffer for optimized operations
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Scalar>) throws -> R) rethrows -> R

    /// Mutate the underlying buffer
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R) rethrows -> R
}

/// Protocol for storage types that support optimized operations
public protocol VectorStorageOperations: VectorStorage {
    /// Compute dot product with another storage
    func dotProduct(_ other: Self) -> Scalar
}

// MARK: - Storage Operations Extension

extension VectorStorageOperations where Scalar == Float {
    /// Default dot product implementation using SwiftSIMDProvider
    public func dotProduct(_ other: Self) -> Float {
        var result: Float = 0
        self.withUnsafeBufferPointer { aBuffer in
            other.withUnsafeBufferPointer { bBuffer in
                result = SIMDOperations.FloatProvider.dot(
                    aBuffer.baseAddress!,
                    bBuffer.baseAddress!,
                    count: aBuffer.count
                )
            }
        }
        return result
    }
}
