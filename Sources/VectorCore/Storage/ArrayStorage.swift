// VectorCore: Array Storage Implementation
//
// Generic storage for arbitrary dimension vectors
//

import Foundation

/// Generic array-based storage for arbitrary dimensions
public struct ArrayStorage<D>: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var data: ContiguousArray<Float>
    
    public var count: Int { data.count }
    
    public init() {
        if D.self == DynamicDimension.self {
            fatalError("DynamicDimension requires explicit size")
        }
        if let dimensionType = D.self as? any StaticDimension.Type {
            self.data = ContiguousArray(repeating: 0, count: dimensionType.value)
        } else {
            fatalError("Unknown dimension type")
        }
    }
    
    public init(repeating value: Float) {
        if D.self == DynamicDimension.self {
            fatalError("DynamicDimension requires explicit size")
        }
        if let dimensionType = D.self as? any StaticDimension.Type {
            self.data = ContiguousArray(repeating: value, count: dimensionType.value)
        } else {
            fatalError("Unknown dimension type")
        }
    }
    
    public init(from values: [Float]) {
        if D.self == DynamicDimension.self {
            self.data = ContiguousArray(values)
        } else if let dimensionType = D.self as? any StaticDimension.Type {
            precondition(values.count == dimensionType.value, "Value count must match dimension")
            self.data = ContiguousArray(values)
        } else {
            fatalError("Unknown dimension type")
        }
    }
    
    /// Special initializer for dynamic dimensions
    public init(dimension: Int) {
        self.data = ContiguousArray(repeating: 0, count: dimension)
    }
    
    /// Special initializer for dynamic dimensions with value
    public init(dimension: Int, repeating value: Float) {
        self.data = ContiguousArray(repeating: value, count: dimension)
    }
    
    @inlinable
    public subscript(index: Int) -> Float {
        get { data[index] }
        set { data[index] = newValue }
    }
    
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try data.withUnsafeBufferPointer(body)
    }
    
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try data.withUnsafeMutableBufferPointer { buffer in
            try body(buffer)
        }
    }
    
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        precondition(count == other.count, "Dimensions must match")
        
        var result: Float = 0
        self.withUnsafeBufferPointer { aBuffer in
            other.withUnsafeBufferPointer { bBuffer in
                result = SIMDOperations.FloatProvider.dot(
                    aBuffer.baseAddress!,
                    bBuffer.baseAddress!,
                    count: count
                )
            }
        }
        return result
    }
}

// MARK: - Specialized Array Storage for Dynamic Dimensions

/// Specialized storage for runtime-determined dimensions
public final class DynamicArrayStorage: @unchecked Sendable, VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var data: ContiguousArray<Float>
    
    public var count: Int { data.count }
    
    public init(dimension: Int) {
        self.data = ContiguousArray(repeating: 0, count: dimension)
    }
    
    public required init() {
        fatalError("Dynamic storage requires explicit dimension")
    }
    
    public init(repeating value: Float) {
        fatalError("Dynamic storage requires explicit dimension")
    }
    
    public init(dimension: Int, repeating value: Float) {
        self.data = ContiguousArray(repeating: value, count: dimension)
    }
    
    public init(from values: [Float]) {
        self.data = ContiguousArray(values)
    }
    
    @inlinable
    public subscript(index: Int) -> Float {
        get { data[index] }
        set { data[index] = newValue }
    }
    
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try data.withUnsafeBufferPointer(body)
    }
    
    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try data.withUnsafeMutableBufferPointer { buffer in
            try body(buffer)
        }
    }
    
    @inlinable
    public func dotProduct(_ other: DynamicArrayStorage) -> Float {
        precondition(count == other.count, "Dimensions must match")
        
        var result: Float = 0
        self.withUnsafeBufferPointer { aBuffer in
            other.withUnsafeBufferPointer { bBuffer in
                result = SIMDOperations.FloatProvider.dot(
                    aBuffer.baseAddress!,
                    bBuffer.baseAddress!,
                    count: count
                )
            }
        }
        return result
    }
}