// VectorCore: Array Storage Implementation
//
// Generic storage for arbitrary dimension vectors
//

import Foundation

/// Generic array-based storage for arbitrary dimensions
public struct ArrayStorage<D: Dimension>: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var data: ContiguousArray<Float>
    
    public var count: Int { data.count }
    
    public init() {
        if D.self is DynamicDimension.Type {
            fatalError("DynamicDimension requires explicit size")
        }
        self.data = ContiguousArray(repeating: 0, count: D.value)
    }
    
    public init(repeating value: Float) {
        if D.self is DynamicDimension.Type {
            fatalError("DynamicDimension requires explicit size")
        }
        self.data = ContiguousArray(repeating: value, count: D.value)
    }
    
    public init(from values: [Float]) {
        if D.self is DynamicDimension.Type {
            self.data = ContiguousArray(values)
        } else {
            precondition(values.count == D.value, "Value count must match dimension")
            self.data = ContiguousArray(values)
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
        
        // Convert to arrays and use SIMD provider
        let a = Array(self.data)
        let b = Array(other.data)
        return Operations.simdProvider.dot(a, b)
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
        
        // Convert to arrays and use SIMD provider
        let a = Array(self.data)
        let b = Array(other.data)
        return Operations.simdProvider.dot(a, b)
    }
}