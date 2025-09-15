// VectorCore: Dimension-Specific Storage Wrappers
//
// Wrapper types that bind storage to specific dimensions for Vector<D> compatibility
//
// These types provide compile-time dimension safety by wrapping AlignedValueStorage
// with fixed sizes. This ensures type safety while reusing the same underlying
// storage implementation.
//

import Foundation

// MARK: - Storage for Dim2

/// Storage specifically for 2-dimensional vectors
public struct Storage2: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: AlignedValueStorage

    public typealias Scalar = Float
    public var count: Int { 2 }

    @inlinable
    public init() {
        self.storage = AlignedValueStorage(count: 2)
    }

    @inlinable
    public init(repeating value: Float) {
        self.storage = AlignedValueStorage(count: 2, repeating: value)
    }

    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 2, "Storage2 requires exactly 2 values")
        self.storage = AlignedValueStorage(from: values)
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }
}

// MARK: - Storage for Dim3

/// Storage specifically for 3-dimensional vectors
public struct Storage3: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: AlignedValueStorage

    public typealias Scalar = Float
    public var count: Int { 3 }

    @inlinable
    public init() {
        self.storage = AlignedValueStorage(count: 3)
    }

    @inlinable
    public init(repeating value: Float) {
        self.storage = AlignedValueStorage(count: 3, repeating: value)
    }

    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 3, "Storage3 requires exactly 3 values")
        self.storage = AlignedValueStorage(from: values)
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }
}

// MARK: - Storage for Dim4

/// Storage specifically for 4-dimensional vectors
public struct Storage4: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: AlignedValueStorage

    public typealias Scalar = Float
    public var count: Int { 4 }

    @inlinable
    public init() {
        self.storage = AlignedValueStorage(count: 4)
    }

    @inlinable
    public init(repeating value: Float) {
        self.storage = AlignedValueStorage(count: 4, repeating: value)
    }

    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 4, "Storage4 requires exactly 4 values")
        self.storage = AlignedValueStorage(from: values)
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }
}

// MARK: - Storage for Dim8

/// Storage specifically for 8-dimensional vectors
public struct Storage8: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: AlignedValueStorage

    public typealias Scalar = Float
    public var count: Int { 8 }

    @inlinable
    public init() {
        self.storage = AlignedValueStorage(count: 8)
    }

    @inlinable
    public init(repeating value: Float) {
        self.storage = AlignedValueStorage(count: 8, repeating: value)
    }

    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 8, "Storage8 requires exactly 8 values")
        self.storage = AlignedValueStorage(from: values)
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }
}

// MARK: - Storage for Dim16

/// Storage specifically for 16-dimensional vectors
public struct Storage16: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: AlignedValueStorage

    public typealias Scalar = Float
    public var count: Int { 16 }

    @inlinable
    public init() {
        self.storage = AlignedValueStorage(count: 16)
    }

    @inlinable
    public init(repeating value: Float) {
        self.storage = AlignedValueStorage(count: 16, repeating: value)
    }

    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 16, "Storage16 requires exactly 16 values")
        self.storage = AlignedValueStorage(from: values)
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }
}

// MARK: - Storage for Dim32

/// Storage specifically for 32-dimensional vectors
public struct Storage32: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: AlignedValueStorage

    public typealias Scalar = Float
    public var count: Int { 32 }

    @inlinable
    public init() {
        self.storage = AlignedValueStorage(count: 32)
    }

    @inlinable
    public init(repeating value: Float) {
        self.storage = AlignedValueStorage(count: 32, repeating: value)
    }

    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 32, "Storage32 requires exactly 32 values")
        self.storage = AlignedValueStorage(from: values)
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }

    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        storage.dotProduct(other.storage)
    }
}

// MARK: - Storage for Dim64

/// Storage specifically for 64-dimensional vectors
public struct Storage64: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: AlignedValueStorage

    public var count: Int { 64 }

    @inlinable
    public init() {
        self.storage = AlignedValueStorage(count: 64)
    }

    @inlinable
    public init(repeating value: Float) {
        self.storage = AlignedValueStorage(count: 64, repeating: value)
    }

    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 64, "Storage64 requires exactly 64 values")
        self.storage = AlignedValueStorage(from: values)
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }

    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        storage.dotProduct(other.storage)
    }
}

// MARK: - Storage for Dim128

/// Storage specifically for 128-dimensional vectors
public struct Storage128: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: AlignedValueStorage

    public var count: Int { 128 }

    @inlinable
    public init() {
        self.storage = AlignedValueStorage(count: 128)
    }

    @inlinable
    public init(repeating value: Float) {
        self.storage = AlignedValueStorage(count: 128, repeating: value)
    }

    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 128, "Storage128 requires exactly 128 values")
        self.storage = AlignedValueStorage(from: values)
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }

    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        storage.dotProduct(other.storage)
    }
}

// MARK: - Storage for Dim256

/// Storage specifically for 256-dimensional vectors
public struct Storage256: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: AlignedValueStorage

    public var count: Int { 256 }

    @inlinable
    public init() {
        self.storage = AlignedValueStorage(count: 256)
    }

    @inlinable
    public init(repeating value: Float) {
        self.storage = AlignedValueStorage(count: 256, repeating: value)
    }

    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 256, "Storage256 requires exactly 256 values")
        self.storage = AlignedValueStorage(from: values)
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }

    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        storage.dotProduct(other.storage)
    }
}

// MARK: - Storage for Dim512

/// Storage specifically for 512-dimensional vectors
public struct Storage512: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: AlignedValueStorage

    public var count: Int { 512 }

    @inlinable
    public init() {
        self.storage = AlignedValueStorage(count: 512)
    }

    @inlinable
    public init(repeating value: Float) {
        self.storage = AlignedValueStorage(count: 512, repeating: value)
    }

    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 512, "Storage512 requires exactly 512 values")
        self.storage = AlignedValueStorage(from: values)
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }

    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        storage.dotProduct(other.storage)
    }
}

// MARK: - Storage for Dim768

/// Storage specifically for 768-dimensional vectors
public struct Storage768: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: AlignedValueStorage

    public var count: Int { 768 }

    @inlinable
    public init() {
        self.storage = AlignedValueStorage(count: 768)
    }

    @inlinable
    public init(repeating value: Float) {
        self.storage = AlignedValueStorage(count: 768, repeating: value)
    }

    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 768, "Storage768 requires exactly 768 values")
        self.storage = AlignedValueStorage(from: values)
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }

    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        storage.dotProduct(other.storage)
    }
}

// MARK: - Storage for Dim1536

/// Storage specifically for 1536-dimensional vectors
public struct Storage1536: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: AlignedValueStorage

    public var count: Int { 1536 }

    @inlinable
    public init() {
        self.storage = AlignedValueStorage(count: 1536)
    }

    @inlinable
    public init(repeating value: Float) {
        self.storage = AlignedValueStorage(count: 1536, repeating: value)
    }

    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 1536, "Storage1536 requires exactly 1536 values")
        self.storage = AlignedValueStorage(from: values)
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }

    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        storage.dotProduct(other.storage)
    }
}

// MARK: - Storage for Dim3072

/// Storage specifically for 3072-dimensional vectors
public struct Storage3072: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: AlignedValueStorage

    public var count: Int { 3072 }

    @inlinable
    public init() {
        self.storage = AlignedValueStorage(count: 3072)
    }

    @inlinable
    public init(repeating value: Float) {
        self.storage = AlignedValueStorage(count: 3072, repeating: value)
    }

    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 3072, "Storage3072 requires exactly 3072 values")
        self.storage = AlignedValueStorage(from: values)
    }

    @inlinable
    public subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }

    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }

    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }

    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        storage.dotProduct(other.storage)
    }
}
