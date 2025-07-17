// VectorCore: Dimension-Specific Storage Wrappers
//
// Wrapper types that bind storage to specific dimensions for Vector<D> compatibility
//

import Foundation
import Accelerate

// MARK: - Storage for Dim32

/// Storage specifically for 32-dimensional vectors
public struct Storage32: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: SmallVectorStorage
    
    public var count: Int { 32 }
    
    @inlinable
    public init() {
        self.storage = SmallVectorStorage(count: 32)
    }
    
    @inlinable
    public init(repeating value: Float) {
        self.storage = SmallVectorStorage(count: 32, repeating: value)
    }
    
    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 32, "Storage32 requires exactly 32 values")
        self.storage = SmallVectorStorage(from: values)
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
    internal var storage: SmallVectorStorage
    
    public var count: Int { 64 }
    
    @inlinable
    public init() {
        self.storage = SmallVectorStorage(count: 64)
    }
    
    @inlinable
    public init(repeating value: Float) {
        self.storage = SmallVectorStorage(count: 64, repeating: value)
    }
    
    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 64, "Storage64 requires exactly 64 values")
        self.storage = SmallVectorStorage(from: values)
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
    internal var storage: MediumVectorStorage
    
    public var count: Int { 128 }
    
    @inlinable
    public init() {
        self.storage = MediumVectorStorage(count: 128)
    }
    
    @inlinable
    public init(repeating value: Float) {
        self.storage = MediumVectorStorage(count: 128, repeating: value)
    }
    
    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 128, "Storage128 requires exactly 128 values")
        self.storage = MediumVectorStorage(from: values)
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
    internal var storage: MediumVectorStorage
    
    public var count: Int { 256 }
    
    @inlinable
    public init() {
        self.storage = MediumVectorStorage(count: 256)
    }
    
    @inlinable
    public init(repeating value: Float) {
        self.storage = MediumVectorStorage(count: 256, repeating: value)
    }
    
    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 256, "Storage256 requires exactly 256 values")
        self.storage = MediumVectorStorage(from: values)
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
    internal var storage: MediumVectorStorage
    
    public var count: Int { 512 }
    
    @inlinable
    public init() {
        self.storage = MediumVectorStorage(count: 512)
    }
    
    @inlinable
    public init(repeating value: Float) {
        self.storage = MediumVectorStorage(count: 512, repeating: value)
    }
    
    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 512, "Storage512 requires exactly 512 values")
        self.storage = MediumVectorStorage(from: values)
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
    internal var storage: LargeVectorStorage
    
    public var count: Int { 768 }
    
    @inlinable
    public init() {
        self.storage = LargeVectorStorage(count: 768)
    }
    
    @inlinable
    public init(repeating value: Float) {
        self.storage = LargeVectorStorage(count: 768, repeating: value)
    }
    
    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 768, "Storage768 requires exactly 768 values")
        self.storage = LargeVectorStorage(from: values)
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
    internal var storage: LargeVectorStorage
    
    public var count: Int { 1536 }
    
    @inlinable
    public init() {
        self.storage = LargeVectorStorage(count: 1536)
    }
    
    @inlinable
    public init(repeating value: Float) {
        self.storage = LargeVectorStorage(count: 1536, repeating: value)
    }
    
    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 1536, "Storage1536 requires exactly 1536 values")
        self.storage = LargeVectorStorage(from: values)
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
    internal var storage: LargeVectorStorage
    
    public var count: Int { 3072 }
    
    @inlinable
    public init() {
        self.storage = LargeVectorStorage(count: 3072)
    }
    
    @inlinable
    public init(repeating value: Float) {
        self.storage = LargeVectorStorage(count: 3072, repeating: value)
    }
    
    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 3072, "Storage3072 requires exactly 3072 values")
        self.storage = LargeVectorStorage(from: values)
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