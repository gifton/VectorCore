// VectorCore: Optimized SIMD Storage
//
// Zero-allocation storage implementations for high-performance vector operations
//

import Foundation
import simd
import Accelerate

// MARK: - SIMD32 Storage (1-32 dimensions)

/// Optimized storage for vectors with dimensions 1-32
public struct SIMDStorage32: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var data: SIMD32<Float>
    
    @usableFromInline
    internal let actualCount: Int
    
    public var count: Int { actualCount }
    
    @inlinable
    public init() {
        self.actualCount = 32
        self.data = .zero
    }
    
    @inlinable
    public init(count: Int = 32) {
        precondition(count > 0 && count <= 32, "Count must be between 1 and 32")
        self.actualCount = count
        self.data = .zero
    }
    
    @inlinable
    public init(repeating value: Float) {
        self.actualCount = 32
        self.data = SIMD32(repeating: value)
    }
    
    @inlinable
    public init(from values: [Float]) {
        precondition(!values.isEmpty && values.count <= 32)
        self.actualCount = values.count
        self.data = .zero
        for i in 0..<values.count {
            self.data[i] = values[i]
        }
    }
    
    @inlinable
    public subscript(index: Int) -> Float {
        get { data[index] }
        set { data[index] = newValue }
    }
    
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try withUnsafePointer(to: data) { ptr in
            let floatPtr = UnsafeRawPointer(ptr).assumingMemoryBound(to: Float.self)
            let buffer = UnsafeBufferPointer(start: floatPtr, count: actualCount)
            return try body(buffer)
        }
    }
    
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try withUnsafeMutablePointer(to: &data) { ptr in
            let floatPtr = UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: Float.self)
            let buffer = UnsafeMutableBufferPointer(start: floatPtr, count: actualCount)
            return try body(buffer)
        }
    }
    
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        var result: Float = 0
        self.withUnsafeBufferPointer { selfBuffer in
            other.withUnsafeBufferPointer { otherBuffer in
                vDSP_dotpr(selfBuffer.baseAddress!, 1,
                          otherBuffer.baseAddress!, 1,
                          &result, vDSP_Length(actualCount))
            }
        }
        return result
    }
}

// MARK: - SIMD64 Storage (33-64 dimensions)

/// Optimized storage for vectors with dimensions 33-64
public struct SIMDStorage64: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var data: SIMD64<Float>
    
    @usableFromInline
    internal let actualCount: Int
    
    public var count: Int { actualCount }
    
    @inlinable
    public init() {
        self.actualCount = 64
        self.data = .zero
    }
    
    @inlinable
    public init(count: Int = 64) {
        precondition(count > 32 && count <= 64, "Count must be between 33 and 64")
        self.actualCount = count
        self.data = .zero
    }
    
    @inlinable
    public init(repeating value: Float) {
        self.actualCount = 64
        self.data = SIMD64(repeating: value)
    }
    
    @inlinable
    public init(from values: [Float]) {
        precondition(values.count > 32 && values.count <= 64)
        self.actualCount = values.count
        self.data = .zero
        for i in 0..<values.count {
            self.data[i] = values[i]
        }
    }
    
    @inlinable
    public subscript(index: Int) -> Float {
        get { data[index] }
        set { data[index] = newValue }
    }
    
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try withUnsafePointer(to: data) { ptr in
            let floatPtr = UnsafeRawPointer(ptr).assumingMemoryBound(to: Float.self)
            let buffer = UnsafeBufferPointer(start: floatPtr, count: actualCount)
            return try body(buffer)
        }
    }
    
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try withUnsafeMutablePointer(to: &data) { ptr in
            let floatPtr = UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: Float.self)
            let buffer = UnsafeMutableBufferPointer(start: floatPtr, count: actualCount)
            return try body(buffer)
        }
    }
    
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        var result: Float = 0
        self.withUnsafeBufferPointer { selfBuffer in
            other.withUnsafeBufferPointer { otherBuffer in
                vDSP_dotpr(selfBuffer.baseAddress!, 1,
                          otherBuffer.baseAddress!, 1,
                          &result, vDSP_Length(actualCount))
            }
        }
        return result
    }
}

// MARK: - Optimized Composite SIMD Storage

/// Zero-allocation storage for 128-dimensional vectors
public struct SIMDStorage128: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var data: (SIMD64<Float>, SIMD64<Float>)
    
    public let count = 128
    
    @inlinable
    public init() {
        self.data = (.zero, .zero)
    }
    
    @inlinable
    public init(repeating value: Float) {
        let v = SIMD64(repeating: value)
        self.data = (v, v)
    }
    
    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 128)
        var first = SIMD64<Float>.zero
        var second = SIMD64<Float>.zero
        
        for i in 0..<64 {
            first[i] = values[i]
            second[i] = values[i + 64]
        }
        
        self.data = (first, second)
    }
    
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < 128)
            if index < 64 {
                return data.0[index]
            } else {
                return data.1[index - 64]
            }
        }
        set {
            precondition(index >= 0 && index < 128)
            if index < 64 {
                data.0[index] = newValue
            } else {
                data.1[index - 64] = newValue
            }
        }
    }
    
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        // Zero-copy access using Swift's tuple memory layout guarantee
        try withUnsafePointer(to: data) { tuplePtr in
            let rawPtr = UnsafeRawPointer(tuplePtr)
            let floatPtr = rawPtr.assumingMemoryBound(to: Float.self)
            let buffer = UnsafeBufferPointer(start: floatPtr, count: 128)
            return try body(buffer)
        }
    }
    
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        // Zero-copy mutable access
        try withUnsafeMutablePointer(to: &data) { tuplePtr in
            let rawPtr = UnsafeMutableRawPointer(tuplePtr)
            let floatPtr = rawPtr.assumingMemoryBound(to: Float.self)
            let buffer = UnsafeMutableBufferPointer(start: floatPtr, count: 128)
            return try body(buffer)
        }
    }
    
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        var result: Float = 0
        self.withUnsafeBufferPointer { selfBuffer in
            other.withUnsafeBufferPointer { otherBuffer in
                vDSP_dotpr(selfBuffer.baseAddress!, 1,
                          otherBuffer.baseAddress!, 1,
                          &result, vDSP_Length(128))
            }
        }
        return result
    }
}

/// Zero-allocation storage for 256-dimensional vectors
public struct SIMDStorage256: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var data: (SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>)
    
    public let count = 256
    
    @inlinable
    public init() {
        self.data = (.zero, .zero, .zero, .zero)
    }
    
    @inlinable
    public init(repeating value: Float) {
        let v = SIMD64(repeating: value)
        self.data = (v, v, v, v)
    }
    
    @inlinable
    public init(from values: [Float]) {
        precondition(values.count == 256)
        var chunks = (SIMD64<Float>.zero, SIMD64<Float>.zero, SIMD64<Float>.zero, SIMD64<Float>.zero)
        
        for i in 0..<64 {
            chunks.0[i] = values[i]
            chunks.1[i] = values[i + 64]
            chunks.2[i] = values[i + 128]
            chunks.3[i] = values[i + 192]
        }
        
        self.data = chunks
    }
    
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < 256)
            let chunk = index >> 6  // Divide by 64 using bit shift
            let offset = index & 63 // Modulo 64 using bit mask
            
            switch chunk {
            case 0: return data.0[offset]
            case 1: return data.1[offset]
            case 2: return data.2[offset]
            case 3: return data.3[offset]
            default: fatalError("Invalid index")
            }
        }
        set {
            precondition(index >= 0 && index < 256)
            let chunk = index >> 6
            let offset = index & 63
            
            switch chunk {
            case 0: data.0[offset] = newValue
            case 1: data.1[offset] = newValue
            case 2: data.2[offset] = newValue
            case 3: data.3[offset] = newValue
            default: fatalError("Invalid index")
            }
        }
    }
    
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        // Zero-copy access using tuple memory layout
        try withUnsafePointer(to: data) { tuplePtr in
            let rawPtr = UnsafeRawPointer(tuplePtr)
            let floatPtr = rawPtr.assumingMemoryBound(to: Float.self)
            let buffer = UnsafeBufferPointer(start: floatPtr, count: 256)
            return try body(buffer)
        }
    }
    
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try withUnsafeMutablePointer(to: &data) { tuplePtr in
            let rawPtr = UnsafeMutableRawPointer(tuplePtr)
            let floatPtr = rawPtr.assumingMemoryBound(to: Float.self)
            let buffer = UnsafeMutableBufferPointer(start: floatPtr, count: 256)
            return try body(buffer)
        }
    }
    
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        var result: Float = 0
        self.withUnsafeBufferPointer { selfBuffer in
            other.withUnsafeBufferPointer { otherBuffer in
                vDSP_dotpr(selfBuffer.baseAddress!, 1,
                          otherBuffer.baseAddress!, 1,
                          &result, vDSP_Length(256))
            }
        }
        return result
    }
}

// MARK: - Aligned Storage for Larger Vectors

/// Aligned storage for optimal SIMD performance
@usableFromInline
final class AlignedStorage: @unchecked Sendable {
    @usableFromInline
    let ptr: UnsafeMutablePointer<Float>
    @usableFromInline
    let count: Int
    
    @usableFromInline
    init(count: Int, alignment: Int = 64) {
        self.count = count
        // Use posix_memalign for guaranteed alignment
        var rawPtr: UnsafeMutableRawPointer?
        posix_memalign(&rawPtr, alignment, count * MemoryLayout<Float>.stride)
        self.ptr = rawPtr!.assumingMemoryBound(to: Float.self)
        self.ptr.initialize(repeating: 0, count: count)
    }
    
    deinit {
        ptr.deinitialize(count: count)
        ptr.deallocate()
    }
}

/// Optimized storage for 512-dimensional vectors with aligned memory
public struct SIMDStorage512: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal let storage: AlignedStorage
    
    public let count = 512
    
    public init() {
        self.storage = AlignedStorage(count: 512)
    }
    
    public init(repeating value: Float) {
        self.storage = AlignedStorage(count: 512)
        self.storage.ptr.initialize(repeating: value, count: 512)
    }
    
    public init(from values: [Float]) {
        precondition(values.count == 512)
        self.storage = AlignedStorage(count: 512)
        self.storage.ptr.initialize(from: values, count: 512)
    }
    
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < 512)
            return storage.ptr[index]
        }
        set {
            precondition(index >= 0 && index < 512)
            storage.ptr[index] = newValue
        }
    }
    
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(start: storage.ptr, count: 512))
    }
    
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeMutableBufferPointer(start: storage.ptr, count: 512))
    }
    
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        var result: Float = 0
        vDSP_dotpr(storage.ptr, 1, other.storage.ptr, 1, &result, vDSP_Length(512))
        return result
    }
}

/// Optimized storage for 768-dimensional vectors
public struct SIMDStorage768: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal let storage: AlignedStorage
    
    public let count = 768
    
    public init() {
        self.storage = AlignedStorage(count: 768)
    }
    
    public init(repeating value: Float) {
        self.storage = AlignedStorage(count: 768)
        self.storage.ptr.initialize(repeating: value, count: 768)
    }
    
    public init(from values: [Float]) {
        precondition(values.count == 768)
        self.storage = AlignedStorage(count: 768)
        self.storage.ptr.initialize(from: values, count: 768)
    }
    
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < 768)
            return storage.ptr[index]
        }
        set {
            precondition(index >= 0 && index < 768)
            storage.ptr[index] = newValue
        }
    }
    
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(start: storage.ptr, count: 768))
    }
    
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeMutableBufferPointer(start: storage.ptr, count: 768))
    }
    
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        var result: Float = 0
        vDSP_dotpr(storage.ptr, 1, other.storage.ptr, 1, &result, vDSP_Length(768))
        return result
    }
}

/// Optimized storage for 1536-dimensional vectors
public struct SIMDStorage1536: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal let storage: AlignedStorage
    
    public let count = 1536
    
    public init() {
        self.storage = AlignedStorage(count: 1536)
    }
    
    public init(repeating value: Float) {
        self.storage = AlignedStorage(count: 1536)
        self.storage.ptr.initialize(repeating: value, count: 1536)
    }
    
    public init(from values: [Float]) {
        precondition(values.count == 1536)
        self.storage = AlignedStorage(count: 1536)
        self.storage.ptr.initialize(from: values, count: 1536)
    }
    
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < 1536)
            return storage.ptr[index]
        }
        set {
            precondition(index >= 0 && index < 1536)
            storage.ptr[index] = newValue
        }
    }
    
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(start: storage.ptr, count: 1536))
    }
    
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeMutableBufferPointer(start: storage.ptr, count: 1536))
    }
    
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        var result: Float = 0
        vDSP_dotpr(storage.ptr, 1, other.storage.ptr, 1, &result, vDSP_Length(1536))
        return result
    }
}

/// Optimized storage for 3072-dimensional vectors
public struct SIMDStorage3072: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal let storage: AlignedStorage
    
    public let count = 3072
    
    public init() {
        self.storage = AlignedStorage(count: 3072)
    }
    
    public init(repeating value: Float) {
        self.storage = AlignedStorage(count: 3072)
        self.storage.ptr.initialize(repeating: value, count: 3072)
    }
    
    public init(from values: [Float]) {
        precondition(values.count == 3072)
        self.storage = AlignedStorage(count: 3072)
        self.storage.ptr.initialize(from: values, count: 3072)
    }
    
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < 3072)
            return storage.ptr[index]
        }
        set {
            precondition(index >= 0 && index < 3072)
            storage.ptr[index] = newValue
        }
    }
    
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(start: storage.ptr, count: 3072))
    }
    
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeMutableBufferPointer(start: storage.ptr, count: 3072))
    }
    
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        var result: Float = 0
        vDSP_dotpr(storage.ptr, 1, other.storage.ptr, 1, &result, vDSP_Length(3072))
        return result
    }
}