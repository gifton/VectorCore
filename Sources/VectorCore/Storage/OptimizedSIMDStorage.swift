// VectorCore: Optimized SIMD Storage Implementations
//
// Zero-allocation SIMD storage for high-performance vector operations
//

import Foundation
import simd
import Accelerate

// MARK: - Optimized SIMDStorage128 (128 dimensions)

/// Zero-allocation storage for 128-dimensional vectors using 2×SIMD64
///
/// Implementation notes:
/// - Uses tuple memory layout for contiguous access without allocation
/// - Leverages Swift's guaranteed tuple layout for homogeneous types
/// - Zero-copy buffer access via type punning
/// - Thread-safe for concurrent reads, requires synchronization for writes
public struct OptimizedSIMDStorage128: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var data: (SIMD64<Float>, SIMD64<Float>)
    
    public let count = 128
    
    public init() {
        self.data = (.zero, .zero)
    }
    
    public init(repeating value: Float) {
        let v = SIMD64(repeating: value)
        self.data = (v, v)
    }
    
    public init(from values: [Float]) {
        precondition(values.count == 128, "OptimizedSIMDStorage128 requires exactly 128 values")
        var first = SIMD64<Float>.zero
        var second = SIMD64<Float>.zero
        
        // Unrolled initialization for better performance
        for i in 0..<64 {
            first[i] = values[i]
            second[i] = values[i + 64]
        }
        
        self.data = (first, second)
    }
    
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < 128, "Index \(index) out of bounds [0, 128)")
            if index < 64 {
                return data.0[index]
            } else {
                return data.1[index - 64]
            }
        }
        set {
            precondition(index >= 0 && index < 128, "Index \(index) out of bounds [0, 128)")
            if index < 64 {
                data.0[index] = newValue
            } else {
                data.1[index - 64] = newValue
            }
        }
    }
    
    /// Zero-copy buffer access using tuple memory layout
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try SIMDMemoryUtilities.withUnsafeBufferPointer(to: data, count: 128, body)
    }
    
    /// Zero-copy mutable buffer access
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try SIMDMemoryUtilities.withUnsafeMutableBufferPointer(to: &data, count: 128, body)
    }
    
    /// Optimized dot product using vDSP
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

// MARK: - Optimized SIMDStorage256 (256 dimensions)

/// Zero-allocation storage for 256-dimensional vectors using 4×SIMD64
public struct OptimizedSIMDStorage256: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var data: (SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>)
    
    public let count = 256
    
    public init() {
        self.data = (.zero, .zero, .zero, .zero)
    }
    
    public init(repeating value: Float) {
        let v = SIMD64(repeating: value)
        self.data = (v, v, v, v)
    }
    
    public init(from values: [Float]) {
        precondition(values.count == 256, "OptimizedSIMDStorage256 requires exactly 256 values")
        var chunks = (SIMD64<Float>.zero, SIMD64<Float>.zero, 
                     SIMD64<Float>.zero, SIMD64<Float>.zero)
        
        // Vectorized initialization
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
            precondition(index >= 0 && index < 256, "Index \(index) out of bounds [0, 256)")
            let chunk = index >> 6  // Divide by 64 using bit shift
            let offset = index & 63 // Modulo 64 using bit mask
            
            switch chunk {
            case 0: return data.0[offset]
            case 1: return data.1[offset]
            case 2: return data.2[offset]
            case 3: return data.3[offset]
            default: fatalError("Unreachable")
            }
        }
        set {
            precondition(index >= 0 && index < 256, "Index \(index) out of bounds [0, 256)")
            let chunk = index >> 6
            let offset = index & 63
            
            switch chunk {
            case 0: data.0[offset] = newValue
            case 1: data.1[offset] = newValue
            case 2: data.2[offset] = newValue
            case 3: data.3[offset] = newValue
            default: fatalError("Unreachable")
            }
        }
    }
    
    /// Zero-copy buffer access
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try SIMDMemoryUtilities.withUnsafeBufferPointer(to: data, count: 256, body)
    }
    
    /// Zero-copy mutable buffer access
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try SIMDMemoryUtilities.withUnsafeMutableBufferPointer(to: &data, count: 256, body)
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

// MARK: - Optimized Contiguous Storage Base

/// Base type for larger SIMD storage using contiguous memory
///
/// For dimensions > 256, we use a single contiguous allocation
/// aligned to 64-byte boundaries for optimal SIMD performance
@usableFromInline
internal final class ContiguousSIMDStorage: @unchecked Sendable {
    @usableFromInline
    internal let buffer: UnsafeMutableBufferPointer<Float>
    
    @usableFromInline
    internal let alignment = 64 // Cache line size
    
    @inlinable
    init(count: Int) {
        // Allocate aligned memory for optimal SIMD performance
        let rawPtr = UnsafeMutableRawPointer.allocate(
            byteCount: count * MemoryLayout<Float>.stride,
            alignment: alignment
        )
        self.buffer = UnsafeMutableBufferPointer(
            start: rawPtr.assumingMemoryBound(to: Float.self),
            count: count
        )
        // Initialize to zero
        buffer.initialize(repeating: 0)
    }
    
    @inlinable
    convenience init(count: Int, repeating value: Float) {
        self.init(count: count)
        buffer.initialize(repeating: value)
    }
    
    deinit {
        buffer.deallocate()
    }
}

// MARK: - Optimized SIMDStorage512

/// Zero-copy storage for 512-dimensional vectors using contiguous memory
public struct OptimizedSIMDStorage512: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: ContiguousSIMDStorage
    
    public let count = 512
    
    public init() {
        self.storage = ContiguousSIMDStorage(count: 512)
    }
    
    public init(repeating value: Float) {
        self.storage = ContiguousSIMDStorage(count: 512, repeating: value)
    }
    
    public init(from values: [Float]) {
        precondition(values.count == 512, "OptimizedSIMDStorage512 requires exactly 512 values")
        self.storage = ContiguousSIMDStorage(count: 512)
        storage.buffer.initialize(from: values)
    }
    
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < 512, "Index \(index) out of bounds [0, 512)")
            return storage.buffer[index]
        }
        set {
            precondition(index >= 0 && index < 512, "Index \(index) out of bounds [0, 512)")
            storage.buffer[index] = newValue
        }
    }
    
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(storage.buffer))
    }
    
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(storage.buffer)
    }
    
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        var result: Float = 0
        vDSP_dotpr(storage.buffer.baseAddress!, 1,
                  other.storage.buffer.baseAddress!, 1,
                  &result, vDSP_Length(512))
        return result
    }
}

// MARK: - Optimized SIMDStorage768

/// Zero-copy storage for 768-dimensional vectors using contiguous memory
public struct OptimizedSIMDStorage768: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: ContiguousSIMDStorage
    
    public let count = 768
    
    public init() {
        self.storage = ContiguousSIMDStorage(count: 768)
    }
    
    public init(repeating value: Float) {
        self.storage = ContiguousSIMDStorage(count: 768, repeating: value)
    }
    
    public init(from values: [Float]) {
        precondition(values.count == 768, "OptimizedSIMDStorage768 requires exactly 768 values")
        self.storage = ContiguousSIMDStorage(count: 768)
        storage.buffer.initialize(from: values)
    }
    
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < 768, "Index \(index) out of bounds [0, 768)")
            return storage.buffer[index]
        }
        set {
            precondition(index >= 0 && index < 768, "Index \(index) out of bounds [0, 768)")
            storage.buffer[index] = newValue
        }
    }
    
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(storage.buffer))
    }
    
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(storage.buffer)
    }
    
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        var result: Float = 0
        vDSP_dotpr(storage.buffer.baseAddress!, 1,
                  other.storage.buffer.baseAddress!, 1,
                  &result, vDSP_Length(768))
        return result
    }
}

// MARK: - Optimized SIMDStorage1536

/// Zero-copy storage for 1536-dimensional vectors using contiguous memory
public struct OptimizedSIMDStorage1536: VectorStorage, VectorStorageOperations {
    @usableFromInline
    internal var storage: ContiguousSIMDStorage
    
    public let count = 1536
    
    public init() {
        self.storage = ContiguousSIMDStorage(count: 1536)
    }
    
    public init(repeating value: Float) {
        self.storage = ContiguousSIMDStorage(count: 1536, repeating: value)
    }
    
    public init(from values: [Float]) {
        precondition(values.count == 1536, "OptimizedSIMDStorage1536 requires exactly 1536 values")
        self.storage = ContiguousSIMDStorage(count: 1536)
        storage.buffer.initialize(from: values)
    }
    
    @inlinable
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < 1536, "Index \(index) out of bounds [0, 1536)")
            return storage.buffer[index]
        }
        set {
            precondition(index >= 0 && index < 1536, "Index \(index) out of bounds [0, 1536)")
            storage.buffer[index] = newValue
        }
    }
    
    @inlinable
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(storage.buffer))
    }
    
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try body(storage.buffer)
    }
    
    @inlinable
    public func dotProduct(_ other: Self) -> Float {
        var result: Float = 0
        vDSP_dotpr(storage.buffer.baseAddress!, 1,
                  other.storage.buffer.baseAddress!, 1,
                  &result, vDSP_Length(1536))
        return result
    }
}