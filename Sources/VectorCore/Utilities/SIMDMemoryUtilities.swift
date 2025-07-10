// VectorCore: SIMD Memory Utilities
//
// Zero-copy memory access utilities for SIMD storage types
//

import Foundation

/// Helper for zero-copy access to tuple-based SIMD storage
@usableFromInline
internal enum SIMDMemoryUtilities {
    
    /// Access contiguous memory of a 2-tuple of SIMD64 as Float buffer
    /// 
    /// Leverages Swift's guaranteed tuple memory layout:
    /// - Tuples of homogeneous types are laid out contiguously
    /// - No padding between elements for aligned types
    /// - Total size = sum of element sizes
    ///
    /// Safety: This is safe because SIMD64<Float> is 256 bytes (64 * 4),
    /// properly aligned, and the tuple layout is guaranteed contiguous.
    @inlinable
    @_alwaysEmitIntoClient
    static func withUnsafeBufferPointer<R>(
        to tuple: (SIMD64<Float>, SIMD64<Float>),
        count: Int,
        _ body: (UnsafeBufferPointer<Float>) throws -> R
    ) rethrows -> R {
        try withUnsafePointer(to: tuple) { tuplePtr in
            let rawPtr = UnsafeRawPointer(tuplePtr)
            let floatPtr = rawPtr.assumingMemoryBound(to: Float.self)
            let buffer = UnsafeBufferPointer(start: floatPtr, count: count)
            return try body(buffer)
        }
    }
    
    /// Mutable access to contiguous memory of a 2-tuple of SIMD64
    @inlinable
    @_alwaysEmitIntoClient
    static func withUnsafeMutableBufferPointer<R>(
        to tuple: inout (SIMD64<Float>, SIMD64<Float>),
        count: Int,
        _ body: (UnsafeMutableBufferPointer<Float>) throws -> R
    ) rethrows -> R {
        try withUnsafeMutablePointer(to: &tuple) { tuplePtr in
            let rawPtr = UnsafeMutableRawPointer(tuplePtr)
            let floatPtr = rawPtr.assumingMemoryBound(to: Float.self)
            let buffer = UnsafeMutableBufferPointer(start: floatPtr, count: count)
            return try body(buffer)
        }
    }
    
    /// Access contiguous memory of a 4-tuple of SIMD64 as Float buffer
    @inlinable
    @_alwaysEmitIntoClient
    static func withUnsafeBufferPointer<R>(
        to tuple: (SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>),
        count: Int,
        _ body: (UnsafeBufferPointer<Float>) throws -> R
    ) rethrows -> R {
        try withUnsafePointer(to: tuple) { tuplePtr in
            let rawPtr = UnsafeRawPointer(tuplePtr)
            let floatPtr = rawPtr.assumingMemoryBound(to: Float.self)
            let buffer = UnsafeBufferPointer(start: floatPtr, count: count)
            return try body(buffer)
        }
    }
    
    /// Mutable access to contiguous memory of a 4-tuple of SIMD64
    @inlinable
    @_alwaysEmitIntoClient
    static func withUnsafeMutableBufferPointer<R>(
        to tuple: inout (SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>),
        count: Int,
        _ body: (UnsafeMutableBufferPointer<Float>) throws -> R
    ) rethrows -> R {
        try withUnsafeMutablePointer(to: &tuple) { tuplePtr in
            let rawPtr = UnsafeMutableRawPointer(tuplePtr)
            let floatPtr = rawPtr.assumingMemoryBound(to: Float.self)
            let buffer = UnsafeMutableBufferPointer(start: floatPtr, count: count)
            return try body(buffer)
        }
    }
}