//
//  SIMDStorage.swift
//  VectorCore
//
//  Protocol for SIMD-compatible storage types used by performance-critical kernels
//

import Foundation
import simd

/// Protocol defining requirements for SIMD-compatible storage types
///
/// This protocol enables type-safe access to SIMD storage in generic contexts,
/// providing both buffer pointer access and subscript operations required by
/// the vector kernels.
///
public protocol SIMDStorage {
    /// The SIMD element type stored in this container
    associatedtype Element

    /// Number of SIMD elements in storage
    var count: Int { get }

    /// Provides unsafe buffer pointer access to the storage
    /// - Parameter body: Closure that receives an unsafe buffer pointer to the storage
    /// - Returns: Result of the closure
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R

    /// Borrows the elements for mutation, passing the buffer descriptor by value.
    ///
    /// Conformer contract: invoke `body` exactly once, including for empty storage,
    /// and propagate its result or error. Element writes before an error remain
    /// mutations; this operation does not promise rollback. The buffer is valid
    /// only during `body`: do not escape its pointers or access this storage through
    /// another route during the borrow. Passing the descriptor by value permits
    /// element writes without permitting replacement of the owner's buffer.
    ///
    /// ContiguousArray behavior is exercised by `SIMDStorageMutationTests`, including
    /// `callbackRunsOnceIncludingEmptyStorage`, `cowCopyAndUntouchedTailArePreserved`
    /// and `throwingBorrowKeepsMutationAndOriginalError`.
    /// - Parameter body: A closure receiving a mutable view of the elements.
    /// - Returns: The result returned by `body`.
    mutating func withUnsafeMutableElements<R>(_ body: (UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R

    /// Subscript access for individual SIMD elements
    subscript(index: Int) -> Element { get set }
}

// MARK: - ContiguousArray Extensions

/// ContiguousArray's SIMDStorage conformance is not restricted to SIMD elements.
extension ContiguousArray: SIMDStorage {
    /// Uses ContiguousArray's mutable borrow and passes its descriptor by value.
    /// Shared storage may be copied by the underlying borrow; the callback's work
    /// is additional. See `SIMDStorageMutationTests` for the tested mutation contract.
    @inlinable
    public mutating func withUnsafeMutableElements<R>(
        _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        try withUnsafeMutableBufferPointer { buffer in
            try body(buffer)
        }
    }
}

// MARK: - Protocol Extensions for Common Operations

public extension SIMDStorage where Element == SIMD4<Float> {
    /// Find minimum and maximum values across all SIMD elements: (NaN, NaN) when any lane is NaN (T16 reduction NaN
    /// contract; `SIMD4.min()`/`max()` ignore a NaN lane and `min`/`max` drop one), and (+infinity, -infinity), the
    /// identities, for empty storage (T16 ruling I1; the earlier +-greatestFiniteMagnitude start also reported
    /// min = greatestFiniteMagnitude for all-+infinity data). Pinned by T16ReductionExceptionalTableTests.
    func minMax() -> (min: Float, max: Float) {
        // The result is returned from inside the closure; no captured var is written in it and read after it
        // (T15 lost-store class, T16 ruling C5).
        return withUnsafeBufferPointer { ptr -> (min: Float, max: Float) in
            var globalMin: Float = .infinity
            var globalMax: Float = -.infinity
            for simd in ptr {
                if any(simd .!= simd) { return (.nan, .nan) }
                globalMin = min(globalMin, simd.min())
                globalMax = max(globalMax, simd.max())
            }
            return (globalMin, globalMax)
        }
    }

    /// Flatten SIMD storage to a regular Float array
    func flattenToArray() -> [Float] {
        var result: [Float] = []
        result.reserveCapacity(count * 4)

        withUnsafeBufferPointer { ptr in
            for simd in ptr {
                result.append(contentsOf: [simd.x, simd.y, simd.z, simd.w])
            }
        }

        return result
    }
}

public extension SIMDStorage where Element == SIMD4<Int8> {
    /// Find minimum and maximum values across all SIMD elements
    func minMax() -> (min: Int8, max: Int8) {
        var globalMin: Int8 = .max
        var globalMax: Int8 = .min

        withUnsafeBufferPointer { ptr in
            for simd in ptr {
                globalMin = min(globalMin, min(min(simd.x, simd.y), min(simd.z, simd.w)))
                globalMax = max(globalMax, max(max(simd.x, simd.y), max(simd.z, simd.w)))
            }
        }

        return (globalMin, globalMax)
    }
}
