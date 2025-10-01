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
/// high-performance vector kernels.
///
public protocol SIMDStorage {
    /// The SIMD element type stored in this container
    associatedtype Element

    /// Number of SIMD elements in storage
    var count: Int { get }

    /// Provides unsafe buffer pointer access for high-performance operations
    /// - Parameter body: Closure that receives an unsafe buffer pointer to the storage
    /// - Returns: Result of the closure
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R

    /// Provides mutable unsafe buffer pointer access for high-performance operations
    /// - Parameter body: Closure that receives an inout mutable unsafe buffer pointer to the storage
    /// - Returns: Result of the closure
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (inout UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R

    /// Subscript access for individual SIMD elements
    subscript(index: Int) -> Element { get set }
}

// MARK: - ContiguousArray Extensions

/// Make ContiguousArray conform to SIMDStorage for any SIMD4 element type
extension ContiguousArray: SIMDStorage {
    // All required methods are already provided by ContiguousArray
    // This works for any Element type, including SIMD4<Float>, SIMD4<Int8>, etc.
}

// MARK: - Protocol Extensions for Common Operations

public extension SIMDStorage where Element == SIMD4<Float> {
    /// Find minimum and maximum values across all SIMD elements
    func minMax() -> (min: Float, max: Float) {
        var globalMin: Float = .greatestFiniteMagnitude
        var globalMax: Float = -.greatestFiniteMagnitude

        withUnsafeBufferPointer { ptr in
            for simd in ptr {
                globalMin = min(globalMin, simd.min())
                globalMax = max(globalMax, simd.max())
            }
        }

        return (globalMin, globalMax)
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
