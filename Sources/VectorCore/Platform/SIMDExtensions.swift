//
//  SIMDExtensions.swift
//  VectorCore
//
//  Shared SIMD extensions for vector operations
//

import Foundation

// MARK: - Float SIMD Extensions

extension SIMD where Scalar == Float {
    /// Sum of the lanes in index order from +0: `(((+0 + self[0]) + self[1]) + ...) + self[scalarCount - 1]`,
    /// as the body below reads.
    ///
    /// Inside VectorCore this shadows the standard library's `SIMD.sum()`. On the Swift 6.4 toolchain the stdlib
    /// member adds the lanes in the same order but starts from -0, so the two differ only for an all-(-0) vector,
    /// where this returns +0 and the stdlib returns -0. Which NaN payload results when two or more lanes are NaN is
    /// unspecified. `SIMDSumSemanticsTests` pins the zero sign and the order, and compares with the stdlib member
    /// on the toolchain under test.
    @inlinable
    func sum() -> Scalar {
        return self.indices.reduce(Scalar.zero) { $0 + self[$1] }
    }
}

// MARK: - Double SIMD Extensions

extension SIMD where Scalar == Double {
    /// Sum of the lanes in index order from +0: `(((+0 + self[0]) + self[1]) + ...) + self[scalarCount - 1]`,
    /// as the body below reads.
    ///
    /// Inside VectorCore this shadows the standard library's `SIMD.sum()`. On the Swift 6.4 toolchain the stdlib
    /// member adds the lanes in the same order but starts from -0, so the two differ only for an all-(-0) vector,
    /// where this returns +0 and the stdlib returns -0. Which NaN payload results when two or more lanes are NaN is
    /// unspecified. `SIMDSumSemanticsTests` pins the zero sign and the order, and compares with the stdlib member
    /// on the toolchain under test.
    @inlinable
    func sum() -> Scalar {
        return self.indices.reduce(Scalar.zero) { $0 + self[$1] }
    }
}

// MARK: - Generic SIMD Extensions

extension SIMD where Scalar: FloatingPoint {
    /// Load data from unsafe pointer with bounds checking
    @inlinable
    static func load(from pointer: UnsafePointer<Scalar>, count: Int, offset: Int) -> Self {
        precondition(offset + Self.scalarCount <= count, "SIMD load would exceed buffer bounds")
        var result = Self()
        for i in 0..<Self.scalarCount {
            result[i] = pointer[offset + i]
        }
        return result
    }

    /// Store data to unsafe pointer with bounds checking
    @inlinable
    func storeSafe(to pointer: UnsafeMutablePointer<Scalar>, count: Int, offset: Int) {
        precondition(offset + Self.scalarCount <= count, "SIMD store would exceed buffer bounds")
        // Store each element manually
        for i in 0..<Self.scalarCount {
            pointer[offset + i] = self[i]
        }
    }
}
