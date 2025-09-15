//
//  SIMDExtensions.swift
//  VectorCore
//
//  Shared SIMD extensions for vector operations
//

import Foundation

// MARK: - Float SIMD Extensions

extension SIMD where Scalar == Float {
    /// Sum all elements in the SIMD vector
    @inlinable
    func sum() -> Scalar {
        return self.indices.reduce(Scalar.zero) { $0 + self[$1] }
    }
}

// MARK: - Double SIMD Extensions

extension SIMD where Scalar == Double {
    /// Sum all elements in the SIMD vector
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
