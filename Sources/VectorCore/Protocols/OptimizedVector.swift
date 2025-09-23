//
//  OptimizedVector.swift
//  VectorCore
//
//  Protocol extension for optimized vector types used by quantization kernels
//

import Foundation
import simd

/// Protocol for optimized vector types with additional properties required by specialized kernels
///
/// This protocol extends VectorProtocol with properties and methods specifically needed
/// by quantization and SIMD-optimized kernels. It provides:
/// - Lane count information for SIMD operations
/// - Dimension information for compile-time optimizations
/// - Static factory methods for kernel implementations
/// - SIMD storage access for high-performance operations
///
public protocol OptimizedVector: VectorProtocol where Scalar == Float, Storage: SIMDStorage, Storage.Element == SIMD4<Float> {

    // MARK: - SIMD Properties

    /// Number of SIMD lanes in the underlying storage
    /// For SIMD4<Float> storage, this would be 4
    static var laneCount: Int { get }


    // MARK: - Factory Methods

    /// Static zero vector instance
    static var zero: Self { get }
}

// MARK: - Default Implementations

public extension OptimizedVector {

    /// Instance accessor for lane count (convenience)
    var laneCount: Int { Self.laneCount }
}