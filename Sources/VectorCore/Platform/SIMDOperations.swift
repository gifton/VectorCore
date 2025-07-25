//
//  SIMDOperations.swift
//  VectorCore
//
//  SIMD-accelerated operations with platform-specific implementations
//

import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

/// SIMD-accelerated operations with automatic fallback to scalar implementations
public enum SIMDOperations {
    // MARK: - Dot Product
    
    /// Compute dot product of two float arrays with SIMD acceleration
    @inlinable
    public static func dotProduct(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        #if canImport(Accelerate)
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(count))
        return result
        #else
        // Scalar fallback
        var sum: Float = 0
        for i in 0..<count {
            sum += a[i] * b[i]
        }
        return sum
        #endif
    }
    
    // MARK: - Distance Operations
    
    /// Compute squared Euclidean distance with SIMD
    @inlinable
    public static func squaredEuclideanDistance(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        #if canImport(Accelerate)
        var result: Float = 0
        vDSP_distancesq(a, 1, b, 1, &result, vDSP_Length(count))
        return result
        #else
        // Scalar fallback
        var sum: Float = 0
        for i in 0..<count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sum
        #endif
    }
    
    // MARK: - Threshold Operations
    
    /// Apply threshold to array elements
    public static func applyThreshold(
        _ input: UnsafePointer<Float>,
        _ output: UnsafeMutablePointer<Float>,
        threshold: Float,
        count: Int,
        replaceWith: Float = 0
    ) {
        #if canImport(Accelerate)
        var thresholdValue = threshold
        var replacementValue = replaceWith
        vDSP_vthres(input, 1, &thresholdValue, output, 1, vDSP_Length(count))
        // Replace values below threshold
        vDSP_vlim(output, 1, &replacementValue, &thresholdValue, output, 1, vDSP_Length(count))
        #else
        // Scalar fallback
        for i in 0..<count {
            output[i] = input[i] >= threshold ? input[i] : replaceWith
        }
        #endif
    }
    
    // MARK: - Vector Operations
    
    /// Element-wise addition
    @inlinable
    public static func add(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        _ result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        #if canImport(Accelerate)
        vDSP_vadd(a, 1, b, 1, result, 1, vDSP_Length(count))
        #else
        for i in 0..<count {
            result[i] = a[i] + b[i]
        }
        #endif
    }
    
    /// Element-wise multiplication
    @inlinable
    public static func multiply(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        _ result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        #if canImport(Accelerate)
        vDSP_vmul(a, 1, b, 1, result, 1, vDSP_Length(count))
        #else
        for i in 0..<count {
            result[i] = a[i] * b[i]
        }
        #endif
    }
    
    // MARK: - Reduction Operations
    
    /// Sum all elements
    @inlinable
    public static func sum(
        _ input: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        #if canImport(Accelerate)
        var result: Float = 0
        vDSP_sve(input, 1, &result, vDSP_Length(count))
        return result
        #else
        return (0..<count).reduce(Float(0)) { $0 + input[$1] }
        #endif
    }
    
    /// Find maximum element
    @inlinable
    public static func maximum(
        _ input: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        #if canImport(Accelerate)
        var result: Float = 0
        vDSP_maxv(input, 1, &result, vDSP_Length(count))
        return result
        #else
        guard count > 0 else { return 0 }
        return (1..<count).reduce(input[0]) { max($0, input[$1]) }
        #endif
    }
    
    // MARK: - Availability Checking
    
    /// Check if SIMD operations are available for the current platform
    public static var isAvailable: Bool {
        #if canImport(Accelerate)
        return true
        #else
        return false
        #endif
    }
    
    /// Get the optimal vector size for SIMD operations
    public static var optimalVectorSize: Int {
        PlatformConfiguration.simdVectorWidth
    }
}