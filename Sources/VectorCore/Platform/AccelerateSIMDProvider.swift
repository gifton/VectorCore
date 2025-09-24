//
//  AccelerateSIMDProvider.swift
//  VectorCore
//
//  Accelerate framework implementation of SIMDProvider
//  Provides hardware-accelerated operations on Apple platforms
//

#if canImport(Accelerate)
import Foundation
import Accelerate

/// Accelerate framework implementation for Float operations
public struct AccelerateFloatProvider: SIMDProvider {
    public typealias Scalar = Float

    // MARK: - Arithmetic Operations

    @inlinable
    public static func add(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        vDSP_vadd(a, 1, b, 1, result, 1, vDSP_Length(count))
    }

    @inlinable
    public static func subtract(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        // vDSP_vsub computes: result = B - A when called as vDSP_vsub(A, 1, B, 1, result, 1, N)
        // So to compute a - b, we need vDSP_vsub(b, 1, a, 1, result, 1, count)
        vDSP_vsub(b, 1, a, 1, result, 1, vDSP_Length(count))
    }

    @inlinable
    public static func multiply(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        vDSP_vmul(a, 1, b, 1, result, 1, vDSP_Length(count))
    }

    @inlinable
    public static func divide(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        vDSP_vdiv(b, 1, a, 1, result, 1, vDSP_Length(count))
    }

    @inlinable
    public static func negate(
        _ a: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        vDSP_vneg(a, 1, result, 1, vDSP_Length(count))
    }

    // MARK: - Scalar Operations

    @inlinable
    public static func addScalar(
        _ a: UnsafePointer<Float>,
        scalar: Float,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var s = scalar
        vDSP_vsadd(a, 1, &s, result, 1, vDSP_Length(count))
    }

    @inlinable
    public static func multiplyScalar(
        _ a: UnsafePointer<Float>,
        scalar: Float,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var s = scalar
        vDSP_vsmul(a, 1, &s, result, 1, vDSP_Length(count))
    }

    @inlinable
    public static func divideByScalar(
        _ a: UnsafePointer<Float>,
        scalar: Float,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var s = scalar
        vDSP_vsdiv(a, 1, &s, result, 1, vDSP_Length(count))
    }

    // MARK: - Reduction Operations

    @inlinable
    public static func dot(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    public static func sum(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var result: Float = 0
        vDSP_sve(a, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    public static func sumOfMagnitudes(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var result: Float = 0
        vDSP_svemg(a, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    public static func sumOfSquares(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var result: Float = 0
        vDSP_svesq(a, 1, &result, vDSP_Length(count))
        return result
    }

    // MARK: - Statistical Operations

    @inlinable
    public static func maximum(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var result: Float = 0
        vDSP_maxv(a, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    public static func minimum(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var result: Float = 0
        vDSP_minv(a, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    public static func maximumMagnitude(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var result: Float = 0
        vDSP_maxmgv(a, 1, &result, vDSP_Length(count))
        return result
    }

    // MARK: - Distance Operations

    @inlinable
    public static func distanceSquared(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var result: Float = 0
        vDSP_distancesq(a, 1, b, 1, &result, vDSP_Length(count))
        return result
    }

    // MARK: - Utility Operations

    @inlinable
    public static func copy(
        source: UnsafePointer<Float>,
        destination: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        // Use memcpy for efficient copying
        destination.initialize(from: source, count: count)
    }

    @inlinable
    public static func fill(
        value: Float,
        destination: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var v = value
        vDSP_vfill(&v, destination, 1, vDSP_Length(count))
    }

    @inlinable
    public static func clip(
        _ a: UnsafePointer<Float>,
        low: Float,
        high: Float,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var lo = low
        var hi = high
        vDSP_vclip(a, 1, &lo, &hi, result, 1, vDSP_Length(count))
    }
}

/// Accelerate framework implementation for Double operations
public struct AccelerateDoubleProvider: SIMDProvider {
    public typealias Scalar = Double

    // MARK: - Arithmetic Operations

    @inlinable
    public static func add(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        vDSP_vaddD(a, 1, b, 1, result, 1, vDSP_Length(count))
    }

    @inlinable
    public static func subtract(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        vDSP_vsubD(b, 1, a, 1, result, 1, vDSP_Length(count))
    }

    @inlinable
    public static func multiply(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        vDSP_vmulD(a, 1, b, 1, result, 1, vDSP_Length(count))
    }

    @inlinable
    public static func divide(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        vDSP_vdivD(b, 1, a, 1, result, 1, vDSP_Length(count))
    }

    @inlinable
    public static func negate(
        _ a: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        vDSP_vnegD(a, 1, result, 1, vDSP_Length(count))
    }

    // MARK: - Scalar Operations

    @inlinable
    public static func addScalar(
        _ a: UnsafePointer<Double>,
        scalar: Double,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        var s = scalar
        vDSP_vsaddD(a, 1, &s, result, 1, vDSP_Length(count))
    }

    @inlinable
    public static func multiplyScalar(
        _ a: UnsafePointer<Double>,
        scalar: Double,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        var s = scalar
        vDSP_vsmulD(a, 1, &s, result, 1, vDSP_Length(count))
    }

    @inlinable
    public static func divideByScalar(
        _ a: UnsafePointer<Double>,
        scalar: Double,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        var s = scalar
        vDSP_vsdivD(a, 1, &s, result, 1, vDSP_Length(count))
    }

    // MARK: - Reduction Operations

    @inlinable
    public static func dot(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        var result: Double = 0
        vDSP_dotprD(a, 1, b, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    public static func sum(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        var result: Double = 0
        vDSP_sveD(a, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    public static func sumOfMagnitudes(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        var result: Double = 0
        vDSP_svemgD(a, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    public static func sumOfSquares(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        var result: Double = 0
        vDSP_svesqD(a, 1, &result, vDSP_Length(count))
        return result
    }

    // MARK: - Statistical Operations

    @inlinable
    public static func maximum(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        var result: Double = 0
        vDSP_maxvD(a, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    public static func minimum(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        var result: Double = 0
        vDSP_minvD(a, 1, &result, vDSP_Length(count))
        return result
    }

    @inlinable
    public static func maximumMagnitude(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        var result: Double = 0
        vDSP_maxmgvD(a, 1, &result, vDSP_Length(count))
        return result
    }

    // MARK: - Distance Operations

    @inlinable
    public static func distanceSquared(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        var result: Double = 0
        vDSP_distancesqD(a, 1, b, 1, &result, vDSP_Length(count))
        return result
    }

    // MARK: - Utility Operations

    @inlinable
    public static func copy(
        source: UnsafePointer<Double>,
        destination: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        // Use memcpy for efficient copying
        destination.initialize(from: source, count: count)
    }

    @inlinable
    public static func fill(
        value: Double,
        destination: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        var v = value
        vDSP_vfillD(&v, destination, 1, vDSP_Length(count))
    }

    @inlinable
    public static func clip(
        _ a: UnsafePointer<Double>,
        low: Double,
        high: Double,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        var lo = low
        var hi = high
        vDSP_vclipD(a, 1, &lo, &hi, result, 1, vDSP_Length(count))
    }
}

#endif // canImport(Accelerate)
