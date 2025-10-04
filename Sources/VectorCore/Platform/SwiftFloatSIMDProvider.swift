//
//  SwiftFloatSIMDProvider_Optimized.swift
//  VectorCore
//
//  Optimized Pure Swift implementation of SIMDProvider for Float
//  Cross-platform SIMD operations using Swift's built-in SIMD types
//

import Foundation

/// Optimized Pure Swift implementation for Float operations
public struct SwiftFloatSIMDProvider: SIMDProvider {
    public typealias Scalar = Float

    // MARK: - Arithmetic Operations

    @inlinable
    public static func add(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var i = 0

        // Process SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vb = SIMD8<Float>(
                b[i], b[i+1], b[i+2], b[i+3],
                b[i+4], b[i+5], b[i+6], b[i+7]
            )
            let vr = va + vb
            result[i] = vr[0]
            result[i+1] = vr[1]
            result[i+2] = vr[2]
            result[i+3] = vr[3]
            result[i+4] = vr[4]
            result[i+5] = vr[5]
            result[i+6] = vr[6]
            result[i+7] = vr[7]
            i += 8
        }

        // Process SIMD4 chunks
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let vb = SIMD4<Float>(b[i], b[i+1], b[i+2], b[i+3])
            let vr = va + vb
            result[i] = vr[0]
            result[i+1] = vr[1]
            result[i+2] = vr[2]
            result[i+3] = vr[3]
            i += 4
        }

        // Handle remaining elements
        while i < count {
            result[i] = a[i] + b[i]
            i += 1
        }
    }

    @inlinable
    public static func subtract(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var i = 0

        // Process SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vb = SIMD8<Float>(
                b[i], b[i+1], b[i+2], b[i+3],
                b[i+4], b[i+5], b[i+6], b[i+7]
            )
            let vr = va - vb
            result[i] = vr[0]
            result[i+1] = vr[1]
            result[i+2] = vr[2]
            result[i+3] = vr[3]
            result[i+4] = vr[4]
            result[i+5] = vr[5]
            result[i+6] = vr[6]
            result[i+7] = vr[7]
            i += 8
        }

        // Process SIMD4 chunks
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let vb = SIMD4<Float>(b[i], b[i+1], b[i+2], b[i+3])
            let vr = va - vb
            result[i] = vr[0]
            result[i+1] = vr[1]
            result[i+2] = vr[2]
            result[i+3] = vr[3]
            i += 4
        }

        // Handle remaining elements
        while i < count {
            result[i] = a[i] - b[i]
            i += 1
        }
    }

    @inlinable
    public static func multiply(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var i = 0

        // Process SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vb = SIMD8<Float>(
                b[i], b[i+1], b[i+2], b[i+3],
                b[i+4], b[i+5], b[i+6], b[i+7]
            )
            let vr = va * vb
            result[i] = vr[0]
            result[i+1] = vr[1]
            result[i+2] = vr[2]
            result[i+3] = vr[3]
            result[i+4] = vr[4]
            result[i+5] = vr[5]
            result[i+6] = vr[6]
            result[i+7] = vr[7]
            i += 8
        }

        // Process SIMD4 chunks
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let vb = SIMD4<Float>(b[i], b[i+1], b[i+2], b[i+3])
            let vr = va * vb
            result[i] = vr[0]
            result[i+1] = vr[1]
            result[i+2] = vr[2]
            result[i+3] = vr[3]
            i += 4
        }

        // Handle remaining elements
        while i < count {
            result[i] = a[i] * b[i]
            i += 1
        }
    }

    @inlinable
    public static func divide(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var i = 0

        // Process SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vb = SIMD8<Float>(
                b[i], b[i+1], b[i+2], b[i+3],
                b[i+4], b[i+5], b[i+6], b[i+7]
            )
            let vr = va / vb
            result[i] = vr[0]
            result[i+1] = vr[1]
            result[i+2] = vr[2]
            result[i+3] = vr[3]
            result[i+4] = vr[4]
            result[i+5] = vr[5]
            result[i+6] = vr[6]
            result[i+7] = vr[7]
            i += 8
        }

        // Process SIMD4 chunks
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let vb = SIMD4<Float>(b[i], b[i+1], b[i+2], b[i+3])
            let vr = va / vb
            result[i] = vr[0]
            result[i+1] = vr[1]
            result[i+2] = vr[2]
            result[i+3] = vr[3]
            i += 4
        }

        // Handle remaining elements
        while i < count {
            result[i] = a[i] / b[i]
            i += 1
        }
    }

    @inlinable
    public static func negate(
        _ a: UnsafePointer<Float>,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var i = 0

        // Process SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vr = -va
            result[i] = vr[0]
            result[i+1] = vr[1]
            result[i+2] = vr[2]
            result[i+3] = vr[3]
            result[i+4] = vr[4]
            result[i+5] = vr[5]
            result[i+6] = vr[6]
            result[i+7] = vr[7]
            i += 8
        }

        // Process SIMD4 chunks
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let vr = -va
            result[i] = vr[0]
            result[i+1] = vr[1]
            result[i+2] = vr[2]
            result[i+3] = vr[3]
            i += 4
        }

        // Handle remaining elements
        while i < count {
            result[i] = -a[i]
            i += 1
        }
    }

    // MARK: - Scalar Operations

    @inlinable
    public static func addScalar(
        _ a: UnsafePointer<Float>,
        scalar: Float,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var i = 0
        let scalarVec8 = SIMD8<Float>(repeating: scalar)
        let scalarVec4 = SIMD4<Float>(repeating: scalar)

        // Process SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vr = va + scalarVec8
            result[i] = vr[0]
            result[i+1] = vr[1]
            result[i+2] = vr[2]
            result[i+3] = vr[3]
            result[i+4] = vr[4]
            result[i+5] = vr[5]
            result[i+6] = vr[6]
            result[i+7] = vr[7]
            i += 8
        }

        // Process SIMD4 chunks
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let vr = va + scalarVec4
            result[i] = vr[0]
            result[i+1] = vr[1]
            result[i+2] = vr[2]
            result[i+3] = vr[3]
            i += 4
        }

        // Handle remaining elements
        while i < count {
            result[i] = a[i] + scalar
            i += 1
        }
    }

    @inlinable
    public static func multiplyScalar(
        _ a: UnsafePointer<Float>,
        scalar: Float,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var i = 0
        let scalarVec8 = SIMD8<Float>(repeating: scalar)
        let scalarVec4 = SIMD4<Float>(repeating: scalar)

        // Process SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vr = va * scalarVec8
            result[i] = vr[0]
            result[i+1] = vr[1]
            result[i+2] = vr[2]
            result[i+3] = vr[3]
            result[i+4] = vr[4]
            result[i+5] = vr[5]
            result[i+6] = vr[6]
            result[i+7] = vr[7]
            i += 8
        }

        // Process SIMD4 chunks
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let vr = va * scalarVec4
            result[i] = vr[0]
            result[i+1] = vr[1]
            result[i+2] = vr[2]
            result[i+3] = vr[3]
            i += 4
        }

        // Handle remaining elements
        while i < count {
            result[i] = a[i] * scalar
            i += 1
        }
    }

    @inlinable
    public static func divideByScalar(
        _ a: UnsafePointer<Float>,
        scalar: Float,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        let invScalar = 1.0 / scalar
        multiplyScalar(a, scalar: invScalar, result: result, count: count)
    }

    // MARK: - Reduction Operations

    @inlinable
    public static func dot(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var sum = SIMD8<Float>.zero
        var i = 0

        // Process in SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vb = SIMD8<Float>(
                b[i], b[i+1], b[i+2], b[i+3],
                b[i+4], b[i+5], b[i+6], b[i+7]
            )
            sum += va * vb
            i += 8
        }

        // Sum the SIMD8 vector
        var result = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7]

        // Process SIMD4
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let vb = SIMD4<Float>(b[i], b[i+1], b[i+2], b[i+3])
            let prod = va * vb
            result += prod[0] + prod[1] + prod[2] + prod[3]
            i += 4
        }

        // Handle remaining elements
        while i < count {
            result += a[i] * b[i]
            i += 1
        }

        return result
    }

    @inlinable
    public static func sum(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var sum = SIMD8<Float>.zero
        var i = 0

        // Process in SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            sum += va
            i += 8
        }

        // Sum the SIMD8 vector
        var result = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7]

        // Process SIMD4
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            result += va[0] + va[1] + va[2] + va[3]
            i += 4
        }

        // Handle remaining
        while i < count {
            result += a[i]
            i += 1
        }

        return result
    }

    @inlinable
    public static func sumOfMagnitudes(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var sum = SIMD8<Float>.zero
        var i = 0

        // Process in SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            sum += va.replacing(with: -va, where: va .< 0)
            i += 8
        }

        // Sum the SIMD8 vector
        var result = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7]

        // Process SIMD4
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let absVa = va.replacing(with: -va, where: va .< 0)
            result += absVa[0] + absVa[1] + absVa[2] + absVa[3]
            i += 4
        }

        // Handle remaining
        while i < count {
            result += abs(a[i])
            i += 1
        }

        return result
    }

    @inlinable
    public static func sumOfSquares(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var sum = SIMD8<Float>.zero
        var i = 0

        // Process in SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            sum += va * va
            i += 8
        }

        // Sum the SIMD8 vector
        var result = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7]

        // Process SIMD4
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let sqr = va * va
            result += sqr[0] + sqr[1] + sqr[2] + sqr[3]
            i += 4
        }

        // Handle remaining
        while i < count {
            result += a[i] * a[i]
            i += 1
        }

        return result
    }

    // MARK: - Statistical Operations

    @inlinable
    public static func maximum(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        guard count != 0 else { return 0 }

        var result = a[0]
        var i = 1

        // Process SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            result = max(result, va.max())
            i += 8
        }

        // Process SIMD4 chunks
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            result = max(result, va.max())
            i += 4
        }

        // Handle remaining elements
        while i < count {
            result = max(result, a[i])
            i += 1
        }

        return result
    }

    @inlinable
    public static func minimum(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        guard count != 0 else { return 0 }

        var result = a[0]
        var i = 1

        // Process SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            result = min(result, va.min())
            i += 8
        }

        // Process SIMD4 chunks
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            result = min(result, va.min())
            i += 4
        }

        // Handle remaining elements
        while i < count {
            result = min(result, a[i])
            i += 1
        }

        return result
    }

    @inlinable
    public static func maximumMagnitude(
        _ a: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        guard count != 0 else { return 0 }

        var result = abs(a[0])
        var i = 1

        // Process SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let absVa = va.replacing(with: -va, where: va .< 0)
            result = Swift.max(result, absVa.max())
            i += 8
        }

        // Process SIMD4 chunks
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let absVa = va.replacing(with: -va, where: va .< 0)
            result = Swift.max(result, absVa.max())
            i += 4
        }

        // Handle remaining elements
        while i < count {
            result = Swift.max(result, Swift.abs(a[i]))
            i += 1
        }

        return result
    }

    // MARK: - Distance Operations

    @inlinable
    public static func distanceSquared(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var sum = SIMD8<Float>.zero
        var i = 0

        // Process SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vb = SIMD8<Float>(
                b[i], b[i+1], b[i+2], b[i+3],
                b[i+4], b[i+5], b[i+6], b[i+7]
            )
            let diff = va - vb
            sum += diff * diff
            i += 8
        }

        // Sum the SIMD8 vector
        var result = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7]

        // Process SIMD4
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let vb = SIMD4<Float>(b[i], b[i+1], b[i+2], b[i+3])
            let diff = va - vb
            let sqr = diff * diff
            result += sqr[0] + sqr[1] + sqr[2] + sqr[3]
            i += 4
        }

        // Handle remaining elements
        while i < count {
            let diff = a[i] - b[i]
            result += diff * diff
            i += 1
        }

        return result
    }

    // MARK: - Utility Operations

    @inlinable
    public static func copy(
        source: UnsafePointer<Float>,
        destination: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        destination.initialize(from: source, count: count)
    }

    @inlinable
    public static func fill(
        value: Float,
        destination: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var i = 0
        let valueVec8 = SIMD8<Float>(repeating: value)
        let valueVec4 = SIMD4<Float>(repeating: value)

        // Process SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            destination[i] = valueVec8[0]
            destination[i+1] = valueVec8[1]
            destination[i+2] = valueVec8[2]
            destination[i+3] = valueVec8[3]
            destination[i+4] = valueVec8[4]
            destination[i+5] = valueVec8[5]
            destination[i+6] = valueVec8[6]
            destination[i+7] = valueVec8[7]
            i += 8
        }

        // Process SIMD4 chunks
        if i + 4 <= count {
            destination[i] = valueVec4[0]
            destination[i+1] = valueVec4[1]
            destination[i+2] = valueVec4[2]
            destination[i+3] = valueVec4[3]
            i += 4
        }

        // Handle remaining elements
        while i < count {
            destination[i] = value
            i += 1
        }
    }

    @inlinable
    public static func clip(
        _ a: UnsafePointer<Float>,
        low: Float,
        high: Float,
        result: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var i = 0
        let lowVec8 = SIMD8<Float>(repeating: low)
        let highVec8 = SIMD8<Float>(repeating: high)
        let lowVec4 = SIMD4<Float>(repeating: low)
        let highVec4 = SIMD4<Float>(repeating: high)

        // Process SIMD8 chunks
        let simd8Count = count & ~7
        while i < simd8Count {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let clipped = va.clamped(lowerBound: lowVec8, upperBound: highVec8)
            result[i] = clipped[0]
            result[i+1] = clipped[1]
            result[i+2] = clipped[2]
            result[i+3] = clipped[3]
            result[i+4] = clipped[4]
            result[i+5] = clipped[5]
            result[i+6] = clipped[6]
            result[i+7] = clipped[7]
            i += 8
        }

        // Process SIMD4 chunks
        if i + 4 <= count {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let clipped = va.clamped(lowerBound: lowVec4, upperBound: highVec4)
            result[i] = clipped[0]
            result[i+1] = clipped[1]
            result[i+2] = clipped[2]
            result[i+3] = clipped[3]
            i += 4
        }

        // Handle remaining elements
        while i < count {
            result[i] = min(max(a[i], low), high)
            i += 1
        }
    }
}
