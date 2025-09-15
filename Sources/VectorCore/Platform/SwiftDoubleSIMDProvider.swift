//
//  SwiftDoubleSIMDProvider.swift
//  VectorCore
//
//  Pure Swift implementation of SIMDProvider for Double
//  Cross-platform SIMD operations using Swift's built-in SIMD types
//

import Foundation

/// Pure Swift implementation for Double operations
public struct SwiftDoubleSIMDProvider: SIMDProvider {
    public typealias Scalar = Double

    // MARK: - Arithmetic Operations

    @inlinable
    public static func add(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        processBinaryOperation(a, b, result: result, count: count) { $0 + $1 }
    }

    @inlinable
    public static func subtract(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        processBinaryOperation(a, b, result: result, count: count) { $0 - $1 }
    }

    @inlinable
    public static func multiply(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        processBinaryOperation(a, b, result: result, count: count) { $0 * $1 }
    }

    @inlinable
    public static func divide(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        processBinaryOperation(a, b, result: result, count: count) { $0 / $1 }
    }

    @inlinable
    public static func multiplyScalar(
        _ a: UnsafePointer<Double>,
        scalar: Double,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        processUnaryOperation(a, result: result, count: count) { $0 * scalar }
    }

    @inlinable
    public static func negate(
        _ a: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        processUnaryOperation(a, result: result, count: count) { -$0 }
    }

    @inlinable
    public static func addScalar(
        _ a: UnsafePointer<Double>,
        scalar: Double,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        processUnaryOperation(a, result: result, count: count) { $0 + scalar }
    }

    // MARK: - Reduction Operations

    @inlinable
    public static func sum(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        var result: Double = 0
        var i = 0

        // Process SIMD8 chunks (Double uses smaller SIMD sizes)
        while i + 8 <= count {
            let va = SIMD8<Double>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            result += va.sum()
            i += 8
        }

        // Process SIMD4 chunks
        while i + 4 <= count {
            let va = SIMD4<Double>(a[i], a[i+1], a[i+2], a[i+3])
            result += va.sum()
            i += 4
        }

        // Process remaining scalars
        while i < count {
            result += a[i]
            i += 1
        }

        return result
    }

    @inlinable
    public static func dot(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        var result: Double = 0
        var i = 0

        // Process SIMD8 chunks (Double uses smaller SIMD sizes)
        while i + 8 <= count {
            let va = SIMD8<Double>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vb = SIMD8<Double>(
                b[i], b[i+1], b[i+2], b[i+3],
                b[i+4], b[i+5], b[i+6], b[i+7]
            )
            result += (va * vb).sum()
            i += 8
        }

        // Process SIMD4 chunks
        while i + 4 <= count {
            let va = SIMD4<Double>(a[i], a[i+1], a[i+2], a[i+3])
            let vb = SIMD4<Double>(b[i], b[i+1], b[i+2], b[i+3])
            result += (va * vb).sum()
            i += 4
        }

        // Process remaining scalars
        while i < count {
            result += a[i] * b[i]
            i += 1
        }

        return result
    }

    @inlinable
    public static func sumOfSquares(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        processReduction(a, count: count, initial: 0) { sum, vec in
            sum + (vec * vec).sum()
        }
    }

    @inlinable
    public static func sumOfMagnitudes(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        processReduction(a, count: count, initial: 0) { sum, vec in
            sum + vec.replacing(with: 0, where: vec .< 0).sum() - vec.replacing(with: 0, where: vec .>= 0).sum()
        }
    }

    @inlinable
    public static func distanceSquared(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        var result: Double = 0
        var i = 0

        // Process SIMD8 chunks
        while i + 8 <= count {
            let va = SIMD8<Double>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vb = SIMD8<Double>(
                b[i], b[i+1], b[i+2], b[i+3],
                b[i+4], b[i+5], b[i+6], b[i+7]
            )
            let diff = va - vb
            result += (diff * diff).sum()
            i += 8
        }

        // Process SIMD4 chunks
        while i + 4 <= count {
            let va = SIMD4<Double>(a[i], a[i+1], a[i+2], a[i+3])
            let vb = SIMD4<Double>(b[i], b[i+1], b[i+2], b[i+3])
            let diff = va - vb
            result += (diff * diff).sum()
            i += 4
        }

        // Process remaining scalars
        while i < count {
            let diff = a[i] - b[i]
            result += diff * diff
            i += 1
        }

        return result
    }

    // MARK: - Statistical Operations

    @inlinable
    public static func maximum(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        guard count > 0 else { return 0 }

        var result = a[0]
        var i = 1

        // Process SIMD8 chunks
        while i + 8 <= count {
            let va = SIMD8<Double>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            result = Swift.max(result, va.max())
            i += 8
        }

        // Process remaining scalars
        while i < count {
            result = Swift.max(result, a[i])
            i += 1
        }

        return result
    }

    @inlinable
    public static func minimum(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        guard count > 0 else { return 0 }

        var result = a[0]
        var i = 1

        // Process SIMD8 chunks
        while i + 8 <= count {
            let va = SIMD8<Double>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            result = Swift.min(result, va.min())
            i += 8
        }

        // Process remaining scalars
        while i < count {
            result = Swift.min(result, a[i])
            i += 1
        }

        return result
    }

    @inlinable
    public static func maximumMagnitude(
        _ a: UnsafePointer<Double>,
        count: Int
    ) -> Double {
        guard count > 0 else { return 0 }

        var result = abs(a[0])
        var i = 1

        // Process SIMD8 chunks
        while i + 8 <= count {
            let va = SIMD8<Double>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let absVa = va.replacing(with: -va, where: va .< 0)
            result = Swift.max(result, absVa.max())
            i += 8
        }

        // Process remaining scalars
        while i < count {
            result = Swift.max(result, abs(a[i]))
            i += 1
        }

        return result
    }

    // MARK: - Utility Operations

    @inlinable
    public static func fill(
        value: Double,
        destination: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        var i = 0
        let valueVec8 = SIMD8<Double>(repeating: value)
        let valueVec4 = SIMD4<Double>(repeating: value)
        let valueVec2 = SIMD2<Double>(repeating: value)

        // Process SIMD8 chunks
        while i + 8 <= count {
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
        while i + 4 <= count {
            destination[i] = valueVec4[0]
            destination[i+1] = valueVec4[1]
            destination[i+2] = valueVec4[2]
            destination[i+3] = valueVec4[3]
            i += 4
        }

        // Process SIMD2 chunks
        while i + 2 <= count {
            destination[i] = valueVec2[0]
            destination[i+1] = valueVec2[1]
            i += 2
        }

        // Process remaining scalars
        while i < count {
            destination[i] = value
            i += 1
        }
    }

    @inlinable
    public static func copy(
        source: UnsafePointer<Double>,
        destination: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        destination.initialize(from: source, count: count)
    }

    // MARK: - Extended Operations

    @inlinable
    public static func clip(
        _ a: UnsafePointer<Double>,
        low: Double,
        high: Double,
        result: UnsafeMutablePointer<Double>,
        count: Int
    ) {
        var i = 0
        let lowVec8 = SIMD8<Double>(repeating: low)
        let highVec8 = SIMD8<Double>(repeating: high)
        let lowVec4 = SIMD4<Double>(repeating: low)
        let highVec4 = SIMD4<Double>(repeating: high)

        // Process SIMD8 chunks
        while i + 8 <= count {
            let va = SIMD8<Double>(
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
        while i + 4 <= count {
            let va = SIMD4<Double>(a[i], a[i+1], a[i+2], a[i+3])
            let clipped = va.clamped(lowerBound: lowVec4, upperBound: highVec4)
            result[i] = clipped[0]
            result[i+1] = clipped[1]
            result[i+2] = clipped[2]
            result[i+3] = clipped[3]
            i += 4
        }

        // Process remaining scalars
        while i < count {
            result[i] = Swift.max(low, Swift.min(high, a[i]))
            i += 1
        }
    }

    // MARK: - Private Helpers

    @inline(__always)
    @usableFromInline
    internal static func processBinaryOperation(
        _ a: UnsafePointer<Double>,
        _ b: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int,
        operation: (SIMD8<Double>, SIMD8<Double>) -> SIMD8<Double>
    ) {
        var i = 0

        // Process SIMD8 chunks
        while i + 8 <= count {
            let va = SIMD8<Double>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vb = SIMD8<Double>(
                b[i], b[i+1], b[i+2], b[i+3],
                b[i+4], b[i+5], b[i+6], b[i+7]
            )
            let vr = operation(va, vb)
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
        while i + 4 <= count {
            let va = SIMD4<Double>(a[i], a[i+1], a[i+2], a[i+3])
            let vb = SIMD4<Double>(b[i], b[i+1], b[i+2], b[i+3])
            // Note: Cast to SIMD8 for uniform operation handling
            let va8 = SIMD8<Double>(lowHalf: va, highHalf: va)
            let vb8 = SIMD8<Double>(lowHalf: vb, highHalf: vb)
            let result8 = operation(va8, vb8)
            let lowHalf = result8.lowHalf
            result[i] = lowHalf[0]
            result[i+1] = lowHalf[1]
            result[i+2] = lowHalf[2]
            result[i+3] = lowHalf[3]
            i += 4
        }

        // Process SIMD2 chunks
        while i + 2 <= count {
            let va = SIMD2<Double>(a[i], a[i+1])
            let vb = SIMD2<Double>(b[i], b[i+1])
            // Note: Cast to SIMD8 for uniform operation handling
            let va8 = SIMD8<Double>(va[0], va[1], va[0], va[1], va[0], va[1], va[0], va[1])
            let vb8 = SIMD8<Double>(vb[0], vb[1], vb[0], vb[1], vb[0], vb[1], vb[0], vb[1])
            let result8 = operation(va8, vb8)
            let firstQuarter = result8.lowHalf.lowHalf
            result[i] = firstQuarter[0]
            result[i+1] = firstQuarter[1]
            i += 2
        }

        // Process remaining scalars
        while i < count {
            let scalarA = a[i]
            let scalarB = b[i]
            // Create a SIMD8 for single operation
            let va8 = SIMD8<Double>(repeating: scalarA)
            let vb8 = SIMD8<Double>(repeating: scalarB)
            let result8 = operation(va8, vb8)
            result[i] = result8[0]
            i += 1
        }
    }

    @inline(__always)
    @usableFromInline
    internal static func processUnaryOperation(
        _ a: UnsafePointer<Double>,
        result: UnsafeMutablePointer<Double>,
        count: Int,
        operation: (SIMD8<Double>) -> SIMD8<Double>
    ) {
        var i = 0

        // Process SIMD8 chunks
        while i + 8 <= count {
            let va = SIMD8<Double>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vr = operation(va)
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
        while i + 4 <= count {
            let va = SIMD4<Double>(a[i], a[i+1], a[i+2], a[i+3])
            // Note: Cast to SIMD8 for uniform operation handling
            let va8 = SIMD8<Double>(lowHalf: va, highHalf: va)
            let result8 = operation(va8)
            let lowHalf = result8.lowHalf
            result[i] = lowHalf[0]
            result[i+1] = lowHalf[1]
            result[i+2] = lowHalf[2]
            result[i+3] = lowHalf[3]
            i += 4
        }

        // Process remaining scalars
        while i < count {
            let scalarA = a[i]
            let va8 = SIMD8<Double>(repeating: scalarA)
            let result8 = operation(va8)
            result[i] = result8[0]
            i += 1
        }
    }

    @inline(__always)
    @usableFromInline
    internal static func processReduction<T>(
        _ a: UnsafePointer<Double>,
        count: Int,
        initial: T,
        operation: (T, SIMD8<Double>) -> T
    ) -> T {
        var result = initial
        var i = 0

        // Process SIMD8 chunks
        while i + 8 <= count {
            let va = SIMD8<Double>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            result = operation(result, va)
            i += 8
        }

        // Process SIMD4 chunks
        while i + 4 <= count {
            let va = SIMD4<Double>(a[i], a[i+1], a[i+2], a[i+3])
            let va8 = SIMD8<Double>(lowHalf: va, highHalf: SIMD4<Double>(repeating: 0))
            result = operation(result, va8)
            i += 4
        }

        // Process remaining scalars
        while i < count {
            let va8 = SIMD8<Double>(a[i], 0, 0, 0, 0, 0, 0, 0)
            result = operation(result, va8)
            i += 1
        }

        return result
    }
}
