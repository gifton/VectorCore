//
//  ArraySIMDProvider.swift
//  VectorCore
//
//  Protocol for array-based SIMD operations used by Operations.swift
//

import Foundation

/// Protocol for SIMD operations on arrays
///
/// This protocol provides a convenient array-based interface for SIMD operations,
/// used by the Operations API. Implementations wrap the lower-level pointer-based
/// SIMDProvider protocol.
public protocol ArraySIMDProvider: Sendable {

    // MARK: - Arithmetic Operations

    /// Add two arrays element-wise
    func add(_ a: [Float], _ b: [Float]) -> [Float]

    /// Subtract two arrays element-wise
    func subtract(_ a: [Float], _ b: [Float]) -> [Float]

    /// Multiply array by scalar
    func multiply(_ a: [Float], by scalar: Float) -> [Float]

    /// Divide array by scalar
    func divide(_ a: [Float], by scalar: Float) -> [Float]

    // MARK: - Reduction Operations

    /// Compute dot product of two arrays
    func dot(_ a: [Float], _ b: [Float]) -> Float

    /// Sum all elements
    func sum(_ a: [Float]) -> Float

    /// Find maximum value
    func max(_ a: [Float]) -> Float

    /// Find minimum value
    func min(_ a: [Float]) -> Float

    /// Compute mean (average) of all elements
    func mean(_ a: [Float]) -> Float

    // MARK: - Vector Operations

    /// Compute magnitude (L2 norm) of vector
    func magnitude(_ a: [Float]) -> Float

    /// Normalize vector to unit length
    func normalize(_ a: [Float]) -> [Float]

    /// Compute squared magnitude
    func magnitudeSquared(_ a: [Float]) -> Float

    // MARK: - Distance Operations

    /// Compute Euclidean distance squared
    func euclideanDistanceSquared(_ a: [Float], _ b: [Float]) -> Float

    /// Compute cosine similarity
    func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float

    // MARK: - Element-wise Operations

    /// Element-wise minimum
    func elementWiseMin(_ a: [Float], _ b: [Float]) -> [Float]

    /// Element-wise maximum
    func elementWiseMax(_ a: [Float], _ b: [Float]) -> [Float]

    /// Element-wise multiplication
    func elementWiseMultiply(_ a: [Float], _ b: [Float]) -> [Float]

    /// Element-wise division
    func elementWiseDivide(_ a: [Float], _ b: [Float]) -> [Float]

    /// Absolute value
    func abs(_ a: [Float]) -> [Float]

    /// Square root
    func sqrt(_ a: [Float]) -> [Float]

    /// Natural logarithm
    func log(_ a: [Float]) -> [Float]

    /// Clip values to range
    func clip(_ a: [Float], min: Float, max: Float) -> [Float]

    // MARK: - Index Operations

    /// Find index of minimum value
    func minIndex(_ a: [Float]) -> Int

    /// Find index of maximum value
    func maxIndex(_ a: [Float]) -> Int
}

/// Default implementation using SwiftFloatSIMDProvider
public struct DefaultArraySIMDProvider: ArraySIMDProvider, Sendable {

    public init() {}

    public func add(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        var result = [Float](repeating: 0, count: a.count)

        a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                result.withUnsafeMutableBufferPointer { resultBuffer in
                    SwiftFloatSIMDProvider.add(
                        aBuffer.baseAddress!,
                        bBuffer.baseAddress!,
                        result: resultBuffer.baseAddress!,
                        count: a.count
                    )
                }
            }
        }

        return result
    }

    public func subtract(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        var result = [Float](repeating: 0, count: a.count)

        a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                result.withUnsafeMutableBufferPointer { resultBuffer in
                    SwiftFloatSIMDProvider.subtract(
                        aBuffer.baseAddress!,
                        bBuffer.baseAddress!,
                        result: resultBuffer.baseAddress!,
                        count: a.count
                    )
                }
            }
        }

        return result
    }

    public func multiply(_ a: [Float], by scalar: Float) -> [Float] {
        var result = [Float](repeating: 0, count: a.count)

        a.withUnsafeBufferPointer { aBuffer in
            result.withUnsafeMutableBufferPointer { resultBuffer in
                SwiftFloatSIMDProvider.multiplyScalar(
                    aBuffer.baseAddress!,
                    scalar: scalar,
                    result: resultBuffer.baseAddress!,
                    count: a.count
                )
            }
        }

        return result
    }

    public func divide(_ a: [Float], by scalar: Float) -> [Float] {
        multiply(a, by: 1.0 / scalar)
    }

    public func dot(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Arrays must have same length")

        return a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                SwiftFloatSIMDProvider.dot(
                    aBuffer.baseAddress!,
                    bBuffer.baseAddress!,
                    count: a.count
                )
            }
        }
    }

    public func sum(_ a: [Float]) -> Float {
        a.withUnsafeBufferPointer { aBuffer in
            SwiftFloatSIMDProvider.sum(
                aBuffer.baseAddress!,
                count: a.count
            )
        }
    }

    public func max(_ a: [Float]) -> Float {
        guard !a.isEmpty else { return -.infinity }

        return a.withUnsafeBufferPointer { aBuffer in
            SwiftFloatSIMDProvider.maximum(
                aBuffer.baseAddress!,
                count: a.count
            )
        }
    }

    public func min(_ a: [Float]) -> Float {
        guard !a.isEmpty else { return .infinity }

        return a.withUnsafeBufferPointer { aBuffer in
            SwiftFloatSIMDProvider.minimum(
                aBuffer.baseAddress!,
                count: a.count
            )
        }
    }

    public func mean(_ a: [Float]) -> Float {
        guard !a.isEmpty else { return 0 }
        return sum(a) / Float(a.count)
    }

    public func magnitude(_ a: [Float]) -> Float {
        Foundation.sqrt(magnitudeSquared(a))
    }

    public func magnitudeSquared(_ a: [Float]) -> Float {
        a.withUnsafeBufferPointer { aBuffer in
            SwiftFloatSIMDProvider.sumOfSquares(
                aBuffer.baseAddress!,
                count: a.count
            )
        }
    }

    public func normalize(_ a: [Float]) -> [Float] {
        let mag = magnitude(a)
        guard mag > 0 else { return a }
        return multiply(a, by: 1.0 / mag)
    }

    public func euclideanDistanceSquared(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Arrays must have same length")

        return a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                SwiftFloatSIMDProvider.distanceSquared(
                    aBuffer.baseAddress!,
                    bBuffer.baseAddress!,
                    count: a.count
                )
            }
        }
    }

    public func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        let dotProduct = dot(a, b)
        let magA = magnitude(a)
        let magB = magnitude(b)

        guard magA > 0 && magB > 0 else { return 0 }
        return dotProduct / (magA * magB)
    }

    public func elementWiseMin(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        var result = [Float](repeating: 0, count: a.count)

        for i in 0..<a.count {
            result[i] = Swift.min(a[i], b[i])
        }

        return result
    }

    public func elementWiseMax(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        var result = [Float](repeating: 0, count: a.count)

        for i in 0..<a.count {
            result[i] = Swift.max(a[i], b[i])
        }

        return result
    }

    public func elementWiseMultiply(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        var result = [Float](repeating: 0, count: a.count)

        a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                result.withUnsafeMutableBufferPointer { resultBuffer in
                    SwiftFloatSIMDProvider.multiply(
                        aBuffer.baseAddress!,
                        bBuffer.baseAddress!,
                        result: resultBuffer.baseAddress!,
                        count: a.count
                    )
                }
            }
        }

        return result
    }

    public func elementWiseDivide(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        var result = [Float](repeating: 0, count: a.count)

        a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                result.withUnsafeMutableBufferPointer { resultBuffer in
                    SwiftFloatSIMDProvider.divide(
                        aBuffer.baseAddress!,
                        bBuffer.baseAddress!,
                        result: resultBuffer.baseAddress!,
                        count: a.count
                    )
                }
            }
        }

        return result
    }

    public func abs(_ a: [Float]) -> [Float] {
        a.map { Swift.abs($0) }
    }

    public func sqrt(_ a: [Float]) -> [Float] {
        a.map { Foundation.sqrt($0) }
    }

    public func log(_ a: [Float]) -> [Float] {
        a.map { Foundation.log($0) }
    }

    public func clip(_ a: [Float], min minVal: Float, max maxVal: Float) -> [Float] {
        var result = [Float](repeating: 0, count: a.count)

        a.withUnsafeBufferPointer { aBuffer in
            result.withUnsafeMutableBufferPointer { resultBuffer in
                SwiftFloatSIMDProvider.clip(
                    aBuffer.baseAddress!,
                    low: minVal,
                    high: maxVal,
                    result: resultBuffer.baseAddress!,
                    count: a.count
                )
            }
        }

        return result
    }

    public func minIndex(_ a: [Float]) -> Int {
        guard !a.isEmpty else { return 0 }

        var minIdx = 0
        var minVal = a[0]

        for i in 1..<a.count {
            if a[i] < minVal {
                minVal = a[i]
                minIdx = i
            }
        }

        return minIdx
    }

    public func maxIndex(_ a: [Float]) -> Int {
        guard !a.isEmpty else { return 0 }

        var maxIdx = 0
        var maxVal = a[0]

        for i in 1..<a.count {
            if a[i] > maxVal {
                maxVal = a[i]
                maxIdx = i
            }
        }

        return maxIdx
    }
}

// MARK: - Legacy SwiftSIMDProvider compatibility
//
// This type provides backward compatibility for code expecting the old
// SwiftSIMDProvider instance-based API

/// Legacy instance-based SIMD provider for backward compatibility
public struct SwiftSIMDProvider: ArraySIMDProvider, Sendable {
    private let provider = DefaultArraySIMDProvider()

    public init() {}

    public func add(_ a: [Float], _ b: [Float]) -> [Float] {
        provider.add(a, b)
    }

    public func subtract(_ a: [Float], _ b: [Float]) -> [Float] {
        provider.subtract(a, b)
    }

    public func multiply(_ a: [Float], by scalar: Float) -> [Float] {
        provider.multiply(a, by: scalar)
    }

    public func divide(_ a: [Float], by scalar: Float) -> [Float] {
        provider.divide(a, by: scalar)
    }

    public func dot(_ a: [Float], _ b: [Float]) -> Float {
        provider.dot(a, b)
    }

    public func sum(_ a: [Float]) -> Float {
        provider.sum(a)
    }

    public func max(_ a: [Float]) -> Float {
        provider.max(a)
    }

    public func min(_ a: [Float]) -> Float {
        provider.min(a)
    }

    public func mean(_ a: [Float]) -> Float {
        provider.mean(a)
    }

    public func magnitude(_ a: [Float]) -> Float {
        provider.magnitude(a)
    }

    public func magnitudeSquared(_ a: [Float]) -> Float {
        provider.magnitudeSquared(a)
    }

    public func normalize(_ a: [Float]) -> [Float] {
        provider.normalize(a)
    }

    public func euclideanDistanceSquared(_ a: [Float], _ b: [Float]) -> Float {
        provider.euclideanDistanceSquared(a, b)
    }

    public func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        provider.cosineSimilarity(a, b)
    }

    public func elementWiseMin(_ a: [Float], _ b: [Float]) -> [Float] {
        provider.elementWiseMin(a, b)
    }

    public func elementWiseMax(_ a: [Float], _ b: [Float]) -> [Float] {
        provider.elementWiseMax(a, b)
    }

    public func elementWiseMultiply(_ a: [Float], _ b: [Float]) -> [Float] {
        provider.elementWiseMultiply(a, b)
    }

    public func elementWiseDivide(_ a: [Float], _ b: [Float]) -> [Float] {
        provider.elementWiseDivide(a, b)
    }

    public func abs(_ a: [Float]) -> [Float] {
        provider.abs(a)
    }

    public func sqrt(_ a: [Float]) -> [Float] {
        provider.sqrt(a)
    }

    public func log(_ a: [Float]) -> [Float] {
        provider.log(a)
    }

    public func clip(_ a: [Float], min: Float, max: Float) -> [Float] {
        provider.clip(a, min: min, max: max)
    }

    public func minIndex(_ a: [Float]) -> Int {
        provider.minIndex(a)
    }

    public func maxIndex(_ a: [Float]) -> Int {
        provider.maxIndex(a)
    }
}
