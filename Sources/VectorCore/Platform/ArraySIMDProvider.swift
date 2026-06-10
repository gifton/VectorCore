//
//  ArraySIMDProvider.swift
//  VectorCore
//
//  Protocol for array-based SIMD operations used by Operations.swift
//

import Foundation

#if canImport(Accelerate)
import Accelerate
#endif

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

    /// Allocate an output array of `count` floats **without** the redundant zero-fill.
    ///
    /// Every arithmetic/element-wise/transcendental op below fully overwrites its result
    /// buffer via the underlying SIMD/vForce primitive, so `[Float](repeating: 0, count:)`
    /// pays for a whole-buffer `memset` that is immediately discarded. `Array`'s
    /// `unsafeUninitializedCapacity:` initializer hands us raw storage and skips the
    /// zero-fill entirely — the closure is responsible for writing all `count` elements.
    ///
    /// - Important: `body` MUST initialize all `count` elements of the buffer. Empty
    ///   requests short-circuit to `[]` so the SIMD primitives never see a nil base pointer.
    @inline(__always)
    private func uninitializedResult(
        count: Int,
        _ body: (UnsafeMutableBufferPointer<Float>) -> Void
    ) -> [Float] {
        guard count > 0 else { return [] }
        return [Float](unsafeUninitializedCapacity: count) { buffer, initializedCount in
            body(buffer)
            initializedCount = count
        }
    }

    public func add(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        let count = a.count
        return uninitializedResult(count: count) { resultBuffer in
            a.withUnsafeBufferPointer { aBuffer in
                b.withUnsafeBufferPointer { bBuffer in
                    SwiftFloatSIMDProvider.add(
                        aBuffer.baseAddress!,
                        bBuffer.baseAddress!,
                        result: resultBuffer.baseAddress!,
                        count: count
                    )
                }
            }
        }
    }

    public func subtract(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        let count = a.count
        return uninitializedResult(count: count) { resultBuffer in
            a.withUnsafeBufferPointer { aBuffer in
                b.withUnsafeBufferPointer { bBuffer in
                    SwiftFloatSIMDProvider.subtract(
                        aBuffer.baseAddress!,
                        bBuffer.baseAddress!,
                        result: resultBuffer.baseAddress!,
                        count: count
                    )
                }
            }
        }
    }

    public func multiply(_ a: [Float], by scalar: Float) -> [Float] {
        let count = a.count
        return uninitializedResult(count: count) { resultBuffer in
            a.withUnsafeBufferPointer { aBuffer in
                SwiftFloatSIMDProvider.multiplyScalar(
                    aBuffer.baseAddress!,
                    scalar: scalar,
                    result: resultBuffer.baseAddress!,
                    count: count
                )
            }
        }
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
        let count = a.count
        return uninitializedResult(count: count) { resultBuffer in
            a.withUnsafeBufferPointer { aBuffer in
                b.withUnsafeBufferPointer { bBuffer in
                    for i in 0..<count {
                        resultBuffer[i] = Swift.min(aBuffer[i], bBuffer[i])
                    }
                }
            }
        }
    }

    public func elementWiseMax(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        let count = a.count
        return uninitializedResult(count: count) { resultBuffer in
            a.withUnsafeBufferPointer { aBuffer in
                b.withUnsafeBufferPointer { bBuffer in
                    for i in 0..<count {
                        resultBuffer[i] = Swift.max(aBuffer[i], bBuffer[i])
                    }
                }
            }
        }
    }

    public func elementWiseMultiply(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        let count = a.count
        return uninitializedResult(count: count) { resultBuffer in
            a.withUnsafeBufferPointer { aBuffer in
                b.withUnsafeBufferPointer { bBuffer in
                    SwiftFloatSIMDProvider.multiply(
                        aBuffer.baseAddress!,
                        bBuffer.baseAddress!,
                        result: resultBuffer.baseAddress!,
                        count: count
                    )
                }
            }
        }
    }

    public func elementWiseDivide(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        let count = a.count
        return uninitializedResult(count: count) { resultBuffer in
            a.withUnsafeBufferPointer { aBuffer in
                b.withUnsafeBufferPointer { bBuffer in
                    SwiftFloatSIMDProvider.divide(
                        aBuffer.baseAddress!,
                        bBuffer.baseAddress!,
                        result: resultBuffer.baseAddress!,
                        count: count
                    )
                }
            }
        }
    }

    public func abs(_ a: [Float]) -> [Float] {
        let count = a.count
        guard count > 0 else { return [] }
        #if canImport(Accelerate)
        return uninitializedResult(count: count) { dst in
            var n = Int32(count)
            a.withUnsafeBufferPointer { src in
                vvfabsf(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        #else
        return a.map { Swift.abs($0) }
        #endif
    }

    public func sqrt(_ a: [Float]) -> [Float] {
        let count = a.count
        guard count > 0 else { return [] }
        #if canImport(Accelerate)
        return uninitializedResult(count: count) { dst in
            var n = Int32(count)
            a.withUnsafeBufferPointer { src in
                vvsqrtf(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        #else
        return a.map { Foundation.sqrt($0) }
        #endif
    }

    public func log(_ a: [Float]) -> [Float] {
        let count = a.count
        guard count > 0 else { return [] }
        #if canImport(Accelerate)
        return uninitializedResult(count: count) { dst in
            var n = Int32(count)
            a.withUnsafeBufferPointer { src in
                vvlogf(dst.baseAddress!, src.baseAddress!, &n)
            }
        }
        #else
        return a.map { Foundation.log($0) }
        #endif
    }

    public func clip(_ a: [Float], min minVal: Float, max maxVal: Float) -> [Float] {
        let count = a.count
        return uninitializedResult(count: count) { resultBuffer in
            a.withUnsafeBufferPointer { aBuffer in
                SwiftFloatSIMDProvider.clip(
                    aBuffer.baseAddress!,
                    low: minVal,
                    high: maxVal,
                    result: resultBuffer.baseAddress!,
                    count: count
                )
            }
        }
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

// MARK: - Accelerate-backed ArraySIMDProvider

#if canImport(Accelerate)
/// `ArraySIMDProvider` whose hot arithmetic and reductions run on Accelerate's
/// vDSP via ``AccelerateFloatProvider``, with the handful of vDSP-uncovered
/// operations (`elementWiseMin`/`elementWiseMax`, `abs`/`sqrt`/`log`,
/// `minIndex`/`maxIndex`) delegated to ``DefaultArraySIMDProvider`` (already
/// vForce-backed or trivial index scans).
///
/// This is the Accelerate analog of ``DefaultArraySIMDProvider``: the bodies are
/// structurally identical, only the backing static-provider type changes from
/// `SwiftFloatSIMDProvider` to `AccelerateFloatProvider`. Install it for the
/// `Operations` API (and `Vector<D>` array math that flows through it) via the
/// scoped `Operations.$simdProvider.withValue(_:operation:)` API.
///
/// - Note: vDSP reductions (`vDSP_sve`, `vDSP_dotpr`, `vDSP_svesq`, …) use a
///   pairwise/tree summation, so results agree with the sequential Swift
///   provider only to within floating-point rounding — never bit-for-bit.
public struct AccelerateArraySIMDProvider: ArraySIMDProvider, Sendable {

    /// Backing provider for the operations vDSP does not cover. These are either
    /// vForce-backed (`abs`/`sqrt`/`log`) or trivial scalar scans
    /// (`elementWiseMin`/`elementWiseMax`/`minIndex`/`maxIndex`), so delegating
    /// them introduces no "fake Accelerate" dishonesty.
    private let fallback = DefaultArraySIMDProvider()

    public init() {}

    /// Allocate an output array of `count` floats **without** the redundant zero-fill.
    ///
    /// Mirrors ``DefaultArraySIMDProvider``'s allocation idiom so allocation
    /// behavior is identical across the two providers: every arithmetic /
    /// element-wise op below fully overwrites its result buffer via the
    /// underlying vDSP primitive, so `Array`'s `unsafeUninitializedCapacity:`
    /// initializer hands us raw storage and skips the zero-fill entirely.
    ///
    /// - Important: `body` MUST initialize all `count` elements of the buffer.
    ///   Empty requests short-circuit to `[]` so the vDSP primitives never see a
    ///   nil base pointer.
    @inline(__always)
    private func uninitializedResult(
        count: Int,
        _ body: (UnsafeMutableBufferPointer<Float>) -> Void
    ) -> [Float] {
        guard count > 0 else { return [] }
        return [Float](unsafeUninitializedCapacity: count) { buffer, initializedCount in
            body(buffer)
            initializedCount = count
        }
    }

    public func add(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        let count = a.count
        return uninitializedResult(count: count) { resultBuffer in
            a.withUnsafeBufferPointer { aBuffer in
                b.withUnsafeBufferPointer { bBuffer in
                    AccelerateFloatProvider.add(
                        aBuffer.baseAddress!,
                        bBuffer.baseAddress!,
                        result: resultBuffer.baseAddress!,
                        count: count
                    )
                }
            }
        }
    }

    public func subtract(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        let count = a.count
        return uninitializedResult(count: count) { resultBuffer in
            a.withUnsafeBufferPointer { aBuffer in
                b.withUnsafeBufferPointer { bBuffer in
                    AccelerateFloatProvider.subtract(
                        aBuffer.baseAddress!,
                        bBuffer.baseAddress!,
                        result: resultBuffer.baseAddress!,
                        count: count
                    )
                }
            }
        }
    }

    public func multiply(_ a: [Float], by scalar: Float) -> [Float] {
        let count = a.count
        return uninitializedResult(count: count) { resultBuffer in
            a.withUnsafeBufferPointer { aBuffer in
                AccelerateFloatProvider.multiplyScalar(
                    aBuffer.baseAddress!,
                    scalar: scalar,
                    result: resultBuffer.baseAddress!,
                    count: count
                )
            }
        }
    }

    public func divide(_ a: [Float], by scalar: Float) -> [Float] {
        multiply(a, by: 1.0 / scalar)
    }

    public func dot(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Arrays must have same length")

        return a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                AccelerateFloatProvider.dot(
                    aBuffer.baseAddress!,
                    bBuffer.baseAddress!,
                    count: a.count
                )
            }
        }
    }

    public func sum(_ a: [Float]) -> Float {
        a.withUnsafeBufferPointer { aBuffer in
            AccelerateFloatProvider.sum(
                aBuffer.baseAddress!,
                count: a.count
            )
        }
    }

    public func max(_ a: [Float]) -> Float {
        guard !a.isEmpty else { return -.infinity }

        return a.withUnsafeBufferPointer { aBuffer in
            AccelerateFloatProvider.maximum(
                aBuffer.baseAddress!,
                count: a.count
            )
        }
    }

    public func min(_ a: [Float]) -> Float {
        guard !a.isEmpty else { return .infinity }

        return a.withUnsafeBufferPointer { aBuffer in
            AccelerateFloatProvider.minimum(
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
            AccelerateFloatProvider.sumOfSquares(
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
                AccelerateFloatProvider.distanceSquared(
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
        // vDSP has no element-wise min; delegate to the trivial scalar-scan impl.
        fallback.elementWiseMin(a, b)
    }

    public func elementWiseMax(_ a: [Float], _ b: [Float]) -> [Float] {
        // vDSP has no element-wise max; delegate to the trivial scalar-scan impl.
        fallback.elementWiseMax(a, b)
    }

    public func elementWiseMultiply(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        let count = a.count
        return uninitializedResult(count: count) { resultBuffer in
            a.withUnsafeBufferPointer { aBuffer in
                b.withUnsafeBufferPointer { bBuffer in
                    AccelerateFloatProvider.multiply(
                        aBuffer.baseAddress!,
                        bBuffer.baseAddress!,
                        result: resultBuffer.baseAddress!,
                        count: count
                    )
                }
            }
        }
    }

    public func elementWiseDivide(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Arrays must have same length")
        let count = a.count
        return uninitializedResult(count: count) { resultBuffer in
            a.withUnsafeBufferPointer { aBuffer in
                b.withUnsafeBufferPointer { bBuffer in
                    AccelerateFloatProvider.divide(
                        aBuffer.baseAddress!,
                        bBuffer.baseAddress!,
                        result: resultBuffer.baseAddress!,
                        count: count
                    )
                }
            }
        }
    }

    public func abs(_ a: [Float]) -> [Float] {
        // vForce-backed (vvfabsf) inside DefaultArraySIMDProvider.
        fallback.abs(a)
    }

    public func sqrt(_ a: [Float]) -> [Float] {
        // vForce-backed (vvsqrtf) inside DefaultArraySIMDProvider.
        fallback.sqrt(a)
    }

    public func log(_ a: [Float]) -> [Float] {
        // vForce-backed (vvlogf) inside DefaultArraySIMDProvider.
        fallback.log(a)
    }

    public func clip(_ a: [Float], min minVal: Float, max maxVal: Float) -> [Float] {
        let count = a.count
        return uninitializedResult(count: count) { resultBuffer in
            a.withUnsafeBufferPointer { aBuffer in
                AccelerateFloatProvider.clip(
                    aBuffer.baseAddress!,
                    low: minVal,
                    high: maxVal,
                    result: resultBuffer.baseAddress!,
                    count: count
                )
            }
        }
    }

    public func minIndex(_ a: [Float]) -> Int {
        // Trivial scalar scan; delegate to the shared implementation.
        fallback.minIndex(a)
    }

    public func maxIndex(_ a: [Float]) -> Int {
        // Trivial scalar scan; delegate to the shared implementation.
        fallback.maxIndex(a)
    }
}
#endif // canImport(Accelerate)
