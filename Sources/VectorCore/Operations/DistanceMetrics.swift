//
//  DistanceMetrics.swift
//  VectorCore
//
//
//  Design Note: Distance metrics do NOT validate dimension compatibility.
//  This is by design — dimension validation happens at the public API layer
//  (e.g., Operations/BatchOperations) to avoid redundant checks in hot paths.
//  Debug assertions are included for development-time safety.
//

import Foundation
import simd

// MARK: - Distance Score

/// Type alias for distance scores
public typealias DistanceScore = Float

// MARK: - Euclidean Distance

/// Euclidean (L2) distance metric with SIMD acceleration
public struct EuclideanDistance: DistanceMetric {
    public typealias Scalar = Float

    public var name: String { "euclidean" }
    public var identifier: String { "euclidean" }

    public init() {}

    /// Compute Euclidean distance between two vectors
    /// - Precondition: Both vectors must have the same dimension
    @inlinable
    public func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        #if DEBUG
        assert(a.scalarCount == b.scalarCount, "Vector dimension mismatch: \(a.scalarCount) != \(b.scalarCount)")
        #endif

        return a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                let result = SIMDOperations.FloatProvider.distanceSquared(
                    aBuffer.baseAddress!,
                    bBuffer.baseAddress!,
                    count: aBuffer.count
                )
                return sqrt(result)
            }
        }
    }

    /// Optimized batch distance computation
    /// - Precondition: Query and all candidates must have the same dimension
    public func batchDistance<Vector: VectorProtocol>(
        query: Vector,
        candidates: [Vector]
    ) -> [DistanceScore] where Vector.Scalar == Float {
        #if DEBUG
        for candidate in candidates {
            assert(query.scalarCount == candidate.scalarCount,
                   "Dimension mismatch in batch: query has \(query.scalarCount), candidate has \(candidate.scalarCount)")
        }
        #endif

        var results = [DistanceScore](repeating: 0, count: candidates.count)

        query.withUnsafeBufferPointer { queryBuffer in
            for (index, candidate) in candidates.enumerated() {
                candidate.withUnsafeBufferPointer { candidateBuffer in
                    let distance = SIMDOperations.FloatProvider.distanceSquared(
                        queryBuffer.baseAddress!,
                        candidateBuffer.baseAddress!,
                        count: queryBuffer.count
                    )
                    results[index] = sqrt(distance)
                }
            }
        }

        return results
    }
}

// MARK: - Cosine Distance

/// Cosine distance metric (1 - cosine similarity)
public struct CosineDistance: DistanceMetric {
    public var name: String { "cosine" }
    public let identifier = "cosine"

    public init() {}

    /// Compute cosine distance between two vectors
    /// - Precondition: Both vectors must have the same dimension
    @inlinable
    public func distance<Vector: VectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore where Vector.Scalar == Float {
        #if DEBUG
        assert(a.scalarCount == b.scalarCount, "Vector dimension mismatch: \(a.scalarCount) != \(b.scalarCount)")
        #endif

        var dotProduct: Float = 0
        var aMagnitudeSq: Float = 0
        var bMagnitudeSq: Float = 0

        a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                // Dot product
                dotProduct = SIMDOperations.FloatProvider.dot(
                    aBuffer.baseAddress!,
                    bBuffer.baseAddress!,
                    count: aBuffer.count
                )

                // Magnitude squared of a
                aMagnitudeSq = SIMDOperations.FloatProvider.sumOfSquares(
                    aBuffer.baseAddress!,
                    count: aBuffer.count
                )

                // Magnitude squared of b
                bMagnitudeSq = SIMDOperations.FloatProvider.sumOfSquares(
                    bBuffer.baseAddress!,
                    count: bBuffer.count
                )
            }
        }

        // Handle zero vectors
        let magnitudeProduct = sqrt(aMagnitudeSq * bMagnitudeSq)
        guard magnitudeProduct > Float.ulpOfOne else {
            return 1.0 // Maximum distance for zero vectors
        }

        // Cosine similarity, clamped to [-1, 1] to handle numerical errors
        let cosineSimilarity = max(-1, min(1, dotProduct / magnitudeProduct))

        // Convert to distance (1 - similarity)
        return 1.0 - cosineSimilarity
    }
}

// MARK: - Manhattan Distance

/// Manhattan (L1) distance metric with SIMD acceleration
public struct ManhattanDistance: DistanceMetric {
    public var name: String { "manhattan" }
    public let identifier = "manhattan"

    public init() {}

    /// Compute Manhattan distance between two vectors
    ///
    /// Uses SIMD4 vectorization for improved performance on generic vectors.
    /// Processes 4 elements per iteration with a scalar remainder loop.
    ///
    /// - Precondition: Both vectors must have the same dimension
    /// - Complexity: O(n) where n is the vector dimension
    @inlinable
    public func distance<Vector: VectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore where Vector.Scalar == Float {
        #if DEBUG
        assert(a.scalarCount == b.scalarCount, "Vector dimension mismatch: \(a.scalarCount) != \(b.scalarCount)")
        #endif

        var result: Float = 0

        a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                let count = aBuffer.count
                let simdCount = count / 4
                let remainder = count % 4

                // SIMD4 vectorized path: process 4 elements per iteration
                var acc = SIMD4<Float>.zero
                for i in 0..<simdCount {
                    let offset = i * 4
                    let a4 = SIMD4<Float>(
                        aBuffer[offset],
                        aBuffer[offset + 1],
                        aBuffer[offset + 2],
                        aBuffer[offset + 3]
                    )
                    let b4 = SIMD4<Float>(
                        bBuffer[offset],
                        bBuffer[offset + 1],
                        bBuffer[offset + 2],
                        bBuffer[offset + 3]
                    )
                    acc += abs(a4 - b4)
                }
                result = acc.sum()

                // Scalar remainder loop for dimensions not divisible by 4
                let remainderStart = simdCount * 4
                for i in 0..<remainder {
                    result += abs(aBuffer[remainderStart + i] - bBuffer[remainderStart + i])
                }
            }
        }

        return result
    }
}

// MARK: - Dot Product Distance

/// Dot product distance (negative dot product for similarity)
public struct DotProductDistance: DistanceMetric {
    public var name: String { "dotproduct" }
    public let identifier = "dotproduct"

    public init() {}

    /// Compute negative dot product as distance
    /// - Precondition: Both vectors must have the same dimension
    @inlinable
    public func distance<Vector: VectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore where Vector.Scalar == Float {
        #if DEBUG
        assert(a.scalarCount == b.scalarCount, "Vector dimension mismatch: \(a.scalarCount) != \(b.scalarCount)")
        #endif

        var result: Float = 0

        a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                result = SIMDOperations.FloatProvider.dot(
                    aBuffer.baseAddress!,
                    bBuffer.baseAddress!,
                    count: aBuffer.count
                )
            }
        }

        // Return negative for distance (higher dot product = smaller distance)
        return -result
    }
}

// MARK: - Chebyshev Distance

/// Chebyshev (L-infinity) distance metric
public struct ChebyshevDistance: DistanceMetric {
    public var name: String { "chebyshev" }
    public let identifier = "chebyshev"

    public init() {}

    /// Compute Chebyshev distance (maximum absolute difference)
    /// - Precondition: Both vectors must have the same dimension
    @inlinable
    public func distance<Vector: VectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore where Vector.Scalar == Float {
        #if DEBUG
        assert(a.scalarCount == b.scalarCount, "Vector dimension mismatch: \(a.scalarCount) != \(b.scalarCount)")
        #endif

        var result: Float = 0

        a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                // Find maximum absolute difference directly
                for i in 0..<aBuffer.count {
                    let diff = abs(aBuffer[i] - bBuffer[i])
                    result = max(result, diff)
                }
            }
        }

        return result
    }
}

// MARK: - Hamming Distance

/// Hamming distance for binary vectors
public struct HammingDistance: DistanceMetric {
    public var name: String { "hamming" }
    public let identifier = "hamming"

    /// Threshold for considering a value as 1 (vs 0)
    public let threshold: Float

    public init(threshold: Float = 0.5) {
        self.threshold = threshold
    }

    /// Compute Hamming distance (number of differing positions)
    /// - Precondition: Both vectors must have the same dimension
    @inlinable
    public func distance<Vector: VectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore where Vector.Scalar == Float {
        #if DEBUG
        assert(a.scalarCount == b.scalarCount, "Vector dimension mismatch: \(a.scalarCount) != \(b.scalarCount)")
        #endif

        var count: Float = 0

        a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                for i in 0..<aBuffer.count {
                    let aBit = aBuffer[i] > threshold ? 1 : 0
                    let bBit = bBuffer[i] > threshold ? 1 : 0
                    if aBit != bBit {
                        count += 1
                    }
                }
            }
        }

        return count
    }
}

// MARK: - Minkowski Distance

/// Generalized Minkowski distance (L_p norm)
public struct MinkowskiDistance: DistanceMetric {
    public let identifier: String
    public let p: Float
    public var name: String { identifier }

    public init(p: Float = 2.0) {
        self.p = p
        self.identifier = "minkowski_\(p)"
    }

    /// Compute Minkowski distance
    /// - Precondition: Both vectors must have the same dimension
    @inlinable
    public func distance<Vector: VectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore where Vector.Scalar == Float {
        #if DEBUG
        assert(a.scalarCount == b.scalarCount, "Vector dimension mismatch: \(a.scalarCount) != \(b.scalarCount)")
        #endif

        // Special cases
        if p == 1 {
            return ManhattanDistance().distance(a, b)
        } else if p == 2 {
            return EuclideanDistance().distance(a, b)
        } else if p == .infinity {
            return ChebyshevDistance().distance(a, b)
        }

        var sum: Float = 0

        a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                for i in 0..<aBuffer.count {
                    let diff = abs(aBuffer[i] - bBuffer[i])
                    sum += pow(diff, p)
                }
            }
        }

        return pow(sum, 1.0 / p)
    }
}
