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

/// Shared finite nonnegative inverse interpretation. The widened Double
/// expression gives a single final rounding to Float; SearchResultScoreContractTests
/// exercises the finite domain, signed zero, rejection matrix and positive endpoint.
private func inverseDistanceRelevance(_ distance: Float) -> Float? {
    guard distance.isFinite, distance >= 0 else { return nil }
    return Float(1 / (1 + Double(distance)))
}

// MARK: - Euclidean Distance

/// Euclidean (L2) distance metric with SIMD acceleration
public struct EuclideanDistance: DistanceMetric {
    public typealias Scalar = Float

    public var name: String { "euclidean" }
    public var identifier: String { "euclidean" }

    public init() {}

    /// Inverse-distance relevance for finite nonnegative input, otherwise nil.
    /// This interpretation does not change the metric's distance computation.
    public func relevanceScore(for distance: Float) -> Float? {
        inverseDistanceRelevance(distance)
    }

    /// Compute Euclidean distance between two vectors
    ///
    /// `Vector512Optimized`, `Vector768Optimized` and `Vector1536Optimized` use `EuclideanKernels`, the arithmetic of
    /// the per-candidate Euclidean search routes on those types, so a concrete call, this witness and search outside the
    /// GEMM routes return the same result (`MetricWitnessEquivalenceTests`, `MetricSearchWitnessConsistencyTests`).
    /// The GEMM routes are a different arithmetic and are not covered: `Operations.findNearestBatch` with the explicit
    /// `BatchSearchCPUStrategy.gram` (its default, `.perQuery`, runs the per-candidate routes), and the off-diagonal
    /// entries of `BatchOperations.pairwiseDistances` when `enableMatrixRouting` is set and the count reaches
    /// `matrixRoutingMinN`. An installed `BatchKernelProvider`
    /// returns its own results from `Operations`. Other vector types use `SIMDOperations.FloatProvider.distanceSquared`.
    /// - Precondition: Both vectors must have the same dimension
    @inlinable
    public func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        #if DEBUG
        assert(a.scalarCount == b.scalarCount, "Vector dimension mismatch: \(a.scalarCount) != \(b.scalarCount)")
        #endif

        if let a = a as? Vector512Optimized, let b = b as? Vector512Optimized { return EuclideanKernels.distance512(a, b) }
        if let a = a as? Vector768Optimized, let b = b as? Vector768Optimized { return EuclideanKernels.distance768(a, b) }
        if let a = a as? Vector1536Optimized, let b = b as? Vector1536Optimized { return EuclideanKernels.distance1536(a, b) }

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
    ///
    /// For `Vector512Optimized`, `Vector768Optimized` and `Vector1536Optimized`, each result is `distance(query, $0)`.
    /// - Precondition: Query and all candidates must have the same dimension
    public func batchDistance<Vector: VectorProtocol>(
        query: Vector,
        candidates: [Vector]
    ) -> [Scalar] where Vector.Scalar == Float {
        #if DEBUG
        for candidate in candidates {
            assert(query.scalarCount == candidate.scalarCount,
                   "Dimension mismatch in batch: query has \(query.scalarCount), candidate has \(candidate.scalarCount)")
        }
        #endif

        if query is Vector512Optimized || query is Vector768Optimized || query is Vector1536Optimized {
            return candidates.map { distance(query, $0) }
        }

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

    /// Inverse-distance relevance for finite nonnegative input, otherwise nil.
    /// This interpretation does not change the metric's distance computation.
    public func relevanceScore(for distance: Float) -> Float? {
        inverseDistanceRelevance(distance)
    }

    /// Compute cosine distance between two vectors
    ///
    /// `Vector512Optimized`, `Vector768Optimized` and `Vector1536Optimized` use `CosineKernels.distance*_fused`, the
    /// arithmetic of every cosine search route on those types (`BatchKernels.range_cosine_fused_*` and the 512 Top-K
    /// kernel), so a concrete call, this witness and search outside the GEMM routes return the same result
    /// (`MetricWitnessEquivalenceTests`, `MetricSearchWitnessConsistencyTests`). On those types a squared norm at or
    /// below `Float.leastNormalMagnitude` makes a vector a zero vector (`CosineKernels.calculateCosineDistance`), so a
    /// tiny vector against a normal one gives 1.0; other types keep the product guard below (T15's per-family zero
    /// classes, `T15CosineExceptionalTableTests`). The GEMM routes are a different arithmetic and are not covered:
    /// `Operations.findNearestBatch` with the explicit `BatchSearchCPUStrategy.gram` (its default, `.perQuery`, runs the
    /// per-candidate routes), and the off-diagonal entries of `BatchOperations.pairwiseDistances` when
    /// `enableMatrixRouting` is set and the count reaches `matrixRoutingMinN`. An installed `BatchKernelProvider` returns its own results from `Operations`. Other vector
    /// types use `SIMDOperations.FloatProvider`.
    /// - Precondition: Both vectors must have the same dimension
    @inlinable
    public func distance<Vector: VectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore where Vector.Scalar == Float {
        #if DEBUG
        assert(a.scalarCount == b.scalarCount, "Vector dimension mismatch: \(a.scalarCount) != \(b.scalarCount)")
        #endif

        if let a = a as? Vector512Optimized, let b = b as? Vector512Optimized { return CosineKernels.distance512_fused(a, b) }
        if let a = a as? Vector768Optimized, let b = b as? Vector768Optimized { return CosineKernels.distance768_fused(a, b) }
        if let a = a as? Vector1536Optimized, let b = b as? Vector1536Optimized { return CosineKernels.distance1536_fused(a, b) }

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

        // Handle zero vectors.
        // Compute the product of magnitudes as sqrt(aMagnitudeSq) * sqrt(bMagnitudeSq)
        // rather than sqrt(aMagnitudeSq * bMagnitudeSq) to avoid intermediate overflow
        // for large-magnitude vectors.
        //
        // The guard threshold must be an *absolute* floor, not the relative precision
        // around 1.0. Using Float.ulpOfOne (~1.19e-7) wrongly rejects valid micro-vectors:
        // two unit-direction vectors of magnitude 1e-4 yield a product of ~1e-8, which is
        // below ulpOfOne yet far above the smallest representable normal magnitude.
        // Float.leastNormalMagnitude (~1.18e-38) is the correct floor: it excludes only
        // genuine zero (or subnormal) vectors where the cosine is undefined.
        let magnitudeProduct = sqrt(aMagnitudeSq) * sqrt(bMagnitudeSq)
        guard magnitudeProduct > Float.leastNormalMagnitude else {
            // Maximum distance for zero vectors; a NaN or infinite squared norm dominates the zero rule (NaN)
            return aMagnitudeSq.isFinite && bMagnitudeSq.isFinite ? 1.0 : .nan
        }

        // Cosine similarity, clamped to [-1, 1] to handle finite rounding error. A non-finite similarity (a NaN or
        // infinite coordinate) gives NaN (T15CosineExceptionalTableTests, witness rows).
        let rawSimilarity = dotProduct / magnitudeProduct
        guard rawSimilarity.isFinite else { return .nan }
        let cosineSimilarity = max(-1, min(1, rawSimilarity))

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

    /// Inverse-distance relevance for finite nonnegative input, otherwise nil.
    /// This interpretation does not change the metric's distance computation.
    public func relevanceScore(for distance: Float) -> Float? {
        inverseDistanceRelevance(distance)
    }

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

    /// Logistic relevance of negative-dot distance, in [0, 1] for finite input.
    /// Float rounding may saturate; the mapping is weakly decreasing, not strictly
    /// ordered or a calibrated probability. Nonfinite input has no score.
    /// SearchResultScoreContractTests.dotLogisticDomain checks poles and endpoints.
    public func relevanceScore(for distance: Float) -> Float? {
        guard distance.isFinite else { return nil }
        if distance >= 0 {
            let exponential = Foundation.exp(-Double(distance))
            return Float(exponential / (1 + exponential))
        }
        let exponential = Foundation.exp(Double(distance))
        return Float(1 / (1 + exponential))
    }

    /// Compute negative dot product as distance
    ///
    /// `Vector512Optimized`, `Vector768Optimized` and `Vector1536Optimized` use `DotKernels`, the arithmetic of the
    /// dot-product search routes on those types, so a concrete call, this witness and search return the same result
    /// (`MetricWitnessEquivalenceTests`, `MetricSearchWitnessConsistencyTests`); the GEMM routes take only Euclidean
    /// and cosine. An installed `BatchKernelProvider` returns its own results from `Operations`. Other vector types use
    /// `SIMDOperations.FloatProvider.dot`.
    /// - Precondition: Both vectors must have the same dimension
    @inlinable
    public func distance<Vector: VectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore where Vector.Scalar == Float {
        #if DEBUG
        assert(a.scalarCount == b.scalarCount, "Vector dimension mismatch: \(a.scalarCount) != \(b.scalarCount)")
        #endif

        if let a = a as? Vector512Optimized, let b = b as? Vector512Optimized { return -DotKernels.dot512(a, b) }
        if let a = a as? Vector768Optimized, let b = b as? Vector768Optimized { return -DotKernels.dot768(a, b) }
        if let a = a as? Vector1536Optimized, let b = b as? Vector1536Optimized { return -DotKernels.dot1536(a, b) }

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

    /// Inverse-distance relevance for finite nonnegative input, otherwise nil.
    /// This interpretation does not change the metric's distance computation.
    public func relevanceScore(for distance: Float) -> Float? {
        inverseDistanceRelevance(distance)
    }

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
                // Find maximum absolute difference directly. A NaN difference (a NaN coordinate, or same-sign
                // infinities) makes the distance NaN: `max(result, NaN)` would drop it
                // (T15MetricExceptionalTableTests.everyMetricPinsItsExceptionalClass).
                for i in 0..<aBuffer.count {
                    let diff = abs(aBuffer[i] - bBuffer[i])
                    if diff.isNaN {
                        result = .nan
                        return
                    }
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

    /// Inverse-distance relevance for finite nonnegative input, otherwise nil.
    /// This interpretation does not change the metric's distance computation.
    public func relevanceScore(for distance: Float) -> Float? {
        inverseDistanceRelevance(distance)
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
                    // A NaN coordinate has no bit value: the distance is NaN, not a 0 bit
                    // (T15MetricExceptionalTableTests.everyMetricPinsItsExceptionalClass).
                    if aBuffer[i].isNaN || bBuffer[i].isNaN {
                        count = .nan
                        return
                    }
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

    /// Inverse-distance relevance for finite nonnegative input, otherwise nil.
    /// This interpretation does not change the metric's distance computation.
    public func relevanceScore(for distance: Float) -> Float? {
        inverseDistanceRelevance(distance)
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
