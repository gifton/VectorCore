// VectorCore: Vector Math Operations
//
// Additional mathematical operations for vectors
//

import Foundation
import simd

#if canImport(Accelerate)
import Accelerate
#endif

// MARK: - Element-wise Operations

extension Vector {
    /// Element-wise multiplication (Hadamard product)
    ///
    /// Computes: cᵢ = aᵢ × bᵢ for all i
    ///
    /// Implementation:
    /// - Copies both operands into arrays and calls `Operations.simdProvider.elementWiseMultiply`
    ///   (the default `SwiftSIMDProvider` is pure Swift; `AccelerateArraySIMDProvider` is opt-in)
    /// - Speed relative to a scalar loop is not measured in this repository
    ///
    /// Use cases:
    /// - Feature scaling and masking
    /// - Neural network gating mechanisms
    /// - Signal processing (windowing)
    /// - Probability computations
    @inlinable
    public static func .* (lhs: Vector<D>, rhs: Vector<D>) -> Vector<D> {
        let lhsArray = lhs.toArray()
        let rhsArray = rhs.toArray()
        let result = Operations.simdProvider.elementWiseMultiply(lhsArray, rhsArray)
        return try! Vector<D>(result)
    }

    /// Element-wise division
    ///
    /// Computes: cᵢ = aᵢ / bᵢ for all i
    ///
    /// Performance characteristics:
    /// - Calls `Operations.simdProvider.elementWiseDivide` (pure Swift by default)
    /// - Note: Division is slower than multiplication
    /// - Consider reciprocal multiplication for repeated operations
    ///
    /// IMPORTANT: No zero-check performed for performance
    /// - Caller must ensure rhs has no zero elements
    /// - Results in ±Inf for division by zero
    @inlinable
    public static func ./ (lhs: Vector<D>, rhs: Vector<D>) -> Vector<D> {
        let lhsArray = lhs.toArray()
        let rhsArray = rhs.toArray()
        let result = Operations.simdProvider.elementWiseDivide(lhsArray, rhsArray)
        return try! Vector<D>(result)
    }

    /// Safe element-wise division with zero checking
    ///
    /// Computes: cᵢ = aᵢ / bᵢ for all i, with zero protection
    ///
    /// - Parameters:
    ///   - lhs: Dividend vector
    ///   - rhs: Divisor vector
    /// - Throws: `VectorError.divisionByZero` if any element in rhs is zero
    /// - Returns: Result of element-wise division
    ///
    /// Performance note: Adds zero-checking overhead compared to `./`
    public static func safeDivide(_ lhs: Vector<D>, by rhs: Vector<D>) throws -> Vector<D> {
        // Check for zeros in divisor
        var hasZero = false
        var zeroIndices: [Int] = []

        rhs.storage.withUnsafeBufferPointer { buffer in
            for i in 0..<D.value {
                if buffer[i] == 0 {
                    hasZero = true
                    zeroIndices.append(i)
                }
            }
        }

        if hasZero {
            throw VectorError.divisionByZero(operation: "safeDivide")
        }

        // Perform division
        return lhs ./ rhs
    }

    /// Safe element-wise division with default value for division by zero
    ///
    /// Computes: cᵢ = (bᵢ ≠ 0) ? aᵢ / bᵢ : defaultValue
    ///
    /// - Parameters:
    ///   - lhs: Dividend vector
    ///   - rhs: Divisor vector
    ///   - defaultValue: Value to use when divisor is zero (default: 0)
    /// - Returns: Result with defaultValue where division by zero would occur
    ///
    /// Use cases:
    /// - Masked operations where zeros indicate "ignore"
    /// - Regularized division with safe fallback
    @inlinable
    public static func safeDivide(
        _ lhs: Vector<D>,
        by rhs: Vector<D>,
        default defaultValue: Float = 0
    ) -> Vector<D> {
        var result = Vector<D>()

        result.storage.withUnsafeMutableBufferPointer { dest in
            lhs.storage.withUnsafeBufferPointer { src1 in
                rhs.storage.withUnsafeBufferPointer { src2 in
                    for i in 0..<D.value {
                        dest[i] = src2[i] != 0 ? src1[i] / src2[i] : defaultValue
                    }
                }
            }
        }

        return result
    }
}

// MARK: - Norms

extension Vector {
    /// L1 norm (Manhattan norm)
    ///
    /// Computes: Σ|xᵢ|
    ///
    /// Route (not a speed claim): up to 32 elements, a scalar loop over the storage; above 32, the elements go
    /// through `Operations.simdProvider.abs` and are summed left to right with `reduce`
    /// (D02DimensionProviderDispatchTests, D02ProviderHookRouteTests).
    ///
    /// Mathematical properties:
    /// - Always non-negative
    /// - Triangle inequality: ||x+y||₁ ≤ ||x||₁ + ||y||₁
    /// - Grows linearly with each component's magnitude (no squaring)
    @inlinable
    public var l1Norm: Float {
        var result: Float = 0
        storage.withUnsafeBufferPointer { buffer in
            // Small vector optimization
            if D.value <= 32 {
                for i in 0..<D.value {
                    result += Swift.abs(buffer[i])
                }
            } else {
                // Large vector: use SIMD provider
                let array = self.toArray()
                let absArray = Operations.simdProvider.abs(array)
                result = absArray.reduce(0, +)
            }
        }
        return result
    }

    /// L2 norm (Euclidean norm) - alias for magnitude
    @inlinable
    public var l2Norm: Float {
        magnitude
    }

    /// L∞ norm (Maximum norm)
    ///
    /// Computes: max(|xᵢ|)
    ///
    /// Route (not a speed claim): a NaN scan, then `Operations.simdProvider.abs` and `Array.max()`; an empty vector
    /// gives 0 (D02ProviderHookRouteTests, D02DegenerateWidthTests).
    ///
    /// Mathematical properties:
    /// - Defines hypercube neighborhoods
    /// - Dual norm to L¹ (sum norm)
    /// - Submultiplicative: ||xy||∞ ≤ ||x||∞||y||∞
    ///
    /// NaN propagation: `Float.nan` when any element is NaN, as `l1Norm` and `magnitude` give NaN (`Array.max()`
    /// alone keeps or drops a NaN depending on its position). Pinned through `chebyshevDistance(to:)` by
    /// T15MetricExceptionalTableTests.everyMetricPinsItsExceptionalClass.
    @inlinable
    public var lInfinityNorm: Float {
        let array = self.toArray()
        if array.contains(where: { $0.isNaN }) { return .nan }
        let absArray = Operations.simdProvider.abs(array)
        return absArray.max() ?? 0
    }
}

// MARK: - Statistical Operations

extension Vector {
    /// Mean value of all elements
    @inlinable
    public var mean: Float {
        let array = self.toArray()
        return Operations.simdProvider.mean(array)
    }

    /// Sum of all elements
    @inlinable
    public var sum: Float {
        let array = self.toArray()
        return Operations.simdProvider.sum(array)
    }

    /// Variance of all elements
    ///
    /// Computes: σ² = (1/n)Σ(xᵢ - μ)²
    ///
    /// Algorithm: two passes (the mean first, then squared deviations from it)
    /// 1. Compute mean μ
    /// 2. Compute sum of squared deviations
    ///
    /// Performance characteristics:
    /// - Two passes over data (mean + variance)
    /// - Temporary buffer for differences
    /// - More stable than single-pass algorithm
    ///
    /// Note: Uses sample variance (n divisor), not population (n-1)
    @inlinable
    public var variance: Float {
        let m = mean
        var result: Float = 0

        // Compute sum of squared differences from mean
        let array = self.toArray()
        let diffs = array.map { $0 - m }
        let squaredDiffs = diffs.map { $0 * $0 }
        result = squaredDiffs.reduce(0, +)

        return result / Float(D.value)
    }

    /// Standard deviation
    @inlinable
    public var standardDeviation: Float {
        Foundation.sqrt(variance)
    }
}

// MARK: - Distance Metrics

extension Vector {
    /// Manhattan distance to another vector
    ///
    /// Optimized implementation that fuses operations for better performance
    @inlinable
    public func manhattanDistance(to other: Vector<D>) -> Float {
        var result: Float = 0

        storage.withUnsafeBufferPointer { buffer1 in
            other.storage.withUnsafeBufferPointer { buffer2 in
                // Small vector optimization: avoid allocation
                if D.value <= 32 {
                    for i in 0..<D.value {
                        result += Swift.abs(buffer1[i] - buffer2[i])
                    }
                } else {
                    // Large vector: use SIMD provider
                    let arr1 = self.toArray()
                    let arr2 = other.toArray()
                    let diffs = Operations.simdProvider.subtract(arr1, arr2)
                    let absDiffs = Operations.simdProvider.abs(diffs)
                    result = absDiffs.reduce(0, +)
                }
            }
        }

        return result
    }

    /// Chebyshev distance (L∞ distance)
    @inlinable
    public func chebyshevDistance(to other: Vector<D>) -> Float {
        (self - other).lInfinityNorm
    }
}

// MARK: - Special Operations

extension Vector {
    /// Apply softmax to the vector
    ///
    /// Computes: softmax(xᵢ) = exp(xᵢ - max(x)) / Σexp(xⱼ - max(x))
    ///
    /// Algorithm: subtract the maximum before exponentiating (the log-sum-exp shift)
    /// 1. Subtract maximum value to prevent overflow
    /// 2. Apply exponential function
    /// 3. Normalize by sum
    ///
    /// Performance characteristics:
    /// - Three passes: max, exp, normalize
    /// - vvexpf: Vectorized exponential
    /// - Stable for large positive/negative values
    ///
    /// Mathematical properties:
    /// - Output sums to 1.0
    /// - All outputs in (0, 1)
    /// - Preserves order: xᵢ > xⱼ ⇒ softmax(xᵢ) > softmax(xⱼ)
    ///
    /// Use cases:
    /// - Neural network output layers
    /// - Probability distribution from logits
    /// - Attention weight computation
    @inlinable
    public func softmax() -> Vector<D> {
        var result = self

        // Find the maximum: after subtracting it no finite shifted value is positive, so exp cannot overflow
        let array = self.toArray()
        let maxVal = Operations.simdProvider.max(array)

        // Subtract max and exp
        let shifted = array.map { $0 - maxVal }
        #if canImport(Accelerate)
        var expValues = [Float](repeating: 0, count: shifted.count)
        if !shifted.isEmpty {
            var n = Int32(shifted.count)
            shifted.withUnsafeBufferPointer { src in
                expValues.withUnsafeMutableBufferPointer { dst in
                    vvexpf(dst.baseAddress!, src.baseAddress!, &n)
                }
            }
        }
        #else
        let expValues = shifted.map { Foundation.exp($0) }
        #endif

        // Sum for normalization
        let sum = expValues.reduce(0, +)

        // Normalize
        let normalized = Operations.simdProvider.divide(expValues, by: sum)
        result = try! Vector<D>(normalized)

        return result
    }

    /// Clamp values to a range
    ///
    /// Computes: clamp(xᵢ, min, max) = max(min, min(max, xᵢ))
    ///
    /// Route (not a speed claim): the elements go through `Operations.simdProvider.clip`, and the result is a new
    /// vector (D02ProviderHookRouteTests).
    ///
    /// Use cases:
    /// - Activation function bounds (e.g., hard sigmoid)
    /// - Pixel value normalization [0, 1]
    /// - Gradient clipping for training stability
    /// - Outlier suppression
    @inlinable
    public func clamped(to range: ClosedRange<Float>) -> Vector<D> {
        var result = self

        let array = result.toArray()
        let clamped = Operations.simdProvider.clip(array, min: range.lowerBound, max: range.upperBound)
        result = try! Vector<D>(clamped)

        return result
    }
}

// MARK: - Batch Operations

/// Advanced mathematical operations for vectors.
///
/// `VectorMath` provides a collection of sophisticated vector operations
/// beyond basic arithmetic, including nearest neighbor search, clustering
/// support, and statistical computations. Accelerate use is not verified per
/// operation here, and `nearestNeighbors(query:in:k:using:)` does not use
/// Accelerate: it calls the supplied metric closure and selects in Swift.
///
/// ## Categories
/// - **Nearest Neighbor Search**: Find similar vectors (`nearestNeighbors(query:in:k:using:)`)
/// - **Clustering Support**: Operations for k-means and related algorithms
/// - **Statistical Operations**: Mean, variance, and other statistics
/// - **Matrix Operations**: Batch vector operations
///
/// ## Performance Notes
/// - vDSP use is not verified per operation; `nearestNeighbors(query:in:k:using:)` does not use vDSP
/// - Operations are designed for batch processing
/// - Consider approximate methods for very large datasets
public enum VectorMath {
    /// Find the k nearest neighbors of `query` in `vectors` under `metric`.
    ///
    /// Order: ascending numeric distances first, then NaN distances; exact ties
    /// (including +0 and -0, and two NaNs) keep the smaller original index. This
    /// is the policy of `TopKSelection.select(k:from:tieBreaker:)` with the default
    /// `.smallerIndex`. Each entry carries the candidate's original index and the
    /// metric's value for it, unmodified (zero sign and NaN payload included).
    ///
    /// Count and metric calls: for positive `k` and nonempty `vectors`, `metric`
    /// runs exactly once per candidate, in increasing index order (NaN and losing
    /// values included), and never during the final extraction or sort; on the
    /// small-k path each distance is offered to the bounded heap as it is computed.
    /// The result has `min(k, vectors.count)` entries (NaN distances fill it when
    /// needed). `k <= 0`, including negative `k`, and empty `vectors` return `[]`
    /// without calling `metric`.
    ///
    /// Strategy and complexity, with n = `vectors.count` and k' = min(k, n):
    /// - k' < n / 10 (integer division): n metric calls, a bounded heap of k'
    ///   entries (O(n log k') comparisons), then a sort of the k' retained entries;
    ///   O(k') working arrays.
    /// - otherwise: n metric calls, then a sort of all n (index, distance) pairs,
    ///   O(n log n); O(n) working arrays.
    /// The scan is serial and does not use Accelerate.
    ///
    /// - Important: Changed behaviour (breaking; see CHANGELOG). Results used to
    ///   come from a sort with `<` on the distance, which left NaN positions
    ///   unspecified; `k == 0` called the metric for every candidate, and a negative
    ///   `k` trapped after calling it for every candidate.
    ///
    /// Tested by `VectorMathNearestContractTests` and `TopKSelectorOracleTests`.
    ///
    /// Use cases:
    /// - k-NN classification/regression
    /// - Similarity search
    /// - Recommendation systems
    /// - Clustering algorithms
    public static func nearestNeighbors<D>(
        query: Vector<D>,
        in vectors: [Vector<D>],
        k: Int,
        using metric: (Vector<D>, Vector<D>) -> Float
    ) -> [(index: Int, distance: Float)] {
        TopKSelection.selectByScore(count: vectors.count, k: k, tieBreaker: .smallerIndex) {
            metric(query, vectors[$0])
        }
    }

    /// Compute pairwise distances between all vectors
    ///
    /// Computes: D[i,j] = metric(vectors[i], vectors[j])
    ///
    /// Algorithm: Symmetric matrix computation
    /// - Only compute upper triangle (i < j)
    /// - Mirror to lower triangle
    /// - Diagonal is zero (distance to self)
    ///
    /// Performance characteristics:
    /// - O(n²) distance computations
    /// - Exploits symmetry: d(a,b) = d(b,a)
    /// - Memory: O(n²) for result matrix
    /// - Consider chunking for large n
    ///
    /// Use cases:
    /// - Hierarchical clustering
    /// - MDS (multidimensional scaling)
    /// - Graph construction
    /// - Correlation analysis
    public static func pairwiseDistances<D>(
        _ vectors: [Vector<D>],
        using metric: (Vector<D>, Vector<D>) -> Float
    ) -> [[Float]] {
        let n = vectors.count
        var distances = Array(repeating: Array(repeating: Float(0), count: n), count: n)

        for i in 0..<n {
            for j in i+1..<n {
                let dist = metric(vectors[i], vectors[j])
                distances[i][j] = dist
                distances[j][i] = dist
            }
        }

        return distances
    }
}

// MARK: - Vector Creation Helpers

extension Vector {
    /// Create a vector filled with ones
    public static var one: Vector<D> {
        Vector(repeating: 1)
    }
}

// MARK: - Quality Metrics

extension Vector {
    /// Calculate sparsity (proportion of near-zero elements)
    ///
    /// - Parameter threshold: Values with absolute value <= threshold are considered zero
    /// - Returns: Proportion of sparse elements (0.0 = dense, 1.0 = all zeros). At width 0 the proportion is 0/0,
    ///   so the result is NaN (D02DegenerateWidthTests; behaviour unchanged by D02, now documented).
    ///
    /// Use cases:
    /// - Compression decisions (a sparse format stores only the nonzero entries)
    /// - Quality assessment (very sparse vectors may indicate issues)
    /// - Feature selection (identify uninformative features)
    @inlinable
    public func sparsity(threshold: Float = Float.ulpOfOne) -> Float {
        var sparseCount = 0
        for i in 0..<D.value {
            let value = self[i]
            // Non-finite values (NaN, Infinity) are not considered sparse
            if value.isFinite && Swift.abs(value) <= threshold {
                sparseCount += 1
            }
        }
        return Float(sparseCount) / Float(D.value)
    }

    /// Truncated entropy of the normalized magnitude distribution
    ///
    /// Treats the vector as a probability distribution $p_i = |x_i| / \sum_j |x_j|$.
    /// Returns $\sum_{p_i > \varepsilon_{Float}} -p_i \ln p_i$ (the documented truncated entropy;
    /// terms with $p_i \le$ `Float.ulpOfOne` are omitted, but those coordinates stay in the
    /// denominator). The cutoff is on the normalized probability, never on the input scale.
    ///
    /// Returns:
    /// - `+0` for all-zero (including signed-zero) or single-spike vectors
    /// - NaN if any coordinate is NaN or infinite
    /// - $\ln D$ for uniform nonzero magnitudes only while each $p_i = 1/D$ exceeds
    ///   `Float.ulpOfOne` (i.e. $D < 2^{23}$); for $D \ge 2^{23}$ every term is omitted and the
    ///   truncated entropy is `+0`. The $\ln D$ claim is tested at widths 1, 4, 8, 63, 64, 65, 128,
    ///   1536 only, by `EntropyRangeContractTests.uniformMagnitudesReturnLogD`; the $D \ge 2^{23}$ case is
    ///   derived from the cutoff and not exercised by a test (unverified).
    ///
    /// For finite coordinates, with $p_i = |x_i| / \sum_j |x_j|$:
    ///
    /// $$H = \mathrm{Float}\Big(\sum_{p_i > \varepsilon_{\mathrm{Float}}} -p_i \ln p_i\Big)$$
    ///
    /// - The cutoff `p_i > Float.ulpOfOne` applies to the *normalized probability only*.
    ///   Coordinates at or below it are omitted from the numerator sum but remain in the
    ///   denominator. This is the documented truncated entropy, not exact Shannon entropy
    ///   for arbitrarily small probability mass. There is no cutoff on the input sum.
    /// - Coordinates are widened exactly to `Double`; magnitudes are summed with
    ///   Neumaier-compensated `Double` arithmetic and entropy terms accumulate in `Double`;
    ///   only the final result is rounded to `Float`.
    /// - All-zero input (including signed zeros) returns `+0`. Any NaN or +/-infinity
    ///   coordinate returns NaN. Signs do not affect finite results.
    /// - Complexity: two passes, O(D) time, O(1) auxiliary storage.
    ///
    /// Verified by `EntropyRangeContractTests` (uniform ln(D) across the Float exponent
    /// range, spikes, cutoff table, non-finite and signed-zero cases) and `VectorEntropySuite`.
    /// The test tolerance (8 * ulpOfOne * max(1, |reference|)) is a regression bound, not an accuracy guarantee.
    /// `entropyFast` is a deprecated spelling of this property (same computation and value).
    @inlinable
    public var entropy: Float {
        // Both width branches run the shared contract implementation (VectorEntropy.swift).
        if D.value <= 64 {
            return entropySmall()
        } else {
            return entropyLarge()
        }
    }

    // TODO: Implement VectorQuality type
    // /// Comprehensive quality assessment
    // ///
    // /// Returns a VectorQuality struct containing multiple metrics for
    // /// assessing vector characteristics and quality.
    // public var quality: VectorQuality {
    //     VectorQuality(
    //         magnitude: magnitude,
    //         variance: variance,
    //         sparsity: sparsity(),
    //         entropy: entropy
    //     )
    // }
}

// MARK: - Serialization

extension Vector {
    /// Base64-encoded representation of the vector
    ///
    /// Uses the binary encoding format with CRC32 checksum for data integrity.
    /// Useful for:
    /// - Transmitting vectors over text-based protocols
    /// - Storing vectors in JSON/XML
    /// - Embedding vectors in URLs
    public var base64Encoded: String {
        let data = encodeBinary()
        return data.base64EncodedString()
    }

    /// Decode vector from base64 string
    ///
    /// - Parameter base64String: Base64-encoded vector data
    /// - Returns: Decoded vector
    /// - Throws: VectorError if decoding fails or dimension mismatch
    public static func base64Decoded(from base64String: String) throws -> Vector<D> {
        guard let data = Data(base64Encoded: base64String) else {
            throw VectorError.invalidDataFormat(
                expected: "base64",
                actual: "invalid base64 string"
            )
        }
        return try decodeBinary(from: data)
    }
}
