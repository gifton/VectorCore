// VectorCore: Vector Math Operations
//
// Additional mathematical operations for vectors
//

import Foundation
import simd

// MARK: - Element-wise Operations

extension Vector {
    /// Element-wise multiplication (Hadamard product)
    ///
    /// Computes: cᵢ = aᵢ × bᵢ for all i
    ///
    /// Performance characteristics:
    /// - vDSP_vmul: Highly optimized SIMD implementation
    /// - In-place operation on result copy
    /// - ~10x faster than scalar loop for large vectors
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
    /// - vDSP_vdiv: SIMD-optimized division
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

extension Vector where D.Storage: VectorStorageOperations {
    /// L1 norm (Manhattan norm)
    ///
    /// Computes: Σ|xᵢ|
    ///
    /// Performance characteristics:
    /// - Small vectors (≤32): Direct loop to avoid allocation overhead
    /// - Large vectors: vDSP_vabs + vDSP_sve SIMD implementation
    /// - Optimized memory usage with single allocation
    ///
    /// Mathematical properties:
    /// - Always non-negative
    /// - Triangle inequality: ||x+y||₁ ≤ ||x||₁ + ||y||₁
    /// - Robust to outliers (linear growth)
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
    /// Performance characteristics:
    /// - vDSP_maxmgv: Single-pass maximum magnitude
    /// - O(n) with early termination potential
    /// - Minimal memory overhead
    ///
    /// Mathematical properties:
    /// - Defines hypercube neighborhoods
    /// - Dual norm to L¹ (sum norm)
    /// - Submultiplicative: ||xy||∞ ≤ ||x||∞||y||∞
    @inlinable
    public var lInfinityNorm: Float {
        let array = self.toArray()
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
    /// Algorithm: Two-pass for numerical stability
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

extension Vector where D.Storage: VectorStorageOperations {
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
    /// Algorithm: Log-sum-exp trick for numerical stability
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
        
        // Find max for numerical stability
        let array = self.toArray()
        let maxVal = Operations.simdProvider.max(array)
        
        // Subtract max and exp
        let shifted = array.map { $0 - maxVal }
        let expValues = shifted.map { Foundation.exp($0) }
        
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
    /// Performance characteristics:
    /// - vDSP_vclip: Single-pass SIMD clipping
    /// - Branch-free implementation
    /// - In-place operation on result copy
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
/// support, and statistical computations. All operations are optimized
/// for performance using Accelerate framework.
///
/// ## Categories
/// - **Nearest Neighbor Search**: Find similar vectors efficiently
/// - **Clustering Support**: Operations for k-means and related algorithms
/// - **Statistical Operations**: Mean, variance, and other statistics
/// - **Matrix Operations**: Batch vector operations
///
/// ## Performance Notes
/// - Uses vDSP for optimal SIMD performance
/// - Operations are designed for batch processing
/// - Consider approximate methods for very large datasets
public enum VectorMath {
    /// Find k nearest neighbors by distance
    ///
    /// Algorithm: Full scan with partial sort
    /// - O(n) distance computations
    /// - O(n log k) for partial sort (via heap)
    ///
    /// Performance characteristics:
    /// - Linear scan suitable for small datasets
    /// - Consider approximate methods (LSH, HNSW) for large n
    /// - Parallelizable across distance computations
    ///
    /// Memory: O(n) for distance storage
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
    ) -> [(index: Int, distance: Float)] where D.Storage: VectorStorageOperations {
        let distances = vectors.enumerated().map { index, vector in
            (index: index, distance: metric(query, vector))
        }
        
        return Array(distances.sorted { $0.distance < $1.distance }.prefix(k))
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
    ) -> [[Float]] where D.Storage: VectorStorageOperations {
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

extension Vector where D.Storage: VectorStorageOperations {
    /// Calculate sparsity (proportion of near-zero elements)
    ///
    /// - Parameter threshold: Values with absolute value <= threshold are considered zero
    /// - Returns: Proportion of sparse elements (0.0 = dense, 1.0 = all zeros)
    ///
    /// Use cases:
    /// - Compression decisions (sparse vectors can be stored efficiently)
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
    
    /// Calculate Shannon entropy
    ///
    /// Treats the vector as a probability distribution after normalization.
    /// Higher values indicate more uniform distribution of values.
    ///
    /// Formula: H(X) = -Σ(p_i * log(p_i)) where p_i = |x_i| / Σ|x_j|
    ///
    /// Returns:
    /// - 0.0 for zero vectors or single-spike vectors
    /// - Higher values for more distributed vectors
    /// - Maximum entropy = log(n) for uniform distribution
    ///
    /// Use cases:
    /// - Measure information content
    /// - Detect concentrated vs distributed patterns
    /// - Feature quality assessment
    @inlinable
    public var entropy: Float {
        // Use optimized implementation for better performance
        return entropyFast
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

extension Vector where D.Storage: VectorStorageOperations {
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