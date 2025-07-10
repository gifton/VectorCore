// VectorCore: Vector Math Operations
//
// Additional mathematical operations for vectors
//

import Foundation
import Accelerate
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
        var result = lhs
        result.storage.withUnsafeMutableBufferPointer { dest in
            rhs.storage.withUnsafeBufferPointer { src in
                vDSP_vmul(dest.baseAddress!, 1, src.baseAddress!, 1,
                         dest.baseAddress!, 1, vDSP_Length(D.value))
            }
        }
        return result
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
        var result = lhs
        result.storage.withUnsafeMutableBufferPointer { dest in
            rhs.storage.withUnsafeBufferPointer { src in
                vDSP_vdiv(src.baseAddress!, 1, dest.baseAddress!, 1,
                         dest.baseAddress!, 1, vDSP_Length(D.value))
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
    /// - vDSP_vabs + vDSP_sve: Two-pass SIMD implementation
    /// - Temporary buffer allocation for absolute values
    /// - Consider Manhattan distance metric for direct computation
    ///
    /// Mathematical properties:
    /// - Always non-negative
    /// - Triangle inequality: ||x+y||₁ ≤ ||x||₁ + ||y||₁
    /// - Robust to outliers (linear growth)
    @inlinable
    public var l1Norm: Float {
        var result: Float = 0
        var temp = [Float](repeating: 0, count: D.value)
        storage.withUnsafeBufferPointer { buffer in
            temp.withUnsafeMutableBufferPointer { tempBuffer in
                vDSP_vabs(buffer.baseAddress!, 1,
                         tempBuffer.baseAddress!, 1, vDSP_Length(D.value))
                vDSP_sve(tempBuffer.baseAddress!, 1, &result, vDSP_Length(D.value))
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
        var result: Float = 0
        storage.withUnsafeBufferPointer { buffer in
            vDSP_maxmgv(buffer.baseAddress!, 1, &result, vDSP_Length(D.value))
        }
        return result
    }
}

// MARK: - Statistical Operations

extension Vector {
    /// Mean value of all elements
    @inlinable
    public var mean: Float {
        var result: Float = 0
        storage.withUnsafeBufferPointer { buffer in
            vDSP_meanv(buffer.baseAddress!, 1, &result, vDSP_Length(D.value))
        }
        return result
    }
    
    /// Sum of all elements
    @inlinable
    public var sum: Float {
        var result: Float = 0
        storage.withUnsafeBufferPointer { buffer in
            vDSP_sve(buffer.baseAddress!, 1, &result, vDSP_Length(D.value))
        }
        return result
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
        
        storage.withUnsafeBufferPointer { buffer in
            // Compute sum of squared differences from mean
            var temp = [Float](repeating: 0, count: D.value)
            temp.withUnsafeMutableBufferPointer { tempBuffer in
                // Subtract mean
                var negMean = -m
                vDSP_vsadd(buffer.baseAddress!, 1, &negMean,
                          tempBuffer.baseAddress!, 1, vDSP_Length(D.value))
                
                // Square the differences
                vDSP_vsq(tempBuffer.baseAddress!, 1,
                        tempBuffer.baseAddress!, 1, vDSP_Length(D.value))
                
                // Sum
                vDSP_sve(tempBuffer.baseAddress!, 1, &result, vDSP_Length(D.value))
            }
        }
        
        return result / Float(D.value)
    }
    
    /// Standard deviation
    @inlinable
    public var standardDeviation: Float {
        sqrt(variance)
    }
}

// MARK: - Distance Metrics

extension Vector where D.Storage: VectorStorageOperations {
    /// Manhattan distance to another vector
    @inlinable
    public func manhattanDistance(to other: Vector<D>) -> Float {
        (self - other).l1Norm
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
        var maxVal: Float = 0
        storage.withUnsafeBufferPointer { buffer in
            vDSP_maxv(buffer.baseAddress!, 1, &maxVal, vDSP_Length(D.value))
        }
        
        result.storage.withUnsafeMutableBufferPointer { buffer in
            // Subtract max and exp
            var negMax = -maxVal
            vDSP_vsadd(buffer.baseAddress!, 1, &negMax,
                      buffer.baseAddress!, 1, vDSP_Length(D.value))
            
            // Apply exp
            var count = Int32(D.value)
            vvexpf(buffer.baseAddress!, buffer.baseAddress!, &count)
            
            // Sum for normalization
            var sum: Float = 0
            vDSP_sve(buffer.baseAddress!, 1, &sum, vDSP_Length(D.value))
            
            // Normalize
            vDSP_vsdiv(buffer.baseAddress!, 1, &sum,
                      buffer.baseAddress!, 1, vDSP_Length(D.value))
        }
        
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
        
        result.storage.withUnsafeMutableBufferPointer { buffer in
            var lower = range.lowerBound
            var upper = range.upperBound
            vDSP_vclip(buffer.baseAddress!, 1, &lower, &upper,
                      buffer.baseAddress!, 1, vDSP_Length(D.value))
        }
        
        return result
    }
}

// MARK: - Batch Operations

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
    /// Create a vector with random values in range
    public static func random(in range: ClosedRange<Float>) -> Vector<D> {
        let values = (0..<D.value).map { _ in
            Float.random(in: range)
        }
        return Vector(values)
    }
    
    /// Create a vector filled with zeros
    public static var zero: Vector<D> {
        Vector()
    }
    
    /// Create a vector filled with ones
    public static var one: Vector<D> {
        Vector(repeating: 1)
    }
}