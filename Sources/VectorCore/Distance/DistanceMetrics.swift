// VectorCore: Distance Metrics
//
// High-performance distance metric implementations
//

import Foundation

// MARK: - Euclidean Distance

/// Optimized Euclidean distance metric (L2 norm)
///
/// Computes: √(Σ(aᵢ - bᵢ)²)
///
/// Performance characteristics:
/// - Small vectors (≤8 dims): Direct computation, O(n)
/// - Large vectors: Loop unrolling with 4 accumulators, ~4x throughput
/// - Memory access pattern: Sequential, cache-friendly
/// - Instruction pipelining: Multiple accumulators hide FP latency
public struct EuclideanDistance: DistanceMetric {
    public let identifier = "euclidean"
    
    public init() {}
    
    @inlinable
    @inline(__always)
    public func distance<Vector: ExtendedVectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore {
        // For small vectors, use direct computation to avoid loop overhead
        if a.scalarCount <= 8 {
            var sum: Float = 0
            for i in 0..<a.scalarCount {
                let diff = Float(a[i]) - Float(b[i])
                sum += diff * diff
            }
            return sqrt(sum)
        }
        
        // For larger vectors, use multiple accumulators to hide FP latency
        // Modern CPUs can execute multiple FP operations in parallel
        var sum0: Float = 0
        var sum1: Float = 0
        var sum2: Float = 0
        var sum3: Float = 0
        
        let count = a.scalarCount
        let mainLoop = count & ~3  // Round down to multiple of 4 using bit mask
        
        // Process 4 elements at a time
        for i in stride(from: 0, to: mainLoop, by: 4) {
            let diff0 = Float(a[i]) - Float(b[i])
            let diff1 = Float(a[i+1]) - Float(b[i+1])
            let diff2 = Float(a[i+2]) - Float(b[i+2])
            let diff3 = Float(a[i+3]) - Float(b[i+3])
            
            sum0 += diff0 * diff0
            sum1 += diff1 * diff1
            sum2 += diff2 * diff2
            sum3 += diff3 * diff3
        }
        
        // Process remaining elements
        for i in mainLoop..<count {
            let diff = Float(a[i]) - Float(b[i])
            sum0 += diff * diff
        }
        
        return sqrt(sum0 + sum1 + sum2 + sum3)
    }
    
    @inlinable
    public func batchDistance<Vector: ExtendedVectorProtocol>(
        query: Vector, 
        candidates: [Vector]
    ) -> [DistanceScore] {
        // Pre-allocate result array
        var results = [DistanceScore](repeating: 0, count: candidates.count)
        
        // Process in parallel-friendly chunks
        for (index, candidate) in candidates.enumerated() {
            results[index] = distance(query, candidate)
        }
        
        return results
    }
}

// MARK: - Cosine Distance

/// Optimized Cosine distance metric
///
/// Computes: 1 - (a·b)/(||a||₂||b||₂)
///
/// Performance characteristics:
/// - Fused computation: Dot product and norms in single pass
/// - Dual accumulators per metric: Improved instruction-level parallelism
/// - Numerical stability: Handles zero vectors gracefully
/// - Returns: [0, 2] where 0 = identical direction, 1 = orthogonal, 2 = opposite
public struct CosineDistance: DistanceMetric {
    public let identifier = "cosine"
    
    public init() {}
    
    @inlinable
    @inline(__always)
    public func distance<Vector: ExtendedVectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore {
        // Use multiple accumulators for better pipelining
        // CPU can compute dot product and norms simultaneously
        var dot0: Float = 0, dot1: Float = 0
        var normA0: Float = 0, normA1: Float = 0
        var normB0: Float = 0, normB1: Float = 0
        
        let count = a.scalarCount
        let mainLoop = count & ~1  // Round down to multiple of 2
        
        // Process 2 elements at a time
        for i in stride(from: 0, to: mainLoop, by: 2) {
            let a0 = Float(a[i])
            let a1 = Float(a[i+1])
            let b0 = Float(b[i])
            let b1 = Float(b[i+1])
            
            dot0 += a0 * b0
            dot1 += a1 * b1
            normA0 += a0 * a0
            normA1 += a1 * a1
            normB0 += b0 * b0
            normB1 += b1 * b1
        }
        
        // Process remaining element if any
        if mainLoop < count {
            let aVal = Float(a[mainLoop])
            let bVal = Float(b[mainLoop])
            dot0 += aVal * bVal
            normA0 += aVal * aVal
            normB0 += bVal * bVal
        }
        
        let dotProduct = dot0 + dot1
        let normA = sqrt(normA0 + normA1)
        let normB = sqrt(normB0 + normB1)
        
        // Handle zero vectors
        guard normA > Float.ulpOfOne && normB > Float.ulpOfOne else { return 1.0 }
        
        let cosine = dotProduct / (normA * normB)
        // Clamp to [-1, 1] to handle floating point errors
        let clampedCosine = max(-1.0, min(1.0, cosine))
        return 1.0 - clampedCosine
    }
}

// MARK: - Dot Product Distance

/// Optimized Dot Product distance metric
///
/// Computes: -(a·b) (negative for similarity-to-distance conversion)
///
/// Performance characteristics:
/// - 4-way unrolled loop: ~4x throughput vs naive implementation
/// - Multiple accumulators: Hide FP multiply-add latency
/// - Sequential memory access: Optimal cache utilization
/// - Note: Returns negative values; larger magnitude = more similar
///
/// Use cases:
/// - Pre-normalized vectors where angle is the only concern
/// - Neural network embeddings with unit norm
/// - When combined with vector magnitudes for cosine similarity
public struct DotProductDistance: DistanceMetric {
    public let identifier = "dotProduct"
    
    public init() {}
    
    @inlinable
    @inline(__always)
    public func distance<Vector: ExtendedVectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore {
        // Use multiple accumulators for better performance
        var sum0: Float = 0
        var sum1: Float = 0
        var sum2: Float = 0
        var sum3: Float = 0
        
        let count = a.scalarCount
        let mainLoop = count & ~3  // Round down to multiple of 4
        
        // Process 4 elements at a time
        for i in stride(from: 0, to: mainLoop, by: 4) {
            sum0 += Float(a[i]) * Float(b[i])
            sum1 += Float(a[i+1]) * Float(b[i+1])
            sum2 += Float(a[i+2]) * Float(b[i+2])
            sum3 += Float(a[i+3]) * Float(b[i+3])
        }
        
        // Process remaining elements
        for i in mainLoop..<count {
            sum0 += Float(a[i]) * Float(b[i])
        }
        
        return -(sum0 + sum1 + sum2 + sum3)  // Negative for similarity to distance
    }
}

// MARK: - Manhattan Distance

/// Optimized Manhattan distance metric (L1 norm)
///
/// Computes: Σ|aᵢ - bᵢ|
///
/// Performance characteristics:
/// - 2-way unrolled loop: Balances code size and performance
/// - Dual accumulators: Better instruction scheduling
/// - Branch-free abs(): Modern CPUs handle abs efficiently
/// - Memory bandwidth limited for large vectors
///
/// Use cases:
/// - Grid-based pathfinding and spatial algorithms
/// - Robust to outliers compared to Euclidean distance
/// - Feature selection and sparse data analysis
/// - City-block distance in urban planning applications
public struct ManhattanDistance: DistanceMetric {
    public let identifier = "manhattan"
    
    public init() {}
    
    @inlinable
    @inline(__always)
    public func distance<Vector: ExtendedVectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore {
        var sum0: Float = 0
        var sum1: Float = 0
        
        let count = a.scalarCount
        let mainLoop = count & ~1  // Round down to multiple of 2
        
        // Process 2 elements at a time
        for i in stride(from: 0, to: mainLoop, by: 2) {
            sum0 += abs(Float(a[i]) - Float(b[i]))
            sum1 += abs(Float(a[i+1]) - Float(b[i+1]))
        }
        
        // Process remaining element if any
        if mainLoop < count {
            sum0 += abs(Float(a[mainLoop]) - Float(b[mainLoop]))
        }
        
        return sum0 + sum1
    }
}

// MARK: - Specialized Distance Functions

/// Static utility functions for distance calculation
public struct DistanceCalculator {
    
    /// Compute distance between two vectors
    @inlinable
    public static func compute<V: ExtendedVectorProtocol>(_ a: V, _ b: V, metric: any DistanceMetric) -> DistanceScore {
        return metric.distance(a, b)
    }
    
    /// Compute squared Euclidean distance (avoids sqrt)
    ///
    /// Performance optimization for:
    /// - k-NN searches (comparison only)
    /// - Clustering algorithms (relative distances)
    /// - Gradient computations (derivatives)
    ///
    /// ~2x faster than full Euclidean distance
    @inlinable
    @inline(__always)
    public static func euclideanSquared<V: ExtendedVectorProtocol>(_ a: V, _ b: V) -> DistanceScore {
        var sum: Float = 0
        for i in 0..<a.scalarCount {
            let diff = Float(a[i]) - Float(b[i])
            sum += diff * diff
        }
        return sum
    }
    
    /// Compute normalized vectors then cosine similarity
    ///
    /// Optimization for pre-normalized vectors:
    /// - Skips magnitude computation (~30% faster)
    /// - Common in embedding models
    /// - Reduces to simple dot product
    ///
    /// IMPORTANT: Assumes ||a||₂ = ||b||₂ = 1
    @inlinable
    public static func normalizedCosine<V: ExtendedVectorProtocol>(_ a: V, _ b: V) -> DistanceScore {
        // Assumes vectors are already normalized
        var dotProduct: Float = 0
        for i in 0..<a.scalarCount {
            dotProduct += Float(a[i]) * Float(b[i])
        }
        return 1.0 - dotProduct
    }
}

// MARK: - Hamming Distance

/// Hamming distance metric (number of differing elements)
///
/// Computes: Σ(aᵢ ≠ bᵢ ? 1 : 0)
///
/// Performance characteristics:
/// - Simple comparison loop: Limited by branch prediction
/// - Consider bit manipulation for binary vectors
/// - O(n) time complexity, no floating-point operations
/// - Cache-friendly sequential access
///
/// Use cases:
/// - Binary vector comparison (error correction codes)
/// - Categorical data distance measurement
/// - DNA sequence comparison in bioinformatics
/// - Hash collision detection
///
/// Note: For binary data, consider specialized bit manipulation versions
public struct HammingDistance: DistanceMetric {
    public let identifier = "hamming"
    
    public init() {}
    
    @inlinable
    public func distance<Vector: ExtendedVectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore {
        var count: Int = 0
        for i in 0..<a.scalarCount {
            if a[i] != b[i] {
                count += 1
            }
        }
        return Float(count)
    }
}

// MARK: - Chebyshev Distance

/// Chebyshev distance metric (L∞ norm)
///
/// Computes: max(|aᵢ - bᵢ|)
///
/// Performance characteristics:
/// - Single pass with running maximum
/// - Early termination possible if threshold known
/// - Minimal memory footprint (single accumulator)
/// - Branch prediction friendly with max operation
///
/// Use cases:
/// - Chess king movement distance
/// - Warehouse/logistics planning (max coordinate difference)
/// - Image processing (maximum pixel difference)
/// - Uniform norm in optimization problems
///
/// Mathematical properties:
/// - Always ≤ Euclidean distance
/// - Defines rectangular neighborhoods
public struct ChebyshevDistance: DistanceMetric {
    public let identifier = "chebyshev"
    
    public init() {}
    
    @inlinable
    public func distance<Vector: ExtendedVectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore {
        var maxDiff: Float = 0
        for i in 0..<a.scalarCount {
            let diff = abs(Float(a[i]) - Float(b[i]))
            maxDiff = max(maxDiff, diff)
        }
        return maxDiff
    }
}

// MARK: - Minkowski Distance

/// Minkowski distance metric (Lp norm)
///
/// Computes: (Σ|aᵢ - bᵢ|^p)^(1/p)
///
/// Performance characteristics:
/// - Special-cased for p=1 (Manhattan) and p=2 (Euclidean)
/// - General case uses pow() - significantly slower
/// - Consider pre-computing 1/p for repeated use
/// - Cache pow() results for common p values
///
/// Use cases:
/// - Generalization of Manhattan (p=1) and Euclidean (p=2)
/// - p→∞ converges to Chebyshev distance
/// - Fractional p for specialized similarity measures
/// - Parameter tuning in machine learning
///
/// Mathematical properties:
/// - Triangle inequality holds for p ≥ 1
/// - Convex unit balls for p ≥ 1
/// - Non-convex for 0 < p < 1
public struct MinkowskiDistance: DistanceMetric {
    public let identifier: String
    public let p: Float
    
    public init(p: Float = 3.0) {
        self.p = p
        self.identifier = "minkowski_p\(p)"
    }
    
    @inlinable
    public func distance<Vector: ExtendedVectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore {
        guard p > 0 else { return 0 }
        
        // Special cases for common values
        if p == 1 {
            return ManhattanDistance().distance(a, b)
        }
        if p == 2 {
            return EuclideanDistance().distance(a, b)
        }
        if p.isInfinite {
            return ChebyshevDistance().distance(a, b)
        }
        
        var sum: Float = 0
        for i in 0..<a.scalarCount {
            let diff = abs(Float(a[i]) - Float(b[i]))
            sum += pow(diff, p)
        }
        return pow(sum, 1.0 / p)
    }
}

// MARK: - Jaccard Distance

/// Jaccard distance metric (for binary/sparse vectors)
///
/// Computes: 1 - |A∩B|/|A∪B| where elements are treated as binary
///
/// Performance characteristics:
/// - Single pass with intersection/union counting
/// - Threshold comparison for binary conversion
/// - Consider specialized sparse vector version
/// - Memory bandwidth limited for dense vectors
///
/// Use cases:
/// - Set similarity measurement
/// - Document similarity (bag-of-words)
/// - Recommendation systems (user preferences)
/// - Ecological diversity indices
/// - Binary feature comparison
///
/// Implementation notes:
/// - Values above threshold are treated as "present" (1)
/// - Zero vectors return distance 0 (identical empty sets)
/// - Consider sparse representations for efficiency
public struct JaccardDistance: DistanceMetric {
    public let identifier = "jaccard"
    public let threshold: Float
    
    public init(threshold: Float = Float.ulpOfOne) {
        self.threshold = threshold
    }
    
    @inlinable
    public func distance<Vector: ExtendedVectorProtocol>(_ a: Vector, _ b: Vector) -> DistanceScore {
        var intersection: Int = 0
        var union: Int = 0
        
        for i in 0..<a.scalarCount {
            let aVal = Float(a[i])
            let bVal = Float(b[i])
            
            // Treat as binary (non-zero is 1)
            let aBinary = abs(aVal) > threshold
            let bBinary = abs(bVal) > threshold
            
            if aBinary && bBinary {
                intersection += 1
            }
            if aBinary || bBinary {
                union += 1
            }
        }
        
        guard union > 0 else { return 0 }
        return 1.0 - Float(intersection) / Float(union)
    }
}