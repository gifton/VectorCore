// VectorCore: Vector Quality Metrics
//
// Composite quality assessment for vectors
//

import Foundation

/// Quality metrics for vector assessment
///
/// Provides a comprehensive view of vector characteristics including:
/// - **Magnitude**: L2 norm indicating vector strength
/// - **Variance**: Measure of element spread
/// - **Sparsity**: Proportion of near-zero elements
/// - **Entropy**: Information content/randomness measure
///
/// Use this for vector quality assessment, compression decisions, or
/// identifying vectors that may need special handling.
public struct VectorQuality: Sendable, Codable, Equatable {
    /// L2 norm (Euclidean magnitude) of the vector
    public let magnitude: Float
    
    /// Statistical variance of vector elements
    public let variance: Float
    
    /// Proportion of elements that are zero or near-zero (0.0 = dense, 1.0 = all zeros)
    public let sparsity: Float
    
    /// Shannon entropy of the normalized vector (0.0 = concentrated, higher = distributed)
    public let entropy: Float
    
    /// Initialize with individual metric values
    public init(magnitude: Float, variance: Float, sparsity: Float, entropy: Float) {
        self.magnitude = magnitude
        self.variance = variance
        self.sparsity = sparsity
        self.entropy = entropy
    }
    
    /// Computed properties for quality assessment
    
    /// Whether the vector is effectively zero (magnitude below threshold)
    public var isZero: Bool {
        magnitude < Float.ulpOfOne
    }
    
    /// Whether the vector is highly sparse (>50% zeros)
    public var isSparse: Bool {
        sparsity > 0.5
    }
    
    /// Whether the vector has low entropy (concentrated in few elements)
    public var isConcentrated: Bool {
        entropy < 1.0  // Threshold based on typical entropy ranges
    }
    
    /// Overall quality score (0-1, higher is better)
    /// Combines multiple factors:
    /// - Non-zero magnitude (good)
    /// - Moderate sparsity (not too sparse)
    /// - Reasonable entropy (not too concentrated)
    public var score: Float {
        var score: Float = 0.0
        
        // Magnitude contribution (normalized to 0-1 range using sigmoid-like function)
        let magnitudeScore = min(1.0, magnitude / 10.0)  // Assumes typical magnitudes < 10
        score += magnitudeScore * 0.4
        
        // Sparsity contribution (prefer moderate sparsity)
        let sparsityScore = 1.0 - abs(sparsity - 0.3) / 0.7  // Peak at 30% sparsity
        score += sparsityScore * 0.3
        
        // Entropy contribution (prefer moderate entropy)
        let entropyScore = min(1.0, entropy / 3.0)  // Normalize to typical range
        score += entropyScore * 0.3
        
        return score
    }
}

// MARK: - CustomStringConvertible

extension VectorQuality: CustomStringConvertible {
    public var description: String {
        """
        VectorQuality(
            magnitude: \(String(format: "%.3f", magnitude)),
            variance: \(String(format: "%.3f", variance)),
            sparsity: \(String(format: "%.1f%%", sparsity * 100)),
            entropy: \(String(format: "%.3f", entropy)),
            score: \(String(format: "%.2f", score))
        )
        """
    }
}

// MARK: - Comparable

extension VectorQuality: Comparable {
    /// Compare by overall quality score
    public static func < (lhs: VectorQuality, rhs: VectorQuality) -> Bool {
        lhs.score < rhs.score
    }
}