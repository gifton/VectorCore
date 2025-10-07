//
//  MixedPrecisionValidation.swift
//  VectorCore
//
//  Precision validation and performance profiling utilities
//  Implements Phase 3 of kernel-specs/11-mixed-precision-kernels-part2.md
//

import Foundation
import simd

// MARK: - Performance Metrics

/// Performance metrics for mixed precision operations
internal struct MixedPrecisionMetrics: Sendable {
    /// Time spent on FP16 conversion (seconds)
    public var fp16ConversionTime: TimeInterval = 0

    /// Time spent on actual computation (seconds)
    public var computationTime: TimeInterval = 0

    /// Memory bandwidth utilization (0.0 to 1.0)
    public var memoryBandwidthUtilization: Double = 0

    /// Maximum observed accuracy loss vs FP32
    public var accuracyLoss: Float = 0

    /// Cache hit rate for SoA conversions (0.0 to 1.0)
    public var cacheHitRate: Double = 0

    /// Estimated speedup compared to FP32 baseline
    public var totalSpeedup: Double {
        let totalTime = fp16ConversionTime + computationTime
        guard totalTime > 0 else { return 0.0 }

        // Heuristic: FP16 typically provides ~2× improvement
        let baselineTime = computationTime * 2.0
        return baselineTime / totalTime
    }

    /// Human-readable summary
    public var summary: String {
        """
        Mixed Precision Metrics:
          Conversion Time:  \(String(format: "%.4f ms", fp16ConversionTime * 1000))
          Computation Time: \(String(format: "%.4f ms", computationTime * 1000))
          Total Speedup:    \(String(format: "%.2fx", totalSpeedup))
          Accuracy Loss:    \(String(format: "%.6f", accuracyLoss))
          Cache Hit Rate:   \(String(format: "%.1f%%", cacheHitRate * 100))
        """
    }
}

// MARK: - Performance Profiler

/// Thread-safe performance profiler for mixed precision operations
///
/// Tracks runtime metrics including conversion overhead, computation time,
/// and accuracy degradation.
///
/// ## Usage Example
/// ```swift
/// let profiler = PerformanceProfiler.shared
///
/// // Record conversion
/// let start = mach_absolute_time()
/// let fp16 = MixedPrecisionKernels.Vector512FP16(from: vector)
/// await profiler.recordConversionTime(timeElapsed(since: start))
///
/// // Record accuracy
/// await profiler.recordAccuracyLoss(relativeError)
///
/// // Get metrics
/// let metrics = await profiler.getMetrics()
/// print(metrics.summary)
/// ```
public actor PerformanceProfiler {

    // MARK: - Singleton

    public static let shared = PerformanceProfiler()

    // MARK: - State

    private var metrics = MixedPrecisionMetrics()

    // MARK: - Initialization

    private init() {}

    // MARK: - Recording

    /// Record FP16 conversion time
    ///
    /// - Parameter time: Time spent on conversion (seconds)
    public func recordConversionTime(_ time: TimeInterval) {
        metrics.fp16ConversionTime += time
    }

    /// Record computation time
    ///
    /// - Parameter time: Time spent on actual computation (seconds)
    public func recordComputationTime(_ time: TimeInterval) {
        metrics.computationTime += time
    }

    /// Record accuracy loss
    ///
    /// - Parameter loss: Relative error vs FP32 baseline
    public func recordAccuracyLoss(_ loss: Float) {
        // Track maximum observed loss
        metrics.accuracyLoss = max(metrics.accuracyLoss, loss)
    }

    /// Record memory bandwidth utilization
    ///
    /// - Parameter utilization: Bandwidth utilization (0.0 to 1.0)
    public func recordMemoryBandwidthUtilization(_ utilization: Double) {
        metrics.memoryBandwidthUtilization = max(metrics.memoryBandwidthUtilization, utilization)
    }

    /// Record cache hit rate
    ///
    /// - Parameter hitRate: Cache hit rate (0.0 to 1.0)
    public func recordCacheHitRate(_ hitRate: Double) {
        metrics.cacheHitRate = hitRate
    }

    // MARK: - Retrieval

    /// Get current metrics snapshot
    ///
    /// - Returns: Copy of current metrics
    public func getMetrics() -> MixedPrecisionMetrics {
        return metrics
    }

    /// Reset all metrics to zero
    public func reset() {
        metrics = MixedPrecisionMetrics()
    }
}

// MARK: - Precision Validator

/// Utilities for validating mixed precision accuracy
internal enum PrecisionValidator {

    // MARK: - Conversion Validation

    /// Validate FP32 → FP16 → FP32 round-trip conversion
    ///
    /// Ensures that converting a vector to FP16 and back to FP32 maintains
    /// acceptable precision within the given tolerance.
    ///
    /// - Parameters:
    ///   - original: Original FP32 vector
    ///   - converted: FP16 representation
    ///   - tolerance: Maximum acceptable absolute difference per element
    /// - Returns: true if conversion is within tolerance
    public static func validateConversion512(
        original: Vector512Optimized,
        converted: MixedPrecisionKernels.Vector512FP16,
        tolerance: Float = 0.001  // 0.1% default
    ) -> Bool {
        let restored = converted.toFP32()
        return validateVectorEquality512(original, restored, tolerance: tolerance)
    }

    public static func validateConversion768(
        original: Vector768Optimized,
        converted: MixedPrecisionKernels.Vector768FP16,
        tolerance: Float = 0.001
    ) -> Bool {
        let restored = converted.toFP32()
        return validateVectorEquality768(original, restored, tolerance: tolerance)
    }

    public static func validateConversion1536(
        original: Vector1536Optimized,
        converted: MixedPrecisionKernels.Vector1536FP16,
        tolerance: Float = 0.001
    ) -> Bool {
        let restored = converted.toFP32()
        return validateVectorEquality1536(original, restored, tolerance: tolerance)
    }

    // MARK: - Distance Accuracy Validation

    /// Validate distance computation accuracy
    ///
    /// Compares a mixed precision distance result against FP32 reference
    /// using relative error metric.
    ///
    /// - Parameters:
    ///   - referenceFP32: Reference FP32 distance
    ///   - computedFP16: Mixed precision computed distance
    ///   - tolerance: Maximum acceptable relative error (default 1%)
    /// - Returns: true if within tolerance
    public static func validateDistanceAccuracy(
        referenceFP32: Float,
        computedFP16: Float,
        tolerance: Float = 0.01  // 1% default
    ) -> Bool {
        guard referenceFP32.isFinite && computedFP16.isFinite else {
            return false
        }

        let relativeError = abs(referenceFP32 - computedFP16) / max(abs(referenceFP32), 1e-8)
        return relativeError <= tolerance
    }

    /// Validate batch distance accuracy
    ///
    /// - Parameters:
    ///   - referenceFP32: Reference FP32 distances
    ///   - computedFP16: Mixed precision distances
    ///   - tolerance: Maximum acceptable relative error
    /// - Returns: (allValid, maxError, meanError)
    public static func validateBatchDistanceAccuracy(
        referenceFP32: [Float],
        computedFP16: [Float],
        tolerance: Float = 0.01
    ) -> (allValid: Bool, maxError: Float, meanError: Float) {
        guard referenceFP32.count == computedFP16.count else {
            return (false, Float.infinity, Float.infinity)
        }

        var maxError: Float = 0
        var sumError: Float = 0
        var allValid = true

        for (ref, computed) in zip(referenceFP32, computedFP16) {
            guard ref.isFinite && computed.isFinite else {
                allValid = false
                maxError = Float.infinity
                continue
            }

            let relativeError = abs(ref - computed) / max(abs(ref), 1e-8)
            maxError = max(maxError, relativeError)
            sumError += relativeError

            if relativeError > tolerance {
                allValid = false
            }
        }

        let meanError = referenceFP32.isEmpty ? 0 : sumError / Float(referenceFP32.count)
        return (allValid, maxError, meanError)
    }

    // MARK: - Vector Equality Validation

    /// Validate two 512D vectors are equal within tolerance
    ///
    /// - Parameters:
    ///   - v1: First vector
    ///   - v2: Second vector
    ///   - tolerance: Maximum acceptable absolute difference per element
    /// - Returns: true if all elements are within tolerance
    public static func validateVectorEquality512(
        _ v1: Vector512Optimized,
        _ v2: Vector512Optimized,
        tolerance: Float
    ) -> Bool {
        for i in 0..<128 {  // 512 / 4 = 128 SIMD4 lanes
            let diff = abs(v1.storage[i] - v2.storage[i])

            // Check if any element exceeds tolerance
            if any(diff .> tolerance) {
                return false
            }
        }
        return true
    }

    public static func validateVectorEquality768(
        _ v1: Vector768Optimized,
        _ v2: Vector768Optimized,
        tolerance: Float
    ) -> Bool {
        for i in 0..<192 {  // 768 / 4 = 192 SIMD4 lanes
            let diff = abs(v1.storage[i] - v2.storage[i])
            if any(diff .> tolerance) {
                return false
            }
        }
        return true
    }

    public static func validateVectorEquality1536(
        _ v1: Vector1536Optimized,
        _ v2: Vector1536Optimized,
        tolerance: Float
    ) -> Bool {
        for i in 0..<384 {  // 1536 / 4 = 384 SIMD4 lanes
            let diff = abs(v1.storage[i] - v2.storage[i])
            if any(diff .> tolerance) {
                return false
            }
        }
        return true
    }

    // MARK: - Statistical Validation

    /// Compute detailed accuracy statistics for a vector comparison
    ///
    /// - Parameters:
    ///   - reference: Reference FP32 vector
    ///   - computed: Computed (possibly FP16) vector
    /// - Returns: Statistics including max/mean/median absolute and relative errors
    public static func computeAccuracyStatistics512(
        reference: Vector512Optimized,
        computed: Vector512Optimized
    ) -> AccuracyStatistics {
        var absoluteErrors: [Float] = []
        var relativeErrors: [Float] = []

        absoluteErrors.reserveCapacity(512)
        relativeErrors.reserveCapacity(512)

        for i in 0..<128 {
            for j in 0..<4 {
                let ref = reference.storage[i][j]
                let comp = computed.storage[i][j]

                let absError = abs(ref - comp)
                let relError = abs(ref - comp) / max(abs(ref), 1e-8)

                absoluteErrors.append(absError)
                relativeErrors.append(relError)
            }
        }

        return AccuracyStatistics(
            absoluteErrors: absoluteErrors,
            relativeErrors: relativeErrors
        )
    }

    /// Accuracy statistics container
    public struct AccuracyStatistics {
        public let absoluteErrors: [Float]
        public let relativeErrors: [Float]

        public var maxAbsoluteError: Float {
            absoluteErrors.max() ?? 0
        }

        public var meanAbsoluteError: Float {
            guard !absoluteErrors.isEmpty else { return 0 }
            return absoluteErrors.reduce(0, +) / Float(absoluteErrors.count)
        }

        public var maxRelativeError: Float {
            relativeErrors.max() ?? 0
        }

        public var meanRelativeError: Float {
            guard !relativeErrors.isEmpty else { return 0 }
            return relativeErrors.reduce(0, +) / Float(relativeErrors.count)
        }

        public var summary: String {
            """
            Accuracy Statistics:
              Max Absolute Error:  \(String(format: "%.6f", maxAbsoluteError))
              Mean Absolute Error: \(String(format: "%.6f", meanAbsoluteError))
              Max Relative Error:  \(String(format: "%.6f (%.4f%%)", maxRelativeError, maxRelativeError * 100))
              Mean Relative Error: \(String(format: "%.6f (%.4f%%)", meanRelativeError, meanRelativeError * 100))
            """
        }
    }
}
