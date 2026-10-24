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
public struct MixedPrecisionMetrics: Sendable {
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

    /// Estimated speedup compared to an FP32 baseline that is assumed, not measured, to take
    /// twice `computationTime`
    public var totalSpeedup: Double {
        let totalTime = fp16ConversionTime + computationTime
        guard totalTime > 0 else { return 0.0 }

        // Assumption (not measured): the FP32 baseline takes 2× `computationTime`
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

/// Performance profiler for mixed precision operations. It is an `actor`, so its metrics are
/// actor-isolated state.
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
    /// Tracks the maximum observed loss. A NaN loss makes the tracked maximum NaN until `reset()` (T16 reduction NaN
    /// contract; `max` alone ignored a NaN). Pinned by T16ReductionProfilerTests.
    ///
    /// - Parameter loss: Relative error vs FP32 baseline
    public func recordAccuracyLoss(_ loss: Float) {
        // Track maximum observed loss
        metrics.accuracyLoss = ReductionNaN.propagatingMax(metrics.accuracyLoss, loss)
    }

    /// Record memory bandwidth utilization
    ///
    /// Tracks the maximum observed utilization. A NaN utilization makes the tracked maximum NaN until `reset()`
    /// (T16 reduction NaN contract). Pinned by T16ReductionProfilerTests.
    ///
    /// - Parameter utilization: Bandwidth utilization (0.0 to 1.0)
    public func recordMemoryBandwidthUtilization(_ utilization: Double) {
        metrics.memoryBandwidthUtilization = ReductionNaN.propagatingMax(metrics.memoryBandwidthUtilization,
                                                                         utilization)
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
    /// Nonfinite coordinates or an invalid tolerance return false.
    /// - Returns: true if every absolute scalar conversion error is within tolerance
    public static func validateConversion512(
        original: Vector512Optimized,
        converted: MixedPrecisionKernels.Vector512FP16,
        tolerance: Float = 0.001  // Absolute scalar error, not a percentage
    ) -> Bool {
        guard tolerance.isFinite, tolerance >= 0 else { return false }
        let restored = converted.toFP32()
        return validateVectorEquality512(original, restored, tolerance: tolerance)
    }

    /// Finite conversion validation with absolute scalar tolerance (default 0.001).
    public static func validateConversion768(
        original: Vector768Optimized,
        converted: MixedPrecisionKernels.Vector768FP16,
        tolerance: Float = 0.001
    ) -> Bool {
        guard tolerance.isFinite, tolerance >= 0 else { return false }
        let restored = converted.toFP32()
        return validateVectorEquality768(original, restored, tolerance: tolerance)
    }

    /// Finite conversion validation with absolute scalar tolerance (default 0.001).
    public static func validateConversion1536(
        original: Vector1536Optimized,
        converted: MixedPrecisionKernels.Vector1536FP16,
        tolerance: Float = 0.001
    ) -> Bool {
        guard tolerance.isFinite, tolerance >= 0 else { return false }
        let restored = converted.toFP32()
        return validateVectorEquality1536(original, restored, tolerance: tolerance)
    }

    // MARK: - Distance Accuracy Validation

    /// Validate distance computation accuracy
    ///
    /// Compares a mixed precision distance result against FP32 reference
    /// using abs(Double(reference) - Double(computed)) divided by the larger of
    /// abs(Double(reference)) and Double(Float(1e-8)). Inputs and tolerance must
    /// be finite, and tolerance must be nonnegative. Widening avoids Float
    /// subtraction overflow for opposite finite extrema.
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
        guard tolerance.isFinite, tolerance >= 0, referenceFP32.isFinite, computedFP16.isFinite else {
            return false
        }

        let relativeError = distanceError(referenceFP32, computedFP16)
        return relativeError <= Double(tolerance)
    }

    /// Validate batch distance accuracy
    ///
    /// - Parameters:
    ///   - referenceFP32: Reference FP32 distances
    ///   - computedFP16: Mixed precision distances
    ///   - tolerance: Maximum acceptable relative error
    /// Uses the scalar distance formula and Double summary accumulation.
    /// Invalid tolerance, mismatched lengths or any nonfinite pair returns
    /// `(false, infinity, infinity)`. Valid empty batches return `(true, 0, 0)`.
    /// - Returns: (allValid, maxError, meanError), with summaries converted to Float once.
    public static func validateBatchDistanceAccuracy(
        referenceFP32: [Float],
        computedFP16: [Float],
        tolerance: Float = 0.01
    ) -> (allValid: Bool, maxError: Float, meanError: Float) {
        guard tolerance.isFinite, tolerance >= 0,
              referenceFP32.count == computedFP16.count else {
            return (false, .infinity, .infinity)
        }

        var maxError: Double = 0
        var sumError: Double = 0
        var allValid = true
        for (reference, computed) in zip(referenceFP32, computedFP16) {
            guard reference.isFinite, computed.isFinite else {
                return (false, .infinity, .infinity)
            }
            let error = distanceError(reference, computed)
            maxError = max(maxError, error)
            sumError += error
            allValid = allValid && error <= Double(tolerance)
        }
        let meanError = referenceFP32.isEmpty ? 0 : sumError / Double(referenceFP32.count)
        return (allValid, Float(maxError), Float(meanError))
    }

    private static func distanceError(_ reference: Float, _ computed: Float) -> Double {
        abs(Double(reference) - Double(computed)) / max(abs(Double(reference)), Double(Float(1e-8)))
    }

    // MARK: - Vector Equality Validation

    /// Validate finite coordinates with finite, nonnegative absolute tolerance.
    /// All three width-specific equality validators use this policy; nonfinite
    /// coordinates are unequal even when both operands have identical bits.
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
        guard tolerance.isFinite, tolerance >= 0 else { return false }
        for i in 0..<128 {
            for lane in 0..<4 {
                let lhs = v1.storage[i][lane], rhs = v2.storage[i][lane]
                guard lhs.isFinite, rhs.isFinite, abs(lhs - rhs) <= tolerance else { return false }
            }
        }
        return true
    }

    public static func validateVectorEquality768(
        _ v1: Vector768Optimized,
        _ v2: Vector768Optimized,
        tolerance: Float
    ) -> Bool {
        guard tolerance.isFinite, tolerance >= 0 else { return false }
        for i in 0..<192 {
            for lane in 0..<4 {
                let lhs = v1.storage[i][lane], rhs = v2.storage[i][lane]
                guard lhs.isFinite, rhs.isFinite, abs(lhs - rhs) <= tolerance else { return false }
            }
        }
        return true
    }

    public static func validateVectorEquality1536(
        _ v1: Vector1536Optimized,
        _ v2: Vector1536Optimized,
        tolerance: Float
    ) -> Bool {
        guard tolerance.isFinite, tolerance >= 0 else { return false }
        for i in 0..<384 {
            for lane in 0..<4 {
                let lhs = v1.storage[i][lane], rhs = v2.storage[i][lane]
                guard lhs.isFinite, rhs.isFinite, abs(lhs - rhs) <= tolerance else { return false }
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

        /// NaN when any error is NaN (T16; `Array.max()` keeps a NaN only at index 0); 0 when empty.
        /// Pinned by T16ReductionExceptionalTableTests.
        public var maxAbsoluteError: Float {
            absoluteErrors.contains { $0.isNaN } ? .nan : (absoluteErrors.max() ?? 0)
        }

        public var meanAbsoluteError: Float {
            guard !absoluteErrors.isEmpty else { return 0 }
            return absoluteErrors.reduce(0, +) / Float(absoluteErrors.count)
        }

        /// NaN when any error is NaN (T16; `Array.max()` keeps a NaN only at index 0); 0 when empty.
        /// Pinned by T16ReductionExceptionalTableTests.
        public var maxRelativeError: Float {
            relativeErrors.contains { $0.isNaN } ? .nan : (relativeErrors.max() ?? 0)
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
