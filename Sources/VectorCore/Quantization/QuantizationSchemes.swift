//
//  QuantizationSchemes.swift
//  VectorCore
//
//  Structure-aware quantization for high-performance vector compression
//  Implements asymmetric INT8 quantization with error analysis
//

import Foundation
import simd

// MARK: - Quantization Parameters

/// Defines the parameters and strategy used for asymmetric INT8 quantization.
public struct QuantizationParams: Sendable, Equatable {
    public enum Strategy: Sendable {
        /// A single scale and offset pair is used for the entire vector.
        case perVector
        /// A distinct scale and offset pair is used for each dimension. (Calibration requires dataset analysis).
        case perDimension
    }

    public let strategy: Strategy
    /// Quantization scales. Length 1 (per-vector) or dim (per-dimension).
    // Using ContiguousArray for performance and safe pointer access.
    public let scales: ContiguousArray<Float>
    /// Quantization offsets (zero points).
    public let offsets: ContiguousArray<Float>

    public init(strategy: Strategy, scales: ContiguousArray<Float>, offsets: ContiguousArray<Float>) {
        self.strategy = strategy
        self.scales = scales
        self.offsets = offsets
    }

    /// Helper to get the scale for a specific dimension (abstracts the strategy).
    @inlinable
    public func scale(at index: Int) -> Float {
        return strategy == .perVector ? scales[0] : scales[index]
    }

    /// Helper to get the offset for a specific dimension.
    @inlinable
    public func offset(at index: Int) -> Float {
        return strategy == .perVector ? offsets[0] : offsets[index]
    }
}

// MARK: - Quantization Schemes

/// Implementation of quantization algorithms and analysis tools.
public enum QuantizationSchemes {

    // Constants for INT8 range
    private static let Q_MIN: Float = -128.0
    private static let Q_MAX: Float = 127.0
    private static let Q_RANGE: Float = 255.0 // Q_MAX - Q_MIN

    /// Computes optimal scale and offset for asymmetric INT8 quantization for a single vector.
    public static func computeQuantizationParams(
        values: [Float],
        strategy: QuantizationParams.Strategy
    ) -> QuantizationParams {

        guard !values.isEmpty else {
            // Handle empty vector case
            return QuantizationParams(strategy: strategy, scales: [1.0], offsets: [0.0])
        }

        switch strategy {
        case .perVector:
            return computePerVectorParams(values: values)
        case .perDimension:
            // Per-dimension calibration requires analysis across a dataset (multiple vectors).
            fatalError(".perDimension calibration requires a dataset-level API and cannot be computed from a single vector input.")
        }
    }

    /// Calculates parameters for the .perVector strategy.
    /// Scale = (max_val - min_val) / 255
    /// Offset = -128 - min_val / scale
    private static func computePerVectorParams(values: [Float]) -> QuantizationParams {
        // 1. Find Min/Max.
        var minVal = values[0]
        var maxVal = values[0]

        for value in values {
            minVal = min(minVal, value)
            maxVal = max(maxVal, value)
        }

        // 2. Calculate Scale
        let scale = (maxVal - minVal) / Q_RANGE

        // Handle the edge case where all values are the same (scale would be near 0).
        let effectiveScale = max(scale, Float.leastNonzeroMagnitude)

        // 3. Calculate Offset (Zero Point)
        let offset = Q_MIN - minVal / effectiveScale

        // Nudging: Round the offset and ensure it is within the representable range.
        let nudgedOffset = max(Q_MIN, min(Q_MAX, round(offset)))

        return QuantizationParams(strategy: .perVector, scales: [effectiveScale], offsets: [nudgedOffset])
    }

    // MARK: - Accuracy Analysis

    public struct QuantizationErrorStats {
        public let maxAbsoluteError: Float
        public let meanSquaredError: Float
        /// Signal-to-Noise Ratio (SNR) in dB.
        public let signalToNoiseRatio: Float
    }

    /// Analyzes the quantization error by performing a round-trip conversion and comparing results.
    public static func analyzeQuantizationError(
        original: [Float],
        quantized: [Int8],
        params: QuantizationParams
    ) -> QuantizationErrorStats {

        precondition(original.count == quantized.count, "Arrays must have the same length.")

        var maxAbsErr: Float = 0.0
        var sumSqErr: Float = 0.0
        var sumSqSignal: Float = 0.0

        for i in 0..<original.count {
            let x = original[i]
            let q = Float(quantized[i])

            // Dequantize: x' = (q - offset) * scale
            let scale = params.scale(at: i)
            let offset = params.offset(at: i)
            let x_prime = (q - offset) * scale

            // Calculate error
            let error = x - x_prime

            maxAbsErr = max(maxAbsErr, abs(error))
            sumSqErr += error * error
            sumSqSignal += x * x
        }

        let count = Float(original.count)
        let mse = original.count != 0 ? sumSqErr / count : 0.0

        // Calculate SNR = 10 * log10( P_signal / P_noise )
        let snr: Float
        if mse <= Float.leastNonzeroMagnitude {
            snr = Float.infinity
        } else if sumSqSignal <= Float.leastNonzeroMagnitude {
            snr = -Float.infinity
        } else {
            snr = 10.0 * log10(sumSqSignal / sumSqErr)
        }

        return QuantizationErrorStats(
            maxAbsoluteError: maxAbsErr,
            meanSquaredError: mse,
            signalToNoiseRatio: snr
        )
    }
}
