//
//  MixedPrecisionKernels.swift
//  VectorCore
//
//  FP16 Mixed-Precision CPU Kernels with Apple Silicon NEON optimization
//  Provides 2× memory bandwidth improvement with maintained FP32 accuracy
//

import Foundation
import simd

// MARK: - FP16 Storage Types

/// FP16 storage for 512-dimensional optimized vectors
public struct Vector512FP16: Sendable {
    public let storage: ContiguousArray<SIMD4<Float16>>  // 128 lanes

    public init(from vector: Vector512Optimized) {
        // Efficient FP32→FP16 conversion using Swift's automatic NEON intrinsics
        let fp16Storage = vector.storage.map { SIMD4<Float16>($0) }
        self.storage = ContiguousArray(fp16Storage)
    }

    public init(storage: ContiguousArray<SIMD4<Float16>>) {
        guard storage.count == 128 else {
            fatalError("Invalid storage size for Vector512FP16: expected 128, got \(storage.count)")
        }
        self.storage = storage
    }

    public func toFP32() -> Vector512Optimized {
        // Efficient FP16→FP32 conversion using Swift's automatic NEON intrinsics
        let fp32Storage = self.storage.map { SIMD4<Float>($0) }
        var result = Vector512Optimized()
        result.storage = ContiguousArray(fp32Storage)
        return result
    }
}

/// FP16 storage for 768-dimensional optimized vectors
public struct Vector768FP16: Sendable {
    public let storage: ContiguousArray<SIMD4<Float16>>  // 192 lanes

    public init(from vector: Vector768Optimized) {
        let fp16Storage = vector.storage.map { SIMD4<Float16>($0) }
        self.storage = ContiguousArray(fp16Storage)
    }

    public init(storage: ContiguousArray<SIMD4<Float16>>) {
        guard storage.count == 192 else {
            fatalError("Invalid storage size for Vector768FP16: expected 192, got \(storage.count)")
        }
        self.storage = storage
    }

    public func toFP32() -> Vector768Optimized {
        let fp32Storage = self.storage.map { SIMD4<Float>($0) }
        var result = Vector768Optimized()
        result.storage = ContiguousArray(fp32Storage)
        return result
    }
}

/// FP16 storage for 1536-dimensional optimized vectors
public struct Vector1536FP16: Sendable {
    public let storage: ContiguousArray<SIMD4<Float16>>  // 384 lanes

    public init(from vector: Vector1536Optimized) {
        let fp16Storage = vector.storage.map { SIMD4<Float16>($0) }
        self.storage = ContiguousArray(fp16Storage)
    }

    public init(storage: ContiguousArray<SIMD4<Float16>>) {
        guard storage.count == 384 else {
            fatalError("Invalid storage size for Vector1536FP16: expected 384, got \(storage.count)")
        }
        self.storage = storage
    }

    public func toFP32() -> Vector1536Optimized {
        let fp32Storage = self.storage.map { SIMD4<Float>($0) }
        var result = Vector1536Optimized()
        result.storage = ContiguousArray(fp32Storage)
        return result
    }
}

// MARK: - Mixed Precision Kernels

/// FP16 Mixed-Precision kernels for high-performance vector similarity computation
///
/// These kernels store candidates in FP16 format (50% memory usage) while maintaining
/// FP32 computation precision. Leverages Apple Silicon NEON FP16 instructions for
/// efficient conversion and processing.
///
/// Key optimizations:
/// - FP16 candidate storage reduces memory bandwidth by 50%
/// - FP32 query processing and accumulation maintains numerical stability
/// - Automatic NEON vcvt.f32.f16 instruction generation
/// - Zero-allocation hot paths with pre-allocated output buffers
public enum MixedPrecisionKernels {

    // MARK: - Euclidean Squared Distance (Mixed Precision)

    /// Euclidean squared distance for 512-dimensional vectors (Mixed Precision)
    ///
    /// Algorithm: Query (FP32) + Candidates (FP16 storage) → FP32 computation
    /// Expected performance: 1.3-1.5× improvement due to reduced memory pressure
    @inlinable
    public static func range_euclid2_mixed_512(
        query: Vector512Optimized,
        candidatesFP16: [Vector512FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        #if DEBUG
        assert(range.lowerBound >= 0 && range.upperBound <= candidatesFP16.count,
               "Range \(range) out of bounds for \(candidatesFP16.count) candidates")
        assert(out.count >= range.count,
               "Output buffer too small: \(out.count) < \(range.count)")
        #endif

        let queryStorage = query.storage
        let laneCount = 128  // 512 / 4

        // Process each candidate in the specified range
        for i in range {
            let candidateStorage = candidatesFP16[i].storage

            // Four FP32 accumulators for optimal SIMD throughput
            var acc0 = SIMD4<Float>.zero
            var acc1 = SIMD4<Float>.zero
            var acc2 = SIMD4<Float>.zero
            var acc3 = SIMD4<Float>.zero

            // Process lanes in groups of 4 for better instruction scheduling
            for lane in stride(from: 0, to: laneCount, by: 4) {
                // Load query lanes (FP32)
                let q0 = queryStorage[lane + 0]
                let q1 = queryStorage[lane + 1]
                let q2 = queryStorage[lane + 2]
                let q3 = queryStorage[lane + 3]

                // Load candidate lanes (FP16) and convert to FP32
                // Uses NEON vcvt.f32.f16 instructions automatically
                let c0 = SIMD4<Float>(candidateStorage[lane + 0])
                let c1 = SIMD4<Float>(candidateStorage[lane + 1])
                let c2 = SIMD4<Float>(candidateStorage[lane + 2])
                let c3 = SIMD4<Float>(candidateStorage[lane + 3])

                // FP32 computation with fused multiply-add
                let d0 = q0 - c0; acc0.addProduct(d0, d0)
                let d1 = q1 - c1; acc1.addProduct(d1, d1)
                let d2 = q2 - c2; acc2.addProduct(d2, d2)
                let d3 = q3 - c3; acc3.addProduct(d3, d3)
            }

            // Horizontal reduction to scalar
            let result = (acc0 + acc1 + acc2 + acc3).sum()
            out[i - range.lowerBound] = result
        }
    }

    /// Euclidean squared distance for 768-dimensional vectors (Mixed Precision)
    @inlinable
    public static func range_euclid2_mixed_768(
        query: Vector768Optimized,
        candidatesFP16: [Vector768FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        #if DEBUG
        assert(range.lowerBound >= 0 && range.upperBound <= candidatesFP16.count,
               "Range out of bounds")
        assert(out.count >= range.count, "Output buffer too small")
        #endif

        let queryStorage = query.storage
        let laneCount = 192  // 768 / 4

        for i in range {
            let candidateStorage = candidatesFP16[i].storage

            var acc0 = SIMD4<Float>.zero
            var acc1 = SIMD4<Float>.zero
            var acc2 = SIMD4<Float>.zero
            var acc3 = SIMD4<Float>.zero

            for lane in stride(from: 0, to: laneCount, by: 4) {
                let q0 = queryStorage[lane + 0]
                let q1 = queryStorage[lane + 1]
                let q2 = queryStorage[lane + 2]
                let q3 = queryStorage[lane + 3]

                let c0 = SIMD4<Float>(candidateStorage[lane + 0])
                let c1 = SIMD4<Float>(candidateStorage[lane + 1])
                let c2 = SIMD4<Float>(candidateStorage[lane + 2])
                let c3 = SIMD4<Float>(candidateStorage[lane + 3])

                let d0 = q0 - c0; acc0.addProduct(d0, d0)
                let d1 = q1 - c1; acc1.addProduct(d1, d1)
                let d2 = q2 - c2; acc2.addProduct(d2, d2)
                let d3 = q3 - c3; acc3.addProduct(d3, d3)
            }

            let result = (acc0 + acc1 + acc2 + acc3).sum()
            out[i - range.lowerBound] = result
        }
    }

    /// Euclidean squared distance for 1536-dimensional vectors (Mixed Precision)
    @inlinable
    public static func range_euclid2_mixed_1536(
        query: Vector1536Optimized,
        candidatesFP16: [Vector1536FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        #if DEBUG
        assert(range.lowerBound >= 0 && range.upperBound <= candidatesFP16.count,
               "Range out of bounds")
        assert(out.count >= range.count, "Output buffer too small")
        #endif

        let queryStorage = query.storage
        let laneCount = 384  // 1536 / 4

        for i in range {
            let candidateStorage = candidatesFP16[i].storage

            var acc0 = SIMD4<Float>.zero
            var acc1 = SIMD4<Float>.zero
            var acc2 = SIMD4<Float>.zero
            var acc3 = SIMD4<Float>.zero

            for lane in stride(from: 0, to: laneCount, by: 4) {
                let q0 = queryStorage[lane + 0]
                let q1 = queryStorage[lane + 1]
                let q2 = queryStorage[lane + 2]
                let q3 = queryStorage[lane + 3]

                let c0 = SIMD4<Float>(candidateStorage[lane + 0])
                let c1 = SIMD4<Float>(candidateStorage[lane + 1])
                let c2 = SIMD4<Float>(candidateStorage[lane + 2])
                let c3 = SIMD4<Float>(candidateStorage[lane + 3])

                let d0 = q0 - c0; acc0.addProduct(d0, d0)
                let d1 = q1 - c1; acc1.addProduct(d1, d1)
                let d2 = q2 - c2; acc2.addProduct(d2, d2)
                let d3 = q3 - c3; acc3.addProduct(d3, d3)
            }

            let result = (acc0 + acc1 + acc2 + acc3).sum()
            out[i - range.lowerBound] = result
        }
    }

    // MARK: - Cosine Distance (Mixed Precision)

    /// Cosine distance for 512-dimensional vectors (Mixed Precision)
    ///
    /// Uses fused single-pass computation for dot product and magnitudes
    /// with FP32 precision throughout the calculation pipeline
    @inlinable
    public static func range_cosine_mixed_512(
        query: Vector512Optimized,
        candidatesFP16: [Vector512FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        #if DEBUG
        assert(range.lowerBound >= 0 && range.upperBound <= candidatesFP16.count,
               "Range out of bounds")
        assert(out.count >= range.count, "Output buffer too small")
        #endif

        let queryStorage = query.storage
        let laneCount = 128  // 512 / 4

        // Precompute query magnitude squared |q|²
        var queryMagSqAcc = SIMD4<Float>.zero
        for lane in 0..<laneCount {
            let q = queryStorage[lane]
            queryMagSqAcc.addProduct(q, q)
        }
        let queryMagSq = queryMagSqAcc.sum()

        for i in range {
            let candidateStorage = candidatesFP16[i].storage

            // Fused computation: dot product and candidate magnitude
            var dotAcc = SIMD4<Float>.zero
            var candMagSqAcc = SIMD4<Float>.zero

            for lane in 0..<laneCount {
                let q = queryStorage[lane]
                let c = SIMD4<Float>(candidateStorage[lane])  // FP16→FP32 conversion

                dotAcc.addProduct(q, c)      // q·c
                candMagSqAcc.addProduct(c, c) // |c|²
            }

            let dot = dotAcc.sum()
            let candMagSq = candMagSqAcc.sum()

            // Calculate cosine distance with zero-magnitude guards
            let denominatorSq = queryMagSq * candMagSq
            let epsilon: Float = 1e-9

            if denominatorSq <= epsilon {
                // Handle zero magnitude cases
                if queryMagSq <= epsilon && candMagSq <= epsilon {
                    out[i - range.lowerBound] = 0.0  // Both zero → distance 0
                } else {
                    out[i - range.lowerBound] = 1.0  // One zero → distance 1
                }
            } else {
                let denominator = sqrt(denominatorSq)
                let similarity = dot / denominator
                let clampedSim = max(-1.0, min(1.0, similarity))
                out[i - range.lowerBound] = 1.0 - clampedSim
            }
        }
    }

    /// Cosine distance for 768-dimensional vectors (Mixed Precision)
    @inlinable
    public static func range_cosine_mixed_768(
        query: Vector768Optimized,
        candidatesFP16: [Vector768FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        #if DEBUG
        assert(range.lowerBound >= 0 && range.upperBound <= candidatesFP16.count,
               "Range out of bounds")
        assert(out.count >= range.count, "Output buffer too small")
        #endif

        let queryStorage = query.storage
        let laneCount = 192  // 768 / 4

        // Precompute query magnitude squared
        var queryMagSqAcc = SIMD4<Float>.zero
        for lane in 0..<laneCount {
            let q = queryStorage[lane]
            queryMagSqAcc.addProduct(q, q)
        }
        let queryMagSq = queryMagSqAcc.sum()

        for i in range {
            let candidateStorage = candidatesFP16[i].storage

            var dotAcc = SIMD4<Float>.zero
            var candMagSqAcc = SIMD4<Float>.zero

            for lane in 0..<laneCount {
                let q = queryStorage[lane]
                let c = SIMD4<Float>(candidateStorage[lane])

                dotAcc.addProduct(q, c)
                candMagSqAcc.addProduct(c, c)
            }

            let dot = dotAcc.sum()
            let candMagSq = candMagSqAcc.sum()

            let denominatorSq = queryMagSq * candMagSq
            let epsilon: Float = 1e-9

            if denominatorSq <= epsilon {
                if queryMagSq <= epsilon && candMagSq <= epsilon {
                    out[i - range.lowerBound] = 0.0
                } else {
                    out[i - range.lowerBound] = 1.0
                }
            } else {
                let denominator = sqrt(denominatorSq)
                let similarity = dot / denominator
                let clampedSim = max(-1.0, min(1.0, similarity))
                out[i - range.lowerBound] = 1.0 - clampedSim
            }
        }
    }

    /// Cosine distance for 1536-dimensional vectors (Mixed Precision)
    @inlinable
    public static func range_cosine_mixed_1536(
        query: Vector1536Optimized,
        candidatesFP16: [Vector1536FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        #if DEBUG
        assert(range.lowerBound >= 0 && range.upperBound <= candidatesFP16.count,
               "Range out of bounds")
        assert(out.count >= range.count, "Output buffer too small")
        #endif

        let queryStorage = query.storage
        let laneCount = 384  // 1536 / 4

        // Precompute query magnitude squared
        var queryMagSqAcc = SIMD4<Float>.zero
        for lane in 0..<laneCount {
            let q = queryStorage[lane]
            queryMagSqAcc.addProduct(q, q)
        }
        let queryMagSq = queryMagSqAcc.sum()

        for i in range {
            let candidateStorage = candidatesFP16[i].storage

            var dotAcc = SIMD4<Float>.zero
            var candMagSqAcc = SIMD4<Float>.zero

            for lane in 0..<laneCount {
                let q = queryStorage[lane]
                let c = SIMD4<Float>(candidateStorage[lane])

                dotAcc.addProduct(q, c)
                candMagSqAcc.addProduct(c, c)
            }

            let dot = dotAcc.sum()
            let candMagSq = candMagSqAcc.sum()

            let denominatorSq = queryMagSq * candMagSq
            let epsilon: Float = 1e-9

            if denominatorSq <= epsilon {
                if queryMagSq <= epsilon && candMagSq <= epsilon {
                    out[i - range.lowerBound] = 0.0
                } else {
                    out[i - range.lowerBound] = 1.0
                }
            } else {
                let denominator = sqrt(denominatorSq)
                let similarity = dot / denominator
                let clampedSim = max(-1.0, min(1.0, similarity))
                out[i - range.lowerBound] = 1.0 - clampedSim
            }
        }
    }

    // MARK: - Batch Conversion Utilities

    /// Convert 512-dimensional vectors from FP32 to FP16 format
    public static func convertToFP16_512(_ vectors: [Vector512Optimized]) -> [Vector512FP16] {
        return vectors.map { Vector512FP16(from: $0) }
    }

    /// Convert 768-dimensional vectors from FP32 to FP16 format
    public static func convertToFP16_768(_ vectors: [Vector768Optimized]) -> [Vector768FP16] {
        return vectors.map { Vector768FP16(from: $0) }
    }

    /// Convert 1536-dimensional vectors from FP32 to FP16 format
    public static func convertToFP16_1536(_ vectors: [Vector1536Optimized]) -> [Vector1536FP16] {
        return vectors.map { Vector1536FP16(from: $0) }
    }

    // MARK: - Performance Analysis

    /// Estimates memory bandwidth improvement from FP16 storage
    public static func estimateMemoryImprovement(candidateCount: Int, dimension: Int) -> String {
        let fp32Bytes = candidateCount * dimension * 4
        let fp16Bytes = candidateCount * dimension * 2
        let savings = Double(fp32Bytes - fp16Bytes) / Double(fp32Bytes) * 100

        return """
        FP16 Mixed-Precision Analysis for \(candidateCount) candidates, \(dimension)D:
        - FP32 storage: \(fp32Bytes) bytes
        - FP16 storage: \(fp16Bytes) bytes
        - Memory savings: \(String(format: "%.1f", savings))%
        - Expected throughput improvement: 1.3-1.5× due to reduced memory pressure
        - Accuracy: Maintains FP32 computation precision
        """
    }
}

// MARK: - Integration Helpers

extension MixedPrecisionKernels {

    /// Check if mixed precision is beneficial for given parameters
    public static func shouldUseMixedPrecision(candidateCount: Int, dimension: Int) -> Bool {
        // Mixed precision beneficial for larger datasets where memory bandwidth is the bottleneck
        return candidateCount >= 100 && dimension >= 512
    }
}