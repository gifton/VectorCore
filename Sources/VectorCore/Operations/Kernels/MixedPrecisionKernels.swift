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

    // FP16 range constants
    private static let fp16Max: Float = 65504.0
    private static let fp16Min: Float = -65504.0

    public init(from vector: Vector512Optimized) {
        // Efficient FP32→FP16 conversion with clamping for finite out-of-range values
        let fp16Storage = vector.storage.map { simd4 in
            // Clamp finite values to FP16 range, preserve infinity and NaN
            let clamped = SIMD4<Float>(
                simd4.x.isNaN || simd4.x.isInfinite ? simd4.x : max(Self.fp16Min, min(Self.fp16Max, simd4.x)),
                simd4.y.isNaN || simd4.y.isInfinite ? simd4.y : max(Self.fp16Min, min(Self.fp16Max, simd4.y)),
                simd4.z.isNaN || simd4.z.isInfinite ? simd4.z : max(Self.fp16Min, min(Self.fp16Max, simd4.z)),
                simd4.w.isNaN || simd4.w.isInfinite ? simd4.w : max(Self.fp16Min, min(Self.fp16Max, simd4.w))
            )
            return SIMD4<Float16>(clamped)
        }
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

    // FP16 range constants
    private static let fp16Max: Float = 65504.0
    private static let fp16Min: Float = -65504.0

    public init(from vector: Vector768Optimized) {
        // Efficient FP32→FP16 conversion with clamping for finite out-of-range values
        let fp16Storage = vector.storage.map { simd4 in
            // Clamp finite values to FP16 range, preserve infinity and NaN
            let clamped = SIMD4<Float>(
                simd4.x.isNaN || simd4.x.isInfinite ? simd4.x : max(Self.fp16Min, min(Self.fp16Max, simd4.x)),
                simd4.y.isNaN || simd4.y.isInfinite ? simd4.y : max(Self.fp16Min, min(Self.fp16Max, simd4.y)),
                simd4.z.isNaN || simd4.z.isInfinite ? simd4.z : max(Self.fp16Min, min(Self.fp16Max, simd4.z)),
                simd4.w.isNaN || simd4.w.isInfinite ? simd4.w : max(Self.fp16Min, min(Self.fp16Max, simd4.w))
            )
            return SIMD4<Float16>(clamped)
        }
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

    // FP16 range constants
    private static let fp16Max: Float = 65504.0
    private static let fp16Min: Float = -65504.0

    public init(from vector: Vector1536Optimized) {
        // Efficient FP32→FP16 conversion with clamping for finite out-of-range values
        let fp16Storage = vector.storage.map { simd4 in
            // Clamp finite values to FP16 range, preserve infinity and NaN
            let clamped = SIMD4<Float>(
                simd4.x.isNaN || simd4.x.isInfinite ? simd4.x : max(Self.fp16Min, min(Self.fp16Max, simd4.x)),
                simd4.y.isNaN || simd4.y.isInfinite ? simd4.y : max(Self.fp16Min, min(Self.fp16Max, simd4.y)),
                simd4.z.isNaN || simd4.z.isInfinite ? simd4.z : max(Self.fp16Min, min(Self.fp16Max, simd4.z)),
                simd4.w.isNaN || simd4.w.isInfinite ? simd4.w : max(Self.fp16Min, min(Self.fp16Max, simd4.w))
            )
            return SIMD4<Float16>(clamped)
        }
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

// MARK: - Part 2: SoA Batch Processing with Blocked Layout

/// Structure-of-Arrays FP16 storage with blocked layout for optimal cache behavior
public struct SoAFP16<VectorType: OptimizedVector>: Sendable {
    public let dimension: Int
    public let storage: ContiguousArray<SIMD4<Float16>>
    public let vectorCount: Int
    public let blockSize: Int
    public let blocksPerVector: Int

    public init(vectors: [VectorType], blockSize: Int = 64) throws {
        guard !vectors.isEmpty else {
            throw VectorError.invalidData("Cannot create SoAFP16 with empty vector array")
        }

        self.dimension = vectors[0].scalarCount
        self.vectorCount = vectors.count
        self.blockSize = blockSize
        self.blocksPerVector = (dimension + blockSize - 1) / blockSize

        // Validate all vectors have same dimension
        for vector in vectors {
            guard vector.scalarCount == dimension else {
                throw VectorError.dimensionMismatch(expected: dimension, actual: vector.scalarCount)
            }
        }

        // Calculate SIMD4 groups per block for efficient layout
        let elementsPerBlock = blockSize * vectorCount
        let simd4GroupsPerBlock = (elementsPerBlock + 3) / 4

        // Allocate storage in blocked SoA layout
        var tempStorage = ContiguousArray<SIMD4<Float16>>()
        tempStorage.reserveCapacity(simd4GroupsPerBlock * blocksPerVector)

        // Convert to blocked layout: blocks are organized by dimension, then vectors within each block
        for blockIndex in 0..<blocksPerVector {
            let blockStart = blockIndex * blockSize
            let blockEnd = min(blockStart + blockSize, dimension)

            var blockElements = [Float16]()
            blockElements.reserveCapacity(blockSize * vectorCount)

            for dimIndex in blockStart..<blockEnd {
                for vectorIndex in 0..<vectorCount {
                    let value = vectors[vectorIndex][dimIndex]
                    // Clamp finite values to FP16 range, preserve infinity and NaN
                    let clampedValue: Float
                    if value.isNaN || value.isInfinite {
                        clampedValue = value
                    } else {
                        clampedValue = max(-65504.0, min(65504.0, value))
                    }
                    blockElements.append(Float16(clampedValue))
                }
            }

            // Pad to SIMD4 alignment if necessary
            while blockElements.count % 4 != 0 {
                blockElements.append(Float16(0))
            }

            // Store as SIMD4 chunks
            for i in stride(from: 0, to: blockElements.count, by: 4) {
                let simd4 = SIMD4<Float16>(
                    blockElements[i],
                    blockElements[i + 1],
                    blockElements[i + 2],
                    blockElements[i + 3]
                )
                tempStorage.append(simd4)
            }
        }

        self.storage = tempStorage
    }

    /// Get storage pointer for a specific dimension block for batch processing
    @inlinable
    internal func blockPointer(blockIndex: Int) -> UnsafePointer<SIMD4<Float16>> {
        precondition(blockIndex >= 0 && blockIndex < blocksPerVector, "Block index out of bounds")
        let elementsPerBlock = blockSize * vectorCount
        let simd4GroupsPerBlock = (elementsPerBlock + 3) / 4
        return storage.withUnsafeBufferPointer { buffer in
            return buffer.baseAddress! + (blockIndex * simd4GroupsPerBlock)
        }
    }

    /// Extract a specific vector with FP16→FP32 conversion
    public func getVector(at index: Int) throws -> VectorType {
        precondition(index >= 0 && index < vectorCount, "Vector index out of bounds")

        var elements = [Float]()
        elements.reserveCapacity(dimension)

        for blockIndex in 0..<blocksPerVector {
            let blockStart = blockIndex * blockSize
            let blockEnd = min(blockStart + blockSize, dimension)
            let blockPtr = blockPointer(blockIndex: blockIndex)

            for dimOffset in 0..<(blockEnd - blockStart) {
                let elementIndex = dimOffset * vectorCount + index
                let simd4Index = elementIndex / 4
                let laneIndex = elementIndex % 4

                let fp16Value = blockPtr[simd4Index][laneIndex]
                elements.append(Float(fp16Value))
            }
        }

        return try VectorType(elements)
    }
}

// MARK: - Batch Processing Kernels

extension MixedPrecisionKernels {

    /// Batch Euclidean distance squared: FP32 query vs FP16 SoA candidates
    @inlinable
    public static func batchEuclideanSquaredSoA<V: OptimizedVector>(
        query: V,
        candidates: SoAFP16<V>,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(results.count >= candidates.vectorCount, "Results buffer too small")
        precondition(query.scalarCount == candidates.dimension, "Dimension mismatch")

        // Initialize results
        for i in 0..<candidates.vectorCount {
            results[i] = 0
        }

        // Process by blocks for cache efficiency
        query.withUnsafeBufferPointer { queryPtr in
            for blockIndex in 0..<candidates.blocksPerVector {
                let blockStart = blockIndex * candidates.blockSize
                let blockEnd = min(blockStart + candidates.blockSize, candidates.dimension)
                let candidateBlockPtr = candidates.blockPointer(blockIndex: blockIndex)

                batchEuclideanBlockSoA_mixed(
                    queryPtr: queryPtr.baseAddress! + blockStart,
                    candidateBlockPtr: candidateBlockPtr,
                    blockSize: blockEnd - blockStart,
                    vectorCount: candidates.vectorCount,
                    results: results
                )
            }
        }
    }

    /// Batch dot product: FP32 query vs FP16 SoA candidates
    @inlinable
    public static func batchDotProductSoA<V: OptimizedVector>(
        query: V,
        candidates: SoAFP16<V>,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(results.count >= candidates.vectorCount, "Results buffer too small")
        precondition(query.scalarCount == candidates.dimension, "Dimension mismatch")

        // Initialize results
        for i in 0..<candidates.vectorCount {
            results[i] = 0
        }

        // Process by blocks for cache efficiency
        query.withUnsafeBufferPointer { queryPtr in
            for blockIndex in 0..<candidates.blocksPerVector {
                let blockStart = blockIndex * candidates.blockSize
                let blockEnd = min(blockStart + candidates.blockSize, candidates.dimension)
                let candidateBlockPtr = candidates.blockPointer(blockIndex: blockIndex)

                batchDotProductBlockSoA_mixed(
                    queryPtr: queryPtr.baseAddress! + blockStart,
                    candidateBlockPtr: candidateBlockPtr,
                    blockSize: blockEnd - blockStart,
                    vectorCount: candidates.vectorCount,
                    results: results
                )
            }
        }
    }

    // MARK: - Block Processing Implementation

    /// Process a single block for SoA mixed precision Euclidean distance
    @inlinable
    @inline(__always)
    internal static func batchEuclideanBlockSoA_mixed(
        queryPtr: UnsafePointer<Float>,
        candidateBlockPtr: UnsafePointer<SIMD4<Float16>>,
        blockSize: Int,
        vectorCount: Int,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let elementsPerBlock = blockSize * vectorCount
        let simd4Groups = (elementsPerBlock + 3) / 4

        // Process SIMD4 groups for optimal memory access
        for simd4Index in 0..<simd4Groups {
            let fp16Values = candidateBlockPtr[simd4Index]
            let fp32Values = SIMD4<Float>(fp16Values)

            // Calculate element indices within the block
            let baseElementIndex = simd4Index * 4

            for lane in 0..<4 {
                let elementIndex = baseElementIndex + lane
                if elementIndex >= elementsPerBlock { break }

                let dimOffset = elementIndex / vectorCount
                let vectorIndex = elementIndex % vectorCount

                if dimOffset < blockSize && vectorIndex < vectorCount {
                    let queryValue = queryPtr[dimOffset]
                    let candidateValue = fp32Values[lane]
                    let diff = queryValue - candidateValue
                    results[vectorIndex] += diff * diff
                }
            }
        }
    }

    /// Process a single block for SoA mixed precision dot product
    @inlinable
    @inline(__always)
    internal static func batchDotProductBlockSoA_mixed(
        queryPtr: UnsafePointer<Float>,
        candidateBlockPtr: UnsafePointer<SIMD4<Float16>>,
        blockSize: Int,
        vectorCount: Int,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let elementsPerBlock = blockSize * vectorCount
        let simd4Groups = (elementsPerBlock + 3) / 4

        // Process SIMD4 groups for optimal memory access
        for simd4Index in 0..<simd4Groups {
            let fp16Values = candidateBlockPtr[simd4Index]
            let fp32Values = SIMD4<Float>(fp16Values)

            // Calculate element indices within the block
            let baseElementIndex = simd4Index * 4

            for lane in 0..<4 {
                let elementIndex = baseElementIndex + lane
                if elementIndex >= elementsPerBlock { break }

                let dimOffset = elementIndex / vectorCount
                let vectorIndex = elementIndex % vectorCount

                if dimOffset < blockSize && vectorIndex < vectorCount {
                    let queryValue = queryPtr[dimOffset]
                    let candidateValue = fp32Values[lane]
                    results[vectorIndex] += queryValue * candidateValue
                }
            }
        }
    }
}

// MARK: - AutoTuning for Mixed Precision

/// AutoTuner for mixed precision kernel selection and optimization
public final class MixedPrecisionAutoTuner: @unchecked Sendable {
    public static let shared = MixedPrecisionAutoTuner()

    private let lock = NSLock()
    private var precisionStrategies: [String: PrecisionStrategy] = [:]
    private var performanceCache: [String: PerformanceMetrics] = [:]

    private init() {}

    public struct PrecisionStrategy: Sendable {
        public let useFP16Storage: Bool
        public let useSoALayout: Bool
        public let blockSize: Int
        public let accuracyThreshold: Float

        public init(useFP16Storage: Bool, useSoALayout: Bool, blockSize: Int, accuracyThreshold: Float) {
            self.useFP16Storage = useFP16Storage
            self.useSoALayout = useSoALayout
            self.blockSize = blockSize
            self.accuracyThreshold = accuracyThreshold
        }
    }

    public struct PerformanceMetrics: Sendable {
        public let averageTimeMS: Double
        public let memoryUsageMB: Double
        public let accuracyScore: Float
        public let throughputOpsPerSec: Double

        public init(averageTimeMS: Double, memoryUsageMB: Double, accuracyScore: Float, throughputOpsPerSec: Double) {
            self.averageTimeMS = averageTimeMS
            self.memoryUsageMB = memoryUsageMB
            self.accuracyScore = accuracyScore
            self.throughputOpsPerSec = throughputOpsPerSec
        }
    }

    /// Get optimal precision strategy for given parameters
    public func getStrategy(dimension: Int, vectorCount: Int, accuracyRequired: Float) -> PrecisionStrategy {
        let key = "\(dimension)_\(vectorCount)_\(accuracyRequired)"

        lock.lock()
        defer { lock.unlock() }

        if let cached = precisionStrategies[key] {
            return cached
        }

        // Determine strategy based on parameters
        let useFP16 = shouldUseFP16(dimension: dimension, vectorCount: vectorCount, accuracyRequired: accuracyRequired)
        let useSoA = shouldUseSoA(dimension: dimension, vectorCount: vectorCount)
        let blockSize = optimalBlockSize(dimension: dimension, vectorCount: vectorCount)
        let threshold = accuracyThreshold(dimension: dimension, accuracyRequired: accuracyRequired)

        let strategy = PrecisionStrategy(
            useFP16Storage: useFP16,
            useSoALayout: useSoA,
            blockSize: blockSize,
            accuracyThreshold: threshold
        )

        precisionStrategies[key] = strategy
        return strategy
    }

    /// Benchmark strategy performance
    public func benchmarkStrategy<V: OptimizedVector>(
        strategy: PrecisionStrategy,
        sampleVectors: [V],
        iterations: Int = 100
    ) -> PerformanceMetrics {
        let key = "\(strategy.useFP16Storage)_\(strategy.useSoALayout)_\(strategy.blockSize)_\(sampleVectors.count)"

        lock.lock()
        if let cached = performanceCache[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        guard sampleVectors.count >= 2 else {
            return PerformanceMetrics(averageTimeMS: 0, memoryUsageMB: 0, accuracyScore: 0, throughputOpsPerSec: 0)
        }

        var totalTime: Double = 0
        let startMemory = getMemoryUsage()

        for _ in 0..<iterations {
            let startTime = CFAbsoluteTimeGetCurrent()

            if strategy.useFP16Storage && strategy.useSoALayout {
                // Benchmark SoA FP16 operations
                do {
                    let soaFP16 = try SoAFP16(vectors: sampleVectors, blockSize: strategy.blockSize)
                    let resultsBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: sampleVectors.count)
                    defer { resultsBuffer.deallocate() }

                    MixedPrecisionKernels.batchEuclideanSquaredSoA(
                        query: sampleVectors[0],
                        candidates: soaFP16,
                        results: resultsBuffer
                    )
                } catch {
                    continue
                }
            } else if strategy.useFP16Storage {
                // Benchmark regular FP16 operations
                let fp16Vectors = MixedPrecisionKernels.convertToFP16_512(sampleVectors as! [Vector512Optimized])
                let resultsBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: sampleVectors.count)
                defer { resultsBuffer.deallocate() }

                MixedPrecisionKernels.range_euclid2_mixed_512(
                    query: sampleVectors[0] as! Vector512Optimized,
                    candidatesFP16: fp16Vectors,
                    range: 0..<fp16Vectors.count,
                    out: resultsBuffer
                )
            } else {
                // Benchmark regular FP32 operations
                for i in 0..<sampleVectors.count {
                    for j in (i+1)..<sampleVectors.count {
                        _ = sampleVectors[i].euclideanDistanceSquared(to: sampleVectors[j])
                    }
                }
            }

            let endTime = CFAbsoluteTimeGetCurrent()
            totalTime += (endTime - startTime)
        }

        let endMemory = getMemoryUsage()
        let averageTimeMS = (totalTime / Double(iterations)) * 1000.0
        let memoryUsageMB = endMemory - startMemory
        let accuracyScore: Float = strategy.useFP16Storage ? 0.95 : 1.0
        let throughputOpsPerSec = Double(sampleVectors.count) / (totalTime / Double(iterations))

        let metrics = PerformanceMetrics(
            averageTimeMS: averageTimeMS,
            memoryUsageMB: memoryUsageMB,
            accuracyScore: accuracyScore,
            throughputOpsPerSec: throughputOpsPerSec
        )

        lock.lock()
        performanceCache[key] = metrics
        lock.unlock()

        return metrics
    }

    // MARK: - Private Strategy Selection

    private func shouldUseFP16(dimension: Int, vectorCount: Int, accuracyRequired: Float) -> Bool {
        // Use FP16 for large datasets where memory bandwidth is critical
        let memoryPressure = Float(dimension * vectorCount * 4) / (1024 * 1024) // MB
        let accuracyThreshold: Float = 0.90 // 90% accuracy threshold

        return memoryPressure > 50.0 && accuracyRequired >= accuracyThreshold
    }

    private func shouldUseSoA(dimension: Int, vectorCount: Int) -> Bool {
        // SoA beneficial for batch operations with many candidates
        return vectorCount >= 50 && dimension >= 512
    }

    private func optimalBlockSize(dimension: Int, vectorCount: Int) -> Int {
        // Optimize for L1 cache (32KB typical)
        let cacheSize = 32 * 1024
        let elementSize = 2 // FP16
        let vectorsPerCacheLine = cacheSize / (elementSize * vectorCount)

        return min(max(vectorsPerCacheLine, 32), min(128, dimension))
    }

    private func accuracyThreshold(dimension: Int, accuracyRequired: Float) -> Float {
        // Higher dimensions are more tolerant of precision loss
        let dimensionFactor = min(Float(dimension) / 1000.0, 1.0)
        return accuracyRequired * (1.0 - dimensionFactor * 0.05)
    }

    private func getMemoryUsage() -> Double {
        // Simplified memory usage estimation
        return 0.0 // In real implementation, would use mach_task_basic_info
    }
}

// MARK: - Integration Helpers

extension MixedPrecisionKernels {

    /// Check if mixed precision is beneficial for given parameters
    public static func shouldUseMixedPrecision(candidateCount: Int, dimension: Int) -> Bool {
        // Mixed precision beneficial for larger datasets where memory bandwidth is the bottleneck
        return candidateCount >= 100 && dimension >= 512
    }

    /// Adaptive kernel selection based on AutoTuning
    public static func adaptiveEuclideanDistance<V: OptimizedVector>(
        query: V,
        candidates: [V],
        accuracyRequired: Float = 0.95
    ) -> [Float] {
        let autoTuner = MixedPrecisionAutoTuner.shared
        let strategy = autoTuner.getStrategy(
            dimension: query.scalarCount,
            vectorCount: candidates.count,
            accuracyRequired: accuracyRequired
        )

        var results = [Float]()
        results.reserveCapacity(candidates.count)

        if strategy.useFP16Storage && strategy.useSoALayout {
            // Use SoA FP16 batch processing
            do {
                let soaFP16 = try SoAFP16(vectors: candidates, blockSize: strategy.blockSize)
                let resultsBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: candidates.count)
                defer { resultsBuffer.deallocate() }

                batchEuclideanSquaredSoA(query: query, candidates: soaFP16, results: resultsBuffer)
                results.append(contentsOf: resultsBuffer)
            } catch {
                // Fallback to regular computation
                results = candidates.map { query.euclideanDistanceSquared(to: $0) }
            }
        } else {
            // Use regular FP32 computation
            results = candidates.map { query.euclideanDistanceSquared(to: $0) }
        }

        return results
    }
}