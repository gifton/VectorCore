//
//  QuantizedKernels.swift
//  VectorCore
//
//  INT8 quantized kernels for high-performance vector compression
//  Provides 4x memory reduction with minimal accuracy loss
//  Phase 1: Core algorithms and distance kernels with SIMD4<Int8> optimization
//

import Foundation
import simd

// MARK: - Linear Quantization Parameters

/// Parameters for linear (affine) quantization: q = round(x/scale + zeroPoint).
public struct LinearQuantizationParams: Sendable, Equatable, Hashable, Codable {
    public let scale: Float
    public let zeroPoint: Int8
    public let minValue: Float
    public let maxValue: Float
    public let isSymmetric: Bool

    public init(minValue: Float, maxValue: Float, symmetric: Bool = true) {
        self.minValue = minValue
        self.maxValue = maxValue
        self.isSymmetric = symmetric

        if symmetric {
            // Symmetric quantization: zeroPoint = 0. Range [-127, 127] is used to maintain symmetry.
            let absMax = max(abs(minValue), abs(maxValue))
            // Ensure scale is not zero or subnormal if the range is zero.
            self.scale = absMax <= Float.leastNormalMagnitude ? 1.0 : absMax / 127.0
            self.zeroPoint = 0
        } else {
            // Asymmetric quantization: Range [-128, 127].
            let range = maxValue - minValue
            self.scale = range <= Float.leastNormalMagnitude ? 1.0 : range / 255.0

            // Calculate zero point: zp = round(-128 - min/scale)
            let zpFloat = -128.0 - (minValue / self.scale)
            // Nudge and clamp the zero point to the valid Int8 range.
            self.zeroPoint = Int8(clamping: Int(zpFloat.rounded()))
        }
    }

    @inlinable
    public func quantize(_ value: Float) -> Int8 {
        let scaledValue = value / scale + Float(zeroPoint)
        // Use clamping conversion for safety and efficiency.
        return Int8(clamping: Int(scaledValue.rounded(.toNearestOrAwayFromZero)))
    }

    @inlinable
    public func dequantize(_ quantizedValue: Int8) -> Float {
        // Inverse transformation: x' = scale * (q - zeroPoint)
        return scale * (Float(quantizedValue) - Float(zeroPoint))
    }
}

// MARK: - INT8 Storage Types

/// Protocol defining the structure and behavior of INT8 quantized vectors.
public protocol QuantizedVectorINT8: Sendable, VectorProtocol {
    associatedtype FloatVectorType: OptimizedVector

    /// Storage uses SIMD4<Int8> for efficient packing (4 bytes per lane).
    var storage: ContiguousArray<SIMD4<Int8>> { get }
    var quantizationParams: LinearQuantizationParams { get }

    static var laneCount: Int { get }

    init(storage: ContiguousArray<SIMD4<Int8>>, params: LinearQuantizationParams)
    init(from vector: FloatVectorType, params: LinearQuantizationParams?)
    func toFP32() -> FloatVectorType
}

extension QuantizedVectorINT8 {
    public var dimension: Int { return Self.laneCount * 4 }
    // Note: laneCount is defined in each concrete struct (Vector512INT8, Vector768INT8, Vector1536INT8)
    // to return the correct number of SIMD4 chunks (e.g., 128 for 512-dim, not 4)
    public var memoryFootprint: Int {
        return Self.laneCount * 4 + MemoryLayout<LinearQuantizationParams>.size
    }

    /// Initializes by quantizing an FP32 vector. Auto-calibrates parameters if not provided.
    public init(from vector: FloatVectorType, params: LinearQuantizationParams? = nil) {
        let finalParams: LinearQuantizationParams

        if let providedParams = params {
            finalParams = providedParams
        } else {
            // Auto-calibrate parameters by finding the min/max range of the input vector.
            // Use SIMDStorage extension for efficient min/max finding
            let (minVal, maxVal) = vector.storage.minMax()
            // Default to symmetric quantization for optimization compatibility.
            finalParams = LinearQuantizationParams(minValue: minVal, maxValue: maxVal, symmetric: true)
        }

        // Perform the quantization and pack into storage (Vectorized).
        let quantizedStorage = Self.vectorizedQuantize(vector: vector, params: finalParams)
        self.init(storage: quantizedStorage, params: finalParams)
    }

    /// Optimized quantization using SIMD operations.
    private static func vectorizedQuantize(vector: FloatVectorType, params: LinearQuantizationParams) -> ContiguousArray<SIMD4<Int8>> {
        let invScale = 1.0 / params.scale
        let vInvScale = SIMD4<Float>(repeating: invScale)
        let vOffset = SIMD4<Float>(repeating: Float(params.zeroPoint))
        let vMin = SIMD4<Float>(repeating: -128.0)
        let vMax = SIMD4<Float>(repeating: 127.0)

        let quantizedStorage = ContiguousArray<SIMD4<Int8>>(unsafeUninitializedCapacity: Self.laneCount) { buffer, count in
            vector.storage.withUnsafeBufferPointer { fp32Ptr in
                for i in 0..<Self.laneCount {
                    let vInput = fp32Ptr[i]

                    // Calculate: (x * invScale) + offset (FMA).
                    var vQuantized = vInput * vInvScale + vOffset

                    // Round and Clamp.
                    vQuantized = simd_max(vMin, simd_min(vMax, vQuantized.rounded(.toNearestOrAwayFromZero)))

                    // Convert FP32 -> INT32 with explicit conversion
                    let vInt32 = SIMD4<Int32>(Int32(vQuantized.x), Int32(vQuantized.y), Int32(vQuantized.z), Int32(vQuantized.w))

                    // Narrowing conversion using `clamping:` (saturating conversion).
                    buffer[i] = SIMD4<Int8>(clamping: vInt32)
                }
            }
            count = Self.laneCount
        }
        return quantizedStorage
    }

    /// Dequantizes the vector back to FP32.
    public func toFP32() -> FloatVectorType {
        let params = self.quantizationParams

        let fp32Storage = ContiguousArray<SIMD4<Float>>(unsafeUninitializedCapacity: Self.laneCount) { buffer, count in
            self.storage.withUnsafeBufferPointer { int8Ptr in
                for i in 0..<Self.laneCount {
                    // Use the optimized conversion helper.
                    buffer[i] = QuantizedKernels.convertToFP32_NEON(
                        int8Vec: int8Ptr[i],
                        scale: params.scale,
                        zeroPoint: params.zeroPoint
                    )
                }
            }
            count = Self.laneCount
        }
        // Convert storage to array for vector initialization
        let fp32Array = fp32Storage.flattenToArray()
        return try! FloatVectorType(fp32Array)
    }
}

// MARK: Specific INT8 Vector Types

public struct Vector512INT8: QuantizedVectorINT8 {
    public typealias FloatVectorType = Vector512Optimized
    public var storage: ContiguousArray<SIMD4<Int8>>
    public let quantizationParams: LinearQuantizationParams

    public static let laneCount: Int = 128  // 512 / 4

    public init(storage: ContiguousArray<SIMD4<Int8>>, params: LinearQuantizationParams) {
        self.storage = storage
        self.quantizationParams = params
    }
}

public struct Vector768INT8: QuantizedVectorINT8 {
    public typealias FloatVectorType = Vector768Optimized
    public var storage: ContiguousArray<SIMD4<Int8>>
    public let quantizationParams: LinearQuantizationParams

    public static let laneCount: Int = 192  // 768 / 4

    public init(storage: ContiguousArray<SIMD4<Int8>>, params: LinearQuantizationParams) {
        self.storage = storage
        self.quantizationParams = params
    }
}

public struct Vector1536INT8: QuantizedVectorINT8 {
    public typealias FloatVectorType = Vector1536Optimized
    public var storage: ContiguousArray<SIMD4<Int8>>
    public let quantizationParams: LinearQuantizationParams

    public static let laneCount: Int = 384  // 1536 / 4

    public init(storage: ContiguousArray<SIMD4<Int8>>, params: LinearQuantizationParams) {
        self.storage = storage
        self.quantizationParams = params
    }
}

// MARK: - VectorProtocol Implementations for Quantized Types

extension Vector512INT8 {
    public typealias Scalar = Float  // We report as Float for compatibility
    public typealias Storage = ContiguousArray<SIMD4<Int8>>
    public var scalarCount: Int { 512 }

    public init() {
        self.init(storage: ContiguousArray(repeating: SIMD4<Int8>(), count: 128),
                  params: LinearQuantizationParams(minValue: 0, maxValue: 1))
    }

    public init(_ array: [Float]) throws {
        guard array.count == 512 else {
            throw VectorError.dimensionMismatch(expected: 512, actual: array.count)
        }

        let minVal = array.min() ?? 0
        let maxVal = array.max() ?? 1
        let params = LinearQuantizationParams(minValue: minVal, maxValue: maxVal)

        var storage = ContiguousArray<SIMD4<Int8>>()
        storage.reserveCapacity(128)

        for i in stride(from: 0, to: 512, by: 4) {
            let simd4 = SIMD4<Int8>(
                params.quantize(array[i]),
                params.quantize(array[i + 1]),
                params.quantize(array[i + 2]),
                params.quantize(array[i + 3])
            )
            storage.append(simd4)
        }

        self.init(storage: storage, params: params)
    }

    public init(repeating value: Float) {
        let params = LinearQuantizationParams(minValue: value, maxValue: value)
        let quantized = params.quantize(value)
        let simd4 = SIMD4<Int8>(repeating: quantized)
        self.init(storage: ContiguousArray(repeating: simd4, count: 128), params: params)
    }

    public func toArray() -> [Float] {
        return toFP32().toArray()
    }

    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        let fp32Vector = toFP32()
        return try fp32Vector.withUnsafeBufferPointer(body)
    }

    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        // Convert to FP32, apply transformation, then quantize back
        var fp32Vector = toFP32()
        let result = try fp32Vector.withUnsafeMutableBufferPointer(body)
        let newArray = fp32Vector.toArray()
        self = try! Vector512INT8(newArray)
        return result
    }
}

extension Vector768INT8 {
    public typealias Scalar = Float
    public typealias Storage = ContiguousArray<SIMD4<Int8>>
    public var scalarCount: Int { 768 }

    public init() {
        self.init(storage: ContiguousArray(repeating: SIMD4<Int8>(), count: 192),
                  params: LinearQuantizationParams(minValue: 0, maxValue: 1))
    }

    public init(_ array: [Float]) throws {
        guard array.count == 768 else {
            throw VectorError.dimensionMismatch(expected: 768, actual: array.count)
        }

        let minVal = array.min() ?? 0
        let maxVal = array.max() ?? 1
        let params = LinearQuantizationParams(minValue: minVal, maxValue: maxVal)

        var storage = ContiguousArray<SIMD4<Int8>>()
        storage.reserveCapacity(192)

        for i in stride(from: 0, to: 768, by: 4) {
            let simd4 = SIMD4<Int8>(
                params.quantize(array[i]),
                params.quantize(array[i + 1]),
                params.quantize(array[i + 2]),
                params.quantize(array[i + 3])
            )
            storage.append(simd4)
        }

        self.init(storage: storage, params: params)
    }

    public init(repeating value: Float) {
        let params = LinearQuantizationParams(minValue: value, maxValue: value)
        let quantized = params.quantize(value)
        let simd4 = SIMD4<Int8>(repeating: quantized)
        self.init(storage: ContiguousArray(repeating: simd4, count: 192), params: params)
    }

    public func toArray() -> [Float] {
        return toFP32().toArray()
    }

    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        let fp32Vector = toFP32()
        return try fp32Vector.withUnsafeBufferPointer(body)
    }

    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        var fp32Vector = toFP32()
        let result = try fp32Vector.withUnsafeMutableBufferPointer(body)
        let newArray = fp32Vector.toArray()
        self = try! Vector768INT8(newArray)
        return result
    }
}

extension Vector1536INT8 {
    public typealias Scalar = Float
    public typealias Storage = ContiguousArray<SIMD4<Int8>>
    public var scalarCount: Int { 1536 }

    public init() {
        self.init(storage: ContiguousArray(repeating: SIMD4<Int8>(), count: 384),
                  params: LinearQuantizationParams(minValue: 0, maxValue: 1))
    }

    public init(_ array: [Float]) throws {
        guard array.count == 1536 else {
            throw VectorError.dimensionMismatch(expected: 1536, actual: array.count)
        }

        let minVal = array.min() ?? 0
        let maxVal = array.max() ?? 1
        let params = LinearQuantizationParams(minValue: minVal, maxValue: maxVal)

        var storage = ContiguousArray<SIMD4<Int8>>()
        storage.reserveCapacity(384)

        for i in stride(from: 0, to: 1536, by: 4) {
            let simd4 = SIMD4<Int8>(
                params.quantize(array[i]),
                params.quantize(array[i + 1]),
                params.quantize(array[i + 2]),
                params.quantize(array[i + 3])
            )
            storage.append(simd4)
        }

        self.init(storage: storage, params: params)
    }

    public init(repeating value: Float) {
        let params = LinearQuantizationParams(minValue: value, maxValue: value)
        let quantized = params.quantize(value)
        let simd4 = SIMD4<Int8>(repeating: quantized)
        self.init(storage: ContiguousArray(repeating: simd4, count: 384), params: params)
    }

    public func toArray() -> [Float] {
        return toFP32().toArray()
    }

    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        let fp32Vector = toFP32()
        return try fp32Vector.withUnsafeBufferPointer(body)
    }

    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        var fp32Vector = toFP32()
        let result = try fp32Vector.withUnsafeMutableBufferPointer(body)
        let newArray = fp32Vector.toArray()
        self = try! Vector1536INT8(newArray)
        return result
    }
}

// MARK: - Core Distance Kernels and Helpers

@usableFromInline
internal enum QuantizedKernels {

    // MARK: Conversion Helpers (NEON Optimization)

    /// Optimized conversion from INT8 to FP32, applying dequantization parameters.
    /// Optimized FMA: x' = (q_f - zp_f) * scale => x' = q_f * scale + (-zp_f * scale)
    @inline(__always)
    @usableFromInline
    internal static func convertToFP32_NEON(
        int8Vec: SIMD4<Int8>,
        scale: Float,
        zeroPoint: Int8
    ) -> SIMD4<Float> {
        // 1. Widen Int8 -> Int32 (Sign extension).
        let int32Vec = SIMD4<Int32>(Int32(int8Vec.x), Int32(int8Vec.y), Int32(int8Vec.z), Int32(int8Vec.w))

        // 2. Convert Int32 -> Float.
        let fp32Vec = SIMD4<Float>(Float(int32Vec.x), Float(int32Vec.y), Float(int32Vec.z), Float(int32Vec.w))

        // 3. Apply FMA optimization.
        let scaleVec = SIMD4<Float>(repeating: scale)

        if zeroPoint == 0 {
            // Symmetric optimization.
            return fp32Vec * scaleVec
        }

        // Asymmetric optimization.
        // Calculate bias: -zp_f * scale
        let bias = (-Float(zeroPoint)) * scale
        let biasVec = SIMD4<Float>(repeating: bias)

        // result = bias + scale * q_f
        return biasVec.addingProduct(scaleVec, fp32Vec)
    }
}

// MARK: Euclidean Distance Kernels

extension QuantizedKernels {

    // MARK: INT8 vs INT8 Euclidean

    /// Generic implementation for INT8 vs INT8 Euclidean distance.
    @inlinable
    internal static func euclidean_generic_int8<V: QuantizedVectorINT8>(query: V, candidate: V) -> Float {

        // Robustness Check: The optimized integer path D_f^2 ≈ S^2 * Sum[(Q_i8 - C_i8)^2] is accurate only if:
        // 1. Quantization is symmetric (ZeroPoints are 0).
        // 2. Scales are identical (S_q == S_c).

        let isSymmetric = query.quantizationParams.isSymmetric && candidate.quantizationParams.isSymmetric

        // Use a relative tolerance for scale comparison.
        let scaleTolerance = 1e-6 * query.quantizationParams.scale
        let scalesMatch = abs(query.quantizationParams.scale - candidate.quantizationParams.scale) <= scaleTolerance

        if !isSymmetric || !scalesMatch {
            // If conditions aren't met, fallback to the mixed-precision path (Dequantize-then-Compute) for correctness.
            return euclidean_generic_mixed(query: query, candidate: candidate.toFP32())
        }

        // Optimized path for identical, symmetric quantization parameters.
        let laneCount = V.laneCount

        // Use Int32 accumulators. Max accumulation for 1536 dimensions is safe within Int32 limits.
        // Max diff^2 = 254^2 = 64516. Total = 1536 * 64516 ≈ 99 million. Int32 max ≈ 2.1 billion.
        var acc1: Int32 = 0; var acc2: Int32 = 0; var acc3: Int32 = 0; var acc4: Int32 = 0

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidate.storage.withUnsafeBufferPointer { candidatePtr in
                let qINT8 = queryPtr.baseAddress!; let cINT8 = candidatePtr.baseAddress!

                // Process in blocks for maximum ILP, handling remainders
                let fullBlocks = laneCount / 16
                let remainder = laneCount % 16

                // Process full blocks of 16 lanes
                for blockIdx in 0..<fullBlocks {
                    let baseIdx = blockIdx * 16

                    // Unroll 16 lanes, distributing across 4 accumulators.
                    accumulateEuclidDiffSq(qINT8[baseIdx+0], cINT8[baseIdx+0], &acc1)
                    accumulateEuclidDiffSq(qINT8[baseIdx+1], cINT8[baseIdx+1], &acc2)
                    accumulateEuclidDiffSq(qINT8[baseIdx+2], cINT8[baseIdx+2], &acc3)
                    accumulateEuclidDiffSq(qINT8[baseIdx+3], cINT8[baseIdx+3], &acc4)

                    accumulateEuclidDiffSq(qINT8[baseIdx+4], cINT8[baseIdx+4], &acc1)
                    accumulateEuclidDiffSq(qINT8[baseIdx+5], cINT8[baseIdx+5], &acc2)
                    accumulateEuclidDiffSq(qINT8[baseIdx+6], cINT8[baseIdx+6], &acc3)
                    accumulateEuclidDiffSq(qINT8[baseIdx+7], cINT8[baseIdx+7], &acc4)

                    accumulateEuclidDiffSq(qINT8[baseIdx+8], cINT8[baseIdx+8], &acc1)
                    accumulateEuclidDiffSq(qINT8[baseIdx+9], cINT8[baseIdx+9], &acc2)
                    accumulateEuclidDiffSq(qINT8[baseIdx+10], cINT8[baseIdx+10], &acc3)
                    accumulateEuclidDiffSq(qINT8[baseIdx+11], cINT8[baseIdx+11], &acc4)

                    accumulateEuclidDiffSq(qINT8[baseIdx+12], cINT8[baseIdx+12], &acc1)
                    accumulateEuclidDiffSq(qINT8[baseIdx+13], cINT8[baseIdx+13], &acc2)
                    accumulateEuclidDiffSq(qINT8[baseIdx+14], cINT8[baseIdx+14], &acc3)
                    accumulateEuclidDiffSq(qINT8[baseIdx+15], cINT8[baseIdx+15], &acc4)
                }

                // Handle remainder lanes
                if remainder > 0 {
                    let baseIdx = fullBlocks * 16
                    for j in 0..<remainder {
                        let accIdx = j % 4
                        switch accIdx {
                        case 0: accumulateEuclidDiffSq(qINT8[baseIdx+j], cINT8[baseIdx+j], &acc1)
                        case 1: accumulateEuclidDiffSq(qINT8[baseIdx+j], cINT8[baseIdx+j], &acc2)
                        case 2: accumulateEuclidDiffSq(qINT8[baseIdx+j], cINT8[baseIdx+j], &acc3)
                        case 3: accumulateEuclidDiffSq(qINT8[baseIdx+j], cINT8[baseIdx+j], &acc4)
                        default: break
                        }
                    }
                }
            }
        }

        // Final reduction and scaling.
        let totalAccumulator = acc1 + acc2 + acc3 + acc4

        // Scaling factor is S^2.
        let scale = query.quantizationParams.scale

        // D = sqrt(S^2 * Sum(diff^2)) = S * sqrt(Sum(diff^2))
        return scale * sqrt(Float(totalAccumulator))
    }

    /// Helper to compute (q-c)^2 and accumulate, promoting to Int16 intermediates.
    @inline(__always)
    @usableFromInline internal static func accumulateEuclidDiffSq(_ q: SIMD4<Int8>, _ c: SIMD4<Int8>, _ acc: inout Int32) {
        // Promote to Int16 before subtraction. Use wrapping subtraction (&-).
        // Sign-extend Int8 to Int16 for proper arithmetic
        let qInt16 = SIMD4<Int16>(Int16(q.x), Int16(q.y), Int16(q.z), Int16(q.w))
        let cInt16 = SIMD4<Int16>(Int16(c.x), Int16(c.y), Int16(c.z), Int16(c.w))
        let diff = qInt16 &- cInt16
        // Square the differences and sum. Use wrapping arithmetic (&+, &*).
        acc &+= Int32(diff.x &* diff.x) &+ Int32(diff.y &* diff.y) &+ Int32(diff.z &* diff.z) &+ Int32(diff.w &* diff.w)
    }

    // MARK: Mixed Precision Euclidean (INT8 vs FP32)

    /// Generic implementation for mixed precision Euclidean distance (Dequantize-then-Compute).
    @inlinable
    internal static func euclidean_generic_mixed<V_INT8: QuantizedVectorINT8, V_FP32: OptimizedVector>(
        query: V_INT8, candidate: V_FP32
    ) -> Float where V_INT8.FloatVectorType == V_FP32 {

        let laneCount = V_INT8.laneCount
        let queryParams = query.quantizationParams
        let scale = queryParams.scale
        let zeroPoint = queryParams.zeroPoint

        // Use SIMD4<Float> accumulators for efficient FMA operations.
        var s0 = SIMD4<Float>.zero; var s1 = SIMD4<Float>.zero
        var s2 = SIMD4<Float>.zero; var s3 = SIMD4<Float>.zero

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidate.storage.withUnsafeBufferPointer { candidatePtr in
                let qINT8 = queryPtr.baseAddress!; let cFP32 = candidatePtr.baseAddress!

                // Process in blocks with proper remainder handling
                let fullBlocks = laneCount / 16
                let remainder = laneCount % 16

                // Process full blocks of 16 lanes
                for blockIdx in 0..<fullBlocks {
                    let baseIdx = blockIdx * 16

                    // Unroll 16 lanes, interleaving dequantization and FMA.
                    for offset in 0..<16 {
                        let idx = baseIdx + offset
                        let qFP32 = convertToFP32_NEON(int8Vec: qINT8[idx], scale: scale, zeroPoint: zeroPoint)
                        let diff = qFP32 - cFP32[idx]

                        // Distribute across 4 accumulators
                        switch offset % 4 {
                        case 0: s0.addProduct(diff, diff)
                        case 1: s1.addProduct(diff, diff)
                        case 2: s2.addProduct(diff, diff)
                        case 3: s3.addProduct(diff, diff)
                        default: break
                        }
                    }
                }

                // Handle remainder lanes
                if remainder > 0 {
                    let baseIdx = fullBlocks * 16
                    for j in 0..<remainder {
                        let idx = baseIdx + j
                        let qFP32 = convertToFP32_NEON(int8Vec: qINT8[idx], scale: scale, zeroPoint: zeroPoint)
                        let diff = qFP32 - cFP32[idx]

                        switch j % 4 {
                        case 0: s0.addProduct(diff, diff)
                        case 1: s1.addProduct(diff, diff)
                        case 2: s2.addProduct(diff, diff)
                        case 3: s3.addProduct(diff, diff)
                        default: break
                        }
                    }
                }
            }
        }

        // Horizontal reduction.
        return sqrt((s0 + s1 + s2 + s3).sum())
    }

    // Public API (512 specialization)
    @inlinable
    public static func euclidean512(query: Vector512INT8, candidate: Vector512INT8) -> Float {
        return euclidean_generic_int8(query: query, candidate: candidate)
    }

    @inlinable
    public static func euclidean512(query: Vector512INT8, candidate: Vector512Optimized) -> Float {
        return euclidean_generic_mixed(query: query, candidate: candidate)
    }

    @inlinable
    public static func euclidean512(query: Vector512Optimized, candidate: Vector512INT8) -> Float {
        return euclidean_generic_mixed(query: candidate, candidate: query)
    }

    // 768-dimensional specializations
    @inlinable
    public static func euclidean768(query: Vector768INT8, candidate: Vector768INT8) -> Float {
        return euclidean_generic_int8(query: query, candidate: candidate)
    }

    @inlinable
    public static func euclidean768(query: Vector768INT8, candidate: Vector768Optimized) -> Float {
        return euclidean_generic_mixed(query: query, candidate: candidate)
    }

    @inlinable
    public static func euclidean768(query: Vector768Optimized, candidate: Vector768INT8) -> Float {
        return euclidean_generic_mixed(query: candidate, candidate: query)
    }

    // 1536-dimensional specializations
    @inlinable
    public static func euclidean1536(query: Vector1536INT8, candidate: Vector1536INT8) -> Float {
        return euclidean_generic_int8(query: query, candidate: candidate)
    }

    @inlinable
    public static func euclidean1536(query: Vector1536INT8, candidate: Vector1536Optimized) -> Float {
        return euclidean_generic_mixed(query: query, candidate: candidate)
    }

    @inlinable
    public static func euclidean1536(query: Vector1536Optimized, candidate: Vector1536INT8) -> Float {
        return euclidean_generic_mixed(query: candidate, candidate: query)
    }
}

// MARK: - Cosine and Dot Product Kernels

extension QuantizedKernels {

    // MARK: Fused INT8 Kernel (Dot + Magnitudes)

    // Helper struct for fused accumulation results.
    @usableFromInline internal struct FusedAccumulators {
        @usableFromInline var dotProduct: Int64 = 0
        @usableFromInline var queryMagnitudeSquared: Int64 = 0
        @usableFromInline var candidateMagnitudeSquared: Int64 = 0

        @usableFromInline init() {}
    }

    // Generic implementation for fused Dot Product and Magnitudes.
    @inlinable
    internal static func accumulate_fused_generic<V: QuantizedVectorINT8>(
        query: V,
        candidate: V
    ) -> FusedAccumulators? {

        // This optimized integer path requires symmetric quantization (ZeroPoint=0).
        guard query.quantizationParams.isSymmetric && candidate.quantizationParams.isSymmetric else {
            return nil // Cannot use optimized path for asymmetric quantization.
        }

        var acc = FusedAccumulators()
        let laneCount = V.laneCount

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidate.storage.withUnsafeBufferPointer { candidatePtr in
                let qINT8 = queryPtr.baseAddress!; let cINT8 = candidatePtr.baseAddress!

                // Fused loop. Rely on compiler optimization for ILP.
                for i in 0..<laneCount {
                    // Promote to Int16 for intermediate multiplication.
                    // Sign-extend Int8 to Int16 for proper arithmetic
                    let qInt8 = qINT8[i]
                    let cInt8 = cINT8[i]
                    let q16 = SIMD4<Int16>(Int16(qInt8.x), Int16(qInt8.y), Int16(qInt8.z), Int16(qInt8.w))
                    let c16 = SIMD4<Int16>(Int16(cInt8.x), Int16(cInt8.y), Int16(cInt8.z), Int16(cInt8.w))

                    // Calculate products and sums. Promote to Int32 for intermediate sums.
                    // Use wrapping arithmetic (&+, &*).
                    let dot = Int32(q16.x &* c16.x) &+ Int32(q16.y &* c16.y) &+ Int32(q16.z &* c16.z) &+ Int32(q16.w &* c16.w)
                    let qMagSq = Int32(q16.x &* q16.x) &+ Int32(q16.y &* q16.y) &+ Int32(q16.z &* q16.z) &+ Int32(q16.w &* q16.w)
                    let cMagSq = Int32(c16.x &* c16.x) &+ Int32(c16.y &* c16.y) &+ Int32(c16.z &* c16.z) &+ Int32(c16.w &* c16.w)

                    // Accumulate into Int64.
                    acc.dotProduct &+= Int64(dot)
                    acc.queryMagnitudeSquared &+= Int64(qMagSq)
                    acc.candidateMagnitudeSquared &+= Int64(cMagSq)
                }
            }
        }
        return acc
    }

    @inlinable
    public static func cosine512(query: Vector512INT8, candidate: Vector512INT8) -> Float {
        guard let accumulators = accumulate_fused_generic(query: query, candidate: candidate) else {
            // Fallback for asymmetric quantization (Dequantize-then-Compute).
            return query.toFP32().cosineDistance(to: candidate.toFP32())
        }

        // Final scaling and calculation in FP32.
        let qScale = query.quantizationParams.scale
        let cScale = candidate.quantizationParams.scale

        // Dot(Q_f, C_f) = Dot(Q_i, C_i) * S_q * S_c
        let dotProductFP32 = Float(accumulators.dotProduct) * (qScale * cScale)

        // Mag(V_f)^2 = Mag(V_i)^2 * S_v^2
        let queryMagSqFP32 = Float(accumulators.queryMagnitudeSquared) * (qScale * qScale)
        let candidateMagSqFP32 = Float(accumulators.candidateMagnitudeSquared) * (cScale * cScale)

        let magnitudeProduct = sqrt(queryMagSqFP32 * candidateMagSqFP32)

        // Handle division by zero.
        if magnitudeProduct <= Float.leastNormalMagnitude {
            return 1.0 // Distance is 1 (Similarity 0) if either vector is zero.
        }

        let similarity = dotProductFP32 / magnitudeProduct
        // Clamp for numerical stability and return distance.
        return 1.0 - max(-1.0, min(1.0, similarity))
    }

    @inlinable
    public static func dotProduct512(query: Vector512INT8, candidate: Vector512INT8) -> Float {
        // Reuse the fused kernel logic.
        guard let accumulators = accumulate_fused_generic(query: query, candidate: candidate) else {
            return query.toFP32().dotProduct(candidate.toFP32())
        }

        let scaleProduct = query.quantizationParams.scale * candidate.quantizationParams.scale
        return Float(accumulators.dotProduct) * scaleProduct
    }

    // 768-dimensional cosine and dot product
    @inlinable
    public static func cosine768(query: Vector768INT8, candidate: Vector768INT8) -> Float {
        guard let accumulators = accumulate_fused_generic(query: query, candidate: candidate) else {
            return query.toFP32().cosineDistance(to: candidate.toFP32())
        }

        let qScale = query.quantizationParams.scale
        let cScale = candidate.quantizationParams.scale
        let dotProductFP32 = Float(accumulators.dotProduct) * (qScale * cScale)
        let queryMagSqFP32 = Float(accumulators.queryMagnitudeSquared) * (qScale * qScale)
        let candidateMagSqFP32 = Float(accumulators.candidateMagnitudeSquared) * (cScale * cScale)
        let magnitudeProduct = sqrt(queryMagSqFP32 * candidateMagSqFP32)

        if magnitudeProduct <= Float.leastNormalMagnitude {
            return 1.0
        }

        let similarity = dotProductFP32 / magnitudeProduct
        return 1.0 - max(-1.0, min(1.0, similarity))
    }

    @inlinable
    public static func dotProduct768(query: Vector768INT8, candidate: Vector768INT8) -> Float {
        guard let accumulators = accumulate_fused_generic(query: query, candidate: candidate) else {
            return query.toFP32().dotProduct(candidate.toFP32())
        }

        let scaleProduct = query.quantizationParams.scale * candidate.quantizationParams.scale
        return Float(accumulators.dotProduct) * scaleProduct
    }

    // 1536-dimensional cosine and dot product
    @inlinable
    public static func cosine1536(query: Vector1536INT8, candidate: Vector1536INT8) -> Float {
        guard let accumulators = accumulate_fused_generic(query: query, candidate: candidate) else {
            return query.toFP32().cosineDistance(to: candidate.toFP32())
        }

        let qScale = query.quantizationParams.scale
        let cScale = candidate.quantizationParams.scale
        let dotProductFP32 = Float(accumulators.dotProduct) * (qScale * cScale)
        let queryMagSqFP32 = Float(accumulators.queryMagnitudeSquared) * (qScale * qScale)
        let candidateMagSqFP32 = Float(accumulators.candidateMagnitudeSquared) * (cScale * cScale)
        let magnitudeProduct = sqrt(queryMagSqFP32 * candidateMagSqFP32)

        if magnitudeProduct <= Float.leastNormalMagnitude {
            return 1.0
        }

        let similarity = dotProductFP32 / magnitudeProduct
        return 1.0 - max(-1.0, min(1.0, similarity))
    }

    @inlinable
    public static func dotProduct1536(query: Vector1536INT8, candidate: Vector1536INT8) -> Float {
        guard let accumulators = accumulate_fused_generic(query: query, candidate: candidate) else {
            return query.toFP32().dotProduct(candidate.toFP32())
        }

        let scaleProduct = query.quantizationParams.scale * candidate.quantizationParams.scale
        return Float(accumulators.dotProduct) * scaleProduct
    }
}

// MARK: - Calibration and Error Analysis

@usableFromInline
internal enum QuantizationStrategy {
    case symmetric, asymmetric, perChannel
}

internal struct QuantizationCalibrator {
    public static func calibrate<V: OptimizedVector>(
        vectors: [V], strategy: QuantizationStrategy = .symmetric
    ) -> LinearQuantizationParams {
        guard !vectors.isEmpty else {
            return LinearQuantizationParams(minValue: -1.0, maxValue: 1.0, symmetric: true)
        }

        var globalMin: Float = .greatestFiniteMagnitude
        var globalMax: Float = -.greatestFiniteMagnitude

        for vector in vectors {
            vector.storage.withUnsafeBufferPointer { ptr in
                for simd in ptr {
                    globalMin = min(globalMin, simd.min())
                    globalMax = max(globalMax, simd.max())
                }
            }
        }

        return LinearQuantizationParams(
            minValue: globalMin,
            maxValue: globalMax,
            symmetric: strategy == .symmetric
        )
    }

    public static func calibrateWithDistribution<V: OptimizedVector>(
        vectors: [V],
        percentile: Float = 99.9
    ) -> LinearQuantizationParams {
        var allValues: [Float] = []

        for vector in vectors {
            vector.storage.withUnsafeBufferPointer { ptr in
                for simd in ptr {
                    allValues.append(contentsOf: [simd.x, simd.y, simd.z, simd.w])
                }
            }
        }

        guard !allValues.isEmpty else {
            return LinearQuantizationParams(minValue: -1.0, maxValue: 1.0, symmetric: true)
        }

        allValues.sort()
        let clipIndex = Int(Float(allValues.count) * percentile / 100.0)
        let safeIndex = min(max(clipIndex, 1), allValues.count - 1)

        let minValue = allValues[allValues.count - safeIndex]
        let maxValue = allValues[safeIndex - 1]

        return LinearQuantizationParams(minValue: minValue, maxValue: maxValue, symmetric: true)
    }
}

internal struct QuantizationError {
    public let meanAbsoluteError: Float
    public let maxAbsoluteError: Float
    public let rootMeanSquareError: Float
}

internal struct QuantizationErrorAnalyzer {
    public static func analyzeError<V: QuantizedVectorINT8>(
        original: V.FloatVectorType,
        quantized: V
    ) -> QuantizationError {
        let restored = quantized.toFP32()
        var errors: [Float] = []

        for i in 0..<original.storage.count {
            let origSIMD = original.storage[i]
            let restSIMD = restored.storage[i]

            let error = [
                abs(origSIMD.x - restSIMD.x),
                abs(origSIMD.y - restSIMD.y),
                abs(origSIMD.z - restSIMD.z),
                abs(origSIMD.w - restSIMD.w)
            ]

            errors.append(contentsOf: error)
        }

        let mae = errors.reduce(0, +) / Float(errors.count)
        let maxError = errors.max() ?? 0
        let rmse = sqrt(errors.map { $0 * $0 }.reduce(0, +) / Float(errors.count))

        return QuantizationError(meanAbsoluteError: mae, maxAbsoluteError: maxError, rootMeanSquareError: rmse)
    }
}

// MARK: - Integration Points (Protocol Conformance)

extension Vector512INT8 {
    public func euclideanDistance(to other: Self) -> Float {
        return QuantizedKernels.euclidean512(query: self, candidate: other)
    }

    public func cosineDistance(to other: Self) -> Float {
        return QuantizedKernels.cosine512(query: self, candidate: other)
    }

    public func dotProduct(_ other: Self) -> Float {
        return QuantizedKernels.dotProduct512(query: self, candidate: other)
    }

    public func validateQuantization() -> Bool {
        return quantizationParams.scale > 0
    }
}

extension Vector768INT8 {
    public func euclideanDistance(to other: Self) -> Float {
        return QuantizedKernels.euclidean768(query: self, candidate: other)
    }

    public func cosineDistance(to other: Self) -> Float {
        return QuantizedKernels.cosine768(query: self, candidate: other)
    }

    public func dotProduct(_ other: Self) -> Float {
        return QuantizedKernels.dotProduct768(query: self, candidate: other)
    }

    public func validateQuantization() -> Bool {
        return quantizationParams.scale > 0
    }
}

extension Vector1536INT8 {
    public func euclideanDistance(to other: Self) -> Float {
        return QuantizedKernels.euclidean1536(query: self, candidate: other)
    }

    public func cosineDistance(to other: Self) -> Float {
        return QuantizedKernels.cosine1536(query: self, candidate: other)
    }

    public func dotProduct(_ other: Self) -> Float {
        return QuantizedKernels.dotProduct1536(query: self, candidate: other)
    }

    public func validateQuantization() -> Bool {
        return quantizationParams.scale > 0
    }
}

// MARK: - Phase 2: SoA Quantized Types (Pure SoA Layout)

/// Structure-of-Arrays (SoA) storage for INT8 quantized vectors.
/// Pure SoA Layout: Organized by dimension index for efficient batch computation.
/// Storage structure:
/// Dim 0: [V0_d0, V1_d0, V2_d0, V3_d0], [V4_d0, V5_d0, V6_d0, V7_d0], ...
/// Dim 1: [V0_d1, V1_d1, V2_d1, V3_d1], [V4_d1, V5_d1, V6_d1, V7_d1], ...
/// ...
public struct SoAINT8<VectorType: OptimizedVector>: Sendable {
    public let dimension: Int
    /// Storage layout: D * ceil(N/4) entries of SIMD4<Int8>.
    public let storage: ContiguousArray<SIMD4<Int8>>
    public let vectorCount: Int
    public let quantizationParams: LinearQuantizationParams

    /// Number of SIMD4 groups needed (ceil(N/4)).
    public let simdGroups: Int

    public init(from vectors: [VectorType], params: LinearQuantizationParams? = nil) {
        self.dimension = vectors.first?.scalarCount ?? 0
        self.vectorCount = vectors.count

        guard !vectors.isEmpty else {
            self.storage = []
            self.simdGroups = 0
            self.quantizationParams = params ?? LinearQuantizationParams(minValue: -1, maxValue: 1, symmetric: true)
            return
        }

        // Determine parameters (calibrate if not provided).
        let finalParams = params ?? Self.calibrateParams(vectors: vectors)
        self.quantizationParams = finalParams

        self.simdGroups = (vectorCount + 3) / 4

        // Initialize storage and perform transposition.
        self.storage = Self.transposeAndQuantize(vectors: vectors, params: finalParams, dimension: self.dimension, simdGroups: self.simdGroups)
    }

    /// Fallback calibration when QuantizationCalibrator is not available
    public static func calibrateParams(vectors: [VectorType]) -> LinearQuantizationParams {
        var globalMin: Float = Float.infinity
        var globalMax: Float = -Float.infinity

        for vector in vectors {
            vector.withUnsafeBufferPointer { buffer in
                for element in buffer {
                    globalMin = min(globalMin, element)
                    globalMax = max(globalMax, element)
                }
            }
        }

        return LinearQuantizationParams(minValue: globalMin, maxValue: globalMax, symmetric: true)
    }

    /// Transposes AoS FP32 vectors to SoA INT8 storage.
    private static func transposeAndQuantize(vectors: [VectorType], params: LinearQuantizationParams, dimension: Int, simdGroups: Int) -> ContiguousArray<SIMD4<Int8>> {

        let capacity = dimension * simdGroups
        var soaStorage = ContiguousArray<SIMD4<Int8>>(repeating: .zero, count: capacity)

        let invScale = 1.0 / params.scale
        let offset = Float(params.zeroPoint)

        // Iterate over dimensions.
        for d in 0..<dimension {
            // Calculate indices for AoS access.
            let laneIndex = d / 4
            let elementIndex = d % 4

            // Iterate over groups of 4 vectors.
            for g in 0..<simdGroups {
                let vStart = g * 4

                // Gather the d-th element from 4 vectors (AoS -> SoA transposition).
                // Handle boundary conditions (tail cases) by padding with 0.0.
                let f0 = vectors.indices.contains(vStart) ? vectors[vStart].storage[laneIndex][elementIndex] : 0.0
                let f1 = vectors.indices.contains(vStart+1) ? vectors[vStart+1].storage[laneIndex][elementIndex] : 0.0
                let f2 = vectors.indices.contains(vStart+2) ? vectors[vStart+2].storage[laneIndex][elementIndex] : 0.0
                let f3 = vectors.indices.contains(vStart+3) ? vectors[vStart+3].storage[laneIndex][elementIndex] : 0.0

                let fVec = SIMD4<Float>(f0, f1, f2, f3)

                // Quantize the 4 elements simultaneously.
                let qVec = quantizeSIMD4(fVec, invScale: invScale, offset: offset)

                // Store in SoA layout: [Dim 0 Groups..., Dim 1 Groups..., ...]
                soaStorage[d * simdGroups + g] = qVec
            }
        }
        return soaStorage
    }

    @inline(__always)
    private static func quantizeSIMD4(_ input: SIMD4<Float>, invScale: Float, offset: Float) -> SIMD4<Int8> {
        // Vectorized quantization helper (Adapted from Part 1).
        let vInvScale = SIMD4<Float>(repeating: invScale)
        let vOffset = SIMD4<Float>(repeating: offset)
        let vMin = SIMD4<Float>(repeating: -128.0)
        let vMax = SIMD4<Float>(repeating: 127.0)

        var vQuantized = input * vInvScale + vOffset
        vQuantized = max(vMin, min(vMax, vQuantized.rounded(.toNearestOrAwayFromZero)))

        let vQuantizedInt32 = SIMD4<Int32>(Int32(vQuantized.x), Int32(vQuantized.y), Int32(vQuantized.z), Int32(vQuantized.w))
        return SIMD4<Int8>(
            Int8(clamping: vQuantizedInt32.x),
            Int8(clamping: vQuantizedInt32.y),
            Int8(clamping: vQuantizedInt32.z),
            Int8(clamping: vQuantizedInt32.w)
        )
    }
}

// Specialized SoA Types
public typealias SoA512INT8 = SoAINT8<Vector512Optimized>
public typealias SoA768INT8 = SoAINT8<Vector768Optimized>
public typealias SoA1536INT8 = SoAINT8<Vector1536Optimized>

// MARK: - Phase 2: Batch Processing Kernels

extension QuantizedKernels {

    // MARK: Hybrid Precision Batch Euclidean (FP32 Query vs INT8 SoA Candidates)

    /// Generic implementation for FP32 Query vs INT8 SoA Candidates (Dequantize-then-Compute).
    @inlinable
    internal static func batchEuclidean_generic_mixed<V: OptimizedVector>(
        query: V, candidates: SoAINT8<V>, results: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(results.count >= candidates.vectorCount)

        let dimension = candidates.dimension
        let simdGroups = candidates.simdGroups
        let candidateParams = candidates.quantizationParams
        let scale = candidateParams.scale
        let zeroPoint = candidateParams.zeroPoint

        // Initialize accumulators for all candidates.
        var accumulators = ContiguousArray<Float>(repeating: 0.0, count: candidates.vectorCount)

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                let queryFP32 = queryPtr // AoS layout

                // Iterate over dimensions.
                for d in 0..<dimension {
                    let laneIndex = d / 4
                    let elementIndex = d % 4

                    // Load the query element for this dimension and broadcast it.
                    let q_d = queryFP32[laneIndex][elementIndex]
                    let qBroadcast = SIMD4<Float>(repeating: q_d)

                    // Iterate over groups of 4 candidates.
                    for g in 0..<simdGroups {
                        let storageIndex = d * simdGroups + g

                        // Load 4 candidate elements for this dimension (SoA layout).
                        let cINT8 = candPtr[storageIndex]

                        // Dequantize on-the-fly.
                        let cFP32 = convertToFP32_NEON(int8Vec: cINT8, scale: scale, zeroPoint: zeroPoint)

                        // Calculate differences: (Q_d - Ck_d)^2
                        let diffs = qBroadcast - cFP32
                        let squaredDiffs = diffs * diffs

                        // Accumulate results.
                        let baseVectorIndex = g * 4
                        accumulators.withUnsafeMutableBufferPointer { accPtr in
                            // Handle tail cases safely.
                            accPtr[baseVectorIndex] += squaredDiffs.x
                            if baseVectorIndex + 1 < candidates.vectorCount {
                                accPtr[baseVectorIndex + 1] += squaredDiffs.y
                            }
                            if baseVectorIndex + 2 < candidates.vectorCount {
                                accPtr[baseVectorIndex + 2] += squaredDiffs.z
                            }
                            if baseVectorIndex + 3 < candidates.vectorCount {
                                accPtr[baseVectorIndex + 3] += squaredDiffs.w
                            }
                        }
                    }
                }
            }
        }

        // Final step: Apply sqrt to all results.
        for i in 0..<candidates.vectorCount {
            results[i] = sqrt(accumulators[i])
        }
    }

    // Public API (512 specialization)
    @inlinable
    public static func batchEuclidean512(query: Vector512Optimized, candidates: SoA512INT8, results: UnsafeMutableBufferPointer<Float>) {
        batchEuclidean_generic_mixed(query: query, candidates: candidates, results: results)
    }

    // MARK: Quantized Batch Euclidean (INT8 Query vs INT8 SoA Candidates)

    /// Generic implementation for INT8 Query vs INT8 SoA Candidates.
    @inlinable
    internal static func batchEuclidean_generic_int8<Q: QuantizedVectorINT8, C: OptimizedVector>(
        query: Q, candidates: SoAINT8<C>, results: UnsafeMutableBufferPointer<Float>
    ) where Q.FloatVectorType == C {

        // Check optimization criteria (symmetric and matching scales).
        let qParams = query.quantizationParams
        let cParams = candidates.quantizationParams

        let isSymmetric = qParams.isSymmetric && cParams.isSymmetric
        let scaleTolerance = 1e-6 * qParams.scale
        let scalesMatch = abs(qParams.scale - cParams.scale) <= scaleTolerance

        guard isSymmetric && scalesMatch else {
            // Fallback to mixed precision if not optimized.
            batchEuclidean_generic_mixed(query: query.toFP32(), candidates: candidates, results: results)
            return
        }

        let dimension = candidates.dimension
        let simdGroups = candidates.simdGroups
        let scale = qParams.scale

        // Initialize accumulators (Int32 is sufficient for up to 1536 dimensions).
        var accumulators = ContiguousArray<Int32>(repeating: 0, count: candidates.vectorCount)

        // Flatten the INT8 query (AoS) for easy dimensional access.
        let queryFlat = Array(query.storage.flatMap { [$0.x, $0.y, $0.z, $0.w] })

        queryFlat.withUnsafeBufferPointer { queryFlatPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in

                // Iterate over dimensions.
                for d in 0..<dimension {
                    // Load the query element and broadcast (promoting to Int16).
                    let q_d = queryFlatPtr[d]
                    let qBroadcast = SIMD4<Int16>(repeating: Int16(q_d))

                    // Iterate over groups of 4 candidates.
                    for g in 0..<simdGroups {
                        let storageIndex = d * simdGroups + g

                        // Load 4 candidate elements (SoA layout) and promote to Int16.
                        let cINT8 = candPtr[storageIndex]
                        let c16 = SIMD4<Int16>(
                            Int16(cINT8.x),
                            Int16(cINT8.y),
                            Int16(cINT8.z),
                            Int16(cINT8.w)
                        )

                        // Calculate differences (Int16).
                        let diffs = qBroadcast &- c16

                        // Square and accumulate (Int16*Int16 -> Int32).
                        let baseVectorIndex = g * 4
                        accumulators.withUnsafeMutableBufferPointer { accPtr in
                            accPtr[baseVectorIndex] &+= Int32(diffs.x &* diffs.x)
                            if baseVectorIndex + 1 < candidates.vectorCount {
                                accPtr[baseVectorIndex + 1] &+= Int32(diffs.y &* diffs.y)
                            }
                            if baseVectorIndex + 2 < candidates.vectorCount {
                                accPtr[baseVectorIndex + 2] &+= Int32(diffs.z &* diffs.z)
                            }
                            if baseVectorIndex + 3 < candidates.vectorCount {
                                accPtr[baseVectorIndex + 3] &+= Int32(diffs.w &* diffs.w)
                            }
                        }
                    }
                }
            }
        }

        // Final step: Apply scaling and sqrt. D = S * sqrt(Sum(diff^2)).
        for i in 0..<candidates.vectorCount {
            results[i] = scale * sqrt(Float(accumulators[i]))
        }
    }

    // Public API (512 specialization)
    @inlinable
    public static func batchEuclidean512(query: Vector512INT8, candidates: SoA512INT8, results: UnsafeMutableBufferPointer<Float>) {
        batchEuclidean_generic_int8(query: query, candidates: candidates, results: results)
    }
}

// MARK: - Phase 2: Quality Analysis Framework

internal struct QuantizationQualityAnalyzer {
    public struct QualityMetrics: Sendable {
        public let meanAbsoluteError: Float
        public let rootMeanSquareError: Float
        public let maxAbsoluteError: Float
        public let signalToNoiseRatio: Float
    }

    public static func analyzeQuality<V_FP32: OptimizedVector, V_INT8: QuantizedVectorINT8>(
        originalVectors: [V_FP32],
        quantizedVectors: [V_INT8]
    ) -> QualityMetrics where V_INT8.FloatVectorType == V_FP32 {

        // Use Double for accumulation to maintain precision.
        var sumAbsError: Double = 0.0
        var sumSqError: Double = 0.0
        var maxAbsError: Float = 0.0
        var sumSqSignal: Double = 0.0
        var elementCount: Int = 0

        for (orig, quant) in zip(originalVectors, quantizedVectors) {
            let restored = quant.toFP32()
            elementCount += orig.scalarCount
            // Calculate errors element-wise since we can't access storage directly in generic context
            for j in 0..<orig.scalarCount {
                let origValue = orig[j]
                let restValue = restored[j]
                let diff = origValue - restValue
                let absDiff = abs(diff)
                let sqDiff = diff * diff
                let sqSignal = origValue * origValue

                sumAbsError += Double(absDiff)
                sumSqError += Double(sqDiff)
                sumSqSignal += Double(sqSignal)

                maxAbsError = max(maxAbsError, absDiff)
            }
        }

        guard elementCount > 0 else {
            return QualityMetrics(meanAbsoluteError: 0, rootMeanSquareError: 0, maxAbsoluteError: 0, signalToNoiseRatio: 0)
        }

        let countD = Double(elementCount)
        let mae = Float(sumAbsError / countD)
        let rmse = Float(sqrt(sumSqError / countD))

        // SNR = 10 * log10(P_signal / P_noise)
        let signalPower = sumSqSignal / countD
        let noisePower = sumSqError / countD

        let snr: Float
        if noisePower <= Double.leastNormalMagnitude {
            snr = Float.infinity
        } else {
            snr = Float(10.0 * log10(signalPower / noisePower))
        }

        return QualityMetrics(
            meanAbsoluteError: mae,
            rootMeanSquareError: rmse,
            maxAbsoluteError: maxAbsError,
            signalToNoiseRatio: snr
        )
    }
}

// MARK: - Phase 2: AutoTuning Integration Framework

public final class QuantizedAutoTuner: @unchecked Sendable {
    public static let shared = QuantizedAutoTuner()
    private var calibrationCache: [String: QuantizationStrategy] = [:]
    private let lock = NSLock() // Ensure thread safety.

    public enum QuantizationStrategy: Sendable {
        case fullFP32, candidatesINT8, bothINT8
    }

    private init() {} // Singleton

    /// Selects optimal quantization strategy based on operation characteristics
    public func selectOptimalQuantization(
        for operation: String,
        dimension: Int,
        candidateCount: Int,
        accuracyRequirement: Float = 0.02
    ) -> QuantizationStrategy {
        let cacheKey = "\(operation)_\(dimension)_\(candidateCount)"

        lock.lock()
        defer { lock.unlock() }

        if let cached = calibrationCache[cacheKey] {
            return cached
        }

        // Heuristic-based strategy selection
        let strategy: QuantizationStrategy
        if candidateCount > 1000 && accuracyRequirement < 0.05 {
            strategy = .candidatesINT8
        } else if candidateCount > 5000 && accuracyRequirement < 0.1 {
            strategy = .bothINT8
        } else {
            strategy = .fullFP32
        }

        calibrationCache[cacheKey] = strategy
        return strategy
    }

    /// Clears the calibration cache (useful for testing or memory management)
    public func clearCache() {
        lock.lock()
        defer { lock.unlock() }
        calibrationCache.removeAll()
    }
}

// MARK: - Phase 2: Advanced Quantization Techniques (Framework Stubs)

internal struct AdaptiveQuantizer {
    public enum AdaptiveStrategy {
        case distributionBased, errorBudgetBased
    }

    /// Placeholder for adaptive quantization - requires advanced statistical analysis
    public static func quantize(vectors: [Vector512Optimized], strategy: AdaptiveStrategy) -> ([Vector512INT8], LinearQuantizationParams) {
        // For now, use the basic symmetric quantization as fallback
        let params = SoAINT8.calibrateParams(vectors: vectors)
        let quantizedVectors = vectors.compactMap { Vector512INT8(from: $0, params: params) }
        return (quantizedVectors, params)
    }
}

// MARK: - Phase 2: Production Deployment Frameworks (Stubs)

internal struct ProductionQuantizationParams: Sendable {
    public let targetAccuracy: Float
    public let memoryBudget: Int
    public let performanceTarget: Double // operations per second

    public init(targetAccuracy: Float = 0.01, memoryBudget: Int = 1024 * 1024 * 100, performanceTarget: Double = 10000) {
        self.targetAccuracy = targetAccuracy
        self.memoryBudget = memoryBudget
        self.performanceTarget = performanceTarget
    }
}

internal struct QuantizationCalibrationPipeline {
    public static func calibrateForProduction(
        testVectors: [Vector512Optimized],
        params: ProductionQuantizationParams
    ) -> LinearQuantizationParams {
        // Placeholder for production calibration pipeline
        return SoAINT8.calibrateParams(vectors: testVectors)
    }
}

public actor QuantizationMonitor {
    private var accuracyMetrics: [String: Float] = [:]
    private var performanceMetrics: [String: Double] = [:]

    public func recordAccuracy(_ accuracy: Float, for operation: String) {
        accuracyMetrics[operation] = accuracy
    }

    public func recordPerformance(_ opsPerSec: Double, for operation: String) {
        performanceMetrics[operation] = opsPerSec
    }

    public func getMetrics() -> (accuracy: [String: Float], performance: [String: Double]) {
        return (accuracyMetrics, performanceMetrics)
    }
}
