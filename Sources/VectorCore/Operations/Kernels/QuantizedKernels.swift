//
//  QuantizedKernels.swift
//  VectorCore
//
//  INT8 quantized kernels
//  One byte per stored scalar, a quarter of the FP32 scalar payload (payload model only, not a measured memory
//  saving); the accuracy loss is not characterized in this file
//  Phase 1: Core algorithms and distance kernels with SIMD4<Int8> storage
//

import Foundation
import simd

// MARK: - Linear Quantization Parameters

/// Parameters for linear (affine) quantization: q = round(x/scale + zeroPoint).
public struct LinearQuantizationParams: Sendable, Equatable, Hashable, Codable {
    public let scale: Float
    /// Affine zero point. Legacy decoded tuples retain the full Int32 domain.
    public let zeroPoint: Int32
    public let minValue: Float
    public let maxValue: Float
    public let isSymmetric: Bool

    public init(minValue: Float, maxValue: Float, symmetric: Bool = true) {
        let calibrated = AffineQuantizationCore.calibrate(minimum: minValue, maximum: maxValue, symmetric: symmetric)
        self.minValue = minValue
        self.maxValue = maxValue
        self.isSymmetric = symmetric
        self.scale = calibrated.0
        self.zeroPoint = calibrated.1
    }

    @usableFromInline internal var isValid: Bool {
        scale.isFinite && scale > 0 && minValue.isFinite && maxValue.isFinite
            && minValue <= maxValue && (!isSymmetric || zeroPoint == 0)
    }

    private enum CodingKeys: String, CodingKey { case scale, zeroPoint, minValue, maxValue, isSymmetric }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        scale = try container.decode(Float.self, forKey: .scale)
        zeroPoint = try container.decode(Int32.self, forKey: .zeroPoint)
        minValue = try container.decode(Float.self, forKey: .minValue)
        maxValue = try container.decode(Float.self, forKey: .maxValue)
        isSymmetric = try container.decode(Bool.self, forKey: .isSymmetric)
        guard isValid else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "Quantization requires valid parameters"))
        }
    }

    @inlinable
    public func quantize(_ value: Float) -> Int8 {
        guard isValid else { fatalError("Quantization requires valid parameters") }
        guard value.isFinite else { fatalError("Quantization requires finite input coordinates") }
        let rounded = (Double(value) / Double(scale) + Double(zeroPoint)).rounded(.toNearestOrAwayFromZero)
        return Int8(max(-128, min(127, rounded)))
    }

    @inlinable
    public func dequantize(_ quantizedValue: Int8) -> Float {
        return Float(Double(scale) * (Double(quantizedValue) - Double(zeroPoint)))
    }
}

// MARK: - INT8 Storage Types

/// Protocol defining the structure and behavior of INT8 quantized vectors.
public protocol QuantizedVectorINT8: Sendable, VectorProtocol {
    associatedtype FloatVectorType: OptimizedVector

    /// Storage is `SIMD4<Int8>`: four Int8 values (4 bytes) per element.
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
        AffineQuantizationCore.requireFinite(vector.toArray())
        let finalParams: LinearQuantizationParams

        if let providedParams = params {
            finalParams = providedParams
        } else {
            // Auto-calibrate parameters by finding the min/max range of the input vector.
            // `storage.minMax()` (a SIMDStorage extension) finds the range.
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
        guard params.isValid else { fatalError("Quantization requires valid parameters") }
        return ContiguousArray((0..<Self.laneCount).map { index in
            let lane = vector.storage[index]
            return SIMD4(params.quantize(lane.x), params.quantize(lane.y), params.quantize(lane.z), params.quantize(lane.w))
        })
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
    private var _storage: ContiguousArray<SIMD4<Int8>>

    /// SIMD storage. Replacement must contain exactly 128 SIMD4 elements.
    /// Inout mutation must preserve this count, including when unwinding a throw.
    public var storage: ContiguousArray<SIMD4<Int8>> {
        get { _storage }
        set {
            guard newValue.count == 128 else { fatalError("Storage must contain exactly 128 SIMD4 elements") }
            _storage = newValue
        }
        _modify {
            defer {
                guard _storage.count == 128 else { fatalError("Storage must contain exactly 128 SIMD4 elements") }
            }
            yield &_storage
        }
    }
    public let quantizationParams: LinearQuantizationParams

    public static let laneCount: Int = 128  // 512 / 4

    /// - Precondition: Storage contains exactly 128 SIMD4 elements.
    public init(storage: ContiguousArray<SIMD4<Int8>>, params: LinearQuantizationParams) {
        guard storage.count == 128 else { fatalError("Storage must contain exactly 128 SIMD4 elements") }
        self._storage = storage
        self.quantizationParams = params
    }

    private enum CodingKeys: String, CodingKey {
        case storage, quantizationParams
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let storage = try container.decode(ContiguousArray<SIMD4<Int8>>.self, forKey: .storage)
        guard storage.count == 128 else {
            throw DecodingError.dataCorruptedError(
                forKey: .storage, in: container,
                debugDescription: "Storage must contain exactly 128 SIMD4 elements"
            )
        }
        let params = try container.decode(LinearQuantizationParams.self, forKey: .quantizationParams)
        self.init(storage: storage, params: params)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(_storage, forKey: .storage)
        try container.encode(quantizationParams, forKey: .quantizationParams)
    }
}

public struct Vector768INT8: QuantizedVectorINT8 {
    public typealias FloatVectorType = Vector768Optimized
    private var _storage: ContiguousArray<SIMD4<Int8>>

    /// SIMD storage. Replacement must contain exactly 192 SIMD4 elements.
    /// Inout mutation must preserve this count, including when unwinding a throw.
    public var storage: ContiguousArray<SIMD4<Int8>> {
        get { _storage }
        set {
            guard newValue.count == 192 else { fatalError("Storage must contain exactly 192 SIMD4 elements") }
            _storage = newValue
        }
        _modify {
            defer {
                guard _storage.count == 192 else { fatalError("Storage must contain exactly 192 SIMD4 elements") }
            }
            yield &_storage
        }
    }
    public let quantizationParams: LinearQuantizationParams

    public static let laneCount: Int = 192  // 768 / 4

    /// - Precondition: Storage contains exactly 192 SIMD4 elements.
    public init(storage: ContiguousArray<SIMD4<Int8>>, params: LinearQuantizationParams) {
        guard storage.count == 192 else { fatalError("Storage must contain exactly 192 SIMD4 elements") }
        self._storage = storage
        self.quantizationParams = params
    }

    private enum CodingKeys: String, CodingKey {
        case storage, quantizationParams
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let storage = try container.decode(ContiguousArray<SIMD4<Int8>>.self, forKey: .storage)
        guard storage.count == 192 else {
            throw DecodingError.dataCorruptedError(
                forKey: .storage, in: container,
                debugDescription: "Storage must contain exactly 192 SIMD4 elements"
            )
        }
        let params = try container.decode(LinearQuantizationParams.self, forKey: .quantizationParams)
        self.init(storage: storage, params: params)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(_storage, forKey: .storage)
        try container.encode(quantizationParams, forKey: .quantizationParams)
    }
}

public struct Vector1536INT8: QuantizedVectorINT8 {
    public typealias FloatVectorType = Vector1536Optimized
    private var _storage: ContiguousArray<SIMD4<Int8>>

    /// SIMD storage. Replacement must contain exactly 384 SIMD4 elements.
    /// Inout mutation must preserve this count, including when unwinding a throw.
    public var storage: ContiguousArray<SIMD4<Int8>> {
        get { _storage }
        set {
            guard newValue.count == 384 else { fatalError("Storage must contain exactly 384 SIMD4 elements") }
            _storage = newValue
        }
        _modify {
            defer {
                guard _storage.count == 384 else { fatalError("Storage must contain exactly 384 SIMD4 elements") }
            }
            yield &_storage
        }
    }
    public let quantizationParams: LinearQuantizationParams

    public static let laneCount: Int = 384  // 1536 / 4

    /// - Precondition: Storage contains exactly 384 SIMD4 elements.
    public init(storage: ContiguousArray<SIMD4<Int8>>, params: LinearQuantizationParams) {
        guard storage.count == 384 else { fatalError("Storage must contain exactly 384 SIMD4 elements") }
        self._storage = storage
        self.quantizationParams = params
    }

    private enum CodingKeys: String, CodingKey {
        case storage, quantizationParams
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let storage = try container.decode(ContiguousArray<SIMD4<Int8>>.self, forKey: .storage)
        guard storage.count == 384 else {
            throw DecodingError.dataCorruptedError(
                forKey: .storage, in: container,
                debugDescription: "Storage must contain exactly 384 SIMD4 elements"
            )
        }
        let params = try container.decode(LinearQuantizationParams.self, forKey: .quantizationParams)
        self.init(storage: storage, params: params)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(_storage, forKey: .storage)
        try container.encode(quantizationParams, forKey: .quantizationParams)
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

        let invalidIndices = array.indices.filter { !array[$0].isFinite }
        guard invalidIndices.isEmpty else {
            throw VectorError.invalidValues(indices: invalidIndices, reason: "Quantization requires finite input coordinates")
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
        guard value.isFinite else { fatalError("Quantization requires finite input coordinates") }
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

    /// A thrown body discards the temporary and preserves raw storage and parameters.
    /// Normal return validates finite output and recalibrates, including a no-op body.
    /// A legacy tuple with infinite Float projections therefore rejects on normal return.
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        // Convert to FP32, apply transformation, then quantize back
        var fp32Vector = toFP32()
        let result = try fp32Vector.withUnsafeMutableBufferPointer(body)
        let newArray = fp32Vector.toArray()
        AffineQuantizationCore.requireFinite(newArray)
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

        let invalidIndices = array.indices.filter { !array[$0].isFinite }
        guard invalidIndices.isEmpty else {
            throw VectorError.invalidValues(indices: invalidIndices, reason: "Quantization requires finite input coordinates")
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
        guard value.isFinite else { fatalError("Quantization requires finite input coordinates") }
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

    /// A thrown body discards the temporary and preserves raw storage and parameters.
    /// Normal return validates finite output and recalibrates, including a no-op body.
    /// A legacy tuple with infinite Float projections therefore rejects on normal return.
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        var fp32Vector = toFP32()
        let result = try fp32Vector.withUnsafeMutableBufferPointer(body)
        let newArray = fp32Vector.toArray()
        AffineQuantizationCore.requireFinite(newArray)
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

        let invalidIndices = array.indices.filter { !array[$0].isFinite }
        guard invalidIndices.isEmpty else {
            throw VectorError.invalidValues(indices: invalidIndices, reason: "Quantization requires finite input coordinates")
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
        guard value.isFinite else { fatalError("Quantization requires finite input coordinates") }
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

    /// A thrown body discards the temporary and preserves raw storage and parameters.
    /// Normal return validates finite output and recalibrates, including a no-op body.
    /// A legacy tuple with infinite Float projections therefore rejects on normal return.
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        var fp32Vector = toFP32()
        let result = try fp32Vector.withUnsafeMutableBufferPointer(body)
        let newArray = fp32Vector.toArray()
        AffineQuantizationCore.requireFinite(newArray)
        self = try! Vector1536INT8(newArray)
        return result
    }
}

// MARK: - Core Distance Kernels and Helpers

@usableFromInline
internal enum QuantizedKernels {

    // MARK: Conversion Helpers (NEON Optimization)

    /// Optimized conversion from INT8 to FP32, applying dequantization parameters.
    /// Each affine coordinate is rounded once to Float after Double evaluation.
    @inline(__always)
    @usableFromInline
    internal static func convertToFP32_NEON(
        int8Vec: SIMD4<Int8>,
        scale: Float,
        zeroPoint: Int32
    ) -> SIMD4<Float> {
        SIMD4<Float>(
            Float(Double(scale) * (Double(int8Vec.x) - Double(zeroPoint))),
            Float(Double(scale) * (Double(int8Vec.y) - Double(zeroPoint))),
            Float(Double(scale) * (Double(int8Vec.z) - Double(zeroPoint))),
            Float(Double(scale) * (Double(int8Vec.w) - Double(zeroPoint))))
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

        let scalesMatch = query.quantizationParams.scale == candidate.quantizationParams.scale
        if !isSymmetric || !scalesMatch || query.quantizationParams.zeroPoint != 0 || candidate.quantizationParams.zeroPoint != 0 {
            var sum: Double = 0
            for i in 0..<V.laneCount {
                for j in 0..<4 {
                    let difference = AffineQuantizationCore.difference(
                        AffineQuantizationCore.product(query.storage[i][j], query.quantizationParams),
                        AffineQuantizationCore.product(candidate.storage[i][j], candidate.quantizationParams))
                    sum += difference * difference
                }
            }
            return Float(sqrt(sum))
        }

        // Optimized path for identical, symmetric quantization parameters.
        let scale = query.quantizationParams.scale

        #if VC_USE_C_KERNELS
        // Use dot product decomposition: ||a-b||^2 = dot(a,a) - 2*dot(a,b) + dot(b,b)
        // Three calls to the hardware-accelerated dotInt8 share data in L1 cache.
        // Overflow-safe for well-formed quantization: max intermediate for 1536-d is ~99M,
        // within Int32 range. Clamp to 0 defensively — wrapping arithmetic could produce
        // negative values if quantization bounds are violated, and sqrt(negative) → NaN.
        var l2sq: Int32 = 0
        withSIMD4Int8Pointer(query.storage) { qPtr, qCount in
            withSIMD4Int8Pointer(candidate.storage) { cPtr, cCount in
                let dotQQ = CKernels.dotInt8(qPtr, qPtr, lanes: qCount)
                let dotQC = CKernels.dotInt8(qPtr, cPtr, lanes: qCount)
                let dotCC = CKernels.dotInt8(cPtr, cPtr, lanes: cCount)
                l2sq = dotQQ &- 2 &* dotQC &+ dotCC
            }
        }
        return Float(Double(scale) * sqrt(Double(max(0, l2sq))))
        #else
        let laneCount = V.laneCount

        // Use Int32 accumulators. Max accumulation for 1536 dimensions is safe within Int32 limits.
        // Max diff^2 = 255^2 = 65025. Total = 1536 * 65025 < 100 million. Int32 max ≈ 2.1 billion.
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

        // D = sqrt(S^2 * Sum(diff^2)) = S * sqrt(Sum(diff^2))
        return Float(Double(scale) * sqrt(Double(totalAccumulator)))
        #endif
    }

    /// Helper to compute (q-c)^2 and accumulate, promoting to Int16 intermediates.
    @inline(__always)
    @usableFromInline internal static func accumulateEuclidDiffSq(_ q: SIMD4<Int8>, _ c: SIMD4<Int8>, _ acc: inout Int32) {
        // Promote to Int16 before subtraction. Use wrapping subtraction (&-).
        // Sign-extend Int8 to Int16 for proper arithmetic
        let qInt16 = SIMD4<Int16>(Int16(q.x), Int16(q.y), Int16(q.z), Int16(q.w))
        let cInt16 = SIMD4<Int16>(Int16(c.x), Int16(c.y), Int16(c.z), Int16(c.w))
        let diff = qInt16 &- cInt16
        // diff ∈ [-255, 255]; diff² can reach 65025 which overflows Int16 [-32768, 32767].
        // Widen to Int32 BEFORE squaring so the square is computed in 32-bit space.
        let d = SIMD4<Int32>(truncatingIfNeeded: diff)
        let sq = d &* d
        acc &+= sq.x &+ sq.y &+ sq.z &+ sq.w
    }

    // MARK: Mixed Precision Euclidean (INT8 vs FP32)

    /// Generic implementation for mixed precision Euclidean distance (Dequantize-then-Compute).
    @inlinable
    internal static func euclidean_generic_mixed<V_INT8: QuantizedVectorINT8, V_FP32: OptimizedVector>(
        query: V_INT8, candidate: V_FP32
    ) -> Float where V_INT8.FloatVectorType == V_FP32 {
        var sum: Double = 0
        for i in 0..<V_INT8.laneCount {
            for j in 0..<4 {
                let difference = AffineQuantizationCore.difference(query.storage[i][j], query.quantizationParams, candidate.storage[i][j])
                sum += difference * difference
            }
        }
        return Float(sqrt(sum))
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
        guard query.quantizationParams.isValid && candidate.quantizationParams.isValid && query.quantizationParams.isSymmetric && candidate.quantizationParams.isSymmetric
            && query.quantizationParams.zeroPoint == 0 && candidate.quantizationParams.zeroPoint == 0 else {
            return nil // Cannot use optimized path for asymmetric quantization.
        }

        var acc = FusedAccumulators()

        #if VC_USE_C_KERNELS
        // Route through hardware-accelerated dot product.
        // dot(q,c), dot(q,q), and dot(c,c) are independent calls sharing L1 cache.
        // Int32 -> Int64 promotion is lossless (max 1536-d value ~24.7M).
        withSIMD4Int8Pointer(query.storage) { qPtr, qCount in
            withSIMD4Int8Pointer(candidate.storage) { cPtr, cCount in
                acc.dotProduct = Int64(CKernels.dotInt8(qPtr, cPtr, lanes: qCount))
                acc.queryMagnitudeSquared = Int64(CKernels.dotInt8(qPtr, qPtr, lanes: qCount))
                acc.candidateMagnitudeSquared = Int64(CKernels.dotInt8(cPtr, cPtr, lanes: cCount))
            }
        }
        #else
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
        #endif
        return acc
    }

    @usableFromInline
    internal static func centeredSums<V: QuantizedVectorINT8>(query: V, candidate: V) -> (Double, Double, Double) {
        if let fast = accumulate_fused_generic(query: query, candidate: candidate) {
            return (Double(fast.dotProduct), Double(fast.queryMagnitudeSquared), Double(fast.candidateMagnitudeSquared))
        }
        var dot = AffineWideInteger(), querySquare = AffineWideInteger(), candidateSquare = AffineWideInteger()
        for i in 0..<V.laneCount {
            for j in 0..<4 {
                let q = Int64(query.storage[i][j]) - Int64(query.quantizationParams.zeroPoint)
                let c = Int64(candidate.storage[i][j]) - Int64(candidate.quantizationParams.zeroPoint)
                dot.add(q * c); querySquare.add(q * q); candidateSquare.add(c * c)
            }
        }
        return (dot.doubleValue, querySquare.doubleValue, candidateSquare.doubleValue)
    }

    @usableFromInline
    internal static func cosine_generic<V: QuantizedVectorINT8>(query: V, candidate: V) -> Float {
        let sums = centeredSums(query: query, candidate: candidate)
        if sums.1 == 0 || sums.2 == 0 { return 1 }
        let similarity = sums.0 / (sqrt(sums.1) * sqrt(sums.2))
        // Unreachable (centeredSums returns integer sums, both nonzero past the check above); Float.nan keeps the T15 policy.
        guard similarity.isFinite else { return .nan }
        return Float(1 - max(-1, min(1, similarity)))
    }

    @usableFromInline
    internal static func dot_generic<V: QuantizedVectorINT8>(query: V, candidate: V) -> Float {
        let dot = centeredSums(query: query, candidate: candidate).0
        return Float(dot * (Double(query.quantizationParams.scale) * Double(candidate.quantizationParams.scale)))
    }

    @inlinable
    public static func cosine512(query: Vector512INT8, candidate: Vector512INT8) -> Float {
        cosine_generic(query: query, candidate: candidate)
    }

    @inlinable
    public static func dotProduct512(query: Vector512INT8, candidate: Vector512INT8) -> Float {
        dot_generic(query: query, candidate: candidate)
    }

    // 768-dimensional cosine and dot product
    @inlinable
    public static func cosine768(query: Vector768INT8, candidate: Vector768INT8) -> Float {
        cosine_generic(query: query, candidate: candidate)
    }

    @inlinable
    public static func dotProduct768(query: Vector768INT8, candidate: Vector768INT8) -> Float {
        dot_generic(query: query, candidate: candidate)
    }

    // 1536-dimensional cosine and dot product
    @inlinable
    public static func cosine1536(query: Vector1536INT8, candidate: Vector1536INT8) -> Float {
        cosine_generic(query: query, candidate: candidate)
    }

    @inlinable
    public static func dotProduct1536(query: Vector1536INT8, candidate: Vector1536INT8) -> Float {
        dot_generic(query: query, candidate: candidate)
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
        guard strategy != .perChannel else { fatalError("Quantization perChannel calibration is unsupported") }
        var minimum = Float.infinity
        var maximum = -Float.infinity
        var hasValues = false
        for vector in vectors {
            vector.storage.withUnsafeBufferPointer { buffer in
                for lane in buffer {
                    for index in 0..<4 {
                        let value = lane[index]
                        guard value.isFinite else { fatalError("Quantization requires finite input coordinates") }
                        if !hasValues {
                            minimum = value
                            maximum = value
                            hasValues = true
                        } else {
                            if value < minimum { minimum = value }
                            if value > maximum { maximum = value }
                        }
                    }
                }
            }
        }
        return LinearQuantizationParams(minValue: hasValues ? minimum : 0,
                                        maxValue: hasValues ? maximum : 0,
                                        symmetric: strategy == .symmetric)
    }

    public static func calibrateWithDistribution<V: OptimizedVector>(
        vectors: [V],
        percentile: Float = 99.9
    ) -> LinearQuantizationParams {
        guard percentile.isFinite && percentile > 0 && percentile <= 100 else { fatalError("Quantization percentile must be finite and in (0,100]") }
        let values = vectors.flatMap { $0.toArray() }
        AffineQuantizationCore.requireFinite(values)
        guard !values.isEmpty else { return LinearQuantizationParams(minValue: 0, maxValue: 0) }
        let magnitudes = values.map { abs($0) }.sorted()
        let index = AffineQuantizationCore.percentileIndex(count: magnitudes.count, percentile: percentile)
        let threshold = magnitudes[index]
        return LinearQuantizationParams(minValue: -threshold, maxValue: threshold)
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
        return quantizationParams.isValid
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
        return quantizationParams.isValid
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
        return quantizationParams.isValid
    }
}

// MARK: - Phase 2: SoA Quantized Types (Pure SoA Layout)

/// Structure-of-Arrays (SoA) storage for INT8 quantized vectors.
/// Pure SoA Layout: organized by dimension index, then by groups of four vectors.
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
        if let params { guard params.isValid else { fatalError("Quantization requires valid parameters") } }
        for vector in vectors { AffineQuantizationCore.requireFinite(vector.toArray()) }
        self.dimension = vectors.first?.scalarCount ?? 0
        self.vectorCount = vectors.count

        guard !vectors.isEmpty else {
            self.storage = []
            self.simdGroups = 0
            self.quantizationParams = params ?? LinearQuantizationParams(minValue: 0, maxValue: 0, symmetric: true)
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
        QuantizationCalibrator.calibrate(vectors: vectors)
    }

    /// Transposes AoS FP32 vectors to SoA INT8 storage.
    private static func transposeAndQuantize(vectors: [VectorType], params: LinearQuantizationParams, dimension: Int, simdGroups: Int) -> ContiguousArray<SIMD4<Int8>> {

        let capacity = dimension * simdGroups
        var soaStorage = ContiguousArray<SIMD4<Int8>>(repeating: .zero, count: capacity)


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
                let qVec = quantizeSIMD4(fVec, params: params)

                // Store in SoA layout: [Dim 0 Groups..., Dim 1 Groups..., ...]
                soaStorage[d * simdGroups + g] = qVec
            }
        }
        return soaStorage
    }

    @inline(__always)
    private static func quantizeSIMD4(_ input: SIMD4<Float>, params: LinearQuantizationParams) -> SIMD4<Int8> {
        SIMD4(params.quantize(input.x), params.quantize(input.y), params.quantize(input.z), params.quantize(input.w))
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
        guard results.count >= candidates.vectorCount else { fatalError("Quantized batch results capacity must cover vectorCount") }
        for k in 0..<candidates.vectorCount {
            var sum: Double = 0
            for d in 0..<candidates.dimension {
                let code = candidates.storage[d * candidates.simdGroups + k / 4][k % 4]
                let difference = AffineQuantizationCore.difference(code, candidates.quantizationParams, query.storage[d / 4][d % 4])
                sum += difference * difference
            }
            results[k] = Float(sqrt(sum))
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

        guard results.count >= candidates.vectorCount else { fatalError("Quantized batch results capacity must cover vectorCount") }
        let qParams = query.quantizationParams
        let cParams = candidates.quantizationParams
        guard qParams.isSymmetric && cParams.isSymmetric && qParams.zeroPoint == 0 && cParams.zeroPoint == 0 && qParams.scale == cParams.scale else {
            for k in 0..<candidates.vectorCount {
                var sum: Double = 0
                for d in 0..<candidates.dimension {
                    let code = candidates.storage[d * candidates.simdGroups + k / 4][k % 4]
                    let difference = AffineQuantizationCore.difference(
                        AffineQuantizationCore.product(query.storage[d / 4][d % 4], qParams),
                        AffineQuantizationCore.product(code, cParams))
                    sum += difference * difference
                }
                results[k] = Float(sqrt(sum))
            }
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

                        // diffs ∈ [-255, 255]; diffs² can reach 65025 which overflows Int16.
                        // Widen to Int32 BEFORE squaring to compute the square in 32-bit space.
                        let d32 = SIMD4<Int32>(truncatingIfNeeded: diffs)
                        let sq = d32 &* d32

                        // Square and accumulate (Int32).
                        let baseVectorIndex = g * 4
                        accumulators.withUnsafeMutableBufferPointer { accPtr in
                            accPtr[baseVectorIndex] &+= sq.x
                            if baseVectorIndex + 1 < candidates.vectorCount {
                                accPtr[baseVectorIndex + 1] &+= sq.y
                            }
                            if baseVectorIndex + 2 < candidates.vectorCount {
                                accPtr[baseVectorIndex + 2] &+= sq.z
                            }
                            if baseVectorIndex + 3 < candidates.vectorCount {
                                accPtr[baseVectorIndex + 3] &+= sq.w
                            }
                        }
                    }
                }
            }
        }

        // Final step: Apply scaling and sqrt. D = S * sqrt(Sum(diff^2)).
        for i in 0..<candidates.vectorCount {
            results[i] = Float(Double(scale) * sqrt(Double(accumulators[i])))
        }
    }

    // Public API (512 specialization)
    @inlinable
    public static func batchEuclidean512(query: Vector512INT8, candidates: SoA512INT8, results: UnsafeMutableBufferPointer<Float>) {
        batchEuclidean_generic_int8(query: query, candidates: candidates, results: results)
    }
}

// MARK: - Phase 2: AutoTuning Integration Framework

/// A heuristic suggestion cache. The hint is not a measured error guarantee.
/// All mutable cache state is protected by `lock`.
public final class QuantizedAutoTuner: @unchecked Sendable {
    public static let shared = QuantizedAutoTuner()
    private struct Request: Hashable {
        let operation: String
        let dimension: Int
        let candidateCount: Int
        let hintBits: UInt32
    }
    private var calibrationCache: [Request: QuantizationStrategy] = [:]
    private let lock = NSLock()

    public enum QuantizationStrategy: Sendable {
        case fullFP32, candidatesINT8, bothINT8
    }

    public init() {}

    /// Suggests a representation using count and a nonnegative relative-error hint.
    /// No calibration or accuracy measurement occurs. Validate actual application
    /// error separately before using a reduced representation.
    public func suggestQuantization(
        for operation: String,
        dimension: Int,
        candidateCount: Int,
        relativeErrorHint: Float = 0.02
    ) throws -> QuantizationStrategy {
        guard dimension > 0 else {
            throw PrecisionTuningSupport.invalid("dimension", "must be positive")
        }
        guard candidateCount >= 0 else {
            throw PrecisionTuningSupport.invalid("candidateCount", "must be nonnegative")
        }
        try PrecisionTuningSupport.validateLimit(relativeErrorHint, field: "relativeErrorHint")
        let request = Request(operation: operation, dimension: dimension,
                              candidateCount: candidateCount,
                              hintBits: relativeErrorHint == 0 ? 0 : relativeErrorHint.bitPattern)
        lock.lock()
        defer { lock.unlock() }
        if let cached = calibrationCache[request] { return cached }
        let strategy: QuantizationStrategy
        if candidateCount == 0 || relativeErrorHint == 0 {
            strategy = .fullFP32
        } else if candidateCount > 5000 && relativeErrorHint >= 0.1 {
            strategy = .bothINT8
        } else if candidateCount > 1000 && relativeErrorHint >= 0.05 {
            strategy = .candidatesINT8
        } else {
            strategy = .fullFP32
        }
        calibrationCache[request] = strategy
        return strategy
    }

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
