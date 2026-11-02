// Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift

import Foundation
import simd

// Import necessary for mathematical functions like ldexp in software fallback.
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

// MARK: - VectorCore Context Assumptions

/*
 This implementation assumes the existence of the following infrastructure within VectorCore:
 1. VectorXXXOptimized structs (e.g., Vector512Optimized) with:
 - internal storage: ContiguousArray<SIMD4<Float>>
 - static lanes property (e.g., 128 for Vector512Optimized)
 - initializer: init(fromSIMDStorage: ContiguousArray<SIMD4<Float>>)
 2. SIMD extensions: SIMD4<Float> with .addProduct() (FMA) and .sum(). SIMD8<Float> with .lowHalf, .highHalf.

 Example Assumed Structures/Extensions (required for compilation):
 extension Vector512Optimized { static let lanes = 128 }
 extension Vector768Optimized { static let lanes = 192 }
 extension Vector1536Optimized { static let lanes = 384 }

 extension SIMD4 where Scalar == Float {
 // Assumes addProduct is implemented elsewhere, e.g.:
 // mutating func addProduct(_ a: SIMD4<Float>, _ b: SIMD4<Float>) {
 //     self = self + (a * b) // illustrative only; the package uses SIMD's fused addProduct
 // }
 // func sum() -> Float { return self[0] + self[1] + self[2] + self[3] }
 }
 */

/// Mixed precision kernel operations with FP16 storage and FP32 computation
///
/// # Error Analysis
///
/// Scope: the facts below hold for finite inputs in the FP16 normal range under round-to-nearest-even. They do not
/// hold through FP16 overflow (finite values from 65520 upward round to infinity), underflow (subnormal or zero
/// results), zero, or cancellation, and they are not an accuracy guarantee for any workload.
///
/// **Unit roundoff**:
/// - FP16 storage: u₁₆ = 2⁻¹¹, the bound on the relative rounding error of one conversion in that domain
///   (FP32→FP16 rounding is pinned against oracle rows by FP16DiagnosticsBoundaryTests.isolatedConstructors)
/// - FP32 arithmetic: u₃₂ = 2⁻²⁴
///
/// **What does not follow** (no bound or property is promised here):
/// - No deterministic relative-error bound for a dot product or distance. Classical worst-case bounds for a
///   length-D sum grow like D·u; √D growth is only a probabilistic estimate; and near-cancelling sums have
///   unbounded relative error.
/// - FP32 accumulation does not prevent catastrophic cancellation.
/// - Rankings are not preserved: distinct FP32 values can round to the same FP16 value and merge ranks.
/// - Conversion maps ±∞ to ±∞ and NaN to a NaN (testFP16SpecialValues; the Q03 oracle rows in
///   FP16DiagnosticsBoundaryTests.isolatedConstructors check ±∞ bit for bit and accept any FP16 NaN for a NaN row). The
///   cosine kernels return `Float.nan` when either side has a NaN or infinite coordinate after conversion
///   (T15CosineExceptionalTableTests.everyImplementationFollowsThePolicy); the other distance kernels do not promise
///   NaN propagation.
/// - NaN policy, encode (FP32→FP16): every NaN becomes the quiet NaN sign | 0x7E00, keeping the sign and dropping the
///   payload, in `fp32ToFp16_scalar`, `convertToFP16`, the `Vector*FP16(from:)` initializers and `convertToFP16_*`,
///   `MixedPrecisionFactory.createVector*FP16(from:)`, the SoA creators, and both the SIMD4 chunks and the tail of
///   `platformOptimizedConvertBatch`
///   (FP16ConverterExhaustiveTests: nanEncodeScalar, nanEncodeConvertToFP16, nanEncodeVectorInitializers,
///   nanEncodeSoACreators, nanEncodePlatformBatchChunks, nanEncodePlatformBatchTail; the batch tests run the loop of
///   the host architecture).
/// - NaN policy, decode (FP16→FP32): every NaN pattern becomes 0x7FC00000 (`Float.nan`), dropping the sign and the
///   payload, in `fp16ToFp32_scalar`, `fp16ToFp32_simd8`, `convertToFP32` and `toFP32()`, and `SoAFP16.getVector(at:)`
///   (FP16ConverterExhaustiveTests: nanDecodeScalar, nanDecodeSIMD8, nanDecodeConvertToFP32, nanDecodeSoAGetVector).
/// - No empirical error rate is published here; the mixed-precision tests check fixture tolerances only.
public enum MixedPrecisionKernels {

    // MARK: - Internal FP16 Conversion Helpers (Scalar)

    // IEEE 754 Half-Precision Format Constants
    private static let fp16ExponentBias: Int = 15
    private static let fp16MantissaBits: Int = 10
    private static let fp16ExponentBits: Int = 5
    private static let fp32ExponentBias: Int = 127

    // These helpers manage the conversion between UInt16 bit patterns and Float32 values.

    // Software conversions: decoding of every FP16 class, and FP32→FP16 encoding with round-to-nearest-even
    // (FP16ConverterExhaustiveTests: decodeAllPatterns, roundTripAllNonNaNPatterns, encodeRoundToNearestEvenNeighbours;
    // oracle rows: FP16DiagnosticsBoundaryTests.isolatedConstructors, Q03). NaN follows the policy in the type doc.

    @inline(__always)
    @usableFromInline
    internal static func fp16ToFp32_scalar(_ fp16: UInt16) -> Float {
        // IEEE 754 half-precision format: Sign: 1 bit, Exponent: 5 bits, Mantissa: 10 bits
        let sign = (fp16 >> 15) & 0x1
        let exponent = (fp16 >> 10) & 0x1F
        let mantissa = fp16 & 0x3FF

        if exponent == 0 {
            if mantissa == 0 {
                // Zero
                return sign == 1 ? -0.0 : 0.0
            } else {
                // Subnormal: value = (-1)^sign × 2^(-14) × (mantissa / 1024)
                let e = -14
                let m = Float(mantissa) / 1024.0
                let value = scalbn(m, e)
                return sign == 1 ? -value : value
            }
        } else if exponent == 0x1F {
            // Infinity or NaN
            if mantissa == 0 {
                return sign == 1 ? -Float.infinity : Float.infinity
            } else {
                return Float.nan
            }
        } else {
            // Normal number: value = (-1)^sign × 2^(exp-15) × (1 + mantissa/1024)
            let e = Int(exponent) - fp16ExponentBias
            let m = 1.0 + Float(mantissa) / 1024.0
            let value = scalbn(m, e)
            return sign == 1 ? -value : value
        }
    }

    /// Software FP32 to FP16 conversion with IEEE 754 round-to-nearest-even
    /// (oracle rows: FP16DiagnosticsBoundaryTests.isolatedConstructors, Q03).
    @inline(__always)
    @usableFromInline
    internal static func fp32ToFp16_scalar(_ fp32: Float) -> UInt16 {
        let bits = fp32.bitPattern
        let sign = UInt16((bits >> 31) & 0x1) << 15
        let exponent = Int((bits >> 23) & 0xFF)
        let mantissa = bits & 0x007FFFFF

        // Handle special cases first
        if exponent == 0xFF {
            // Infinity or NaN
            if mantissa == 0 {
                return sign | 0x7C00  // Infinity
            } else {
                return sign | 0x7C00 | 0x0200  // NaN (set a mantissa bit)
            }
        }

        if exponent == 0 && mantissa == 0 {
            // Zero
            return sign
        }

        // Adjust exponent from FP32 bias (127) to FP16 bias (15)
        var exp16 = exponent - fp32ExponentBias + fp16ExponentBias

        // Handle overflow (too large for FP16)
        if exp16 >= 0x1F {
            return sign | 0x7C00  // Return infinity
        }

        // Handle underflow (too small for FP16 normal, might be subnormal)
        if exp16 <= 0 {
            // Subnormal or flush to zero
            if exp16 < -10 {
                // Too small even for subnormal, flush to zero
                return sign
            }

            // Convert to subnormal: shift the mantissa (with implicit leading 1) right.
            let shift = 1 - exp16
            let mantissa32 = mantissa | 0x00800000  // Add implicit 1
            let shiftAmount = 13 + shift
            let mantissa16Raw = mantissa32 >> shiftAmount

            // Round-to-nearest-even. The round bit is the bit immediately below the kept
            // bits; the sticky bits are everything below that. (The previous code extracted
            // a 2-bit field that included the kept LSB, which spuriously rounded exactly
            // representable subnormals such as 2^-24 (0x0001) up by one ULP to 0x0002.)
            let roundBit = (mantissa32 >> (shiftAmount - 1)) & 0x1
            let stickyMask = (UInt32(1) << (shiftAmount - 1)) &- 1
            let stickyBits = mantissa32 & stickyMask
            let roundUp = roundBit == 1 && (stickyBits != 0 || (mantissa16Raw & 1) == 1)
            // A rounding carry (0x3FF -> 0x400) correctly produces the smallest normal.
            let mantissa16 = UInt16(roundUp ? mantissa16Raw + 1 : mantissa16Raw)

            return sign | mantissa16
        }

        // Normal number conversion with round-to-nearest-even
        // Extract 23-bit mantissa, need to round to 10 bits
        // Round bit is bit 12, sticky bits are bits 0-11
        let roundBit = (mantissa >> 12) & 1
        let stickyBits = mantissa & 0xFFF
        let mantissa16Raw = UInt16(mantissa >> 13)

        // Round to nearest even (banker's rounding)
        // Round up if: round bit is 1 AND (sticky bits are non-zero OR mantissa is odd)
        let shouldRoundUp = roundBit == 1 && (stickyBits != 0 || (mantissa16Raw & 1) == 1)
        var mantissa16 = shouldRoundUp ? mantissa16Raw + 1 : mantissa16Raw

        // Check for mantissa overflow (rounding caused carry-out)
        if mantissa16 > 0x3FF {
            mantissa16 = 0
            exp16 += 1

            // Check for exponent overflow after rounding
            if exp16 >= 0x1F {
                return sign | 0x7C00  // Return infinity
            }
        }

        return sign | (UInt16(exp16) << 10) | mantissa16
    }

    // MARK: - SIMD Conversion Helpers

    /// FP16 (UInt16) to FP32 conversion for SIMD8: eight `fp16ToFp32_scalar` calls.
    ///
    /// Whether the compiler vectorizes them or emits vcvt instructions is not verified.
    @inline(__always)
    @usableFromInline
    internal static func fp16ToFp32_simd8(_ fp16: SIMD8<UInt16>) -> SIMD8<Float> {
        return SIMD8<Float>(
            fp16ToFp32_scalar(fp16[0]), fp16ToFp32_scalar(fp16[1]),
            fp16ToFp32_scalar(fp16[2]), fp16ToFp32_scalar(fp16[3]),
            fp16ToFp32_scalar(fp16[4]), fp16ToFp32_scalar(fp16[5]),
            fp16ToFp32_scalar(fp16[6]), fp16ToFp32_scalar(fp16[7])
        )
    }

    // MARK: - FP16 Vector Storage Types

    /// 512-dimensional vector with FP16 storage (stored as UInt16 bit patterns).
    ///
    /// Scalar payload: 512 × 2 bytes = 1024 bytes of FP16 bit patterns, versus 2048 bytes of FP32 scalars. This is a
    /// payload model only: it excludes the ContiguousArray header and capacity and the handle, and it is not process memory.
    public struct Vector512FP16: Sendable {
        @usableFromInline
        internal let storage: ContiguousArray<UInt16>
        public static let dimension = 512

        /// Initialize from FP32 vector with conversion
        ///
        /// Performs FP32→FP16 conversion with round-to-nearest-even rounding.
        /// Values just above 65504 may round to finite 65504; sufficiently large
        /// finite values round to infinity, and tiny nonzero values may round to zero
        /// (oracle rows: FP16DiagnosticsBoundaryTests.isolatedConstructors, Q03).
        public init(from vector: Vector512Optimized) {
            // Validate input vector has correct storage size
            precondition(
                vector.storage.count == 128,
                "Vector512Optimized must have exactly 128 SIMD4 lanes, got \(vector.storage.count)"
            )

            self.storage = MixedPrecisionKernels.convertToFP16(vector.storage)
            MixedPrecisionKernels.recordConversionsIfEnabled(vector.storage, rounded: self.storage)

            // Verify conversion produced correct element count
            #if DEBUG
            assert(
                self.storage.count == Self.dimension,
                "FP16 conversion must produce \(Self.dimension) elements, got \(self.storage.count)"
            )
            #endif
        }

        /// Initialize from raw FP16 values (as UInt16 bit patterns)
        public init(fp16Values: [UInt16]) {
            guard fp16Values.count == Self.dimension else {
                fatalError("Invalid size for Vector512FP16: expected \(Self.dimension), got \(fp16Values.count)")
            }
            self.storage = ContiguousArray(fp16Values)
        }

        /// Convert back to FP32 representation
        public func toFP32() -> Vector512Optimized {
            let fp32Storage = MixedPrecisionKernels.convertToFP32(self.storage, laneCount: Vector512Optimized.lanes)
            return Vector512Optimized(fromSIMDStorage: fp32Storage)
        }

        /// Borrow the fixed-dimension UInt16 bit patterns for a synchronous callback.
        /// The buffer and its base pointer must not escape through a return value,
        /// capture, task, foreign callback, or stored pointer. Generic R does not
        /// enforce this unsafe caller obligation. For retained or asynchronous data,
        /// return an owned Array created inside the callback instead.
        @inlinable
        public func withUnsafeFP16BufferPointer<R>(
            _ body: (UnsafeBufferPointer<UInt16>) throws -> R
        ) rethrows -> R {
            try storage.withUnsafeBufferPointer(body)
        }
    }

    /// 768-dimensional vector with FP16 storage.
    public struct Vector768FP16: Sendable {
        @usableFromInline
        internal let storage: ContiguousArray<UInt16>
        public static let dimension = 768

        public init(from vector: Vector768Optimized) {
            // Validate input vector has correct storage size
            precondition(
                vector.storage.count == 192,
                "Vector768Optimized must have exactly 192 SIMD4 lanes, got \(vector.storage.count)"
            )

            self.storage = MixedPrecisionKernels.convertToFP16(vector.storage)
            MixedPrecisionKernels.recordConversionsIfEnabled(vector.storage, rounded: self.storage)

            // Verify conversion produced correct element count
            #if DEBUG
            assert(
                self.storage.count == Self.dimension,
                "FP16 conversion must produce \(Self.dimension) elements, got \(self.storage.count)"
            )
            #endif
        }

        public init(fp16Values: [UInt16]) {
            guard fp16Values.count == Self.dimension else {
                fatalError("Invalid size for Vector768FP16: expected \(Self.dimension), got \(fp16Values.count)")
            }
            self.storage = ContiguousArray(fp16Values)
        }

        public func toFP32() -> Vector768Optimized {
            let fp32Storage = MixedPrecisionKernels.convertToFP32(self.storage, laneCount: Vector768Optimized.lanes)
            return Vector768Optimized(fromSIMDStorage: fp32Storage)
        }

        /// Borrow the fixed-dimension UInt16 bit patterns for a synchronous callback.
        /// The buffer and its base pointer must not escape through a return value,
        /// capture, task, foreign callback, or stored pointer. Generic R does not
        /// enforce this unsafe caller obligation. For retained or asynchronous data,
        /// return an owned Array created inside the callback instead.
        @inlinable
        public func withUnsafeFP16BufferPointer<R>(
            _ body: (UnsafeBufferPointer<UInt16>) throws -> R
        ) rethrows -> R {
            try storage.withUnsafeBufferPointer(body)
        }
    }

    /// 1536-dimensional vector with FP16 storage.
    public struct Vector1536FP16: Sendable {
        @usableFromInline
        internal let storage: ContiguousArray<UInt16>
        public static let dimension = 1536

        public init(from vector: Vector1536Optimized) {
            // Validate input vector has correct storage size
            precondition(
                vector.storage.count == 384,
                "Vector1536Optimized must have exactly 384 SIMD4 lanes, got \(vector.storage.count)"
            )

            self.storage = MixedPrecisionKernels.convertToFP16(vector.storage)
            MixedPrecisionKernels.recordConversionsIfEnabled(vector.storage, rounded: self.storage)

            // Verify conversion produced correct element count
            #if DEBUG
            assert(
                self.storage.count == Self.dimension,
                "FP16 conversion must produce \(Self.dimension) elements, got \(self.storage.count)"
            )
            #endif
        }

        public init(fp16Values: [UInt16]) {
            guard fp16Values.count == Self.dimension else {
                fatalError("Invalid size for Vector1536FP16: expected \(Self.dimension), got \(fp16Values.count)")
            }
            self.storage = ContiguousArray(fp16Values)
        }

        public func toFP32() -> Vector1536Optimized {
            let fp32Storage = MixedPrecisionKernels.convertToFP32(self.storage, laneCount: Vector1536Optimized.lanes)
            return Vector1536Optimized(fromSIMDStorage: fp32Storage)
        }

        /// Borrow the fixed-dimension UInt16 bit patterns for a synchronous callback.
        /// The buffer and its base pointer must not escape through a return value,
        /// capture, task, foreign callback, or stored pointer. Generic R does not
        /// enforce this unsafe caller obligation. For retained or asynchronous data,
        /// return an owned Array created inside the callback instead.
        @inlinable
        public func withUnsafeFP16BufferPointer<R>(
            _ body: (UnsafeBufferPointer<UInt16>) throws -> R
        ) rethrows -> R {
            try storage.withUnsafeBufferPointer(body)
        }
    }

    // MARK: - Structure-of-Arrays (SoA) FP16 Types

    /// Generic Structure-of-Arrays container for FP16 vectors.
    ///
    /// **Memory Layout (Blocked SoA):**
    /// Vectors are grouped in sets of 4 and stored dimension-major within each group.
    /// For N vectors with D dimensions:
    /// ```
    /// [group0: d0(v0,v1,v2,v3), d1(v0,v1,v2,v3), ..., dD(v0,v1,v2,v3),
    ///  group1: d0(v4,v5,v6,v7), d1(v4,v5,v6,v7), ..., dD(v4,v5,v6,v7),
    ///  ...]
    /// ```
    ///
    /// **Usage:**
    /// ```swift
    /// let vectors: [Vector512Optimized] = ...
    /// let soa = SoAFP16<Vector512Optimized>(from: vectors)
    ///
    /// var results = [Float](repeating: 0, count: vectors.count)
    /// results.withUnsafeMutableBufferPointer { buffer in
    ///     MixedPrecisionKernels.batchEuclidean512(
    ///         query: queryFP16,
    ///         candidates: soa,
    ///         results: buffer
    ///     )
    /// }
    /// ```
    public struct SoAFP16<VectorType: OptimizedVector>: Sendable {
        /// FP16 storage as UInt16 bit patterns (matches VectorCore architecture)
        public let storage: ContiguousArray<UInt16>

        /// Number of vectors stored
        public let vectorCount: Int

        /// Dimension of each vector
        public let dimension: Int

        /// Number of groups (ceil(vectorCount / 4))
        @usableFromInline
        internal let groupCount: Int

        /// Internal init for specialized initializers
        @usableFromInline
        internal init(storage: ContiguousArray<UInt16>, vectorCount: Int, dimension: Int) {
            self.storage = storage
            self.vectorCount = vectorCount
            self.dimension = dimension
            self.groupCount = (vectorCount + 3) / 4
        }

        /// Initialize empty SoA with given capacity
        public init(capacity: Int, dimension: Int) {
            let groupCap = (capacity + 3) / 4
            let storageSize = groupCap * dimension * 4
            self.storage = ContiguousArray<UInt16>(repeating: 0, count: storageSize)
            self.vectorCount = 0
            self.dimension = dimension
            self.groupCount = groupCap
        }

        /// Convenience initializer from vectors (for test compatibility).
        ///
        /// - Note: `SoAFP16` uses vector-grouping (groups of 4 vectors, dimension-first),
        ///   not dimension-blocking; there is intentionally no `blockSize` parameter. (A
        ///   prior no-op `blockSize:` was removed in 0.3.0.)
        /// - Parameters:
        ///   - vectors: Array of optimized vectors to convert
        /// - Throws: VectorError if vectors have incompatible dimensions
        public init(vectors: [VectorType]) throws {
            // An empty vector array yields a valid empty SoA (vectorCount == 0); the
            // specialized builders below already handle that degenerate case without throwing.
            // Delegate to the specialized creation function for each supported dimension.
            if VectorType.self == Vector512Optimized.self {
                let soa = MixedPrecisionKernels.createSoA512FP16(from: exactArray(vectors, as: Vector512Optimized.self)!)
                self = soa as! Self
            } else if VectorType.self == Vector768Optimized.self {
                let soa = MixedPrecisionKernels.createSoA768FP16(from: exactArray(vectors, as: Vector768Optimized.self)!)
                self = soa as! Self
            } else if VectorType.self == Vector1536Optimized.self {
                let soa = MixedPrecisionKernels.createSoA1536FP16(from: exactArray(vectors, as: Vector1536Optimized.self)!)
                self = soa as! Self
            } else {
                throw VectorError.invalidData(
                    "Unsupported vector type \(VectorType.self) for SoAFP16; expected Vector512/768/1536Optimized"
                )
            }
        }

        /// Extract a single vector from SoA layout (for testing/validation)
        ///
        /// Converts FP16 values back to FP32. Useful for round-trip accuracy testing.
        ///
        /// **Storage Layout:** Vectors are stored in groups of 4, transposed dimension-first:
        /// - Group 0, dim 0: [v0[0], v1[0], v2[0], v3[0]]
        /// - Group 0, dim 1: [v0[1], v1[1], v2[1], v3[1]]
        /// - ...
        ///
        /// - Parameter index: Vector index (must be < vectorCount)
        /// - Returns: Array of FP32 values for the requested vector
        /// - Throws: VectorError if index is out of bounds
        public func getVector(at index: Int) throws -> [Float] {
            guard index >= 0 && index < vectorCount else {
                throw VectorError.indexOutOfBounds(index: index, dimension: vectorCount)
            }

            var result = [Float](repeating: 0, count: dimension)

            // Determine which group this vector belongs to
            let groupIndex = index / 4
            let vectorInGroup = index % 4  // Position within group (0-3)

            // Calculate starting offset for this group
            let groupOffset = groupIndex * dimension * 4

            // Extract values for each dimension
            for d in 0..<dimension {
                // In the SoA layout, dimension d's values for a group of 4 vectors are consecutive
                let dimOffset = groupOffset + (d * 4)
                let fp16Bits = storage[dimOffset + vectorInGroup]

                // Convert FP16 to FP32
                result[d] = MixedPrecisionKernels.fp16ToFp32_scalar(fp16Bits)
            }

            return result
        }
    }

    /// Specialized SoA typealias for 512-dim vectors
    public typealias SoA512FP16 = SoAFP16<Vector512Optimized>

    /// Specialized SoA typealias for 768-dim vectors
    public typealias SoA768FP16 = SoAFP16<Vector768Optimized>

    /// Specialized SoA typealias for 1536-dim vectors
    public typealias SoA1536FP16 = SoAFP16<Vector1536Optimized>

    // MARK: - SoA Initialization Helpers

    /// Helper function to initialize SoA512FP16 from FP32 vectors with 4×4 block transposition
    @inlinable
    public static func createSoA512FP16(from vectors: [Vector512Optimized]) -> SoA512FP16 {
        let vectorCount = vectors.count
        guard vectorCount > 0 else {
            return SoA512FP16(capacity: 0, dimension: 512)
        }

        let dimension = 512
        let groupCount = (vectorCount + 3) / 4
        let storageSize = groupCount * dimension * 4

        // Initialize in place through ContiguousArray(unsafeUninitializedCapacity:)
        let storage = ContiguousArray<UInt16>(unsafeUninitializedCapacity: storageSize) { buffer, initializedCount in
            guard let storagePtr = buffer.baseAddress else {
                initializedCount = 0
                return
            }

            var writeIndex = 0

            // Process groups of 4 vectors
            for groupStart in stride(from: 0, to: vectorCount, by: 4) {
                let actualCount = min(4, vectorCount - groupStart)

                // Convert vectors to FP16 once
                let v0_fp16 = Vector512FP16(from: vectors[groupStart])
                let v1_fp16 = actualCount > 1 ? Vector512FP16(from: vectors[groupStart + 1]) : nil
                let v2_fp16 = actualCount > 2 ? Vector512FP16(from: vectors[groupStart + 2]) : nil
                let v3_fp16 = actualCount > 3 ? Vector512FP16(from: vectors[groupStart + 3]) : nil

                // Access FP16 storage
                v0_fp16.storage.withUnsafeBufferPointer { v0Ptr in
                    let v1Storage = v1_fp16?.storage ?? ContiguousArray<UInt16>(repeating: 0, count: 512)
                    let v2Storage = v2_fp16?.storage ?? ContiguousArray<UInt16>(repeating: 0, count: 512)
                    let v3Storage = v3_fp16?.storage ?? ContiguousArray<UInt16>(repeating: 0, count: 512)

                    v1Storage.withUnsafeBufferPointer { v1Ptr in
                        v2Storage.withUnsafeBufferPointer { v2Ptr in
                            v3Storage.withUnsafeBufferPointer { v3Ptr in
                                guard let v0 = v0Ptr.baseAddress,
                                      let v1 = v1Ptr.baseAddress,
                                      let v2 = v2Ptr.baseAddress,
                                      let v3 = v3Ptr.baseAddress else { return }

                                // Transpose: dimension-major within group
                                for d in 0..<dimension {
                                    storagePtr[writeIndex + 0] = v0[d]
                                    storagePtr[writeIndex + 1] = v1[d]
                                    storagePtr[writeIndex + 2] = v2[d]
                                    storagePtr[writeIndex + 3] = v3[d]
                                    writeIndex += 4
                                }
                            }
                        }
                    }
                }
            }

            initializedCount = storageSize
        }

        // Use the private init
        return SoA512FP16(storage: storage, vectorCount: vectorCount, dimension: dimension)
    }

    /// Helper function to initialize SoA768FP16 from FP32 vectors with 4×4 block transposition
    @inlinable
    public static func createSoA768FP16(from vectors: [Vector768Optimized]) -> SoA768FP16 {
        let vectorCount = vectors.count
        guard vectorCount > 0 else {
            return SoA768FP16(capacity: 0, dimension: 768)
        }

        let dimension = 768
        let groupCount = (vectorCount + 3) / 4
        let storageSize = groupCount * dimension * 4

        // Initialize in place through ContiguousArray(unsafeUninitializedCapacity:)
        let storage = ContiguousArray<UInt16>(unsafeUninitializedCapacity: storageSize) { buffer, initializedCount in
            guard let storagePtr = buffer.baseAddress else {
                initializedCount = 0
                return
            }

            var writeIndex = 0

            // Process groups of 4 vectors
            for groupStart in stride(from: 0, to: vectorCount, by: 4) {
                let actualCount = min(4, vectorCount - groupStart)

                // Convert vectors to FP16 once
                let v0_fp16 = Vector768FP16(from: vectors[groupStart])
                let v1_fp16 = actualCount > 1 ? Vector768FP16(from: vectors[groupStart + 1]) : nil
                let v2_fp16 = actualCount > 2 ? Vector768FP16(from: vectors[groupStart + 2]) : nil
                let v3_fp16 = actualCount > 3 ? Vector768FP16(from: vectors[groupStart + 3]) : nil

                // Access FP16 storage
                v0_fp16.storage.withUnsafeBufferPointer { v0Ptr in
                    let v1Storage = v1_fp16?.storage ?? ContiguousArray<UInt16>(repeating: 0, count: 768)
                    let v2Storage = v2_fp16?.storage ?? ContiguousArray<UInt16>(repeating: 0, count: 768)
                    let v3Storage = v3_fp16?.storage ?? ContiguousArray<UInt16>(repeating: 0, count: 768)

                    v1Storage.withUnsafeBufferPointer { v1Ptr in
                        v2Storage.withUnsafeBufferPointer { v2Ptr in
                            v3Storage.withUnsafeBufferPointer { v3Ptr in
                                guard let v0 = v0Ptr.baseAddress,
                                      let v1 = v1Ptr.baseAddress,
                                      let v2 = v2Ptr.baseAddress,
                                      let v3 = v3Ptr.baseAddress else { return }

                                // Transpose: dimension-major within group
                                for d in 0..<dimension {
                                    storagePtr[writeIndex + 0] = v0[d]
                                    storagePtr[writeIndex + 1] = v1[d]
                                    storagePtr[writeIndex + 2] = v2[d]
                                    storagePtr[writeIndex + 3] = v3[d]
                                    writeIndex += 4
                                }
                            }
                        }
                    }
                }
            }

            initializedCount = storageSize
        }

        // Use the private init
        return SoA768FP16(storage: storage, vectorCount: vectorCount, dimension: dimension)
    }

    /// Helper function to initialize SoA1536FP16 from FP32 vectors with 4×4 block transposition
    @inlinable
    public static func createSoA1536FP16(from vectors: [Vector1536Optimized]) -> SoA1536FP16 {
        let vectorCount = vectors.count
        guard vectorCount > 0 else {
            return SoA1536FP16(capacity: 0, dimension: 1536)
        }

        let dimension = 1536
        let groupCount = (vectorCount + 3) / 4
        let storageSize = groupCount * dimension * 4

        // Initialize in place through ContiguousArray(unsafeUninitializedCapacity:)
        let storage = ContiguousArray<UInt16>(unsafeUninitializedCapacity: storageSize) { buffer, initializedCount in
            guard let storagePtr = buffer.baseAddress else {
                initializedCount = 0
                return
            }

            var writeIndex = 0

            // Process groups of 4 vectors
            for groupStart in stride(from: 0, to: vectorCount, by: 4) {
                let actualCount = min(4, vectorCount - groupStart)

                // Convert vectors to FP16 once
                let v0_fp16 = Vector1536FP16(from: vectors[groupStart])
                let v1_fp16 = actualCount > 1 ? Vector1536FP16(from: vectors[groupStart + 1]) : nil
                let v2_fp16 = actualCount > 2 ? Vector1536FP16(from: vectors[groupStart + 2]) : nil
                let v3_fp16 = actualCount > 3 ? Vector1536FP16(from: vectors[groupStart + 3]) : nil

                // Access FP16 storage
                v0_fp16.storage.withUnsafeBufferPointer { v0Ptr in
                    let v1Storage = v1_fp16?.storage ?? ContiguousArray<UInt16>(repeating: 0, count: 1536)
                    let v2Storage = v2_fp16?.storage ?? ContiguousArray<UInt16>(repeating: 0, count: 1536)
                    let v3Storage = v3_fp16?.storage ?? ContiguousArray<UInt16>(repeating: 0, count: 1536)

                    v1Storage.withUnsafeBufferPointer { v1Ptr in
                        v2Storage.withUnsafeBufferPointer { v2Ptr in
                            v3Storage.withUnsafeBufferPointer { v3Ptr in
                                guard let v0 = v0Ptr.baseAddress,
                                      let v1 = v1Ptr.baseAddress,
                                      let v2 = v2Ptr.baseAddress,
                                      let v3 = v3Ptr.baseAddress else { return }

                                // Transpose: dimension-major within group
                                for d in 0..<dimension {
                                    storagePtr[writeIndex + 0] = v0[d]
                                    storagePtr[writeIndex + 1] = v1[d]
                                    storagePtr[writeIndex + 2] = v2[d]
                                    storagePtr[writeIndex + 3] = v3[d]
                                    writeIndex += 4
                                }
                            }
                        }
                    }
                }
            }

            initializedCount = storageSize
        }

        // Use the private init
        return SoA1536FP16(storage: storage, vectorCount: vectorCount, dimension: dimension)
    }

    // MARK: - Conversion Utilities (Public Bulk Conversion)

    /// FP32 storage to FP16 conversion: one `fp32ToFp16_scalar` call per element.
    ///
    /// Rounds to nearest, ties to even (oracle rows: FP16DiagnosticsBoundaryTests.isolatedConstructors, Q03).
    @inlinable
    public static func convertToFP16(
        _ values: ContiguousArray<SIMD4<Float>>
    ) -> ContiguousArray<UInt16> {
        let floatCount = values.count * 4
        var result = ContiguousArray<UInt16>()
        result.reserveCapacity(floatCount)

        // Process 1 SIMD4<Float> at a time.
        for simd in values {
            // Convert with the scalar round-to-nearest-even helper
            result.append(fp32ToFp16_scalar(simd[0]))
            result.append(fp32ToFp16_scalar(simd[1]))
            result.append(fp32ToFp16_scalar(simd[2]))
            result.append(fp32ToFp16_scalar(simd[3]))
        }

        // Validate output size
        #if DEBUG
        assert(
            result.count == floatCount,
            "FP16 conversion produced \(result.count) elements, expected \(floatCount)"
        )
        #endif

        return result
    }

    /// Record original inputs and actual rounded output bits at the diagnostic
    /// recorder's locked commit. Its synchronized disabled hint may skip preparation;
    /// concurrent toggles do not promise to count every physical conversion
    /// (FP16DiagnosticsBoundaryTests.suppliedRoundedBitsDriveClassification, isolatedConcurrency).
    @inline(__always)
    internal static func recordConversionsIfEnabled(
        _ values: ContiguousArray<SIMD4<Float>>, rounded: ContiguousArray<UInt16>
    ) {
        MixedPrecisionDiagnostics.shared.recordBatchConversion(values, rounded: rounded)
    }

    /// FP16 to FP32 storage conversion: SIMD8 loads, each converted by eight scalar conversions.
    @inlinable
    public static func convertToFP32(
        _ fp16Values: ContiguousArray<UInt16>,
        laneCount: Int
    ) -> ContiguousArray<SIMD4<Float>> {
        // Validate input size matches expected lane count
        let expectedElementCount = laneCount * 4
        precondition(
            fp16Values.count == expectedElementCount,
            "FP16 input size mismatch: expected \(expectedElementCount) elements for \(laneCount) lanes, got \(fp16Values.count)"
        )

        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(laneCount)

        // Process 8 UInt16 (8 elements) at a time → 2 SIMD4<Float>.
        // Dimensions (512, 768, 1536) are multiples of 8.
        fp16Values.withUnsafeBufferPointer { buffer in
            guard let basePtr = buffer.baseAddress else { return }

            var i = 0
            while i + 8 <= fp16Values.count {
                // One SIMD8<UInt16> load via loadUnaligned
                let simd8_u16 = UnsafeRawPointer(basePtr.advanced(by: i))
                    .loadUnaligned(as: SIMD8<UInt16>.self)

                // Convert with fp16ToFp32_simd8 (eight scalar conversions)
                let simd8_f32 = fp16ToFp32_simd8(simd8_u16)

                result.append(simd8_f32.lowHalf)
                result.append(simd8_f32.highHalf)
                i += 8
            }

            // Handle remaining elements (should not occur for our dimensions, but for safety)
            while i < fp16Values.count {
                let v0 = i < fp16Values.count ? fp16ToFp32_scalar(basePtr[i]) : 0
                let v1 = i+1 < fp16Values.count ? fp16ToFp32_scalar(basePtr[i+1]) : 0
                let v2 = i+2 < fp16Values.count ? fp16ToFp32_scalar(basePtr[i+2]) : 0
                let v3 = i+3 < fp16Values.count ? fp16ToFp32_scalar(basePtr[i+3]) : 0
                result.append(SIMD4<Float>(v0, v1, v2, v3))
                i += 4
            }
        }

        // Validate output size
        #if DEBUG
        assert(
            result.count == laneCount,
            "Conversion produced \(result.count) lanes, expected \(laneCount)"
        )
        #endif

        return result
    }

    // MARK: - Core Kernel Implementations

    /// Core FP16 dot product with FP32 accumulation.
    ///
    /// Processes 32 FP16 elements (64 bytes per operand) per iteration with 4 independent FP32 accumulators.
    @inline(__always)
    @usableFromInline
    internal static func dotFP16Core(
        aPtr: UnsafePointer<UInt16>,
        bPtr: UnsafePointer<UInt16>,
        count: Int
    ) -> Float {
        // Four independent accumulators.
        var acc0 = SIMD4<Float>.zero
        var acc1 = SIMD4<Float>.zero
        var acc2 = SIMD4<Float>.zero
        var acc3 = SIMD4<Float>.zero

        // Assuming count is a multiple of 32 (true for 512, 768, 1536).
        var i = 0
        while i + 32 <= count {
            // --- Batch 0: elements 0-7 ---
            // Load 8 FP16 values with one SIMD8 load
            let a0_fp16 = UnsafeRawPointer(aPtr.advanced(by: i))
                .loadUnaligned(as: SIMD8<UInt16>.self)
            let b0_fp16 = UnsafeRawPointer(bPtr.advanced(by: i))
                .loadUnaligned(as: SIMD8<UInt16>.self)

            // Convert to FP32
            let a0_fp32 = fp16ToFp32_simd8(a0_fp16)
            let b0_fp32 = fp16ToFp32_simd8(b0_fp16)

            // FMA accumulation
            acc0.addProduct(a0_fp32.lowHalf, b0_fp32.lowHalf)
            acc1.addProduct(a0_fp32.highHalf, b0_fp32.highHalf)

            // --- Batch 1: elements 8-15 ---
            let a1_fp16 = UnsafeRawPointer(aPtr.advanced(by: i + 8))
                .loadUnaligned(as: SIMD8<UInt16>.self)
            let b1_fp16 = UnsafeRawPointer(bPtr.advanced(by: i + 8))
                .loadUnaligned(as: SIMD8<UInt16>.self)

            let a1_fp32 = fp16ToFp32_simd8(a1_fp16)
            let b1_fp32 = fp16ToFp32_simd8(b1_fp16)

            acc2.addProduct(a1_fp32.lowHalf, b1_fp32.lowHalf)
            acc3.addProduct(a1_fp32.highHalf, b1_fp32.highHalf)

            // --- Batch 2: elements 16-23 ---
            let a2_fp16 = UnsafeRawPointer(aPtr.advanced(by: i + 16))
                .loadUnaligned(as: SIMD8<UInt16>.self)
            let b2_fp16 = UnsafeRawPointer(bPtr.advanced(by: i + 16))
                .loadUnaligned(as: SIMD8<UInt16>.self)

            let a2_fp32 = fp16ToFp32_simd8(a2_fp16)
            let b2_fp32 = fp16ToFp32_simd8(b2_fp16)

            // Batches 2 and 3 reuse acc0...acc3, so each accumulator receives two products per iteration.
            acc0.addProduct(a2_fp32.lowHalf, b2_fp32.lowHalf)
            acc1.addProduct(a2_fp32.highHalf, b2_fp32.highHalf)

            // --- Batch 3: elements 24-31 ---
            let a3_fp16 = UnsafeRawPointer(aPtr.advanced(by: i + 24))
                .loadUnaligned(as: SIMD8<UInt16>.self)
            let b3_fp16 = UnsafeRawPointer(bPtr.advanced(by: i + 24))
                .loadUnaligned(as: SIMD8<UInt16>.self)

            let a3_fp32 = fp16ToFp32_simd8(a3_fp16)
            let b3_fp32 = fp16ToFp32_simd8(b3_fp16)

            acc2.addProduct(a3_fp32.lowHalf, b3_fp32.lowHalf)
            acc3.addProduct(a3_fp32.highHalf, b3_fp32.highHalf)

            i += 32
        }

        // Handle remaining elements (for dimensions not divisible by 32)
        while i < count {
            let a_fp32 = fp16ToFp32_scalar(aPtr[i])
            let b_fp32 = fp16ToFp32_scalar(bPtr[i])
            acc0[0] += a_fp32 * b_fp32
            i += 1
        }

        // Final horizontal reduction
        let sum = (acc0 + acc1) + (acc2 + acc3)
        return sum.sum()
    }

    /// Core Mixed Precision Dot Product (FP32 Query, FP16 Candidate).
    ///
    /// Converts each FP16 candidate block to FP32 immediately before the multiply-add. The FP32 query is used
    /// without conversion, so query values are not rounded to FP16.
    @inline(__always)
    @usableFromInline
    internal static func dotMixedCore(
        queryStorage: ContiguousArray<SIMD4<Float>>, // FP32 Query
        candidatePtr: UnsafePointer<UInt16>,         // FP16 Candidate
        lanes: Int // Number of SIMD4 lanes
    ) -> Float {
        var acc0 = SIMD4<Float>.zero
        var acc1 = SIMD4<Float>.zero
        var acc2 = SIMD4<Float>.zero
        var acc3 = SIMD4<Float>.zero

        // Process 4 lanes (16 elements) at a time
        // Lanes (128, 192, 384) are multiples of 4.
        var i = 0
        while i + 4 <= lanes {
            // Load query (FP32; no conversion)
            let q0 = queryStorage[i]
            let q1 = queryStorage[i+1]
            let q2 = queryStorage[i+2]
            let q3 = queryStorage[i+3]

            // Load candidate (FP16) - 16 elements starting at index i*4
            let offset = i * 4

            // Batch 0: elements 0-7 (one SIMD8 load)
            let c0_fp16 = UnsafeRawPointer(candidatePtr.advanced(by: offset))
                .loadUnaligned(as: SIMD8<UInt16>.self)
            let c0_fp32 = fp16ToFp32_simd8(c0_fp16)

            // Batch 1: elements 8-15
            let c1_fp16 = UnsafeRawPointer(candidatePtr.advanced(by: offset + 8))
                .loadUnaligned(as: SIMD8<UInt16>.self)
            let c1_fp32 = fp16ToFp32_simd8(c1_fp16)

            // FMA accumulation: Match FP32 query lane with corresponding converted candidate data.
            acc0.addProduct(q0, c0_fp32.lowHalf)
            acc1.addProduct(q1, c0_fp32.highHalf)
            acc2.addProduct(q2, c1_fp32.lowHalf)
            acc3.addProduct(q3, c1_fp32.highHalf)

            i += 4
        }

        // Handle remaining lanes
        while i < lanes {
            let q = queryStorage[i]
            let offset = i * 4

            let c0 = fp16ToFp32_scalar(candidatePtr[offset])
            let c1 = fp16ToFp32_scalar(candidatePtr[offset + 1])
            let c2 = fp16ToFp32_scalar(candidatePtr[offset + 2])
            let c3 = fp16ToFp32_scalar(candidatePtr[offset + 3])

            let c = SIMD4<Float>(c0, c1, c2, c3)
            acc0.addProduct(q, c)
            i += 1
        }

        let sum = (acc0 + acc1) + (acc2 + acc3)
        return sum.sum()
    }

    // MARK: - Public API: FP16 Dot Products

    /// Compute dot product with both vectors in FP16 storage, FP32 accumulation.
    ///
    /// Use this when both query and candidates are stored in FP16 format. Both operands
    /// then carry FP16 rounding; no bandwidth or latency figure is measured.
    @inlinable
    public static func dotFP16_512(_ a: Vector512FP16, _ b: Vector512FP16) -> Float {
        return a.storage.withUnsafeBufferPointer { aBuffer in
            return b.storage.withUnsafeBufferPointer { bBuffer in
                guard let aPtr = aBuffer.baseAddress, let bPtr = bBuffer.baseAddress else { return 0.0 }
                return dotFP16Core(
                    aPtr: aPtr,
                    bPtr: bPtr,
                    count: Vector512FP16.dimension
                )
            }
        }
    }

    @inlinable
    public static func dotFP16_768(_ a: Vector768FP16, _ b: Vector768FP16) -> Float {
        return a.storage.withUnsafeBufferPointer { aBuffer in
            return b.storage.withUnsafeBufferPointer { bBuffer in
                guard let aPtr = aBuffer.baseAddress, let bPtr = bBuffer.baseAddress else { return 0.0 }
                return dotFP16Core(
                    aPtr: aPtr,
                    bPtr: bPtr,
                    count: Vector768FP16.dimension
                )
            }
        }
    }

    @inlinable
    public static func dotFP16_1536(_ a: Vector1536FP16, _ b: Vector1536FP16) -> Float {
        return a.storage.withUnsafeBufferPointer { aBuffer in
            return b.storage.withUnsafeBufferPointer { bBuffer in
                guard let aPtr = aBuffer.baseAddress, let bPtr = bBuffer.baseAddress else { return 0.0 }
                return dotFP16Core(
                    aPtr: aPtr,
                    bPtr: bPtr,
                    count: Vector1536FP16.dimension
                )
            }
        }
    }

    // MARK: - Public API: Mixed Precision Dot Products

    /// Mixed precision dot product: FP32 query × FP16 candidate.
    ///
    /// Preserves full query precision (the FP32 query is not converted) with FP16 candidate storage:
    /// - Query: FP32, not converted
    /// - Candidates: FP16 bit patterns (half the scalar payload of FP32; payload model only)
    /// - Computation: FP32 multiply-add after converting each candidate value
    @inlinable
    public static func dotMixed512(
        query: Vector512Optimized,
        candidate: Vector512FP16
    ) -> Float {
        return candidate.storage.withUnsafeBufferPointer { cBuffer in
            guard let cPtr = cBuffer.baseAddress else { return 0.0 }
            return dotMixedCore(
                queryStorage: query.storage,
                candidatePtr: cPtr,
                lanes: Vector512Optimized.lanes
            )
        }
    }

    @inlinable
    public static func dotMixed768(
        query: Vector768Optimized,
        candidate: Vector768FP16
    ) -> Float {
        return candidate.storage.withUnsafeBufferPointer { cBuffer in
            guard let cPtr = cBuffer.baseAddress else { return 0.0 }
            return dotMixedCore(
                queryStorage: query.storage,
                candidatePtr: cPtr,
                lanes: Vector768Optimized.lanes
            )
        }
    }

    @inlinable
    public static func dotMixed1536(
        query: Vector1536Optimized,
        candidate: Vector1536FP16
    ) -> Float {
        return candidate.storage.withUnsafeBufferPointer { cBuffer in
            guard let cPtr = cBuffer.baseAddress else { return 0.0 }
            return dotMixedCore(
                queryStorage: query.storage,
                candidatePtr: cPtr,
                lanes: Vector1536Optimized.lanes
            )
        }
    }

    // MARK: - Euclidean Distance Kernels

    /// Core FP16 Euclidean distance with FP32 accumulation.
    ///
    /// Uses 4 independent accumulators and processes 32 FP16 elements (64 bytes per operand) per iteration.
    @inline(__always)
    @usableFromInline
    internal static func euclideanFP16Core(
        aPtr: UnsafePointer<UInt16>,
        bPtr: UnsafePointer<UInt16>,
        count: Int
    ) -> Float {
        var acc0 = SIMD4<Float>.zero
        var acc1 = SIMD4<Float>.zero
        var acc2 = SIMD4<Float>.zero
        var acc3 = SIMD4<Float>.zero

        var i = 0
        while i + 32 <= count {
            // Batch 0: elements 0-7
            let a0_fp16 = UnsafeRawPointer(aPtr.advanced(by: i))
                .loadUnaligned(as: SIMD8<UInt16>.self)
            let b0_fp16 = UnsafeRawPointer(bPtr.advanced(by: i))
                .loadUnaligned(as: SIMD8<UInt16>.self)

            let a0_fp32 = fp16ToFp32_simd8(a0_fp16)
            let b0_fp32 = fp16ToFp32_simd8(b0_fp16)

            let diff0_low = a0_fp32.lowHalf - b0_fp32.lowHalf
            let diff0_high = a0_fp32.highHalf - b0_fp32.highHalf
            acc0.addProduct(diff0_low, diff0_low)
            acc1.addProduct(diff0_high, diff0_high)

            // Batch 1: elements 8-15
            let a1_fp16 = UnsafeRawPointer(aPtr.advanced(by: i + 8))
                .loadUnaligned(as: SIMD8<UInt16>.self)
            let b1_fp16 = UnsafeRawPointer(bPtr.advanced(by: i + 8))
                .loadUnaligned(as: SIMD8<UInt16>.self)

            let a1_fp32 = fp16ToFp32_simd8(a1_fp16)
            let b1_fp32 = fp16ToFp32_simd8(b1_fp16)

            let diff1_low = a1_fp32.lowHalf - b1_fp32.lowHalf
            let diff1_high = a1_fp32.highHalf - b1_fp32.highHalf
            acc2.addProduct(diff1_low, diff1_low)
            acc3.addProduct(diff1_high, diff1_high)

            // Batch 2: elements 16-23
            let a2_fp16 = UnsafeRawPointer(aPtr.advanced(by: i + 16))
                .loadUnaligned(as: SIMD8<UInt16>.self)
            let b2_fp16 = UnsafeRawPointer(bPtr.advanced(by: i + 16))
                .loadUnaligned(as: SIMD8<UInt16>.self)

            let a2_fp32 = fp16ToFp32_simd8(a2_fp16)
            let b2_fp32 = fp16ToFp32_simd8(b2_fp16)

            let diff2_low = a2_fp32.lowHalf - b2_fp32.lowHalf
            let diff2_high = a2_fp32.highHalf - b2_fp32.highHalf
            acc0.addProduct(diff2_low, diff2_low)
            acc1.addProduct(diff2_high, diff2_high)

            // Batch 3: elements 24-31
            let a3_fp16 = UnsafeRawPointer(aPtr.advanced(by: i + 24))
                .loadUnaligned(as: SIMD8<UInt16>.self)
            let b3_fp16 = UnsafeRawPointer(bPtr.advanced(by: i + 24))
                .loadUnaligned(as: SIMD8<UInt16>.self)

            let a3_fp32 = fp16ToFp32_simd8(a3_fp16)
            let b3_fp32 = fp16ToFp32_simd8(b3_fp16)

            let diff3_low = a3_fp32.lowHalf - b3_fp32.lowHalf
            let diff3_high = a3_fp32.highHalf - b3_fp32.highHalf
            acc2.addProduct(diff3_low, diff3_low)
            acc3.addProduct(diff3_high, diff3_high)

            i += 32
        }

        // Handle remaining elements
        while i < count {
            let a_fp32 = fp16ToFp32_scalar(aPtr[i])
            let b_fp32 = fp16ToFp32_scalar(bPtr[i])
            let diff = a_fp32 - b_fp32
            acc0[0] += diff * diff
            i += 1
        }

        let sum = (acc0 + acc1) + (acc2 + acc3)
        return sqrt(sum.sum())
    }

    /// Core Mixed Precision Euclidean Distance (FP32 Query, FP16 Candidate).
    @inline(__always)
    @usableFromInline
    internal static func euclideanMixedCore(
        queryStorage: ContiguousArray<SIMD4<Float>>,
        candidatePtr: UnsafePointer<UInt16>,
        lanes: Int
    ) -> Float {
        var acc0 = SIMD4<Float>.zero
        var acc1 = SIMD4<Float>.zero
        var acc2 = SIMD4<Float>.zero
        var acc3 = SIMD4<Float>.zero

        var i = 0
        while i + 4 <= lanes {
            let q0 = queryStorage[i]
            let q1 = queryStorage[i+1]
            let q2 = queryStorage[i+2]
            let q3 = queryStorage[i+3]

            let offset = i * 4

            // Load 16 FP16 elements
            let c0_fp16 = UnsafeRawPointer(candidatePtr.advanced(by: offset))
                .loadUnaligned(as: SIMD8<UInt16>.self)
            let c0_fp32 = fp16ToFp32_simd8(c0_fp16)

            let c1_fp16 = UnsafeRawPointer(candidatePtr.advanced(by: offset + 8))
                .loadUnaligned(as: SIMD8<UInt16>.self)
            let c1_fp32 = fp16ToFp32_simd8(c1_fp16)

            let diff0 = q0 - c0_fp32.lowHalf
            let diff1 = q1 - c0_fp32.highHalf
            let diff2 = q2 - c1_fp32.lowHalf
            let diff3 = q3 - c1_fp32.highHalf

            acc0.addProduct(diff0, diff0)
            acc1.addProduct(diff1, diff1)
            acc2.addProduct(diff2, diff2)
            acc3.addProduct(diff3, diff3)

            i += 4
        }

        // Handle remaining lanes
        while i < lanes {
            let q = queryStorage[i]
            let offset = i * 4

            let c0 = fp16ToFp32_scalar(candidatePtr[offset])
            let c1 = fp16ToFp32_scalar(candidatePtr[offset + 1])
            let c2 = fp16ToFp32_scalar(candidatePtr[offset + 2])
            let c3 = fp16ToFp32_scalar(candidatePtr[offset + 3])

            let c = SIMD4<Float>(c0, c1, c2, c3)
            let diff = q - c
            acc0.addProduct(diff, diff)
            i += 1
        }

        let sum = (acc0 + acc1) + (acc2 + acc3)
        return sqrt(sum.sum())
    }

    // MARK: - Public API: FP16 Euclidean Distance

    /// Euclidean distance with both vectors in FP16 storage.
    @inlinable
    public static func euclidean512(_ a: Vector512FP16, _ b: Vector512FP16) -> Float {
        return a.storage.withUnsafeBufferPointer { aBuffer in
            return b.storage.withUnsafeBufferPointer { bBuffer in
                guard let aPtr = aBuffer.baseAddress, let bPtr = bBuffer.baseAddress else { return 0.0 }
                return euclideanFP16Core(
                    aPtr: aPtr,
                    bPtr: bPtr,
                    count: Vector512FP16.dimension
                )
            }
        }
    }

    @inlinable
    public static func euclidean768(_ a: Vector768FP16, _ b: Vector768FP16) -> Float {
        return a.storage.withUnsafeBufferPointer { aBuffer in
            return b.storage.withUnsafeBufferPointer { bBuffer in
                guard let aPtr = aBuffer.baseAddress, let bPtr = bBuffer.baseAddress else { return 0.0 }
                return euclideanFP16Core(
                    aPtr: aPtr,
                    bPtr: bPtr,
                    count: Vector768FP16.dimension
                )
            }
        }
    }

    @inlinable
    public static func euclidean1536(_ a: Vector1536FP16, _ b: Vector1536FP16) -> Float {
        return a.storage.withUnsafeBufferPointer { aBuffer in
            return b.storage.withUnsafeBufferPointer { bBuffer in
                guard let aPtr = aBuffer.baseAddress, let bPtr = bBuffer.baseAddress else { return 0.0 }
                return euclideanFP16Core(
                    aPtr: aPtr,
                    bPtr: bPtr,
                    count: Vector1536FP16.dimension
                )
            }
        }
    }

    // MARK: - Public API: Mixed Precision Euclidean Distance

    /// Mixed precision Euclidean distance: FP32 query × FP16 candidate.
    ///
    /// Preserves full query precision (the FP32 query is not converted) with FP16 candidate storage.
    @inlinable
    public static func euclidean512(
        query: Vector512Optimized,
        candidate: Vector512FP16
    ) -> Float {
        return candidate.storage.withUnsafeBufferPointer { cBuffer in
            guard let cPtr = cBuffer.baseAddress else { return 0.0 }
            return euclideanMixedCore(
                queryStorage: query.storage,
                candidatePtr: cPtr,
                lanes: Vector512Optimized.lanes
            )
        }
    }

    @inlinable
    public static func euclidean768(
        query: Vector768Optimized,
        candidate: Vector768FP16
    ) -> Float {
        return candidate.storage.withUnsafeBufferPointer { cBuffer in
            guard let cPtr = cBuffer.baseAddress else { return 0.0 }
            return euclideanMixedCore(
                queryStorage: query.storage,
                candidatePtr: cPtr,
                lanes: Vector768Optimized.lanes
            )
        }
    }

    @inlinable
    public static func euclidean1536(
        query: Vector1536Optimized,
        candidate: Vector1536FP16
    ) -> Float {
        return candidate.storage.withUnsafeBufferPointer { cBuffer in
            guard let cPtr = cBuffer.baseAddress else { return 0.0 }
            return euclideanMixedCore(
                queryStorage: query.storage,
                candidatePtr: cPtr,
                lanes: Vector1536Optimized.lanes
            )
        }
    }

    /// Mixed precision Euclidean distance: FP16 query × FP32 candidate.
    @inlinable
    public static func euclidean512(
        query: Vector512FP16,
        candidate: Vector512Optimized
    ) -> Float {
        // No conversion here: Euclidean distance is symmetric, so the FP32 candidate is passed as the FP32 operand
        // and the FP16 query is converted block by block inside euclideanMixedCore.
        return query.storage.withUnsafeBufferPointer { qBuffer in
            guard let qPtr = qBuffer.baseAddress else { return 0.0 }
            return euclideanMixedCore(
                queryStorage: candidate.storage,
                candidatePtr: qPtr,
                lanes: Vector512Optimized.lanes
            )
        }
    }

    @inlinable
    public static func euclidean768(
        query: Vector768FP16,
        candidate: Vector768Optimized
    ) -> Float {
        return query.storage.withUnsafeBufferPointer { qBuffer in
            guard let qPtr = qBuffer.baseAddress else { return 0.0 }
            return euclideanMixedCore(
                queryStorage: candidate.storage,
                candidatePtr: qPtr,
                lanes: Vector768Optimized.lanes
            )
        }
    }

    @inlinable
    public static func euclidean1536(
        query: Vector1536FP16,
        candidate: Vector1536Optimized
    ) -> Float {
        return query.storage.withUnsafeBufferPointer { qBuffer in
            guard let qPtr = qBuffer.baseAddress else { return 0.0 }
            return euclideanMixedCore(
                queryStorage: candidate.storage,
                candidatePtr: qPtr,
                lanes: Vector1536Optimized.lanes
            )
        }
    }

    // MARK: - Cosine Distance Kernels

    /// Core FP16 Cosine distance with FP32 accumulation.
    ///
    /// Computes 1 - (dot product / (||a|| * ||b||)), with the similarity clamped to [-1, 1], when the
    /// denominator sqrt(||a||² * ||b||²) is finite and greater than 1e-9. In exact arithmetic identical
    /// vectors then give 0 and opposite vectors give 2; with rounding an identical pair can give a small
    /// positive distance.
    ///
    /// Otherwise (the branch at the end of the body) it returns `Float.nan` when either squared norm is not
    /// finite (an FP16 infinity or NaN, or a sum that overflows) and 1 in every other case: two zero vectors, one
    /// zero vector, and identical nonzero vectors with ||a|| at or below about 3.2e-5 (||a||² ≤ 1e-9)
    /// (T15CosineExceptionalTableTests.everyImplementationFollowsThePolicy, fp16 rows).
    @inline(__always)
    @usableFromInline
    internal static func cosineFP16Core(
        aPtr: UnsafePointer<UInt16>,
        bPtr: UnsafePointer<UInt16>,
        count: Int
    ) -> Float {
        var dotAcc = SIMD4<Float>.zero
        var aMagAcc = SIMD4<Float>.zero
        var bMagAcc = SIMD4<Float>.zero

        var i = 0
        while i + 8 <= count {
            let a_fp16 = UnsafeRawPointer(aPtr.advanced(by: i))
                .loadUnaligned(as: SIMD8<UInt16>.self)
            let b_fp16 = UnsafeRawPointer(bPtr.advanced(by: i))
                .loadUnaligned(as: SIMD8<UInt16>.self)

            let a_fp32 = fp16ToFp32_simd8(a_fp16)
            let b_fp32 = fp16ToFp32_simd8(b_fp16)

            dotAcc.addProduct(a_fp32.lowHalf, b_fp32.lowHalf)
            dotAcc.addProduct(a_fp32.highHalf, b_fp32.highHalf)

            aMagAcc.addProduct(a_fp32.lowHalf, a_fp32.lowHalf)
            aMagAcc.addProduct(a_fp32.highHalf, a_fp32.highHalf)

            bMagAcc.addProduct(b_fp32.lowHalf, b_fp32.lowHalf)
            bMagAcc.addProduct(b_fp32.highHalf, b_fp32.highHalf)

            i += 8
        }

        // Handle remaining elements
        while i < count {
            let a_fp32 = fp16ToFp32_scalar(aPtr[i])
            let b_fp32 = fp16ToFp32_scalar(bPtr[i])
            dotAcc[0] += a_fp32 * b_fp32
            aMagAcc[0] += a_fp32 * a_fp32
            bMagAcc[0] += b_fp32 * b_fp32
            i += 1
        }

        let dotProduct = dotAcc.sum()
        let aMagSq = aMagAcc.sum()
        let bMagSq = bMagAcc.sum()

        let magnitude = sqrt(aMagSq * bMagSq)

        if magnitude > 1e-9 && magnitude.isFinite {
            let similarity = dotProduct / magnitude
            return 1.0 - max(-1.0, min(1.0, similarity))
        } else {
            guard aMagSq.isFinite && bMagSq.isFinite else { return .nan }
            return 1.0
        }
    }

    /// Core Mixed Precision Cosine Distance (FP32 Query, FP16 Candidate).
    ///
    /// The branch at the end of the body follows `cosineFP16Core`: `Float.nan` when either squared norm is not
    /// finite, otherwise 1 (T15CosineExceptionalTableTests.everyImplementationFollowsThePolicy, fp16.q32c16 and
    /// fp16.q16c32 rows).
    ///
    /// Domain: ‖q‖² · ‖c‖² < Float.greatestFiniteMagnitude, narrower than the documented cosine domain
    /// ‖v‖² < Float.greatestFiniteMagnitude (Docs/Numerical_Behavior.md). The denominator is sqrt(‖q‖² · ‖c‖²), so
    /// when that product overflows the call takes the end branch and returns 1 for finite inputs in any direction,
    /// parallel ones included (adjacent code; not in the T15 table). FP16 × FP16 (`cosineFP16Core`) cannot reach
    /// this at widths up to 1536: each finite squared norm is at most 1536 · 65504² up to Float rounding, so their
    /// product stays far below Float.greatestFiniteMagnitude.
    @inline(__always)
    @usableFromInline
    internal static func cosineMixedCore(
        queryStorage: ContiguousArray<SIMD4<Float>>,
        candidatePtr: UnsafePointer<UInt16>,
        lanes: Int
    ) -> Float {
        var dotAcc = SIMD4<Float>.zero
        var qMagAcc = SIMD4<Float>.zero
        var cMagAcc = SIMD4<Float>.zero

        var i = 0
        while i + 2 <= lanes {
            let q0 = queryStorage[i]
            let q1 = queryStorage[i+1]

            let offset = i * 4

            let c_fp16 = UnsafeRawPointer(candidatePtr.advanced(by: offset))
                .loadUnaligned(as: SIMD8<UInt16>.self)
            let c_fp32 = fp16ToFp32_simd8(c_fp16)

            dotAcc.addProduct(q0, c_fp32.lowHalf)
            dotAcc.addProduct(q1, c_fp32.highHalf)

            qMagAcc.addProduct(q0, q0)
            qMagAcc.addProduct(q1, q1)

            cMagAcc.addProduct(c_fp32.lowHalf, c_fp32.lowHalf)
            cMagAcc.addProduct(c_fp32.highHalf, c_fp32.highHalf)

            i += 2
        }

        // Handle remaining lanes
        while i < lanes {
            let q = queryStorage[i]
            let offset = i * 4

            let c0 = fp16ToFp32_scalar(candidatePtr[offset])
            let c1 = fp16ToFp32_scalar(candidatePtr[offset + 1])
            let c2 = fp16ToFp32_scalar(candidatePtr[offset + 2])
            let c3 = fp16ToFp32_scalar(candidatePtr[offset + 3])

            let c = SIMD4<Float>(c0, c1, c2, c3)

            dotAcc.addProduct(q, c)
            qMagAcc.addProduct(q, q)
            cMagAcc.addProduct(c, c)

            i += 1
        }

        let dotProduct = dotAcc.sum()
        let qMagSq = qMagAcc.sum()
        let cMagSq = cMagAcc.sum()

        let magnitude = sqrt(qMagSq * cMagSq)

        if magnitude > 1e-9 && magnitude.isFinite {
            let similarity = dotProduct / magnitude
            return 1.0 - max(-1.0, min(1.0, similarity))
        } else {
            guard qMagSq.isFinite && cMagSq.isFinite else { return .nan }
            return 1.0
        }
    }

    // MARK: - Public API: FP16 Cosine Distance

    /// Cosine distance with both vectors in FP16 storage.
    @inlinable
    public static func cosine512(_ a: Vector512FP16, _ b: Vector512FP16) -> Float {
        return a.storage.withUnsafeBufferPointer { aBuffer in
            return b.storage.withUnsafeBufferPointer { bBuffer in
                guard let aPtr = aBuffer.baseAddress, let bPtr = bBuffer.baseAddress else { return 1.0 }
                return cosineFP16Core(
                    aPtr: aPtr,
                    bPtr: bPtr,
                    count: Vector512FP16.dimension
                )
            }
        }
    }

    @inlinable
    public static func cosine768(_ a: Vector768FP16, _ b: Vector768FP16) -> Float {
        return a.storage.withUnsafeBufferPointer { aBuffer in
            return b.storage.withUnsafeBufferPointer { bBuffer in
                guard let aPtr = aBuffer.baseAddress, let bPtr = bBuffer.baseAddress else { return 1.0 }
                return cosineFP16Core(
                    aPtr: aPtr,
                    bPtr: bPtr,
                    count: Vector768FP16.dimension
                )
            }
        }
    }

    @inlinable
    public static func cosine1536(_ a: Vector1536FP16, _ b: Vector1536FP16) -> Float {
        return a.storage.withUnsafeBufferPointer { aBuffer in
            return b.storage.withUnsafeBufferPointer { bBuffer in
                guard let aPtr = aBuffer.baseAddress, let bPtr = bBuffer.baseAddress else { return 1.0 }
                return cosineFP16Core(
                    aPtr: aPtr,
                    bPtr: bPtr,
                    count: Vector1536FP16.dimension
                )
            }
        }
    }

    // MARK: - Public API: Mixed Precision Cosine Distance

    /// Mixed precision cosine distance: FP32 query × FP16 candidate.
    @inlinable
    public static func cosine512(
        query: Vector512Optimized,
        candidate: Vector512FP16
    ) -> Float {
        return candidate.storage.withUnsafeBufferPointer { cBuffer in
            guard let cPtr = cBuffer.baseAddress else { return 1.0 }
            return cosineMixedCore(
                queryStorage: query.storage,
                candidatePtr: cPtr,
                lanes: Vector512Optimized.lanes
            )
        }
    }

    @inlinable
    public static func cosine768(
        query: Vector768Optimized,
        candidate: Vector768FP16
    ) -> Float {
        return candidate.storage.withUnsafeBufferPointer { cBuffer in
            guard let cPtr = cBuffer.baseAddress else { return 1.0 }
            return cosineMixedCore(
                queryStorage: query.storage,
                candidatePtr: cPtr,
                lanes: Vector768Optimized.lanes
            )
        }
    }

    @inlinable
    public static func cosine1536(
        query: Vector1536Optimized,
        candidate: Vector1536FP16
    ) -> Float {
        return candidate.storage.withUnsafeBufferPointer { cBuffer in
            guard let cPtr = cBuffer.baseAddress else { return 1.0 }
            return cosineMixedCore(
                queryStorage: query.storage,
                candidatePtr: cPtr,
                lanes: Vector1536Optimized.lanes
            )
        }
    }

    /// Mixed precision cosine distance: FP16 query × FP32 candidate.
    @inlinable
    public static func cosine512(
        query: Vector512FP16,
        candidate: Vector512Optimized
    ) -> Float {
        return query.storage.withUnsafeBufferPointer { qBuffer in
            guard let qPtr = qBuffer.baseAddress else { return 1.0 }
            return cosineMixedCore(
                queryStorage: candidate.storage,
                candidatePtr: qPtr,
                lanes: Vector512Optimized.lanes
            )
        }
    }

    @inlinable
    public static func cosine768(
        query: Vector768FP16,
        candidate: Vector768Optimized
    ) -> Float {
        return query.storage.withUnsafeBufferPointer { qBuffer in
            guard let qPtr = qBuffer.baseAddress else { return 1.0 }
            return cosineMixedCore(
                queryStorage: candidate.storage,
                candidatePtr: qPtr,
                lanes: Vector768Optimized.lanes
            )
        }
    }

    @inlinable
    public static func cosine1536(
        query: Vector1536FP16,
        candidate: Vector1536Optimized
    ) -> Float {
        return query.storage.withUnsafeBufferPointer { qBuffer in
            guard let qPtr = qBuffer.baseAddress else { return 1.0 }
            return cosineMixedCore(
                queryStorage: candidate.storage,
                candidatePtr: qPtr,
                lanes: Vector1536Optimized.lanes
            )
        }
    }

    // MARK: - Batch Operations (Blocked)

    /// Helper for the blocked FP16×FP16 batch loop (blocks of 64 candidates).
    ///
    /// Strategy: convert the FP32 query to FP16 once, then use the FP16 × FP16 kernel. The query then carries
    /// FP16 rounding as well; whether that changes a caller's results is not verified here.
    @inline(__always)
    @usableFromInline
    internal static func batchDotFP16_blocked<V_FP32, V_FP16>(
        query: V_FP32,
        candidates: [V_FP16],
        out: UnsafeMutableBufferPointer<Float>,
        dotFunction: (V_FP16, V_FP16) -> Float,
        fp16Converter: (V_FP32) -> V_FP16
    ) {
        let N = candidates.count
        guard N > 0 else { return }

        guard out.count >= N else {
            fatalError("MixedPrecision batch output capacity must cover the candidate count")
        }

        let blockSize = 64  // 64 candidates per block (64 x D x 2 bytes of FP16 payload); not tuned by a measurement

        // Convert the FP32 query to FP16 once, outside the candidate loop.
        let queryFP16 = fp16Converter(query)

        for blockStart in stride(from: 0, to: N, by: blockSize) {
            let blockEnd = min(blockStart + blockSize, N)

            // Discarded read of the next block's first candidate handle; not a prefetch of its payload or instruction
            if blockEnd < N {
                _ = candidates[blockEnd]
            }

            // Process current block
            for i in blockStart..<blockEnd {
                // Utilize the specialized dot function (e.g., dotFP16_512)
                out[i] = dotFunction(queryFP16, candidates[i])
            }
        }
    }

    /// Helper function for mixed-precision batch operations preserving query precision.
    ///
    /// **Recommended over batchDotFP16_blocked for similarity search**: Keeps query in FP32.
    /// Ideal when query is computed once and reused across many candidates.
    @inline(__always)
    @usableFromInline
    internal static func batchDotMixed_blocked<V_FP32, V_FP16>(
        query: V_FP32,
        candidates: [V_FP16],
        out: UnsafeMutableBufferPointer<Float>,
        dotMixedFunction: (V_FP32, V_FP16) -> Float
    ) {
        let N = candidates.count
        guard N > 0 else { return }

        guard out.count >= N else {
            fatalError("MixedPrecision batch output capacity must cover the candidate count")
        }

        let blockSize = 64  // 64 candidates per block; not tuned by a measurement

        for blockStart in stride(from: 0, to: N, by: blockSize) {
            let blockEnd = min(blockStart + blockSize, N)

            // Discarded read of the next block's first candidate handle; not a prefetch
            if blockEnd < N {
                _ = candidates[blockEnd]
            }

            // Process current block with mixed precision (FP32 query, FP16 candidates)
            for i in blockStart..<blockEnd {
                out[i] = dotMixedFunction(query, candidates[i])
            }
        }
    }

    // MARK: - Batch FP16×FP16 (Query converted to FP16)

    @inlinable
    public static func batchDotFP16_512(
        query: Vector512Optimized,
        candidates: [Vector512FP16],
        out: UnsafeMutableBufferPointer<Float>
    ) {
        batchDotFP16_blocked(
            query: query,
            candidates: candidates,
            out: out,
            dotFunction: dotFP16_512,
            fp16Converter: Vector512FP16.init(from:)
        )
    }

    @inlinable
    public static func batchDotFP16_768(
        query: Vector768Optimized,
        candidates: [Vector768FP16],
        out: UnsafeMutableBufferPointer<Float>
    ) {
        batchDotFP16_blocked(
            query: query,
            candidates: candidates,
            out: out,
            dotFunction: dotFP16_768,
            fp16Converter: Vector768FP16.init(from:)
        )
    }

    @inlinable
    public static func batchDotFP16_1536(
        query: Vector1536Optimized,
        candidates: [Vector1536FP16],
        out: UnsafeMutableBufferPointer<Float>
    ) {
        batchDotFP16_blocked(
            query: query,
            candidates: candidates,
            out: out,
            dotFunction: dotFP16_1536,
            fp16Converter: Vector1536FP16.init(from:)
        )
    }

    // MARK: - Batch Mixed Precision (FP32 Query, FP16 Candidates)

    /// Batch dot product with FP32 query and FP16 candidates (preserves query precision).
    ///
    /// Structure:
    /// - Query remains in FP32 (not converted)
    /// - Candidates in FP16 (half the scalar payload; payload model only)
    @inlinable
    public static func batchDotMixed512(
        query: Vector512Optimized,
        candidates: [Vector512FP16],
        out: UnsafeMutableBufferPointer<Float>
    ) {
        batchDotMixed_blocked(
            query: query,
            candidates: candidates,
            out: out,
            dotMixedFunction: dotMixed512
        )
    }

    @inlinable
    public static func batchDotMixed768(
        query: Vector768Optimized,
        candidates: [Vector768FP16],
        out: UnsafeMutableBufferPointer<Float>
    ) {
        batchDotMixed_blocked(
            query: query,
            candidates: candidates,
            out: out,
            dotMixedFunction: dotMixed768
        )
    }

    @inlinable
    public static func batchDotMixed1536(
        query: Vector1536Optimized,
        candidates: [Vector1536FP16],
        out: UnsafeMutableBufferPointer<Float>
    ) {
        batchDotMixed_blocked(
            query: query,
            candidates: candidates,
            out: out,
            dotMixedFunction: dotMixed1536
        )
    }

    // MARK: - SoA Batch Processing Kernels

    /// Batch Euclidean Distance using SoA layout (FP16 query × FP16 SoA candidates).
    ///
    /// Walks groups of 4 candidates dimension by dimension, converting each FP16 value with the scalar
    /// software conversion. No speed or throughput figure is measured.
    ///
    /// - Parameters:
    ///   - query: FP16 query vector
    ///   - candidates: SoA-layout FP16 candidate vectors
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    @inlinable
    public static func batchEuclidean512(
        query: Vector512FP16,
        candidates: SoA512FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }
        #if DEBUG
        assert(candidates.dimension == 512, "Dimension mismatch")
        #endif

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP16 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                var storageIndex = 0
                for candidateGroup in stride(from: 0, to: vectorCount, by: 4) {
                    let actualCount = min(4, vectorCount - candidateGroup)

                    // One SIMD4 accumulator holds the 4 candidates of the group
                    var accumulators = SIMD4<Float>.zero

                    // Process dimensions in blocks of 4
                    var d = 0
                    while d + 3 < 512 {
                        // Process 4 dimensions at once
                        for offset in 0..<4 {
                            let q = fp16ToFp32_scalar(queryFP16[d + offset])

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[storageIndex + 0]),
                                fp16ToFp32_scalar(candidatesFP16[storageIndex + 1]),
                                fp16ToFp32_scalar(candidatesFP16[storageIndex + 2]),
                                fp16ToFp32_scalar(candidatesFP16[storageIndex + 3])
                            )

                            let diff = SIMD4<Float>(repeating: q) - candidates
                            accumulators.addProduct(diff, diff)

                            storageIndex += 4
                        }
                        d += 4
                    }

                    // Handle tail dimensions if any
                    while d < 512 {
                        let q = fp16ToFp32_scalar(queryFP16[d])

                        let candidates = SIMD4<Float>(
                            fp16ToFp32_scalar(candidatesFP16[storageIndex + 0]),
                            fp16ToFp32_scalar(candidatesFP16[storageIndex + 1]),
                            fp16ToFp32_scalar(candidatesFP16[storageIndex + 2]),
                            fp16ToFp32_scalar(candidatesFP16[storageIndex + 3])
                        )

                        let diff = SIMD4<Float>(repeating: q) - candidates
                        accumulators.addProduct(diff, diff)

                        storageIndex += 4
                        d += 1
                    }

                    // Store results with sqrt
                    for i in 0..<actualCount {
                        results[candidateGroup + i] = sqrt(accumulators[i])
                    }
                }
            }
        }
    }

    /// Batch Euclidean Distance using SoA layout (FP16 query × FP16 SoA candidates).
    ///
    /// Walks groups of 4 candidates dimension by dimension, converting each FP16 value with the scalar
    /// software conversion. No speed or throughput figure is measured.
    ///
    /// - Parameters:
    ///   - query: FP16 query vector
    ///   - candidates: SoA-layout FP16 candidate vectors
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    @inlinable
    public static func batchEuclidean768(
        query: Vector768FP16,
        candidates: SoA768FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }
        #if DEBUG
        assert(candidates.dimension == 768, "Dimension mismatch")
        #endif

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP16 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                var storageIndex = 0
                for candidateGroup in stride(from: 0, to: vectorCount, by: 4) {
                    let actualCount = min(4, vectorCount - candidateGroup)

                    // One SIMD4 accumulator holds the 4 candidates of the group
                    var accumulators = SIMD4<Float>.zero

                    // Process dimensions in blocks of 4
                    var d = 0
                    while d + 3 < 768 {
                        // Process 4 dimensions at once
                        for offset in 0..<4 {
                            let q = fp16ToFp32_scalar(queryFP16[d + offset])

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[storageIndex + 0]),
                                fp16ToFp32_scalar(candidatesFP16[storageIndex + 1]),
                                fp16ToFp32_scalar(candidatesFP16[storageIndex + 2]),
                                fp16ToFp32_scalar(candidatesFP16[storageIndex + 3])
                            )

                            let diff = SIMD4<Float>(repeating: q) - candidates
                            accumulators.addProduct(diff, diff)

                            storageIndex += 4
                        }
                        d += 4
                    }

                    // Handle tail dimensions if any
                    while d < 768 {
                        let q = fp16ToFp32_scalar(queryFP16[d])

                        let candidates = SIMD4<Float>(
                            fp16ToFp32_scalar(candidatesFP16[storageIndex + 0]),
                            fp16ToFp32_scalar(candidatesFP16[storageIndex + 1]),
                            fp16ToFp32_scalar(candidatesFP16[storageIndex + 2]),
                            fp16ToFp32_scalar(candidatesFP16[storageIndex + 3])
                        )

                        let diff = SIMD4<Float>(repeating: q) - candidates
                        accumulators.addProduct(diff, diff)

                        storageIndex += 4
                        d += 1
                    }

                    // Store results with sqrt
                    for i in 0..<actualCount {
                        results[candidateGroup + i] = sqrt(accumulators[i])
                    }
                }
            }
        }
    }

    /// Batch Euclidean Distance (FP32 query × FP16 SoA candidates).
    ///
    /// Preserves full query precision (the FP32 query is not converted).
    ///
    /// **Structure:**
    /// - One SIMD4 accumulator per group of 4 candidates
    /// - The lane loop is unrolled by 2
    /// - Candidate values are converted with the scalar software conversion
    ///
    /// - Parameters:
    ///   - query: FP32 query vector
    ///   - candidates: SoA-layout FP16 candidate vectors
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    @inlinable
    public static func batchEuclidean512(
        query: Vector512Optimized,
        candidates: SoA512FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP32 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                var storageIndex = 0
                for candidateGroup in stride(from: 0, to: vectorCount, by: 4) {
                    let actualCount = min(4, vectorCount - candidateGroup)

                    // One SIMD4 accumulator per group of 4 candidates
                    var accumulators = SIMD4<Float>.zero

                    // Process 128 SIMD4 lanes with 2× unrolling
                    var lane = 0
                    while lane + 1 < 128 {
                        // Lane 0
                        let queryLane0 = queryFP32[lane]
                        for d in 0..<4 {
                            let q = queryLane0[d]
                            let idx = storageIndex + d * 4

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 3])
                            )

                            let diff = SIMD4<Float>(repeating: q) - candidates
                            accumulators.addProduct(diff, diff)
                        }
                        storageIndex += 16

                        // Lane 1 (unrolled)
                        let queryLane1 = queryFP32[lane + 1]
                        for d in 0..<4 {
                            let q = queryLane1[d]
                            let idx = storageIndex + d * 4

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 3])
                            )

                            let diff = SIMD4<Float>(repeating: q) - candidates
                            accumulators.addProduct(diff, diff)
                        }
                        storageIndex += 16
                        lane += 2
                    }

                    // Handle tail lane if odd count
                    if lane < 128 {
                        let queryLane = queryFP32[lane]
                        for d in 0..<4 {
                            let q = queryLane[d]
                            let idx = storageIndex + d * 4

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 3])
                            )

                            let diff = SIMD4<Float>(repeating: q) - candidates
                            accumulators.addProduct(diff, diff)
                        }
                        storageIndex += 16
                    }

                    // Store results with sqrt
                    for i in 0..<actualCount {
                        results[candidateGroup + i] = sqrt(accumulators[i])
                    }
                }
            }
        }
    }

    /// Batch Euclidean Distance (FP32 query × FP16 SoA candidates).
    ///
    /// Preserves full query precision (the FP32 query is not converted).
    ///
    /// **Structure:**
    /// - One SIMD4 accumulator per group of 4 candidates
    /// - The lane loop is unrolled by 2
    /// - Candidate values are converted with the scalar software conversion
    ///
    /// - Parameters:
    ///   - query: FP32 query vector
    ///   - candidates: SoA-layout FP16 candidate vectors
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    @inlinable
    public static func batchEuclidean768(
        query: Vector768Optimized,
        candidates: SoA768FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP32 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                var storageIndex = 0
                for candidateGroup in stride(from: 0, to: vectorCount, by: 4) {
                    let actualCount = min(4, vectorCount - candidateGroup)

                    // One SIMD4 accumulator per group of 4 candidates
                    var accumulators = SIMD4<Float>.zero

                    // Process 192 SIMD4 lanes with 2× unrolling
                    var lane = 0
                    while lane + 1 < 192 {
                        // Lane 0
                        let queryLane0 = queryFP32[lane]
                        for d in 0..<4 {
                            let q = queryLane0[d]
                            let idx = storageIndex + d * 4

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 3])
                            )

                            let diff = SIMD4<Float>(repeating: q) - candidates
                            accumulators.addProduct(diff, diff)
                        }
                        storageIndex += 16

                        // Lane 1 (unrolled)
                        let queryLane1 = queryFP32[lane + 1]
                        for d in 0..<4 {
                            let q = queryLane1[d]
                            let idx = storageIndex + d * 4

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 3])
                            )

                            let diff = SIMD4<Float>(repeating: q) - candidates
                            accumulators.addProduct(diff, diff)
                        }
                        storageIndex += 16
                        lane += 2
                    }

                    // Handle tail lane if odd count
                    if lane < 192 {
                        let queryLane = queryFP32[lane]
                        for d in 0..<4 {
                            let q = queryLane[d]
                            let idx = storageIndex + d * 4

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 3])
                            )

                            let diff = SIMD4<Float>(repeating: q) - candidates
                            accumulators.addProduct(diff, diff)
                        }
                        storageIndex += 16
                    }

                    // Store results with sqrt
                    for i in 0..<actualCount {
                        results[candidateGroup + i] = sqrt(accumulators[i])
                    }
                }
            }
        }
    }

    /// Batch Euclidean Distance using SoA layout (FP16 query × FP16 SoA candidates).
    ///
    /// Walks groups of 4 candidates dimension by dimension, converting each FP16 value with the scalar
    /// software conversion. No speed or throughput figure is measured.
    ///
    /// - Parameters:
    ///   - query: FP16 query vector
    ///   - candidates: SoA-layout FP16 candidate vectors
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    @inlinable
    public static func batchEuclidean1536(
        query: Vector1536FP16,
        candidates: SoA1536FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }
        #if DEBUG
        assert(candidates.dimension == 1536, "Dimension mismatch")
        #endif

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP16 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                var storageIndex = 0
                for candidateGroup in stride(from: 0, to: vectorCount, by: 4) {
                    let actualCount = min(4, vectorCount - candidateGroup)

                    // One SIMD4 accumulator holds the 4 candidates of the group
                    var accumulators = SIMD4<Float>.zero

                    // Process dimensions in blocks of 4
                    var d = 0
                    while d + 3 < 1536 {
                        // Process 4 dimensions at once
                        for offset in 0..<4 {
                            let q = fp16ToFp32_scalar(queryFP16[d + offset])

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[storageIndex + 0]),
                                fp16ToFp32_scalar(candidatesFP16[storageIndex + 1]),
                                fp16ToFp32_scalar(candidatesFP16[storageIndex + 2]),
                                fp16ToFp32_scalar(candidatesFP16[storageIndex + 3])
                            )

                            let diff = SIMD4<Float>(repeating: q) - candidates
                            accumulators.addProduct(diff, diff)

                            storageIndex += 4
                        }
                        d += 4
                    }

                    // Handle tail dimensions if any
                    while d < 1536 {
                        let q = fp16ToFp32_scalar(queryFP16[d])

                        let candidates = SIMD4<Float>(
                            fp16ToFp32_scalar(candidatesFP16[storageIndex + 0]),
                            fp16ToFp32_scalar(candidatesFP16[storageIndex + 1]),
                            fp16ToFp32_scalar(candidatesFP16[storageIndex + 2]),
                            fp16ToFp32_scalar(candidatesFP16[storageIndex + 3])
                        )

                        let diff = SIMD4<Float>(repeating: q) - candidates
                        accumulators.addProduct(diff, diff)

                        storageIndex += 4
                        d += 1
                    }

                    // Store results with sqrt
                    for i in 0..<actualCount {
                        results[candidateGroup + i] = sqrt(accumulators[i])
                    }
                }
            }
        }
    }

    /// Batch Euclidean Distance (FP32 query × FP16 SoA candidates).
    ///
    /// Preserves full query precision (the FP32 query is not converted).
    ///
    /// **Structure:**
    /// - One SIMD4 accumulator per group of 4 candidates
    /// - The lane loop is unrolled by 2
    /// - Candidate values are converted with the scalar software conversion
    ///
    /// - Parameters:
    ///   - query: FP32 query vector
    ///   - candidates: SoA-layout FP16 candidate vectors
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    @inlinable
    public static func batchEuclidean1536(
        query: Vector1536Optimized,
        candidates: SoA1536FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP32 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                var storageIndex = 0
                for candidateGroup in stride(from: 0, to: vectorCount, by: 4) {
                    let actualCount = min(4, vectorCount - candidateGroup)

                    // One SIMD4 accumulator per group of 4 candidates
                    var accumulators = SIMD4<Float>.zero

                    // Process 384 SIMD4 lanes with 2× unrolling
                    var lane = 0
                    while lane + 1 < 384 {
                        // Lane 0
                        let queryLane0 = queryFP32[lane]
                        for d in 0..<4 {
                            let q = queryLane0[d]
                            let idx = storageIndex + d * 4

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 3])
                            )

                            let diff = SIMD4<Float>(repeating: q) - candidates
                            accumulators.addProduct(diff, diff)
                        }
                        storageIndex += 16

                        // Lane 1 (unrolled)
                        let queryLane1 = queryFP32[lane + 1]
                        for d in 0..<4 {
                            let q = queryLane1[d]
                            let idx = storageIndex + d * 4

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 3])
                            )

                            let diff = SIMD4<Float>(repeating: q) - candidates
                            accumulators.addProduct(diff, diff)
                        }
                        storageIndex += 16
                        lane += 2
                    }

                    // Handle tail lane if odd count
                    if lane < 384 {
                        let queryLane = queryFP32[lane]
                        for d in 0..<4 {
                            let q = queryLane[d]
                            let idx = storageIndex + d * 4

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 3])
                            )

                            let diff = SIMD4<Float>(repeating: q) - candidates
                            accumulators.addProduct(diff, diff)
                        }
                        storageIndex += 16
                    }

                    // Store results with sqrt
                    for i in 0..<actualCount {
                        results[candidateGroup + i] = sqrt(accumulators[i])
                    }
                }
            }
        }
    }

    // MARK: - Batch Dot Product (FP32 query × FP16 SoA candidates)

    /// Batch Dot Product with FP32 query and FP16 SoA candidates (512D).
    ///
    /// Computes dot products between a single FP32 query vector and multiple FP16 candidates
    /// stored in Structure-of-Arrays (SoA) layout.
    ///
    /// **Algorithm**:
    /// For each candidate i: result[i] = Σ(query[j] * candidate[i][j]) for all dimensions j
    ///
    /// **Structure**: one SIMD4 accumulator processes the 4 candidates of a group together. No speed figure is measured.
    ///
    /// **Accuracy**: candidates carry FP16 rounding; no relative-error bound is promised (see the type's
    /// error-analysis notes: cancellation makes the relative error unbounded).
    ///
    /// - Parameters:
    ///   - query: FP32 query vector (512 dimensions)
    ///   - candidates: FP16 candidates in SoA layout
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    ///
    /// - Complexity: O(N × D) where N = candidates, D = 512
    @inlinable
    public static func batchDotProductSoA(
        query: Vector512Optimized,
        candidates: SoA512FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP32 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                var storageIndex = 0
                for candidateGroup in stride(from: 0, to: vectorCount, by: 4) {
                    let actualCount = min(4, vectorCount - candidateGroup)

                    // One SIMD4 accumulator per group of 4 candidates
                    var accumulators = SIMD4<Float>.zero

                    // Process 128 SIMD4 lanes with 2× unrolling
                    var lane = 0
                    while lane + 1 < 128 {
                        // Lane 0
                        let queryLane0 = queryFP32[lane]
                        for d in 0..<4 {
                            let q = queryLane0[d]
                            let idx = storageIndex + d * 4

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 3])
                            )

                            // Dot product: accumulate q * c (no difference, no sqrt)
                            accumulators.addProduct(SIMD4<Float>(repeating: q), candidates)
                        }
                        storageIndex += 16

                        // Lane 1 (unrolled)
                        let queryLane1 = queryFP32[lane + 1]
                        for d in 0..<4 {
                            let q = queryLane1[d]
                            let idx = storageIndex + d * 4

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 3])
                            )

                            // Dot product: accumulate q * c
                            accumulators.addProduct(SIMD4<Float>(repeating: q), candidates)
                        }
                        storageIndex += 16
                        lane += 2
                    }

                    // Handle tail lane if odd count
                    if lane < 128 {
                        let queryLane = queryFP32[lane]
                        for d in 0..<4 {
                            let q = queryLane[d]
                            let idx = storageIndex + d * 4

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 3])
                            )

                            // Dot product: accumulate q * c
                            accumulators.addProduct(SIMD4<Float>(repeating: q), candidates)
                        }
                        storageIndex += 16
                    }

                    // Store results (no sqrt for dot product)
                    for i in 0..<actualCount {
                        results[candidateGroup + i] = accumulators[i]
                    }
                }
            }
        }
    }

    /// Batch Dot Product with FP32 query and FP16 SoA candidates (768D).
    ///
    /// Computes dot products between a single FP32 query vector and multiple FP16 candidates
    /// stored in Structure-of-Arrays (SoA) layout.
    ///
    /// **Algorithm**:
    /// For each candidate i: result[i] = Σ(query[j] * candidate[i][j]) for all dimensions j
    ///
    /// **Structure**: one SIMD4 accumulator processes the 4 candidates of a group together. No speed figure is measured.
    ///
    /// **Accuracy**: candidates carry FP16 rounding; no relative-error bound is promised (see the type's
    /// error-analysis notes: cancellation makes the relative error unbounded).
    ///
    /// - Parameters:
    ///   - query: FP32 query vector (768 dimensions)
    ///   - candidates: FP16 candidates in SoA layout
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    ///
    /// - Complexity: O(N × D) where N = candidates, D = 768
    @inlinable
    public static func batchDotProductSoA(
        query: Vector768Optimized,
        candidates: SoA768FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }
        #if DEBUG
        assert(candidates.dimension == 768, "Dimension mismatch")
        #endif

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP32 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                var storageIndex = 0
                for candidateGroup in stride(from: 0, to: vectorCount, by: 4) {
                    let actualCount = min(4, vectorCount - candidateGroup)

                    // One SIMD4 accumulator holds the 4 candidates of the group
                    var accumulators = SIMD4<Float>.zero

                    // Process 192 SIMD4 lanes (768 / 4 = 192)
                    for lane in 0..<192 {
                        let queryLane = queryFP32[lane]
                        for d in 0..<4 {
                            let q = queryLane[d]
                            let idx = storageIndex + d * 4

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 3])
                            )

                            // Dot product: accumulate q * c
                            accumulators.addProduct(SIMD4<Float>(repeating: q), candidates)
                        }
                        storageIndex += 16
                    }

                    // Store results (no sqrt for dot product)
                    for i in 0..<actualCount {
                        results[candidateGroup + i] = accumulators[i]
                    }
                }
            }
        }
    }

    /// Batch Dot Product with FP32 query and FP16 SoA candidates (1536D).
    ///
    /// Computes dot products between a single FP32 query vector and multiple FP16 candidates
    /// stored in Structure-of-Arrays (SoA) layout.
    ///
    /// **Algorithm**:
    /// For each candidate i: result[i] = Σ(query[j] * candidate[i][j]) for all dimensions j
    ///
    /// **Structure**: one SIMD4 accumulator processes the 4 candidates of a group together. No speed figure is measured.
    ///
    /// **Accuracy**: candidates carry FP16 rounding; no relative-error bound is promised (see the type's
    /// error-analysis notes: cancellation makes the relative error unbounded).
    ///
    /// - Parameters:
    ///   - query: FP32 query vector (1536 dimensions)
    ///   - candidates: FP16 candidates in SoA layout
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    ///
    /// - Complexity: O(N × D) where N = candidates, D = 1536
    @inlinable
    public static func batchDotProductSoA(
        query: Vector1536Optimized,
        candidates: SoA1536FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }
        #if DEBUG
        assert(candidates.dimension == 1536, "Dimension mismatch")
        #endif

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP32 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                var storageIndex = 0
                for candidateGroup in stride(from: 0, to: vectorCount, by: 4) {
                    let actualCount = min(4, vectorCount - candidateGroup)

                    // One SIMD4 accumulator holds the 4 candidates of the group
                    var accumulators = SIMD4<Float>.zero

                    // Process 384 SIMD4 lanes (1536 / 4 = 384)
                    for lane in 0..<384 {
                        let queryLane = queryFP32[lane]
                        for d in 0..<4 {
                            let q = queryLane[d]
                            let idx = storageIndex + d * 4

                            let candidates = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx + 3])
                            )

                            // Dot product: accumulate q * c
                            accumulators.addProduct(SIMD4<Float>(repeating: q), candidates)
                        }
                        storageIndex += 16
                    }

                    // Store results (no sqrt for dot product)
                    for i in 0..<actualCount {
                        results[candidateGroup + i] = accumulators[i]
                    }
                }
            }
        }
    }

    // MARK: - Register-Blocked Batch Processing

    /// Register-blocked Batch Euclidean Distance (processes 8 candidates simultaneously).
    ///
    /// **Structure:**
    /// - Processes 2 SoA groups (8 candidates) per iteration, halving the outer-loop iterations of the
    ///   one-group loop
    ///
    /// No speed ratio versus batchEuclidean is measured, and no candidate-count threshold is derived here.
    ///
    /// - Parameters:
    ///   - query: FP16 query vector
    ///   - candidates: SoA-layout FP16 candidate vectors
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    @inlinable
    public static func batchEuclideanBlocked512(
        query: Vector512FP16,
        candidates: SoA512FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }

        let blockSize = 8  // Process 8 candidates per iteration

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP16 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                // Process in blocks of 8 candidates
                for blockStart in stride(from: 0, to: vectorCount, by: blockSize) {
                    let blockCount = min(blockSize, vectorCount - blockStart)

                    // Dual SIMD4 accumulators for 8 candidates
                    var acc1 = SIMD4<Float>.zero  // Candidates 0-3
                    var acc2 = SIMD4<Float>.zero  // Candidates 4-7

                    // Calculate storage indices for both groups
                    let group1Start = (blockStart / 4) * 512 * 4
                    let group2Start = ((blockStart + 4) / 4) * 512 * 4

                    var idx1 = group1Start
                    var idx2 = group2Start

                    // Process all 512 dimensions with dual accumulators
                    for d in 0..<512 {
                        let q = fp16ToFp32_scalar(queryFP16[d])

                        // Group 1 (candidates 0-3)
                        let cand1 = SIMD4<Float>(
                            fp16ToFp32_scalar(candidatesFP16[idx1 + 0]),
                            fp16ToFp32_scalar(candidatesFP16[idx1 + 1]),
                            fp16ToFp32_scalar(candidatesFP16[idx1 + 2]),
                            fp16ToFp32_scalar(candidatesFP16[idx1 + 3])
                        )
                        let diff1 = SIMD4<Float>(repeating: q) - cand1
                        acc1.addProduct(diff1, diff1)
                        idx1 += 4

                        // Group 2 (candidates 4-7) - only if we have more than 4 candidates in block
                        if blockCount > 4 {
                            let cand2 = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx2 + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx2 + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx2 + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx2 + 3])
                            )
                            let diff2 = SIMD4<Float>(repeating: q) - cand2
                            acc2.addProduct(diff2, diff2)
                            idx2 += 4
                        }
                    }

                    // Store results for group 1 (candidates 0-3)
                    let count1 = min(4, blockCount)
                    for i in 0..<count1 {
                        results[blockStart + i] = sqrt(acc1[i])
                    }

                    // Store results for group 2 (candidates 4-7) if applicable
                    if blockCount > 4 {
                        let count2 = blockCount - 4
                        for i in 0..<count2 {
                            results[blockStart + 4 + i] = sqrt(acc2[i])
                        }
                    }
                }
            }
        }
    }

    /// Register-blocked Batch Euclidean Distance (FP32 query × FP16 candidates).
    ///
    /// **FP32 query + register blocking:**
    /// - The FP32 query is not converted
    /// - FP16 candidates (half the scalar payload; payload model only)
    /// - 8 candidates per iteration
    ///
    /// - Parameters:
    ///   - query: FP32 query vector
    ///   - candidates: SoA-layout FP16 candidate vectors
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    @inlinable
    public static func batchEuclideanBlocked512(
        query: Vector512Optimized,
        candidates: SoA512FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }

        let blockSize = 8

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP32 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                for blockStart in stride(from: 0, to: vectorCount, by: blockSize) {
                    let blockCount = min(blockSize, vectorCount - blockStart)

                    var acc1 = SIMD4<Float>.zero
                    var acc2 = SIMD4<Float>.zero

                    let group1Start = (blockStart / 4) * 512 * 4
                    let group2Start = ((blockStart + 4) / 4) * 512 * 4

                    var idx1 = group1Start
                    var idx2 = group2Start

                    // Process 128 SIMD4 lanes
                    for lane in 0..<128 {
                        let queryLane = queryFP32[lane]

                        // Process 4 dimensions per lane
                        for d in 0..<4 {
                            let q = queryLane[d]

                            // Group 1
                            let cand1 = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx1 + d * 4 + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx1 + d * 4 + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx1 + d * 4 + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx1 + d * 4 + 3])
                            )
                            let diff1 = SIMD4<Float>(repeating: q) - cand1
                            acc1.addProduct(diff1, diff1)

                            // Group 2
                            if blockCount > 4 {
                                let cand2 = SIMD4<Float>(
                                    fp16ToFp32_scalar(candidatesFP16[idx2 + d * 4 + 0]),
                                    fp16ToFp32_scalar(candidatesFP16[idx2 + d * 4 + 1]),
                                    fp16ToFp32_scalar(candidatesFP16[idx2 + d * 4 + 2]),
                                    fp16ToFp32_scalar(candidatesFP16[idx2 + d * 4 + 3])
                                )
                                let diff2 = SIMD4<Float>(repeating: q) - cand2
                                acc2.addProduct(diff2, diff2)
                            }
                        }

                        idx1 += 16
                        idx2 += 16
                    }

                    // Store results
                    let count1 = min(4, blockCount)
                    for i in 0..<count1 {
                        results[blockStart + i] = sqrt(acc1[i])
                    }

                    if blockCount > 4 {
                        let count2 = blockCount - 4
                        for i in 0..<count2 {
                            results[blockStart + 4 + i] = sqrt(acc2[i])
                        }
                    }
                }
            }
        }
    }

    /// Register-blocked Batch Euclidean Distance (processes 8 candidates simultaneously) - 768D.
    ///
    /// **Structure:**
    /// - Processes 2 SoA groups (8 candidates) per iteration, halving the outer-loop iterations of the
    ///   one-group loop
    ///
    /// No speed ratio versus batchEuclidean is measured, and no candidate-count threshold is derived here.
    ///
    /// - Parameters:
    ///   - query: FP16 query vector
    ///   - candidates: SoA-layout FP16 candidate vectors
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    @inlinable
    public static func batchEuclideanBlocked768(
        query: Vector768FP16,
        candidates: SoA768FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }

        let blockSize = 8  // Process 8 candidates per iteration

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP16 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                // Process in blocks of 8 candidates
                for blockStart in stride(from: 0, to: vectorCount, by: blockSize) {
                    let blockCount = min(blockSize, vectorCount - blockStart)

                    // Dual SIMD4 accumulators for 8 candidates
                    var acc1 = SIMD4<Float>.zero  // Candidates 0-3
                    var acc2 = SIMD4<Float>.zero  // Candidates 4-7

                    // Calculate storage indices for both groups
                    let group1Start = (blockStart / 4) * 768 * 4
                    let group2Start = ((blockStart + 4) / 4) * 768 * 4

                    var idx1 = group1Start
                    var idx2 = group2Start

                    // Process all 768 dimensions with dual accumulators
                    for d in 0..<768 {
                        let q = fp16ToFp32_scalar(queryFP16[d])

                        // Group 1 (candidates 0-3)
                        let cand1 = SIMD4<Float>(
                            fp16ToFp32_scalar(candidatesFP16[idx1 + 0]),
                            fp16ToFp32_scalar(candidatesFP16[idx1 + 1]),
                            fp16ToFp32_scalar(candidatesFP16[idx1 + 2]),
                            fp16ToFp32_scalar(candidatesFP16[idx1 + 3])
                        )
                        let diff1 = SIMD4<Float>(repeating: q) - cand1
                        acc1.addProduct(diff1, diff1)
                        idx1 += 4

                        // Group 2 (candidates 4-7) - only if we have more than 4 candidates in block
                        if blockCount > 4 {
                            let cand2 = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx2 + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx2 + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx2 + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx2 + 3])
                            )
                            let diff2 = SIMD4<Float>(repeating: q) - cand2
                            acc2.addProduct(diff2, diff2)
                            idx2 += 4
                        }
                    }

                    // Store results for group 1 (candidates 0-3)
                    let count1 = min(4, blockCount)
                    for i in 0..<count1 {
                        results[blockStart + i] = sqrt(acc1[i])
                    }

                    // Store results for group 2 (candidates 4-7) if applicable
                    if blockCount > 4 {
                        let count2 = blockCount - 4
                        for i in 0..<count2 {
                            results[blockStart + 4 + i] = sqrt(acc2[i])
                        }
                    }
                }
            }
        }
    }

    /// Register-blocked Batch Euclidean Distance (FP32 query × FP16 candidates) - 768D.
    ///
    /// **FP32 query + register blocking:**
    /// - The FP32 query is not converted
    /// - FP16 candidates (half the scalar payload; payload model only)
    /// - 8 candidates per iteration
    ///
    /// - Parameters:
    ///   - query: FP32 query vector
    ///   - candidates: SoA-layout FP16 candidate vectors
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    @inlinable
    public static func batchEuclideanBlocked768(
        query: Vector768Optimized,
        candidates: SoA768FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }

        let blockSize = 8

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP32 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                for blockStart in stride(from: 0, to: vectorCount, by: blockSize) {
                    let blockCount = min(blockSize, vectorCount - blockStart)

                    var acc1 = SIMD4<Float>.zero
                    var acc2 = SIMD4<Float>.zero

                    let group1Start = (blockStart / 4) * 768 * 4
                    let group2Start = ((blockStart + 4) / 4) * 768 * 4

                    var idx1 = group1Start
                    var idx2 = group2Start

                    // Process 192 SIMD4 lanes (768 dimensions / 4)
                    for lane in 0..<192 {
                        let queryLane = queryFP32[lane]

                        // Process 4 dimensions per lane
                        for d in 0..<4 {
                            let q = queryLane[d]

                            // Group 1
                            let cand1 = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx1 + d * 4 + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx1 + d * 4 + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx1 + d * 4 + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx1 + d * 4 + 3])
                            )
                            let diff1 = SIMD4<Float>(repeating: q) - cand1
                            acc1.addProduct(diff1, diff1)

                            // Group 2
                            if blockCount > 4 {
                                let cand2 = SIMD4<Float>(
                                    fp16ToFp32_scalar(candidatesFP16[idx2 + d * 4 + 0]),
                                    fp16ToFp32_scalar(candidatesFP16[idx2 + d * 4 + 1]),
                                    fp16ToFp32_scalar(candidatesFP16[idx2 + d * 4 + 2]),
                                    fp16ToFp32_scalar(candidatesFP16[idx2 + d * 4 + 3])
                                )
                                let diff2 = SIMD4<Float>(repeating: q) - cand2
                                acc2.addProduct(diff2, diff2)
                            }
                        }

                        idx1 += 16
                        idx2 += 16
                    }

                    // Store results
                    let count1 = min(4, blockCount)
                    for i in 0..<count1 {
                        results[blockStart + i] = sqrt(acc1[i])
                    }

                    if blockCount > 4 {
                        let count2 = blockCount - 4
                        for i in 0..<count2 {
                            results[blockStart + 4 + i] = sqrt(acc2[i])
                        }
                    }
                }
            }
        }
    }

    /// Register-blocked Batch Euclidean Distance (processes 8 candidates simultaneously) - 1536D.
    ///
    /// **Structure:**
    /// - Processes 2 SoA groups (8 candidates) per iteration, halving the outer-loop iterations of the
    ///   one-group loop
    ///
    /// No speed ratio versus batchEuclidean is measured, and no candidate-count threshold is derived here.
    ///
    /// - Parameters:
    ///   - query: FP16 query vector
    ///   - candidates: SoA-layout FP16 candidate vectors
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    @inlinable
    public static func batchEuclideanBlocked1536(
        query: Vector1536FP16,
        candidates: SoA1536FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }

        let blockSize = 8  // Process 8 candidates per iteration

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP16 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                // Process in blocks of 8 candidates
                for blockStart in stride(from: 0, to: vectorCount, by: blockSize) {
                    let blockCount = min(blockSize, vectorCount - blockStart)

                    // Dual SIMD4 accumulators for 8 candidates
                    var acc1 = SIMD4<Float>.zero  // Candidates 0-3
                    var acc2 = SIMD4<Float>.zero  // Candidates 4-7

                    // Calculate storage indices for both groups
                    let group1Start = (blockStart / 4) * 1536 * 4
                    let group2Start = ((blockStart + 4) / 4) * 1536 * 4

                    var idx1 = group1Start
                    var idx2 = group2Start

                    // Process all 1536 dimensions with dual accumulators
                    for d in 0..<1536 {
                        let q = fp16ToFp32_scalar(queryFP16[d])

                        // Group 1 (candidates 0-3)
                        let cand1 = SIMD4<Float>(
                            fp16ToFp32_scalar(candidatesFP16[idx1 + 0]),
                            fp16ToFp32_scalar(candidatesFP16[idx1 + 1]),
                            fp16ToFp32_scalar(candidatesFP16[idx1 + 2]),
                            fp16ToFp32_scalar(candidatesFP16[idx1 + 3])
                        )
                        let diff1 = SIMD4<Float>(repeating: q) - cand1
                        acc1.addProduct(diff1, diff1)
                        idx1 += 4

                        // Group 2 (candidates 4-7) - only if we have more than 4 candidates in block
                        if blockCount > 4 {
                            let cand2 = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx2 + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx2 + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx2 + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx2 + 3])
                            )
                            let diff2 = SIMD4<Float>(repeating: q) - cand2
                            acc2.addProduct(diff2, diff2)
                            idx2 += 4
                        }
                    }

                    // Store results for group 1 (candidates 0-3)
                    let count1 = min(4, blockCount)
                    for i in 0..<count1 {
                        results[blockStart + i] = sqrt(acc1[i])
                    }

                    // Store results for group 2 (candidates 4-7) if applicable
                    if blockCount > 4 {
                        let count2 = blockCount - 4
                        for i in 0..<count2 {
                            results[blockStart + 4 + i] = sqrt(acc2[i])
                        }
                    }
                }
            }
        }
    }

    /// Register-blocked Batch Euclidean Distance (FP32 query × FP16 candidates) - 1536D.
    ///
    /// **FP32 query + register blocking:**
    /// - The FP32 query is not converted
    /// - FP16 candidates (half the scalar payload; payload model only)
    /// - 8 candidates per iteration
    ///
    /// - Parameters:
    ///   - query: FP32 query vector
    ///   - candidates: SoA-layout FP16 candidate vectors
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    @inlinable
    public static func batchEuclideanBlocked1536(
        query: Vector1536Optimized,
        candidates: SoA1536FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        let vectorCount = candidates.vectorCount
        guard vectorCount > 0 else { return }

        guard results.count >= vectorCount else {
            fatalError("MixedPrecision batch results capacity must cover vectorCount")
        }

        let blockSize = 8

        query.storage.withUnsafeBufferPointer { queryPtr in
            candidates.storage.withUnsafeBufferPointer { candPtr in
                guard let queryFP32 = queryPtr.baseAddress,
                      let candidatesFP16 = candPtr.baseAddress else { return }

                for blockStart in stride(from: 0, to: vectorCount, by: blockSize) {
                    let blockCount = min(blockSize, vectorCount - blockStart)

                    var acc1 = SIMD4<Float>.zero
                    var acc2 = SIMD4<Float>.zero

                    let group1Start = (blockStart / 4) * 1536 * 4
                    let group2Start = ((blockStart + 4) / 4) * 1536 * 4

                    var idx1 = group1Start
                    var idx2 = group2Start

                    // Process 384 SIMD4 lanes (1536 dimensions / 4)
                    for lane in 0..<384 {
                        let queryLane = queryFP32[lane]

                        // Process 4 dimensions per lane
                        for d in 0..<4 {
                            let q = queryLane[d]

                            // Group 1
                            let cand1 = SIMD4<Float>(
                                fp16ToFp32_scalar(candidatesFP16[idx1 + d * 4 + 0]),
                                fp16ToFp32_scalar(candidatesFP16[idx1 + d * 4 + 1]),
                                fp16ToFp32_scalar(candidatesFP16[idx1 + d * 4 + 2]),
                                fp16ToFp32_scalar(candidatesFP16[idx1 + d * 4 + 3])
                            )
                            let diff1 = SIMD4<Float>(repeating: q) - cand1
                            acc1.addProduct(diff1, diff1)

                            // Group 2
                            if blockCount > 4 {
                                let cand2 = SIMD4<Float>(
                                    fp16ToFp32_scalar(candidatesFP16[idx2 + d * 4 + 0]),
                                    fp16ToFp32_scalar(candidatesFP16[idx2 + d * 4 + 1]),
                                    fp16ToFp32_scalar(candidatesFP16[idx2 + d * 4 + 2]),
                                    fp16ToFp32_scalar(candidatesFP16[idx2 + d * 4 + 3])
                                )
                                let diff2 = SIMD4<Float>(repeating: q) - cand2
                                acc2.addProduct(diff2, diff2)
                            }
                        }

                        idx1 += 16
                        idx2 += 16
                    }

                    // Store results
                    let count1 = min(4, blockCount)
                    for i in 0..<count1 {
                        results[blockStart + i] = sqrt(acc1[i])
                    }

                    if blockCount > 4 {
                        let count2 = blockCount - 4
                        for i in 0..<count2 {
                            results[blockStart + 4 + i] = sqrt(acc2[i])
                        }
                    }
                }
            }
        }
    }

    // MARK: - SoA Convenience

    /// Batch Euclidean **distance** (FP16 query × FP16 SoA candidates) using SoA layout.
    ///
    /// Returns the actual Euclidean distance (i.e. with `sqrt`), identical to `batchEuclidean512`.
    /// - Parameters:
    ///   - query: FP16 query vector
    ///   - candidates: SoA-layout FP16 candidate vectors
    ///   - results: Output buffer; must have capacity ≥ candidates.vectorCount (a shorter buffer traps before any write, in every build)
    @inlinable
    public static func batchEuclideanSoA(
        query: Vector512FP16,
        candidates: SoA512FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        batchEuclidean512(query: query, candidates: candidates, results: results)
    }

    /// Deprecated misnomer: despite the name, this returns the **actual** (non-squared)
    /// Euclidean distance. Use ``batchEuclideanSoA`` instead. (If you need squared distances,
    /// `BatchKernels_SoA.batchEuclideanSquared512` returns the true squared value.)
    @available(*, deprecated, renamed: "batchEuclideanSoA", message: "Returns the actual Euclidean distance (not squared) despite the old name; use batchEuclideanSoA.")
    @inlinable
    public static func batchEuclideanSquaredSoA(
        query: Vector512FP16,
        candidates: SoA512FP16,
        results: UnsafeMutableBufferPointer<Float>
    ) {
        batchEuclideanSoA(query: query, candidates: candidates, results: results)
    }

    /// Computes a direct FP32 baseline once and accepts a reduced result only
    /// when its actual relative error is within `threshold`, compared in Double
    /// (PrecisionSelectionContractTests.adaptiveBoundAgainstActualBaseline).
    /// Invalid/nonpositive thresholds and zero/nonfinite baselines return the
    /// baseline unchanged. Both operands must fit finite FP16 before conversion.
    /// This verification incurs baseline plus conversion/kernel work; it is not
    /// an acceleration guarantee. Complexity is O(512).
    @inlinable
    public static func adaptiveEuclideanDistance(
        query: Vector512Optimized,
        candidate: Vector512Optimized,
        threshold: Float = 0.001
    ) -> Float {
        let reference = query.euclideanDistance(to: candidate)
        guard threshold.isFinite, threshold > 0,
              reference.isFinite, reference > 0 else { return reference }
        for index in 0..<512 {
            let q = query[index]
            let c = candidate[index]
            guard q.isFinite, c.isFinite, abs(q) <= 65504, abs(c) <= 65504 else {
                return reference
            }
        }
        let queryFP16 = Vector512FP16(from: query)
        let candidatesSoA = createSoA512FP16(from: [candidate])
        var results: [Float] = [0]
        // The result is read through the buffer inside the closure, not as `results[0]` after it: Swift 6.4.0 at -O
        // can forward the literal 0 across an out-of-line `withUnsafeMutableBufferPointer` closure (SIL
        // RedundantLoadElimination; swiftlang/swift#92562; PR #92580 and its 6.4.x backports are believed to fix this
        // mechanism (unverified for this form; see T15/lost-store-investigation), the shape that
        // LostStoreCanaryTests.literalArrayStoreIsObservedAfterOutOfLineClosure reproduces. A stale 0 here would be
        // replaced by the reference for a threshold below 1 and returned for a threshold of 1 or larger (for a positive
        // reference, abs(0 - reference) <= threshold * reference holds iff threshold >= 1); the reduced result is pinned
        // at threshold 0.001 by AdaptiveReducedResultTests.acceptedReducedResultIsReturned.
        let reduced: Float = results.withUnsafeMutableBufferPointer {
            batchEuclidean512(query: queryFP16, candidates: candidatesSoA, results: $0)
            return $0[0]
        }
        guard reduced.isFinite,
              abs(Double(reduced) - Double(reference)) <= Double(threshold) * Double(reference)
        else { return reference }
        return reduced
    }

    /// Returns the constant 2.0, the FP32:FP16 scalar-width ratio (4 bytes / 2 bytes).
    /// - Parameter vectorCount: Ignored.
    /// - Returns: The scalar payload ratio. It excludes headers, capacity, handles and sharing, so it is
    ///   not a measured memory saving.
    @inlinable
    public static func estimateMemoryImprovement(vectorCount: Int) -> Float {
        // Scalar width ratio 4 / 2 (payload model only)
        return 2.0
    }

    // MARK: - Precision Analysis

    /// Precision profile for a dataset of vectors
    public struct PrecisionProfile: Sendable {
        public let minValue: Float
        public let maxValue: Float
        public let meanValue: Float
        public let stdDev: Float
        public let dynamicRange: Float
        public let recommendedPrecision: Precision
        public let expectedError: Float
        public let outlierCount: Int
        public let dimension: Int

        public enum Precision: String, Sendable {
            case fp32       // Full precision needed
            case fp16       // Half precision sufficient
            case int8       // Can quantize to INT8
            case mixed      // Use mixed strategies
        }

        /// Human-readable summary of the precision analysis. The expected-error figure and the reasoning
        /// text come from analyzePrecision's heuristic model and are not verified (Q11 runtime text); the
        /// reasoning's storage lines are scalar payload sizes (see `precisionReasoning`).
        public var summary: String {
            """
            Precision Analysis Results:
            ---------------------------
            Dimension: \(dimension)
            Value Range: [\(String(format: "%.6f", minValue)), \(String(format: "%.6f", maxValue))]
            Mean: \(String(format: "%.6f", meanValue)) ± \(String(format: "%.6f", stdDev))
            Dynamic Range: \(String(format: "%.2f", dynamicRange)) (\(String(format: "%.2f", 20 * log10(dynamicRange)))dB)
            Outliers: \(outlierCount) values exceed 3σ

            Recommended Precision: \(recommendedPrecision.rawValue.uppercased())
            Expected Relative Error: \(String(format: "%.4f%%", expectedError * 100))

            Reasoning:
            \(precisionReasoning)
            """
        }

        /// Fixed reasoning text per recommendation. Not verified: the FP16 error percentage is not derived from a
        /// measurement (Q11 runtime text). The storage lines are scalar payload sizes (FP16 2 bytes, INT8 1 byte,
        /// FP32 4 bytes per scalar), not a measured process-memory figure (D02 RD-R2).
        private var precisionReasoning: String {
            switch recommendedPrecision {
            case .fp32:
                return "• Values exceed FP16 range or require full precision\n• Dynamic range too large for safe quantization"
            case .fp16:
                return "• All values fit in FP16 range (±65504)\n• Expected error < 0.1% for typical operations\n• FP16: 2 bytes per scalar, FP32: 4 (scalar payload)"
            case .int8:
                return "• Small dynamic range suitable for 8-bit quantization\n• INT8: 1 byte per scalar, FP32: 4 (scalar payload)"
            case .mixed:
                return "• Use FP16 for candidates, FP32 for queries\n• Candidates: 2 bytes per scalar, queries: 4 (scalar payload)"
            }
        }
    }

    // Define a protocol to allow analyzePrecision to work generically over the OptimizedVector types for analysis.
    public protocol AnalyzableVector {
        var storage: ContiguousArray<SIMD4<Float>> { get }
        var scalarCount: Int { get }
    }
    // Assume VectorXXXOptimized conform to this protocol in the actual project context.

    /// Heuristic precision recommendation from simple statistics (range, mean, standard deviation and 3σ outliers).
    ///
    /// The expected-error figures are a fixed heuristic model (0.05% × D^(1/4) for FP16, 0.5% for INT8),
    /// not verified against measured error for any dataset.
    public static func analyzePrecision<V: AnalyzableVector>(
        _ vectors: [V]
    ) -> PrecisionProfile {
        guard !vectors.isEmpty else {
            return PrecisionProfile(
                minValue: 0, maxValue: 0, meanValue: 0, stdDev: 0,
                dynamicRange: 0, recommendedPrecision: .fp32, expectedError: 0,
                outlierCount: 0, dimension: 0
            )
        }

        let dimension = vectors[0].scalarCount

        // First pass: compute min, max, and sum for mean
        var minVal = Float.infinity
        var maxVal = -Float.infinity
        var sum: Double = 0
        var valueCount: Int = 0

        for vector in vectors {
            for simd4 in vector.storage {
                for i in 0..<4 {
                    let v = simd4[i]
                    if v.isFinite {
                        minVal = min(minVal, v)
                        maxVal = max(maxVal, v)
                        sum += Double(v)
                        valueCount += 1
                    }
                }
            }
        }

        let meanValue = Float(sum / Double(valueCount))

        // Second pass: compute variance for standard deviation
        var sumSquaredDiff: Double = 0
        for vector in vectors {
            for simd4 in vector.storage {
                for i in 0..<4 {
                    let v = simd4[i]
                    if v.isFinite {
                        let diff = Double(v - meanValue)
                        sumSquaredDiff += diff * diff
                    }
                }
            }
        }

        let variance = sumSquaredDiff / Double(valueCount)
        let stdDev = Float(sqrt(variance))

        // Third pass: count outliers (values beyond 3 standard deviations)
        var outlierCount = 0
        let outlierThreshold = 3.0 * stdDev
        for vector in vectors {
            for simd4 in vector.storage {
                for i in 0..<4 {
                    let v = simd4[i]
                    if v.isFinite && abs(v - meanValue) > outlierThreshold {
                        outlierCount += 1
                    }
                }
            }
        }

        // Compute dynamic range
        let dynamicRange = maxVal - minVal

        // Determine recommended precision based on analysis
        let fp16Max = Float(65504.0)
        let fp16Min = Float(-65504.0)

        var recommendedPrecision: PrecisionProfile.Precision
        var expectedError: Float

        // Decision logic for precision recommendation
        if maxVal > fp16Max || minVal < fp16Min {
            // Values exceed FP16 range — must use FP32.
            recommendedPrecision = .fp32
            expectedError = 0.0
        } else {
            // All values are within FP16 range. Heuristic model (not verified): 0.05% per element scaled by
            // D^(1/4).
            let dimensionFactor = sqrt(Float(dimension))
            let fp16RelativeError: Float = 0.0005  // assumed per-element error (heuristic constant; not verified)
            let fp16ExpectedError = fp16RelativeError * dimensionFactor / dimensionFactor.squareRoot()

            // INT8 is only worthwhile for non-negative, low-dynamic-range data where 8-bit
            // fixed-point preserves relative precision. Signed near-zero data (e.g. normalized
            // embeddings, components ~±1/√D) is poorly served by INT8 — prefer FP16/mixed.
            // (Previously this branch fired for ANY small range including signed embeddings,
            // shadowing the FP16 path, and the FP16 branch was unreachable for signed data
            // because of a `minVal > fp16MinNormal` guard.) This rationale is a heuristic, not verified.
            let int8Suitable = dynamicRange < 2.0 && abs(meanValue) < 10.0 && minVal >= 0.0

            if int8Suitable {
                recommendedPrecision = .int8
                expectedError = 0.005  // assumed INT8 error (heuristic constant; not verified)
            } else if fp16ExpectedError < 0.001 {  // model threshold: modelled FP16 error below 0.001
                recommendedPrecision = .fp16
                expectedError = fp16ExpectedError
            } else {
                recommendedPrecision = .mixed
                expectedError = fp16ExpectedError
            }
        }

        return PrecisionProfile(
            minValue: minVal,
            maxValue: maxVal,
            meanValue: meanValue,
            stdDev: stdDev,
            dynamicRange: dynamicRange,
            recommendedPrecision: recommendedPrecision,
            expectedError: expectedError,
            outlierCount: outlierCount,
            dimension: dimension
        )
    }

    /// Selects a precision from analyzePrecision's heuristic profile, escalating one step when its modelled
    /// error exceeds `errorTolerance` (the model is not verified).
    ///
    /// - Parameters:
    ///   - vectors: Sample vectors to analyze
    ///   - errorTolerance: Maximum acceptable relative error (default: 0.1%)
    /// - Returns: Recommended precision mode
    public static func selectOptimalPrecision<V: AnalyzableVector>(
        for vectors: [V],
        errorTolerance: Float = 0.001
    ) -> PrecisionProfile.Precision {
        let profile = analyzePrecision(vectors)

        // If expected error exceeds tolerance, escalate to higher precision
        if profile.expectedError > errorTolerance {
            switch profile.recommendedPrecision {
            case .int8:
                return .fp16
            case .fp16:
                return .mixed
            case .mixed:
                return .fp32
            case .fp32:
                return .fp32
            }
        }

        return profile.recommendedPrecision
    }
}

// MARK: - Benchmark Utilities

/// Benchmarking utilities for measuring precision/performance tradeoffs
internal struct MixedPrecisionBenchmark {

    /// Performance measurement result
    public struct BenchmarkResult: Sendable {
        public let operationName: String
        public let meanTimeNs: Double
        public let medianTimeNs: Double
        public let stdDevNs: Double
        public let minTimeNs: Double
        public let maxTimeNs: Double
        public let throughputOpsPerSec: Double
        public let memoryBandwidthGBps: Double?

        public var summary: String {
            let bw = memoryBandwidthGBps.map { String(format: "%.2f GB/s", $0) } ?? "N/A"
            return """
            \(operationName):
              Mean:   \(String(format: "%8.2f ns", meanTimeNs))
              Median: \(String(format: "%8.2f ns", medianTimeNs))
              StdDev: \(String(format: "%8.2f ns", stdDevNs))
              Range:  [\(String(format: "%.2f", minTimeNs)) - \(String(format: "%.2f", maxTimeNs))] ns
              Throughput: \(String(format: "%.2f", throughputOpsPerSec / 1_000_000)) M ops/sec
              Bandwidth: \(bw)
            """
        }
    }

    /// Accuracy measurement result
    public struct AccuracyResult: Sendable {
        public let operationName: String
        public let meanRelativeError: Float
        public let maxRelativeError: Float
        public let meanAbsoluteError: Float
        public let maxAbsoluteError: Float
        public let rankCorrelation: Float  // Spearman's rank correlation

        public var summary: String {
            """
            \(operationName) Accuracy:
              Mean Relative Error: \(String(format: "%.6f%%", meanRelativeError * 100))
              Max Relative Error:  \(String(format: "%.6f%%", maxRelativeError * 100))
              Mean Absolute Error: \(String(format: "%.8f", meanAbsoluteError))
              Max Absolute Error:  \(String(format: "%.8f", maxAbsoluteError))
              Rank Correlation:    \(String(format: "%.6f", rankCorrelation))
            """
        }
    }

    /// Benchmark FP16 vs FP32 dot product performance
    public static func benchmarkDotProduct512(
        iterations: Int = 1000,
        warmupIterations: Int = 100
    ) -> (fp32: BenchmarkResult, fp16: BenchmarkResult, speedup: Double) {
        // Deterministic test vectors (seeded) for reproducible benchmarks.
        var rng = SeededRNG(seed: 0xB1A5_0512)
        let vec1 = try! Vector512Optimized((0..<512).map { _ in rng.next() * 2 - 1 })
        let vec2 = try! Vector512Optimized((0..<512).map { _ in rng.next() * 2 - 1 })

        let vec1_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec1)
        let vec2_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec2)

        // Warmup
        for _ in 0..<warmupIterations {
            _ = DotKernels.dot512(vec1, vec2)
            _ = MixedPrecisionKernels.dotFP16_512(vec1_fp16, vec2_fp16)
        }

        // Benchmark FP32
        var fp32Times: [Double] = []
        for _ in 0..<iterations {
            let start = mach_absolute_time()
            _ = DotKernels.dot512(vec1, vec2)
            let end = mach_absolute_time()
            fp32Times.append(machTimeToNanoseconds(end - start))
        }

        // Benchmark FP16
        var fp16Times: [Double] = []
        for _ in 0..<iterations {
            let start = mach_absolute_time()
            _ = MixedPrecisionKernels.dotFP16_512(vec1_fp16, vec2_fp16)
            let end = mach_absolute_time()
            fp16Times.append(machTimeToNanoseconds(end - start))
        }

        let fp32Result = computeStatistics(times: fp32Times, operationName: "Dot Product 512 (FP32)", memoryBytes: 512 * 4 * 2)
        let fp16Result = computeStatistics(times: fp16Times, operationName: "Dot Product 512 (FP16)", memoryBytes: 512 * 2 * 2)
        let speedup = fp32Result.meanTimeNs / fp16Result.meanTimeNs

        return (fp32Result, fp16Result, speedup)
    }

    /// Measure accuracy of FP16 vs FP32 operations
    public static func measureAccuracy512(testVectors: Int = 1000) -> AccuracyResult {
        var relativeErrors: [Float] = []
        var absoluteErrors: [Float] = []
        var fp32Results: [Float] = []
        var fp16Results: [Float] = []

        // Deterministic data so accuracy measurement is reproducible (unseeded
        // Float.random made maxRelativeError flake on near-orthogonal draws).
        var rng = SeededRNG(seed: 0xACC0_0512)
        for _ in 0..<testVectors {
            let vec1 = try! Vector512Optimized((0..<512).map { _ in rng.next() * 2 - 1 })
            let vec2 = try! Vector512Optimized((0..<512).map { _ in rng.next() * 2 - 1 })

            let vec1_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec1)
            let vec2_fp16 = MixedPrecisionKernels.Vector512FP16(from: vec2)

            let fp32Result = DotKernels.dot512(vec1, vec2)
            let fp16Result = MixedPrecisionKernels.dotFP16_512(vec1_fp16, vec2_fp16)

            fp32Results.append(fp32Result)
            fp16Results.append(fp16Result)

            let absError = abs(fp32Result - fp16Result)
            absoluteErrors.append(absError)

            if abs(fp32Result) > 1e-6 {
                let relError = absError / abs(fp32Result)
                relativeErrors.append(relError)
            }
        }

        // Compute Spearman's rank correlation
        let rankCorr = computeSpearmanCorrelation(fp32Results, fp16Results)

        return AccuracyResult(
            operationName: "Dot Product 512",
            meanRelativeError: relativeErrors.reduce(0, +) / Float(relativeErrors.count),
            maxRelativeError: relativeErrors.max() ?? 0,
            meanAbsoluteError: absoluteErrors.reduce(0, +) / Float(absoluteErrors.count),
            maxAbsoluteError: absoluteErrors.max() ?? 0,
            rankCorrelation: rankCorr
        )
    }

    /// Benchmark batch operations
    public static func benchmarkBatchOperations512(
        candidateCount: Int = 1000,
        iterations: Int = 100
    ) -> (mixed: BenchmarkResult, fp16: BenchmarkResult) {
        // Deterministic test data (seeded) for reproducible benchmarks.
        var rng = SeededRNG(seed: 0xBA7C_0512)
        let query = try! Vector512Optimized((0..<512).map { _ in rng.next() * 2 - 1 })
        let candidates: [Vector512Optimized] = (0..<candidateCount).map { _ in
            try! Vector512Optimized((0..<512).map { _ in rng.next() * 2 - 1 })
        }

        let candidatesFP16 = candidates.map { MixedPrecisionKernels.Vector512FP16(from: $0) }
        var results = [Float](repeating: 0, count: candidateCount)

        // Warmup
        results.withUnsafeMutableBufferPointer { buffer in
            for _ in 0..<10 {
                MixedPrecisionKernels.batchDotMixed512(query: query, candidates: candidatesFP16, out: buffer)
            }
        }

        // Benchmark mixed precision
        var mixedTimes: [Double] = []
        results.withUnsafeMutableBufferPointer { buffer in
            for _ in 0..<iterations {
                let start = mach_absolute_time()
                MixedPrecisionKernels.batchDotMixed512(query: query, candidates: candidatesFP16, out: buffer)
                let end = mach_absolute_time()
                mixedTimes.append(machTimeToNanoseconds(end - start))
            }
        }

        // Benchmark FP16 (converts query)
        var fp16Times: [Double] = []
        results.withUnsafeMutableBufferPointer { buffer in
            for _ in 0..<iterations {
                let start = mach_absolute_time()
                MixedPrecisionKernels.batchDotFP16_512(query: query, candidates: candidatesFP16, out: buffer)
                let end = mach_absolute_time()
                fp16Times.append(machTimeToNanoseconds(end - start))
            }
        }

        let mixedResult = computeStatistics(
            times: mixedTimes,
            operationName: "Batch Mixed 512 (\(candidateCount) candidates)",
            memoryBytes: candidateCount * 512 * 2 + 512 * 4
        )
        let fp16Result = computeStatistics(
            times: fp16Times,
            operationName: "Batch FP16 512 (\(candidateCount) candidates)",
            memoryBytes: candidateCount * 512 * 2 + 512 * 2
        )

        return (mixedResult, fp16Result)
    }

    // MARK: - Helper Functions

    private static func machTimeToNanoseconds(_ time: UInt64) -> Double {
        var timebase = mach_timebase_info_data_t()
        mach_timebase_info(&timebase)
        return Double(time) * Double(timebase.numer) / Double(timebase.denom)
    }

    private static func computeStatistics(
        times: [Double],
        operationName: String,
        memoryBytes: Int
    ) -> BenchmarkResult {
        let sorted = times.sorted()
        let mean = times.reduce(0, +) / Double(times.count)
        let median = sorted[sorted.count / 2]
        let variance = times.map { pow($0 - mean, 2) }.reduce(0, +) / Double(times.count)
        let stdDev = sqrt(variance)
        let throughput = 1_000_000_000.0 / mean  // ops per second

        // Memory bandwidth in GB/s
        let bandwidthGBps = (Double(memoryBytes) / mean) // bytes per nanosecond = GB/s

        return BenchmarkResult(
            operationName: operationName,
            meanTimeNs: mean,
            medianTimeNs: median,
            stdDevNs: stdDev,
            minTimeNs: sorted.first ?? 0,
            maxTimeNs: sorted.last ?? 0,
            throughputOpsPerSec: throughput,
            memoryBandwidthGBps: bandwidthGBps
        )
    }

    private static func computeSpearmanCorrelation(_ x: [Float], _ y: [Float]) -> Float {
        guard x.count == y.count else { return 0 }

        // Compute ranks
        let xRanks = computeRanks(x)
        let yRanks = computeRanks(y)

        // Compute correlation of ranks (Pearson on ranks = Spearman only without ties: computeRanks gives tied
        // values distinct sequential ranks, not averaged ranks)
        let n = Float(x.count)
        let xMean = xRanks.reduce(0, +) / n
        let yMean = yRanks.reduce(0, +) / n

        var numerator: Float = 0
        var xVar: Float = 0
        var yVar: Float = 0

        for i in 0..<x.count {
            let xDiff = xRanks[i] - xMean
            let yDiff = yRanks[i] - yMean
            numerator += xDiff * yDiff
            xVar += xDiff * xDiff
            yVar += yDiff * yDiff
        }

        return numerator / sqrt(xVar * yVar)
    }

    private static func computeRanks(_ values: [Float]) -> [Float] {
        let indexed = values.enumerated().sorted { $0.element < $1.element }
        var ranks = [Float](repeating: 0, count: values.count)
        for (rank, (index, _)) in indexed.enumerated() {
            ranks[index] = Float(rank + 1)
        }
        return ranks
    }
}

// MARK: - Utility Extensions

/// Maximum representable finite value in IEEE 754 half-precision (Float16).
private let MaxFloat16Value: Float = 65504.0

// MARK: - Range Validation

public extension MixedPrecisionKernels.Vector512FP16 {
    /// Validates that all components are finite and within the representable range of FP16.
    ///
    /// Returns `true` if all values are finite and within ±65504.
    /// Use `canRepresent(_:)` to validate FP32 inputs before conversion.
    func validateRange() -> Bool {
        return storage.withUnsafeBufferPointer { buffer in
            guard let ptr = buffer.baseAddress else { return false }

            for i in 0..<storage.count {
                // An all-ones binary16 exponent denotes infinity or NaN.
                // MixedPrecisionRangeValidationTests (rejectsNonFinitePatterns) covers both signs and all widths.
                guard ptr[i] & 0x7C00 != 0x7C00 else {
                    return false
                }
            }
            return true
        }
    }

    /// Returns true when every value is finite and |value| <= 65504. This is stricter than overflow:
    /// values in (65504, 65520) still round to finite 65504.
    ///
    /// - Parameter values: Array of FP32 values to validate
    /// - Returns: `true` if all values fit in FP16 range
    static func canRepresent(_ values: [Float]) -> Bool {
        return values.allSatisfy { value in
            value.isFinite && abs(value) <= MaxFloat16Value
        }
    }
}

public extension MixedPrecisionKernels.Vector768FP16 {
    func validateRange() -> Bool {
        return storage.withUnsafeBufferPointer { buffer in
            guard let ptr = buffer.baseAddress else { return false }

            for i in 0..<storage.count {
                // An all-ones binary16 exponent denotes infinity or NaN.
                // MixedPrecisionRangeValidationTests (rejectsNonFinitePatterns) covers both signs and all widths.
                guard ptr[i] & 0x7C00 != 0x7C00 else {
                    return false
                }
            }
            return true
        }
    }

    static func canRepresent(_ values: [Float]) -> Bool {
        return values.allSatisfy { value in
            value.isFinite && abs(value) <= MaxFloat16Value
        }
    }
}

public extension MixedPrecisionKernels.Vector1536FP16 {
    func validateRange() -> Bool {
        return storage.withUnsafeBufferPointer { buffer in
            guard let ptr = buffer.baseAddress else { return false }

            for i in 0..<storage.count {
                // An all-ones binary16 exponent denotes infinity or NaN.
                // MixedPrecisionRangeValidationTests (rejectsNonFinitePatterns) covers both signs and all widths.
                guard ptr[i] & 0x7C00 != 0x7C00 else {
                    return false
                }
            }
            return true
        }
    }

    static func canRepresent(_ values: [Float]) -> Bool {
        return values.allSatisfy { value in
            value.isFinite && abs(value) <= MaxFloat16Value
        }
    }
}

// MARK: - Mixed Precision Factory

/// Factory methods for creating FP16 vectors from raw Float arrays.
///
/// Provides convenient initialization from Float arrays with automatic FP32→FP16 conversion.
public enum MixedPrecisionFactory {

    /// Create Vector512FP16 from array of FP32 values.
    ///
    /// - Parameter values: Exactly 512 Float values
    /// - Returns: Vector512FP16 with converted FP16 storage
    /// - Precondition: `values.count == 512`
    public static func createVector512FP16(from values: [Float]) -> MixedPrecisionKernels.Vector512FP16 {
        precondition(values.count == 512, "Vector512FP16 requires exactly 512 values, got \(values.count)")

        // Convert to ContiguousArray<SIMD4<Float>> first
        var simd4Storage = ContiguousArray<SIMD4<Float>>()
        simd4Storage.reserveCapacity(128)

        for i in stride(from: 0, to: 512, by: 4) {
            simd4Storage.append(SIMD4<Float>(
                values[i], values[i+1], values[i+2], values[i+3]
            ))
        }

        // Use MixedPrecisionKernels conversion
        let fp16Storage = MixedPrecisionKernels.convertToFP16(simd4Storage)
        return MixedPrecisionKernels.Vector512FP16(fp16Values: Array(fp16Storage))
    }

    /// Create Vector768FP16 from array of FP32 values.
    ///
    /// - Parameter values: Exactly 768 Float values
    /// - Returns: Vector768FP16 with converted FP16 storage
    /// - Precondition: `values.count == 768`
    public static func createVector768FP16(from values: [Float]) -> MixedPrecisionKernels.Vector768FP16 {
        precondition(values.count == 768, "Vector768FP16 requires exactly 768 values, got \(values.count)")

        var simd4Storage = ContiguousArray<SIMD4<Float>>()
        simd4Storage.reserveCapacity(192)

        for i in stride(from: 0, to: 768, by: 4) {
            simd4Storage.append(SIMD4<Float>(
                values[i], values[i+1], values[i+2], values[i+3]
            ))
        }

        let fp16Storage = MixedPrecisionKernels.convertToFP16(simd4Storage)
        return MixedPrecisionKernels.Vector768FP16(fp16Values: Array(fp16Storage))
    }

    /// Create Vector1536FP16 from array of FP32 values.
    ///
    /// - Parameter values: Exactly 1536 Float values
    /// - Returns: Vector1536FP16 with converted FP16 storage
    /// - Precondition: `values.count == 1536`
    public static func createVector1536FP16(from values: [Float]) -> MixedPrecisionKernels.Vector1536FP16 {
        precondition(values.count == 1536, "Vector1536FP16 requires exactly 1536 values, got \(values.count)")

        var simd4Storage = ContiguousArray<SIMD4<Float>>()
        simd4Storage.reserveCapacity(384)

        for i in stride(from: 0, to: 1536, by: 4) {
            simd4Storage.append(SIMD4<Float>(
                values[i], values[i+1], values[i+2], values[i+3]
            ))
        }

        let fp16Storage = MixedPrecisionKernels.convertToFP16(simd4Storage)
        return MixedPrecisionKernels.Vector1536FP16(fp16Values: Array(fp16Storage))
    }
}

// MARK: - Overflow Detection

extension MixedPrecisionKernels {
    #if !(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))
    /// Utility function to check if an FP32 value can be safely converted to FP16.
    ///
    /// Available where the SDK provides native Float16; excluded on Intel macOS
    /// and Mac Catalyst. `validateBatch(values:)` remains available there.
    ///
    /// - Parameter value: FP32 value to validate
    /// - Returns: Float16 value if conversion is safe, nil if overflow would occur
    ///
    /// Use this before conversion to detect overflow (non-finite input or |value| > 65504); it does not
    /// detect rounding loss or underflow:
    /// ```swift
    /// if let fp16 = MixedPrecisionKernels.detectOverflow(value: largeFloat) {
    ///     // Safe to use fp16
    /// } else {
    ///     // Handle overflow: value exceeds ±65504 or is NaN/Inf
    /// }
    /// ```
    public static func detectOverflow(value: Float) -> Float16? {
        guard value.isFinite else { return nil }
        guard abs(value) <= MaxFloat16Value else { return nil }
        return Float16(value)
    }
    #endif

    /// Batch validate FP32 values for FP16 conversion.
    ///
    /// - Parameter values: Array of FP32 values
    /// - Returns: Tuple of (all valid, overflow count, first overflow index)
    public static func validateBatch(values: [Float]) -> (allValid: Bool, overflowCount: Int, firstOverflowIndex: Int?) {
        var overflowCount = 0
        var firstOverflowIndex: Int?

        for (index, value) in values.enumerated() {
            if !value.isFinite || abs(value) > MaxFloat16Value {
                overflowCount += 1
                if firstOverflowIndex == nil {
                    firstOverflowIndex = index
                }
            }
        }

        return (overflowCount == 0, overflowCount, firstOverflowIndex)
    }

    // MARK: - Range-Based Batch Kernels (Spec #32)

    // MARK: - Core Batch Helpers (Internal)

    /// Core batch Euclidean² kernel with 2-way register blocking
    ///
    /// **Algorithm**: process 2 candidates per iteration, loading each query lane once for both
    /// - FP16 candidates stored as UInt16 bit patterns
    /// - Loaded as SIMD8<UInt16> and converted by `fp16ToFp32_simd8` (scalar software conversions)
    /// - 8 accumulators (4 per candidate)
    ///
    /// **Complexity**: O(lanes × candidates); no throughput ratio is measured
    @inline(__always)
    @usableFromInline
    internal static func range_euclid2_mixed_core(
        queryStorage: ContiguousArray<SIMD4<Float>>,
        candidatesArray: [ContiguousArray<UInt16>],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>,
        lanes: Int
    ) {
        var candidateIdx = range.lowerBound
        var outIdx = 0
        let end = range.upperBound

        // Active in every build: the public range wrappers reject these first, and this core keeps its own check.
        precondition(out.count >= range.count, "MixedPrecision range core output capacity must cover the range")
        precondition(end <= candidatesArray.count, "MixedPrecision range core range exceeds the candidate count")

        // Main loop: 2-way blocking
        while candidateIdx + 1 < end {
            let c0_storage = candidatesArray[candidateIdx]
            let c1_storage = candidatesArray[candidateIdx + 1]

            c0_storage.withUnsafeBufferPointer { c0Buffer in
                c1_storage.withUnsafeBufferPointer { c1Buffer in
                    guard let c0Ptr = c0Buffer.baseAddress,
                          let c1Ptr = c1Buffer.baseAddress else {
                        out[outIdx] = 0
                        out[outIdx + 1] = 0
                        return
                    }

                    // 8 accumulators: 4 for candidate 0, 4 for candidate 1
                    var a0 = SIMD4<Float>.zero, a1 = SIMD4<Float>.zero
                    var a2 = SIMD4<Float>.zero, a3 = SIMD4<Float>.zero
                    var b0 = SIMD4<Float>.zero, b1 = SIMD4<Float>.zero
                    var b2 = SIMD4<Float>.zero, b3 = SIMD4<Float>.zero

                    // Process 4 lanes at a time (16 floats = 2× SIMD8 conversions)
                    var i = 0
                    while i + 4 <= lanes {
                        let q0 = queryStorage[i]
                        let q1 = queryStorage[i + 1]
                        let q2 = queryStorage[i + 2]
                        let q3 = queryStorage[i + 3]

                        let offset = i * 4

                        // Load and convert candidate 0 (16 FP16 → 16 FP32)
                        let c0_fp16_0 = UnsafeRawPointer(c0Ptr.advanced(by: offset))
                            .loadUnaligned(as: SIMD8<UInt16>.self)
                        let c0_fp32_0 = fp16ToFp32_simd8(c0_fp16_0)

                        let c0_fp16_1 = UnsafeRawPointer(c0Ptr.advanced(by: offset + 8))
                            .loadUnaligned(as: SIMD8<UInt16>.self)
                        let c0_fp32_1 = fp16ToFp32_simd8(c0_fp16_1)

                        // Load and convert candidate 1
                        let c1_fp16_0 = UnsafeRawPointer(c1Ptr.advanced(by: offset))
                            .loadUnaligned(as: SIMD8<UInt16>.self)
                        let c1_fp32_0 = fp16ToFp32_simd8(c1_fp16_0)

                        let c1_fp16_1 = UnsafeRawPointer(c1Ptr.advanced(by: offset + 8))
                            .loadUnaligned(as: SIMD8<UInt16>.self)
                        let c1_fp32_1 = fp16ToFp32_simd8(c1_fp16_1)

                        // Compute differences and accumulate (candidate 0)
                        let diff_a0 = q0 - c0_fp32_0.lowHalf
                        let diff_a1 = q1 - c0_fp32_0.highHalf
                        let diff_a2 = q2 - c0_fp32_1.lowHalf
                        let diff_a3 = q3 - c0_fp32_1.highHalf

                        a0.addProduct(diff_a0, diff_a0)
                        a1.addProduct(diff_a1, diff_a1)
                        a2.addProduct(diff_a2, diff_a2)
                        a3.addProduct(diff_a3, diff_a3)

                        // Compute differences and accumulate (candidate 1)
                        let diff_b0 = q0 - c1_fp32_0.lowHalf
                        let diff_b1 = q1 - c1_fp32_0.highHalf
                        let diff_b2 = q2 - c1_fp32_1.lowHalf
                        let diff_b3 = q3 - c1_fp32_1.highHalf

                        b0.addProduct(diff_b0, diff_b0)
                        b1.addProduct(diff_b1, diff_b1)
                        b2.addProduct(diff_b2, diff_b2)
                        b3.addProduct(diff_b3, diff_b3)

                        i += 4
                    }

                    // Tail: handle remaining lanes
                    while i < lanes {
                        let q = queryStorage[i]
                        let offset = i * 4

                        // Candidate 0
                        let c0_0 = fp16ToFp32_scalar(c0Ptr[offset])
                        let c0_1 = fp16ToFp32_scalar(c0Ptr[offset + 1])
                        let c0_2 = fp16ToFp32_scalar(c0Ptr[offset + 2])
                        let c0_3 = fp16ToFp32_scalar(c0Ptr[offset + 3])
                        let c0 = SIMD4<Float>(c0_0, c0_1, c0_2, c0_3)
                        let diff0 = q - c0
                        a0.addProduct(diff0, diff0)

                        // Candidate 1
                        let c1_0 = fp16ToFp32_scalar(c1Ptr[offset])
                        let c1_1 = fp16ToFp32_scalar(c1Ptr[offset + 1])
                        let c1_2 = fp16ToFp32_scalar(c1Ptr[offset + 2])
                        let c1_3 = fp16ToFp32_scalar(c1Ptr[offset + 3])
                        let c1 = SIMD4<Float>(c1_0, c1_1, c1_2, c1_3)
                        let diff1 = q - c1
                        b0.addProduct(diff1, diff1)

                        i += 1
                    }

                    // Horizontal reduction and store
                    let sum0 = ((a0 + a1) + (a2 + a3)).sum()
                    let sum1 = ((b0 + b1) + (b2 + b3)).sum()

                    out[outIdx] = sum0
                    out[outIdx + 1] = sum1
                }
            }

            candidateIdx += 2
            outIdx += 2
        }

        // Handle odd candidate count
        if candidateIdx < end {
            let c_storage = candidatesArray[candidateIdx]
            c_storage.withUnsafeBufferPointer { cBuffer in
                guard let cPtr = cBuffer.baseAddress else {
                    out[outIdx] = 0
                    return
                }

                var acc0 = SIMD4<Float>.zero
                var acc1 = SIMD4<Float>.zero
                var acc2 = SIMD4<Float>.zero
                var acc3 = SIMD4<Float>.zero

                var i = 0
                while i + 4 <= lanes {
                    let q0 = queryStorage[i]
                    let q1 = queryStorage[i + 1]
                    let q2 = queryStorage[i + 2]
                    let q3 = queryStorage[i + 3]

                    let offset = i * 4

                    let c_fp16_0 = UnsafeRawPointer(cPtr.advanced(by: offset))
                        .loadUnaligned(as: SIMD8<UInt16>.self)
                    let c_fp32_0 = fp16ToFp32_simd8(c_fp16_0)

                    let c_fp16_1 = UnsafeRawPointer(cPtr.advanced(by: offset + 8))
                        .loadUnaligned(as: SIMD8<UInt16>.self)
                    let c_fp32_1 = fp16ToFp32_simd8(c_fp16_1)

                    let diff0 = q0 - c_fp32_0.lowHalf
                    let diff1 = q1 - c_fp32_0.highHalf
                    let diff2 = q2 - c_fp32_1.lowHalf
                    let diff3 = q3 - c_fp32_1.highHalf

                    acc0.addProduct(diff0, diff0)
                    acc1.addProduct(diff1, diff1)
                    acc2.addProduct(diff2, diff2)
                    acc3.addProduct(diff3, diff3)

                    i += 4
                }

                while i < lanes {
                    let q = queryStorage[i]
                    let offset = i * 4

                    let c0 = fp16ToFp32_scalar(cPtr[offset])
                    let c1 = fp16ToFp32_scalar(cPtr[offset + 1])
                    let c2 = fp16ToFp32_scalar(cPtr[offset + 2])
                    let c3 = fp16ToFp32_scalar(cPtr[offset + 3])

                    let c = SIMD4<Float>(c0, c1, c2, c3)
                    let diff = q - c
                    acc0.addProduct(diff, diff)

                    i += 1
                }

                let sum = ((acc0 + acc1) + (acc2 + acc3)).sum()
                out[outIdx] = sum
            }
        }
    }

    /// Core batch Cosine distance kernel with 2-way register blocking
    ///
    /// **Algorithm**: Fused dot product + magnitude computation
    /// - Computes: distance = 1 - (dot(q,c) / (||q|| × ||c||))
    /// - Single pass through vectors for all three metrics
    /// - Distance is `Float.nan` when the query or candidate magnitude is not finite; otherwise similarity is zero
    ///   when the denominator is not finite or is at most epsilon, then clamped to [-1, 1]
    ///
    /// **Complexity**: O(lanes × candidates) with compute-bound characteristics
    @inline(__always)
    @usableFromInline
    internal static func range_cosine_mixed_core(
        queryStorage: ContiguousArray<SIMD4<Float>>,
        candidatesArray: [ContiguousArray<UInt16>],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>,
        lanes: Int
    ) {
        let epsilon: Float = 1e-9

        // Pre-compute query magnitude once
        var qMagAcc = SIMD4<Float>.zero
        for i in 0..<lanes {
            qMagAcc.addProduct(queryStorage[i], queryStorage[i])
        }
        let queryMag = sqrt(qMagAcc.sum())

        var candidateIdx = range.lowerBound
        var outIdx = 0
        let end = range.upperBound

        // Main loop: 2-way blocking
        while candidateIdx + 1 < end {
            let c0_storage = candidatesArray[candidateIdx]
            let c1_storage = candidatesArray[candidateIdx + 1]

            c0_storage.withUnsafeBufferPointer { c0Buffer in
                c1_storage.withUnsafeBufferPointer { c1Buffer in
                    guard let c0Ptr = c0Buffer.baseAddress,
                          let c1Ptr = c1Buffer.baseAddress else {
                        out[outIdx] = 1.0
                        out[outIdx + 1] = 1.0
                        return
                    }

                    var dot0 = SIMD4<Float>.zero, mag0 = SIMD4<Float>.zero
                    var dot1 = SIMD4<Float>.zero, mag1 = SIMD4<Float>.zero

                    var i = 0
                    while i + 2 <= lanes {
                        let q0 = queryStorage[i]
                        let q1 = queryStorage[i + 1]
                        let offset = i * 4

                        // Candidate 0
                        let c0_fp16 = UnsafeRawPointer(c0Ptr.advanced(by: offset))
                            .loadUnaligned(as: SIMD8<UInt16>.self)
                        let c0_fp32 = fp16ToFp32_simd8(c0_fp16)

                        dot0.addProduct(q0, c0_fp32.lowHalf)
                        dot0.addProduct(q1, c0_fp32.highHalf)
                        mag0.addProduct(c0_fp32.lowHalf, c0_fp32.lowHalf)
                        mag0.addProduct(c0_fp32.highHalf, c0_fp32.highHalf)

                        // Candidate 1
                        let c1_fp16 = UnsafeRawPointer(c1Ptr.advanced(by: offset))
                            .loadUnaligned(as: SIMD8<UInt16>.self)
                        let c1_fp32 = fp16ToFp32_simd8(c1_fp16)

                        dot1.addProduct(q0, c1_fp32.lowHalf)
                        dot1.addProduct(q1, c1_fp32.highHalf)
                        mag1.addProduct(c1_fp32.lowHalf, c1_fp32.lowHalf)
                        mag1.addProduct(c1_fp32.highHalf, c1_fp32.highHalf)

                        i += 2
                    }

                    // Tail
                    while i < lanes {
                        let q = queryStorage[i]
                        let offset = i * 4

                        let c0_0 = fp16ToFp32_scalar(c0Ptr[offset])
                        let c0_1 = fp16ToFp32_scalar(c0Ptr[offset + 1])
                        let c0_2 = fp16ToFp32_scalar(c0Ptr[offset + 2])
                        let c0_3 = fp16ToFp32_scalar(c0Ptr[offset + 3])
                        let c0 = SIMD4<Float>(c0_0, c0_1, c0_2, c0_3)
                        dot0.addProduct(q, c0)
                        mag0.addProduct(c0, c0)

                        let c1_0 = fp16ToFp32_scalar(c1Ptr[offset])
                        let c1_1 = fp16ToFp32_scalar(c1Ptr[offset + 1])
                        let c1_2 = fp16ToFp32_scalar(c1Ptr[offset + 2])
                        let c1_3 = fp16ToFp32_scalar(c1Ptr[offset + 3])
                        let c1 = SIMD4<Float>(c1_0, c1_1, c1_2, c1_3)
                        dot1.addProduct(q, c1)
                        mag1.addProduct(c1, c1)

                        i += 1
                    }

                    // Cosine distances: NaN for a non-finite magnitude; zero similarity for a tiny or non-finite
                    // denominator, then clamp to [-1, 1]
                    let dp0 = dot0.sum()
                    let mag0_val = sqrt(mag0.sum())
                    let denom0 = queryMag * mag0_val
                    let similarity0 = (denom0 > epsilon && denom0.isFinite) ? (dp0 / denom0) : 0.0
                    out[outIdx] = (queryMag.isFinite && mag0_val.isFinite) ? 1.0 - max(-1.0, min(1.0, similarity0)) : .nan

                    let dp1 = dot1.sum()
                    let mag1_val = sqrt(mag1.sum())
                    let denom1 = queryMag * mag1_val
                    let similarity1 = (denom1 > epsilon && denom1.isFinite) ? (dp1 / denom1) : 0.0
                    out[outIdx + 1] = (queryMag.isFinite && mag1_val.isFinite) ? 1.0 - max(-1.0, min(1.0, similarity1)) : .nan
                }
            }

            candidateIdx += 2
            outIdx += 2
        }

        // Handle odd candidate
        if candidateIdx < end {
            let c_storage = candidatesArray[candidateIdx]
            c_storage.withUnsafeBufferPointer { cBuffer in
                guard let cPtr = cBuffer.baseAddress else {
                    out[outIdx] = 1.0
                    return
                }

                var dot = SIMD4<Float>.zero
                var mag = SIMD4<Float>.zero

                var i = 0
                while i + 2 <= lanes {
                    let q0 = queryStorage[i]
                    let q1 = queryStorage[i + 1]
                    let offset = i * 4

                    let c_fp16 = UnsafeRawPointer(cPtr.advanced(by: offset))
                        .loadUnaligned(as: SIMD8<UInt16>.self)
                    let c_fp32 = fp16ToFp32_simd8(c_fp16)

                    dot.addProduct(q0, c_fp32.lowHalf)
                    dot.addProduct(q1, c_fp32.highHalf)
                    mag.addProduct(c_fp32.lowHalf, c_fp32.lowHalf)
                    mag.addProduct(c_fp32.highHalf, c_fp32.highHalf)

                    i += 2
                }

                while i < lanes {
                    let q = queryStorage[i]
                    let offset = i * 4

                    let c0 = fp16ToFp32_scalar(cPtr[offset])
                    let c1 = fp16ToFp32_scalar(cPtr[offset + 1])
                    let c2 = fp16ToFp32_scalar(cPtr[offset + 2])
                    let c3 = fp16ToFp32_scalar(cPtr[offset + 3])
                    let c = SIMD4<Float>(c0, c1, c2, c3)

                    dot.addProduct(q, c)
                    mag.addProduct(c, c)

                    i += 1
                }

                let dp = dot.sum()
                let mag_val = sqrt(mag.sum())
                let denom = queryMag * mag_val
                let similarity = (denom > epsilon && denom.isFinite) ? (dp / denom) : 0.0
                out[outIdx] = (queryMag.isFinite && mag_val.isFinite) ? 1.0 - max(-1.0, min(1.0, similarity)) : .nan
            }
        }
    }

    /// Core batch Dot Product kernel with 2-way register blocking
    ///
    /// **Formula**: dot(q, c) = Σᵢ qᵢ × cᵢ
    /// **Numerical precision**: FP32 accumulation of FP16-rounded candidates; no relative-error bound is
    /// promised (near-cancelling sums have unbounded relative error)
    @inline(__always)
    @usableFromInline
    internal static func range_dot_mixed_core(
        queryStorage: ContiguousArray<SIMD4<Float>>,
        candidatesArray: [ContiguousArray<UInt16>],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>,
        lanes: Int
    ) {
        var candidateIdx = range.lowerBound
        var outIdx = 0
        let end = range.upperBound

        // Main loop: 2-way blocking
        while candidateIdx + 1 < end {
            let c0_storage = candidatesArray[candidateIdx]
            let c1_storage = candidatesArray[candidateIdx + 1]

            c0_storage.withUnsafeBufferPointer { c0Buffer in
                c1_storage.withUnsafeBufferPointer { c1Buffer in
                    guard let c0Ptr = c0Buffer.baseAddress,
                          let c1Ptr = c1Buffer.baseAddress else {
                        out[outIdx] = 0
                        out[outIdx + 1] = 0
                        return
                    }

                    var dot0 = SIMD4<Float>.zero
                    var dot1 = SIMD4<Float>.zero

                    var i = 0
                    while i + 2 <= lanes {
                        let q0 = queryStorage[i]
                        let q1 = queryStorage[i + 1]
                        let offset = i * 4

                        // Candidate 0
                        let c0_fp16 = UnsafeRawPointer(c0Ptr.advanced(by: offset))
                            .loadUnaligned(as: SIMD8<UInt16>.self)
                        let c0_fp32 = fp16ToFp32_simd8(c0_fp16)
                        dot0.addProduct(q0, c0_fp32.lowHalf)
                        dot0.addProduct(q1, c0_fp32.highHalf)

                        // Candidate 1
                        let c1_fp16 = UnsafeRawPointer(c1Ptr.advanced(by: offset))
                            .loadUnaligned(as: SIMD8<UInt16>.self)
                        let c1_fp32 = fp16ToFp32_simd8(c1_fp16)
                        dot1.addProduct(q0, c1_fp32.lowHalf)
                        dot1.addProduct(q1, c1_fp32.highHalf)

                        i += 2
                    }

                    while i < lanes {
                        let q = queryStorage[i]
                        let offset = i * 4

                        let c0_0 = fp16ToFp32_scalar(c0Ptr[offset])
                        let c0_1 = fp16ToFp32_scalar(c0Ptr[offset + 1])
                        let c0_2 = fp16ToFp32_scalar(c0Ptr[offset + 2])
                        let c0_3 = fp16ToFp32_scalar(c0Ptr[offset + 3])
                        let c0 = SIMD4<Float>(c0_0, c0_1, c0_2, c0_3)
                        dot0.addProduct(q, c0)

                        let c1_0 = fp16ToFp32_scalar(c1Ptr[offset])
                        let c1_1 = fp16ToFp32_scalar(c1Ptr[offset + 1])
                        let c1_2 = fp16ToFp32_scalar(c1Ptr[offset + 2])
                        let c1_3 = fp16ToFp32_scalar(c1Ptr[offset + 3])
                        let c1 = SIMD4<Float>(c1_0, c1_1, c1_2, c1_3)
                        dot1.addProduct(q, c1)

                        i += 1
                    }

                    out[outIdx] = dot0.sum()
                    out[outIdx + 1] = dot1.sum()
                }
            }

            candidateIdx += 2
            outIdx += 2
        }

        // Handle odd candidate
        if candidateIdx < end {
            let c_storage = candidatesArray[candidateIdx]
            c_storage.withUnsafeBufferPointer { cBuffer in
                guard let cPtr = cBuffer.baseAddress else {
                    out[outIdx] = 0
                    return
                }

                var dot = SIMD4<Float>.zero

                var i = 0
                while i + 2 <= lanes {
                    let q0 = queryStorage[i]
                    let q1 = queryStorage[i + 1]
                    let offset = i * 4

                    let c_fp16 = UnsafeRawPointer(cPtr.advanced(by: offset))
                        .loadUnaligned(as: SIMD8<UInt16>.self)
                    let c_fp32 = fp16ToFp32_simd8(c_fp16)

                    dot.addProduct(q0, c_fp32.lowHalf)
                    dot.addProduct(q1, c_fp32.highHalf)

                    i += 2
                }

                while i < lanes {
                    let q = queryStorage[i]
                    let offset = i * 4

                    let c0 = fp16ToFp32_scalar(cPtr[offset])
                    let c1 = fp16ToFp32_scalar(cPtr[offset + 1])
                    let c2 = fp16ToFp32_scalar(cPtr[offset + 2])
                    let c3 = fp16ToFp32_scalar(cPtr[offset + 3])
                    let c = SIMD4<Float>(c0, c1, c2, c3)

                    dot.addProduct(q, c)

                    i += 1
                }

                out[outIdx] = dot.sum()
            }
        }
    }

    /// Validates public wrapper endpoints before count arithmetic or any output write.
    /// Unsafe pointer initialization and aliasing remain the caller's responsibility.
    @usableFromInline
    internal static func validateMixedCandidateRange(
        _ range: Range<Int>, candidateCount: Int, outputCount: Int
    ) -> Int {
        guard range.lowerBound >= 0 && range.upperBound <= candidateCount else {
            fatalError("MixedPrecision candidate range violation")
        }
        let count = range.upperBound - range.lowerBound
        guard outputCount >= count else {
            fatalError("MixedPrecision range output capacity violation")
        }
        return count
    }

    // MARK: - Public Range-Based API (Euclidean squared distance)

    /// Computes 512-dimensional Euclidean squared distance: Σᵢ (qᵢ - cᵢ)².
    /// Writes compact results: out[j] corresponds to candidatesFP16[range.lowerBound + j].
    /// The remaining output suffix is untouched. MixedPrecisionRangeWorkTests covers
    /// compact output, offset views, empty ranges, and named precondition diagnostics.
    /// - Complexity: O(range.count) temporary metadata; selected FP16 payloads stay shared.
    /// - Precondition: Range endpoints lie in 0...candidatesFP16.count and out.count
    ///   is at least range.count. Violations terminate before output writes. A valid
    ///   empty range returns without accessing query elements, candidates, or output.
    /// - Note: The caller must provide initialized, correctly aliased output storage.
    @inlinable
    public static func range_euclid2_mixed_512(
        query: Vector512Optimized,
        candidatesFP16: [Vector512FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let count = validateMixedCandidateRange(range, candidateCount: candidatesFP16.count, outputCount: out.count)
        guard count > 0 else { return }
        let candidatesArray = candidatesFP16[range].map { $0.storage }
        range_euclid2_mixed_core(
            queryStorage: query.storage,
            candidatesArray: candidatesArray,
            range: 0..<count,
            out: out,
            lanes: Vector512Optimized.lanes
        )
    }

    /// Computes 768-dimensional Euclidean squared distance: Σᵢ (qᵢ - cᵢ)².
    /// Writes compact results: out[j] corresponds to candidatesFP16[range.lowerBound + j].
    /// The remaining output suffix is untouched. MixedPrecisionRangeWorkTests covers
    /// compact output, offset views, empty ranges, and named precondition diagnostics.
    /// - Complexity: O(range.count) temporary metadata; selected FP16 payloads stay shared.
    /// - Precondition: Range endpoints lie in 0...candidatesFP16.count and out.count
    ///   is at least range.count. Violations terminate before output writes. A valid
    ///   empty range returns without accessing query elements, candidates, or output.
    /// - Note: The caller must provide initialized, correctly aliased output storage.
    @inlinable
    public static func range_euclid2_mixed_768(
        query: Vector768Optimized,
        candidatesFP16: [Vector768FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let count = validateMixedCandidateRange(range, candidateCount: candidatesFP16.count, outputCount: out.count)
        guard count > 0 else { return }
        let candidatesArray = candidatesFP16[range].map { $0.storage }
        range_euclid2_mixed_core(
            queryStorage: query.storage,
            candidatesArray: candidatesArray,
            range: 0..<count,
            out: out,
            lanes: Vector768Optimized.lanes
        )
    }

    /// Computes 1536-dimensional Euclidean squared distance: Σᵢ (qᵢ - cᵢ)².
    /// Writes compact results: out[j] corresponds to candidatesFP16[range.lowerBound + j].
    /// The remaining output suffix is untouched. MixedPrecisionRangeWorkTests covers
    /// compact output, offset views, empty ranges, and named precondition diagnostics.
    /// - Complexity: O(range.count) temporary metadata; selected FP16 payloads stay shared.
    /// - Precondition: Range endpoints lie in 0...candidatesFP16.count and out.count
    ///   is at least range.count. Violations terminate before output writes. A valid
    ///   empty range returns without accessing query elements, candidates, or output.
    /// - Note: The caller must provide initialized, correctly aliased output storage.
    @inlinable
    public static func range_euclid2_mixed_1536(
        query: Vector1536Optimized,
        candidatesFP16: [Vector1536FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let count = validateMixedCandidateRange(range, candidateCount: candidatesFP16.count, outputCount: out.count)
        guard count > 0 else { return }
        let candidatesArray = candidatesFP16[range].map { $0.storage }
        range_euclid2_mixed_core(
            queryStorage: query.storage,
            candidatesArray: candidatesArray,
            range: 0..<count,
            out: out,
            lanes: Vector1536Optimized.lanes
        )
    }

    // MARK: - Public Range-Based API (cosine distance)

    /// Computes 512-dimensional cosine distance: 1 - (q·c)/(||q|| × ||c||).
    /// Writes compact results: out[j] corresponds to candidatesFP16[range.lowerBound + j].
    /// The remaining output suffix is untouched. MixedPrecisionRangeWorkTests covers
    /// compact output, offset views, empty ranges, and named precondition diagnostics.
    /// - Complexity: O(range.count) temporary metadata; selected FP16 payloads stay shared.
    /// - Precondition: Range endpoints lie in 0...candidatesFP16.count and out.count
    ///   is at least range.count. Violations terminate before output writes. A valid
    ///   empty range returns without accessing query elements, candidates, or output.
    /// - Note: The caller must provide initialized, correctly aliased output storage.
    ///   The core returns `Float.nan` when the query or candidate magnitude is not finite;
    ///   otherwise it uses similarity zero when the denominator is not finite or is at most
    ///   1e-9, then clamps similarity to [-1, 1] (exceptional inputs pinned against the core
    ///   by exceptionalSameCoreParity; the class by
    ///   T15CosineExceptionalTableTests.everyImplementationFollowsThePolicy).
    @inlinable
    public static func range_cosine_mixed_512(
        query: Vector512Optimized,
        candidatesFP16: [Vector512FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let count = validateMixedCandidateRange(range, candidateCount: candidatesFP16.count, outputCount: out.count)
        guard count > 0 else { return }
        let candidatesArray = candidatesFP16[range].map { $0.storage }
        range_cosine_mixed_core(
            queryStorage: query.storage,
            candidatesArray: candidatesArray,
            range: 0..<count,
            out: out,
            lanes: Vector512Optimized.lanes
        )
    }

    /// Computes 768-dimensional cosine distance: 1 - (q·c)/(||q|| × ||c||).
    /// Writes compact results: out[j] corresponds to candidatesFP16[range.lowerBound + j].
    /// The remaining output suffix is untouched. MixedPrecisionRangeWorkTests covers
    /// compact output, offset views, empty ranges, and named precondition diagnostics.
    /// - Complexity: O(range.count) temporary metadata; selected FP16 payloads stay shared.
    /// - Precondition: Range endpoints lie in 0...candidatesFP16.count and out.count
    ///   is at least range.count. Violations terminate before output writes. A valid
    ///   empty range returns without accessing query elements, candidates, or output.
    /// - Note: The caller must provide initialized, correctly aliased output storage.
    ///   The core returns `Float.nan` when the query or candidate magnitude is not finite;
    ///   otherwise it uses similarity zero when the denominator is not finite or is at most
    ///   1e-9, then clamps similarity to [-1, 1] (exceptional inputs pinned against the core
    ///   by exceptionalSameCoreParity; the class by
    ///   T15CosineExceptionalTableTests.everyImplementationFollowsThePolicy).
    @inlinable
    public static func range_cosine_mixed_768(
        query: Vector768Optimized,
        candidatesFP16: [Vector768FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let count = validateMixedCandidateRange(range, candidateCount: candidatesFP16.count, outputCount: out.count)
        guard count > 0 else { return }
        let candidatesArray = candidatesFP16[range].map { $0.storage }
        range_cosine_mixed_core(
            queryStorage: query.storage,
            candidatesArray: candidatesArray,
            range: 0..<count,
            out: out,
            lanes: Vector768Optimized.lanes
        )
    }

    /// Computes 1536-dimensional cosine distance: 1 - (q·c)/(||q|| × ||c||).
    /// Writes compact results: out[j] corresponds to candidatesFP16[range.lowerBound + j].
    /// The remaining output suffix is untouched. MixedPrecisionRangeWorkTests covers
    /// compact output, offset views, empty ranges, and named precondition diagnostics.
    /// - Complexity: O(range.count) temporary metadata; selected FP16 payloads stay shared.
    /// - Precondition: Range endpoints lie in 0...candidatesFP16.count and out.count
    ///   is at least range.count. Violations terminate before output writes. A valid
    ///   empty range returns without accessing query elements, candidates, or output.
    /// - Note: The caller must provide initialized, correctly aliased output storage.
    ///   The core returns `Float.nan` when the query or candidate magnitude is not finite;
    ///   otherwise it uses similarity zero when the denominator is not finite or is at most
    ///   1e-9, then clamps similarity to [-1, 1] (exceptional inputs pinned against the core
    ///   by exceptionalSameCoreParity; the class by
    ///   T15CosineExceptionalTableTests.everyImplementationFollowsThePolicy).
    @inlinable
    public static func range_cosine_mixed_1536(
        query: Vector1536Optimized,
        candidatesFP16: [Vector1536FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let count = validateMixedCandidateRange(range, candidateCount: candidatesFP16.count, outputCount: out.count)
        guard count > 0 else { return }
        let candidatesArray = candidatesFP16[range].map { $0.storage }
        range_cosine_mixed_core(
            queryStorage: query.storage,
            candidatesArray: candidatesArray,
            range: 0..<count,
            out: out,
            lanes: Vector1536Optimized.lanes
        )
    }

    // MARK: - Public Range-Based API (positive dot product)

    /// Computes 512-dimensional positive dot product: Σᵢ qᵢ × cᵢ.
    /// Writes compact results: out[j] corresponds to candidatesFP16[range.lowerBound + j].
    /// The remaining output suffix is untouched. MixedPrecisionRangeWorkTests covers
    /// compact output, offset views, empty ranges, and named precondition diagnostics.
    /// - Complexity: O(range.count) temporary metadata; selected FP16 payloads stay shared.
    /// - Precondition: Range endpoints lie in 0...candidatesFP16.count and out.count
    ///   is at least range.count. Violations terminate before output writes. A valid
    ///   empty range returns without accessing query elements, candidates, or output.
    /// - Note: The caller must provide initialized, correctly aliased output storage.
    @inlinable
    public static func range_dot_mixed_512(
        query: Vector512Optimized,
        candidatesFP16: [Vector512FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let count = validateMixedCandidateRange(range, candidateCount: candidatesFP16.count, outputCount: out.count)
        guard count > 0 else { return }
        let candidatesArray = candidatesFP16[range].map { $0.storage }
        range_dot_mixed_core(
            queryStorage: query.storage,
            candidatesArray: candidatesArray,
            range: 0..<count,
            out: out,
            lanes: Vector512Optimized.lanes
        )
    }

    /// Computes 768-dimensional positive dot product: Σᵢ qᵢ × cᵢ.
    /// Writes compact results: out[j] corresponds to candidatesFP16[range.lowerBound + j].
    /// The remaining output suffix is untouched. MixedPrecisionRangeWorkTests covers
    /// compact output, offset views, empty ranges, and named precondition diagnostics.
    /// - Complexity: O(range.count) temporary metadata; selected FP16 payloads stay shared.
    /// - Precondition: Range endpoints lie in 0...candidatesFP16.count and out.count
    ///   is at least range.count. Violations terminate before output writes. A valid
    ///   empty range returns without accessing query elements, candidates, or output.
    /// - Note: The caller must provide initialized, correctly aliased output storage.
    @inlinable
    public static func range_dot_mixed_768(
        query: Vector768Optimized,
        candidatesFP16: [Vector768FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let count = validateMixedCandidateRange(range, candidateCount: candidatesFP16.count, outputCount: out.count)
        guard count > 0 else { return }
        let candidatesArray = candidatesFP16[range].map { $0.storage }
        range_dot_mixed_core(
            queryStorage: query.storage,
            candidatesArray: candidatesArray,
            range: 0..<count,
            out: out,
            lanes: Vector768Optimized.lanes
        )
    }

    /// Computes 1536-dimensional positive dot product: Σᵢ qᵢ × cᵢ.
    /// Writes compact results: out[j] corresponds to candidatesFP16[range.lowerBound + j].
    /// The remaining output suffix is untouched. MixedPrecisionRangeWorkTests covers
    /// compact output, offset views, empty ranges, and named precondition diagnostics.
    /// - Complexity: O(range.count) temporary metadata; selected FP16 payloads stay shared.
    /// - Precondition: Range endpoints lie in 0...candidatesFP16.count and out.count
    ///   is at least range.count. Violations terminate before output writes. A valid
    ///   empty range returns without accessing query elements, candidates, or output.
    /// - Note: The caller must provide initialized, correctly aliased output storage.
    @inlinable
    public static func range_dot_mixed_1536(
        query: Vector1536Optimized,
        candidatesFP16: [Vector1536FP16],
        range: Range<Int>,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        let count = validateMixedCandidateRange(range, candidateCount: candidatesFP16.count, outputCount: out.count)
        guard count > 0 else { return }
        let candidatesArray = candidatesFP16[range].map { $0.storage }
        range_dot_mixed_core(
            queryStorage: query.storage,
            candidatesArray: candidatesArray,
            range: 0..<count,
            out: out,
            lanes: Vector1536Optimized.lanes
        )
    }

    // MARK: - Batch Conversion Helpers

    /// Convert an array of FP32 vectors to FP16 (half the scalar payload)
    ///
    /// **Use Case**: Pre-convert candidate sets once, reuse for multiple queries
    /// **Payload**: 512-dim scalars 2048 B → 1024 B per vector (excludes array headers and handles;
    /// payload model only, not process memory)
    @inlinable
    public static func convertToFP16_512(_ vectors: [Vector512Optimized]) -> [Vector512FP16] {
        return vectors.map { Vector512FP16(from: $0) }
    }

    @inlinable
    public static func convertToFP16_768(_ vectors: [Vector768Optimized]) -> [Vector768FP16] {
        return vectors.map { Vector768FP16(from: $0) }
    }

    @inlinable
    public static func convertToFP16_1536(_ vectors: [Vector1536Optimized]) -> [Vector1536FP16] {
        return vectors.map { Vector1536FP16(from: $0) }
    }

    // MARK: - Decision Heuristics

    /// Platform hints for `shouldUseMixedPrecision(candidateCount:dimension:platformHint:)`. The cache sizes
    /// in the case comments are unverified hardware assumptions.
    public enum PlatformHint: Sendable {
        case appleM1, appleM2  // assumed 8-12 MB last-level cache (unverified)
        case appleM3, appleM4  // assumed 16-24 MB last-level cache (unverified)
        case intel, amd        // cache varies; thresholds chosen without a measurement (unverified)
        case automatic         // Auto-detect based on generic heuristic
    }

    /// Heuristic: returns true when the FP32 candidate payload (candidateCount × dimension × 4 bytes) exceeds
    /// an assumed 12 MiB last-level cache and candidateCount >= 100.
    ///
    /// The cache size and both thresholds are unverified assumptions; no speedup is measured in this repository.
    ///
    /// - Parameters:
    ///   - candidateCount: Number of candidate vectors
    ///   - dimension: Vector dimensionality (512, 768, or 1536)
    /// - Returns: true if FP16 mixed precision should be used
    @inlinable
    public static func shouldUseMixedPrecision(
        candidateCount: Int,
        dimension: Int
    ) -> Bool {
        // FP32 scalar payload in bytes, as an Int64 product (a product beyond Int64 traps rather than wrapping)
        let fp32Footprint = Int64(candidateCount) * Int64(dimension) * Int64(MemoryLayout<Float>.size)

        // Assumed last-level cache size (unverified hardware assumption)
        let l3CacheSize: Int64 = 12 * 1024 * 1024  // 12 MB

        // Enable FP16 when:
        // 1. The FP32 payload exceeds the assumed cache size
        // 2. candidateCount >= 100
        let isMemoryBound = fp32Footprint > l3CacheSize
        let isBatchLargeEnough = candidateCount >= 100

        return isMemoryBound && isBatchLargeEnough
    }

    /// Per-hint thresholds: payload above 8 MiB (M1/M2), 16 MiB (M3/M4) or 16 MiB with candidateCount >= 200
    /// (Intel/AMD), and candidateCount >= 100 otherwise; `.automatic` uses the 12 MiB heuristic. The cache
    /// sizes behind these numbers are unverified hardware assumptions.
    ///
    /// - Parameters:
    ///   - candidateCount: Number of candidates
    ///   - dimension: Vector dimension
    ///   - platformHint: Selects the threshold pair
    /// - Returns: true if FP16 should be used for this platform
    @inlinable
    public static func shouldUseMixedPrecision(
        candidateCount: Int,
        dimension: Int,
        platformHint: PlatformHint
    ) -> Bool {
        let fp32Footprint = Int64(candidateCount) * Int64(dimension) * Int64(MemoryLayout<Float>.size)

        switch platformHint {
        case .appleM1, .appleM2:
            // Assumed 8-12 MB last-level cache (unverified)
            return fp32Footprint > (8 * 1024 * 1024) && candidateCount >= 100

        case .appleM3, .appleM4:
            // Assumed 16-24 MB last-level cache (unverified)
            return fp32Footprint > (16 * 1024 * 1024) && candidateCount >= 100

        case .intel, .amd:
            // x86: cache sizes vary; threshold and minimum batch chosen without a measurement (unverified)
            return fp32Footprint > (16 * 1024 * 1024) && candidateCount >= 200

        case .automatic:
            return shouldUseMixedPrecision(candidateCount: candidateCount, dimension: dimension)
        }
    }
}

// Import mach timing functions
#if canImport(Darwin)
import Darwin.Mach
#endif

// MARK: - FP16 Vector Pool and Platform Capabilities

// MARK: FP16 Vector Pool

/// A bounded cache of immutable FP16 vector values protected by an NSLock.
/// Acquire/release use the reserved array in O(1); this does not provide mutable
/// conversion-buffer reuse or eliminate the allocations of new conversions.
/// A released value can be acquired later with its existing contents
/// (MixedPrecisionPhase4Tests.testVectorPool_BasicOperations, testVectorPool_ConcurrentAccess).
public final class FP16VectorPool: @unchecked Sendable {
    private let capacity: Int
    private var available: [MixedPrecisionKernels.Vector512FP16]
    private let lock = NSLock()
    private let dimension: Int

    /// Initialize pool with pre-allocated FP16 vectors
    /// - Parameters:
    ///   - capacity: Maximum number of pooled vectors (pre-allocated)
    ///   - dimension: Vector dimension (currently only 512 supported)
    public init(capacity: Int, dimension: Int = 512) {
        precondition(dimension == 512, "FP16VectorPool currently only supports dimension 512")
        self.capacity = capacity
        self.dimension = dimension
        self.available = []

        // Pre-allocate pool with zero-initialized vectors
        self.available.reserveCapacity(capacity)
        for _ in 0..<capacity {
            let fp16Values = [UInt16](repeating: 0, count: dimension)
            let vector = MixedPrecisionKernels.Vector512FP16(fp16Values: fp16Values)
            available.append(vector)
        }
    }

    /// Acquire a vector from the pool
    /// - Returns: Vector512FP16 if available, nil if pool is empty
    /// - Complexity: O(1)
    public func acquire() -> MixedPrecisionKernels.Vector512FP16? {
        lock.lock()
        defer { lock.unlock() }
        return available.popLast()
    }

    /// Release a vector back to the pool
    /// - Parameter vector: Vector to return to pool
    /// - Note: If the pool is at capacity, this reference to `vector` is dropped; its payload is freed only when no
    ///   other copy of the value remains.
    /// - Complexity: O(1)
    public func release(_ vector: MixedPrecisionKernels.Vector512FP16) {
        lock.lock()
        defer { lock.unlock() }
        if available.count < capacity {
            available.append(vector)
        }
        // Otherwise drop this reference (the payload is freed only when no other copy remains)
    }

    /// Current number of available vectors in pool
    public var availableCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return available.count
    }

    /// Remove all pooled vectors (testVectorPool_Drain). `removeAll(keepingCapacity: true)` keeps the array's capacity,
    /// and each payload is freed only when no other copy of it remains.
    public func drain() {
        lock.lock()
        defer { lock.unlock() }
        available.removeAll(keepingCapacity: true)
    }
}

// MARK: - Platform Capabilities & Runtime Detection

/// Compile-time platform facts (by `#if arch`), not runtime CPU detection. No caller in Sources/ or Benchmarks/ uses
/// them for dispatch.
///
/// Architecture notes (cited, not verified here): arm64 has 128-bit NEON registers; AVX2 registers are 256-bit;
/// AVX-512 is not assumed.
///
/// # Usage
/// ```swift
/// if PlatformCapabilities.hasNativeHardwareFP16 {
///     // Use hardware FP16 conversion
/// } else {
///     // Fallback to software conversion
/// }
/// ```
public enum PlatformCapabilities {

    /// Compile-time answer: true on arm64 builds, false otherwise. That arm64 Apple silicon has FP16 arithmetic and
    /// common x86_64 chips do not is a cited hardware fact, not verified here.
    public static let hasNativeHardwareFP16: Bool = {
        #if arch(arm64)
        // All Apple Silicon has NEON with FP16 support
        return true
        #else
        // x86_64 generally lacks hardware FP16 (except Sapphire Rapids AVX-512 FP16)
        return false
        #endif
    }()

    /// Compile-time answer: true on arm64 builds (AArch64 includes Advanced SIMD), false otherwise.
    public static let hasNEON: Bool = {
        #if arch(arm64)
        return true
        #else
        return false
        #endif
    }()

    /// Compile-time answer: true on every x86_64 build, false otherwise. It assumes AVX2 without runtime detection,
    /// so it can be wrong on x86_64 CPUs without AVX2 (not verified).
    public static let hasAVX2: Bool = {
        #if arch(x86_64)
        // Assumed, not detected (unverified on the running CPU)
        return true
        #else
        return false
        #endif
    }()

    /// Always false: no AVX-512 detection is performed.
    public static let hasAVX512: Bool = {
        #if arch(x86_64)
        // Not detected; always false. (AVX-512 frequency and thermal behaviour is a cited concern, not verified here.)
        return false
        #else
        return false
        #endif
    }()

    /// SIMD register width in Float elements by architecture (compile-time; "optimal" is not measured)
    ///
    /// - ARM NEON: 4 elements (128-bit registers)
    /// - x86 AVX2: 8 elements (256-bit registers)
    /// - x86 AVX-512: 16 elements (512-bit registers, disabled)
    public static let optimalSIMDWidth: Int = {
        #if arch(arm64)
        return 4  // NEON 128-bit: SIMD4<Float>
        #elseif arch(x86_64)
        return 8  // AVX2 256-bit: SIMD8<Float>
        #else
        return 4  // Fallback to scalar (treated as SIMD4)
        #endif
    }()

    /// Platform identifier for heuristics and logging
    public static let platformName: String = {
        #if arch(arm64)
        return "Apple Silicon (ARM64 NEON)"
        #elseif arch(x86_64)
        return "Intel/AMD (x86_64 AVX)"
        #else
        return "Unknown Architecture"
        #endif
    }()

    /// Assumed last-level cache size in bytes by architecture (compile-time constant: 8 MiB on arm64, 16 MiB on
    /// x86_64, 4 MiB otherwise). These are unverified hardware assumptions, not detected sizes, and no caller in
    /// Sources/ or Benchmarks/ uses them.
    public static let estimatedL3CacheSize: Int = {
        #if arch(arm64)
        // Apple Silicon: conservative estimate for M1/M2
        return 8 * 1024 * 1024  // 8 MB
        #elseif arch(x86_64)
        // x86: assumed (unverified)
        return 16 * 1024 * 1024  // 16 MB
        #else
        return 4 * 1024 * 1024   // 4 MB fallback
        #endif
    }()
}

// MARK: - Mixed Precision Provider Integration

/// Internal DistanceProvider that routes 512-dim Euclidean batches of at least `minBatchSizeForMixedPrecision`
/// candidates through the FP16 SoA kernels; every other case uses FP32 kernels.
///
/// # Precision Selection
/// The only inputs are the candidate count (against the threshold, whose default differs by architecture),
/// the vector type and the metric. The stored `autoTuner` is not consulted.
///
/// # Example Usage
/// ```swift
/// let provider = MixedPrecisionProvider()
/// let distances = try await provider.batchDistance(
///     from: queryVector,
///     to: candidateVectors,
///     metric: .euclidean
/// )
/// ```
///
/// - Note: The optional MixedPrecisionAutoTuner is stored but not used by `distance` or `batchDistance`.
internal struct MixedPrecisionProvider: DistanceProvider {

    /// Batch-size threshold: below it, FP32 is used. The conversion-overhead rationale is not verified.
    private let minBatchSizeForMixedPrecision: Int

    /// AutoTuner reference (stored; not consulted by distance or batchDistance)
    private let autoTuner: MixedPrecisionAutoTuner?

    /// Initialize provider with custom thresholds
    /// - Parameters:
    ///   - minBatchSize: Minimum batch size to enable FP16 (default: 32 on ARM, 64 on x86)
    ///   - autoTuner: Optional AutoTuner (stored, currently unused)
    public init(
        minBatchSize: Int? = nil,
        autoTuner: MixedPrecisionAutoTuner? = nil
    ) {
        // Platform-specific defaults
        if let minBatchSize = minBatchSize {
            self.minBatchSizeForMixedPrecision = minBatchSize
        } else {
            #if arch(arm64)
            // arm64 default 32 (rationale not verified; package builds use the software FP16 conversion)
            self.minBatchSizeForMixedPrecision = 32
            #else
            // other architectures: default 64 (rationale not verified)
            self.minBatchSizeForMixedPrecision = 64
            #endif
        }
        self.autoTuner = autoTuner
    }

    /// Compute pairwise distance between two vectors
    /// - Parameters:
    ///   - vector1: First vector
    ///   - vector2: Second vector
    ///   - metric: Distance metric (only euclidean and cosine supported)
    /// - Returns: Distance value
    public func distance<T: VectorProtocol>(
        from vector1: T,
        to vector2: T,
        metric: SupportedDistanceMetric
    ) async throws -> Float where T.Scalar == Float {

        // Single distances always use the FP32 kernels (no FP16 path); no cost comparison is measured
        switch metric {
        case .euclidean:
            if let v1 = vector1 as? Vector512Optimized, let v2 = vector2 as? Vector512Optimized {
                return EuclideanKernels.distance512(v1, v2)
            } else {
                return vector1.distance(to: vector2, metric: .euclidean)
            }

        case .cosine:
            if let v1 = vector1 as? Vector512Optimized, let v2 = vector2 as? Vector512Optimized {
                return CosineKernels.distance512_fused(v1, v2)
            } else {
                return vector1.distance(to: vector2, metric: .cosine)
            }

        default:
            // Fallback to standard protocol implementation
            return vector1.distance(to: vector2, metric: metric)
        }
    }

    /// Compute batch distances; FP16 is used only for 512-dim Euclidean batches at or above the threshold
    /// - Parameters:
    ///   - query: Query vector
    ///   - candidates: Array of candidate vectors
    ///   - metric: Distance metric (only euclidean has an FP16 path; cosine and the rest use FP32)
    /// - Returns: Array of distances
    public func batchDistance<T: VectorProtocol>(
        from query: T,
        to candidates: [T],
        metric: SupportedDistanceMetric
    ) async throws -> [Float] where T.Scalar == Float {

        // Check if we should use mixed precision
        let useMixedPrecision = candidates.count >= minBatchSizeForMixedPrecision

        // Only Vector512Optimized has an FP16 path
        guard let query512 = query as? Vector512Optimized,
              let candidates512 = exactArray(candidates, as: Vector512Optimized.self),
              useMixedPrecision else {
            // Fallback to standard per-vector computation
            return candidates.map { query.distance(to: $0, metric: metric) }
        }

        // Use mixed precision kernels for supported metrics
        switch metric {
        case .euclidean:
            return await computeBatchEuclideanFP16(query: query512, candidates: candidates512)

        case .cosine:
            // Cosine FP16 not yet implemented - fallback to FP32
            return candidates512.map { CosineKernels.distance512_fused(query512, $0) }

        default:
            // Other metrics: fallback to standard implementation
            return candidates.map { query.distance(to: $0, metric: metric) }
        }
    }

    /// Internal: Compute batch Euclidean distances using FP16 SoA kernels
    private func computeBatchEuclideanFP16(
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) async -> [Float] {

        // Convert candidates to SoA FP16 layout
        let candidatesSoA = MixedPrecisionKernels.createSoA512FP16(from: candidates)

        // Allocate results buffer
        var results = [Float](repeating: 0, count: candidates.count)

        // Decide whether to use blocked kernel based on batch size
        let useBlockedKernel = candidates.count >= 16

        results.withUnsafeMutableBufferPointer { buffer in
            if useBlockedKernel {
                // Register-blocked kernel for 16 or more candidates (no speed comparison is measured)
                MixedPrecisionKernels.batchEuclideanBlocked512(
                    query: query,
                    candidates: candidatesSoA,
                    results: buffer
                )
            } else {
                // Use standard kernel for small batches
                MixedPrecisionKernels.batchEuclidean512(
                    query: query,
                    candidates: candidatesSoA,
                    results: buffer
                )
            }
        }

        return results
    }
}

// MARK: - Platform-Specific Conversion Paths

extension MixedPrecisionKernels {

    /// FP32→FP16 batch conversion with a per-architecture loop
    ///
    /// arm64 converts the non-NaN lanes of SIMD4-sized chunks through `Float16(_:)`, and NaN lanes and the tail through
    /// `fp32ToFp16_scalar`; x86_64 uses the software `fp32ToFp16_scalar` in SIMD4-sized chunks; other architectures use
    /// the scalar loop. Which instructions the compiler emits, and their latency, are not verified.
    ///
    /// NaN policy: every NaN input becomes sign | 0x7E00, in chunks and tail alike; no payload is kept
    /// (FP16ConverterExhaustiveTests.nanEncodePlatformBatchChunks, nanEncodePlatformBatchTail). On the round-trip and
    /// rounding inputs of roundTripAllNonNaNPatterns and encodeRoundToNearestEvenNeighbours, chunk and tail lanes equal
    /// `fp32ToFp16_scalar`. Those tests run the loop of the host architecture in the default floating-point
    /// environment; agreement of the arm64 chunk lanes under another rounding or flush-to-zero mode is not verified.
    ///
    /// - Parameters:
    ///   - source: FP32 source buffer
    ///   - destination: FP16 destination buffer (as UInt16 bit patterns)
    /// - Complexity: O(n) where n = source.count
    @inlinable
    public static func platformOptimizedConvertBatch(
        source: UnsafeBufferPointer<Float>,
        destination: UnsafeMutableBufferPointer<UInt16>
    ) {
        precondition(source.count == destination.count, "Source and destination buffers must have same count")

        #if arch(arm64)
        // arm64 path: Float16(_:) for non-NaN chunk lanes, fp32ToFp16_scalar for NaN lanes and the tail (instruction
        // selection not verified)
        convertBatchNEON(source: source, destination: destination)
        #elseif arch(x86_64)
        // x86_64 path: software conversion in SIMD4-sized chunks (vectorization not verified)
        convertBatchAVX2(source: source, destination: destination)
        #else
        // Fallback: scalar conversion
        for i in 0..<source.count {
            destination[i] = fp32ToFp16_scalar(source[i])
        }
        #endif
    }

    #if arch(arm64)
    /// arm64 FP16 conversion: `Float16(_:)` for the non-NaN lanes of each SIMD4 chunk, and `fp32ToFp16_scalar` for NaN
    /// lanes (sign | 0x7E00, the NaN policy in the type doc) and for the tail
    /// - Note: The emitted instruction is not verified.
    @inlinable
    static func convertBatchNEON(
        source: UnsafeBufferPointer<Float>,
        destination: UnsafeMutableBufferPointer<UInt16>
    ) {
        // Process in SIMD4 chunks (NEON 128-bit registers)
        let count = source.count
        var i = 0

        // SIMD4 main loop
        while i + 3 < count {
            let fp32Vec = SIMD4<Float>(
                source[i + 0],
                source[i + 1],
                source[i + 2],
                source[i + 3]
            )

            // A NaN lane takes fp32ToFp16_scalar, which returns sign | 0x7E00 (the NaN policy in the type doc); any
            // other lane converts through Float16(_:) (instruction selection not verified)
            destination[i + 0] = fp32Vec[0].isNaN ? fp32ToFp16_scalar(fp32Vec[0]) : Float16(fp32Vec[0]).bitPattern
            destination[i + 1] = fp32Vec[1].isNaN ? fp32ToFp16_scalar(fp32Vec[1]) : Float16(fp32Vec[1]).bitPattern
            destination[i + 2] = fp32Vec[2].isNaN ? fp32ToFp16_scalar(fp32Vec[2]) : Float16(fp32Vec[2]).bitPattern
            destination[i + 3] = fp32Vec[3].isNaN ? fp32ToFp16_scalar(fp32Vec[3]) : Float16(fp32Vec[3]).bitPattern

            i += 4
        }

        // Scalar tail loop
        while i < count {
            destination[i] = fp32ToFp16_scalar(source[i])
            i += 1
        }
    }
    #endif

    #if arch(x86_64)
    /// x86_64 software FP16 conversion in SIMD4-sized chunks. No AVX2-specific code is used; whether the compiler
    /// vectorizes the loop is not verified.
    @inlinable
    static func convertBatchAVX2(
        source: UnsafeBufferPointer<Float>,
        destination: UnsafeMutableBufferPointer<UInt16>
    ) {
        // AVX2 has 256-bit registers = SIMD8<Float>
        // However, Swift SIMD doesn't expose AVX2 directly, so fall back to SIMD4
        let count = source.count
        var i = 0

        // Process in SIMD4-sized chunks (auto-vectorization not verified)
        while i + 3 < count {
            let fp32Vec = SIMD4<Float>(
                source[i + 0],
                source[i + 1],
                source[i + 2],
                source[i + 3]
            )

            // Software conversion (vectorization not verified)
            destination[i + 0] = fp32ToFp16_scalar(fp32Vec[0])
            destination[i + 1] = fp32ToFp16_scalar(fp32Vec[1])
            destination[i + 2] = fp32ToFp16_scalar(fp32Vec[2])
            destination[i + 3] = fp32ToFp16_scalar(fp32Vec[3])

            i += 4
        }

        // Scalar tail
        while i < count {
            destination[i] = fp32ToFp16_scalar(source[i])
            i += 1
        }
    }
    #endif
}
