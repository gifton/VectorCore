//
//  Vector384Optimized.swift
//  VectorCore
//
//  Optimized 384-dimensional vector with SIMD acceleration
//  Standard dimension for MiniLM and Sentence-BERT models
//

import Foundation
import simd
import Accelerate

/// Highly optimized 384-dimensional vector using SIMD
///
/// This is the standard dimension for MiniLM models (all-MiniLM-L6-v2) and
/// many Sentence-BERT variants, making it critical for EmbedKit integration.
///
/// Performance characteristics:
/// - Dot product: ~75ns (4x unrolled with multiple accumulators)
/// - Distance: ~90ns (optimized squared distance calculation)
/// - Normalization: ~110ns (using vDSP when beneficial)
///
/// Memory layout:
/// - Stored as 96 SIMD4<Float> vectors for optimal alignment
/// - 16-byte aligned for SIMD operations
/// - Cache-line optimized for Apple Silicon (24 cache lines)
///
public struct Vector384Optimized: Sendable {
    public typealias Scalar = Float

    /// Internal storage as SIMD4 chunks for optimal performance
    public var storage: ContiguousArray<SIMD4<Float>>

    /// Number of scalar elements (always 384)
    public let scalarCount: Int = 384

    // MARK: - Initialization

    /// Initialize with zeros
    @inlinable
    public init() {
        storage = ContiguousArray(repeating: SIMD4<Float>(), count: 96)
    }

    /// Initialize with repeating value
    @inlinable
    public init(repeating value: Scalar) {
        let simd4 = SIMD4<Float>(repeating: value)
        storage = ContiguousArray(repeating: simd4, count: 96)
    }

    /// Initialize from array with optimized bulk memory operations
    @inlinable
    public init(_ array: [Scalar]) throws {
        guard array.count == 384 else {
            throw VectorError.dimensionMismatch(expected: 384, actual: array.count)
        }

        storage = ContiguousArray<SIMD4<Float>>()
        storage.reserveCapacity(96)

        // Use bulk memory operations for efficient initialization
        array.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else {
                storage = ContiguousArray(repeating: SIMD4<Float>(), count: 96)
                return
            }

            // Cast the buffer to SIMD4 chunks for direct copying
            let simd4Buffer = UnsafeRawPointer(baseAddress).bindMemory(
                to: SIMD4<Float>.self,
                capacity: 96
            )

            // Bulk append using unsafe buffer
            storage.append(contentsOf: UnsafeBufferPointer(start: simd4Buffer, count: 96))
        }
    }

    /// Initialize from sequence
    @inlinable
    public init<S: Sequence>(_ sequence: S) throws where S.Element == Scalar {
        let array = Array(sequence)
        try self.init(array)
    }

    /// Initialize with generator function
    @inlinable
    public init(generator: (Int) throws -> Scalar) rethrows {
        storage = ContiguousArray<SIMD4<Float>>()
        storage.reserveCapacity(96)

        for i in stride(from: 0, to: 384, by: 4) {
            let simd4 = try SIMD4<Float>(
                generator(i),
                generator(i + 1),
                generator(i + 2),
                generator(i + 3)
            )
            storage.append(simd4)
        }
    }

    /// Initialize from SIMD storage directly
    @inlinable
    public init(fromSIMDStorage storage: ContiguousArray<SIMD4<Float>>) {
        precondition(storage.count == 96, "Storage must contain exactly 96 SIMD4 elements")
        self.storage = storage
    }

    // MARK: - Element Access

    /// Access individual elements
    @inlinable
    public subscript(index: Int) -> Scalar {
        get {
            precondition(index >= 0 && index < 384, "Index \(index) out of bounds [0..<384]")
            let vectorIndex = index >> 2  // Divide by 4
            let scalarIndex = index & 3   // Modulo 4
            return storage[vectorIndex][scalarIndex]
        }
        set {
            precondition(index >= 0 && index < 384, "Index \(index) out of bounds [0..<384]")
            let vectorIndex = index >> 2
            let scalarIndex = index & 3
            storage[vectorIndex][scalarIndex] = newValue
        }
    }
}

// MARK: - VectorProtocol Conformance

extension Vector384Optimized: VectorProtocol {
    public typealias Storage = ContiguousArray<SIMD4<Float>>

    /// Convert to array
    public func toArray() -> [Scalar] {
        var result = [Scalar]()
        result.reserveCapacity(384)

        storage.withUnsafeBufferPointer { buffer in
            let floatBuffer = UnsafeRawPointer(buffer.baseAddress!)
                .bindMemory(to: Float.self, capacity: 384)
            result.append(contentsOf: UnsafeBufferPointer(start: floatBuffer, count: 384))
        }

        return result
    }

    /// Access storage for reading
    @inlinable
    public func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try storage.withUnsafeBufferPointer { simd4Buffer in
            let floatBuffer = UnsafeRawPointer(simd4Buffer.baseAddress!)
                .bindMemory(to: Float.self, capacity: 384)
            return try body(UnsafeBufferPointer(start: floatBuffer, count: 384))
        }
    }

    /// Access storage for writing
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { simd4Buffer in
            let floatBuffer = UnsafeMutableRawPointer(simd4Buffer.baseAddress!)
                .bindMemory(to: Float.self, capacity: 384)
            return try body(UnsafeMutableBufferPointer(start: floatBuffer, count: 384))
        }
    }
}

// MARK: - Optimized SIMD Operations

extension Vector384Optimized {

    /// Highly optimized dot product using 4 accumulators
    /// Performance: ~75ns for 384 dimensions
    @inlinable
    @inline(__always)
    public func dotProduct(_ other: Vector384Optimized) -> Scalar {
        DotKernels.dot384(self, other)
    }

    /// Optimized squared Euclidean distance
    /// Performance: ~90ns for 384 dimensions
    @inlinable
    @inline(__always)
    public func euclideanDistanceSquared(to other: Vector384Optimized) -> Scalar {
        EuclideanKernels.squared384(self, other)
    }

    /// Euclidean distance (with square root)
    @inlinable
    public func euclideanDistance(to other: Vector384Optimized) -> Scalar {
        EuclideanKernels.distance384(self, other)
    }

    /// Magnitude squared using stable SIMD kernels
    ///
    /// Uses Kahan's two-pass scaling algorithm to prevent overflow.
    /// This is numerically stable but requires computing the square root internally.
    @inlinable
    public var magnitudeSquared: Scalar {
        NormalizeKernels.magnitudeSquared(storage: storage, laneCount: 96)
    }

    /// Magnitude (L2 norm) using stable SIMD kernels
    ///
    /// Uses Kahan's algorithm through optimized SIMD operations.
    /// Prevents overflow for large values (> sqrt(Float.max)).
    @inlinable
    public var magnitude: Scalar {
        NormalizeKernels.magnitude(storage: storage, laneCount: 96)
    }

    /// Normalized vector
    @inlinable
    public func normalized() -> Result<Vector384Optimized, VectorError> {
        NormalizeKernels.normalized384(self)
    }

    /// Normalized vector without error checking.
    ///
    /// Bypasses zero-vector validation for maximum performance in hot paths.
    /// - Precondition: `magnitude > 0` (asserted in debug builds only)
    /// - Warning: Calling on a zero vector produces NaN/Inf values
    @inlinable
    public func normalizedUnchecked() -> Vector384Optimized {
        NormalizeKernels.normalizedUnchecked384(self)
    }

    /// Cosine similarity
    @inlinable
    public func cosineSimilarity(to other: Vector384Optimized) -> Scalar {
        let dot = dotProduct(other)
        let magSelf = magnitude
        let magOther = other.magnitude

        guard magSelf > 0 && magOther > 0 else { return 0 }
        return dot / (magSelf * magOther)
    }
}

// MARK: - Arithmetic Operations

extension Vector384Optimized {

    /// Addition with SIMD
    @inlinable
    public static func + (lhs: Vector384Optimized, rhs: Vector384Optimized) -> Vector384Optimized {
        var result = Vector384Optimized()
        for i in 0..<96 {
            result.storage[i] = lhs.storage[i] + rhs.storage[i]
        }
        return result
    }

    /// Subtraction with SIMD
    @inlinable
    public static func - (lhs: Vector384Optimized, rhs: Vector384Optimized) -> Vector384Optimized {
        var result = Vector384Optimized()
        for i in 0..<96 {
            result.storage[i] = lhs.storage[i] - rhs.storage[i]
        }
        return result
    }

    /// Scalar multiplication
    @inlinable
    public static func * (lhs: Vector384Optimized, rhs: Scalar) -> Vector384Optimized {
        var result = Vector384Optimized()
        let scalar = SIMD4<Float>(repeating: rhs)
        for i in 0..<96 {
            result.storage[i] = lhs.storage[i] * scalar
        }
        return result
    }

    /// Scalar division
    @inlinable
    public static func / (lhs: Vector384Optimized, rhs: Scalar) -> Vector384Optimized {
        precondition(rhs != 0, "Division by zero")
        return lhs * (1.0 / rhs)
    }

    /// Element-wise multiplication (Hadamard product)
    @inlinable
    public static func .* (lhs: Vector384Optimized, rhs: Vector384Optimized) -> Vector384Optimized {
        var result = Vector384Optimized()
        for i in 0..<96 {
            result.storage[i] = lhs.storage[i] * rhs.storage[i]
        }
        return result
    }
}

// MARK: - Collection Conformance

extension Vector384Optimized: Collection {
    public typealias Index = Int
    public typealias Element = Scalar

    public var startIndex: Int { 0 }
    public var endIndex: Int { 384 }

    public func index(after i: Int) -> Int {
        i + 1
    }
}

// MARK: - Equatable & Hashable

extension Vector384Optimized: Equatable {
    @inlinable
    public static func == (lhs: Vector384Optimized, rhs: Vector384Optimized) -> Bool {
        for i in 0..<96 {
            if lhs.storage[i] != rhs.storage[i] {
                return false
            }
        }
        return true
    }
}

extension Vector384Optimized: Hashable {
    public func hash(into hasher: inout Hasher) {
        #if DEBUG
        assert(storage.count >= 96, "Invalid storage size for hashing")
        #endif

        // Hash first and last few SIMD4 vectors for efficiency
        hasher.combine(storage[0])
        hasher.combine(storage[1])
        hasher.combine(storage[94])
        hasher.combine(storage[95])
        hasher.combine(384) // Include dimension
    }
}

// MARK: - Codable

extension Vector384Optimized: Codable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        let array = try container.decode([Float].self)
        try self.init(array)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(toArray())
    }
}

// MARK: - Debug Support

extension Vector384Optimized: CustomDebugStringConvertible {
    public var debugDescription: String {
        let elements = self.prefix(10).map { String(format: "%.4f", $0) }
        return "Vector384Optimized[\(elements.joined(separator: ", ")), ... (384 total)]"
    }
}

// MARK: - OptimizedVector Conformance

extension Vector384Optimized: OptimizedVector {
    public static var laneCount: Int { 4 }
}

// MARK: - VectorType Conformance

extension Vector384Optimized: VectorType {}

// MARK: - AnalyzableVector Conformance

extension Vector384Optimized: MixedPrecisionKernels.AnalyzableVector {}

// MARK: - Arithmetic Extensions for Streaming K-means

extension Vector384Optimized {
    /// Add another vector element-wise
    @inlinable
    public func adding(_ other: Vector384Optimized) -> Vector384Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(96)

        for i in 0..<96 {
            result.append(storage[i] + other.storage[i])
        }

        return Vector384Optimized(fromSIMDStorage: result)
    }

    /// Subtract another vector element-wise
    @inlinable
    public func subtracting(_ other: Vector384Optimized) -> Vector384Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(96)

        for i in 0..<96 {
            result.append(storage[i] - other.storage[i])
        }

        return Vector384Optimized(fromSIMDStorage: result)
    }

    /// Scale by a scalar value
    @inlinable
    public func scaled(by scalar: Float) -> Vector384Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(96)

        let simdScalar = SIMD4<Float>(repeating: scalar)
        for i in 0..<96 {
            result.append(storage[i] * simdScalar)
        }

        return Vector384Optimized(fromSIMDStorage: result)
    }

    /// Static zero vector
    @inlinable
    public static var zero: Vector384Optimized {
        Vector384Optimized()
    }
}
