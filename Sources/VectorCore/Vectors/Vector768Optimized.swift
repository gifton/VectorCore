//
//  Vector768Optimized.swift
//  VectorCore
//
//  Optimized 768-dimensional vector with SIMD acceleration
//  Common dimension for OpenAI embeddings
//

import Foundation
import simd
import Accelerate

/// Highly optimized 768-dimensional vector using SIMD
///
/// Performance characteristics:
/// - Dot product: ~150ns (4x unrolled with multiple accumulators)
/// - Distance: ~180ns (optimized squared distance calculation)
/// - Normalization: ~225ns (using vDSP when beneficial)
///
/// Memory layout:
/// - Stored as 192 SIMD4<Float> vectors for optimal alignment
/// - 16-byte aligned for SIMD operations
/// - Cache-line optimized for Apple Silicon
///
public struct Vector768Optimized: Sendable {
    public typealias Scalar = Float

    /// Internal storage as SIMD4 chunks for optimal performance
    public var storage: ContiguousArray<SIMD4<Float>>

    /// Number of scalar elements (always 768)
    public let scalarCount: Int = 768

    // MARK: - Initialization

    /// Initialize with zeros
    @inlinable
    public init() {
        storage = ContiguousArray(repeating: SIMD4<Float>(), count: 192)
    }

    /// Initialize with repeating value
    @inlinable
    public init(repeating value: Scalar) {
        let simd4 = SIMD4<Float>(repeating: value)
        storage = ContiguousArray(repeating: simd4, count: 192)
    }

    /// Initialize from array with optimized bulk memory operations
    @inlinable
    public init(_ array: [Scalar]) throws {
        guard array.count == 768 else {
            throw VectorError.dimensionMismatch(expected: 768, actual: array.count)
        }

        storage = ContiguousArray<SIMD4<Float>>()
        storage.reserveCapacity(192)

        // Use bulk memory operations for efficient initialization
        array.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else {
                storage = ContiguousArray(repeating: SIMD4<Float>(), count: 192)
                return
            }

            // Cast the buffer to SIMD4 chunks for direct copying
            let simd4Buffer = UnsafeRawPointer(baseAddress).bindMemory(
                to: SIMD4<Float>.self,
                capacity: 192
            )

            // Bulk append using unsafe buffer
            storage.append(contentsOf: UnsafeBufferPointer(start: simd4Buffer, count: 192))
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
        storage.reserveCapacity(192)

        for i in stride(from: 0, to: 768, by: 4) {
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
        precondition(storage.count == 192, "Storage must contain exactly 192 SIMD4 elements")
        self.storage = storage
    }

    // MARK: - Element Access

    /// Access individual elements
    @inlinable
    public subscript(index: Int) -> Scalar {
        get {
            precondition(index >= 0 && index < 768, "Index \(index) out of bounds [0..<768]")
            let vectorIndex = index >> 2  // Divide by 4
            let scalarIndex = index & 3   // Modulo 4
            return storage[vectorIndex][scalarIndex]
        }
        set {
            precondition(index >= 0 && index < 768, "Index \(index) out of bounds [0..<768]")
            let vectorIndex = index >> 2
            let scalarIndex = index & 3
            storage[vectorIndex][scalarIndex] = newValue
        }
    }
}

// MARK: - VectorProtocol Conformance

extension Vector768Optimized: VectorProtocol {
    public typealias Storage = ContiguousArray<SIMD4<Float>>

    /// Convert to array
    public func toArray() -> [Scalar] {
        var result = [Scalar]()
        result.reserveCapacity(768)

        storage.withUnsafeBufferPointer { buffer in
            let floatBuffer = UnsafeRawPointer(buffer.baseAddress!)
                .bindMemory(to: Float.self, capacity: 768)
            result.append(contentsOf: UnsafeBufferPointer(start: floatBuffer, count: 768))
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
                .bindMemory(to: Float.self, capacity: 768)
            return try body(UnsafeBufferPointer(start: floatBuffer, count: 768))
        }
    }

    /// Access storage for writing
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { simd4Buffer in
            let floatBuffer = UnsafeMutableRawPointer(simd4Buffer.baseAddress!)
                .bindMemory(to: Float.self, capacity: 768)
            return try body(UnsafeMutableBufferPointer(start: floatBuffer, count: 768))
        }
    }
}

// MARK: - Optimized SIMD Operations

extension Vector768Optimized {

    /// Highly optimized dot product using 4 accumulators
    /// Performance: ~150ns for 768 dimensions
    @inlinable
    @inline(__always)
    public func dotProduct(_ other: Vector768Optimized) -> Scalar {
        DotKernels.dot768(self, other)
    }

    /// Optimized squared Euclidean distance
    /// Performance: ~180ns for 768 dimensions
    @inlinable
    @inline(__always)
    public func euclideanDistanceSquared(to other: Vector768Optimized) -> Scalar {
        EuclideanKernels.squared768(self, other)
    }

    /// Euclidean distance (with square root)
    @inlinable
    public func euclideanDistance(to other: Vector768Optimized) -> Scalar {
        EuclideanKernels.distance768(self, other)
    }

    /// Magnitude squared using stable SIMD kernels
    ///
    /// Uses Kahan's two-pass scaling algorithm to prevent overflow.
    /// This is numerically stable but requires computing the square root internally.
    @inlinable
    public var magnitudeSquared: Scalar {
        NormalizeKernels.magnitudeSquared(storage: storage, laneCount: 192)
    }

    /// Magnitude (L2 norm) using stable SIMD kernels
    ///
    /// Uses Kahan's algorithm through optimized SIMD operations.
    /// Prevents overflow for large values (> sqrt(Float.max)).
    @inlinable
    public var magnitude: Scalar {
        NormalizeKernels.magnitude(storage: storage, laneCount: 192)
    }

    /// Normalized vector
    @inlinable
    public func normalized() -> Result<Vector768Optimized, VectorError> {
        NormalizeKernels.normalized768(self)
    }

    /// Normalized vector without error checking.
    ///
    /// Bypasses zero-vector validation for maximum performance in hot paths.
    /// - Precondition: `magnitude > 0` (asserted in debug builds only)
    /// - Warning: Calling on a zero vector produces NaN/Inf values
    @inlinable
    public func normalizedUnchecked() -> Vector768Optimized {
        NormalizeKernels.normalizedUnchecked768(self)
    }

    /// Cosine similarity
    @inlinable
    public func cosineSimilarity(to other: Vector768Optimized) -> Scalar {
        let dot = dotProduct(other)
        let magSelf = magnitude
        let magOther = other.magnitude

        guard magSelf > 0 && magOther > 0 else { return 0 }
        return dot / (magSelf * magOther)
    }
}

// MARK: - Arithmetic Operations

extension Vector768Optimized {

    /// Addition with SIMD
    @inlinable
    public static func + (lhs: Vector768Optimized, rhs: Vector768Optimized) -> Vector768Optimized {
        var result = Vector768Optimized()
        for i in 0..<192 {
            result.storage[i] = lhs.storage[i] + rhs.storage[i]
        }
        return result
    }

    /// Subtraction with SIMD
    @inlinable
    public static func - (lhs: Vector768Optimized, rhs: Vector768Optimized) -> Vector768Optimized {
        var result = Vector768Optimized()
        for i in 0..<192 {
            result.storage[i] = lhs.storage[i] - rhs.storage[i]
        }
        return result
    }

    /// Scalar multiplication
    @inlinable
    public static func * (lhs: Vector768Optimized, rhs: Scalar) -> Vector768Optimized {
        var result = Vector768Optimized()
        let scalar = SIMD4<Float>(repeating: rhs)
        for i in 0..<192 {
            result.storage[i] = lhs.storage[i] * scalar
        }
        return result
    }

    /// Scalar division
    @inlinable
    public static func / (lhs: Vector768Optimized, rhs: Scalar) -> Vector768Optimized {
        precondition(rhs != 0, "Division by zero")
        return lhs * (1.0 / rhs)
    }

    /// Element-wise multiplication (Hadamard product)
    @inlinable
    public static func .* (lhs: Vector768Optimized, rhs: Vector768Optimized) -> Vector768Optimized {
        var result = Vector768Optimized()
        for i in 0..<192 {
            result.storage[i] = lhs.storage[i] * rhs.storage[i]
        }
        return result
    }
}

// MARK: - Collection Conformance

extension Vector768Optimized: Collection {
    public typealias Index = Int
    public typealias Element = Scalar

    public var startIndex: Int { 0 }
    public var endIndex: Int { 768 }

    public func index(after i: Int) -> Int {
        i + 1
    }
}

// MARK: - Equatable & Hashable

extension Vector768Optimized: Equatable {
    @inlinable
    public static func == (lhs: Vector768Optimized, rhs: Vector768Optimized) -> Bool {
        for i in 0..<192 {
            if lhs.storage[i] != rhs.storage[i] {
                return false
            }
        }
        return true
    }
}

extension Vector768Optimized: Hashable {
    public func hash(into hasher: inout Hasher) {
        #if DEBUG
        assert(storage.count >= 192, "Invalid storage size for hashing")
        #endif

        // Hash first and last few SIMD4 vectors for efficiency
        hasher.combine(storage[0])
        hasher.combine(storage[1])
        hasher.combine(storage[190])
        hasher.combine(storage[191])
        hasher.combine(768) // Include dimension
    }
}

// MARK: - Codable

extension Vector768Optimized: Codable {
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

extension Vector768Optimized: CustomDebugStringConvertible {
    public var debugDescription: String {
        let elements = self.prefix(10).map { String(format: "%.4f", $0) }
        return "Vector768Optimized[\(elements.joined(separator: ", ")), ... (768 total)]"
    }
}

// MARK: - OptimizedVector Conformance

extension Vector768Optimized: OptimizedVector {
    public static var laneCount: Int { 4 }
}

// MARK: - VectorType Conformance

extension Vector768Optimized: VectorType {}

// MARK: - AnalyzableVector Conformance

extension Vector768Optimized: MixedPrecisionKernels.AnalyzableVector {}

// MARK: - Arithmetic Extensions for Streaming K-means

extension Vector768Optimized {
    /// Add another vector element-wise
    @inlinable
    public func adding(_ other: Vector768Optimized) -> Vector768Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(192)

        for i in 0..<192 {
            result.append(storage[i] + other.storage[i])
        }

        return Vector768Optimized(fromSIMDStorage: result)
    }

    /// Subtract another vector element-wise
    @inlinable
    public func subtracting(_ other: Vector768Optimized) -> Vector768Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(192)

        for i in 0..<192 {
            result.append(storage[i] - other.storage[i])
        }

        return Vector768Optimized(fromSIMDStorage: result)
    }

    /// Scale by a scalar value
    @inlinable
    public func scaled(by scalar: Float) -> Vector768Optimized {
        var result = ContiguousArray<SIMD4<Float>>()
        result.reserveCapacity(192)

        let simdScalar = SIMD4<Float>(repeating: scalar)
        for i in 0..<192 {
            result.append(storage[i] * simdScalar)
        }

        return Vector768Optimized(fromSIMDStorage: result)
    }

    /// Static zero vector
    @inlinable
    public static var zero: Vector768Optimized {
        Vector768Optimized()
    }
}
