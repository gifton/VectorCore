//
//  Vector512Optimized.swift
//  VectorCore
//
//  Optimized 512-dimensional vector with SIMD acceleration
//  Ported from VectorStoreKit monolith with adaptations for VectorCore
//

import Foundation
import simd
import Accelerate

/// Highly optimized 512-dimensional vector using SIMD
///
/// Performance characteristics:
/// - Dot product: ~100ns (4x unrolled with multiple accumulators)
/// - Distance: ~120ns (optimized squared distance calculation)
/// - Normalization: ~150ns (using vDSP when beneficial)
///
/// Memory layout:
/// - Stored as 128 SIMD4<Float> vectors for optimal alignment
/// - 16-byte aligned for SIMD operations
/// - Cache-line optimized for Apple Silicon
///
public struct Vector512Optimized: Sendable {
    public typealias Scalar = Float

    /// Internal storage as SIMD4 chunks for optimal performance
    public var storage: ContiguousArray<SIMD4<Float>>

    /// Number of scalar elements (always 512)
    public let scalarCount: Int = 512

    // MARK: - Initialization

    /// Initialize with zeros
    @inlinable
    public init() {
        storage = ContiguousArray(repeating: SIMD4<Float>(), count: 128)
    }

    /// Initialize with repeating value
    @inlinable
    public init(repeating value: Scalar) {
        let simd4 = SIMD4<Float>(repeating: value)
        storage = ContiguousArray(repeating: simd4, count: 128)
    }

    /// Initialize from array with optimized bulk memory operations
    @inlinable
    public init(_ array: [Scalar]) throws {
        guard array.count == 512 else {
            throw VectorError.dimensionMismatch(expected: 512, actual: array.count)
        }

        storage = ContiguousArray<SIMD4<Float>>()
        storage.reserveCapacity(128)

        // Use bulk memory operations for efficient initialization
        array.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else {
                storage = ContiguousArray(repeating: SIMD4<Float>(), count: 128)
                return
            }

            // Cast the buffer to SIMD4 chunks for direct copying
            let simd4Buffer = UnsafeRawPointer(baseAddress).bindMemory(
                to: SIMD4<Float>.self,
                capacity: 128
            )

            // Bulk append using unsafe buffer
            storage.append(contentsOf: UnsafeBufferPointer(start: simd4Buffer, count: 128))
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
        storage.reserveCapacity(128)

        for i in stride(from: 0, to: 512, by: 4) {
            let simd4 = try SIMD4<Float>(
                generator(i),
                generator(i + 1),
                generator(i + 2),
                generator(i + 3)
            )
            storage.append(simd4)
        }
    }

    // MARK: - Element Access

    /// Access individual elements
    @inlinable
    public subscript(index: Int) -> Scalar {
        get {
            precondition(index >= 0 && index < 512, "Index \(index) out of bounds [0..<512]")
            let vectorIndex = index >> 2  // Divide by 4
            let scalarIndex = index & 3   // Modulo 4
            return storage[vectorIndex][scalarIndex]
        }
        set {
            precondition(index >= 0 && index < 512, "Index \(index) out of bounds [0..<512]")
            let vectorIndex = index >> 2
            let scalarIndex = index & 3
            storage[vectorIndex][scalarIndex] = newValue
        }
    }
}

// MARK: - VectorProtocol Conformance

extension Vector512Optimized: VectorProtocol {
    public typealias Storage = ContiguousArray<SIMD4<Float>>

    /// Convert to array
    public func toArray() -> [Scalar] {
        var result = [Scalar]()
        result.reserveCapacity(512)

        storage.withUnsafeBufferPointer { buffer in
            let floatBuffer = UnsafeRawPointer(buffer.baseAddress!)
                .bindMemory(to: Float.self, capacity: 512)
            result.append(contentsOf: UnsafeBufferPointer(start: floatBuffer, count: 512))
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
                .bindMemory(to: Float.self, capacity: 512)
            return try body(UnsafeBufferPointer(start: floatBuffer, count: 512))
        }
    }

    /// Access storage for writing
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { simd4Buffer in
            let floatBuffer = UnsafeMutableRawPointer(simd4Buffer.baseAddress!)
                .bindMemory(to: Float.self, capacity: 512)
            return try body(UnsafeMutableBufferPointer(start: floatBuffer, count: 512))
        }
    }
}

// MARK: - Optimized SIMD Operations

extension Vector512Optimized {

    /// Highly optimized dot product using 4 accumulators
    /// Performance: ~100ns for 512 dimensions
    @inlinable
    @inline(__always)
    public func dotProduct(_ other: Vector512Optimized) -> Scalar {
        DotKernels.dot512(self, other)
    }

    /// Optimized squared Euclidean distance
    /// Performance: ~120ns for 512 dimensions
    @inlinable
    @inline(__always)
    public func euclideanDistanceSquared(to other: Vector512Optimized) -> Scalar {
        EuclideanKernels.squared512(self, other)
    }

    /// Euclidean distance (with square root)
    @inlinable
    public func euclideanDistance(to other: Vector512Optimized) -> Scalar {
        EuclideanKernels.distance512(self, other)
    }

    /// Magnitude squared (more efficient when square root not needed)
    @inlinable
    public var magnitudeSquared: Scalar {
        dotProduct(self)
    }

    /// Magnitude (L2 norm)
    @inlinable
    public var magnitude: Scalar {
        sqrt(magnitudeSquared)
    }

    /// Normalized vector
    @inlinable
    public func normalized() -> Result<Vector512Optimized, VectorError> {
        NormalizeKernels.normalized512(self)
    }

    /// Cosine similarity
    @inlinable
    public func cosineSimilarity(to other: Vector512Optimized) -> Scalar {
        let dot = dotProduct(other)
        let magSelf = magnitude
        let magOther = other.magnitude

        guard magSelf > 0 && magOther > 0 else { return 0 }
        return dot / (magSelf * magOther)
    }
}

// MARK: - Arithmetic Operations

extension Vector512Optimized {

    /// Addition with SIMD
    @inlinable
    public static func + (lhs: Vector512Optimized, rhs: Vector512Optimized) -> Vector512Optimized {
        var result = Vector512Optimized()
        for i in 0..<128 {
            result.storage[i] = lhs.storage[i] + rhs.storage[i]
        }
        return result
    }

    /// Subtraction with SIMD
    @inlinable
    public static func - (lhs: Vector512Optimized, rhs: Vector512Optimized) -> Vector512Optimized {
        var result = Vector512Optimized()
        for i in 0..<128 {
            result.storage[i] = lhs.storage[i] - rhs.storage[i]
        }
        return result
    }

    /// Scalar multiplication
    @inlinable
    public static func * (lhs: Vector512Optimized, rhs: Scalar) -> Vector512Optimized {
        var result = Vector512Optimized()
        let scalar = SIMD4<Float>(repeating: rhs)
        for i in 0..<128 {
            result.storage[i] = lhs.storage[i] * scalar
        }
        return result
    }

    /// Scalar division
    @inlinable
    public static func / (lhs: Vector512Optimized, rhs: Scalar) -> Vector512Optimized {
        precondition(rhs != 0, "Division by zero")
        return lhs * (1.0 / rhs)
    }

    /// Element-wise multiplication (Hadamard product)
    @inlinable
    public static func .* (lhs: Vector512Optimized, rhs: Vector512Optimized) -> Vector512Optimized {
        var result = Vector512Optimized()
        for i in 0..<128 {
            result.storage[i] = lhs.storage[i] * rhs.storage[i]
        }
        return result
    }
}

// MARK: - Collection Conformance

extension Vector512Optimized: Collection {
    public typealias Index = Int
    public typealias Element = Scalar

    public var startIndex: Int { 0 }
    public var endIndex: Int { 512 }

    public func index(after i: Int) -> Int {
        i + 1
    }
}

// MARK: - Equatable & Hashable

extension Vector512Optimized: Equatable {
    @inlinable
    public static func == (lhs: Vector512Optimized, rhs: Vector512Optimized) -> Bool {
        for i in 0..<128 {
            if lhs.storage[i] != rhs.storage[i] {
                return false
            }
        }
        return true
    }
}

extension Vector512Optimized: Hashable {
    public func hash(into hasher: inout Hasher) {
        #if DEBUG
        assert(storage.count >= 128, "Invalid storage size for hashing")
        #endif

        // Hash first and last few SIMD4 vectors for efficiency
        hasher.combine(storage[0])
        hasher.combine(storage[1])
        hasher.combine(storage[126])
        hasher.combine(storage[127])
        hasher.combine(512) // Include dimension
    }
}

// MARK: - Codable

extension Vector512Optimized: Codable {
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

extension Vector512Optimized: CustomDebugStringConvertible {
    public var debugDescription: String {
        let elements = self.prefix(10).map { String(format: "%.4f", $0) }
        return "Vector512Optimized[\(elements.joined(separator: ", ")), ... (512 total)]"
    }
}

// MARK: - VectorType Conformance

extension Vector512Optimized: VectorType {}
