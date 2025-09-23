//
//  Vector1536Optimized.swift
//  VectorCore
//
//  Optimized 1536-dimensional vector with SIMD acceleration
//  Common dimension for large embedding models
//

import Foundation
import simd
import Accelerate

/// Highly optimized 1536-dimensional vector using SIMD
///
/// Performance characteristics:
/// - Dot product: ~300ns (4x unrolled with multiple accumulators)
/// - Distance: ~360ns (optimized squared distance calculation)
/// - Normalization: ~450ns (using vDSP when beneficial)
///
/// Memory layout:
/// - Stored as 384 SIMD4<Float> vectors for optimal alignment
/// - 16-byte aligned for SIMD operations
/// - Cache-line optimized for Apple Silicon
///
public struct Vector1536Optimized: Sendable {
    public typealias Scalar = Float

    /// Internal storage as SIMD4 chunks for optimal performance
    public var storage: ContiguousArray<SIMD4<Float>>

    /// Number of scalar elements (always 1536)
    public let scalarCount: Int = 1536

    // MARK: - Initialization

    /// Initialize with zeros
    @inlinable
    public init() {
        storage = ContiguousArray(repeating: SIMD4<Float>(), count: 384)
    }

    /// Initialize with repeating value
    @inlinable
    public init(repeating value: Scalar) {
        let simd4 = SIMD4<Float>(repeating: value)
        storage = ContiguousArray(repeating: simd4, count: 384)
    }

    /// Initialize from array with optimized bulk memory operations
    @inlinable
    public init(_ array: [Scalar]) throws {
        guard array.count == 1536 else {
            throw VectorError.dimensionMismatch(expected: 1536, actual: array.count)
        }

        storage = ContiguousArray<SIMD4<Float>>()
        storage.reserveCapacity(384)

        // Use bulk memory operations for efficient initialization
        array.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else {
                storage = ContiguousArray(repeating: SIMD4<Float>(), count: 384)
                return
            }

            // Cast the buffer to SIMD4 chunks for direct copying
            let simd4Buffer = UnsafeRawPointer(baseAddress).bindMemory(
                to: SIMD4<Float>.self,
                capacity: 384
            )

            // Bulk append using unsafe buffer
            storage.append(contentsOf: UnsafeBufferPointer(start: simd4Buffer, count: 384))
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
        storage.reserveCapacity(384)

        for i in stride(from: 0, to: 1536, by: 4) {
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
            precondition(index >= 0 && index < 1536, "Index \(index) out of bounds [0..<1536]")
            let vectorIndex = index >> 2  // Divide by 4
            let scalarIndex = index & 3   // Modulo 4
            return storage[vectorIndex][scalarIndex]
        }
        set {
            precondition(index >= 0 && index < 1536, "Index \(index) out of bounds [0..<1536]")
            let vectorIndex = index >> 2
            let scalarIndex = index & 3
            storage[vectorIndex][scalarIndex] = newValue
        }
    }
}

// MARK: - VectorProtocol Conformance

extension Vector1536Optimized: VectorProtocol {
    public typealias Storage = ContiguousArray<SIMD4<Float>>

    /// Convert to array
    public func toArray() -> [Scalar] {
        var result = [Scalar]()
        result.reserveCapacity(1536)

        storage.withUnsafeBufferPointer { buffer in
            let floatBuffer = UnsafeRawPointer(buffer.baseAddress!)
                .bindMemory(to: Float.self, capacity: 1536)
            result.append(contentsOf: UnsafeBufferPointer(start: floatBuffer, count: 1536))
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
                .bindMemory(to: Float.self, capacity: 1536)
            return try body(UnsafeBufferPointer(start: floatBuffer, count: 1536))
        }
    }

    /// Access storage for writing
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { simd4Buffer in
            let floatBuffer = UnsafeMutableRawPointer(simd4Buffer.baseAddress!)
                .bindMemory(to: Float.self, capacity: 1536)
            return try body(UnsafeMutableBufferPointer(start: floatBuffer, count: 1536))
        }
    }
}

// MARK: - Optimized SIMD Operations

extension Vector1536Optimized {

    /// Highly optimized dot product using 4 accumulators
    /// Performance: ~300ns for 1536 dimensions
    @inlinable
    @inline(__always)
    public func dotProduct(_ other: Vector1536Optimized) -> Scalar {
        DotKernels.dot1536(self, other)
    }

    /// Optimized squared Euclidean distance
    /// Performance: ~360ns for 1536 dimensions
    @inlinable
    @inline(__always)
    public func euclideanDistanceSquared(to other: Vector1536Optimized) -> Scalar {
        EuclideanKernels.squared1536(self, other)
    }

    /// Euclidean distance (with square root)
    @inlinable
    public func euclideanDistance(to other: Vector1536Optimized) -> Scalar {
        EuclideanKernels.distance1536(self, other)
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
    public func normalized() -> Result<Vector1536Optimized, VectorError> {
        NormalizeKernels.normalized1536(self)
    }

    /// Cosine similarity
    @inlinable
    public func cosineSimilarity(to other: Vector1536Optimized) -> Scalar {
        let dot = dotProduct(other)
        let magSelf = magnitude
        let magOther = other.magnitude

        guard magSelf > 0 && magOther > 0 else { return 0 }
        return dot / (magSelf * magOther)
    }
}

// MARK: - Arithmetic Operations

extension Vector1536Optimized {

    /// Addition with SIMD
    @inlinable
    public static func + (lhs: Vector1536Optimized, rhs: Vector1536Optimized) -> Vector1536Optimized {
        var result = Vector1536Optimized()
        for i in 0..<384 {
            result.storage[i] = lhs.storage[i] + rhs.storage[i]
        }
        return result
    }

    /// Subtraction with SIMD
    @inlinable
    public static func - (lhs: Vector1536Optimized, rhs: Vector1536Optimized) -> Vector1536Optimized {
        var result = Vector1536Optimized()
        for i in 0..<384 {
            result.storage[i] = lhs.storage[i] - rhs.storage[i]
        }
        return result
    }

    /// Scalar multiplication
    @inlinable
    public static func * (lhs: Vector1536Optimized, rhs: Scalar) -> Vector1536Optimized {
        var result = Vector1536Optimized()
        let scalar = SIMD4<Float>(repeating: rhs)
        for i in 0..<384 {
            result.storage[i] = lhs.storage[i] * scalar
        }
        return result
    }

    /// Scalar division
    @inlinable
    public static func / (lhs: Vector1536Optimized, rhs: Scalar) -> Vector1536Optimized {
        precondition(rhs != 0, "Division by zero")
        return lhs * (1.0 / rhs)
    }

    /// Element-wise multiplication (Hadamard product)
    @inlinable
    public static func .* (lhs: Vector1536Optimized, rhs: Vector1536Optimized) -> Vector1536Optimized {
        var result = Vector1536Optimized()
        for i in 0..<384 {
            result.storage[i] = lhs.storage[i] * rhs.storage[i]
        }
        return result
    }
}

// MARK: - Collection Conformance

extension Vector1536Optimized: Collection {
    public typealias Index = Int
    public typealias Element = Scalar

    public var startIndex: Int { 0 }
    public var endIndex: Int { 1536 }

    public func index(after i: Int) -> Int {
        i + 1
    }
}

// MARK: - Equatable & Hashable

extension Vector1536Optimized: Equatable {
    @inlinable
    public static func == (lhs: Vector1536Optimized, rhs: Vector1536Optimized) -> Bool {
        for i in 0..<384 {
            if lhs.storage[i] != rhs.storage[i] {
                return false
            }
        }
        return true
    }
}

extension Vector1536Optimized: Hashable {
    public func hash(into hasher: inout Hasher) {
        #if DEBUG
        assert(storage.count >= 384, "Invalid storage size for hashing")
        #endif

        // Hash first and last few SIMD4 vectors for efficiency
        hasher.combine(storage[0])
        hasher.combine(storage[1])
        hasher.combine(storage[382])
        hasher.combine(storage[383])
        hasher.combine(1536) // Include dimension
    }
}

// MARK: - Codable

extension Vector1536Optimized: Codable {
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

extension Vector1536Optimized: CustomDebugStringConvertible {
    public var debugDescription: String {
        let elements = self.prefix(10).map { String(format: "%.4f", $0) }
        return "Vector1536Optimized[\(elements.joined(separator: ", ")), ... (1536 total)]"
    }
}

// MARK: - OptimizedVector Conformance

extension Vector1536Optimized: OptimizedVector {
    public static var laneCount: Int { 4 }
}

// MARK: - VectorType Conformance

extension Vector1536Optimized: VectorType {}
