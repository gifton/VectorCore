//
//  SoA.swift
//  VectorCore
//
//  Structure-of-Arrays (SoA) storage for optimized batch scoring
//  Provides cache-friendly memory layout for vector similarity operations
//

import Foundation
import simd

// MARK: - Protocol for SoA Compatible Vectors

/// Protocol defining the requirements for vectors compatible with SoA layout
public protocol SoACompatible: Sendable {
    static var dimension: Int { get }
    static var lanes: Int { get }
    var storage: ContiguousArray<SIMD4<Float>> { get }
}

// Extend existing optimized vectors to be SoA compatible
extension Vector512Optimized: SoACompatible {
    public static var dimension: Int { 512 }
    public static var lanes: Int { 128 } // 512 / 4
}

extension Vector768Optimized: SoACompatible {
    public static var dimension: Int { 768 }
    public static var lanes: Int { 192 } // 768 / 4
}

extension Vector1536Optimized: SoACompatible {
    public static var dimension: Int { 1536 }
    public static var lanes: Int { 384 } // 1536 / 4
}

// MARK: - SoA Container

/// Structure-of-Arrays (SoA) container for optimized batch scoring.
///
/// Memory layout: lanes-major then candidate index.
/// `buffer[lane * N + j]` stores SIMD4 for candidate j at lane `lane`.
///
/// This provides superior cache locality when processing multiple candidates
/// with the same query vector, as all candidates' data for a given lane
/// are stored contiguously in memory.
public final class SoA<Vector: SoACompatible>: @unchecked Sendable {
    public let lanes: Int
    public let count: Int

    // Manually managed storage buffer for stable pointer access
    @usableFromInline internal let buffer: UnsafeMutablePointer<SIMD4<Float>>
    private let bufferCapacity: Int

    private init(count: Int) {
        self.count = count
        self.lanes = Vector.lanes
        self.bufferCapacity = self.lanes * self.count

        if bufferCapacity > 0 {
            // Allocate memory with proper alignment for SIMD4<Float> (16 bytes)
            self.buffer = UnsafeMutablePointer<SIMD4<Float>>.allocate(capacity: bufferCapacity)
            // Initialize to zero
            self.buffer.initialize(repeating: .zero, count: bufferCapacity)
        } else {
            // Handle empty case
            self.buffer = UnsafeMutablePointer<SIMD4<Float>>.allocate(capacity: 0)
        }
    }

    deinit {
        if bufferCapacity > 0 {
            buffer.deinitialize(count: bufferCapacity)
        }
        buffer.deallocate()
    }

    /// Builds the SoA structure from an Array-of-Structures (AoS) candidate set.
    ///
    /// Transforms the memory layout from [Candidate][Lane] to [Lane][Candidate]
    /// for optimal cache performance during batch scoring operations.
    public static func build(from candidates: [Vector]) -> SoA<Vector> {
        let N = candidates.count
        let soa = SoA<Vector>(count: N)

        guard N > 0 else { return soa }

        let lanes = Vector.lanes
        let bufferPtr = soa.buffer

        // Transposition loop: Transform AoS to SoA
        // Optimized for sequential candidate reading (cache-friendly for input)
        for j in 0..<N {
            let candidate = candidates[j]
            let candidateStorage = candidate.storage

            // Copy each lane from this candidate to the appropriate SoA location
            for i in 0..<lanes {
                // SoA destination index: lane * N + candidate_index
                let destinationIndex = i * N + j
                bufferPtr[destinationIndex] = candidateStorage[i]
            }
        }

        return soa
    }

    /// Returns a pointer to the start of the specified lane's data block.
    ///
    /// The resulting pointer provides access to `count` contiguous SIMD4<Float> elements,
    /// representing all candidates' data for the specified lane.
    ///
    /// This is the key optimization: all candidates for a given lane are contiguous,
    /// enabling efficient vectorized processing with minimal cache misses.
    @inlinable
    public func lanePointer(_ lane: Int) -> UnsafePointer<SIMD4<Float>> {
        assert(lane >= 0 && lane < self.lanes, "Lane index \(lane) out of bounds [0..<\(lanes)]")
        return UnsafePointer(buffer + (lane * count))
    }

    /// Memory footprint in bytes
    public var memoryFootprint: Int {
        return bufferCapacity * MemoryLayout<SIMD4<Float>>.size
    }

    /// Memory savings compared to AoS storage (for reference)
    public var memoryEfficiency: String {
        let aosSize = count * lanes * MemoryLayout<SIMD4<Float>>.size
        let soaSize = memoryFootprint
        return "SoA: \(soaSize) bytes, AoS: \(aosSize) bytes (same, but better cache locality)"
    }
}

// MARK: - Type Aliases for Specific Dimensions

/// SoA container for 512-dimensional optimized vectors
public typealias SoA512 = SoA<Vector512Optimized>

/// SoA container for 768-dimensional optimized vectors
public typealias SoA768 = SoA<Vector768Optimized>

/// SoA container for 1536-dimensional optimized vectors
public typealias SoA1536 = SoA<Vector1536Optimized>

// MARK: - Builder Extensions
// Note: Use SoA<VectorType>.build(from:) directly instead of these type aliases