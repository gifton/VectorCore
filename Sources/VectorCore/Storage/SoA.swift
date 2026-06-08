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
extension Vector384Optimized: SoACompatible {
    public static var dimension: Int { 384 }
    public static var lanes: Int { 96 } // 384 / 4
}

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

    /// True when the buffer was allocated via AlignedMemory (posix_memalign) and so must be freed with free(); false for the default Swift allocation.
    private let usedAlignedAlloc: Bool

    /// Total allocated bytes. Page-rounded when page-aligned (the length to pass to
    /// makeBuffer(bytesNoCopy:)); otherwise bufferCapacity * sizeof(SIMD4<Float>).
    private let allocatedByteCount: Int

    private init(count: Int, pageAligned: Bool = false) {
        self.count = count
        self.lanes = Vector.lanes
        self.bufferCapacity = self.lanes * self.count
        let alloc = SoA.allocateBuffer(capacity: self.bufferCapacity, pageAligned: pageAligned)
        self.buffer = alloc.buffer
        self.usedAlignedAlloc = alloc.usedAlignedAlloc
        self.allocatedByteCount = alloc.allocatedBytes
    }

    /// Allocate (and zero) the SoA buffer. When `pageAligned` and non-empty, uses
    /// AlignedMemory (page-aligned base + page-rounded length) so the region is valid
    /// for MTLDevice.makeBuffer(bytesNoCopy:); otherwise the default 16-byte allocation.
    private static func allocateBuffer(
        capacity: Int, pageAligned: Bool
    ) -> (buffer: UnsafeMutablePointer<SIMD4<Float>>, usedAlignedAlloc: Bool, allocatedBytes: Int) {
        let elemSize = MemoryLayout<SIMD4<Float>>.stride   // 16
        if pageAligned && capacity > 0 {
            let page = PlatformConfiguration.pageSize
            let logical = capacity * elemSize
            let padded = ((logical + page - 1) / page) * page   // round up to a whole page
            let paddedCapacity = padded / elemSize
            let ptr: UnsafeMutablePointer<SIMD4<Float>>
            do {
                ptr = try AlignedMemory.allocateAligned(type: SIMD4<Float>.self,
                                                        count: paddedCapacity, alignment: page)
            } catch {
                fatalError("SoA: page-aligned allocation of \(padded) bytes failed: \(error)")
            }
            ptr.initialize(repeating: .zero, count: paddedCapacity)   // zero logical + padding
            return (ptr, true, padded)
        } else {
            let ptr = UnsafeMutablePointer<SIMD4<Float>>.allocate(capacity: max(0, capacity))
            if capacity > 0 { ptr.initialize(repeating: .zero, count: capacity) }
            return (ptr, false, max(0, capacity) * elemSize)
        }
    }

    deinit {
        if usedAlignedAlloc {
            // posix_memalign memory MUST be freed with free(), never .deallocate().
            AlignedMemory.deallocate(buffer)
        } else {
            if bufferCapacity > 0 { buffer.deinitialize(count: bufferCapacity) }
            buffer.deallocate()
        }
    }

    /// Public initializer for test compatibility: creates SoA from vectors
    ///
    /// - Parameters:
    ///   - vectors: Array of vectors to convert to SoA layout
    ///   - blockSize: Ignored parameter for API compatibility (reserved for future chunking optimizations)
    ///   - pageAligned: When true, allocates via posix_memalign with page-aligned base and
    ///     page-rounded length, making the region usable with MTLDevice.makeBuffer(bytesNoCopy:).
    public init(vectors: [Vector], blockSize: Int = 32, pageAligned: Bool = false) {
        let count = vectors.count
        self.count = count
        self.lanes = Vector.lanes
        self.bufferCapacity = self.lanes * self.count
        let alloc = SoA.allocateBuffer(capacity: self.bufferCapacity, pageAligned: pageAligned)
        self.buffer = alloc.buffer
        self.usedAlignedAlloc = alloc.usedAlignedAlloc
        self.allocatedByteCount = alloc.allocatedBytes

        // Populate the SoA structure from vectors
        guard count > 0 else { return }
        let lanes = Vector.lanes
        let bufferPtr = self.buffer
        for j in 0..<count {
            let vectorStorage = vectors[j].storage
            for i in 0..<lanes {
                bufferPtr[i * count + j] = vectorStorage[i]
            }
        }
    }

    /// Builds the SoA structure from an Array-of-Structures (AoS) candidate set.
    ///
    /// Transforms the memory layout from [Candidate][Lane] to [Lane][Candidate]
    /// for optimal cache performance during batch scoring operations.
    ///
    /// - Parameters:
    ///   - candidates: Source vectors in AoS layout.
    ///   - pageAligned: When true, allocates via posix_memalign with page-aligned base and
    ///     page-rounded length, enabling zero-copy GPU import via
    ///     `MTLDevice.makeBuffer(bytesNoCopy:)`. Access the pointer pair via `pageAlignedBytes`.
    public static func build(from candidates: [Vector], pageAligned: Bool = false) -> SoA<Vector> {
        let N = candidates.count
        let soa = SoA<Vector>(count: N, pageAligned: pageAligned)

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

    /// Page-aligned base pointer + page-rounded byte length for zero-copy GPU import
    /// via `MTLDevice.makeBuffer(bytesNoCopy:)`, or `nil` if this SoA was not built
    /// page-aligned (`build(from:pageAligned: true)` / `init(..., pageAligned: true)`).
    ///
    /// - Important: The `SoA` MUST outlive any `MTLBuffer` created from this pointer.
    ///   The memory is freed on `SoA` deinit, so hold a strong reference to the `SoA`
    ///   for the buffer's lifetime (or arrange a deallocator handshake).
    public var pageAlignedBytes: (base: UnsafeRawPointer, byteCount: Int)? {
        guard usedAlignedAlloc else { return nil }
        return (UnsafeRawPointer(buffer), allocatedByteCount)
    }

    /// Scoped read-only access to the raw SoA data (logical bytes = `count * lanes * 16`).
    /// The pointer is valid only for the duration of `body`.
    public func withUnsafeRawBuffer<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        let byteCount = bufferCapacity * MemoryLayout<SIMD4<Float>>.stride
        return try body(UnsafeRawBufferPointer(start: UnsafeRawPointer(buffer), count: byteCount))
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

/// SoA container for 384-dimensional optimized vectors (MiniLM, SBERT)
public typealias SoA384 = SoA<Vector384Optimized>

/// SoA container for 512-dimensional optimized vectors
public typealias SoA512 = SoA<Vector512Optimized>

/// SoA container for 768-dimensional optimized vectors
public typealias SoA768 = SoA<Vector768Optimized>

/// SoA container for 1536-dimensional optimized vectors
public typealias SoA1536 = SoA<Vector1536Optimized>

// MARK: - Builder Extensions
// Note: Use SoA<VectorType>.build(from:) directly instead of these type aliases
