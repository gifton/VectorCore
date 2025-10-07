// VectorCore: Aligned Memory Utilities
//
// Common utilities for memory alignment across storage types
//

import Foundation

/// Common alignment utilities for VectorCore storage types
internal enum AlignedMemory {
    /// Platform-specific optimal alignment for SIMD operations
    internal static var optimalAlignment: Int {
        #if arch(arm64)
        // Apple Silicon prefers 64-byte alignment (cache line size)
        return 64
        #elseif arch(x86_64)
        // Intel x86_64 typically uses 32-byte alignment for AVX
        // but 64-byte is still good for cache lines
        return 64
        #else
        // Conservative default
        return 16
        #endif
    }

    /// Minimum alignment required for SIMD operations
    internal static let minimumAlignment: Int = 16

    /// Check if a pointer is properly aligned
    @inlinable
    internal static func isAligned<T>(_ pointer: UnsafePointer<T>, to alignment: Int) -> Bool {
        Int(bitPattern: pointer) % alignment == 0
    }

    /// Check if a mutable pointer is properly aligned
    @inlinable
    internal static func isAligned<T>(_ pointer: UnsafeMutablePointer<T>, to alignment: Int) -> Bool {
        Int(bitPattern: pointer) % alignment == 0
    }

    /// Allocate aligned memory for Float arrays
    /// - Throws: VectorError.allocationFailed if allocation fails
    @inlinable
    internal static func allocateAligned(
        count: Int,
        alignment: Int = optimalAlignment
    ) throws -> UnsafeMutablePointer<Float> {
        try allocateAligned(type: Float.self, count: count, alignment: alignment)
    }

    /// Allocate aligned memory for any type
    /// - Throws: VectorError.allocationFailed if allocation fails
    @inlinable
    internal static func allocateAligned<T>(
        type: T.Type,
        count: Int,
        alignment: Int = optimalAlignment
    ) throws -> UnsafeMutablePointer<T> {
        precondition(alignment > 0 && (alignment & (alignment - 1)) == 0,
                     "Alignment must be a power of 2")
        precondition(alignment >= minimumAlignment,
                     "Alignment must be at least \(minimumAlignment) bytes")

        var rawPtr: UnsafeMutableRawPointer?
        let byteCount = count * MemoryLayout<T>.stride
        let result = posix_memalign(&rawPtr, alignment, byteCount)

        guard result == 0, let ptr = rawPtr else {
            throw VectorError.allocationFailed(size: byteCount, reason: "posix_memalign error \(result)")
        }

        return ptr.assumingMemoryBound(to: T.self)
    }

    /// Deallocate memory previously allocated via posix_memalign or aligned_alloc.
    ///
    /// - Parameter ptr: Pointer returned by allocateAligned or posix_memalign
    ///
    /// - Important: Memory allocated with `posix_memalign` MUST be freed with `free()`,
    ///   not with Swift's `.deallocate()`. Using `.deallocate()` on memory allocated
    ///   by `posix_memalign` causes undefined behavior and heap corruption.
    @inlinable
    internal static func deallocate<T>(_ ptr: UnsafeMutablePointer<T>) {
        free(UnsafeMutableRawPointer(ptr))
    }

    /// Deallocate raw memory previously allocated via posix_memalign or aligned_alloc.
    ///
    /// - Parameter ptr: Raw pointer returned by allocateAligned or posix_memalign
    ///
    /// - Important: Memory allocated with `posix_memalign` MUST be freed with `free()`,
    ///   not with Swift's `.deallocate()`.
    @inlinable
    internal static func deallocate(_ ptr: UnsafeMutableRawPointer) {
        free(ptr)
    }
}

/// Protocol for storage types that guarantee memory alignment
public protocol AlignedStorage {
    /// The alignment guarantee provided by this storage
    var guaranteedAlignment: Int { get }

    /// Verify that the storage maintains its alignment guarantee
    func verifyAlignment() -> Bool
}

// Make existing aligned storage types conform
extension AlignedValueStorage: AlignedStorage {
    public var guaranteedAlignment: Int { storage.alignment }

    public func verifyAlignment() -> Bool {
        withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return false }
            return AlignedMemory.isAligned(baseAddress, to: guaranteedAlignment)
        }
    }
}

// Note: The SmallVectorStorage extension has been removed as that type
// no longer exists. The dimension-specific storage types now use
// AlignedValueStorage which already handles alignment properly.
