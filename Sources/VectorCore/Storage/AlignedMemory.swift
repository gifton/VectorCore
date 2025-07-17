// VectorCore: Aligned Memory Utilities
//
// Common utilities for memory alignment across storage types
//

import Foundation

/// Common alignment utilities for VectorCore storage types
public enum AlignedMemory {
    /// Platform-specific optimal alignment for SIMD operations
    public static var optimalAlignment: Int {
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
    public static let minimumAlignment: Int = 16
    
    /// Check if a pointer is properly aligned
    @inlinable
    public static func isAligned<T>(_ pointer: UnsafePointer<T>, to alignment: Int) -> Bool {
        Int(bitPattern: pointer) % alignment == 0
    }
    
    /// Check if a mutable pointer is properly aligned
    @inlinable
    public static func isAligned<T>(_ pointer: UnsafeMutablePointer<T>, to alignment: Int) -> Bool {
        Int(bitPattern: pointer) % alignment == 0
    }
    
    /// Allocate aligned memory for Float arrays
    @inlinable
    public static func allocateAligned(
        count: Int,
        alignment: Int = optimalAlignment
    ) -> UnsafeMutablePointer<Float> {
        precondition(alignment > 0 && (alignment & (alignment - 1)) == 0,
                    "Alignment must be a power of 2")
        precondition(alignment >= minimumAlignment,
                    "Alignment must be at least \(minimumAlignment) bytes")
        
        var rawPtr: UnsafeMutableRawPointer?
        let byteCount = count * MemoryLayout<Float>.stride
        let result = posix_memalign(&rawPtr, alignment, byteCount)
        
        guard result == 0, let ptr = rawPtr else {
            fatalError("Failed to allocate aligned memory: error \(result)")
        }
        
        return ptr.assumingMemoryBound(to: Float.self)
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

// Extension to make SmallVectorStorage report its alignment
extension SmallVectorStorage: AlignedStorage {
    public var guaranteedAlignment: Int {
        // SIMD64 is naturally aligned to at least 16 bytes
        AlignedMemory.minimumAlignment
    }
    
    public func verifyAlignment() -> Bool {
        withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return false }
            return AlignedMemory.isAligned(baseAddress, to: guaranteedAlignment)
        }
    }
}