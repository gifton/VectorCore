//
//  PlatformConfiguration.swift
//  VectorCore
//
//  Centralized platform-specific configuration and capability detection
//

import Foundation

/// Platform-specific configuration and capability detection
public enum PlatformConfiguration {
    // MARK: - Cache Configuration

    /// L1 data cache line size in bytes
    public static var l1CacheLineSize: Int {
        #if arch(x86_64) || arch(arm64)
        return 64
        #else
        return 32
        #endif
    }

    /// L2 cache size in bytes (approximate)
    public static var l2CacheSize: Int {
        #if arch(x86_64)
        return 256 * 1024  // 256KB typical for Intel
        #elseif arch(arm64)
        return 128 * 1024  // 128KB typical for Apple Silicon
        #else
        return 64 * 1024   // Conservative default
        #endif
    }

    // MARK: - SIMD Capabilities

    /// Whether SIMD operations are available on this platform
    public static var hasSIMD: Bool {
        #if arch(x86_64) || arch(arm64)
        return true
        #else
        return false
        #endif
    }

    /// Optimal SIMD vector width in elements for Float
    public static var simdVectorWidth: Int {
        #if arch(x86_64)
        return 16  // AVX-512
        #elseif arch(arm64)
        return 4   // NEON
        #else
        return 1
        #endif
    }

    // MARK: - Memory Configuration

    /// Optimal memory alignment for SIMD operations
    public static var optimalAlignment: Int {
        #if arch(x86_64)
        return 64  // AVX-512 alignment
        #elseif arch(arm64)
        #if os(macOS) || os(iOS)
        return 128  // Apple Silicon AMX alignment
        #else
        return 16   // NEON alignment
        #endif
        #else
        return 8
        #endif
    }

    /// Page size for memory allocation
    public static var pageSize: Int {
        Int(getpagesize())
    }

    // MARK: - Platform Detection

    /// Whether running on Apple Silicon
    public static var isAppleSilicon: Bool {
        #if arch(arm64) && (os(macOS) || os(iOS))
        return true
        #else
        return false
        #endif
    }

    /// Whether AMX (Apple Matrix Extensions) are available
    public static var hasAMX: Bool {
        // AMX is available on M1 and later
        isAppleSilicon
    }

    // MARK: - Recommended Sizes

    /// Recommended buffer size for batch operations
    public static var recommendedBatchSize: Int {
        // Based on L2 cache size and typical vector dimensions
        let floatSize = MemoryLayout<Float>.stride
        let vectorsInL2 = l2CacheSize / (256 * floatSize)  // Assume 256-dim vectors
        return max(32, min(1024, vectorsInL2))
    }

    /// Recommended block size for matrix operations
    public static var recommendedBlockSize: Int {
        // Optimize for L1 cache
        let floatSize = MemoryLayout<Float>.stride
        let elementsInL1 = l1CacheLineSize * 8 / floatSize
        return Int(sqrt(Double(elementsInL1)))
    }
}
