//
//  AlignmentStrategy.swift
//  VectorCore
//
//

import Foundation

// MARK: - Alignment Strategy

/// Memory alignment strategy for SIMD operations
public enum AlignmentStrategy: Sendable {
    /// Fixed 64-byte alignment (cache line)
    case cacheLine
    
    /// Platform-specific optimal alignment
    case platform
    
    /// Custom alignment value
    case custom(Int)
    
    /// Default alignment for the element type
    case elementDefault
    
    /// Get the alignment value in bytes
    public var value: Int {
        switch self {
        case .cacheLine:
            return 64
            
        case .platform:
            #if arch(arm64)
            return 16  // ARM NEON requires 16-byte alignment
            #elseif arch(x86_64)
            return 32  // AVX/AVX2 prefers 32-byte alignment
            #else
            return 16  // Conservative default
            #endif
            
        case .custom(let alignment):
            precondition(alignment > 0 && alignment.nonzeroBitCount == 1,
                        "Alignment must be a positive power of 2")
            return alignment
            
        case .elementDefault:
            return 16  // Will be specialized per type
        }
    }
}

// MARK: - Alignment Utilities

public enum AlignmentUtility {
    /// Check if a pointer is aligned to the specified boundary
    @inlinable
    public static func isAligned<T>(_ pointer: UnsafePointer<T>, to alignment: Int) -> Bool {
        Int(bitPattern: pointer) & (alignment - 1) == 0
    }
    
    /// Check if a mutable pointer is aligned to the specified boundary
    @inlinable
    public static func isAligned<T>(_ pointer: UnsafeMutablePointer<T>, to alignment: Int) -> Bool {
        Int(bitPattern: pointer) & (alignment - 1) == 0
    }
    
    /// Round up size to the nearest alignment boundary
    @inlinable
    public static func alignedSize(_ size: Int, alignment: Int) -> Int {
        (size + alignment - 1) & ~(alignment - 1)
    }
    
    /// Get recommended alignment for a type
    @inlinable
    public static func recommendedAlignment<T>(for type: T.Type) -> Int {
        // For SIMD types, use larger alignment
        if type == Float.self || type == Double.self {
            return AlignmentStrategy.platform.value
        }
        return max(MemoryLayout<T>.alignment, 16)
    }
    
    /// Validate alignment is a power of 2
    @inlinable
    public static func isValidAlignment(_ alignment: Int) -> Bool {
        alignment > 0 && alignment.nonzeroBitCount == 1
    }
}

// MARK: - Aligned Allocator

/// Utility for aligned memory allocation
public struct AlignedAllocator {
    /// Allocate aligned memory
    @inlinable
    public static func allocate(
        byteCount: Int,
        alignment: AlignmentStrategy = .platform
    ) -> UnsafeMutableRawPointer {
        let alignmentValue = alignment.value
        precondition(AlignmentUtility.isValidAlignment(alignmentValue),
                    "Invalid alignment: \(alignmentValue)")
        
        return UnsafeMutableRawPointer.allocate(
            byteCount: byteCount,
            alignment: alignmentValue
        )
    }
    
    /// Allocate aligned memory for a specific type
    @inlinable
    public static func allocate<T>(
        capacity: Int,
        of type: T.Type,
        alignment: AlignmentStrategy = .platform
    ) -> UnsafeMutablePointer<T> {
        let alignmentValue = max(alignment.value, MemoryLayout<T>.alignment)
        let byteCount = capacity * MemoryLayout<T>.stride
        
        let rawPointer = allocate(byteCount: byteCount,
                                 alignment: .custom(alignmentValue))
        return rawPointer.bindMemory(to: T.self, capacity: capacity)
    }
}

// MARK: - Storage Alignment Extension

extension ManagedStorage {
    /// Create storage with specific alignment strategy
    public init(capacity: Int, alignment: AlignmentStrategy) {
        let alignmentValue = max(alignment.value,
                                AlignmentUtility.recommendedAlignment(for: Element.self))
        self.init(capacity: capacity, alignment: alignmentValue)
    }
    
    /// Check if storage is properly aligned for SIMD operations
    public var isSimdAligned: Bool {
        withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return false }
            return AlignmentUtility.isAligned(baseAddress, to: AlignmentStrategy.platform.value)
        }
    }
}

extension HybridStorage {
    /// Check if storage is properly aligned for SIMD operations
    public var isSimdAligned: Bool {
        withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return false }
            return AlignmentUtility.isAligned(baseAddress, to: AlignmentStrategy.platform.value)
        }
    }
}

// MARK: - Platform-Specific Optimizations

#if arch(arm64)
extension AlignmentStrategy {
    /// Optimal alignment for NEON SIMD operations
    public static let simd = AlignmentStrategy.custom(16)
}
#elseif arch(x86_64)
extension AlignmentStrategy {
    /// Optimal alignment for AVX operations
    public static let simd = AlignmentStrategy.custom(32)
    
    /// Optimal alignment for AVX-512 operations
    public static let simd512 = AlignmentStrategy.custom(64)
}
#endif

// MARK: - Alignment Assertions

/// Debug assertion for alignment
@inlinable
public func assertAligned<T>(
    _ pointer: UnsafePointer<T>,
    to alignment: Int,
    file: StaticString = #file,
    line: UInt = #line
) {
    #if DEBUG
    assert(AlignmentUtility.isAligned(pointer, to: alignment),
           "Pointer not aligned to \(alignment) bytes",
           file: file, line: line)
    #endif
}

/// Debug assertion for mutable pointer alignment
@inlinable
public func assertAligned<T>(
    _ pointer: UnsafeMutablePointer<T>,
    to alignment: Int,
    file: StaticString = #file,
    line: UInt = #line
) {
    #if DEBUG
    assert(AlignmentUtility.isAligned(pointer, to: alignment),
           "Pointer not aligned to \(alignment) bytes",
           file: file, line: line)
    #endif
}