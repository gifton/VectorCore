// VectorCore: Optimization Attributes
//
// Central location for optimization-related attributes and compiler hints
//

import Foundation

// MARK: - Optimization Flags

/// Check if SIMD optimizations are enabled
@inlinable
public var isSIMDEnabled: Bool {
    #if VECTORCORE_ENABLE_SIMD
    return true
    #else
    return false
    #endif
}

/// Check if running in release mode
@inlinable
public var isReleaseMode: Bool {
    #if SWIFT_RELEASE_MODE
    return true
    #else
    return false
    #endif
}

// MARK: - Performance Annotations

/// Marks a function as performance-critical and always inline
@inlinable
@inline(__always)
public func performanceCritical<T>(_ operation: () throws -> T) rethrows -> T {
    try operation()
}

/// Hints to the compiler that a condition is likely true
@inlinable
@inline(__always)
public func likely(_ condition: Bool) -> Bool {
    // In release mode, this could use __builtin_expect
    condition
}

/// Hints to the compiler that a condition is likely false
@inlinable
@inline(__always)
public func unlikely(_ condition: Bool) -> Bool {
    // In release mode, this could use __builtin_expect
    condition
}

// MARK: - Memory Optimization Hints

/// Prefetch data for read access
@inlinable
public func prefetchRead<T>(_ pointer: UnsafePointer<T>, offset: Int = 0, count: Int) {
    #if arch(x86_64) || arch(arm64)
    // Bounds check in debug mode
    performanceAssert(offset >= 0 && offset < count, "Prefetch offset out of bounds")
    // Compiler should recognize this pattern for prefetching
    _ = pointer.advanced(by: offset).pointee
    #endif
}

/// Prefetch data for write access
@inlinable
public func prefetchWrite<T>(_ pointer: UnsafeMutablePointer<T>, offset: Int = 0, count: Int) {
    #if arch(x86_64) || arch(arm64)
    // Bounds check in debug mode
    performanceAssert(offset >= 0 && offset < count, "Prefetch offset out of bounds")
    // Touch the memory location to hint at future write
    pointer.advanced(by: offset).pointee = pointer.advanced(by: offset).pointee
    #endif
}

// MARK: - Loop Optimization Hints

/// Hints that a loop should be unrolled
@inlinable
public func unrolledLoop<T>(
    count: Int,
    by stride: Int = 4,
    _ body: (Int) throws -> T
) rethrows {
    var i = 0
    
    // Process in groups of stride
    while i + stride <= count {
        _ = try body(i)
        _ = try body(i + 1)
        if stride > 2 { _ = try body(i + 2) }
        if stride > 3 { _ = try body(i + 3) }
        i += stride
    }
    
    // Process remaining elements
    while i < count {
        _ = try body(i)
        i += 1
    }
}

/// Hints that a loop should be vectorized
@inlinable
public func vectorizedLoop(
    count: Int,
    vectorWidth: Int = 4,
    _ body: (Range<Int>) throws -> Void
) rethrows {
    var i = 0
    
    // Process vector-width elements at a time
    while i + vectorWidth <= count {
        try body(i..<(i + vectorWidth))
        i += vectorWidth
    }
    
    // Process remaining elements
    if i < count {
        try body(i..<count)
    }
}

// MARK: - Branch Prediction Hints

/// Execute code with branch prediction hint
@inlinable
public func withLikelyBranch<T>(
    _ condition: @autoclosure () -> Bool,
    then: () throws -> T,
    else: () throws -> T
) rethrows -> T {
    if likely(condition()) {
        return try then()
    } else {
        return try `else`()
    }
}

/// Execute code with unlikely branch prediction hint
@inlinable
public func withUnlikelyBranch<T>(
    _ condition: @autoclosure () -> Bool,
    then: () throws -> T,
    else: () throws -> T
) rethrows -> T {
    if unlikely(condition()) {
        return try then()
    } else {
        return try `else`()
    }
}

// MARK: - Alignment Hints

/// Ensures a pointer is aligned to the specified boundary
@inlinable
public func assumeAligned<T>(
    _ pointer: UnsafePointer<T>,
    to alignment: Int = MemoryLayout<T>.alignment
) -> UnsafePointer<T> {
    assert(Int(bitPattern: pointer) % alignment == 0, 
           "Pointer is not aligned to \(alignment) bytes")
    return pointer
}

/// Ensures a mutable pointer is aligned to the specified boundary
@inlinable
public func assumeAligned<T>(
    _ pointer: UnsafeMutablePointer<T>,
    to alignment: Int = MemoryLayout<T>.alignment
) -> UnsafeMutablePointer<T> {
    assert(Int(bitPattern: pointer) % alignment == 0,
           "Pointer is not aligned to \(alignment) bytes")
    return pointer
}

// MARK: - Optimization Barriers

/// Prevents the compiler from optimizing across this barrier
@inline(never)
public func optimizationBarrier() {
    // This function intentionally does nothing but prevents optimization
}

/// Prevents the compiler from optimizing away a value
@inline(never)
public func doNotOptimize<T>(_ value: T) -> T {
    return value
}

// MARK: - Debug vs Release Helpers

/// Executes code only in debug builds
@inlinable
public func debugOnly(_ block: () throws -> Void) rethrows {
    #if DEBUG
    try block()
    #endif
}

/// Executes code only in release builds
@inlinable
public func releaseOnly(_ block: () throws -> Void) rethrows {
    #if SWIFT_RELEASE_MODE
    try block()
    #endif
}

/// Returns different values for debug vs release
@inlinable
public func debugOrRelease<T>(debug: T, release: T) -> T {
    #if DEBUG
    return debug
    #else
    return release
    #endif
}

// MARK: - Performance Assertions

/// Assert that only runs in debug mode
@inlinable
public func performanceAssert(
    _ condition: @autoclosure () -> Bool,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #file,
    line: UInt = #line
) {
    #if DEBUG
    assert(condition(), message(), file: file, line: line)
    #endif
}

/// Precondition that can be disabled in release for performance
@inlinable
public func performancePrecondition(
    _ condition: @autoclosure () -> Bool,
    _ message: @autoclosure () -> String = "",
    file: StaticString = #file,
    line: UInt = #line
) {
    #if !SWIFT_RELEASE_MODE
    precondition(condition(), message(), file: file, line: line)
    #endif
}