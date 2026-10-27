// VectorCore: Optimization Attributes
//
// Optimization-intent helpers. None of them emits a compiler hint, prefetch or fence beyond what its own body
// and attributes state; several are identity wrappers kept for API compatibility (disposition: D02).
//

import Foundation

// MARK: - Performance Annotations

/// Calls `operation` and returns its result. `@inline(__always)` requests inlining of this wrapper only; it does not
/// mark `operation` always-inline. No caller in this package (claim ledger callerInventory).
@inlinable
@inline(__always)
public func performanceCritical<T>(_ operation: () throws -> T) rethrows -> T {
    try operation()
}

/// Returns `condition` unchanged. No branch-weight hint is emitted.
@inlinable
@inline(__always)
public func likely(_ condition: Bool) -> Bool {
    // Identity function: no __builtin_expect equivalent is used.
    condition
}

/// Returns `condition` unchanged (it is not negated). No branch-weight hint is emitted.
@inlinable
@inline(__always)
public func unlikely(_ condition: Bool) -> Bool {
    // Identity function: no __builtin_expect equivalent is used.
    condition
}

// MARK: - Memory Touch Helpers

/// Reads and discards `pointer[offset]` on x86_64 and arm64. This is a discarded pointee load that the optimizer may
/// remove; it is not a prefetch instruction and no prefetch intrinsic is emitted. The bounds check is `performanceAssert`
/// (DEBUG builds only). No caller in this package (claim ledger callerInventory).
@inlinable
public func prefetchRead<T>(_ pointer: UnsafePointer<T>, offset: Int = 0, count: Int) {
    #if arch(x86_64) || arch(arm64)
    // Bounds check in debug mode
    performanceAssert(offset >= 0 && offset < count, "Prefetch offset out of bounds")
    // Discarded load; not a prefetch instruction.
    _ = pointer.advanced(by: offset).pointee
    #endif
}

/// Reads `pointer[offset]` and stores the same value back on x86_64 and arm64: a self-assignment the optimizer may remove.
/// It is not a prefetch instruction. When kept it is a real store, so it must not target memory another thread may access
/// concurrently. The bounds check is `performanceAssert` (DEBUG builds only). No caller in this package (claim ledger callerInventory).
@inlinable
public func prefetchWrite<T>(_ pointer: UnsafeMutablePointer<T>, offset: Int = 0, count: Int) {
    #if arch(x86_64) || arch(arm64)
    // Bounds check in debug mode
    performanceAssert(offset >= 0 && offset < count, "Prefetch offset out of bounds")
    // Self-assignment store; not a prefetch instruction.
    pointer.advanced(by: offset).pointee = pointer.advanced(by: offset).pointee
    #endif
}

// MARK: - Loop Helpers

/// Visits indices in groups of the requested width. The name expresses optimization intent.
/// Requires count >= 0 and stride > 0, including empty work.
/// Exact coverage and error stopping: ExecutionBoundaryTests.testArbitraryLoopWidthsExactCoverage
/// and ExecutionBoundaryTests.testLoopErrorsStopAtFirstInteriorAndTail.
@inlinable
public func unrolledLoop<T>(
    count: Int,
    by stride: Int = 4,
    _ body: (Int) throws -> T
) rethrows {
    precondition(count >= 0, "count must be nonnegative")
    precondition(stride > 0, "width must be positive")
    var i = 0
    while i < count {
        let end = i + min(stride, count - i)
        while i < end {
            _ = try body(i)
            i += 1
        }
    }
}

/// Visits consecutive nonempty ranges of at most vectorWidth indices.
/// Requires count >= 0 and vectorWidth > 0, including empty work.
/// Exact coverage and error stopping: ExecutionBoundaryTests.testArbitraryLoopWidthsExactCoverage
/// and ExecutionBoundaryTests.testLoopErrorsStopAtFirstInteriorAndTail.
@inlinable
public func vectorizedLoop(
    count: Int,
    vectorWidth: Int = 4,
    _ body: (Range<Int>) throws -> Void
) rethrows {
    precondition(count >= 0, "count must be nonnegative")
    precondition(vectorWidth > 0, "width must be positive")
    var i = 0
    while i < count {
        let end = i + min(vectorWidth, count - i)
        try body(i..<end)
        i = end
    }
}

// MARK: - Branch Helpers

/// Runs `then` when `condition()` is true, otherwise `else`. `likely` is an identity function, so no branch-prediction
/// hint is emitted.
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

/// Runs `then` when `condition()` is true, otherwise `else`. `unlikely` is an identity function, so no branch-prediction
/// hint is emitted.
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

// MARK: - Alignment Assertions

/// Returns `pointer` unchanged after an `assert` that it is aligned to `alignment`. The check runs only where `assert` is
/// active (Debug builds); in Release nothing is checked, and no alignment assumption is passed to the optimizer.
@inlinable
public func assumeAligned<T>(
    _ pointer: UnsafePointer<T>,
    to alignment: Int = MemoryLayout<T>.alignment
) -> UnsafePointer<T> {
    assert(Int(bitPattern: pointer) % alignment == 0,
           "Pointer is not aligned to \(alignment) bytes")
    return pointer
}

/// Returns `pointer` unchanged after an `assert` that it is aligned to `alignment` (Debug builds only); in Release nothing
/// is checked, and no alignment assumption is passed to the optimizer.
@inlinable
public func assumeAligned<T>(
    _ pointer: UnsafeMutablePointer<T>,
    to alignment: Int = MemoryLayout<T>.alignment
) -> UnsafeMutablePointer<T> {
    assert(Int(bitPattern: pointer) % alignment == 0,
           "Pointer is not aligned to \(alignment) bytes")
    return pointer
}

// MARK: - Opaque Calls

/// An empty `@inline(never)` function. It is not a memory or compiler fence: no ordering or anti-elision effect is
/// promised beyond the optimizer honouring `@inline(never)`. No caller in this package (claim ledger callerInventory).
@inline(never)
public func optimizationBarrier() {
    // Intentionally empty.
}

/// Returns `value` through an `@inline(never)` call. Whether that keeps the computation of `value` from being removed is
/// not verified, and there is no guarantee that it does (the benchmark sink is `blackHole`). No caller in this package (claim ledger callerInventory).
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

/// Returns different values for debug vs release
@inlinable
public func debugOrRelease<T>(debug: T, release: T) -> T {
    #if DEBUG
    return debug
    #else
    return release
    #endif
}

// MARK: - Assertions

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
