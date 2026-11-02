// VectorCore: Memory Utilities
//
// Memory management and optimization utilities

import Foundation

// MARK: - Memory Management Namespace

/// Namespace for memory management types and utilities
internal enum Memory {
    // MARK: - Pressure Monitor

    /// Monitor system memory pressure
    public final class PressureMonitor {
        private var source: (any DispatchSourceMemoryPressure)?

        public enum Level {
            case normal
            case warning
            case critical
        }

        public init() {}

        public func startMonitoring(handler: @escaping (Level) -> Void) {
            let source = DispatchSource.makeMemoryPressureSource(eventMask: .all)

            source.setEventHandler {
                let event = source.data

                if event.contains(.critical) {
                    handler(.critical)
                } else if event.contains(.warning) {
                    handler(.warning)
                } else {
                    handler(.normal)
                }
            }

            self.source = source
            source.resume()
        }

        public func stopMonitoring() {
            source?.cancel()
            source = nil
        }

        deinit {
            stopMonitoring()
        }
    }
}

// MARK: - Memory Utilities

/// Get memory size of a SIMD vector type.
///
/// - Parameter type: The SIMD vector type to query
/// - Returns: Size in bytes required to store the vector
///
/// ## Example
/// ```swift
/// let size = vectorMemorySize(SIMD16<Float>.self)  // Returns 64 bytes
/// ```
internal func vectorMemorySize<Vector: SIMD>(_ type: Vector.Type) -> Int where Vector.Scalar: BinaryFloatingPoint {
    return MemoryLayout<Vector>.size
}

/// Get memory alignment requirement of a SIMD vector type.
///
/// - Parameter type: The SIMD vector type to query
/// - Returns: `MemoryLayout<Vector>.alignment` in bytes
///
/// ## Example
/// ```swift
/// let alignment = vectorMemoryAlignment(SIMD8<Float>.self)  // Returns 32 bytes
/// ```
internal func vectorMemoryAlignment<Vector: SIMD>(_ type: Vector.Type) -> Int where Vector.Scalar: BinaryFloatingPoint {
    return MemoryLayout<Vector>.alignment
}

/// Check if a pointer is properly aligned to a specified boundary.
///
/// Proper alignment is crucial for SIMD operations and can significantly
/// impact performance. Misaligned access may cause crashes on some architectures.
///
/// - Parameters:
///   - pointer: The pointer to check
///   - alignment: Required alignment boundary in bytes
/// - Returns: true if pointer address is a multiple of alignment
///
/// ## Example
/// ```swift
/// let buffer = UnsafeMutablePointer<Float>.allocate(capacity: 256)
/// if isAligned(buffer, to: 32) {
///     // Safe to use with 32-byte aligned SIMD operations
/// }
/// ```
public func isAligned<T>(_ pointer: UnsafePointer<T>, to alignment: Int) -> Bool {
    return alignment > 0 && Int(bitPattern: pointer) % alignment == 0
}

/// Replace `count` initialized destination elements with initialized source elements.
/// Both ranges must be valid, naturally aligned and nonoverlapping. Old destination
/// values are released. For uninitialized storage, use `initialize(from:count:)`.
/// `preferredAlignment` is a positive compatibility hint; it does not change ownership
/// operations or promise a measured optimization. `count` must be nonnegative.
public func alignedCopy<T>(
    from source: UnsafePointer<T>,
    to destination: UnsafeMutablePointer<T>,
    count: Int,
    preferredAlignment: Int = 64
) {
    guard count >= 0 else { fatalError("Count must be nonnegative") }
    guard preferredAlignment > 0 else { fatalError("Preferred alignment must be positive") }
    destination.update(from: source, count: count)
}

// MARK: - Buffer Utilities

/// Create a temporary buffer with automatic cleanup.
///
/// This function allocates an aligned temporary buffer, executes the provided
/// closure with the buffer, and automatically deallocates the memory
/// regardless of how the closure exits (normally or via throw).
///
/// - Parameters:
///   - type: The element type for the buffer
///   - capacity: Number of elements to allocate
///   - alignment: Memory alignment in bytes (default: 64 for cache lines)
///   - body: Closure to execute with the allocated buffer
/// - Returns: Result returned by the closure
/// - Throws: Rethrows any error thrown by the closure
///
/// ## Example
/// ```swift
/// let result = withTemporaryBuffer(of: Float.self, capacity: 1024) { buffer in
///     // Use buffer safely, automatic cleanup guaranteed
///     vDSP_vfill([1.0], buffer.baseAddress!, 1, vDSP_Length(1024))
///     return buffer.reduce(0, +)
/// }
/// ```
internal func withTemporaryBuffer<T, Result>(
    of type: T.Type,
    capacity: Int,
    alignment: Int = 64,
    _ body: (UnsafeMutableBufferPointer<T>) throws -> Result
) rethrows -> Result {
    let rawPointer = UnsafeMutableRawPointer.allocate(
        byteCount: capacity * MemoryLayout<T>.stride,
        alignment: alignment
    )
    defer { rawPointer.deallocate() }

    let pointer = rawPointer.bindMemory(to: T.self, capacity: capacity)
    let buffer = UnsafeMutableBufferPointer(start: pointer, count: capacity)

    return try body(buffer)
}

/// Zero out memory securely to prevent data leakage.
///
/// This function uses memset_s to ensure memory is zeroed in a way that
/// prevents compiler optimizations from removing the operation. Essential
/// for handling sensitive data like cryptographic keys.
///
/// - Parameter buffer: Buffer to securely zero
///
/// ## Security Notes
/// - Uses memset_s for guaranteed zeroing
/// - Cannot be optimized away by compiler
/// - Should be called before deallocating sensitive data
///
/// ## Example
/// ```swift
/// var sensitiveData = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: 32)
/// defer {
///     secureZero(sensitiveData)
///     sensitiveData.deallocate()
/// }
/// // Use sensitive data...
/// ```
internal func secureZero<T>(_ buffer: UnsafeMutableBufferPointer<T>) {
    guard let baseAddress = buffer.baseAddress else { return }

    // Use memset_s for secure zeroing when available
    let byteCount = buffer.count * MemoryLayout<T>.stride
    _ = memset_s(baseAddress, byteCount, 0, byteCount)
}

// MARK: - Sendable Buffer Wrapper

/// `@unchecked Sendable` wrapper for UnsafeMutablePointer: it owns the allocation and
/// frees it in `deinit`. It adds no synchronization.
///
/// The pointer is reachable from any isolation domain (`nonisolated var pointer`); the
/// compiler does not check accesses through it.
///
/// - Warning: The caller must synchronize concurrent reads and writes through `pointer`
internal final class SendableBufferWrapper<T> {
    private let buffer: UnsafeMutablePointer<T>
    public let capacity: Int

    /// Initialize with allocated capacity
    public init(capacity: Int) {
        self.capacity = capacity
        self.buffer = UnsafeMutablePointer<T>.allocate(capacity: capacity)
    }

    /// Access the underlying pointer from any isolation domain
    public nonisolated var pointer: UnsafeMutablePointer<T> {
        buffer
    }

    /// Clean up allocated memory
    deinit {
        buffer.deallocate()
    }
}

extension SendableBufferWrapper: @unchecked Sendable {}

/// `@unchecked Sendable` wrapper for UnsafeMutableBufferPointer; it adds no synchronization
internal final class SendableMutableBufferWrapper<T> {
    private let buffer: UnsafeMutableBufferPointer<T>

    /// Initialize with allocated capacity
    public init(capacity: Int) {
        let ptr = UnsafeMutablePointer<T>.allocate(capacity: capacity)
        self.buffer = UnsafeMutableBufferPointer(start: ptr, count: capacity)
    }

    /// Access the underlying buffer pointer from any isolation domain
    public nonisolated var bufferPointer: UnsafeMutableBufferPointer<T> {
        buffer
    }

    /// Access the base pointer from any isolation domain
    public nonisolated var pointer: UnsafeMutablePointer<T>? {
        buffer.baseAddress
    }

    /// Clean up allocated memory
    deinit {
        buffer.baseAddress?.deallocate()
    }
}

extension SendableMutableBufferWrapper: @unchecked Sendable {}
