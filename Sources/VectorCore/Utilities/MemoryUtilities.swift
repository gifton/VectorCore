// VectorCore: Memory Utilities
//
// Memory management and optimization utilities

import Foundation

// MARK: - Memory Management Namespace

/// Namespace for memory management types and utilities
public enum Memory {
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
public func vectorMemorySize<Vector: SIMD>(_ type: Vector.Type) -> Int where Vector.Scalar: BinaryFloatingPoint {
    return MemoryLayout<Vector>.size
}

/// Get memory alignment requirement of a SIMD vector type.
///
/// - Parameter type: The SIMD vector type to query
/// - Returns: Required alignment in bytes for optimal performance
///
/// ## Example
/// ```swift
/// let alignment = vectorMemoryAlignment(SIMD8<Float>.self)  // Returns 32 bytes
/// ```
public func vectorMemoryAlignment<Vector: SIMD>(_ type: Vector.Type) -> Int where Vector.Scalar: BinaryFloatingPoint {
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
    return Int(bitPattern: pointer) % alignment == 0
}

/// Copy memory with alignment optimization.
///
/// This function checks alignment and uses optimized copy operations
/// when both source and destination meet alignment requirements.
///
/// - Parameters:
///   - source: Source memory pointer
///   - destination: Destination memory pointer
///   - count: Number of elements to copy
///   - preferredAlignment: Preferred alignment for optimization (default: 64)
///
/// ## Performance Notes
/// - Aligned copy uses bulk memory operations
/// - Unaligned copy falls back to element-wise operations
/// - 64-byte alignment optimal for cache lines
///
/// ## Example
/// ```swift
/// alignedCopy(
///     from: sourceBuffer,
///     to: destBuffer,
///     count: 1024,
///     preferredAlignment: 64
/// )
/// ```
public func alignedCopy<T>(
    from source: UnsafePointer<T>,
    to destination: UnsafeMutablePointer<T>,
    count: Int,
    preferredAlignment: Int = 64
) {
    if isAligned(source, to: preferredAlignment) && isAligned(destination, to: preferredAlignment) {
        // Use optimized copy for aligned memory
        destination.initialize(from: source, count: count)
    } else {
        // Fallback to element-wise copy
        for i in 0..<count {
            destination[i] = source[i]
        }
    }
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
public func withTemporaryBuffer<T, Result>(
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
public func secureZero<T>(_ buffer: UnsafeMutableBufferPointer<T>) {
    guard let baseAddress = buffer.baseAddress else { return }

    // Use memset_s for secure zeroing when available
    let byteCount = buffer.count * MemoryLayout<T>.stride
    _ = memset_s(baseAddress, byteCount, 0, byteCount)
}
