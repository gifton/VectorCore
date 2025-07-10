// VectorCore: Memory Utilities
//
// Memory management and optimization utilities

import Foundation

// MARK: - Memory Management Namespace

/// Namespace for memory management types and utilities
public enum Memory {
    
    // MARK: - Aligned Memory Buffer
    
    /// Aligned memory allocation for SIMD operations
    public final class AlignedBuffer<T> {
        private let pointer: UnsafeMutablePointer<T>
        private let capacity: Int
        private let alignment: Int
        
        public init(capacity: Int, alignment: Int = 64) {
            self.capacity = capacity
            self.alignment = alignment
            
            let byteCount = capacity * MemoryLayout<T>.stride
            let rawPointer = UnsafeMutableRawPointer.allocate(
                byteCount: byteCount,
                alignment: alignment
            )
            self.pointer = rawPointer.bindMemory(to: T.self, capacity: capacity)
        }
        
        deinit {
            pointer.deallocate()
        }
        
        public func withUnsafePointer<Result>(_ body: (UnsafePointer<T>) throws -> Result) rethrows -> Result {
            try body(pointer)
        }
        
        public func withUnsafeMutablePointer<Result>(_ body: (UnsafeMutablePointer<T>) throws -> Result) rethrows -> Result {
            try body(pointer)
        }
        
        public subscript(index: Int) -> T {
            get {
                precondition(index >= 0 && index < capacity, "Index out of bounds")
                return pointer[index]
            }
            set {
                precondition(index >= 0 && index < capacity, "Index out of bounds")
                pointer[index] = newValue
            }
        }
    }
    
    // MARK: - Reusable Memory Pool
    
    /// Simple memory pool for reusing allocations
    public final class Pool<T> {
        private var available: [UnsafeMutableBufferPointer<T>] = []
        private let capacity: Int
        private let alignment: Int
        
        public init(capacity: Int, alignment: Int = 64) {
            self.capacity = capacity
            self.alignment = alignment
        }
        
        deinit {
            // Clean up all allocated buffers
            for buffer in available {
                buffer.baseAddress?.deallocate()
            }
        }
        
        public func acquire() -> UnsafeMutableBufferPointer<T> {
            if let buffer = available.popLast() {
                return buffer
            } else {
                // Allocate new buffer
                let rawPointer = UnsafeMutableRawPointer.allocate(
                    byteCount: capacity * MemoryLayout<T>.stride,
                    alignment: alignment
                )
                let pointer = rawPointer.bindMemory(to: T.self, capacity: capacity)
                return UnsafeMutableBufferPointer(start: pointer, count: capacity)
            }
        }
        
        public func release(_ buffer: UnsafeMutableBufferPointer<T>) {
            // Buffer is already initialized, just append it back
            available.append(buffer)
        }
    }
    
    
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

/// Get memory size of a vector type
public func vectorMemorySize<Vector: SIMD>(_ type: Vector.Type) -> Int where Vector.Scalar: BinaryFloatingPoint {
    return MemoryLayout<Vector>.size
}

/// Get memory alignment of a vector type
public func vectorMemoryAlignment<Vector: SIMD>(_ type: Vector.Type) -> Int where Vector.Scalar: BinaryFloatingPoint {
    return MemoryLayout<Vector>.alignment
}

/// Check if pointer is aligned
public func isAligned<T>(_ pointer: UnsafePointer<T>, to alignment: Int) -> Bool {
    return Int(bitPattern: pointer) % alignment == 0
}

/// Copy memory with alignment check
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

/// Create a temporary buffer with automatic cleanup
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

/// Zero out memory securely
public func secureZero<T>(_ buffer: UnsafeMutableBufferPointer<T>) {
    guard let baseAddress = buffer.baseAddress else { return }
    
    // Use memset_s for secure zeroing when available
    let byteCount = buffer.count * MemoryLayout<T>.stride
    _ = memset_s(baseAddress, byteCount, 0, byteCount)
}


