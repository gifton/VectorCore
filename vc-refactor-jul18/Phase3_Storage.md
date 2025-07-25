# Phase 3: Unified Storage System

## Overview

Leveraging Swift 6.0's move-only types and ownership features, we'll create a unified storage system that eliminates all duplication while maximizing performance. This phase replaces 8 dimension-specific storage types with a single, generic implementation.

## Storage Architecture

### Core Storage Design

```swift
// Move-only storage for zero-copy semantics
public struct Storage<Element: BinaryFloatingPoint & SIMDScalar>: ~Copyable {
    private let pointer: UnsafeMutableRawPointer
    private let capacity: Int
    private let alignment: Int
    
    // Swift 6.0 move semantics
    public consuming func into() -> UnsafeMutableBufferPointer<Element> {
        let typed = pointer.bindMemory(to: Element.self, capacity: capacity)
        consume self
        return UnsafeMutableBufferPointer(start: typed, count: capacity)
    }
    
    // Explicit copy when needed
    public borrowing func copy() -> Storage {
        let newPointer = UnsafeMutableRawPointer.allocate(
            byteCount: capacity * MemoryLayout<Element>.stride,
            alignment: alignment
        )
        newPointer.copyMemory(from: pointer, byteCount: capacity * MemoryLayout<Element>.stride)
        return Storage(pointer: newPointer, capacity: capacity, alignment: alignment)
    }
    
    deinit {
        pointer.deallocate()
    }
}
```

### Dimension-Aware Storage Wrapper

```swift
// Generic storage that knows its dimension at compile time
public struct DimensionStorage<D: Dimension, Element: BinaryFloatingPoint & SIMDScalar> {
    private var storage: ManagedBuffer<Header, Element>
    
    private struct Header {
        let dimension: Int
        let alignment: Int
        var isUniquelyReferenced: Bool
    }
    
    public init() {
        self.storage = ManagedBuffer<Header, Element>.create(
            minimumCapacity: D.size
        ) { buffer in
            Header(dimension: D.size, alignment: 64, isUniquelyReferenced: true)
        }
        
        // Initialize to zero using SIMD
        storage.withUnsafeMutablePointerToElements { elements in
            memset(elements, 0, D.size * MemoryLayout<Element>.stride)
        }
    }
    
    // Copy-on-write implementation
    private mutating func ensureUniquelyReferenced() {
        if !isKnownUniquelyReferenced(&storage) {
            storage = storage.copy()
        }
    }
    
    // Borrowing for read operations
    public borrowing func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        try storage.withUnsafeMutablePointerToElements { elements in
            try body(UnsafeBufferPointer(start: elements, count: D.size))
        }
    }
    
    // Consuming for write operations
    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        ensureUniquelyReferenced()
        return try storage.withUnsafeMutablePointerToElements { elements in
            try body(UnsafeMutableBufferPointer(start: elements, count: D.size))
        }
    }
}
```

### 🛑 DECISION POINT: Memory Alignment Strategy

What alignment should we use for SIMD operations?

**Option A: Fixed 64-byte (cache line)**
```swift
private static let alignment = 64
```

**Option B: Platform-specific**
```swift
private static var alignment: Int {
    #if arch(arm64)
    return 16  // ARM NEON
    #else
    return 32  // AVX/AVX2
    #endif
}
```

**Option C: Configurable with default**
```swift
public init(alignment: Int = MemoryLayout<Element>.recommendedAlignment) {
    // Use provided or recommended alignment
}
```

### Optimized Allocation Strategy

```swift
// Memory pool for common sizes to reduce allocation overhead
public actor StoragePool {
    private var available: [Int: [UnsafeMutableRawPointer]] = [:]
    
    // Common ML embedding dimensions
    private static let pooledSizes = [128, 256, 384, 512, 768, 1024, 1536, 3072]
    
    public func acquire(size: Int, alignment: Int = 64) -> UnsafeMutableRawPointer {
        if Self.pooledSizes.contains(size),
           let cached = available[size]?.popLast() {
            return cached
        }
        
        return UnsafeMutableRawPointer.allocate(
            byteCount: size * MemoryLayout<Float>.stride,
            alignment: alignment
        )
    }
    
    public func release(_ pointer: UnsafeMutableRawPointer, size: Int) {
        if Self.pooledSizes.contains(size) {
            available[size, default: []].append(pointer)
        } else {
            pointer.deallocate()
        }
    }
}
```

### Small Vector Optimization

```swift
// Stack allocation for small vectors
public struct HybridStorage<Element: BinaryFloatingPoint & SIMDScalar> {
    private enum Backing {
        case inline(InlineBuffer)
        case heap(Storage<Element>)
    }
    
    // 64 bytes of inline storage (8 Float64 or 16 Float32)
    private struct InlineBuffer {
        var data: (Element, Element, Element, Element,
                   Element, Element, Element, Element,
                   Element, Element, Element, Element,
                   Element, Element, Element, Element) = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        
        mutating func withUnsafeMutableBufferPointer<R>(
            count: Int,
            _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
        ) rethrows -> R {
            try withUnsafeMutableBytes(of: &data) { bytes in
                let pointer = bytes.bindMemory(to: Element.self)
                let buffer = UnsafeMutableBufferPointer(start: pointer.baseAddress!, count: count)
                return try body(buffer)
            }
        }
    }
    
    private var backing: Backing
    private let count: Int
    
    public init(count: Int) {
        self.count = count
        if count <= 16 / MemoryLayout<Element>.stride {
            self.backing = .inline(InlineBuffer())
        } else {
            self.backing = .heap(Storage(capacity: count))
        }
    }
}
```

### GPU-Ready Storage

```swift
// Unified memory for CPU/GPU operations
public struct UnifiedStorage<Element: BinaryFloatingPoint & SIMDScalar> {
    private let buffer: any GPUBuffer
    private let device: ComputeDevice
    
    public init(count: Int, device: ComputeDevice) async throws {
        switch device {
        case .cpu:
            self.buffer = CPUBuffer(count: count)
        case .gpu(let index):
            self.buffer = try await MetalBuffer(count: count, device: index)
        case .neural:
            self.buffer = try await NeuralEngineBuffer(count: count)
        }
        self.device = device
    }
    
    // Automatic migration between devices
    public func migrateTo(device: ComputeDevice) async throws -> UnifiedStorage {
        if self.device == device {
            return self
        }
        
        let newStorage = try await UnifiedStorage(count: buffer.count, device: device)
        try await buffer.copyTo(newStorage.buffer)
        return newStorage
    }
}

// Protocol for different buffer types
protocol GPUBuffer: Sendable {
    associatedtype Element
    var count: Int { get }
    func copyTo(_ other: any GPUBuffer) async throws
}
```

### 🛑 DECISION POINT: Storage Optimization Strategy

Which storage optimizations should we implement?

**Option A: Simple heap allocation only**
- Pros: Simplest implementation
- Cons: May have allocation overhead

**Option B: Small vector optimization + heap**
- Pros: Faster for small vectors
- Cons: More complex implementation

**Option C: Full tiered system (inline, pool, heap)**
- Pros: Maximum performance
- Cons: Most complex

### Performance-Critical Operations

```swift
extension DimensionStorage {
    // Zero-copy slice operations
    @inlinable
    public borrowing func slice(from start: Int, to end: Int) -> Slice<Element> {
        precondition(start >= 0 && end <= D.size && start < end)
        
        return withUnsafeBufferPointer { buffer in
            Slice(base: buffer.baseAddress! + start, count: end - start)
        }
    }
    
    // SIMD-optimized fill
    @inlinable
    public mutating func fill(with value: Element) {
        withUnsafeMutableBufferPointer { buffer in
            if Element.self == Float.self {
                let floatValue = Float(value)
                vDSP_vfill([floatValue], buffer.baseAddress!.assumingMemoryBound(to: Float.self), 1, vDSP_Length(buffer.count))
            } else {
                for i in 0..<buffer.count {
                    buffer[i] = value
                }
            }
        }
    }
    
    // Efficient copy operations
    @inlinable
    public borrowing func copyInto(_ destination: inout DimensionStorage<D, Element>) {
        withUnsafeBufferPointer { source in
            destination.withUnsafeMutableBufferPointer { dest in
                dest.initialize(from: source)
            }
        }
    }
}
```

## Implementation Checklist

- [ ] Basic Storage<Element> implementation
- [ ] DimensionStorage<D, Element> wrapper
- [ ] Copy-on-write semantics
- [ ] Memory alignment guarantees
- [ ] Small vector optimization (if chosen)
- [ ] Storage pool (if chosen)
- [ ] GPU-ready abstractions
- [ ] Performance benchmarks

## Validation

```swift
// Verify alignment
func testStorageAlignment() {
    let storage = DimensionStorage<Dim512, Float>()
    storage.withUnsafeBufferPointer { buffer in
        let address = Int(bitPattern: buffer.baseAddress!)
        assert(address % 64 == 0, "Storage not properly aligned")
    }
}

// Verify COW behavior
func testCopyOnWrite() {
    var storage1 = DimensionStorage<Dim128, Float>()
    var storage2 = storage1  // Should not copy
    
    storage2.fill(with: 1.0)  // Should trigger copy
    
    // Verify storage1 unchanged
}
```

## Performance Considerations

1. **Allocation Strategy**: Pool common sizes to reduce malloc overhead
2. **Alignment**: Ensure 64-byte alignment for optimal cache performance
3. **COW Overhead**: Use isKnownUniquelyReferenced for efficiency
4. **SIMD Usage**: Always use vDSP when available
5. **Memory Ordering**: Consider false sharing in parallel operations

## Next Steps

With unified storage complete:
1. Benchmark allocation/deallocation performance
2. Verify SIMD operation efficiency
3. Test COW behavior
4. Proceed to [Phase 4: Vector Implementation](Phase4_Vectors.md)