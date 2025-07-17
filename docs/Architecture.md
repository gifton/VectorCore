# VectorCore Architecture

VectorCore is a high-performance Swift library for vector operations, designed with a focus on performance, type safety, and usability. This document provides a comprehensive overview of the library's architecture and design decisions.

## Design Principles

### 1. Performance First
- **SIMD Optimization**: Leverages Apple's Accelerate framework and SIMD instructions
- **Memory Alignment**: Ensures optimal memory alignment for vectorized operations
- **Zero-Cost Abstractions**: Compile-time optimizations eliminate abstraction overhead

### 2. Type Safety
- **Compile-Time Dimensions**: Vector dimensions are encoded in the type system
- **No Runtime Dimension Errors**: Dimension mismatches caught at compile time
- **Protocol-Oriented Design**: Clear interfaces with strong guarantees

### 3. Value Semantics
- **Predictable Behavior**: All vector types behave as values, not references
- **Thread Safety**: Inherent thread safety through immutability
- **Copy-on-Write**: Efficient memory usage for large vectors

### 4. Composability
- **Modular Architecture**: Clear separation of concerns
- **Protocol Extensions**: Rich functionality through protocol composition
- **Flexible Storage**: Multiple storage strategies for different use cases

## Core Architecture

### Type System

```
┌─────────────────────────────────────────────────────────┐
│                    VectorProtocol                       │
│  (Unified interface for all vector types)               │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                 │
┌───────▼────────┐              ┌────────▼────────┐
│ Vector<D>      │              │ DynamicVector   │
│ (Static dims)  │              │ (Runtime dims)  │
└────────────────┘              └─────────────────┘
        │                                 │
        │                                 │
┌───────▼────────────────────────────────▼────────┐
│              Storage Layer                      │
│  • SmallVectorStorage (1-64 dims)              │
│  • MediumVectorStorage (65-512 dims)           │
│  • LargeVectorStorage (513+ dims)              │
│  • COWDynamicStorage (runtime sized)           │
└─────────────────────────────────────────────────┘
```

### Storage Strategy

VectorCore uses a tiered storage approach optimized for different vector sizes:

#### 1. SmallVectorStorage (1-64 dimensions)
```swift
public struct SmallVectorStorage {
    internal var data: SIMD64<Float>
    internal let actualCount: Int
}
```
- **Stack Allocated**: No heap allocation overhead
- **SIMD Register**: Direct CPU register usage
- **Partial Usage**: Supports any size from 1-64 with actualCount tracking

#### 2. MediumVectorStorage (65-512 dimensions)
```swift
public struct MediumVectorStorage {
    internal var buffer: UnsafeMutableBufferPointer<Float>
    internal let count: Int
}
```
- **Heap Allocated**: Managed memory with value semantics
- **Cache Friendly**: Sized for L1/L2 cache optimization
- **16-byte Aligned**: Optimal SIMD performance

#### 3. LargeVectorStorage (513+ dimensions)
```swift
public struct LargeVectorStorage {
    internal var memory: AlignedMemory<Float>
    internal let capacity: Int
}
```
- **Page Aligned**: Optimal for large memory operations
- **Lazy Allocation**: Memory allocated on first use
- **Batch Optimized**: Designed for bulk operations

#### 4. COWDynamicStorage (Runtime dimensions)
```swift
public struct COWDynamicStorage {
    internal var storage: AlignedDynamicArrayStorage
}
```
- **Copy-on-Write**: Efficient sharing until mutation
- **Dynamic Sizing**: Size determined at runtime
- **Reference Counting**: Automatic memory management

### Dimension System

VectorCore uses phantom types to encode dimensions at compile time:

```swift
// Dimension protocol
public protocol Dimension {
    static var value: Int { get }
    associatedtype Storage: VectorStorage
}

// Concrete dimensions
public struct Dim32: Dimension {
    public static let value = 32
    public typealias Storage = Storage32
}

public struct Dim512: Dimension {
    public static let value = 512
    public typealias Storage = Storage512
}

// Usage
let vec32: Vector<Dim32> = [1, 2, 3, /* ... 29 more */]
let vec512: Vector<Dim512> = Vector.random(in: -1...1)
```

### Protocol Hierarchy

```
BaseVectorProtocol
    │
    ├─► ExtendedVectorProtocol
    │       │
    │       └─► VectorType (full interface)
    │
    └─► Minimal interface for basic operations
```

#### BaseVectorProtocol
- Core functionality: initialization, array conversion
- Minimal requirements for vector types

#### ExtendedVectorProtocol
- Mathematical operations: dot product, magnitude, normalization
- Distance metrics: Euclidean, cosine similarity

#### VectorType
- Complete vector interface
- Used by factory methods and batch operations

## Operation Pipeline

### 1. Element Access
```swift
// Direct access path
vector[index] 
    → storage[index]
    → bounds check
    → memory access
```

### 2. Mathematical Operations
```swift
// Dot product pipeline
vector1.dotProduct(vector2)
    → storage1.dotProduct(storage2)
    → withUnsafeBufferPointer
    → vDSP_dotpr (Accelerate)
    → SIMD instructions
```

### 3. Batch Operations
```swift
// Parallel processing pipeline
BatchOperations.findNearest(query, vectors, k)
    → Task group creation
    → Chunk distribution
    → Parallel distance calculation
    → Priority queue merge
    → Result collection
```

## Memory Model

### Alignment Strategy
- **16-byte alignment**: For SIMD128 compatibility
- **64-byte alignment**: For cache line optimization (large vectors)
- **Page alignment**: For huge vectors (memory mapping ready)

### Allocation Patterns
```swift
// Small vectors: Stack allocation
Vector<Dim32>()  // 256 bytes on stack

// Medium vectors: Heap with value semantics
Vector<Dim256>() // ~1KB on heap

// Large vectors: Aligned allocation
Vector<Dim1536>() // ~6KB aligned heap

// Dynamic vectors: COW optimization
DynamicVector(dimension: 10000) // Shared until mutation
```

## Performance Optimizations

### 1. Compile-Time Optimizations
- **@inlinable**: Critical paths marked for cross-module inlining
- **@inline(__always)**: Force inlining for hot paths
- **Generic Specialization**: Compiler generates optimal code per dimension

### 2. Runtime Optimizations
- **Fast Paths**: Special cases for common operations
```swift
// Example: Multiplication by special values
static func * (lhs: Vector<D>, rhs: Float) -> Vector<D> {
    switch rhs {
    case 0: return Vector<D>()      // Return zero vector
    case 1: return lhs              // Return copy
    case -1: return -lhs            // Use negation
    default: // General multiplication
    }
}
```

### 3. SIMD Utilization
- **Accelerate Framework**: Hardware-optimized operations
- **Manual SIMD**: Direct SIMD types for small vectors
- **Vectorization**: Compiler auto-vectorization hints

## Extension Points

### 1. Custom Dimensions
```swift
// Define custom dimension
public struct Dim384: Dimension {
    public static let value = 384
    public typealias Storage = Storage384
}

// Define matching storage
public struct Storage384: VectorStorage {
    // Implementation using MediumVectorStorage
}
```

### 2. Custom Operations
```swift
extension Vector where D.Storage: VectorStorageOperations {
    // Add custom mathematical operations
    public func customOperation() -> Float {
        // Implementation
    }
}
```

### 3. Serialization Formats
```swift
// Implement custom serialization
extension Vector: MyCustomSerializable {
    func serialize(to encoder: MyEncoder) {
        // Custom serialization logic
    }
}
```

## Error Handling Philosophy

### Compile-Time Safety
- Dimension mismatches prevented by type system
- Storage requirements enforced at compile time

### Runtime Validation
- Bounds checking in debug builds
- Graceful handling of edge cases (NaN, infinity)
- Clear error messages with recovery hints

```swift
public enum VectorError: Error {
    case dimensionMismatch(expected: Int, actual: Int)
    case indexOutOfBounds(index: Int, bounds: Range<Int>)
    case invalidDimension(Int)
    case serializationError(String)
}
```

## Future Architecture Considerations

### 1. GPU Acceleration
- Metal Performance Shaders integration points
- Unified memory model for CPU/GPU operations

### 2. Distributed Computing
- Serialization for network transport
- Partitioning strategies for large datasets

### 3. Specialized Hardware
- Apple Neural Engine integration
- Custom accelerator support

### 4. Advanced Storage
- Memory-mapped file support
- Compressed vector storage
- Quantized representations

## Design Decisions Rationale

### Why Phantom Types?
- **Type Safety**: Dimension errors caught at compile time
- **Performance**: No runtime dimension checks needed
- **Clarity**: Vector dimensions visible in type signatures

### Why Multiple Storage Types?
- **Optimization**: Each size range has different performance characteristics
- **Memory Efficiency**: Avoid waste for small vectors
- **Cache Utilization**: Match storage to cache hierarchies

### Why Value Semantics?
- **Predictability**: Easier to reason about
- **Thread Safety**: Inherent safety without locks
- **Swift Integration**: Follows Swift conventions

### Why Protocol-Oriented?
- **Flexibility**: Easy to extend functionality
- **Testability**: Mock implementations for testing
- **Composability**: Build complex operations from simple ones

## Summary

VectorCore's architecture balances several competing concerns:
- **Performance** through specialized storage and SIMD operations
- **Safety** through compile-time dimension checking
- **Usability** through clean APIs and value semantics
- **Flexibility** through protocol-oriented design

The tiered storage system ensures optimal performance across all vector sizes, while the type system prevents common errors at compile time. This architecture provides a solid foundation for high-performance vector computations in Swift.