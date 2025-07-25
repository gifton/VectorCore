# Phase 2: Core Protocol Architecture

## Overview

With Swift 6.0 as our foundation, we can leverage the latest language features to create a cutting-edge protocol architecture. This phase establishes the protocol hierarchy that will unify all vector operations while maintaining zero-cost abstractions.

## Swift 6.0 Features We'll Leverage

1. **Parameter Packs** - For variadic generic operations
2. **Improved Existentials** - Better protocol-based type erasure
3. **Typed Throws** - More precise error handling
4. **Borrowing and Consuming** - Optimal memory management
5. **Strict Concurrency** - Data race safety by design

## Core Protocol Hierarchy

### Base Protocol

```swift
// Leverage Swift 6.0's improved protocol design
public protocol VectorProtocol: Sendable {
    associatedtype Scalar: BinaryFloatingPoint & SIMDScalar
    associatedtype Storage: VectorStorage where Storage.Element == Scalar
    
    // Borrowing for read-only access
    borrowing var storage: Storage { get }
    var scalarCount: Int { get }
    
    // Consuming for move semantics
    consuming func into() -> Storage
    
    // Subscript with bounds checking in debug
    subscript(index: Int) -> Scalar { get }
    
    // Swift 6.0 typed throws for precise errors
    init(storage: consuming Storage) throws(VectorError)
}

// Leverage parameter packs for variadic operations
public protocol VectorArithmetic: VectorProtocol {
    // Using parameter packs for n-ary operations
    static func sum<each T: VectorProtocol>(_ vectors: repeat each T) -> Self 
        where repeat (each T).Scalar == Scalar
    
    // Borrowing for efficient read-only operations
    static func add(_ lhs: borrowing Self, _ rhs: borrowing Self) -> Self
    static func subtract(_ lhs: borrowing Self, _ rhs: borrowing Self) -> Self
    static func multiply(_ lhs: borrowing Self, _ rhs: borrowing Self) -> Self
}
```

### Storage Protocol with Swift 6.0 Features

```swift
public protocol VectorStorage: ~Copyable, Sendable {
    associatedtype Element: BinaryFloatingPoint & SIMDScalar
    
    var count: Int { get }
    
    // Borrowing and consuming for optimal performance
    borrowing func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Element>) throws -> R
    ) rethrows -> R
    
    consuming func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
    ) rethrows -> R
    
    // Swift 6.0 move-only type support
    init(count: Int, alignment: Int) throws(AllocationError)
    
    // Explicit copy when needed
    func copy() -> Self
}

// Concrete storage can be move-only for efficiency
public struct AlignedStorage<Element: BinaryFloatingPoint & SIMDScalar>: VectorStorage, ~Copyable {
    private let buffer: UnsafeMutableBufferPointer<Element>
    private let alignment: Int
    
    public consuming func into() -> UnsafeMutableBufferPointer<Element> {
        consume self
        return buffer
    }
}
```

### Protocol Extensions with Modern Swift

```swift
extension VectorProtocol {
    // Leverage Swift 6.0's improved generics
    @inlinable
    public borrowing func magnitude() -> Scalar {
        var sum: Scalar = 0
        withUnsafeBuffer { buffer in
            vDSP.sumOfSquares(buffer, result: &sum)
        }
        return sqrt(sum)
    }
    
    // Parameter packs for flexible operations
    @inlinable
    public static func dotProduct<each T: VectorProtocol>(
        _ vectors: repeat each T
    ) -> Scalar where repeat (each T).Scalar == Scalar {
        // Implementation using parameter pack expansion
    }
    
    // Consuming self for move semantics
    @inlinable
    public consuming func normalized() throws(VectorError) -> Self {
        let mag = magnitude()
        guard mag > Scalar.ulpOfOne else {
            throw VectorError.zeroMagnitude
        }
        
        var storage = self.into()
        storage.withUnsafeMutableBufferPointer { buffer in
            vDSP.divideScalar(buffer, by: mag, result: buffer)
        }
        
        return try Self(storage: consume storage)
    }
}
```

### 🛑 DECISION POINT: Error Handling Strategy

With Swift 6.0's typed throws, how should we handle errors?

**Option A: Specific Error Types**
```swift
enum VectorError: Error {
    case dimensionMismatch(expected: Int, actual: Int)
    case allocationFailed(size: Int)
    case zeroMagnitude
}

func add(_ other: Vector) throws(VectorError) -> Vector
```

**Option B: Protocol-Based Errors**
```swift
protocol VectorError: Error {}
struct DimensionError: VectorError { }
struct AllocationError: VectorError { }

func add<E: VectorError>(_ other: Vector) throws(E) -> Vector
```

**Option C: Result Type with Typed Errors**
```swift
typealias VectorResult<T> = Result<T, VectorError>

func add(_ other: Vector) -> VectorResult<Vector>
```

### Advanced Protocol Features

```swift
// Leverage existential improvements
public protocol AnyVector: VectorProtocol {
    // Type-erased operations using any keyword improvements
    func distance(to other: any AnyVector, using metric: any DistanceMetric) -> Scalar
}

// Conditional conformances with Swift 6.0 improvements
extension Array: VectorProtocol where Element: BinaryFloatingPoint & SIMDScalar {
    public typealias Scalar = Element
    public typealias Storage = ContiguousArray<Element>
    
    public borrowing var storage: Storage {
        ContiguousArray(self)
    }
}

// Actor-based operations for thread safety
public actor VectorProcessor {
    private var cache: [String: any VectorProtocol] = [:]
    
    public func process<V: VectorProtocol>(_ vector: consuming V) async -> V {
        // Thread-safe processing with actor isolation
    }
}
```

### Protocol Composition for Features

```swift
// Compose protocols for different capabilities
public typealias NumericVector = VectorProtocol & VectorArithmetic & VectorComparison

public protocol VectorComparison: VectorProtocol {
    static func < (lhs: borrowing Self, rhs: borrowing Self) -> Bool
    static func == (lhs: borrowing Self, rhs: borrowing Self) -> Bool
}

public protocol VectorSerialization: VectorProtocol {
    func encode() throws -> Data
    init(from data: Data) throws(DecodingError)
}

// GPU-ready protocol
public protocol AcceleratedVector: VectorProtocol {
    associatedtype AcceleratedStorage: VectorStorage
    
    func toAccelerated(device: ComputeDevice) async throws -> AcceleratedStorage
    init(accelerated: consuming AcceleratedStorage) throws
}
```

### 🛑 DECISION POINT: Operator Design

Should we use custom operators or standard methods?

**Option A: Custom Operators (Current)**
```swift
public static func .* (lhs: Self, rhs: Self) -> Self  // Element-wise multiply
public static func ./ (lhs: Self, rhs: Self) -> Self  // Element-wise divide
```

**Option B: Standard Methods Only**
```swift
public func elementwiseMultiply(with other: Self) -> Self
public func elementwiseDivide(by other: Self) -> Self
```

**Option C: Both with Operators as Sugar**
```swift
public func multiplied(elementwiseBy other: Self) -> Self
public static func .* (lhs: Self, rhs: Self) -> Self {
    lhs.multiplied(elementwiseBy: rhs)
}
```

### Implementation Strategy

```swift
// Start with core protocols
internal protocol _VectorCore {
    associatedtype _Storage
    borrowing var _storage: _Storage { get }
}

// Build public API on top
public protocol VectorProtocol: _VectorCore { }

// This allows us to change internals without breaking API
```

## Validation Steps

Before proceeding to Phase 3:

- [ ] Core protocols compile with Swift 6.0
- [ ] No ambiguous type requirements
- [ ] Borrowing/consuming used appropriately
- [ ] Parameter packs work as expected
- [ ] Actor isolation is correct
- [ ] Typed throws cover all error cases

## Performance Considerations

```swift
// Always inline hot paths
@inlinable
@inline(__always)
public borrowing func dot(_ other: borrowing Self) -> Scalar {
    // Direct vDSP call
}

// Use @_alwaysEmitIntoClient for ABI stability
@_alwaysEmitIntoClient
public borrowing func sum() -> Scalar {
    // Implementation
}
```

## Code Examples

### Creating a Vector Type

```swift
public struct Vector<D: Dimension>: VectorProtocol {
    public typealias Scalar = Float
    public typealias Storage = AlignedStorage<Float>
    
    private let _storage: Storage
    
    public borrowing var storage: Storage { _storage }
    public var scalarCount: Int { D.size }
    
    public init(storage: consuming Storage) throws(VectorError) {
        guard storage.count == D.size else {
            throw VectorError.dimensionMismatch(expected: D.size, actual: storage.count)
        }
        self._storage = consume storage
    }
}
```

### Using Parameter Packs

```swift
extension VectorProtocol {
    // Sum any number of vectors
    public static func sum<each V: VectorProtocol>(
        _ vectors: repeat each V
    ) -> Self where repeat (each V).Scalar == Scalar {
        var result = Self.zero
        repeat result = result + (each vectors)
        return result
    }
}

// Usage
let sum = Vector.sum(v1, v2, v3, v4, v5)
```

This protocol architecture leverages Swift 6.0's cutting-edge features to create a type-safe, performant foundation for VectorCore.

## Next Steps

Once protocols are defined:
1. Implement core protocol conformances
2. Verify zero-cost abstraction with benchmarks
3. Test Swift 6.0 features thoroughly
4. Proceed to [Phase 3: Unified Storage System](Phase3_Storage.md)