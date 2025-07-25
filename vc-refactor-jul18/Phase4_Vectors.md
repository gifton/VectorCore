# Phase 4: Vector Implementation

## Overview

With our Swift 6.0 protocol architecture and unified storage system in place, we can now implement the vector types themselves. This phase creates a single, unified implementation that eliminates ~400 lines of duplication between Vector and DynamicVector.

## Design Decisions Applied

Based on previous phases:
- Swift 6.0 with borrowing/consuming semantics
- Platform-specific alignment (16 bytes on Apple Silicon)
- Small vector optimization for ≤16 elements
- Protocol-based code sharing

## Core Vector Implementation

### Base Vector Type

```swift
// Compile-time dimensioned vector
public struct Vector<D: Dimension>: VectorProtocol, Sendable {
    public typealias Scalar = Float
    public typealias Storage = DimensionStorage<D, Float>
    
    @usableFromInline
    internal var storage: Storage
    
    @inlinable
    public var scalarCount: Int { D.size }
    
    // Swift 6.0 memberwise init with consuming
    @inlinable
    public init(storage: consuming Storage) {
        self.storage = consume storage
    }
    
    // Convenience initializers
    @inlinable
    public init(repeating value: Scalar = 0) {
        self.storage = Storage(repeating: value)
    }
    
    @inlinable
    public init(_ elements: borrowing [Scalar]) throws(VectorError) {
        guard elements.count == D.size else {
            throw VectorError.dimensionMismatch(expected: D.size, actual: elements.count)
        }
        self.storage = Storage(elements)
    }
    
    // Borrowing for read-only access
    @inlinable
    public borrowing func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }
    
    // Copy-on-write for mutations
    @inlinable
    public mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }
}
```

### Dynamic Vector Implementation

```swift
// Runtime-dimensioned vector
public struct DynamicVector: VectorProtocol, Sendable {
    public typealias Scalar = Float
    public typealias Storage = HybridStorage<Float>  // Uses SVO
    
    @usableFromInline
    internal var storage: Storage
    
    @usableFromInline
    internal let dimension: Int
    
    @inlinable
    public var scalarCount: Int { dimension }
    
    @inlinable
    public init(dimension: Int, storage: consuming Storage) throws(VectorError) {
        guard storage.count == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: storage.count)
        }
        self.dimension = dimension
        self.storage = consume storage
    }
    
    // Convenience initializers matching Vector<D>
    @inlinable
    public init(dimension: Int, repeating value: Scalar = 0) {
        self.dimension = dimension
        self.storage = Storage(count: dimension, repeating: value)
    }
    
    @inlinable
    public init(_ elements: borrowing [Scalar]) {
        self.dimension = elements.count
        self.storage = Storage(elements)
    }
}
```

### 🛑 DECISION POINT: Subscript Behavior

How should we handle subscript bounds checking?

**Option A: Debug-only checks**
```swift
@inlinable
public subscript(index: Int) -> Scalar {
    get {
        #if DEBUG
        precondition(index >= 0 && index < scalarCount, "Index out of bounds")
        #endif
        return storage[index]
    }
}
```

**Option B: Always check with throwing**
```swift
public subscript(index: Int) throws(VectorError) -> Scalar {
    get throws {
        guard index >= 0 && index < scalarCount else {
            throw VectorError.indexOutOfBounds(index, max: scalarCount - 1)
        }
        return storage[index]
    }
}
```

**Option C: Safe and unsafe variants**
```swift
@inlinable
public subscript(index: Int) -> Scalar {
    get {
        precondition(index >= 0 && index < scalarCount)
        return storage[index]
    }
}

@inlinable
public subscript(unchecked index: Int) -> Scalar {
    get { storage.unsafelyUnwrapped[index] }
}
```

### Unified Protocol Extensions

All shared functionality lives in protocol extensions:

```swift
// Arithmetic operations - shared by all vectors
extension VectorProtocol {
    // Addition with borrowing semantics
    @inlinable
    public static func + (lhs: borrowing Self, rhs: borrowing Self) throws(VectorError) -> Self {
        guard lhs.scalarCount == rhs.scalarCount else {
            throw VectorError.dimensionMismatch(expected: lhs.scalarCount, actual: rhs.scalarCount)
        }
        
        var result = Self(storage: lhs.storage.copy())
        result.withUnsafeMutableBufferPointer { resultBuffer in
            rhs.withUnsafeBufferPointer { rhsBuffer in
                vDSP_vadd(
                    resultBuffer.baseAddress!, 1,
                    rhsBuffer.baseAddress!, 1,
                    resultBuffer.baseAddress!, 1,
                    vDSP_Length(resultBuffer.count)
                )
            }
        }
        return result
    }
    
    // Element-wise multiplication (Hadamard product)
    @inlinable
    public static func .* (lhs: borrowing Self, rhs: borrowing Self) throws(VectorError) -> Self {
        guard lhs.scalarCount == rhs.scalarCount else {
            throw VectorError.dimensionMismatch(expected: lhs.scalarCount, actual: rhs.scalarCount)
        }
        
        var result = Self(storage: lhs.storage.copy())
        result.withUnsafeMutableBufferPointer { resultBuffer in
            rhs.withUnsafeBufferPointer { rhsBuffer in
                vDSP_vmul(
                    resultBuffer.baseAddress!, 1,
                    rhsBuffer.baseAddress!, 1,
                    resultBuffer.baseAddress!, 1,
                    vDSP_Length(resultBuffer.count)
                )
            }
        }
        return result
    }
    
    // Scalar operations
    @inlinable
    public static func * (lhs: borrowing Self, rhs: Scalar) -> Self {
        var result = Self(storage: lhs.storage.copy())
        result.withUnsafeMutableBufferPointer { buffer in
            var scalar = rhs
            vDSP_vsmul(
                buffer.baseAddress!, 1,
                &scalar,
                buffer.baseAddress!, 1,
                vDSP_Length(buffer.count)
            )
        }
        return result
    }
}

// Mathematical operations
extension VectorProtocol {
    @inlinable
    public borrowing func magnitude() -> Scalar {
        var result: Scalar = 0
        withUnsafeBufferPointer { buffer in
            vDSP_svesq(
                buffer.baseAddress!, 1,
                &result,
                vDSP_Length(buffer.count)
            )
        }
        return sqrt(result)
    }
    
    @inlinable
    public borrowing func normalized() throws(VectorError) -> Self {
        let mag = magnitude()
        guard mag > Scalar.ulpOfOne else {
            throw VectorError.zeroMagnitude
        }
        return self * (1.0 / mag)
    }
    
    @inlinable
    public static func dot(_ lhs: borrowing Self, _ rhs: borrowing Self) throws(VectorError) -> Scalar {
        guard lhs.scalarCount == rhs.scalarCount else {
            throw VectorError.dimensionMismatch(expected: lhs.scalarCount, actual: rhs.scalarCount)
        }
        
        var result: Scalar = 0
        lhs.withUnsafeBufferPointer { lhsBuffer in
            rhs.withUnsafeBufferPointer { rhsBuffer in
                vDSP_dotpr(
                    lhsBuffer.baseAddress!, 1,
                    rhsBuffer.baseAddress!, 1,
                    &result,
                    vDSP_Length(lhsBuffer.count)
                )
            }
        }
        return result
    }
}

// Quality metrics
extension VectorProtocol {
    @inlinable
    public borrowing func quality() -> VectorQuality {
        var sum: Scalar = 0
        var sumSquares: Scalar = 0
        var zeros = 0
        
        withUnsafeBufferPointer { buffer in
            // Use vDSP for efficiency
            vDSP_sve(buffer.baseAddress!, 1, &sum, vDSP_Length(buffer.count))
            vDSP_svesq(buffer.baseAddress!, 1, &sumSquares, vDSP_Length(buffer.count))
            
            // Count zeros
            for i in 0..<buffer.count {
                if abs(buffer[i]) < Scalar.ulpOfOne {
                    zeros += 1
                }
            }
        }
        
        let mean = sum / Scalar(scalarCount)
        let variance = (sumSquares / Scalar(scalarCount)) - (mean * mean)
        let sparsity = Scalar(zeros) / Scalar(scalarCount)
        let magnitude = sqrt(sumSquares)
        
        // Simplified entropy calculation
        let entropy = calculateEntropy()
        
        return VectorQuality(
            magnitude: magnitude,
            variance: variance,
            sparsity: sparsity,
            entropy: entropy
        )
    }
}
```

### Type-Specific Features

```swift
// Compile-time vector specific features
extension Vector {
    // Static factory methods
    @inlinable
    public static var zero: Self {
        Self(repeating: 0)
    }
    
    @inlinable
    public static var ones: Self {
        Self(repeating: 1)
    }
    
    // Random generation
    @inlinable
    public static func random(in range: ClosedRange<Scalar> = 0...1) -> Self {
        var result = Self.zero
        result.withUnsafeMutableBufferPointer { buffer in
            for i in 0..<buffer.count {
                buffer[i] = Scalar.random(in: range)
            }
        }
        return result
    }
}

// Dynamic vector specific features
extension DynamicVector {
    // Resizing operations
    @inlinable
    public consuming func resized(to newDimension: Int, fill: Scalar = 0) -> DynamicVector {
        var newStorage = Storage(count: newDimension, repeating: fill)
        let copyCount = min(dimension, newDimension)
        
        withUnsafeBufferPointer { source in
            newStorage.withUnsafeMutableBufferPointer { dest in
                dest.baseAddress!.initialize(from: source.baseAddress!, count: copyCount)
            }
        }
        
        consume self
        return DynamicVector(dimension: newDimension, storage: newStorage)
    }
}
```

### Serialization Support

```swift
// Codable conformance
extension Vector: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let elements = try container.decode([Scalar].self)
        try self.init(elements)
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(Array(self))
    }
}

// Binary format for performance
extension VectorProtocol {
    public borrowing func encodeBinary() -> Data {
        var data = Data(capacity: scalarCount * MemoryLayout<Scalar>.size + 8)
        
        // Header: dimension (4 bytes) + checksum placeholder (4 bytes)
        withUnsafeBytes(of: UInt32(scalarCount).littleEndian) { data.append(contentsOf: $0) }
        data.append(contentsOf: [0, 0, 0, 0])  // Checksum placeholder
        
        // Vector data
        withUnsafeBufferPointer { buffer in
            buffer.forEach { element in
                withUnsafeBytes(of: element) { data.append(contentsOf: $0) }
            }
        }
        
        // Calculate and update checksum
        let checksum = data.dropFirst(8).reduce(UInt32(0), { $0 &+ UInt32($1) })
        data.replaceSubrange(4..<8, with: withUnsafeBytes(of: checksum.littleEndian, Array.init))
        
        return data
    }
}
```

### 🛑 DECISION POINT: Collection Conformance

Should vectors conform to Collection/Sequence?

**Option A: Full Collection conformance**
```swift
extension Vector: Collection {
    public var startIndex: Int { 0 }
    public var endIndex: Int { D.size }
    // ... full implementation
}
```

**Option B: Sequence only**
```swift
extension Vector: Sequence {
    public func makeIterator() -> IndexingIterator<Self> {
        // Simpler, iteration only
    }
}
```

**Option C: No collection conformance**
- Pros: Cleaner API, no confusion with arrays
- Cons: Can't use collection algorithms directly

## Implementation Validation

```swift
// Verify no code duplication
func testProtocolSharing() {
    let v1 = Vector<Dim128>(repeating: 1.0)
    let v2 = Vector<Dim128>(repeating: 2.0)
    let v3 = DynamicVector(dimension: 128, repeating: 1.0)
    let v4 = DynamicVector(dimension: 128, repeating: 2.0)
    
    // Both use same protocol implementation
    let staticResult = try v1 + v2
    let dynamicResult = try v3 + v4
    
    // Verify same operation
    assert(staticResult.magnitude() == dynamicResult.magnitude())
}
```

## Performance Considerations

1. **Inlining**: All hot paths marked @inlinable
2. **COW**: Leverages storage's copy-on-write
3. **SIMD**: Direct vDSP usage throughout
4. **Allocation**: Small vectors use stack storage
5. **Borrowing**: Read operations don't copy

## Next Steps

With unified vectors complete:
1. Verify zero code duplication
2. Benchmark vs original implementation
3. Test all operations
4. Proceed to [Phase 5: Modern Operations Layer](Phase5_Operations.md)