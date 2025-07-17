# VectorCore Error Handling Guidelines

## Philosophy

VectorCore follows Swift's error handling conventions: **fast by default, safe by opt-in**. This approach provides:
- Maximum performance for the common case
- Safety options when error recovery is needed
- Clear API contracts through naming conventions

## Swift Error Handling Conventions

### 1. Preconditions for Programming Errors (Fast Path)

**When to use:**
- Array/vector bounds checking in subscripts
- Dimension validation in constructors
- Invariant violations that indicate bugs
- Performance-critical inner loops

**Rationale:** These represent programming errors that should be caught during development. In release builds, preconditions compile to minimal checks, maximizing performance.

**Examples:**
```swift
// Standard subscript - fast, crashes on programming error
public subscript(index: Int) -> Float {
    get {
        precondition(index >= 0 && index < D.value, "Index out of bounds")
        return storage[index]
    }
}

// Standard initialization - fast, crashes on wrong dimension
public init(_ values: [Float]) {
    precondition(values.count == D.value, "Array count must match dimension")
    self.storage = D.Storage(from: values)
}
```

### 2. Safe Variants with Optionals (Opt-in Safety)

**When to use:**
- Operations where bounds checking is expected
- User input validation
- Dynamic data processing
- Any operation where graceful failure is preferred

**Naming convention:** Use descriptive names or `safe:` parameter labels

**Examples:**
```swift
// Safe subscript access - returns nil instead of crashing
public func at(_ index: Int) -> Float? {
    guard index >= 0 && index < D.value else { return nil }
    return storage[index]
}

// Safe initialization - returns nil on dimension mismatch
public init?(safe values: [Float]) {
    guard values.count == D.value else { return nil }
    self.storage = D.Storage(from: values)
}

// Safe basis vector creation
public static func basis(safe index: Int) -> Vector<D>? {
    guard index >= 0 && index < D.value else { return nil }
    var vector = Vector<D>()
    vector[index] = 1.0
    return vector
}
```

### 3. Throwing Functions for Complex Errors

**When to use:**
- Operations that can fail in multiple ways
- When detailed error information is needed
- I/O operations (serialization, file access)
- Network operations
- Complex validation scenarios

**Examples:**
```swift
// Factory method with detailed error
public static func vector(of dimension: Int, from values: [Float]) throws -> any VectorType {
    guard values.count == dimension else {
        throw VectorError.dimensionMismatch(expected: dimension, actual: values.count)
    }
    // ... implementation
}

// Safe division with error details
public static func safeDivide(_ lhs: Vector<D>, by rhs: Vector<D>) throws -> Vector<D> {
    // Check for zeros and throw detailed error
    if hasZeros(in: rhs) {
        throw VectorError.divisionByZero(operation: "safeDivide")
    }
    return lhs ./ rhs
}

// Serialization with multiple failure modes
public static func decodeBinary(from data: Data) throws -> Vector<D> {
    // Can throw: insufficientData, invalidDataFormat, dataCorruption
    try BinaryFormat.validateHeader(in: data)
    try BinaryFormat.validateChecksum(in: data)
    // ... implementation
}
```

### 4. Result Type for Functional Error Handling

**When to use:**
- Functional programming style
- Error aggregation in batch operations
- When you want to chain operations

**Examples:**
```swift
// Result-based API
public func safeDotProduct(_ other: Vector<D>) -> Result<Float, VectorError> {
    guard magnitude > 0 && other.magnitude > 0 else {
        return .failure(.zeroVectorError(operation: "dot product"))
    }
    return .success(dotProduct(other))
}

// Batch processing with error aggregation
func processBatch(_ vectors: [Vector<D>]) -> (successes: [Float], failures: [VectorError]) {
    vectors.reduce(into: ([], [])) { result, vector in
        switch vector.safeMagnitude() {
        case .success(let magnitude):
            result.successes.append(magnitude)
        case .failure(let error):
            result.failures.append(error)
        }
    }
}
```

## API Design Patterns

### Pattern 1: Default Fast, Opt-in Safe

Provide both fast (precondition-based) and safe (optional/throwing) variants:

```swift
// Fast default - used in 90% of cases
let value = vector[index]  // Crashes on bad index

// Safe variant - used when index might be invalid
if let value = vector.at(index) {
    // Handle value
}

// Fast initialization
let v1 = Vector<Dim128>(values)  // Crashes if count != 128

// Safe initialization  
if let v2 = Vector<Dim128>(safe: values) {
    // Handle successful creation
}
```

### Pattern 2: Progressive Error Detail

Provide multiple levels of error handling based on needs:

```swift
// Level 1: Boolean success
if vector.setAt(index, to: value) {
    // Success
}

// Level 2: Optional result
if let normalized = vector.normalizedSafe() {
    // Use normalized vector
}

// Level 3: Detailed error
do {
    let result = try vector.validateRange(0...1)
} catch let error as VectorError {
    // Handle specific error with full context
}
```

### Pattern 3: Batch Operations with Error Collection

For operations on collections, collect rather than fail fast:

```swift
public static func normalizeAll(_ vectors: [Vector<D>]) -> (normalized: [Vector<D>], errors: [(index: Int, error: VectorError)]) {
    var normalized: [Vector<D>] = []
    var errors: [(index: Int, error: VectorError)] = []
    
    for (index, vector) in vectors.enumerated() {
        if vector.magnitude > 0 {
            normalized.append(vector.normalized())
        } else {
            errors.append((index, .zeroVectorError(operation: "normalize")))
        }
    }
    
    return (normalized, errors)
}
```

## Performance Considerations

### Choose the Right Tool

1. **Hot paths**: Use preconditions or unsafe operations
   ```swift
   // Inner loop of matrix multiplication
   result[i] = vector1[i] * vector2[i]  // No bounds checking
   ```

2. **User-facing APIs**: Provide safe variants
   ```swift
   // Public API processing user input
   guard let vector = Vector<Dim512>(safe: userProvidedData) else {
       return .invalidInput
   }
   ```

3. **Batch operations**: Use error aggregation
   ```swift
   // Process thousands of vectors, don't fail on first error
   let results = BatchOperations.processWithErrors(vectors)
   ```

### Performance Impact

| Operation | Precondition | Optional | Throwing | Result |
|-----------|-------------|----------|----------|---------|
| Overhead | ~0% | ~5-10% | ~10-20% | ~15-25% |
| Use case | Hot paths | Simple validation | Complex errors | Functional style |

## Migration Guide

### From Unsafe to Safe APIs

1. **Keep existing fast APIs** - Don't break existing code
2. **Add safe variants** - Use parameter labels or new method names
3. **Document clearly** - Explain when to use each variant
4. **Deprecate carefully** - Only remove unsafe APIs if truly dangerous

Example migration:
```swift
// Phase 1: Add safe variant
extension Vector {
    // Existing fast API
    public subscript(index: Int) -> Float { 
        get { precondition(...); return storage[index] }
    }
    
    // New safe API
    public func at(_ index: Int) -> Float? {
        guard index >= 0 && index < dimension else { return nil }
        return storage[index]
    }
}

// Phase 2: Guide users to safe API where appropriate
// (Keep fast API for performance-critical code)
```

## Error Types Reference

VectorCore uses a unified `VectorError` type with rich context:

```swift
public struct VectorError: Error {
    public let kind: ErrorKind        // Categorized error type
    public let context: ErrorContext  // Source location, timestamp, etc.
    public let underlyingError: Error? // Optional wrapped error
    public var errorChain: [VectorError] // For error composition
}
```

Common error factories:
- `VectorError.dimensionMismatch(expected:actual:)`
- `VectorError.indexOutOfBounds(index:dimension:)`
- `VectorError.divisionByZero(operation:)`
- `VectorError.zeroVectorError(operation:)`
- `VectorError.invalidValues(indices:reason:)`

## Best Practices

### DO:
✅ Use preconditions for programming errors in hot paths
✅ Provide safe variants for public APIs that process external data
✅ Return optionals for simple "found/not found" scenarios
✅ Throw detailed errors when multiple failure modes exist
✅ Document performance characteristics of each API variant
✅ Use clear naming: `at()` for safe subscript, `safe:` parameter labels

### DON'T:
❌ Use `fatalError` in production code
❌ Throw errors in performance-critical inner loops
❌ Force users to handle errors for programming mistakes
❌ Mix error handling styles inconsistently
❌ Remove fast APIs without performance justification

## Summary

VectorCore's error handling follows Swift conventions:
- **Fast by default**: Preconditions for maximum performance
- **Safe when needed**: Optional and throwing variants available
- **Clear naming**: Safe variants are clearly marked
- **Progressive detail**: From booleans to detailed errors
- **No surprises**: Consistent patterns throughout the library

Choose the right tool for each situation, prioritizing performance in hot paths and safety in user-facing APIs.