# VectorCore Error Handling Quick Reference

## Choose Your Error Handling Approach

### 🏃‍♂️ Fast Path (Default)
Use when: Performance is critical, input is trusted
```swift
let value = vector[10]                        // Subscript access
let v = Vector<Dim128>(array)                 // Initialization  
let basis = Vector<Dim128>.basis(at: 5)       // Basis vector
let result = v1 ./ v2                         // Division (no zero check)
```

### 🛡️ Safe Path (Opt-in)
Use when: Input is untrusted, graceful failure needed
```swift
let value = vector.at(10)                     // Returns Float?
let v = Vector<Dim128>(safe: array)           // Returns Vector?
let basis = Vector<Dim128>.basis(safe: 5)     // Returns Vector?
let result = try Vector.safeDivide(v1, by: v2) // Throws on zero
```

## Quick Decision Tree

```
Is this a hot path (inner loop)?
├─ YES → Use precondition-based API (fast)
└─ NO → Is the input trusted?
    ├─ YES → Use precondition-based API (fast)
    └─ NO → Do you need detailed errors?
        ├─ YES → Use throwing variant
        └─ NO → Use optional variant
```

## API Patterns at a Glance

| Operation | Fast (Default) | Safe (Optional) | Safe (Throwing) |
|-----------|---------------|-----------------|-----------------|
| **Subscript** | `vector[i]` | `vector.at(i)` | - |
| **Set Value** | `vector[i] = x` | `vector.setAt(i, to: x)` | - |
| **Initialize** | `Vector(values)` | `Vector(safe: values)` | `VectorFactory.create(...)` |
| **Basis** | `.basis(at: i)` | `.basis(safe: i)` | - |
| **Division** | `v1 ./ v2` | `Vector.safeDivide(v1, by: v2, default: 0)` | `try Vector.safeDivide(v1, by: v2)` |
| **Pattern** | `.repeatingPattern(p)` | `.repeatingPattern(safe: p)` | - |

## Common Scenarios

### Processing User Input
```swift
// ❌ Don't: May crash on invalid input
let vector = Vector<Dim512>(userInput)

// ✅ Do: Handle invalid dimensions gracefully
guard let vector = Vector<Dim512>(safe: userInput) else {
    return .error("Invalid vector dimension")
}
```

### Batch Processing
```swift
// ❌ Don't: Fail on first error
for array in arrays {
    let vector = Vector<Dim128>(array)  // Crashes on wrong size
}

// ✅ Do: Collect all errors
let results = arrays.compactMap { Vector<Dim128>(safe: $0) }
let errors = arrays.enumerated().compactMap { index, array in
    Vector<Dim128>(safe: array) == nil ? index : nil
}
```

### Performance-Critical Loops
```swift
// ✅ Do: Use fast path in hot loops
for i in 0..<vector.scalarCount {
    result[i] = vector1[i] * vector2[i]  // No bounds checking
}

// ❌ Don't: Add unnecessary safety overhead
for i in 0..<vector.scalarCount {
    if let v1 = vector1.at(i), let v2 = vector2.at(i) {  // Wasteful
        result.setAt(i, to: v1 * v2)
    }
}
```

### Dynamic Index Access
```swift
// ❌ Don't: Assume index is valid
func getValue(at index: Int) -> Float {
    return vector[index]  // May crash
}

// ✅ Do: Use safe access for dynamic indices
func getValue(at index: Int) -> Float? {
    return vector.at(index)
}
```

## Error Types

All throwing functions use `VectorError`:

```swift
do {
    let result = try operation()
} catch let error as VectorError {
    switch error.kind {
    case .dimensionMismatch:
        // Handle dimension errors
    case .divisionByZero:
        // Handle division errors
    case .indexOutOfBounds:
        // Handle index errors
    default:
        // Handle other errors
    }
}
```

## Performance Guide

| Variant | Overhead | When to Use |
|---------|----------|-------------|
| Precondition | ~0% | Hot paths, trusted input |
| Optional | ~5-10% | Simple validation |
| Throwing | ~10-20% | Need error details |
| Result | ~15-25% | Functional style |

## Remember

- **Default = Fast**: Standard APIs use preconditions
- **Opt-in = Safe**: Look for `safe:` parameters or alternative method names
- **Document Usage**: Make it clear which variant to use when
- **Measure First**: Profile before choosing safety over performance