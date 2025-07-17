# VectorCore Thread Safety Guide

## Overview

VectorCore is designed with thread safety in mind. Most operations work on value types (structs) which are inherently thread-safe. This guide documents the thread safety guarantees and best practices.

## Thread Safety Guarantees

### ✅ Thread-Safe by Design

1. **Vector Operations**: All `Vector<D>` operations are thread-safe
   - Vectors are value types (structs)
   - No shared mutable state
   - Copy-on-write semantics for storage

2. **Immutable Operations**: All mathematical operations create new values
   ```swift
   // Thread-safe: Each thread gets its own copy
   let v1 = Vector<Dim128>.random(in: -1...1)
   let v2 = Vector<Dim128>.random(in: -1...1)
   
   // Safe to use from multiple threads
   let sum = v1 + v2
   let dot = v1.dotProduct(v2)
   ```

3. **Logger Configuration**: Thread-safe with internal locking
   ```swift
   // Safe to configure from any thread
   Logger.configuration.minimumLevel = .warning
   ```

### ⚠️ Thread-Safe with Caveats

1. **BatchOperations Configuration**: Thread-safe access via dedicated API
   ```swift
   // ✅ Correct: Use thread-safe update method
   BatchOperations.updateConfiguration { config in
       config.parallelThreshold = 500
       config.minimumChunkSize = 128
   }
   
   // ✅ Correct: Safe read access
   let threshold = BatchOperations.currentConfiguration.parallelThreshold
   
   // ❌ Wrong: Direct access no longer available
   // BatchOperations.configuration.parallelThreshold = 500  // Won't compile
   ```

2. **Global Configuration**: Immutable after initialization
   ```swift
   // Set once at app startup
   let config = VectorCore.configuration  // Immutable
   ```

### ❌ Not Thread-Safe

Currently, all core VectorCore operations are thread-safe. The library avoids mutable shared state.

## Best Practices

### 1. Configuration at Startup

Configure VectorCore settings during app initialization before concurrent operations begin:

```swift
// AppDelegate or main()
func setupVectorCore() {
    // Configure batch operations
    BatchOperations.updateConfiguration { config in
        config.parallelThreshold = 1000
        config.oversubscription = 2.0
    }
    
    // Configure logging
    Logger.configuration.minimumLevel = .info
    Logger.configuration.enableFileOutput = true
}
```

### 2. Concurrent Vector Processing

Vectors can be safely shared across threads:

```swift
// Safe concurrent processing
await withTaskGroup(of: Float.self) { group in
    let sharedVector = Vector<Dim512>.random(in: -1...1)
    
    for otherVector in vectors {
        group.addTask {
            // Each task can safely read sharedVector
            return sharedVector.distance(to: otherVector)
        }
    }
}
```

### 3. Batch Operations

BatchOperations automatically handle concurrency:

```swift
// Automatically parallelized for large datasets
let results = await BatchOperations.findNearest(
    to: query,
    in: largeVectorSet,  // Thread-safe parallel processing
    k: 100
)
```

### 4. Safe Division Operations

Use safe variants when division by zero is possible:

```swift
// Thread-safe error handling
do {
    let result = try Vector.safeDivide(v1, by: v2)
} catch {
    // Handle division by zero
}

// Or with default value
let result = Vector.safeDivide(v1, by: v2, default: 0.0)
```

## Actor-Based Patterns

For complex stateful operations, consider using Swift actors:

```swift
actor VectorProcessor {
    private var cache: [String: Vector<Dim512>] = [:]
    
    func process(key: String, vector: Vector<Dim512>) -> Vector<Dim512> {
        if let cached = cache[key] {
            return cached
        }
        
        let processed = vector.normalized()
        cache[key] = processed
        return processed
    }
}
```

## Performance Considerations

1. **Value Type Copying**: Vectors are copied when passed between threads
   - Small vectors (≤512 dimensions): Negligible overhead
   - Large vectors: Consider using `ArraySlice` or indices

2. **Parallel Thresholds**: Automatic parallelization kicks in for:
   - General operations: >1000 vectors
   - Pairwise distances: >100 vectors
   - Adjust via `BatchOperations.updateConfiguration`

3. **Memory Contention**: For best performance:
   - Process vectors in batches
   - Use `BatchOperations` for automatic optimization
   - Avoid false sharing by processing independent data

## Testing Thread Safety

Always test concurrent code thoroughly:

```swift
func testConcurrentVectorOperations() async {
    let vectors = (0..<1000).map { _ in Vector<Dim256>.random(in: -1...1) }
    
    // Concurrent reads are safe
    await withTaskGroup(of: Float.self) { group in
        for vector in vectors {
            group.addTask {
                vector.magnitude  // Safe concurrent access
            }
        }
    }
}
```

## Summary

- ✅ All vector operations are thread-safe
- ✅ Configuration has thread-safe accessors  
- ✅ Batch operations handle concurrency automatically
- ✅ Logger is thread-safe with internal synchronization
- ❌ Avoid modifying configuration during concurrent operations