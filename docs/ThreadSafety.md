# VectorCore Thread Safety Guide

VectorCore is designed with thread safety as a core principle, enabling safe concurrent operations without explicit synchronization in most cases. This guide covers the thread safety guarantees, design patterns, and best practices.

## Core Thread Safety Guarantees

### 1. Value Semantics
All vector types in VectorCore have **value semantics**, making them inherently thread-safe:

```swift
let vec1 = Vector512.random(in: -1...1)
let vec2 = vec1  // Creates a copy, not a reference

// Safe concurrent access from multiple threads
Task { print(vec1.magnitude) }
Task { print(vec2.magnitude) }
```

### 2. Immutable Operations
Most vector operations create new instances rather than modifying existing ones:

```swift
let a = Vector256.random(in: -1...1)
let b = Vector256.random(in: -1...1)

// All of these are thread-safe
let sum = a + b          // Creates new vector
let scaled = a * 2.0     // Creates new vector
let normalized = a.normalized()  // Creates new vector
```

### 3. Copy-on-Write (COW)
`DynamicVector` uses COW optimization for efficient memory usage while maintaining thread safety:

```swift
var vec1 = DynamicVector(dimension: 1000)
let vec2 = vec1  // Cheap copy, shares storage

// Thread 1
Task {
    vec1[0] = 42.0  // Triggers COW, creates separate storage
}

// Thread 2
Task {
    print(vec2[0])  // Always reads 0.0, unaffected by vec1 changes
}
```

## What's Thread-Safe

### ✅ Safe for Concurrent Use

1. **All Read Operations**
   ```swift
   // Multiple threads can read the same vector concurrently
   let shared = Vector768.random(in: -1...1)
   
   await withTaskGroup(of: Float.self) { group in
       for _ in 0..<100 {
           group.addTask {
               shared.magnitude  // Safe concurrent read
           }
       }
   }
   ```

2. **Mathematical Operations**
   ```swift
   // All math operations are thread-safe
   let v1 = Vector512.random(in: -1...1)
   let v2 = Vector512.random(in: -1...1)
   
   Task { let dot = v1.dotProduct(v2) }
   Task { let dist = v1.distance(to: v2) }
   Task { let cos = v1.cosineSimilarity(to: v2) }
   ```

3. **Factory Methods**
   ```swift
   // Creating vectors is thread-safe
   await withTaskGroup(of: Void.self) { group in
       for _ in 0..<1000 {
           group.addTask {
               _ = VectorFactory.random(dimension: 256)
               _ = VectorFactory.ones(dimension: 512)
               _ = VectorFactory.basis(dimension: 768, at: 100)
           }
       }
   }
   ```

4. **Batch Operations**
   ```swift
   // BatchOperations are designed for concurrent use
   let vectors = (0..<1000).map { _ in Vector512.random(in: -1...1) }
   let query = Vector512.random(in: -1...1)
   
   // Safe to call from multiple threads
   Task {
       let nearest = await BatchOperations.findNearest(to: query, in: vectors, k: 10)
   }
   ```

## What's NOT Thread-Safe

### ❌ Unsafe for Concurrent Use

1. **Mutating Operations on Shared Variables**
   ```swift
   var shared = Vector256.random(in: -1...1)
   
   // UNSAFE: Multiple threads modifying the same variable
   Task { shared[0] = 1.0 }  // Race condition!
   Task { shared[1] = 2.0 }  // Race condition!
   ```

2. **In-Place Operations**
   ```swift
   var vec = Vector512.random(in: -1...1)
   
   // UNSAFE: Concurrent in-place modifications
   Task { vec += Vector512.ones() }  // Race condition!
   Task { vec *= 2.0 }               // Race condition!
   ```

## Safe Concurrent Patterns

### 1. Actor-Based State Management
```swift
actor VectorStore {
    private var vectors: [Vector512] = []
    
    func add(_ vector: Vector512) {
        vectors.append(vector)
    }
    
    func findNearest(to query: Vector512, k: Int) -> [(index: Int, distance: Float)] {
        vectors.enumerated()
            .map { ($0.offset, query.distance(to: $0.element)) }
            .sorted { $0.1 < $1.1 }
            .prefix(k)
            .map { ($0.0, $0.1) }
    }
}

// Usage
let store = VectorStore()

// Safe concurrent access
await withTaskGroup(of: Void.self) { group in
    for _ in 0..<100 {
        group.addTask {
            await store.add(Vector512.random(in: -1...1))
        }
    }
}
```

### 2. Immutable Shared State
```swift
// Create immutable dataset
let dataset = (0..<10_000).map { _ in Vector768.random(in: -1...1) }

// Safe concurrent queries
await withTaskGroup(of: [(Int, Float)].self) { group in
    for _ in 0..<100 {
        group.addTask {
            let query = Vector768.random(in: -1...1)
            return await BatchOperations.findNearest(to: query, in: dataset, k: 50)
        }
    }
}
```

### 3. Thread-Local Processing
```swift
func processVectorsConcurrently(_ vectors: [Vector256]) async -> [Float] {
    await withTaskGroup(of: [Float].self) { group in
        let chunkSize = vectors.count / ProcessInfo.processInfo.activeProcessorCount
        
        for chunk in vectors.chunked(into: chunkSize) {
            group.addTask {
                // Each task processes its own chunk
                chunk.map { $0.magnitude }
            }
        }
        
        var results: [Float] = []
        for await chunkResults in group {
            results.append(contentsOf: chunkResults)
        }
        return results
    }
}
```

## Storage-Specific Considerations

### Small/Medium/Large Vector Storage
These storage types use value semantics and are always thread-safe:

```swift
let small = SmallVectorStorage(count: 32)   // Stack-allocated
let medium = MediumVectorStorage(count: 256) // Heap-allocated, value semantics
let large = LargeVectorStorage(count: 1536)  // Heap-allocated, value semantics

// All safe for concurrent read access
Task { print(small.count) }
Task { print(medium.count) }
Task { print(large.count) }
```

### COWDynamicStorage
Uses copy-on-write for efficiency while maintaining thread safety:

```swift
var storage1 = COWDynamicStorage(dimension: 1000)
let storage2 = storage1  // Shares underlying storage

// First write triggers copy
storage1[0] = 42.0  // Now has separate storage

// Safe to use both concurrently after COW
Task { print(storage1[0]) }  // 42.0
Task { print(storage2[0]) }  // 0.0
```

## Best Practices

### 1. Prefer Immutable Patterns
```swift
// Good: Create new vectors
let normalized = vector.normalized()
let scaled = vector * 2.0

// Avoid: Mutating shared state
var shared = vector
shared.normalize()  // Potential race condition if shared
```

### 2. Use Actors for Mutable State
```swift
actor VectorProcessor {
    private var cache: [String: Vector512] = [:]
    
    func process(_ key: String) async -> Vector512 {
        if let cached = cache[key] {
            return cached
        }
        
        let result = computeExpensiveVector(key)
        cache[key] = result
        return result
    }
}
```

### 3. Batch Operations for Performance
```swift
// Good: Process in batches
let results = await BatchOperations.map(vectors) { $0.normalized() }

// Less efficient: Individual concurrent tasks
let results = await withTaskGroup(of: Vector512.self) { group in
    for vector in vectors {
        group.addTask { vector.normalized() }
    }
    // ... collect results
}
```

### 4. Profile Concurrent Code
```swift
let start = CFAbsoluteTimeGetCurrent()

let results = await withTaskGroup(of: Float.self) { group in
    for vector in vectors {
        group.addTask { vector.magnitude }
    }
    // ... collect results
}

let elapsed = CFAbsoluteTimeGetCurrent() - start
print("Concurrent processing: \(elapsed)s")
```

## Common Pitfalls

### 1. Shared Mutable State
```swift
// BAD: Race condition
var counter = 0
await withTaskGroup(of: Void.self) { group in
    for _ in 0..<1000 {
        group.addTask { counter += 1 }  // RACE CONDITION!
    }
}

// GOOD: Use actor
actor Counter {
    private var value = 0
    func increment() { value += 1 }
    func getValue() -> Int { value }
}
```

### 2. False Sharing
```swift
// BAD: Multiple threads updating adjacent array elements
var results = [Float](repeating: 0, count: 100)
await withTaskGroup(of: Void.self) { group in
    for i in 0..<100 {
        group.addTask {
            results[i] = expensiveComputation(i)  // False sharing!
        }
    }
}

// GOOD: Collect results separately
let results = await withTaskGroup(of: (Int, Float).self) { group in
    for i in 0..<100 {
        group.addTask {
            (i, expensiveComputation(i))
        }
    }
    
    var collected = [Float](repeating: 0, count: 100)
    for await (index, value) in group {
        collected[index] = value
    }
    return collected
}
```

## Performance Considerations

### 1. Contention vs. Parallelism
Balance the number of concurrent tasks with the work being done:

```swift
// For light operations, use fewer tasks
let optimalTasks = min(vectors.count, ProcessInfo.processInfo.activeProcessorCount * 2)

// For heavy operations, can use more tasks
let optimalTasks = min(vectors.count, ProcessInfo.processInfo.activeProcessorCount * 4)
```

### 2. Memory Bandwidth
Vector operations are often memory-bandwidth limited:

```swift
// May not scale linearly with cores due to memory bandwidth
await BatchOperations.pairwiseDistances(vectors)

// Consider chunking for better cache utilization
let chunkSize = 1000  // Tune based on your system
for chunk in vectors.chunked(into: chunkSize) {
    await processChunk(chunk)
}
```

## Testing Thread Safety

Example test patterns for verifying thread safety:

```swift
func testConcurrentAccess() async {
    let vector = Vector512.random(in: -1...1)
    let iterations = 10_000
    
    // Concurrent reads should always be safe
    await withTaskGroup(of: Float.self) { group in
        for _ in 0..<100 {
            group.addTask {
                var sum: Float = 0
                for _ in 0..<iterations {
                    sum += vector.magnitude
                }
                return sum
            }
        }
        
        let expected = vector.magnitude * Float(iterations)
        for await sum in group {
            XCTAssertEqual(sum, expected, accuracy: 1e-3)
        }
    }
}
```

## Summary

VectorCore achieves thread safety through:
1. **Value semantics** for all vector types
2. **Immutable operations** that create new instances
3. **Copy-on-Write** for efficient memory usage
4. **Sendable conformance** for all public types

By following the patterns and practices in this guide, you can safely use VectorCore in concurrent environments without explicit synchronization in most cases.