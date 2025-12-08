# Parallelization Strategy

> **Reading time:** 10 minutes
> **Prerequisites:** [Cache-Friendly Code](./02-Cache-Friendly-Code.md)

---

## The Concept

Parallelization means doing work on multiple CPU cores simultaneously. It can provide dramatic speedups—but it has overhead. The key is knowing when parallelization helps and when it hurts.

**Amdahl's Law:**
```
Speedup = 1 / ((1 - P) + P/N)

Where:
  P = Fraction of work that's parallelizable
  N = Number of cores
```

If 90% of your work is parallelizable and you have 8 cores:
```
Speedup = 1 / (0.1 + 0.9/8) = 1 / 0.2125 ≈ 4.7x
```

Not 8x—the sequential 10% limits your speedup.

---

## Why It Matters

### Parallelization Isn't Free

Every parallel operation has overhead:
- **Thread creation/scheduling:** ~1-10 microseconds
- **Work distribution:** Dividing and collecting results
- **Synchronization:** Locks, atomics, barriers
- **Cache effects:** Cores may compete for the same cache lines

For small operations, overhead exceeds the benefit:

```swift
// ❌ Too small to parallelize: 512 operations
let vector = Vector512Optimized(...)
let magnitude = vector.magnitude  // ~150 ns

// Thread overhead: ~1000+ ns
// Parallel version would be SLOWER

// ✅ Worth parallelizing: 1 million operations
let vectors = [Vector512Optimized](count: 1_000_000)
let distances = vectors.parallelMap { $0.euclideanDistance(to: query) }
// Total work: ~1M × 100ns = 100ms
// Even with overhead, parallelization wins
```

---

## The Technique

### Rule 1: Know Your Threshold

As a rough guide:

| Work Per Item | Items Needed for Parallelism |
|---------------|------------------------------|
| 100 ns | 10,000+ |
| 1 μs | 1,000+ |
| 10 μs | 100+ |
| 100 μs | 10+ |
| 1 ms | Always parallel |

### Rule 2: Use Adaptive Parallelization

Don't always parallelize—decide at runtime:

```swift
func processVectors(_ vectors: [Vector512Optimized]) -> [Float] {
    let threshold = 1000

    if vectors.count < threshold {
        // Serial: Less overhead for small inputs
        return vectors.map { $0.magnitude }
    } else {
        // Parallel: Worth it for large inputs
        return vectors.parallelMap { $0.magnitude }
    }
}
```

### Rule 3: Batch to Amortize Overhead

Instead of parallelizing per-item, parallelize per-chunk:

```swift
// ❌ Fine-grained: High overhead
DispatchQueue.concurrentPerform(iterations: 1_000_000) { i in
    results[i] = vectors[i].magnitude  // One vector per dispatch
}

// ✅ Coarse-grained: Amortized overhead
let chunkSize = 10_000
let chunkCount = (vectors.count + chunkSize - 1) / chunkSize

DispatchQueue.concurrentPerform(iterations: chunkCount) { chunk in
    let start = chunk * chunkSize
    let end = min(start + chunkSize, vectors.count)
    for i in start..<end {
        results[i] = vectors[i].magnitude  // Many vectors per dispatch
    }
}
```

### Rule 4: Avoid False Sharing

When multiple cores write to nearby memory locations, they invalidate each other's caches:

```swift
// ❌ False sharing: Results are adjacent in memory
var results = [Float](repeating: 0, count: 8)

DispatchQueue.concurrentPerform(iterations: 8) { i in
    results[i] = expensiveComputation(i)
    // Core 0 writes results[0], invalidates core 1's cache line
    // Core 1 writes results[1], invalidates core 0's cache line
    // They fight over the same cache line!
}

// ✅ Padding to separate cache lines
let paddedSize = 16  // 16 floats = 64 bytes = 1 cache line
var paddedResults = [Float](repeating: 0, count: 8 * paddedSize)

DispatchQueue.concurrentPerform(iterations: 8) { i in
    paddedResults[i * paddedSize] = expensiveComputation(i)
    // Each core writes to its own cache line
}
```

### Rule 5: Use GCD Correctly

Swift's Grand Central Dispatch handles thread management:

```swift
// Parallel iteration (GCD manages thread count)
DispatchQueue.concurrentPerform(iterations: n) { i in
    // Work for iteration i
}

// Custom concurrent queue
let queue = DispatchQueue(label: "processing", attributes: .concurrent)
let group = DispatchGroup()

for chunk in chunks {
    queue.async(group: group) {
        process(chunk)
    }
}

group.wait()  // Block until all complete
```

---

## In VectorCore

VectorCore uses adaptive parallelization based on heuristics:

### Parallel Heuristics

```swift
// Simplified version of VectorCore's approach
enum ParallelHeuristic {
    /// Determine if work should be parallelized
    static func shouldParallelize(
        itemCount: Int,
        workPerItem: WorkClass,
        vectorType: VectorClass
    ) -> Bool {
        let threshold: Int

        switch (workPerItem, vectorType) {
        case (.light, .optimized):  // e.g., magnitude on Vector512Optimized
            threshold = 10_000  // ~1ms total before parallelizing

        case (.medium, .optimized):  // e.g., distance computation
            threshold = 1_000

        case (.heavy, _):  // e.g., batch with sorting
            threshold = 100

        default:
            threshold = 1_000
        }

        return itemCount >= threshold
    }
}
```

### Batch Operations

**📍 See:** `Sources/VectorCore/Operations/BatchOperations.swift`

Batch operations automatically choose serial vs. parallel:

```swift
public func findNearest<V: VectorProtocol>(
    to query: V,
    in candidates: [V],
    k: Int
) -> [(index: Int, distance: V.Scalar)] {
    // Decide parallelization based on candidate count
    if ParallelHeuristic.shouldParallelize(
        itemCount: candidates.count,
        workPerItem: .medium,
        vectorType: V.self
    ) {
        return parallelFindNearest(to: query, in: candidates, k: k)
    } else {
        return serialFindNearest(to: query, in: candidates, k: k)
    }
}
```

### Pairwise Distance Parallelization

For O(n²) operations, the threshold is lower:

```swift
public func pairwiseDistances<V: VectorProtocol>(_ vectors: [V]) -> [[V.Scalar]] {
    // n² complexity means even small n benefits from parallelization
    let threshold = 100  // 100 vectors = 10,000 pairs

    if vectors.count >= threshold {
        return parallelPairwiseDistances(vectors)
    } else {
        return serialPairwiseDistances(vectors)
    }
}
```

---

## Swift Concurrency (async/await)

Modern Swift offers structured concurrency:

```swift
// Parallel processing with task groups
func processParallel(_ vectors: [Vector512Optimized]) async -> [Float] {
    await withTaskGroup(of: (Int, Float).self) { group in
        for (index, vector) in vectors.enumerated() {
            group.addTask {
                return (index, vector.magnitude)
            }
        }

        var results = [Float](repeating: 0, count: vectors.count)
        for await (index, result) in group {
            results[index] = result
        }
        return results
    }
}
```

Task groups have more overhead than `concurrentPerform` but integrate better with async code.

---

## When NOT to Parallelize

1. **Single vector operations:** Always serial (too small)
2. **I/O bound work:** Parallelizing doesn't help if you're waiting for disk/network
3. **Already on background thread:** Don't nest parallelism unnecessarily
4. **Memory-bandwidth limited:** More cores won't help if RAM is the bottleneck

---

## Key Takeaways

1. **Parallelization has overhead.** Only parallelize when work exceeds ~1,000 operations.

2. **Use adaptive thresholds.** VectorCore decides at runtime based on input size.

3. **Batch to amortize overhead.** Process chunks, not individual items.

4. **Avoid false sharing.** Pad results to separate cache lines.

5. **Amdahl's Law limits speedup.** Even infinite cores can't parallelize the sequential part.

---

## Chapter Complete!

You now understand:
- How to measure performance accurately
- What makes code cache-friendly
- When parallelization helps (and when it hurts)

Time for the capstone—putting everything together by reading real kernel code:

**[→ Chapter 6: Capstone](../06-Capstone/README.md)**
