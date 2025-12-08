# Cache-Friendly Code

> **Reading time:** 10 minutes
> **Prerequisites:** [Chapter 1: Memory Fundamentals](../01-Memory-Fundamentals/README.md)

---

## The Concept

Modern CPUs are much faster than memory. To bridge this gap, CPUs use caches—small, fast memory that stores recently accessed data.

The memory hierarchy:

```
┌─────────────┐
│  Registers  │  ~0 cycles (instant)
└──────┬──────┘
       │
┌──────▼──────┐
│  L1 Cache   │  ~4 cycles (~1 ns)
│   (64 KB)   │
└──────┬──────┘
       │
┌──────▼──────┐
│  L2 Cache   │  ~12 cycles (~4 ns)
│  (512 KB)   │
└──────┬──────┘
       │
┌──────▼──────┐
│  L3 Cache   │  ~40 cycles (~12 ns)
│  (8-24 MB)  │
└──────┬──────┘
       │
┌──────▼──────┐
│    RAM      │  ~200 cycles (~60-100 ns)
│  (16+ GB)   │
└─────────────┘
```

A cache miss (data not in cache) costs 100x more than a cache hit. Writing cache-friendly code is about maximizing hits.

---

## Why It Matters

### Sequential vs. Random Access

```swift
let size = 1_000_000
var array = [Int](repeating: 0, count: size)
var indices = Array(0..<size).shuffled()  // Random order

// Sequential access: ~10ms
for i in 0..<size {
    array[i] += 1
}

// Random access: ~100ms (10x slower!)
for i in indices {
    array[i] += 1
}
```

Same number of operations, 10x difference. The sequential version triggers prefetching; the random version thrashes the cache.

---

## The Technique

### Pattern 1: Sequential Access

Access memory in order whenever possible:

```swift
// ✅ Sequential: Cache-friendly
for i in 0..<n {
    result += data[i]
}

// ❌ Strided: Less cache-friendly
for i in stride(from: 0, to: n, by: 64) {
    result += data[i]
}

// ❌ Random: Cache-unfriendly
for i in randomIndices {
    result += data[i]
}
```

### Pattern 2: Contiguous Storage

Keep related data together:

```swift
// ❌ Array of objects: Scattered on heap
class Point {
    var x: Float
    var y: Float
    var z: Float
}
var points: [Point] = ...  // Pointers are contiguous; data is scattered

// ✅ Array of structs: Contiguous
struct Point {
    var x: Float
    var y: Float
    var z: Float
}
var points: [Point] = ...  // Data is contiguous

// ✅✅ Struct of arrays (SoA): Best for SIMD
struct Points {
    var xs: [Float]
    var ys: [Float]
    var zs: [Float]
}
```

Memory layout comparison:

```
Array of Structs (AoS):
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ x0 y0 z0    │ x1 y1 z1    │ x2 y2 z2    │ x3 y3 z3    │
└─────────────┴─────────────┴─────────────┴─────────────┘

Struct of Arrays (SoA):
┌─────────────────────────────────────────────────────────┐
│ x0   x1   x2   x3   x4   x5   ...  (all x values)       │
├─────────────────────────────────────────────────────────┤
│ y0   y1   y2   y3   y4   y5   ...  (all y values)       │
├─────────────────────────────────────────────────────────┤
│ z0   z1   z2   z3   z4   z5   ...  (all z values)       │
└─────────────────────────────────────────────────────────┘
```

SoA is better when you process all x values, then all y values, etc. AoS is better when you process all components of each point together.

### Pattern 3: Loop Tiling (Blocking)

For operations on large data, process in cache-sized blocks:

```swift
// ❌ Processes entire arrays at once (may exceed cache)
func matrixMultiply(_ a: [[Float]], _ b: [[Float]]) -> [[Float]] {
    for i in 0..<n {
        for j in 0..<n {
            for k in 0..<n {
                c[i][j] += a[i][k] * b[k][j]
            }
        }
    }
}

// ✅ Processes in cache-sized tiles
let tileSize = 64  // Fits in L1 cache
for ii in stride(from: 0, to: n, by: tileSize) {
    for jj in stride(from: 0, to: n, by: tileSize) {
        for kk in stride(from: 0, to: n, by: tileSize) {
            // Process tile
            for i in ii..<min(ii + tileSize, n) {
                for j in jj..<min(jj + tileSize, n) {
                    for k in kk..<min(kk + tileSize, n) {
                        c[i][j] += a[i][k] * b[k][j]
                    }
                }
            }
        }
    }
}
```

### Pattern 4: Avoid Pointer Chasing

```swift
// ❌ Linked list: Each node is a cache miss
class Node {
    var value: Int
    var next: Node?
}

// ✅ Array: Contiguous, prefetchable
var values: [Int]
```

### Pattern 5: Pack Hot Data Together

```swift
// ❌ Cold data mixed with hot data
struct Entity {
    var id: UUID           // Hot
    var metadata: String   // Cold (rarely accessed)
    var position: SIMD3<Float>  // Hot
    var createdAt: Date    // Cold
    var velocity: SIMD3<Float>  // Hot
}

// ✅ Hot data packed together
struct EntityHot {
    var id: UUID
    var position: SIMD3<Float>
    var velocity: SIMD3<Float>
}

struct EntityCold {
    var id: UUID  // Foreign key
    var metadata: String
    var createdAt: Date
}
```

When iterating over entities for physics, `EntityHot` fits more elements per cache line.

---

## In VectorCore

### ContiguousArray Over Array

**📍 See:** `Sources/VectorCore/Vectors/Vector512Optimized.swift:29`

```swift
public var storage: ContiguousArray<SIMD4<Float>>
```

`ContiguousArray` guarantees:
- No bridging to NSArray
- Contiguous element storage
- Pointer stability within `withUnsafe...` closures

Regular `Array` might bridge to Objective-C, causing copies and indirection.

### SIMD4 Storage Layout

Each `SIMD4<Float>` is 16 bytes, perfectly aligned:

```
Vector512Optimized storage:
┌────────────────┬────────────────┬────────────────┬─────┬────────────────┐
│   SIMD4[0]     │   SIMD4[1]     │   SIMD4[2]     │ ... │  SIMD4[127]    │
│ f0  f1  f2  f3 │ f4  f5  f6  f7 │ f8  f9 f10 f11 │     │ f508..f511     │
│    16 bytes    │    16 bytes    │    16 bytes    │     │    16 bytes    │
└────────────────┴────────────────┴────────────────┴─────┴────────────────┘
│←─────────────────────── 2048 bytes ───────────────────────────────────→│
│←───── Fits in L1 cache (typically 64KB) ──────→│
```

A 512-dimensional vector (2KB) fits entirely in L1 cache. Operations on it don't cause cache misses once loaded.

### Sequential Processing in Kernels

**📍 See:** `Sources/VectorCore/Operations/Kernels/DotKernels.swift:30-54`

```swift
for i in stride(from: 0, to: laneCount, by: 16) {
    acc0 += storageA[i+0] * storageB[i+0]
    acc1 += storageA[i+1] * storageB[i+1]
    // ... sequential access pattern
}
```

Accessing `i+0, i+1, i+2...` sequentially triggers hardware prefetching. The next cache line is loaded before we need it.

---

## Measuring Cache Performance

### Using Instruments

The "Counters" instrument can measure:
- L1/L2/L3 cache hits and misses
- Memory bandwidth
- TLB misses

### Quick Heuristics

If your code is slower than expected for the number of operations:
1. **Check memory access pattern.** Is it sequential?
2. **Check data size.** Does working set fit in cache?
3. **Check layout.** Is hot data packed together?
4. **Check indirection.** Are you chasing pointers?

---

## Key Takeaways

1. **Cache misses are 100x slower than hits.** Optimize for cache, not instruction count.

2. **Sequential access enables prefetching.** Access arrays in order.

3. **Keep data contiguous.** Structs in arrays, not pointers to objects.

4. **Pack hot data together.** Separate frequently-accessed fields from cold data.

5. **VectorCore's 2KB vectors fit in L1.** Operations are cache-optimal.

---

## Next Up

Cache-friendly code runs fast on one core. For more speed, we need multiple cores:

**[→ Parallelization Strategy](./03-Parallelization-Strategy.md)**
