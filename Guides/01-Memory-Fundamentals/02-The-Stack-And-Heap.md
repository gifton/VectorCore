# The Stack and Heap

> **Reading time:** 12 minutes
> **Prerequisites:** [How Swift Stores Data](./01-How-Swift-Stores-Data.md)

---

## The Concept

Your program has two main regions for storing data at runtime:

**The Stack**: Fast, automatic, limited
- Like a stack of plates—you can only add/remove from the top
- Allocation is just moving a pointer
- Memory is automatically reclaimed when a function returns
- Limited size (typically 1-8 MB per thread)

**The Heap**: Flexible, manual*, unlimited
- Like a warehouse—you can allocate anywhere, any size
- Allocation requires finding free space, bookkeeping
- Memory must be explicitly freed (ARC does this for you)
- Limited only by available RAM

*Swift's ARC automates heap management, but the runtime cost is still there.

---

## Why It Matters

### Stack Allocation: Essentially Free

When you call a function, the stack pointer moves:

```swift
func doSomething() {
    var x: Float = 1.0  // Stack allocation: ~0 cycles
    var y: Float = 2.0  // Stack allocation: ~0 cycles
    // ... use x and y ...
}  // Deallocation: ~0 cycles (stack pointer moves back)
```

```
Before call:        During call:        After return:

     Stack              Stack               Stack
┌────────────┐    ┌────────────┐      ┌────────────┐
│            │    │            │      │            │
│            │    │            │      │            │
│            │    ├────────────┤      │            │
│            │    │  y: 2.0    │      │            │
├────────────┤    ├────────────┤      ├────────────┤
│  (caller)  │    │  x: 1.0    │      │  (caller)  │
│            │    ├────────────┤      │            │
│            │    │ return addr│      │            │
│            │    ├────────────┤      │            │
└────────────┘    │  (caller)  │      └────────────┘
      ↑           └────────────┘            ↑
      SP                ↑                   SP
                        SP
```

The entire allocation is just `SP -= 8` (or whatever size). No bookkeeping, no searching for free space, no locks.

### Heap Allocation: Surprisingly Expensive

```swift
func doSomethingElse() {
    let array = [Float](repeating: 0, count: 100)  // Heap allocation
    // ... use array ...
}  // ARC decrements refcount, possibly frees memory
```

A single heap allocation involves:
1. **Lock acquisition** (in multi-threaded apps)
2. **Free list traversal** to find suitable space
3. **Bookkeeping updates** (mark memory as used)
4. **Possibly: new page mapping** from the OS

A typical `malloc` call takes 50-100+ nanoseconds. Compare that to stack allocation's essentially 0 nanoseconds.

### The Real-World Impact

Consider processing a million vectors:

```swift
// Version A: Heap allocation in loop
func processVectors(_ vectors: [Vector512Optimized]) -> [Float] {
    return vectors.map { vector in
        // Array created here, on heap, for each vector
        let temp = vector.toArray()
        return temp.reduce(0, +)
    }
}

// Version B: No heap allocation in loop
func processVectorsFast(_ vectors: [Vector512Optimized]) -> [Float] {
    return vectors.map { vector in
        // Direct access to SIMD storage, no allocation
        vector.sum
    }
}
```

Version A allocates 1 million temporary arrays. At ~100ns each, that's **100 milliseconds** just in allocation overhead—before doing any actual work.

---

## The Technique

### Knowing When Heap Allocation Happens

Heap allocation occurs when:

1. **Classes are instantiated**
```swift
let obj = MyClass()  // Heap allocation
```

2. **Arrays grow beyond their capacity**
```swift
var arr = [Int]()
arr.append(1)  // May trigger heap allocation
```

3. **Closures capture values**
```swift
var counter = 0
let increment = { counter += 1 }  // Heap allocation for captured context
```

4. **Strings are created or modified**
```swift
let str = "Hello, World!"  // Heap allocation (usually)
```

5. **Large value types that don't fit in registers** (compiler decides)

### Avoiding Unnecessary Allocations

**Strategy 1: Pre-allocate with capacity**
```swift
// Bad: Multiple reallocations as array grows
var results = [Float]()
for i in 0..<1000 {
    results.append(Float(i))
}

// Good: Single allocation
var results = [Float]()
results.reserveCapacity(1000)
for i in 0..<1000 {
    results.append(Float(i))
}
```

**Strategy 2: Use in-place operations**
```swift
// Bad: Creates new vector
let scaled = vector * 2.0

// Good: Modifies existing storage
var vector = vector
vector.inPlaceMultiply(2.0)
```

**Strategy 3: Reuse buffers**
```swift
// Bad: Allocate buffer for each call
func computeDistances(query: Vector512Optimized, candidates: [Vector512Optimized]) -> [Float] {
    return candidates.map { query.euclideanDistance(to: $0) }  // Array allocated
}

// Good: Caller provides buffer
func computeDistances(query: Vector512Optimized, candidates: [Vector512Optimized],
                      into buffer: inout [Float]) {
    for (i, candidate) in candidates.enumerated() {
        buffer[i] = query.euclideanDistance(to: candidate)
    }
}
```

---

## Stack Size Limits

The stack is fast but limited. Typical sizes:

| Platform | Main Thread | Worker Threads |
|----------|-------------|----------------|
| macOS    | 8 MB        | 512 KB        |
| iOS      | 1 MB        | 512 KB        |
| Linux    | 8 MB        | 2-8 MB        |

This means you can't put huge arrays on the stack:

```swift
func badIdea() {
    var hugeArray = (Float, Float, Float, ...)  // 10 MB of floats
    // BOOM: Stack overflow
}
```

For large data, you *need* heap allocation—but you can still be smart about it.

---

## In VectorCore

VectorCore balances stack efficiency with heap necessity:

**Vector types are value types (stack-friendly):**
```swift
public struct Vector512Optimized: Sendable {
    public var storage: ContiguousArray<SIMD4<Float>>
    // ...
}
```

But `ContiguousArray` stores its elements on the heap. The struct itself (with its metadata) lives on the stack; the actual data lives on the heap.

```
Stack:                          Heap:
┌──────────────────────┐        ┌────────────────────────────────┐
│ Vector512Optimized   │        │ ContiguousArray backing store  │
│ ├─ storage.ptr ──────────────►│ SIMD4[0] SIMD4[1] ... SIMD4[127]│
│ └─ storage.count     │        └────────────────────────────────┘
└──────────────────────┘
     ~16 bytes                           2048 bytes
```

This gives us:
- **Copy-on-write semantics**: Cheap to pass around until modified
- **No size limit**: 2KB per vector is fine for heap
- **Value semantics**: Behaves like a value type for reasoning about code

**📍 See:** `Sources/VectorCore/Vectors/Vector512Optimized.swift:38-46`

```swift
/// Initialize with zeros
@inlinable
public init() {
    storage = ContiguousArray(repeating: SIMD4<Float>(), count: 128)
}
```

One allocation when created. The vector can then be used in thousands of operations without additional allocations.

---

## Buffer Pooling

For operations that need temporary buffers, VectorCore uses **buffer pooling**—pre-allocating buffers that can be reused:

```
Without pooling:                With pooling:

Operation 1:                    First use:
  alloc → use → free            alloc → use → return to pool
                                                    │
Operation 2:                    Second use:         │
  alloc → use → free              ←─────────────────┘
                                       use → return to pool
Operation 3:                                        │
  alloc → use → free            Third use:          │
                                  ←─────────────────┘
                                       use → return to pool

Cost: 3 allocs + 3 frees        Cost: 1 alloc + 0 frees
```

This amortizes allocation cost across many operations.

---

## Copy-on-Write (CoW)

Swift's collections (and VectorCore's vectors) use copy-on-write:

```swift
var a = Vector512Optimized(repeating: 1.0)
var b = a  // No copy! Just shares reference to storage

print(a[0])  // 1.0
print(b[0])  // 1.0, reading from same memory

b[0] = 2.0   // NOW it copies, because b needs unique storage

print(a[0])  // 1.0 (unchanged)
print(b[0])  // 2.0 (b has its own copy now)
```

```
After b = a:                    After b[0] = 2.0:

Stack:                          Stack:
┌──────┐ ┌──────┐               ┌──────┐ ┌──────┐
│  a   │ │  b   │               │  a   │ │  b   │
│  │   │ │  │   │               │  │   │ │  │   │
└──┼───┘ └──┼───┘               └──┼───┘ └──┼───┘
   │        │                      │        │
   └────┬───┘                      │        │
        │                          ▼        ▼
        ▼                       ┌─────┐  ┌─────┐
     ┌─────┐                    │1.0..│  │2.0..│
     │1.0..│                    └─────┘  └─────┘
     └─────┘
   refcount: 2                  refcount: 1 each
```

CoW lets you pass vectors around freely without copying, paying the copy cost only when you actually mutate.

---

## Key Takeaways

1. **Stack allocation is nearly free; heap allocation costs ~100ns.** In hot loops, this adds up fast.

2. **Value types aren't always on the stack.** Swift decides based on size and usage. Large value types often have heap-allocated storage.

3. **Pre-allocate when you know the size.** Use `reserveCapacity` for arrays, reuse buffers when possible.

4. **Copy-on-Write gives you the best of both worlds.** Cheap copies until you actually need to modify.

5. **VectorCore uses struct wrappers around heap storage.** The struct provides value semantics; the heap provides space for 2KB of data.

---

## Next Up

We've covered *where* data lives and *how much* allocation costs. Now let's explore *how* the CPU accesses that memory efficiently:

**[→ Why Alignment Matters](./03-Why-Alignment-Matters.md)**
