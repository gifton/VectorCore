# What Is SIMD?

> **Reading time:** 8 minutes
> **Prerequisites:** [Chapter 1: Memory Fundamentals](../01-Memory-Fundamentals/README.md)

---

## The Concept

SIMD stands for **Single Instruction, Multiple Data**. It's a CPU feature that processes multiple values with one instruction.

Think of it like this:

**Without SIMD (Scalar):**
You're a cashier with one register. To process 4 customers, you ring up each one separately. 4 customers = 4 transactions.

**With SIMD (Vectorized):**
You have a magical register that can ring up 4 customers simultaneously. Same work, one-quarter the time.

In hardware terms:

```
Scalar Registers:              SIMD Register (128-bit):
┌─────────┐                    ┌─────────────────────────────────────┐
│ 32 bits │  ← one Float       │ 32b │ 32b │ 32b │ 32b │  ← 4 Floats │
└─────────┘                    │ [0] │ [1] │ [2] │ [3] │             │
                               └─────────────────────────────────────┘
                                           "lanes"
```

Modern CPUs have wide SIMD registers:
- **128-bit** (NEON on ARM, SSE on Intel): 4 floats
- **256-bit** (AVX on Intel): 8 floats
- **512-bit** (AVX-512 on Intel Xeon): 16 floats

Apple Silicon (M1/M2/M3) uses 128-bit NEON registers, processing 4 floats at once.

---

## Why It Matters

### The Dot Product Example

Consider computing the dot product of two 512-dimensional vectors:

```
dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + ... + a[511]*b[511]
```

**Scalar approach:** 512 multiplications, 511 additions = 1023 operations

**SIMD approach with SIMD4:**
- Load `a[0..3]` and `b[0..3]` into SIMD registers
- Multiply: one instruction produces 4 products
- Repeat for all 128 groups of 4
- Sum the results

Instead of 512 multiply instructions, we need 128. That's **4x fewer instructions**.

### Real Performance Numbers

From VectorCore benchmarks on Apple Silicon:

| Operation | Scalar (estimated) | SIMD (measured) | Speedup |
|-----------|-------------------|-----------------|---------|
| 512-dim dot product | ~400ns | ~100ns | 4x |
| 512-dim Euclidean distance | ~500ns | ~120ns | 4x |
| 512-dim normalization | ~600ns | ~150ns | 4x |

The speedup is close to the theoretical 4x from using SIMD4.

---

## The Technique

### Anatomy of a SIMD Operation

Let's trace through a SIMD multiply-add:

```swift
import simd

let a = SIMD4<Float>(1.0, 2.0, 3.0, 4.0)
let b = SIMD4<Float>(5.0, 6.0, 7.0, 8.0)

let product = a * b  // One instruction!
// product = SIMD4<Float>(5.0, 12.0, 21.0, 32.0)
```

What happens at the CPU level:

```
Step 1: Load a into SIMD register
┌───────────────────────────────────────┐
│  1.0  │  2.0  │  3.0  │  4.0  │  ← Register V0
└───────────────────────────────────────┘

Step 2: Load b into SIMD register
┌───────────────────────────────────────┐
│  5.0  │  6.0  │  7.0  │  8.0  │  ← Register V1
└───────────────────────────────────────┘

Step 3: FMUL V2, V0, V1  (one instruction!)
┌───────────────────────────────────────┐
│  5.0  │ 12.0  │ 21.0  │ 32.0  │  ← Register V2
└───────────────────────────────────────┘
    ↑       ↑       ↑       ↑
   1×5     2×6     3×7     4×8

   All 4 multiplications happen simultaneously
```

### Horizontal Operations

Sometimes you need to combine lanes. For example, to sum all elements:

```swift
let v = SIMD4<Float>(5.0, 12.0, 21.0, 32.0)
let sum = v.sum()  // 70.0
```

This is called a **horizontal** operation because it works across lanes rather than between vectors. Horizontal operations are slower because lanes need to communicate:

```
Vertical (fast):          Horizontal (slower):
┌─────┐   ┌─────┐         ┌───────────────────┐
│ a0  │   │ b0  │         │ v0 │ v1 │ v2 │ v3 │
│ a1  │ × │ b1  │         └───┬───┬───┬───┬───┘
│ a2  │   │ b2  │             │   │   │   │
│ a3  │   │ b3  │             └─+─┴─+─┴─+─┘
└──┬──┘   └──┬──┘                   │
   │         │               ┌──────┴──────┐
   └────×────┘               │    sum      │
        │                    └─────────────┘
   ┌────┴────┐
   │ product │
   └─────────┘

All lanes                 Lanes must
independent               communicate
```

VectorCore minimizes horizontal operations by delaying reductions until the end.

---

## SIMD Vocabulary

| Term | Meaning |
|------|---------|
| **Lane** | One slot in a SIMD register (e.g., SIMD4 has 4 lanes) |
| **Vector width** | How many elements a SIMD type holds |
| **Vertical operation** | Same operation applied to each lane independently |
| **Horizontal operation** | Combines values across lanes (e.g., sum, max) |
| **Vectorization** | Converting scalar code to use SIMD operations |
| **Auto-vectorization** | Compiler automatically converts loops to SIMD |

---

## When SIMD Works Well

SIMD excels when:

1. **Same operation on many elements:** `for i in 0..<n { c[i] = a[i] + b[i] }`

2. **Data is contiguous:** Elements are adjacent in memory (arrays, not linked lists)

3. **No data dependencies between iterations:** Each element can be computed independently

4. **Aligned data:** Memory addresses are multiples of the SIMD width

## When SIMD Struggles

SIMD is less effective when:

1. **Branching per element:** `if a[i] > 0 { ... } else { ... }`

2. **Non-contiguous access:** `a[indices[i]]` (gather operations are slow)

3. **Dependent computations:** `a[i] = a[i-1] + b[i]` (can't parallelize)

4. **Very short data:** Processing 3 elements with SIMD4 wastes a lane

---

## In VectorCore

VectorCore stores vectors as `ContiguousArray<SIMD4<Float>>`:

**📍 See:** `Sources/VectorCore/Vectors/Vector512Optimized.swift:29`

```swift
/// Internal storage as SIMD4 chunks for optimal performance
public var storage: ContiguousArray<SIMD4<Float>>
```

A 512-dimensional vector is stored as 128 SIMD4 values:

```
Vector512Optimized.storage:
┌─────────────┬─────────────┬─────────────┬─────┬──────────────┐
│ SIMD4[0]    │ SIMD4[1]    │ SIMD4[2]    │ ... │ SIMD4[127]   │
│ f0,f1,f2,f3 │ f4,f5,f6,f7 │ f8,f9,f10,f11│     │f508..f511    │
└─────────────┴─────────────┴─────────────┴─────┴──────────────┘
     16B           16B           16B                  16B

Total: 128 × 16 = 2048 bytes = 512 floats
```

This layout means:
- Every operation can use SIMD4 directly
- No conversion or repacking needed
- Memory access is perfectly aligned for SIMD

---

## Key Takeaways

1. **SIMD processes multiple values with one instruction.** SIMD4 handles 4 floats at once, potentially 4x faster.

2. **Modern CPUs all have SIMD.** ARM NEON (Apple Silicon), Intel SSE/AVX—it's everywhere.

3. **Vertical operations are fast; horizontal operations are slower.** Minimize cross-lane communication.

4. **Data layout matters.** SIMD works best with contiguous, aligned data.

5. **VectorCore uses `SIMD4<Float>` as its fundamental unit.** The entire library is designed around 4-wide operations.

---

## Next Up

Now that you understand what SIMD is, let's see how to use it in Swift:

**[→ SIMD in Swift](./02-SIMD-In-Swift.md)**
