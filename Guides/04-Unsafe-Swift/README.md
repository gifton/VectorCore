# Chapter 4: Unsafe Swift

> **With great power comes great responsibility. And great performance.**

Swift is designed to be safe by default—bounds checking, type safety, memory management. But sometimes you need to drop the safety net for performance. This chapter teaches you to work with Swift's unsafe APIs correctly and safely.

---

## What You'll Learn

| Guide | Time | What You'll Learn |
|-------|------|-------------------|
| [1. Pointer Primer](./01-Pointer-Primer.md) | 12 min | UnsafePointer, memory binding, lifetime |
| [2. Buffer Pointers](./02-Buffer-Pointers.md) | 10 min | Working with contiguous memory regions |
| [3. C Interop Patterns](./03-C-Interop-Patterns.md) | 10 min | posix_memalign, free, bridging |

---

## The Big Picture

Swift's "unsafe" types give you direct memory access:

```
Safe Swift World                    Unsafe Swift World
──────────────────                  ──────────────────
Array<Float>                        UnsafeBufferPointer<Float>
  ├── Bounds checking               ├── No bounds checking
  ├── Copy-on-write                 ├── Direct memory access
  └── Reference counting            └── Manual lifetime

                     ↓
              withUnsafeBufferPointer { ptr in
                  // Cross the boundary safely
              }
                     ↓

                    C / System Calls
                    ──────────────────
                    posix_memalign
                    memcpy
                    SIMD loads
```

Unsafe doesn't mean dangerous—it means the compiler won't check your work. Done correctly, it's just as reliable as safe code, but faster.

---

## Why Swift Developers Should Learn This

You might never write unsafe code directly. But you'll use libraries that do:

1. **VectorCore** uses unsafe pointers throughout for SIMD access
2. **Foundation** uses unsafe code for String and Data
3. **Swift collections** use unsafe code internally
4. **Any C library** requires unsafe bridging

Understanding unsafe code helps you:
- Read library implementations
- Debug crashes in production
- Know when unsafe code is worth the tradeoff
- Write correct bridging code

---

## VectorCore Connections

| VectorCore Feature | Unsafe Pattern |
|-------------------|----------------|
| `withUnsafeBufferPointer` | Safe entry into unsafe territory |
| `bindMemory(to:)` | Reinterpreting SIMD4 as Float |
| `posix_memalign` + `free()` | C memory allocation |
| `ContiguousArray` | Guaranteed pointer stability |

---

## Safety Guidelines

Before diving in, some ground rules:

1. **Prefer safe code.** Use unsafe only when there's a measurable benefit.

2. **Minimize unsafe scope.** Use `withUnsafe...` closures rather than storing pointers.

3. **Never escape pointers.** The pointer is only valid within its closure.

4. **Match allocators to deallocators.** Swift allocation → Swift deallocation. C allocation → C deallocation.

5. **Test with sanitizers.** Use Address Sanitizer to catch memory bugs.

---

## Start Here

**[→ Pointer Primer](./01-Pointer-Primer.md)**

---

*Chapter 4 of 6 • [← Numerical Computing](../03-Numerical-Computing/README.md) | [Next: Performance Patterns →](../05-Performance-Patterns/README.md)*
