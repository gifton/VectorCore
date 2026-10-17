# O04 independent review

Status: ACCEPT revision2 for the scoped O04 contract; no blocking findings. Final frozen source and evidence verification completed on 2026-09-22. Reviewed archive `/private/tmp/vectorcore-stabilization-O04-implementation`, base `cd9bbe392901d1467318c00cba7e96dd973ea1a9`, against the brief and diagnostic addendum. No implementation edits or builds were performed by this reviewer.

## Source and test assessment

- Pointer normalization changes all eleven SIMD loads across the maximum, square-sum and output passes to unaligned loads. Full groups, four-vector unrolling, residual vectors and scalar tails stay within the caller's initialized capacity. Trivial raw stores retain the Float owner's binding. Dimension zero returns before access; negative dimension now has an explicit fatal diagnostic. The moderate-value Double oracle covers four Float-aligned offsets and dimensions 0–5, 15–17 and 31–33 with exact surrounding sentinels. This does not establish arbitrary-range numerical correctness; N01 retains that work.
- The four Float-array constructors build SIMD values from scalar lanes. The twelve reverse borrows remain scoped inside owning ContiguousArray borrows, with capacity in Float elements and no simultaneous use of the original SIMD pointer. The changed Vector384 sites match that pattern. Throwing exits, subsequent SIMD access, retained vector independence and source-array reuse are exercised at every width. S01 count/backing guards are unchanged. This is a source-established binding repair, not evidence of a reproduced optimizer miscompile.
- `alignedCopy` uniformly uses `update(from:count:)` on initialized, naturally aligned, nonoverlapping ranges. This releases replaced references and retains copied references. The nontrivial Payload and three-reference tests independently offset source/destination and observe exact death counts; the zero-count control initializes both pointers. There are no production call sites to migrate in this tree. External users with uninitialized destinations must use `initialize(from:count:)` themselves. The positive alignment hint no longer changes ownership semantics.
- `allocateAligned<T>` checks count arithmetic before allocation, requests at least one element for logical zero, validates type alignment, binds fresh storage, and returns it uninitialized. POSIX allocations retain their `free` pairing. AlignedDynamic constructors share one validated path; primary failure requests the same checked byte size and alignment, binds Float, and records raw Swift deallocation. Array initializers update initialized zero values. The injection seam is per-call/internal; its contract requires Float-bound storage suitable for free. The tests establish forced-primary-failure alignment and destruction, not successful recovery from actual exhaustion.
- The revised subprocess tests require the intended per-case diagnostic for the new pointer, copy and storage boundaries in both configurations. Existing common-allocator alignment preconditions require named Debug diagnostics and Release signal rejection, as the brief permits. Each child has a deadline and no accepted-input marker may appear.

The scoped SIMD4<Float> → Float interpretation follows the homogeneous aggregate and destination-element capacity rules in [SE-0333](https://github.com/swiftlang/swift-evolution/blob/main/proposals/0333-with-memory-rebound.md). Float needs no stronger alignment than its SIMD owner, and the runtime layout checks restrict the tested model. [SE-0349](https://github.com/swiftlang/swift-evolution/blob/main/proposals/0349-unaligned-loads-and-stores.md) supports the trivial unaligned raw loads/stores used here. Neither establishes arbitrary generic aggregate layout or permits pointers to escape unsafe borrow lifetimes.

## Caller audit and retained limitations

Read allocator users in PageAlignedBuffer/UnifiedVectorBuffer, page-aligned SoA, TieredStorage AlignedBuffer and MemoryPool, plus the UInt8 test buffer provider and alignment/pointer tests. The successful allocation paths are compatible with bound, uninitialized storage and their existing initialization strategy. Existing typed contents/layout and SoA ownership are unchanged.

Separate pre-existing concern: `Sources/VectorCore/Utilities/MemoryPool.swift:203` uses `assumingMemoryBound` on fresh raw fallback memory. O04 does not repair that fallback or establish broader pool ownership correctness. This is recorded for separate adjudication, not an expansion request. Existing concurrency, numerical-range and performance claims remain outside acceptance.

## Artifact verification

Independently verified all nine source/test SHA-256 values against the implementation archive and all eighteen evidence hashes against the published files before this review update. Regenerated the complete nine-file unified diff from base `cd9bbe392901d1467318c00cba7e96dd973ea1a9`; it exactly matches `O04-full.diff` (SHA-256 `db4174d1e7e69e038cf3e82bbbe4a357446319046790c0d46213b5430c62f437`). The final source manifest hash is `8667092891d58648d5796230487c2c3d1adb1e194f0dba04faa7080f6993c79c`. Compared all base Sources/Tests/Package files: exactly the eight authorized existing source files changed; the new test is included in the full diff/manifest. Reviewed the revision1-to-final diff: only explicit fatal diagnostics and corresponding per-case assertions changed.

Inspected retained RED logs: `initializedCopyLifetimes` observes the old destination reference surviving replacement, and Debug `pointerOracle` traps on the misaligned raw load. No Release baseline failure or runtime fallback misalignment is inferred from those logs.

Inspected final test logs and their selected test names, results and failure markers:

| Evidence | Observed result |
| --- | --- |
| `O04-final-debug.log` | 55 tests / 8 suites passed in 9.072s; bounded invalid-input diagnostic test passed |
| `O04-final-release.log` | 55 tests / 8 suites passed in 8.566s; bounded invalid-input diagnostic test passed |
| `O04-final-asan.log` | All eight named valid-path tests / one suite passed in 0.015s; no sanitizer diagnostics |

Execution owner records exit0 for each command. This reviewer inspected those frozen outputs and did not duplicate builds. The test source requires named new-boundary diagnostics in both configurations, so the successful child test covers the root addendum. ASan excludes deliberate crash children and is not proof of binding legality; source analysis supplies that part of the review. Build warnings remain visible, including weak-local mutability suggestions. No warning or sanitizer suppression was introduced.

`O04-evidence-sha256.json` included the provisional review hash; root must refresh that one entry after this final review edit. Its other seventeen entries were independently verified unchanged. Root adjudication/integration remains separate from this review acceptance.
