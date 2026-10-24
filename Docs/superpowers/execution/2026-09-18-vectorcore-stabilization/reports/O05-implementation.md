# O05 implementation — frozen v2, validation complete

Base: `94b73440eae64e2d97726138f2ee9752708e5736`. Source archive: canonical `.stabilization-recovery/2026-09-24/workspaces/O05`. Full thirteen-file patch and base/final code plus owned-doc inventory are `O05-frozen-v2.diff` and `O05-frozen-v2-manifest.json`; source backup is `.stabilization-recovery/2026-09-24/O05/frozen-v2/source`. No Package, index, commit, sibling, arithmetic, scheduler or SoA production edits.

## Disposition and migration

Removed the unused internal StorageSlice declaration and ManagedStorage/DimensionStorage slice helpers, plus the disabled MoveOnlyStorage experimental block. ManagedStorage core bytes and accepted O03 mutation paths remain unchanged. No move-only gate was enabled, no replacement transfer API was introduced and no expired-pointer dereference was executed. This closes dormant source-established routes by removal, not by demonstrating a new transfer implementation or reproducing supported-use UAF.

The three fixed FP16 wrappers now expose synchronous `withUnsafeFP16BufferPointer<R>` callbacks, forwarding inside immutable ContiguousArray scoped access. Return values and identical thrown errors propagate. The old fp16Storage getters are removed as an authorized source break. This is explicitly unsafe: buffer/base pointers must not escape through returns, captures, tasks, foreign callbacks or stored pointers. Generic R does not enforce nonescape. Documentation migrates retained/asynchronous data to an owned Array copied inside the callback; it does not introduce an async borrowing guarantee.

SoAFP16.storage remains an owned ContiguousArray value. Public DynamicVector.slice remains an owned-vector operation. Direct FP16 kernels retain their existing nested scoped borrows. The frozen FP32 SoA layout, explicit allocation transfer, logical versus allocated byte distinction and caller synchronization obligations are unchanged. There is no new transfer, GPU, concurrency, allocation-performance or numerical-quality claim.

The two permitted docs distinguish FP16 callback borrows from persistent SoA/page allocations. Their adjacent AlignedMemory text now states accepted O04 behavior: nonnegative counts with representable checked byte product, zero count backed by one uninitialized element, invalid count throws, and power-of-two/minimum/type alignment checked by preconditions. It cites existing UnsafeMemoryContractTests without promising validation of later pointer arithmetic or lifetime.

## Caller inventory

Before deletion, Core Sources/Tests/Benchmarks/Package contained only the dormant slice declaration/forwarders, disabled experimental block, three getters and the distinct public DynamicVector slice test. Read-only VectorAccelerate Sources/Tests/Package, EmbedKit Sources and VectorIndex Sources searches found no removed-symbol or getter callers. Exact commands, statuses and complete outputs are in `O05-caller-inventory-baseline.json`. This is current source inventory, not downstream build/runtime validation. Existing direct-kernel call and suite inventory is in `O05-kernel-and-suite-inventory.json`. A read-only search of the exact base commit also found no removed-symbol or gate references in Scripts, .github or Package.swift (`O05-build-script-inventory.json`).

## Compiler RED and public fixtures

Before production edits, the unchanged Release library was built with six jobs. Each width was checked independently from non-testable Swift 6 source:

| Width | Legacy getter typecheck | Desired scoped API typecheck |
|---|---:|---:|
| 512 | 0 | 1, intended missing member |
| 768 | 0 | 1, intended missing member |
| 1536 | 0 | 1, intended missing member |

No legacy fixture dereferences its pointer. The baseline checks are compile-time migration RED, not memory-corruption reproduction. Baseline module/object/fixture bytes, commands and diagnostics were copied into an immutable snapshot and canonical O05/baseline before source edits. Typechecks create no executable; that is explicit, not a missing executable hash.

The final driver expects all three legacy accesses to fail at the missing fp16Storage member, all three new API sources to typecheck, and each new source to link and execute with a width-specific completion marker. Each positive source includes a normal UInt64 checksum result, a throwing temporary-owner callback with sentinel identity, and an owned Array that survives owner removal. The driver also checks the entire Sources tree for the removed declarations/member-access/gate families; unrelated local variables named fp16Storage remain permitted.

Each child installs signal cleanup handlers before launch, captures output to a file, uses a 15-second execution deadline and at most two seconds of owned-process-group cleanup. The immutable snapshot records full argv, expected/actual exits, intended diagnostics, markers, PIDs, fixture and module/object hashes, and final executable hashes. No empty filter is credited.

## Runtime coverage

Seven new Swift Testing functions all execute real bodies:

- Three width-specific bit-pattern matrices (512/768/1536) check every UInt16, dimension/endpoints, single callback invocation, exact checksum, nested independent borrows, temporary construction, identical sentinel propagation and retained Array copies after optional-owner reassignment/removal. Patterns include positive/negative zero, subnormals, finite boundaries, infinities and NaN payload encodings.
- One direct-kernel matrix checks existing dot/Euclidean/cosine routes at all widths against independent Double scalar references for exactly representable ordinary inputs, plus existing zero-vector policies. It makes no kernel rewrite or expanded numerical claim.
- One valid external-style dimension-8/lanes-2 SoA conformer asserts exactly two SIMD4 storage lanes. N=0/1/5 in ordinary/page-aligned modes checks every lane/scalar, index formula, logical count, allocation rounding, base alignment and empty behavior. OS sysconf supplies the independent page size. No invalid unsafe conformer, transfer race or GPU use is executed.
- One public DynamicVector slice control spans inline/heap source sizes and verifies owned contents after source reassignment and destination mutation.
- One all-width FP16 SoA control retains the public ContiguousArray after dropping its owner and checks the full blocked layout, including padded lanes.

Existing SoALayout descriptor/parity suites, SoAPageAlignTests, UnifiedVectorBufferTests, DynamicVectorTests and MixedPrecisionPart1ValidationTests run alongside the new suite. The brief's SoALayoutContractTests filename contains differently named suites; the actual filter uses SoALayout. Existing transfer controls verify pointer identity and matching recipient deallocation; these remain CPU ownership tests only.

## Validation state

Focused Debug: exit 0, 63 tests across nine suites. Initial Debug setup compilation failed because the new non-testable test referred to internal PlatformConfiguration; it was corrected to the OS page-size query. That full failure log is retained separately; no existing test expectation changed. Production and Swift test/fixture bytes remain frozen from v1. Final Release library build: exit 0. All twelve public-consumer phases passed their exact expected outcomes, including three missing-getter diagnostics and three successful executable completion markers. The six Swift fixture sources match the baseline byte-for-byte; the original final-v1 driver also matched the baseline driver; all library source hashes match the frozen base/final manifest. Focused Release: exit 0, 63 tests across nine suites. ASan: exit 0, 23 valid tests across three suites, including the seven new O05 tests and existing SoA/Page allocation transfer controls; no sanitizer findings.

## Versioned scanner refinement and final evidence

V2 changes only the Python driver. The initial generic `var fp16Storage` declaration scan could reject an unrelated local variable. Root approved narrowing it to the actual removed public getter. Eight pure policy cases now preserve local let/var identifiers while rejecting the removed public getter, member access, slice declaration/helper and move-only declaration/gate. All eight pass, the Sources scan has no matches, and all twelve final consumer phases pass again. V1 sources and runtime/probe evidence remain retained. No Swift rebuild was needed for this scanner-only delta: all production, test and six Swift fixture hashes are unchanged. Final v2 used byte-identical library artifacts to final v1. See O05-consumer-comparison-v2.json.

Exact commands, exits and full log hashes are in O05-test-command-status.json and O05-build-consumer-command-status.json. Public snapshots include all source/module/object/fixture/executable hashes and child logs. O05-test-binary-association.json distinguishes the SwiftPM helper from the loaded Debug/Release/ASan bundles and records their module/object hashes and v2 mapping. The final 286-path code/owned-doc inventory matches frozen v2. Worker evidence is canonically backed up outside /tmp, indexed separately from root and reviewer reports. Independent review and root adjudication remain separate from these implementation results.
