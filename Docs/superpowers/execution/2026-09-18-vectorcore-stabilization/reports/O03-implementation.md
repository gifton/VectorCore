# O03 implementation — frozen v1, validation complete

Base: `6d47ec508d55ef7e863ca3b9d51ac0d2f5cca10b`. Exclusive archive: `/private/tmp/vectorcore-stabilization-O03-implementation`. The complete eight-file patch and full base/final code inventory are `O03-frozen-v1.diff` and `O03-frozen-v1-manifest.json`. Source backup is in canonical `.stabilization-recovery/2026-09-23/O03/frozen-v1/source`; original library modules, objects, linked probes, logs and source copies are retained separately under `O03/baseline`. No index, commit, Package, provider implementation, UnifiedStorage or sibling edits.

## Change and ownership argument

HybridStorage removes its redundant outer copy. The heap enum payload is replaced with a nonallocating inline placeholder before delegating scalar mutation, mutable borrow or the uniqueness diagnostic to ManagedStorage. ManagedStorage remains the sole heap COW boundary. Throwing borrows restore the extracted value with `defer`, preserving partial writes. Inline borrows restore their local value copy. Fill uses one mutable borrow and one element pass. Hybrid copyFrom and DynamicVector resized use update into their already initialized elements.

DimensionStorage moves out both hybrid and managed payloads before mutation, borrow and diagnostics. makeUnique uses a no-op mutable borrow, so the same underlying COW boundary handles actual aliases. Default mode selection and always-managed generator construction are unchanged. Dynamic min/max/clamped each remove only the unused zero-array initialization; the selected ArraySIMDProvider witness and all argument/shape behavior remain.

Root explicitly approved managed mutation move-out after the new linked public-client baseline demonstrated copying. The historical OWN-26 zero-allocation observation came from an optimized source fixture; this probe is a separate, build-specific observation. Neither observation establishes that all default generic vector paths are quadratic. No audit closure or universal compiler claim is made here.

## Actual RED and retained setup attempts

The baseline library was built in Release before production edits. The authoritative before-04 workload completed all three phases with exit 0. At Float512, Hybrid repeating allocated 2,052 times / 2,142,288 requested bytes; ten unique writes allocated 40 times / 41,760 bytes; fill allocated 2,050 times / 2,140,200 bytes. A concrete managed control with direct start/stop and no generic measurement closure allocated ten 40-byte owners and ten 2,048-byte payloads for ten writes. Before-01/02 are retained exploratory probe revisions that added progressively narrower concrete controls; before-03 is an isolated compiler-cache sandbox setup failure.

Behavioral RED ran four new ownership matrix tests against unchanged production: exit 1 with 105 uniqueness assertion failures. The original RED loaded executable and test source were copied and hashed before rebuilding. Earlier red-01 failed to compile because Swift Testing's macro captures the receiver of a mutating getter immutably; explicit local Boolean observations corrected the new test. Expanded-green-01 had new-test generic constraint/throwing-expression setup errors. Expanded-green-02 exposed incorrect new-test metadata key assumptions, corrected to the existing expected_dimension/actual_dimension keys. No existing test expectation was edited. All logs and command statuses are retained separately.

## Behavioral coverage

Eight new Swift Testing functions are real in-process matrices; none is a no-op child entry. Hybrid Float covers 0/1/15/16/17/384/512/768/1536 and Double covers 0/1/7/8/9/16/17/384. The 17 cases check constructors, inline metadata, scalar endpoints, repeated fills, copying, real aliases, truthful uniqueness, ordinary and throwing borrows, exact return/error identity and alternate-copy mutation. Each borrow helper checks every element, including a separately retained pre-throw snapshot and an empty closure invocation.

Dimension Float covers 16/17/384/512/768/1536; Double covers 8/9/16/17, each in default/hybrid/heap mode: 30 cases. Constructor metadata, managed generator behavior and generator error identity, scalar/bulk/throwing operations, makeUnique, diagnostics, and retained aliases around fill/add/multiply/copyFrom are checked. Independent integer arrays supply finite value references.

Dynamic repeating and resize use 0/1/16/17/384/512/768/1536, with destination sets spanning zero, grow, shrink, equal and inline boundaries; prefix, nonzero fill-tail, mutated-result/source independence are asserted. Five actual fixed-vector convenience factories cover dimensions 16/17/384/512/768/1536 with independent arrays.

Dynamic provider matrices cover the eight dimensions above; fixed provider matrices cover 0/1/2/16/17/384/512/768/1536. TaskLocal spies supply distinctive outputs, record full bitwise argument sequences and exact call counts, and verify mismatches precede a Dynamic min/max provider call. Finite hand references are independent. NaN/infinity/signed-zero results are compared bitwise against the selected default provider, preserving its existing semantics rather than inventing a new numerical policy. No process-global provider replacement occurs.

The brief's example VectorCoreOperations filter has no matching existing suite. The final focused filter uses actual OperationsTests and OperationsValidationTests alongside DynamicVectorTests, VectorShapeContractTests and the new suite. Existing consumer search output is retained in `O03-existing-consumer-inventory.json`.

## Linked allocation comparison

Before-04 and after-01 use byte-identical C callback, Swift workload and Python driver sources and identical compile/link flags. The library command is `swift build -c release --target VectorCore --jobs 6 --scratch-path /private/tmp/vectorcore-stabilization-O03-library`. The driver snapshots immutable module/object bytes before linking with `swiftc -O -swift-version 6`. Each compile/link/run uses a file capture, 15-second execution deadline, owned-process-group kill if necessary, and at most two seconds of cleanup. No phase timed out. Complete commands, statuses, hashes and PIDs are in the provenance records. Output includes completion markers and complete result assertions after measurement.

The external malloc_logger callback observes successful allocation notifications and records requested size histograms. Synthetic notification filtering, actual malloc/posix_memalign allocations, a known Array allocation and a no-op region calibrate coverage. This is allocator requested-byte/event evidence; it is not live memory, RSS, latency or throughput. Setup and warm-up occur outside mutation measurement; complete values are consumed after measurement, with retained owners kept alive around shared mutation.

`O03-allocation-comparison.json` records all 627 aligned rows: 322 unchanged and 305 changed. It mechanically validates 268 zero-allocation unique mutation/diagnostic rows and 47 shared-detach rows (inline zero or one 40-byte owner plus one payload). Construction, resize and fixed-factory controls have bounded counts and linear requested payload bytes over the specified dimensions.

At Float512, final Hybrid repeating is two allocations / 2,088 bytes; unique writes and fill are zero; first shared detach is two allocations / 2,088 bytes. Default managed ten writes become zero; managed shared detach remains two allocations. Fixed range becomes two allocations / 2,088 bytes in this linked client. The concrete no-measurement-closure managed control also becomes zero.

OWN-22 has a measured bounded result in this build: Dynamic min/max each go from six to five allocations and clamp from five to four at Float512, each saving one 2,080-byte Array allocation. Fixed counterparts remain five/five/four. All selected provider counts and output checks remain intact. Input/provider-result array-interface opportunities remain; no broader provider redesign or tuning/performance target is claimed. Root owns final audit disposition.

## Validation state

Focused Debug: exit 0, 53 tests in nine suites. Final linked Release allocation compile/link/run: 0/0/0. Focused Release: exit 0, 53 tests in nine suites. ASan: exit 0, all eight valid new matrix tests pass with no sanitizer findings. No source edits after v1 freeze. Final commands/statuses/log hashes are in O03-test-command-status.json and O03-build-command-status.json. Test helper and loaded Debug/Release/ASan bundle executables are independently hashed in O03-test-binary-association.json. Worker evidence index excludes root and independent review artifacts. Canonical evidence and binaries are backed up outside /tmp.
