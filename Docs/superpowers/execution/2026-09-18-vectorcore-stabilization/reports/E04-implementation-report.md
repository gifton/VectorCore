# E04 implementation and migration

The actor pools now own both checked-out and cached raw allocations. Every acquisition creates a fresh lease identity, including reuse of the same address. A return validates identity, requested count and address before consuming the live lease. Cache admission uses physical payload bytes and never exceeds its configured byte or per-size cap. Clearing a pool releases cached storage while active leases remain valid. Actor destruction releases both sets of allocations.

Actor handles remain borrowed metadata. Callers must retain the provider or typed pool until release, avoid pointer use after release, and synchronize pointee access independently. Sending handle metadata does not make its pointee Sendable. Generic storage is uninitialized on acquisition and must be uninitialized again before return: callers initialize and deinitialize exactly the elements used on every exit.

`TypedBufferPool.acquire` now returns its immutable `Handle`, and `release` accepts that handle. Replace raw-buffer release calls with `let handle = await pool.acquire(count: n)`, use `handle.buffer` or `handle.pointer` while the lease is active, then `await pool.release(handle)`. Construction now requires an explicit `maxCachedBytes`; `maxBuffersPerSize` still defaults to 10. The actor-isolated synchronous `withBuffer` helper accepts a Sendable closure and Sendable result. Its pointer must not escape, which is a caller duty rather than a type-system guarantee.

`BufferProvider.withBuffer` is an async throwing helper because acquisition can fail independently of the body. It retains the provider through the body and awaits release before success, error or observed cancellation returns to the caller. A body that ignores cancellation retains its lease until it finishes. `OperationContext.withTemporaryBuffer` now likewise awaits release on both exits instead of scheduling detached cleanup.

MemoryPool handles retain an allocation owner and a weak pool. They can outlive the pool, unlike the actor borrowed handles. Reuse preserves actual alignment, capacity, byte count and allocation family. Maintained cached bytes replace the return-time flattening scan. Independent bucket summation remains available internally for invariant checks. Its forced direct and final Swift raw fallbacks preserve requested alignment, bind fresh storage, and dispatch the matching deallocator. Final real allocator exhaustion may terminate; the implementation does not promise recovery from genuine OOM.

Operational counts and byte products are checked. Cumulative diagnostic counters saturate; disabled SwiftBufferPool tracking leaves cumulative and peak diagnostics zero while operational current usage remains observable. Invalid nonthrowing configurations and internal/typed requests use named fatal guards. SwiftBufferPool invalid sizes throw `BufferError.invalidRequest(size:)`; MemoryPool invalid acquisitions return nil. Nonpositive, nonfinite and beyond-horizon timer intervals disable automatic cleanup, while explicit cleanup retains the configured age-cutoff policy.

The previous ordinary-acquisition test named as fallback coverage was renamed without dropping its assertions. Float and UInt8 fixtures now initialize and deinitialize their used elements. Actual fallback coverage uses per-instance failure stages and real allocations, separately checking reference destruction and raw frees. The standalone probe compiles the same production owner helper; its watchdog control flow and deadlines are unchanged.

## Baseline and verification status

The unchanged ae0fd908 baseline was compiled directly with an isolated address-filtered allocator callback. Both Swift raw allocation/deallocation and posix_memalign/free controls observed exactly one free. All eight burst buffers were observed freed exactly once by explicit clear. With each actor weak-confirmed destroyed, each watched returned payload had zero frees rather than one. Separately, eight fresh SwiftBufferPool returns retained 512 bytes under a 64-byte cap. The retained executable replay exits 1 with the same four findings. Baseline command, source, object, executable and log hashes are retained in the E04 baseline records.

The final v2 full diff is `91b53d9683ac7f202c862088526a6cf1379bc29cacdb37a05f0ad081ab908816`. The 16 changed/new files are enumerated in `E04-source-manifest.json`. Complete code inventory verifies 255 base files and 262 final files across Sources, Tests, Scripts and Package.swift, with exactly the allowed 16 differences and no missing base files. The canonical recovery archive and its hash are recorded in `E04-backup.json`.

Independent v1 source review and root full reconstruction accepted the implementation. V2 adds only `internalCachedAllocationIsFreedByDestructionWithoutClear`, closing the self-audited missing internal cached-destruction assertion. Every production, standalone and public-consumer byte remains identical to v1; `E04-v1-v2-source-association.json` records that comparison. V1 diff, manifest, inventory and canonical archive remain preserved separately. All final worker checks below completed successfully. Root owns final independent runtime acceptance.

No universal allocation-free, latency, contention, Linux, GPU or downstream runtime claim is made. Root owns the repeated linked Release allocation probe and its measured interpretation.

## Final checks and exact evidence

`E04-v2-final-commands.json` contains exact commands, environment overrides, exit statuses, durations, log SHA256 values and frozen diff association. Debug and Release each exit 0 and report 48 test functions in four suites. Each includes 20 actual named rejection children, retained individually in `E04-guard-captures`; the parent `guardChildProbe` entry is a no-op without its environment marker. `E04-frozen-child-capture-provenance.json` associates all 80 children from the two v1 and two v2 frozen parent runs with their parent command and source diff. The separate calibrated replay records 40 exact Debug/Release child commands, overrides, bundle/host hashes, statuses and log hashes in `E04-direct-guard-provenance.json`. Each replay installs its termination handler before launch, writes to files, and uses a 15-second execution deadline plus at most two one-second cleanup waits on its own PID.

ASan exits 0 with the same reported 48 functions. Actual valid-path coverage is 46 functions, or 63 invocations after expanding the four valid parameter matrices. The parent childProbe function does nothing, and the 20 guard-wrapper invocations return early under `E04_SKIP_GUARDS=1`. No intentional rejection child is credited as sanitizer coverage. `ASAN_OPTIONS=detect_leaks=0` is explicit; raw ownership balance comes from the real allocation ledgers, and this is not a LeakSanitizer claim.

`E04-final-commands.json` retains standalone Debug and Release exits 0, each covering lifetime, retention, cleanup and saturation with both ordinary and strict cooperative executors. All three watchdog regressions pass. These inputs are byte-identical in v2, as recorded above, so their v1 association remains valid without unnecessary reruns.

The positive public consumer compiles, links and runs with Swift 6 strict concurrency, exit 0 at both stages, and prints `E04_PUBLIC_SCOPED_CONSUMERS_COMPLETE`. It exercises the exact provider documentation example and nonisolated, actor and MainActor contexts. Three negative fixtures exit 1: non-Sendable results and captures are rejected at both scoped APIs, while the legacy fixture rejects the omitted byte budget and the old typed raw acquire/release shape. Full commands, diagnostics and object/module hashes are retained in the v2 command and public-linked-artifact records.

The initial positive consumer link omitted six nested VectorCoreC objects and failed on LAPACK symbols. The corrected recursive inventory links the unchanged source successfully. The initial two compile attempts that failed on a private synthesized initializer and recursive nested Testing macro expansion are retained in the intermediate logs. Report-only replay setup failures are likewise retained: direct execution of a Mach-O bundle, missing testing framework paths, and omission of the Swift Testing selector. The calibrated final driver changes only its host invocation/environment, not library or test source. No setup failure is represented as a behavioral pass.

## Measured return bookkeeping

Root's unchanged linked Release workload measured these values identically in three repetitions:

| Cached buffers before return | Final allocation events | Final requested bytes |
| --- | ---: | ---: |
| 0 | 3 | 256 |
| 1, 8, 32, 128 | 2 | 184 |

The baseline return allocation bytes grew from 328 at one cached buffer to 5,408 at 128. The final measurement establishes removal of that cache-size growth, while retaining bookkeeping allocations. Timing samples were collected with other builds active and do not establish a latency or contention speedup. Root retains the independent profile provenance and raw results in its separately owned `E04-root-*` records. `E04-release-artifact-inventory.json` hashes 133 copied Release objects/module artifacts; the final recheck confirms both originals and snapshots still match. Production hashes are identical across v1 and v2, so the linked measurement applies to the final implementation.

## Handoff and recovery

`E04-worker-evidence-hashes.json` inventories worker-owned source evidence, commands, logs, children and this report. It deliberately excludes root-owned `E04-root-*`, the independently maintained implementation review, and earlier design records. Those have separate verification lifecycles.

Canonical recovery is `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-09-23/E04`. It contains the preserved v1 and final source archives, retained baseline/public/replay executables and allocation object with hashes, and a byte-identical copy of the worker evidence files under `reports`. The archive source remains untouched after the final v2 freeze. No Package changes, sibling edits, routing-gate changes, index operations or commits were made by this worker.
