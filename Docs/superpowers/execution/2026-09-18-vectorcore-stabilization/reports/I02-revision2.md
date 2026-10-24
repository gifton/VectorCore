# I02 — exact FP16 cache identity and bounded retention

Status: implementation frozen for independent review; revision2 focused Debug and Release passed.

Base: `98388fc96fb62b013eea84dcf97ed092fdb76f8d`.
Archive: `/private/tmp/vectorcore-stabilization-I02-implementation`.
Frozen deliverables: `I02.patch` and `I02-manifest.json`. Full patch includes both modified files and the newly created identity test file. The manifest records base/final source hashes and patch SHA256.

## Contract and migration

The dictionary key retains every ordered candidate coordinate as an immutable UInt32 Float-bit snapshot plus count/dimension. A fingerprint only selects buckets; equality compares full snapshots, including NaN payloads and signed zero. The internal constant-fingerprint initializer exercises real dictionary get/getOrCreate collisions. Retained entry keys are reused for LRU updates so transient equal lookup snapshots are not retained as duplicate allocations.

The initializer requires `maxBytes`, with independent default `maxSize: 100`. Zero in either limit disables retention while conversion continues. Negative limits fail at initialization with named `fatalError` diagnostics. Consumers replace removed `shared` with an owned budgeted actor and replace removed public arbitrary-payload `store` with `getOrCreate`. The original statistics tuple is unchanged; `retentionStatistics()` exposes used bytes and byte limit.

Retention counts `key.bits.capacity * MemoryLayout<UInt32>.stride + soa.storage.capacity * MemoryLayout<UInt16>.stride`, using checked arithmetic. This includes padded SoA capacity, not just logical candidate count. Dictionary/LRU metadata, transient allocations, and allocations held only by callers are excluded. These are allocated payload capacities, not RSS or a production memory envelope. Immutable returned SoA COW storage remains valid after eviction/clear; its caller-held allocation is no longer retained/accounted by the actor.

Oversized/overflow-cost entries bypass insertion before eviction. LRU eviction continues until both count and byte constraints fit, using subtraction to avoid addition overflow. Clear resets retained bytes and hit/miss statistics.

## Verification evidence

- `I02-red.log`: initial sandbox compiler-cache denial, not behavioral RED. Preserved separately.
- `I02-red-runtime.log`: unchanged production cache plus first regression; exit 1. All 512 coordinates tested: 496 false cache hits, 17 misses instead of 513; 498 recorded issues (496 storage mismatches plus two statistics failures). No source repair preceded this runtime RED.
- `I02-revision2-debug.log`: `swift test --jobs 6 --scratch-path /private/tmp/vectorcore-stabilization-I02-debug --filter 'SoAFP16CacheIdentityTests|MixedPrecisionPhase3'`; exit 0, 22 tests/2 suites. Negative child traps passed with named diagnostics; all tests finished in 0.142 seconds after build.
- `I02-revision2-release.log`: same filter with `-c release` and distinct scratch `/private/tmp/vectorcore-stabilization-I02-release`; exit 0, 22 tests/2 suites. Optimized negative child diagnostics passed using the bounded harness.

The Debug/Release build source association is the final SHA256 set in the manifest. Existing mixed-precision tests were migrated only at constructor/manual-store call sites. Builds retain platform/compiler validation flags and use at most six jobs, sequentially.

Coverage: full-coordinate independent uncached oracle; candidate count/order and counts 0/1/3/4/5/7; source mutation with retained baseline; freshly constructed identical inputs; deliberate collisions; signed zeros/NaN payloads; zero/one count capacity, repeated eviction, hit recency; exact-byte fit/one short, padded capacity, oversize bypass preserving existing data, multi-entry byte eviction; hit accounting/clear; overflow helper artificial near-Int.max values; retained results after eviction and clear.

Negative-limit tests spawn isolated copies of the Swift test process, filter to the child probe, disable backtraces and capture stdout/stderr in a temporary file. A termination semaphore provides a monotonic 15-second deadline. Timeout sends SIGKILL only to that owned Process child and allows at most two additional seconds for cleanup; the test records failure with captured output without an unbounded wait/read. Normal completion requires signal termination plus the named initialization diagnostic. Revision1 patch/manifest/report/Debug/Release logs are preserved as I02-revision1*. Production bytes are unchanged in revision2. The timeout branch is source-reviewed; normal child completion is exercised by both focused suites.

Existing unrelated warnings include deprecated Apple BLAS calls, Float literal underflow in NumericalStabilityTests, and other legacy test diagnostics. No throughput, latency, RSS, workload-sizing, or additional cache-width claims are made. Independent review and root integration remain pending.
