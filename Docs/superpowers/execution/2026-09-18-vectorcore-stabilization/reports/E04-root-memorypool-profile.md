# MemoryPool profile before E04 design

Root measured the actual production MemoryPool at accepted source028b4aacc80249b745a89454008c7d2b9e5da154, using the preserved complete milestone03 Release module/object snapshot from the Q01 baseline probe. The initial attempt against the original build failed because temporary-file loss removed its VectorCoreC module metadata; that setup log/provenance is retained separately. The preserved snapshot compiled and ran successfully. Exact commands and copied artifact/source/log hashes are retained in E04-root-memorypool-baseline-provenance.json.

The existing O02 process-local allocation callback passed its event-filter and actual malloc/posix_memalign calibration. Pool setup, preloading, pointer checks, statistics and printing are outside the measured return interval. An explicit noninline inout-optional release drops the sole handle inside that interval. Each case verifies no active handles and the expected retained count after return; no initialized payload is accessed. Timer cleanup is disabled. Three independent repetitions agree:

| Cached buffers before return | Allocation events on return | Requested bytes |
|---:|---:|---:|
| 0 | 4 | 320 |
| 1 | 4 | 328 |
| 8 | 4 | 608 |
| 32 | 4 | 1,568 |
| 128 | 4 | 5,408 |

Separate exploratory contention samples exercise one or four workers, 1,000 acquire/release pairs each, with 0 or128 unrelated cached buffers. Five raw samples per condition, medians/minimum/maximum and hardware are in E04-root-memorypool-profile-summary.json. Other worker builds were active and thermal state was uncontrolled. These samples characterize this host/run only; they establish no production latency target or projected speedup.

Root decision: replace the return-time flattened cache scan with a maintained cached-payload-byte counter as part of E04. Source inspection establishes that the scan/temporary arrays execute under the lock, and the measured requested-byte growth confirms work amplification with retained buffer count. This does not require a user-selected performance target. The counter must be updated on every reuse, admission, eviction, cleanup and destruction path and verified against independently recomputed type/size bucket sums. Preserve active/allocated accounting and late-handle ownership. Repeat the allocation probe after repair; any remaining dictionary/bookkeeping allocations must be reported rather than mislabeled as zero-allocation return.

This report settles the API-32 implementation choice only. E04 teardown, checked allocation/fallback, actor-handle ownership and scoped API contracts still require the complete brief and independent review before implementation.
