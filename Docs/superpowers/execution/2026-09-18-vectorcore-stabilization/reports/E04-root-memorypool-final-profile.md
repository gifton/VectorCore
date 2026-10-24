# E04 root allocation verification

The actual frozen E04 Release module was linked with the unchanged baseline Swift workload and C allocation callback. Root verified all 133 module/object/module-map artifact hashes against the corrected worker inventory, source hashes, command exits and captured logs. The callback calibration again recorded two events and 24,576 bytes. Each measured condition was repeated three times with identical allocation results.

| Cached buffers before return | Baseline events / bytes | E04 events / bytes |
|---:|---:|---:|
| 0 | 4 / 320 | 3 / 256 |
| 1 | 4 / 328 | 2 / 184 |
| 8 | 4 / 608 | 2 / 184 |
| 32 | 4 / 1,568 | 2 / 184 |
| 128 | 4 / 5,408 | 2 / 184 |

Pool setup, preload, pointer/statistics checks and printing are outside the sole-handle return interval. The empty-cache case includes its different first-bucket bookkeeping. E04 removes the measured growth in requested allocation bytes as retained cache size increases; return still allocates bookkeeping storage. Source inspection confirms that cached-byte admission now uses a maintained counter rather than flattened cache arrays. This supports API-32 within the exercised return path, not a universal zero-allocation or constant-time claim.

Raw contention samples from the same workload are retained for provenance only. Other worker builds were active and thermal state was uncontrolled. No throughput, wall-clock speedup, RSS reduction or production target is claimed. The host remains the same Apple M3 Max with 48 GiB memory used for the baseline.

The first compiler attempt failed because the sandbox denied Swift compiler-cache writes; its exact setup log and provenance are retained separately. The bounded v2 driver changes only execution/cleanup deadlines, leaving the measured Swift/C workload and compiler optimization arguments unchanged. The successful run did not time out. Root verified that all production files are byte-identical between E04 v1 and v2; the sole v2 source change adds an internal pool cached-destruction test. The linked artifact association is therefore to the unchanged final production implementation, with final v2 tests recorded separately.

Exact records: E04-root-memorypool-final-v1-provenance.json, E04-root-memorypool-final-v1-summary.json and phase logs. The retained executable SHA256 is in that provenance. A canonical backup of all linked artifacts, probe sources, executable and logs is recorded by E04-root-memorypool-final-v1-backup.json outside /tmp.
