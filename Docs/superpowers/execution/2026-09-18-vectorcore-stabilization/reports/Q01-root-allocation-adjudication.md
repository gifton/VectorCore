# Q01 root allocation review — revision 2

Root measured actual linked Release production routes after finding dynamic arrays in the new affine-difference expansion. This is a work-amplification check, not a latency, throughput, RSS or operating-envelope claim. The probe uses the previously calibrated O02 process-local allocation callback; its filter and malloc/posix_memalign calibrations passed again (two large allocation events, 24,576 requested bytes).

The baseline build is integration milestone03 at source028b4aacc80249b745a89454008c7d2b9e5da154. Q01 artifacts were copied after the worker reported completed/stable VectorCore compilation, with production source hashes matching Q01-review-v2-manifest.json (also unchanged in v3). Each probe links its own snapshot of the module and all objects. Snapshot artifact SHA256 values, exact compiler/link commands and exits are retained in Q01-root-allocation-{baseline,v2}-provenance.json, with phase logs. The initial sandbox compiler-cache denial is retained separately as baseline-setup. Both unsandboxed probes compiled and ran with exit0.

The probe warms each route before measuring a single call. Construction, JSON decoding, warmup and printing are outside the allocation interval. Results are consumed after measurement; SoA sentinels are checked. Source: Q01-root-allocation-probe.swift; orchestration: Q01-root-run-allocation-probe.py; callback: retained O02-allocation.c. Optional C kernels remain disabled. The generic functions measured are the actual production implementations called by the public wrappers, not copied arithmetic.

| Route | Dimension/candidates | Baseline events/bytes | Q01 v2 events/bytes |
|---|---:|---:|---:|
| Scalar equal-scale control | 512/1 and1536/1 | 0/0 | 0/0 |
| Scalar unequal scales | 512/1 | 0/0 | 2,048/81,920 |
| Scalar mixed | 512/1 | 0/0 | 2,048/81,920 |
| Scalar unequal scales or mixed | 1536/1 | 0/0 | 6,144/245,760 |
| SoA unequal scales | 1536/5 | 9/5,080 | 30,720/1,228,800 |
| SoA mixed | 1536/5 | 1/52 | 30,720/1,228,800 |
| Ordinary calibration | 1536/8 | 0/0 | 28/156,288 |

The baseline unequal-scale distance is known incorrect and result bits differ; this table does not establish equivalent numerical behavior or a speed ratio. It establishes new per-coordinate heap work in the reviewed repair. Mixed scalar and ordinary calibration control result bits match in these fixtures. Full outputs also include dimension512, candidate counts0/1/5, and calibration counts1/8.

Root ruling: fix the expansion's avoidable dynamic scratch with bounded value storage while preserving the same four-term insertion/two-sum arithmetic. Four inserted terms require at most four expansion components. Restore streaming finite validation/extrema in the ordinary calibrator; retain canonical empty and unsupported-strategy validation order. Distribution calibration may still materialize values for sorting. No independent latency target is introduced. Keep revision2/3 evidence, publish a complete revision4 manifest/diff, rerun affected correctness checks, then repeat this allocation probe before acceptance.

## Revision4 verification

Root repeated the exact probe against frozen revision4 production after the worker confirmed completed/stable VectorCore compilation. Compile and run exited0; callback calibrations passed. All22 measured rows reported zero allocation events and zero requested bytes, and every result Float bit pattern matches revision2. This resolves the resource finding for these bounded fixtures. Complete v4 artifact/source hashes, commands and logs are retained as Q01-root-allocation-v4-provenance.json and Q01-root-allocation-v4-phase-*.log. Source review independently accepted the four-component scratch bound and streaming calibrator. Final task acceptance still requires the worker correctness/sanitizer evidence.
