# Storage measurements

Measured in Release on Apple M3 Max (Mac15,9), 16 logical cores, 48 GiB RAM; macOS 27.0.1 (26A434), Xcode 27.1 (27A9275), Swift 6.4. Both sequential runs reported thermal state 0 (nominal). No other audit build was active.

Baseline `84798d3`; repaired implementation `f6bb24e`. Each cell uses two warm-up batches and nine timed batches: 50,000 iterations for scalar/borrow and scalar/dot, 20,000 for copy/borrow. Values below are median nanoseconds per iteration. Raw samples, largest-sample percentile estimate, checksums and allocator snapshots are in [storage-measurements.json](storage-measurements.json).

| Storage | Width | Operation | Baseline ns | Repaired ns |
|---|---:|---|---:|---:|
| AlignedDynamicArrayStorage | 512 | scalarWriteAndReadBorrow | 0.65 | 8.93 |
| AlignedDynamicArrayStorage | 512 | scalarWriteAndDot | 24.17 | 35.12 |
| AlignedDynamicArrayStorage | 512 | copyAndMutableBorrow | 0.68 | 105.49 |
| AlignedDynamicArrayStorage | 1536 | scalarWriteAndReadBorrow | 0.60 | 8.53 |
| AlignedDynamicArrayStorage | 1536 | scalarWriteAndDot | 69.02 | 73.65 |
| AlignedDynamicArrayStorage | 1536 | copyAndMutableBorrow | 0.67 | 226.43 |
| DynamicArrayStorage | 512 | scalarWriteAndReadBorrow | 20.67 | 9.87 |
| DynamicArrayStorage | 512 | scalarWriteAndDot | 45.17 | 23.82 |
| DynamicArrayStorage | 512 | copyAndMutableBorrow | 29.51 | 77.52 |
| DynamicArrayStorage | 1536 | scalarWriteAndReadBorrow | 19.73 | 10.45 |
| DynamicArrayStorage | 1536 | scalarWriteAndDot | 90.96 | 68.77 |
| DynamicArrayStorage | 1536 | copyAndMutableBorrow | 25.65 | 124.64 |
| COWDynamicStorage | 512 | scalarWriteAndReadBorrow | 1.32 | 6.91 |
| COWDynamicStorage | 512 | scalarWriteAndDot | 23.31 | 25.32 |
| COWDynamicStorage | 512 | copyAndMutableBorrow | 135.17 | 82.23 |
| COWDynamicStorage | 1536 | scalarWriteAndReadBorrow | 1.32 | 6.91 |
| COWDynamicStorage | 1536 | scalarWriteAndDot | 67.81 | 70.49 |
| COWDynamicStorage | 1536 | copyAndMutableBorrow | 279.95 | 197.76 |

The aligned value adds a uniqueness check before mutation. In this probe, its scalar-write/read-borrow operation increased by about 8 ns; its scalar-write/dot operation increased by about 5–11 ns. The existing COW wrapper scalar/borrow increased by about 6 ns, while its copy/borrow became faster in this sample. DynamicArrayStorage scalar/dot became faster in this sample. These observations do not establish a performance budget or an application-level speedup/slowdown.

For the two former public classes, copy/borrow now performs an independent copy: the baseline benchmark instead mutates the shared object. Their before/after copy timings therefore measure different semantics. Holding 16 mutated copies now retains separate buffers; the former classes retained no extra buffers. The allocator snapshot captures net live blocks and bytes, with allocator caching visible across rows. It does not count allocation calls or measure peak process memory.

The harness and build commands are retained in the external repair evidence directory as `storage-benchmark.swift`, `run-storage-benchmark.py`, and `logs/benchmark-*-build.log`. No throughput guarantee is inferred from this single-host microbenchmark.
