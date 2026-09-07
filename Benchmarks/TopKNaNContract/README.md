# Finite Top-K comparison

This standalone package uses the existing VectorCore library without adding a
benchmark framework or changing the root package. Build each revision in a separate
scratch directory, then run the executables while other builds/tests are idle.

```sh
VECTORCORE_BENCH_SOURCE=/absolute/path/to/source swift build -c release \
  --package-path Benchmarks/TopKNaNContract \
  --scratch-path /private/tmp/topk-baseline-build
/private/tmp/topk-baseline-build/release/TopKNaNContractBench > /private/tmp/topk-baseline.csv
```

Repeat with the final source path and a separate scratch path. Save `swift --version`,
`sw_vers`, `sysctl -n machdep.cpu.brand_string`, and `sysctl -n hw.memsize` alongside
the results. Use the same machine and power settings for both revisions. The build
is Release (`-O`), without extra optimization flags.

The fixture seed is `0x5EED202609050042`, using SplitMix64. Mixed scores are finite,
signed multiples of 1/128 from a 24-bit random integer; duplicate-heavy scores have
17 distinct integer values. Both use n=100,000 and the default smaller-index policy.
k=10 exercises the heap route; k=20,000 exercises the sort route, for both array and
pointer APIs with no external IDs.

Each case has five warmup samples and fifteen measured samples, each containing ten
selections. CSV rows include the median nanoseconds per selection and all sample
averages. Timing includes selection and result allocation, and excludes fixture
generation, output hashing, and result destruction. Every selected index and Float
bit pattern feeds an order-sensitive checksum outside the timed region; a
precondition verifies repeated outputs, and the accumulated checksum is printed.
Compare checksums between revisions and APIs. Report `(final / baseline - 1) * 100`
for timing deltas; this harness defines no pass/fail performance threshold.

The checked-in [measurement record](RESULTS.md) uses two runs per revision in
baseline, final, final, baseline order. Regenerate its table and checksum checks
with `python3 Benchmarks/TopKNaNContract/compare.py`.
