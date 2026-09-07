# Top-K NaN contract verification — 2026-09-05

Implementation and measurements: September 5. Full verification: September 6.

Baseline: `ad39ab6` (`main`, VectorCore 0.3.2). The owner subsequently requested
the patch bump to 0.3.3 and a PR containing this implementation. The pre-existing
`.antigravitycli/` and `GEMINI.md` are excluded. Swift tools version 6.0 and
deployment targets are unchanged; no dependency-cache edits were performed.

## Contract and implementation

`TopKSelection.orderedBefore` provides best-first ordering for both directions.
Non-NaN scores, including infinities, precede NaNs. Numeric inequality uses the
requested direction; exact numeric equality (including signed zeros) and two
NaNs use the tie policy. `.smallerIndex` and `.insertionOrder` compare original
indices; `.smallerValue` leaves equivalent identities/order unspecified.
Index policies form a strict total order for unique indices; value-only policy
forms a strict weak order. This is an application contract, not IEEE `totalOrder`.

The comparator drives heap admission/repair, pair and pointer selection,
ascending/descending extraction, and CPU wrapper output. Merge construction
preserves the destination policy and checks that all participating policies
match. Existing production merge callers all use `.smallerIndex` and global
candidate indices. The pointer heap still scans n scores using O(k) auxiliary
storage and O(n log k) selection work; array/generic pair materialization was
already O(n) and remains so.

Changed paths:

- `TopKSelection.select`: array, pointer and generic elements; generic nearest
  and batch-nearest wrappers inherit the same selection behavior.
- All optimized `TopKSelection` paths through `TopKBuffer` and result extraction,
  including descending `nearestDotProduct512` similarity output.
- `Operations.findNearest`: generic selection and optimized parallel Euclidean,
  cosine and negated-dot output.
- `Operations.findNearestBatch`: selected Euclidean GEMM NaNs remain NaN.
- `BatchOperations.findNearest`: serial/parallel pair selection shares the
  canonical heap/sort helpers without re-enumerating candidate identities.

Public documentation is in `TopKSelection.swift`, `Operations.swift`, and
`BatchOperations.swift`; the observable fix is recorded under Changelog 0.3.3.

## Literal fixtures and coverage

`Tests/ComprehensiveTests/TopKNaNContractTests.swift` holds the shared fixtures:

```text
index:       0    1     2    3     4    5    6    7    8
score:     NaN    2  -Inf    2  +Inf   -0   +0  NaN   -3
min k=9:  [2, 8, 5, 6, 1, 3, 4, 0, 7]
max k=9:  [4, 1, 3, 5, 6, 8, 2, 0, 7]
min k=5:  [2, 8, 5, 6, 1]
max k=5:  [4, 1, 3, 5, 6]
```

The suite checks both heap directions, public heap/sort crossover, cardinality,
all-NaN inputs, numeric infinity over NaN, exact finite ties, adjacent Float
values, signed-zero bits, nonmonotonic/duplicate pointer IDs, generic element
labels, all three tie policies, and comparator laws. Disjoint global-index
chunks are merged under two partitions and three orders, including all-NaN and
cutoff-tie fixtures. Value-only equivalence does not assert signed-zero identity.

`Tests/ComprehensiveTests/TopKNaNWrapperTests.swift` checks actual CPU routes
with `CPUComputeProvider`: generic selection, five optimized finite-tie output
paths, parallel Euclidean and negated-dot NaNs, batch serial/parallel heap and
sort selection, and a real 8-query/256-candidate Euclidean GEMM batch. Direct
metric assertions establish computed NaNs before testing selection/formatting.
Empty/nonpositive-k/cardinality controls supplement existing
`OperationsValidationTests`, which retains throwing validation behavior.

## Red-before-green evidence

| Run before the relevant production fix | Discovered tests | Result |
| --- | ---: | --- |
| Initial core mechanism regressions | 2 | Both failed; 5 issues |
| Existing finite-tie controls | 7 | Passed |
| Expanded core fixture/merge regressions | 12 | Failed; 584 issues |
| CPU wrapper regressions | 7 | All failed; 55 issues |

The initial heap kept index 0's NaN instead of admitting index 1's score 4 in
both modes. Canonical array sorting returned `[0,2,5,6,1,3,4,7,8]` instead of the
literal minimum order. Expanded merge failures included resetting destination
`.smallerValue`/`.insertionOrder` to `.smallerIndex`. Wrapper controls isolated
finite-tie permutations such as `[2,1,0]`; the GEMM matrix contained NaNs before
formatting converted the selected NaNs to zero.

Initial two-test failures were recorded in the tool output; the expanded run
reused `/private/tmp/vectorcore-topk-core-red.log`. Other diagnostic logs are
`/private/tmp/vectorcore-topk-ties-baseline.log`,
`/private/tmp/vectorcore-topk-wrapper-red.log`, and
`/private/tmp/vectorcore-topk-combined-green.log`.

## Final verification

The combined implementation run passed 65 Swift Testing tests in 9 suites,
including all seven initial wrapper regressions. The final mandated filter
`swift test --filter 'TopKNaNContractTests|TopKTieBreakingTests|TopKSelectionSuite'`
passed 41 tests in 3 suites, zero failures/skips. XCTest discovered zero tests
under these filters; these counts come from Swift Testing, not that empty runner.

`swift test` completed successfully on September 6: 1,138 Swift Testing tests in
167 suites (18 disabled/skipped), plus 91 XCTest cases (27 skipped), zero failures.
Swift Testing reports 1,138 tests including disabled tests; 1,120 ran. XCTest's
91 also includes its skips; 64 ran. The Swift Testing run took 387.480 seconds,
dominated by existing auto-tuner calibration tests. Raw log:
`/private/tmp/vectorcore-topk-debug.log`.

`swift test -c release` also completed successfully: the same 1,138 Swift Testing
tests in 167 suites (1,120 run, 18 skipped), and 91 XCTest cases (64 run,
27 skipped), zero failures. Release compilation took 147.98 seconds; Swift
Testing took 22.728 seconds. Raw log: `/private/tmp/vectorcore-topk-release.log`.
Skipped tests are the existing opt-in performance/extended cases. No pre-existing
test failures or new failures were observed in either final full run. Existing
deprecation and test-source compiler warnings remain outside this change.

`git diff --check` passed after final edits. `python3
Benchmarks/TopKNaNContract/compare.py` reproduced the benchmark table and verified
all baseline/final and array/pointer checksums.

After the requested 0.3.3 patch bump, the combined Top-K/core/wrapper filter passed
49 tests in 4 suites. A fresh full Release run passed with the same 1,138 Swift
Testing and 91 XCTest totals and skip counts above; Swift Testing took 22.417
seconds. Logs: `/private/tmp/vectorcore-033-targeted.log` and
`/private/tmp/vectorcore-033-release.log`. A source consistency check confirmed
the version string and major/minor/patch components all resolve to `0.3.3`.

## Finite-input release measurements

The standalone harness in `Benchmarks/TopKNaNContract` uses fixed-seed mixed and
17-value duplicate-heavy inputs with n=100,000, k=10/20,000, and array/pointer
entry points. Baseline and final were separately compiled in Release. Each run
used five warmup samples and fifteen samples of ten selections, consuming every
output index and score in an untimed checksum. Two runs per revision executed
in baseline/final/final/baseline order; the medians below pool thirty samples.

Environment: Apple M3 Max, 48 GiB, arm64, macOS 26.5.2, Swift 6.3.3. Power varied
between battery and AC with battery discharging; settings were left unchanged.

| Input | API | k | Baseline ms | Final ms | Change |
| --- | --- | ---: | ---: | ---: | ---: |
| Mixed | Array | 10 | 0.4471 | 0.4489 | +0.4% |
| Mixed | Pointer | 10 | 0.3299 | 0.3318 | +0.6% |
| Mixed | Array | 20,000 | 7.4085 | 7.8470 | +5.9% |
| Mixed | Pointer | 20,000 | 6.8817 | 7.5591 | +9.8% |
| Duplicates | Array | 10 | 0.4515 | 0.4500 | -0.3% |
| Duplicates | Pointer | 10 | 0.3311 | 0.3341 | +0.9% |
| Duplicates | Array | 20,000 | 4.4386 | 4.6979 | +5.8% |
| Duplicates | Pointer | 20,000 | 3.1834 | 3.7762 | +18.6% |

All eight baseline/final checksums agree; array/pointer outputs agree per case.
Public-import optimized Euclidean and dot calls also passed a downstream
compilation/runtime smoke check, exercising the public inlinable visibility chain.
There is measured sort-path overhead; no acceptance threshold was defined.
These local timings do not establish statistical significance or allocation
counts. See `Benchmarks/TopKNaNContract/RESULTS.md` for raw samples, fingerprints,
the comparison script, and reproduction instructions.

## Scope and release handoff

There is no metric-formula, cosine-degeneracy, overflow-rescue, GPU-routing,
concurrency, public-signature, `TopKResult.Equatable`, or Codable change.
Optimized Euclidean membership still uses squared scores before output rooting;
distinct squared scores may round to equal rooted Floats without becoming ties
at admission. CPU/GPU computations may produce different rounded scores.
Third-party `BatchKernelProvider` implementations retain responsibility for
their own results. No signaling-NaN/payload preservation or ordering guarantee
was added, and overlapping candidate-set deduplication remains outside scope.

Call-graph review found no remaining value-only selection comparator in the
four changed source files. Other sorts in `BatchOperations` reassemble chunks
by original indices. Separate selectors remain outside this graph:

- `VectorMath.nearestNeighbors` in `Operations/VectorMath.swift`.
- Internal `SyncBatchOperations.findNearest` and its selection helpers.
- Internal `ExecutionOperations.findNearest` and public `KNearestHeap` in
  `Operations/MinHeap.swift`.
- `KNNGraph.bruteForce` has its own insertion selector.

The separate `BatchOperations.gemmPairwiseTyped` Euclidean matrix-formatting
clamp is also unchanged. This report does not claim a repository-wide selection
contract or a fix to that matrix API.

Independent read-only review found no blocking correctness issues; its minor
empty/pointer/batch edge-coverage findings were addressed in the tests.

The owner selected 0.3.3 and retains publication control. After publication, send
VectorAccelerate the version/tag and commit SHA, this record, the two contract
test paths, and the benchmark comparison. This PR prepares 0.3.3 without creating
a release tag. VectorAccelerate's Metal implementation and shader-build verification
remain separate work; this handoff does not mark VA3-016 fixed.
