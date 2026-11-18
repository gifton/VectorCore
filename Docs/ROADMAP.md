# VectorCore roadmap

Release-preparation references reviewed November 4, 2026. This is a
direction-setting document, not a release schedule or an assertion that
proposed work is implemented. The
[0.4.0 qualification record](reviews/2026-11-04-release-qualification/README.md)
tracks verified results and remaining release limits.
The [API overview](API_Overview_Map.md) describes current code; the
[changelog](../CHANGELOG.md) records releases.

## Current foundation

The package already includes:

- Generic, dynamic, and optimized 384/512/768/1536-dimensional vectors.
- CPU vector/batch math, distance metrics, and Top-K selection.
- Matrix distance operations and reusable prepared candidates.
- Provider interfaces, aligned buffers, and the published FP32 SoA layout.
- Dense linear algebra, PCA, neighbor-graph interchange, and UMAP layout.
- Quantization and serialization primitives.

These are not open feature requests. Their existence also does not imply
universal routing coverage, allocation-free execution, or identical numerical
behavior between implementations. The current limits belong in the reference
docs, not in a list of future promises.

## Documentation maintenance

Keep the README and reference material aligned with the code and distinguish
historical designs from supported interfaces.

Acceptance criteria:

- Public examples compile and run as downstream consumers, without
  `@testable import` or internal symbols.
- Provider types, routing gates, score conventions, and memory ownership
  match implementation and named regression coverage.
- Performance statements point to reproducible measurements and state
  their limits; deployment targets are not presented as executed device tests.
- Package descriptions identify real libraries and avoid claiming unverified
  version combinations.
- Historical plans remain discoverable but are clearly not current API docs.

The coordinated tutorial refresh remains a separate follow-up for
[`Guides/`](../Guides/): review prerequisites, progression, duplication, and
runnable examples. The [performance reference](Performance_Guide.md) and
[nearest-neighbor how-to](HowTo_NearestNeighbor.md) are maintained references;
their code fragments are outside the standalone consumer-program check.

## Future candidates

The following are proposals to evaluate against real consumers, not approved
implementation commitments.

| Direction | Why it may belong in Core | Evidence needed before implementation |
|---|---|---|
| Numerical-contract consistency | Consumers should be able to reason about scores across routes | Explicit non-finite/zero-norm policies and tests for every affected route; assess compatibility impact |
| Bounded-memory batch distance/search | Current matrix routes materialize query-by-candidate scores | Peak-memory and latency profiles from a real workload; comparison with explicit caller batching |
| Reusable prepared-data interfaces | Consumers repeatedly pack or convert the same candidates | Demonstrated duplicate work, lifetime requirements, and measured benefit |
| Cross-package contract fixtures | Shared types cross independent release boundaries | Pinned consumer revisions, exercised APIs, and a maintainable verification matrix |
| Measured kernel/routing improvements | Shape, metric, and precision change the useful crossover | Release measurements with correctness checks, raw data, hardware, and representative sizes |

A proposal should identify the consumer, public contract, failure modes,
compatibility cost, and validation plan. Favor a small reproducible case over
a broad checklist of algorithms.

## Work that stays downstream

Index traversal and persistent index structures belong with VectorIndex.
Metal resource management and GPU execution belong with VectorAccelerate.
Model/tokenization pipelines belong with EmbedKit; topic extraction and
clustering policy belong with SwiftTopics. These are ownership directions,
not a claim that all duplicated functionality has already been consolidated.
See [Package Boundaries](Package_Boundaries.md).

Core should not acquire a database, embedding runtime, or application UI merely
to make the ecosystem diagram look complete.

## Historical plans

Earlier roadmaps mixed shipped features, speculative APIs, implementation
checklists, and target dates. They are superseded as current planning
references by this document. The [documentation index](README.md#historical-designs-and-plans)
preserves links to the original design series and audits. Consult Git history
for earlier versions of this roadmap; dated verification records remain
evidence for the specific revisions they describe.
