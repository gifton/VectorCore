# VectorCore documentation

The references below describe the code in this checkout. For an installed
release, read the files at its tag and consult the [changelog](../CHANGELOG.md).
A feature on `main` is not necessarily in the latest published release.

## Current references

| Document | Purpose |
|---|---|
| [Project README](../README.md) | Installation, a complete quick start, capabilities, and support limits |
| [API overview](API_Overview_Map.md) | Public entry points, provider bindings, and actual search/matrix routing |
| [Package boundaries](Package_Boundaries.md) | What Core owns and where the surrounding libraries fit |
| [Numerical behavior](Numerical_Behavior.md) | Score conventions, Top-K ordering, precision, and validation boundaries |
| [Memory alignment](Memory_Alignment.md) | Contiguity, allocation, borrowing, transfer, and caller obligations |
| [SoA layout contract](SoA_Layout_Contract.md) | Frozen FP32 lane layout and logical versus allocated bytes |
| [Linear algebra and projection](Linear_Algebra_and_Projection.md) | Factorization shapes, PCA, KNNGraph, UMAP, and scale limits |
| [Roadmap](ROADMAP.md) | Current documentation priority and explicitly proposed future directions |
| [Contributing](../CONTRIBUTING.md) | Development, verification, performance evidence, and support policy |
| [Maintaining and forks](Maintaining.md) | Review, release, and repository-policy maintenance |
| [Security reporting](../SECURITY.md) / [Code of Conduct](../CODE_OF_CONDUCT.md) | Private reporting and community expectations |

The [source](../Sources/VectorCore/) defines exact signatures.
The [tests](../Tests/) exercise particular contracts; neither a green test run
nor a source comment proves a broader unsupported guarantee.

Complete Swift blocks in the README Quick Start and the refreshed API, memory,
SoA, and projection references are compiled and executed by
[`consumer_smoke.py`](../Scripts/ci/consumer_smoke.py). This checks public
consumer usage, not exhaustive numerical correctness or external GPU integration.

## Tutorials: separate refresh pending

The following are retained unchanged for a coordinated guide pass:

- [Guides directory](../Guides/)
- [Performance Guide](Performance_Guide.md)
- [Nearest-neighbor how-to](HowTo_NearestNeighbor.md)

These tutorials can contain outdated signatures, routing descriptions, or
performance claims. Until refreshed, use the current references above for
contracts and the [contribution benchmark instructions](../CONTRIBUTING.md#performance-changes)
for the benchmark CLI. They are not part of the refreshed-example check.

## Verification and measurements

These are dated evidence, not rolling promises of support or performance:

- [0.3.1 verification baseline](verification-baseline-0.3.1.md)
- [Top-K contract and 0.3.3 verification](verification-topk-nan-contract-2026-09-05.md)
- [Public-readiness acceptance, September 7, 2026](verification-public-readiness-2026-09-07.md)
- [Top-K benchmark procedure](../Benchmarks/TopKNaNContract/README.md)
  and [results](../Benchmarks/TopKNaNContract/RESULTS.md)
- [Benchmark baseline storage](../Benchmarks/baselines/README.md)

Each record's commit, hardware, toolchain, selected cases, skips, and stated
limitations bound its claims. Current jobs are defined in
[CI](../.github/workflows/ci.yml); inspect run results for the revision of
interest. A compile check is not a runtime device test.

## Historical designs and plans

Historical documents remain at their original paths to preserve source,
guide, and external links. Their APIs, dates, checklists, and architectural
assertions may be superseded. They are not installation instructions or an
active backlog.

- [Early Vector(N) evolution](Vector_N_evolution_1.md),
  [early future roadmap](future_roadmap.md), and
  [March 2026 implementation plan](vc_implementatino_plan_mar26.md)
- [Beta evolution 2](beta-evolution-2/): optimization, build, concurrency,
  API, benchmark, and C-kernel proposals
- [Beta evolution 3](beta-evolution-3/): design documents, master plan,
  and historical test triage
- [Beta evolution 4](beta-evolution-4/): matrix distance, quantization,
  provider/buffer interfaces, and sibling integration proposals
- [Semantic-search gap analysis](gap-analysis-hn-semantic-search.md)
  and [its implementation plan](gap-analysis-hn-semantic-search-PLAN.md)
- [Copied index kernel specifications](../VectorIndex/kernel-specs/README.md)
  (not a VectorIndex implementation in this package)
- [GitHub readiness review, September 6](GitHub_Readiness_Review_2026-09-06.md)
  and [hardening implementation plan](GitHub_Hardening_Implementation_Plan.md);
  read the later [acceptance record](verification-public-readiness-2026-09-07.md)
  for the verified follow-up
- [Documentation-refresh work plan](superpowers/plans/2026-09-08-documentation-refresh.md)

Use [Package Boundaries](Package_Boundaries.md) and the
[current roadmap](ROADMAP.md) when deciding what belongs in Core today.
