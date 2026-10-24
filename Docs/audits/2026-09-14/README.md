# VectorCore validated audit and improvement guide

**Reviewed revision:** `8b02e75bca7055ae11170873d15c06f1538709b7` (`main`, version 0.3.3).  
**Audit date:** September 14, 2026, America/Los_Angeles.  
**Purpose:** guide the next improvement round in architecture, code quality and performance. This is a review artifact; implementation fixes have not been made.

## Assessment

The main concerns from the [initial twelve-component review](../../reviews/2026-09-14/README.md) survive validation. The strongest reasons to prioritize another hardening round are public storage invariants, compact-storage copy isolation, stale normalization/cache metadata, metric substitution, unchecked aggregation, and numerical failures on finite inputs. These have direct consequences for callers of a foundational numerical library.

Performance work should begin with ownership and measurement reliability. There is evidence of unnecessary full-buffer copying in actual Release code, and there are benchmark paths that can succeed without selecting or validating the requested work. Larger-scale dispatch thresholds need trustworthy, representative measurements after correctness is established.

This document supersedes the earlier summary for prioritization and wording. The original review remains intact as the historical record. A passing test suite supports the exercised cases; it does not close the reproduced gaps below. Proposed remedies are design recommendations with acceptance criteria, not already-validated fixes.

The register contains **131 adjudicated records** covering all 46 original canonical labels and their supplemental observations: 57 reproduced, 42 established by source, 28 qualified, three unverified and one rejected inference. Records overlap and include context/design opportunities; these totals are not independent defect counts.

## How to use this audit

- [Claim register](CLAIMS.md): each actionable principal and supplemental claim, its verdict, exposure, source, evidence, correction and completion criteria.
- [Machine-readable register](claims.json): the same adjudicated records for tracking and checking coverage.
- [Original-ID crosswalk](CROSSWALK.md): maps every A1–A10, C1–C20, PERF1–PERF9 and B1–B7 reference to its current disposition. These labels overlap; they are not 46 independent defects.
- [Validation record and replay instructions](evidence/README.md): retained probes, observations, provenance and limitations.
- [Ownership validation](validation/ownership.md), [numerical validation](validation/numerics.md), [execution/API validation](validation/execution-api.md): detailed independent second-pass analysis and component-subsection crosswalks. The root reviewer owns the final register and priorities.

### Verdict and priority conventions

| Verdict | Meaning |
|---|---|
| Confirmed—reproduced | A focused executable or mechanical check demonstrated the stated observation. The evidence distinguishes real-library clients from instrumented fixtures. |
| Confirmed—source | Current source and reachable control flow establish the claim; no runtime manifestation is implied. |
| Qualified | A narrower claim is supported, or a documented limitation/design opportunity needs a decision before being treated as a defect. |
| Rejected | The stated inference is not supported; do not create a fix task for it. |
| Unverified | Evidence is insufficient to decide the potential failure. Investigate before implementation. |

**P1:** address before depending on the affected API at scale; includes validation infrastructure that gates subsequent changes. **P2:** meaningful correctness, reliability or scale work. **P3:** lower-exposure boundaries, documentation, dormant paths or measured-optimization candidates. Priority reflects impact and exposure, not a claim that a production incident has occurred. Internal or gated findings become prerequisites if a new route would expose them.

## Important corrections from the second pass

1. **Triangular matrix copy cost begins at public size 33, not only 65.** Standard-tier `ContiguousArray` extraction also causes a full detach on each scalar write. Compact copy isolation and generic element destruction are separate defects. Fixing the aligned tier alone would leave the standard-tier cost intact. See OWN-05, OWN-10, OWN-13 and OWN-14.
2. **HybridStorage's quadratic traffic has evidence beyond an instrumented fixture.** Actual linked Release instructions retain an owner immediately before uniqueness checks and then allocate/copy. The exact 1,026 allocations at dimension 512 belong to the counter fixture; the production instruction path supports the copy mechanism and complexity. An unchanged pointer address did not disprove copying because intermediate allocations can recycle addresses. Do not generalize this to default generic `DimensionStorage`. See OWN-12 and OWN-26.
3. **The alignment fixture crashes in Debug; the tested Release ARM fixtures succeed.** The source contract still requires attention. No Release crash, sanitizer finding or 384D optimizer miscompilation is claimed. Scoped rebinding alone is not proof of correct alignment. See OWN-08 and OWN-09.
4. **FP16 shape validation is not the INT8 shape defect.** FP16 values have immutable, length-checked storage. INT8 raw construction, mutation and synthesized decoding admit wrong lengths. FP16's escaped pointer lifetime is a separate unsafe API question. See OWN-01, OWN-02 and OWN-17.
5. **Hashing more coordinates does not fix cache identity by itself.** The 512D cache falsely hits for 496 independent interior-coordinate changes. A digest still needs equality verification or a reliable identity/version contract. Legal hash collisions are a performance concern; treating a hash as exact identity is a correctness defect. See OWN-04 and OWN-07.
6. **Replacing reciprocal multiplication with Accelerate division is not a validated fix.** The fresh host probe also gets `[Inf, NaN]` from Accelerate division for `[s, 0] / s`, `s = 1e-40`. Normalization must preserve finite direction without requiring the reciprocal or full norm to fit in Float. Pointer normalization also produces all zeros for maximum-finite equal components at tested dimensions 4 and above. See the numerical C3 records.
7. **FP16 diagnostics need rounding-aware classification.** `65505` converts to finite `65504` while the diagnostic counts overflow; `1e-12` rounds to zero but is counted as subnormal. Swapping two branches alone is insufficient. See the numerical diagnostics record.
8. **GEMM cancellation, full-matrix scratch and several unsafe caller obligations are already documented.** They remain important workload/architecture decisions. They should not be described as undisclosed corruption or defects caused by valid use of an explicitly unchecked contract. Asymmetric pairwise metrics similarly need a contract decision. See the execution/API and numerical qualifications.
9. **Missing result consumption is not proof of measured dead-code elimination.** The timing-scope inconsistencies are source-confirmed; their speed impact is unmeasured. Binary scalar-load allocation is also a profiling candidate, not an established per-scalar Release heap allocation. See TOOL-07 and SER-03.

## 1. Architecture

### Establish enforceable public contracts

**Storage and metadata ownership:** fixed-dimension vectors must retain their exact count through every public mutation and decoding path. Runtime dimension must stay consistent with storage. A normalization hint must either own an immutable value or invalidate after every mutation, including exceptional exits. Cache identity must distinguish changed values independently of incidental hash collisions. These are the first architecture decisions because later numerical checks assume trustworthy shape and metadata.

**Metric and result semantics:** facade search must honor each supported metric or reject it explicitly. Define negative-dot score conversion, metric-specific diagonals and similarity sign. Collection replacement must preserve existing data when validation fails; generated IDs must not overwrite explicit IDs. The reference collection is distinct from VectorIndex's actual storage engine.

**Execution and configuration:** take one validated live configuration snapshot per operation and mutate configuration atomically. Enforce outstanding-work bounds in shared defaults; a provider's declared concurrency limit must have an observable meaning. Define cancellation and nested execution behavior. Enabling ignored settings can expose currently dormant arithmetic, chunking and storage defects—validate those paths first.

**Resource and extension boundaries:** make pool allocation ownership and byte budgets explicit. Publish which provider/configuration hooks are effective and which protocol methods are customization points. Drive any contiguous matrix interface from actual VectorIndex/VectorAccelerate call sites and their ownership needs. Preserve existing SoA layout/transfer contracts while making changes.

### Decisions to settle before choosing implementations

| Decision | Why it matters | Evidence needed for closure |
|---|---|---|
| Checked versus unchecked shape/mutation APIs | Unsafe kernels cannot recover from fabricated buffer lengths. | All constructors, Codable paths, setters and generic wrappers enforce the same chosen invariant. |
| Hint/cache identity and invalidation | Stale values can alter downstream distance evaluation. | Valid-to-stale mutation tests, throwing exits, snapshot isolation and collision controls. |
| Exact versus approximate distance routing | GEMM distance cancellation can change nearest-neighbor membership. | Explicit policy and route tests on near-equal large-offset vectors, with direct-reference comparison. |
| Bounded memory and outstanding work | Full Q×N scratch, per-worker heaps and count-only caches scale differently. | Analytical live-buffer model plus measured peak/retained bytes at the intended workload. |
| Public extension points | Extension-only methods can select different defaults in generic code. | External concrete/generic/provider consumers demonstrate the promised route and behavior. |
| Sibling compatibility | Separate package releases do not form an automatic compatibility matrix. | Explicit supported revisions, consumer builds and selected end-to-end tests before release claims. |

## 2. Code quality

The immediate quality work is behavioral: array bounds and shape checks, copy isolation, element destruction, atomic updates, finite-aware numerical policies and truthful error/reporting paths. Cosmetic changes should follow the established project conventions and avoid introducing extra hot-path work without reason.

### Numerical acceptance should cover operation families

Use finite exponent sweeps and independent references across generic, optimized, dynamic, pointer, provider, batch and reduced-precision routes. Include zero, subnormal, normal and large finite values; explicit NaN/Infinity policy; route boundaries; ragged and empty batches; and output capacity beyond the logical count.

For normalization, assert direction and unit norm when the documented domain permits them. For QR/SVD, assert residuals, orthogonality, spectra and finiteness. For variance/PCA, assert translation and scale behavior using representable input values and appropriately wider references. For entropy and quantized cosine, assert the defining scale invariances. For UMAP calibration, check the defining local equations and convergence under changes of units; a plausible-looking projection is insufficient.

### Make verification claims auditable

Replace comment-only tests and mislabeled concrete-type coverage with real assertions or clearly separate future-work notes. Repair benchmark correctness before using it to approve numerical/performance changes. Tests must validate the provider/kernel route that actually ran. Exact bit-pattern and ordering checks belong where identity or wire representation matters; approximate numerical checks need explicit finite/tolerance handling.

Every robustness/performance comment should name evidence, follow directly from adjacent code, or be an explicit unverified contract. A passing test count does not justify comments such as “all fixed,” universal speedups or “works for any dimension.” Apply the requested claim discipline to future changes and to review descriptions.

## 3. Performance

### Prioritize demonstrated work amplification

1. Repair HybridStorage and triangular scalar-write ownership before tuning arithmetic kernels. Verify unique writes avoid full copies while genuinely shared values still detach correctly.
2. Remove whole-candidate FP16 mapping from each small range call. The requested range should bound work; test counts/allocations as well as elapsed time.
3. Bound search scratch, chunk scheduling, per-worker heaps and cache retention by explicit workload/budget policy. For Q queries and N candidates, an FP32 Q×N matrix alone uses `4QN` bytes: Q=10,000 and N=1,000,000 is 40 GB, before packed inputs and other buffers. This is a sizing calculation, not a measured allocation.
4. Consolidate streaming selection and reusable matrix packing where callers benefit. Full sorting, temporary tuples, duplicate validation and flat-to-nested matrix copies are real source-level work; crossover decisions still need representative Release measurements.

### Repair measurement before publishing improvements

The benchmark must fail invalid numerical output, missing required validation, unintended zero-case selection and requested artifact-write errors. Record effective build/kernel configuration. Pair A/B cases correctly. Identify setup, packing, allocations and output consumption consistently. Distinguish prepacked kernel throughput from end-to-end operation cost.

Measure Release builds with warmups, raw samples, medians/variation, allocation counts and peak/retained memory. Record hardware, toolchain, compile flags and workload. Current shared-harness p90 summarizes per-sample average time; it is not an individual-operation latency percentile. Historical 2025 baselines and this audit's observation programs do not establish current throughput guarantees.

## Recommended improvement work packages

These are dependency-ordered scopes for the next round, not implementation authorization or promised separate PRs. Each package should reference the relevant register records and close the whole repeated defect class.

| Order | Work package | Required closure |
|---|---|---|
| 0 | **Trustworthy validation** — TOOL-01 through TOOL-06, TOOL-11, SER-02 | Strict correctness rejects invalid/missing results; selected case counts and artifact exits are checked; effective build metadata and A/B pairing are tested. Establish reusable finite-aware and route-sentinel fixtures. |
| 1 | **Public value and identity contracts** — A1–A4, C1/C2/C4/C5/C17, C12, OWN-10 | Constructors/mutation/decoding preserve shape; copies isolate writes; generic elements release; unsafe alignment/binding/lifetime contracts are explicit; hints/cache invalidate correctly; metrics and aggregation validate before unsafe work. Test all widths/storage tiers and generic wrappers. |
| 2 | **Execution, collections and resource ownership** — A5–A9, C10/C11/C15/C18 | Atomic config and replacement; bounded peak active/pending work; pool teardown and byte limits; logical output counts; correct metric/result semantics; validation before arithmetic/stride/indexing. |
| 3 | **Range-safe numerical contracts** — C3/C6–C9/C13/C14/C16/C19 and PCA/diagnostics supplements | Independent equation/reference checks across exponent and route boundaries; documented failure/nonfinite/accuracy behavior. Do not assume a library primitive replacement fixes the reproduced case. |
| 4 | **Storage and batch work reduction** — PERF1/PERF3–PERF6, OWN-14 | Release allocation/instruction evidence and shared-copy controls; range work independent of unrelated candidates; correct throw/writeback behavior before adopting bulk paths. No dormant route enabled without reachability review. |
| 5 | **Scale interfaces and memory policy** — A10, PERF2/PERF7/PERF8 | Real consumer-backed matrix views/reuse, exactness policy, bytes/work bounds, cancellation and supported sibling revision tests. Measure representative end-to-end workloads. |
| 6 | **Contract/documentation cleanup and deferred investigation** — C20, SER-03–SER-07, remaining P3/qualified/unverified records | Resolve wire/API contracts, preserve compatibility, remove unsupported claims, and profile candidates before promising gains. Promote dormant paths only after their prerequisites pass. |

Packages 0–3 can overlap where they do not share code, but changes to the measurement gate, public invariants or route configuration must be reviewed before dependent optimization work. A register entry is closed only when its acceptance criterion is met and evidence is linked; changing code or passing unrelated tests is insufficient.

## Validation coverage and limits

The original review covered twelve components: vectors; memory/storage; execution/concurrency; scalar/SIMD/C kernels; search/Top-K; batch/matrix operations; reduced precision; serialization; linear algebra/PCA; UMAP; downstream interfaces; and validation/benchmark/CI tooling. Three independent second-pass reviewers rechecked the ownership, numerical and execution/API groups while the root reviewer validated tooling/serialization, challenged evidence, resolved overlap and made final decisions.

Fresh second-pass checks include a Release baseline (**1,144 Swift Testing tests in 168 suites; 91 XCTest cases with 27 skipped; no reported failures**), 25 Python CI tests and the nine-file GitHub YAML/policy validator. Focused actual-library Release clients, isolated Debug/Release boundary subprocesses, exact-source tooling/pool fixtures, source arithmetic and Release disassembly supply the finding-specific evidence. Observation clients deliberately print bad outcomes and may exit zero; they are not passing correctness tests.

The earlier same-revision Debug run, seven standalone documentation consumers and pinned SwiftLint results remain historical first-pass evidence, not fresh second-pass executions. No new sibling end-to-end build, GPU run, Intel runtime run, sanitizer campaign, sustained million-vector benchmark, thermal-controlled speed comparison or peak-RSS campaign was performed. Source-derived task counts and byte models remain estimates; hypothetical allocator failure and mobile benchmarking-product availability remain unverified.

Downstream source inspection used VectorIndex `6f114923d228739c207734685cfde18dd1c6cf60` and VectorAccelerate `3cd1442d4f4e092dda4f9bcd5e0878a9f8cad8bb`. These traces establish potential consumer exposure, not executed GPU effects or deployed-state corruption. The [package boundaries](../../Package_Boundaries.md) already describe the absence of an exhaustive cross-package compatibility matrix.

## Completion checklist for a follow-up change

- [ ] Identify register records, affected public routes and compatibility decisions.
- [ ] Add a failing regression that demonstrates the current class, including its second occurrence/variants.
- [ ] Implement the chosen behavior and document failure, ownership, numeric and complexity contracts.
- [ ] Re-derive routes exposed by any configuration/gate change; include dormant prerequisites.
- [ ] Run appropriate Debug/Release, external-consumer and sanitizer checks for the changed surface.
- [ ] For performance claims, retain equivalent-workload measurements and allocation evidence.
- [ ] Update the register with the fixing revision, test names and evidence; retain the original observation.
