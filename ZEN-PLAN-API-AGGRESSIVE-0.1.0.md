# VectorCore v0.1.0 — Aggressive API Consolidation Plan (No BWC)

Goal: Ship a clean, minimal, and consistent public API for VectorCore v0.1.0 with zero migration or deprecation paths. Remove redundant facades, standardize return types and naming, and simplify error and operator semantics while preserving performance philosophy.

## Guiding Principles

- One public facade for operations: `Operations` (provider‑based).
- Validate at API boundaries; keep hot paths free of redundant checks.
- Prefer throwing APIs for public entry points; keep operators fast with preconditions.
- Eliminate duplicate or confusing surfaces; no deprecations, remove outright.
- Keep Float‑only for now; document clearly.

## Public Surface: Keep vs Remove

Keep (public):
- Core vectors: `Vector<D>`, `DynamicVector`, optimized vectors (e.g., `Vector512Optimized`, `Vector768Optimized`, `Vector1536Optimized`).
- Protocols: `VectorProtocol` (trimmed), `VectorFactory`, `DistanceMetric`, provider protocols (`ComputeProvider`, `ArraySIMDProvider`, `BufferProvider`).
- Facade: `Operations` (provider‑based API only).
- Utilities: `SyncBatchOperations` (synchronous helpers only, small surface).
- Errors: `VectorError` (throwing error type), supporting error context.
- Distance metrics: `EuclideanDistance`, `CosineDistance`, `ManhattanDistance`, `DotProductDistance`, `ChebyshevDistance`.

Remove or Internalize (no BWC):
- `ExecutionOperations`, `BatchOperations` (duplicate or overlapping with `Operations`).
- `VectorCore` namespace conveniences (replace with README guidance and `VectorTypeFactory`).
- Execution contexts (`CPUContext`, `ExecutionContext`, GPU placeholders) from public surface; keep internal or behind providers.
- Helper heaps/validation types as internal (`KNearestHeap`, validation helpers).
- Public logging (`Logger`) unless explicitly needed; otherwise internal.

## API Changes (Breaking, Immediate)

Standardize return types and names:
- Nearest neighbors: use `[NearestNeighborResult]` across all operations.
- Keep names: `findNearest`, `findNearestBatch`, `distanceMatrix`, `normalize`, `statistics` — all on `Operations` only.
- Remove tuple return variants `[(index: Int, distance: Float)]` from public APIs.

Error handling:
- Public entry points throw `VectorError` for invalid input (dimension mismatch, invalid k, empty collections).
- Add `normalizedThrowing() -> Self` to `VectorProtocol` (throws on zero magnitude). Remove `normalized() -> Result<..., VectorError>` from public surface.
- Keep operator preconditions for hot paths (documented “fast path”).

Operators & allocation:
- Remove duplicate operator implementations from `VectorProtocol`. Keep a single, safe generic implementation via `VectorFactory` (already present), ensuring correct allocation for dynamic vectors.
- Ensure `VectorProtocol` does not expose `random(in:)` (dimension ambiguous for dynamic vectors). Keep `Vector.random(in:)` (static‑dim) and `DynamicVector.random(dimension:in:)`.

Factories:
- Keep `VectorTypeFactory` for dimension‑based creation. Optionally rename to `VectorBuilder` for clarity; update all references in code and README.
- Ensure centroid, map/reduce utilities use `VectorFactory` for allocation.

Documentation:
- README reflects the single public facade (`Operations`), Float‑only scope, error philosophy, performance notes, and basic usage.
- Add short API reference (DocC optional for later minor).

## Implementation Checklist

1) Public Facade Consolidation
- [x] Move all public operation entry points to `Operations`.
- [x] Remove `ExecutionOperations` and `BatchOperations` from public (delete or make `internal`).
- [ ] Ensure `SyncBatchOperations` surface is minimal and documents intended use.

2) Return Type Unification
- [x] Change any public nearest‑neighbor API to return `[NearestNeighborResult]`.
- [x] Update internal call sites accordingly.
- [x] Keep `NearestNeighborResult` as the single result struct (index, distance).

3) Error Semantics
- [x] Ensure `Operations` validates input and throws `VectorError` (no `precondition` for public entry points).
- [x] Add `normalizedThrowing()` to `VectorProtocol`; remove `Result`‑based normalized from public.
- [x] Keep operators using `precondition` for hot path consistency.

4) Operators & Allocation
- [x] Remove/privatize `VectorProtocol` operator implementations; rely on `VectorFactory` generic operators.
- [x] Confirm all operations allocate result with correct length for dynamic vectors.
- [x] Keep element‑wise `.*` and `./` (document in README).

5) Factories & Utilities
- [x] Remove `VectorProtocol.random(in:)` (dimension ambiguity). Keep `Vector.random(in:)` and `DynamicVector.random(dimension:in:)`.
- [x] Update `Operations.centroid` to use `VectorFactory.create(from:)` instead of `ExpressibleByArrayLiteral`.
- [ ] Decide on renaming `VectorTypeFactory` → `VectorBuilder` (perform rename and adjust imports/docs if chosen).

6) Providers & Contexts
- [x] Keep provider pattern (`ComputeProvider`, `ArraySIMDProvider`) public; make execution contexts (CPU/GPU) internal implementation details.
- [x] Ensure GPU placeholders are not public.

7) Cleanup & Internals
- [x] Mark helper types (`KNearestHeap`, validation helpers) internal.
- [x] Ensure logging is internal unless required as a public API.

8) Tests
- [ ] Update tests to use `Operations` facade exclusively.
- [ ] Remove/replace tests that import `ExecutionOperations`/`BatchOperations`.
- [ ] Keep sync baseline tests via `SyncBatchOperations` where useful.
- [ ] Validate parallel correctness via `Operations`.

9) Docs & Release
- [ ] Update README to match consolidated API, examples, platform support.
- [ ] Add `CHANGELOG.md` with 0.1.0 (initial public release, breaking changes note irrelevant as no prior release).
- [ ] Tag `v0.1.0` after CI green.

## Acceptance Criteria

- Only `Operations` and `SyncBatchOperations` expose operations publicly; no `ExecutionOperations`/`BatchOperations` in public API.
- All nearest‑neighbor APIs return `[NearestNeighborResult]`.
- `VectorProtocol` has `normalizedThrowing()`; no public `Result`‑based normalized.
- No `VectorProtocol.random(in:)` remains; random APIs are dimension‑explicit.
- Operators function for both static and dynamic vectors (no buffer overruns); comprehensive tests pass.
- README aligns with the final public surface; CI passes on supported platforms.

## Risks & Mitigations

- API churn in a single sweep: mitigated by comprehensive tests and clear README.
- Hidden dependencies on removed types: search and refactor tests and internal uses thoroughly.
- Performance regressions due to refactors: keep hot paths unchanged; validate via perf smoke tests later.

## Work Breakdown (Suggested Order)

1. Remove/privatize `ExecutionOperations`, `BatchOperations`, execution contexts.
2. Unify return types to `[NearestNeighborResult]` and adapt internals.
3. Switch public callers/tests to `Operations` only.
4. Replace `normalized()` with `normalizedThrowing()` in public usage; update call sites.
5. Remove duplicate `VectorProtocol` operators; ensure `VectorFactory` operators cover all cases.
6. Remove `VectorProtocol.random(in:)`; verify callers use correct APIs.
7. Adjust `Operations.centroid` allocation via `VectorFactory`.
8. Finalize docs (`README`, `CHANGELOG`), run CI, tag v0.1.0.

---

This plan is intentionally aggressive with no deprecation or compatibility shims, suitable for an unreleased 0.1.0.
