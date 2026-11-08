# SIMDStorage by-value mutation migration

> For agentic workers: implement the authorized migration in the reviewed worktree, using test-driven development and independent review. Retain RED/GREEN evidence and keep compiler qualification separate.

**Goal:** Replace the public SIMDStorage descriptor-inout requirement with an explicitly named by-value mutation boundary for the forthcoming 0.4.0 release.

**Architecture:** `SIMDStorage` requires `mutating func withUnsafeMutableElements<R>(_ body: (UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R`. ContiguousArray implements this through its standard-library borrow, passing the descriptor to the callback by value. There is no default bridge or deprecated inout requirement; existing conformers and generic callers migrate explicitly.

**Tech stack:** Swift 6, native SwiftPM, existing Python consumer/evidence tooling.

**Spec and authorization:** The user requested the recommended distinct-name API migration after the [completed compatibility audit](../2026-11-06-workaround-audit/README.md). The exact method name is an implementation decision. The source break follows that migration; it is documented under Unreleased and in a dedicated migration guide.

## Constraints and review focus

- Continue at `/Users/goftin/dev/projects/VectorCore-fixes-2026-10-27-84798d3`; preserve the earlier uncommitted audit work and all archived evidence.
- Keep the unchanged compiler reproducer, negative controls, known-issue handling and release-admission policy.
- Preserve generic witness dispatch, `rethrows`, callback return values, empty-buffer invocation, element type, count, COW isolation and mutations before a thrown error.
- Do not add a same-name by-value overload, which the compatibility audit showed can cause ambiguity.
- Old-only conformers and old generic callers must fail to compile; a new-only conformer and ordinary concrete stdlib closure must compile.
- Concrete ContiguousArray storage still exposes the stdlib inout API; this migration does not certify affected compilers or arbitrary caller pointer handling.
- No performance acceptance, support expansion, release branch/tag, merge, push or publication.

## Task 1: Public requirement and behavioral coverage

Files: `Sources/VectorCore/Protocols/SIMDStorage.swift`, new `Tests/ComprehensiveTests/SIMDStorageMutationTests.swift`.

- [x] Add tests for direct and generic mutation, an independently implemented external-style conformer, empty/nonempty callback count and return, Float/Double/SIMD elements, COW/tail preservation, mutation followed by the original error, and generic OptimizedVector storage.
- [x] Capture expected missing-new-member/conformance RED against the old actual module before editing production.
- [x] Replace the requirement and add the explicit public inlinable ContiguousArray implementation; document the bounded borrow contract and tested behaviors.
- [x] Run targeted new tests in Debug and Release, then the normally enabled suites against fresh source-bound products.

## Task 2: Downstream contracts and mechanical guard

Files: `Scripts/ci/consumer_buffer_value.py`, its existing test/fixture files, new compile-contract fixtures, `Scripts/ci/tests/test_pointer_descriptor_inout.py`; any directly affected D02 protocol-inventory policy.

- [x] Remove the guard's production exception, retain rejection controls for the old declaration and all existing alias/constraint classes, and capture live-tree RED before migration.
- [x] Migrate all relevant consumer calls and the external witness module to the new requirement; retain the 24 independent numerical rows and complete omitted-write oracle.
- [x] Add successful new-conformer/concrete-closure compile contracts and expected failures for old conformers/generic callers. Require the specific expected diagnostic, not arbitrary failure.
- [x] Update retained SIL checks to the new witness member. Keep concrete stdlib exposure as a separate two-row numerical diagnostic.
- [x] Execute both consumer optimization modes against fresh producer products; preserve every failure, exact input/artifact identity and producer binding.
- [x] Review protocol inventory/release-claim reachability after the requirement change. Run final Python discovery and workflow validation after all policies are stable.

## Task 3: Migration guidance and final review

Files: `Docs/SIMDStorage_Migration.md`, `Docs/README.md`, `CHANGELOG.md`, and this review directory's final report/results.

- [x] Show old/new generic calls and conformer implementations, pointer lifetime obligations, `&` removal for element-only writers, and source/binary rebuild requirements.
- [x] Explain why concrete stdlib methods and independent compiler failures remain separate; link test evidence without making universal optimizer-safety claims.
- [x] Obtain independent review, resolve material findings, verify retained artifact hashes and summarize remaining release gates.

## Execution ledger

- Starting HEAD: `bdf990847babf567be1cbce22d2606e7102b1a47`, with the completed workaround audit's uncommitted tooling and reports present.
- Baseline: prior audit's six native suites and 624 Python tests are retained. Xcode Release has the known unchanged compiler-canary failure; the official Swift.org control passes. This authorized migration proceeds with that baseline documented.
- Design review found no method-name collision or blocker. It confirms a real source/binary migration is required and that callers could still copy a by-value descriptor and pass the copy inout themselves.
- Ruling: replace the requirement rather than add a fallback — a fallback would preserve the old protocol boundary. Existing conformers must explicitly adopt the new contract; migration guidance and compile contracts make this visible.
- Parent owns production code, documentation and native execution. Astra agents independently own external fixtures, descriptor guard changes and downstream policy review, with separate file ownership.

- Completed implementation: the new named by-value requirement and explicit ContiguousArray adapter replace the descriptor-inout boundary. No legacy fallback or same-name overload remains in SIMDStorage.
- All seven new API tests pass in Debug and Release under Xcode 27.2, Xcode 27.3 beta and official Swift.org 6.4.2. Each full suite executes 1,947 tests with 46 skips. The only recorded issues are the original compiler canary in the two Xcode Release suites; compiler policy remains unchanged.
- Separate-module validation passes all 288 positive numerical observations, 12 full omitted-write executions and 36 migration compile contracts; six producer/product bindings verify. The ordinary consumer smoke entry point passes eight examples and all 24 contracts.
- Final Python discovery passes 628 tests; workflow and whitespace checks pass. The initial stale generated downstream appendix failure is retained alongside the corrected final run.
- Public declaration comparison finds only the old requirement removed and the new requirement plus ContiguousArray adapter added. No other compared declaration changes.
- Independent Astra code review reported no material findings. A follow-up evidence review's two hash/source-check omissions were fixed and confirmed; consolidation successfully reverified the added inventories and compiler/SDK bindings.
- Compiled source inputs remain unchanged across validation. The final report links the source-bound summary and selected evidence inventory; the earlier audit snapshot is preserved separately.
