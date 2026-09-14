# Documentation Refresh Implementation Plan

> **For agentic workers:** Use the planning and verification workflow for this documentation-only branch. The user authorized implementation in this session; no separate plan approval is needed.

**Goal:** Make VectorCore's major documentation accurate against current source and distinguish current references from historical proposals.

**Architecture:** Keep existing reference URLs where practical. Add a documentation index and focused numerical/projection references, replace outdated API and performance descriptions, and label retained historical records. Verify complete Swift examples as downstream consumers.

**Tech Stack:** Markdown, Swift Package Manager, existing Python CI tooling.

**Spec:** User request of September 8, 2026: create a branch, address major documentation, remove/add files where useful, and save all Guides for a separate pass.

## Global constraints

- Work on `docs/documentation-refresh`, based on `origin/main` at `f562270`.
- Do not modify `Guides/`, `Docs/Performance_Guide.md`, or `Docs/HowTo_NearestNeighbor.md`; all tutorials are deferred to the guide pass unless the user requests otherwise. Do not change sibling packages, runtime sources, public APIs, or package versions.
- Preserve unrelated untracked `.antigravitycli/` and `GEMINI.md`.
- Retain dated verification evidence; do not turn old measurements into current guarantees.
- Tie correctness claims to source or named tests. Separate deployment targets, compiled platforms, and executed tests.
- User authorization covers documentation deletion, but retaining historical paths avoids breaking external and Guides links.

## Tasks

- [x] Audit root/community docs, current `Docs/` references, historical plans, and benchmark instructions against source and existing CI.
- [x] Update README, API overview, package boundaries, and roadmap; add `Docs/README.md` to identify current and historical documentation.
- [x] Document actual routing and scoped provider binding in the API reference. Defer the performance and nearest-neighbor tutorials to the guide pass.
- [x] Reconcile memory alignment and SoA ownership/layout docs with implementations; add numerical behavior and projection references.
- [x] Label historical proposals in place and explain their status in the documentation index; retain verification records and benchmark data.
- [x] Expand the existing downstream consumer check to compile and run current documentation examples; check local links and run relevant repository-policy checks.
- [x] Review the full diff, verify `Guides/` and runtime sources are unchanged, and hand off the completed branch.

## Verification

Run `python3 Scripts/ci/consumer_smoke.py`, the relevant `Scripts/ci/tests` suite, `ruby Scripts/ci/validate_github.rb`, and `git diff --check`. Inspect active-document links and source references; historical links may refer to the original implementation state and must be marked accordingly. No full runtime test rerun is needed unless runtime source changes, which is outside this pass.

## Results

- Seven independent downstream Swift programs compiled and ran in Release. The
  first build caught an overcomplicated UMAP example expression; explicit
  intermediate types resolved the compiler inference failure.
- Added five example-selection tests. Against the old script, four new
  behaviors failed, while its existing single-Quick-Start guard passed.
  All five pass with the expanded collector.
- All 25 repository-tooling tests pass; the Ruby validator accepts nine
  GitHub YAML/template files and their policies.
- Checked 161 local file/heading links across current references and community
  docs; none were missing. `git diff --check` passed.
- Confirmed no differences in runtime sources/tests, package manifest/resolution,
  deferred guides, dated verification records, or benchmark data.
- Existing Swift warnings about deprecated CBLAS imports and an unused closure
  result were observed; no source changes were made to silence them.
- Historical files retain their contents and paths, with status banners added.
  No documentation files were deleted.
- Independent read-only review found no critical or important issues. Its
  minor CI step-label correction was applied; job IDs and merge gates did
  not change.
- At the September 8 handoff, changes were left local and uncommitted on
  `docs/documentation-refresh`; no remote branch or pull request was created.
  On September 13, the owner requested publication as a pull request.
