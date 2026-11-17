# VectorCore 0.4.0 remaining-gate investigation plan

> **For agentic workers:** Use the dispatched parallel investigation assignments; implementation changes require the applicable debugging and verification workflow. This continues the approved phase-two follow-up rather than introducing a new product design.

**Goal:** Complete the available compiler, platform/coverage and performance investigations, preserve their evidence, and identify the exact remaining release decisions or unavailable infrastructure.

**Architecture:** Three Astra agents own separate evidence directories. The coordinator verifies their reports, resolves cross-track prerequisites and grants the exclusive measurement window. The shipping code and support policy remain those of the carried candidate unless evidence justifies a separately reviewed change.

**Tech stack:** Swift/Xcode, SwiftPM, Apple SDK/runtime tools, existing Python qualification tooling and strict benchmark comparator.

**Spec:** User's three-track follow-up and `Docs/reviews/2026-11-04-release-phase-two/README.md` at `27a045dd648c01a4ef9c5d8d36c5afc92f36f6fb`.

## Constraints and ownership

- Preserve all previous evidence, commits, worktrees and failed outcomes. Do not merge main, publish, tag, bump versions or narrow support implicitly.
- Use the clean managed worktree `/Users/goftin/.codex/worktrees/0bd4/VectorCore` for integration; the carried branch remains untouched while investigations run.
- Compiler agent owns `compiler/`; platform agent owns `platform/`; performance agent owns `performance/` under `/private/tmp/vectorcore-release-gates-2026-11-05`. Coordinator owns `coordinator/` and consolidated records.
- Preserve the unchanged numerical reproducer, Release optimization, all correctness/comparison thresholds and compatibility requirements.
- No toolchain/runtime installation, global Xcode selection, device provisioning or termination of user processes. Cap compiler jobs at two per worker.
- Never read, hash or archive the earlier excluded raw environment diagnostic. Capture only selected relevant environment values.
- Compilation is not runtime acceptance; skips are not passing bodies; standalone Swift.org qualification is not Xcode/App Store qualification; point-comparison success alone is not controlled-host evidence.

## Review focus

1. Verify compiler identity and the actual numerical output, including known-issue cases hidden by aggregate CI success.
2. Trace source and binary boundaries for source SwiftPM, client specialization and any artifact-based support proposal.
3. Enumerate every skipped body, its gate and substitute coverage; identify ordinary numerical paths hidden by host-only conditions.
4. Keep deployment floors, simulator/device architecture and sanitizer/leak scope distinct.
5. Preserve all timing repeats and quantify fixed-order/between-process drift without selecting a favorable run or loosening the gate.

## Work and acceptance

- [x] Establish clean carried commit and create an isolated integration worktree; snapshot all tracked source hashes.
- [x] Dispatch exactly three Astra agents with exclusive path ownership and explicit no-timing-yet rule.
- [x] Compiler: compare official available candidates and local exact binaries, run justified unchanged controls, and provide qualified path or concrete support options. No suitable corrected Xcode qualified; broad support intent retained.
- [x] Platform: derive policy/evidence matrix, classify skips, and execute valuable available missing checks with exact source/tool/runtime identity. Catalyst executed with the canary failure; physical action rejected tool-hosted tests before any body.
- [x] Performance: inspect all retained samples and host observability; prepare chosen native compiler products; wait for explicit exclusive window. One formal readiness observation rejected the busy host; no new timings.
- [x] Coordinator: approve diagnostic/acceptance scope based on compiler results, confirm build/test quiescence, then grant measurement window if host readiness permits. Granted exclusive readiness window; measurement admission failed.
- [x] Independently review all three reports and source/binary associations, resolve findings and perform focused verification appropriate to actual changes. Scoped cross-reviews and five focused integration checks passed; 15 lost-store scanner tests passed.
- [x] Freeze bounded evidence and integrate a consolidated report with exact remaining gates; 1,458 selected payload files (123,894,870 bytes) copied to the durable archive and reread successfully. All required release gates have not closed; release acceptance remains false.
