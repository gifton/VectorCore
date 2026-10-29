# F007 Progress Percentage Repair Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans; continue without routine approvals.

**Goal:** Give every accepted `OperationProgress` a nontrapping integer percentage.

**Architecture:** Preserve the current floating-point fraction and truncation. Saturate only the percentage conversion at `Int.max`; retain progress above 100%, input clamping and completion semantics.

**Tech Stack:** Swift, Swift Testing, existing package and Python CI checks.

**Spec:** User-authorized F007 repair and frozen `VC-F007` in `/Users/goftin/dev/projects/VectorCore-audit-2026-10-27-84798d3/evidence/FINDINGS.md`.

## Global Constraints

- Start at `6e7a136aa7723007b3155caade0c656438a29197` in the existing isolated `codex/audit-top-six-2026-10-27` worktree.
- Preserve original/audit checkouts and previous raw evidence. New evidence: `/Users/goftin/dev/projects/VectorCore-progress-fix-2026-10-28-evidence`.
- `percentage` stays a nonthrowing `Int`; `current`, `total`, `fraction`, `isComplete` and factory behavior remain unchanged.
- Do not change FP16 code. Discuss consumer needs with the user after this repair.
- Use Astra for substantive source review. Keep the branch unmerged.

## Review Focus

- Extreme numerator / small denominator: saturate instead of trapping.
- `Double(Int.max)` equality and the adjacent representable percentage: reject unsafe conversion without capping valid values prematurely.
- Zero totals and negative initialization: preserve zero percentage.
- Ordinary overshoot and fractional percentages: preserve values above 100 and truncation.
- Public access in optimized builds: use ordinary `import VectorCore` in the progress tests and run Debug/Release.

## Task 1: Saturating percentage conversion

**Files:** `Sources/VectorCore/Progress/OperationProgress.swift`, `Tests/ComprehensiveTests/OperationProgressTests.swift`, `CHANGELOG.md`; maintained downstream behavior documentation if its contract needs updating.

**Interface:** Existing `public var percentage: Int`; no new public declarations or storage changes.

- [x] Add regression tables over current/total 0, 1 and Int.max, negative initialization, 150% overshoot and fractional truncation; include the floating conversion boundary. Use the public import.
- [x] Run focused progress tests before the fix. Expected: termination at the actual out-of-range percentage conversion; retain the log and tested patch.
- [x] Saturate when the scaled percentage is at least `Double(Int.max)`, otherwise keep the original integer conversion. Document the behavior and named regressions.
- [x] Run progress/stream tests in Debug and Release. Expected: all pass, including adjacent boundary values.
- [x] Independently challenge the final diff with Astra; run full Debug/Release and applicable CI checks. Expected: no new failures; retain existing skips/known compiler canary explicitly.
- [x] Record disposition, exact source/commands/results and evidence hashes; commit and preserve the worktree. Resume the FP16 consumer discussion.
