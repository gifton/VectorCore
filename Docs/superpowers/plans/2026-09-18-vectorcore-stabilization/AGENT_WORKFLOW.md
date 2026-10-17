# Agent workflow for the stabilization program

[Planning draft](README.md) · [Work items](WORK_ITEMS.md)

**State: planning only.** This is the proposed execution protocol for a later authorized implementation phase.

## Model and responsibility

All planning workers, implementers and independent agent reviewers use this exact selection:

```json
{
  "model": "gpt-6-astra",
  "reasoning_effort": "medium",
  "fork_turns": "none"
}
```

Each dispatch supplies a self-contained brief instead of conversation history. The orchestrator retains the complete audit/program context and decides contracts, algorithmic tradeoffs, review disputes, integration and release readiness. Worker difficulty triggers a concise question to the orchestrator; it does not authorize switching model/effort or spawning more agents. A fresh worker after repeated failed fixes uses the same requested model settings.

Use no more than two concurrent implementers and keep a reviewer slot available, within the runtime's actual limit. Reuse the original implementer for scoped corrections. Read-only planning/review may overlap useful orchestrator work. Avoid extra agents for bookkeeping or single mechanical edits that do not justify an independent assignment.

## Before the first implementation dispatch

- [ ] Verify the approved plan version, execution authorization, source revision, working tree and available slots.
- [ ] Create/verify an isolated implementation checkout; carry the approved untracked audit/plan artifacts into it deliberately. Do not lose or move unrelated user files.
- [ ] Establish a persistent execution ledger with plan identity, task states, decisions, reports and review ranges. This planning catalog is not evidence of implementation progress.
- [ ] Turn each selected catalog item into a bounded brief. Resolve public behavior, exact source ownership, test oracles and any cross-task interface first.
- [ ] Build a conflict table from all source, test, documentation and generated-output files the selected tasks will write. Shared files serialize even when agents use separate worktrees.
- [ ] Record a BASE revision before each task. Decide commit ownership and review packaging before parallel work; agents must not race a shared Git index.
- [ ] Schedule compiler/test invocations deliberately. Use isolated Swift scratch paths where appropriate; serialize expensive full suites and performance measurements.

For the first batch, V01 owns benchmark harness/model/test-integration changes, O01 owns TieredStorage/TriangularMatrix changes, R01 owns search metric dispatch/factory changes, and I01 owns IndexableVector hint changes. If a worker needs another owner's file, it returns the dependency to the orchestrator before editing it. Root adjusts the task boundary or serializes work.

## Required task brief

Provide one file containing:

1. Task ID, relevant audit records and original confidence/exposure/corrections.
2. Approved observable behavior, compatibility decision, explicit non-goals and dependencies already satisfied.
3. Exact permitted source/test/doc files and the shared interfaces consumed/produced. Mark proposed new files clearly.
4. Existing reproduction source and the independent oracle. Include concrete regression code and expected pre-fix failure once the design is settled.
5. The defect family enumeration: widths, tiers, direct/generic/provider routes, decode paths, boundary values, successful/throwing exits as relevant.
6. Exact focused Debug/Release commands, bounded crash/hang subprocess rules, and sanitizer/consumer checks justified by this task.
7. Acceptance criteria and report path. State that the worker must not broaden scope or claim a speedup without measurements.

The brief includes relevant AGENTS.md rules and the program's Global Constraints verbatim. Workers read only their selected audit rows, source/tests and necessary contract documents. They should not reread the whole 131-record register or repeat the original review.

## Implement → review → orchestrator decision

1. **Implementer:** add a meaningful failing regression; establish why it fails; repair the selected class; run focused checks; self-review. For source-proven unsafe access, a checked-boundary assertion may be preferable to deliberately executing an overrun. Preserve unrelated tests.
2. **Report:** write changed files, actual test names/commands/exit results, before/after observations, route coverage, source revision, compatibility effects and unresolved concerns to the task report. Return a short summary and path.
3. **Independent reviewer:** receive the brief, report and a complete diff package from recorded BASE through the task's final revision. Review spec compliance and code quality together. Inspect ownership, mathematics and newly reachable paths; request targeted new evidence only when a concrete concern justifies it.
4. **Orchestrator:** inspect the diff, evidence and review; resolve cross-task assumptions and qualifications. A worker's passing test report does not close the audit record by itself.
5. **Corrections:** send scoped findings back to the original worker and obtain a scoped re-review. Do not rerun unrelated full suites on every correction. Repeated failures trigger root redesign/reassignment; they never justify declaring a load-bearing defect closed.
6. **Integration:** run focused cross-task checks after integration and a full appropriate validation gate at a milestone. Evidence on an earlier task commit does not automatically prove the integrated revision.

No new public correctness/safety defect can be parked merely to exhaust a review-loop budget. Deferred opportunities and documented limitations receive an explicit disposition and owner. A plan conflict is recorded as a root decision with its rationale and cost if wrong.

## Evidence and closure record

For each closed audit record, preserve:

```text
Audit record(s):
Task / approved contract:
Fix revision / integrated revision:
Changed files:
Regression names and pre-fix observation:
Post-fix commands, exit results and artifact paths:
Class/route coverage and remaining exclusions:
Consumer/platform/sanitizer evidence where required:
Independent review outcome:
Orchestrator ruling and compatibility consequences:
Disposition: fixed | contract-resolved | investigated-no-defect | deferred-with-reason
```

These are future report field labels, not claims that evidence already exists. `deferred-with-reason` does not mean fixed and cannot satisfy a release gate that requires a fix. Original audit observations remain immutable.

## Validation levels

- **Task:** focused regression plus affected neighboring tests; Debug and Release for optimizer-, lifetime- or numerical-sensitive changes.
- **Safety/concurrency:** bounded subprocesses for intentional traps/nontermination checks; targeted ASan/TSan where supported and relevant. Record tool limitations; do not infer sanitizer coverage from ordinary tests.
- **Integration:** full Debug/Release and relevant CI policy/consumer checks on the integrated revision, plus public-symbol/compatibility review where signatures or conformances changed.
- **Performance:** repaired harness, equivalent work, Release builds, warmups/raw samples, hardware/toolchain/flags, medians/variation, allocations and peak/retained memory. Timing claims require before/after evidence; source complexity claims must be labeled as such.
- **Downstream:** pin actual sibling revisions, exercise changed contracts and route thresholds, and distinguish Core, VA CPU and GPU evidence. Prepare sibling changes separately; do not edit another repository under Core-only task authorization.

## Token and time controls

- Reuse validated evidence; investigate only a task's unresolved contract or reproduction gap.
- Keep one source of truth for state; completed tasks are never redispatched after context compaction.
- One implementer and one independent review role per bounded task; root performs final adjudication.
- Ask workers for short final messages linked to complete reports, not duplicated narratives.
- Use narrow re-reviews and targeted checks after fixes. Run broad suites at integration boundaries or when a specific risk requires them.
- Token savings come from scope and context discipline. Do not reduce acceptance coverage or silently weaken numerical tolerances to shorten a task.
