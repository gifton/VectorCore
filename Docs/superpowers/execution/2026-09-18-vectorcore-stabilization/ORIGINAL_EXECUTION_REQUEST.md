# Recovered original execution request

User message from thread `01a0a347-3759-7d80-b6ef-a82386372197`, timestamp `2026-09-18T23:55:33.618Z`, transcript line 1529. The body below is copied verbatim; its final sentence was truncated in the original record.

```text
You are the lead orchestrator for stabilizing VectorCore. I want you to execute this work
  through subagents, while retaining the difficult architectural, numerical, ownership and
  integration decisions yourself.

  Repository:
    <repo>

  Start by reading:
    Docs/superpowers/plans/2026-09-18-vectorcore-stabilization/README.md
    Docs/superpowers/plans/2026-09-18-vectorcore-stabilization/AGENT_WORKFLOW.md
    Docs/superpowers/plans/2026-09-18-vectorcore-stabilization/DECISIONS.md
    Docs/superpowers/plans/2026-09-18-vectorcore-stabilization/WORK_ITEMS.md
    Docs/superpowers/plans/2026-09-18-vectorcore-stabilization/tracking.json
    Docs/superpowers/plans/2026-09-18-vectorcore-stabilization/SHARED_FILES.md
    Docs/audits/2026-09-14/README.md

  Read FINDINGS.md and individual audit records as relevant to each task. Follow the
  applicable AGENTS.md instructions and subagent-driven-development, verification and testing
  skills.

  AUTHORIZATION AND SCOPE

  This message authorizes beginning implementation; it supersedes the planning-only
  restriction in those documents.

  The catalog contains 43 candidate work items across 12 workstreams. It is not an executable
  specification: resolve each selected item's design checkpoint and prepare a bounded
  implementation brief before dispatch.

  Begin with the proposed first batch:
  - V01: strict benchmark correctness and failure propagation.
  - O01: compact-storage copy isolation and element lifetime.
  - R01: requested metric preservation.
  - I01: normalization-hint invalidation.

  Respect prerequisites and staged closure. Design the S01 shape contract and I02 cache
  identity approach early because they constrain subsequent work.

  Continue through the stabilization program in dependency order. Do not ask whether to
  continue between routine tasks. Prefer preserving public source and wire compatibility. If
  an unavoidable breaking change is necessary, prepare the concrete change and migration
  implications for my decision before applying it.

  Do not invent production workload or latency/memory targets. Correctness work can proceed;
  scale-readiness claims require an explicitly defined operating envelope.

  AGENT POLICY

  You are the Astra xhigh orchestrator. Every delegated implementer and independent reviewer
  must use:
    model: gpt-6-astra
    reasoning_effort: medium
    fork_turns: none

  Give workers self-contained, narrowly scoped briefs. Do not send the entire conversation or
  audit to every worker. Do not silently change models/effort or permit nested agent spawning.
  Difficult questions come back to you.

  Use at most two simultaneous implementers, reserving capacity for a reviewer within the
  actual available limit. Never allow concurrent edits to the same files or races on a shared
  Git index/build directory.

  BASELINE AND WORKSPACE

  The audited revision was:
    8b02e75bca7055ae11170873d15c06f1538709b7

  Check current HEAD and local changes before acting. Do not reset the repository to the
  audited revision. If code has changed, validate affected findings against current source.

  The audit/review/plan directories were untracked when planning finished. Ensure the approved
  documents and necessary evidence are deliberately available in any isolated implementation
  checkout. Creating a worktree from HEAD alone may omit them.

  Preserve unrelated user files, including .antigravitycli/ and GEMINI.md. Keep the historical
  audit and evidence unchanged. Maintain a separate persistent execution ledger and closure
  records. Keep changes local; do not push, merge, publish or modify sibling repositories
  without authorization.

  EXECUTION AND REVIEW

  For each task:
  1. Resolve observable behavior, compatibility, mathematical/ownership contracts and
  dependencies.
  2. Specify exact write ownership, concrete failing regressions, independent oracles,
  commands and acceptance criteria.
  3. Have an Astra/medium implementer establish the failure, repair the defect class, run
  focused checks and write an evidence report.
  4. Have an independent Astra/medium reviewer inspect the brief, report and complete diff
  from the recorded task base.
  5. Personally adjudicate findings and cross-task consequences. Return corrections to the
  implementer and obtain scoped re-review.
  6. Record integration revision, tests, remaining limitations and audit-record closure
  accurately.

  Use focused tests during task work and broader checks at integration milestones. Do not
  repeatedly run unrelated full suites without a concrete reason. Never close a finding merely
  because code changed or unrelated tests pass.

  IMPORTANT TECHNICAL CONSTRAINTS

  - Close repeated defect classes across widths, storage tiers, generic/concrete/provider
  routes and decoding boundaries.
  - Re-derive reachability whenever a configuration or routing gate changes.
  - Full-coordinate hashing alone does not establish exact cache identity.
  - Repair dormant bulk writeback before activating it for performance.
  - O01 alone does not close the full OWN-05 acceptance matrix; O02 supplies remaining tier/
  bulk evidence.
  - Keep HybridStorage copying findings distinct from default generic DimensionStorage; the
  blanket extrapolation was rejected.
  - Accelerate division already failed the audited subnormal case; replacing a primitive is
  not proof of repair.
  - Numerical references must be independent and finite-aware. Double extremes require scaled/
  exponent-based references.
  - Preserve existing wire bytes, SoA ownership/layout contracts and defined NaN/tie semantics
  unless an approved migration changes them.
  - Distinguish reproduced defects, source-established findings, qualified limitations,
  profiling opportunities and unverified concerns.
  - Downstream source inspection is not executed CPU/GPU integration evidence.

  Maintain concise progress updates and a durable ledger so another session can resume without
  repeating completed work. Begin by inspecting t
```
