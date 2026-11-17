# Remaining exposure and qualification proposal plan

> For agentic workers: execute the authorized investigation in the reviewed worktree. Product/API/support changes require the user's decision on the concrete proposal; do not change release admission during this audit.

**Goal:** Determine what the completed by-value migration removes, what valid client compositions remain exposed, and what evidence a bounded qualification policy would need.

**Architecture:** Three independent Astra source/policy/platform reviews, followed by small downstream numerical probes against the already source-bound migrated Release products. Preserve original compiler failures and negative controls. Record a proposal separately from current policy.

**Tech stack:** Swift 6, existing native SwiftPM products, `v03_process`, `swiftpm_products`, and the existing complete numerical-output judge.

**Spec and authorization:** The user approved the preceding audit/qualification approach and requested involvement at important decision points. This authorizes investigation and a concrete proposal, not accepting an affected compiler, narrowing support, or another public API change.

## Constraints

- Worktree: `/Users/goftin/dev/projects/VectorCore-fixes-2026-10-27-84798d3`; base HEAD `bdf990847babf567be1cbce22d2606e7102b1a47` plus the preserved migration delta.
- Keep Sources, Tests, Package.swift, CI, canary predicates and existing qualification policies unchanged.
- Retain the prior migration evidence; verify actual inputs before reusing its Release products. These are reused products, not a new build.
- No performance acceptance, installations, device queries, external messages, commits, publication, or access to excluded inherited-environment diagnostics.
- Do not equate a passing source shape with absence of the bug class. Keep exploratory failures explicit.

## Review focus

- A caller can copy a by-value descriptor and pass that copy inout without violating the scoped borrow's lifetime.
- Actual optimized vector storage has SIMD elements and a checked computed property; standalone ContiguousArray<Float> evidence does not establish its numerical outcome.
- A custom conformer may satisfy the new function type while forwarding through an internal inout descriptor boundary.
- Generic pointer-bearing state, imported aliases and caller-generated code exceed the mechanical source guard's proof scope.
- Current required CI compiler/platform configurations differ from the newly tested native compiler matrix.

## Tasks

- [x] Independently review source exposure, policy/gate reachability, and platform qualification requirements.
- [x] Retain minimal positive/by-value, reconstructed-inout, custom-conformer and actual vector probes; require full omitted-write controls and exact output inventories.
- [x] Execute both client optimization modes against the matching retained Release library for Xcode 27.2, Xcode 27.3 beta and Swift.org 6.4.2. Hash source/compiler/SDK/products/output and preserve all results.
- [x] Consolidate a source map, numerical findings, explicit policy alternatives, and the smallest useful platform follow-up.
- [x] Present the support-boundary decision to the user. Keep release admission unchanged pending that decision and subsequent qualification.

The direction question has been delivered. The user's answer remains pending;
all completed checks concern the investigation, not an adopted exception or release.
