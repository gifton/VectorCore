# VectorCore first audit repairs

Baseline: `84798d3fe3548ea78d4ed5da5dba4d723a85bf32`. The original audit and its source snapshot remain frozen. This work is on `codex/audit-top-six-2026-10-27` in a separate worktree.

The user wants to drop watchOS and address the six most consequential audit findings before concentrating on the acceleration package. The selected set is VC-F001, F002, F004, F005, F006 and F008. This prioritizes the unusable public tuning API over the extreme progress-percentage overflow. F007, F009–012 remain follow-ups; F003 remains a defect in the audited snapshot but falls outside the revised supported-platform promise. No merge, push or release is part of this work.

## Contracts and implementation choices

### Platform scope

Remove watchOS from advertised deployment platforms, required CI destinations and maintained support documentation. Keep macOS14, iOS17, tvOS17, visionOS1 and Mac Catalyst behavior. SwiftPM's platform list is not an exclusion mechanism: incidental compilation on watchOS does not constitute supported behavior. Keep harmless OS identification/conditional code where removing it would invent an unrelated failure. Update existing platform-policy tests to cover the new supported set; do not rewrite historical audit/design records.

### VC-F001 — reference storage

The user explicitly selected copy-on-write value types, accepting the source compatibility change for clients relying on shared mutation or mutating a `let` instance. Convert AlignedDynamicArrayStorage and DynamicArrayStorage to public structs with mutating write APIs. Keep initializer labels, count, subscript, scoped borrow, alignment and dot-product operations. DynamicArrayStorage uses ContiguousArray value semantics; AlignedDynamicArrayStorage owns a private reference allocation and clones before mutation whenever shared. No public backing reference may escape. COWDynamicStorage delegates to this value storage rather than independently checking reference identity.

Sendable safety relies on immutable sharing plus uniqueness before every write, including the complete mutable-borrow path. Concurrent independent copies must not change one another. Sharing a mutable variable across Sendable closures must be rejected by the compiler; a caller can still misuse an explicitly unsafe escaped pointer, which remains outside the scoped-borrow contract. Array exclusivity applies inside mutable borrows; no recursive access promise is added. No storage locks or arbitrary user callbacks under locks are introduced.

Regression coverage: both public types, copies modified by scalar setters and throwing/nonthrowing buffer borrows, self/cross dot products, initializer alignment/fallback/lifetime, and COWDynamicStorage integration. An ordinary-import negative consumer must reject the original `let` shared-mutation spelling; a positive consumer must execute concurrent independent-copy mutations under TSan. Measure representative Release borrow/dot costs against the unchanged baseline, reporting observations rather than universal performance guarantees.

### VC-F002 — collection lifetime

Keep replacement/membership/ID behavior. Retain replaced values or the removed dictionary until after the lock is released; then permit destruction. A destructor may read or mutate the collection. Cover update, removeAll and single removal with a conforming checked-Sendable custom value and bounded child execution, including nested re-entry and thrown/missing-ID behavior. Do not solve this by making the collection lock recursive: destruction must occur outside the critical section.

### VC-F004 — interpolation

Use Double intermediates for finite Float endpoints in fixed/dynamic linspace, fixed clamped/unclamped lerp, dynamic lerp, and smoothstep through lerp. Return endpoints exactly where promised, preserving signed zeros there. Interior values use the retained Double difference/weight before Float publication. Legitimately unrepresentable extrapolation may still produce infinity. Preserve existing clamping and nonfinite-input behavior where the repair does not require a change. Fixed widths0/1, dynamic widths0/1, mismatched dimensions and nonfinite controls are explicit tests.

### VC-F005 — zero shortcut

Only an exactly zero query (including signed zero) qualifies for the squared-distance zero shortcut. Do not change the separate approximate-zero policy used elsewhere. Verify singleton/multiple candidate identity and independent direct-difference references for tiny nonzero vectors; preserve the existing squared-distance return convention.

### VC-F006 — precision diagnostics

Retain the mean as Double for residuals, population variance and outlier classification; widen operands before subtraction and use a Double three-sigma threshold. Publish the existing Float fields. Preserve the heuristic recommendation policy, finite-value filtering and existing empty/all-nonfinite policy in this scoped repair. Verify the opposite-extrema fixture, adjacent large values, constants, mixed finite/nonfinite values and an analytically known outlier count.

### VC-F008 — public tuning construction

Add explicit public initializers for Tuning and CalibrationProbes with their existing property labels. Tuning requires nonnegative breakEvenN and positive minChunk (nonthrowing initializer preconditions); modeBias is explicit. CalibrationProbes retains raw observations, including invalid values, because the existing calibration calculation deliberately falls back for invalid measurements. Trace consumers before making construction newly reachable. Verify ordinary-import compiled clients can construct and use overrides/calibration; @testable alone is insufficient.

## Completion and evidence

Implement with failing regressions first. Use existing Swift/Python/consumer checks, Debug and Release, plus targeted TSan and bounded deadlock probes. Run the full default suite at integration; disclose unrelated baseline failures if encountered. Update affected source claims, maintained docs and claim-ledger identities where required by existing policy, with named tests rather than new unverified correctness promises. A fresh Astra reviewer checks the final change and its boundary cases. No routine phase approval is needed under the user's standing autonomy instruction; material contract conflicts are raised if source investigation reveals one.
