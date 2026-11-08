# Bounded native qualification investigation

**Authorization:** The user selected “Continue bounded qualification” after the
[remaining-exposure proposal](../2026-11-06-workaround-qualification/qualification-proposal.md).
This authorizes further investigation and a draft qualification mechanism. Final
support wording, platform coverage decisions and release admission remain pending.

**Goal:** Establish a reproducible native evidence assessment for the exact
released Xcode 27.2 candidate, retain its compiler defects explicitly, and refresh
the optional-kernel and generic platform compilation evidence for the migrated source.

**Scope:** The current reviewed worktree at base HEAD
`bdf990847babf567be1cbce22d2606e7102b1a47` plus the retained by-value migration.
The 389 compiled inputs are pinned by manifest SHA-256
`f2869bf68bb9286d33f2107566a2fb53d6913f95c76dced68d9affde448bd6eb`.
The native compiler frontend is
`448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f`;
its macOS SDKSettings identity is
`82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6`.

## Constraints

- Keep original canaries, numerical expectations, shipping source, existing CI
  admission and published support policy unchanged during this investigation.
- The draft checker reports compiler profile separately from native evidence.
  It never declares a compiler fixed, adopts support policy or admits a release.
- Reuse only source/product-bound migration and residual records. No claim that
  historical platform results qualify the migrated source.
- Build ordinary required optional-kernel and generic platform configurations;
  performance timing, physical-device execution and support expansion remain separate.
- Preserve all earlier snapshots, avoid excluded inherited-environment diagnostics
  and raw device identifiers, and make no installations, external messages,
  commits, pushes or publication.

## Tasks and ownership

- [x] Astra policy worker: implement `Scripts/ci/bounded_native_qualification.py`
  and behavioral tests in `Scripts/ci/tests/test_bounded_native_qualification.py`.
  Capture RED before implementation. Reject incomplete, mismatched or newly
  incorrect evidence; exercise unknown identities and altered records.
- [x] Astra platform worker: refresh the eight existing product/platform Release
  compilation cells under Xcode 27.2, retaining source/compiler/SDK/product evidence.
  This does not replace required Xcode 26.3 hosted coverage or establish runtime coverage.
- [x] Astra kernel worker: after Scripts inputs stabilize, run the existing four
  Debug/Release × C-on/off build-mode cells, their numerical public routes, direct
  tail controls and CLI mismatch checks. These are correctness/routing contracts,
  not performance acceptance.
- [x] Parent: review the draft mechanism, execute it on actual retained evidence,
  run the full Python policy suite and workflow validation, and obtain independent review.
- [x] Parent: consolidate verified results and bring the next coverage decision
  to the user with concrete available/unavailable combinations and infrastructure costs.

## Review focus

1. A known compiler failure must remain a failure even when native evidence is accepted.
2. Extra failures, missing/duplicate cases, crashed controls, altered hashes or
   mismatched producer/client inputs must reject the native assessment.
3. Expected diagnostic profiles must be exact; there is no compiler-version or
   issue-number wildcard exception.
4. The proposed restriction covers additional descriptor/view inout handoffs
   transitively after the specifically reviewed borrow-entry adapter. It is not
   a proof of safety for all remaining source shapes or external conformers.
5. Current required legacy CI, optional C routes and architecture/platform branches
   remain explicit obligations. Compilation, sanitizer, runtime and performance
   evidence must not substitute for one another.

## Outcome

The draft native evidence assessment accepts the pinned selection while retaining
`compilerProfile=known-failing`, `supportPolicyAdopted=false` and
`releaseAdmitted=false`. All four C-mode cells passed their bounded numerical/
routing controls. All eight platform actions exited zero with artifacts; seven
original console diagnostics remain unresolved, with a clean structured iOS
discriminator retained separately. The final Python suite passed 644 tests and
workflow validation passed. Independent review's digest-registration finding is
fixed and covered by regressions; re-review found no remaining material issue in
the reviewed changes.

The [coverage decision](coverage-decision.md) has been presented to the user.
Its disposition and final support/release approval remain pending; completing
this investigation plan does not close those gates.
