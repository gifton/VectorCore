# Independent implementation review and fix verification — 2026-11-12

A fresh-context independent reviewer assessed implementation range
`6683ff4..dca55fb`, the adopted policy, plan, ledger rulings and retained evidence.
The review found no Critical issues and two Important verifier gaps. The author
reproduced and fixed both in one pass. No second reviewer pass was used; fixes
are verified by failing-then-passing regressions and the full repository suite.
Actual hosted qualification and final release admission remain open.

## Findings and disposition

1. **Logical source drift — fixed.** Final canonical-file hashing did not detect
   a newly added compiled source or a retargeted source symlink. The verifier
   now rechecks the complete logical source inventory, content and canonical
   binding map before returning. Regression tests cover additions under all
   three compiled-source roots and retargeting to both different and equal
   bytes; canonical first-registration-wins pins remain intact.
2. **Driver/frontend cross-binding — fixed.** A self-consistent reviewed native
   receipt could name another pinned compiler without matching the declared
   frontend. A shared role check now binds `swift`/`swiftc` to the selected
   frontend's adjacent pinned drivers, including `SWIFT_EXEC` overrides.
   Tests cover native producers, consumers, witnesses, every C-on/off build
   and link cell, and wrong tool roles in both iOS app/witness compilations.
3. **README restriction — regraded Important and fixed.** The reviewer marked
   this Minor. The author graded the effect on a consumer looking at the primary
   compatibility section: omitting the active restriction could conceal the
   silent-wrong-value risk central to this release decision. README now links
   the exact bounded compiler policy and current scope, identifies valid-Swift
   miscompilation and the transitive descriptor/view restriction, and retains
   the admission caveat. This prose change was reviewed against the adopted
   policy and existing runtime evidence. Its claim-ledger entry pins the adopted
   policy document; the existing ledger checks validate that entry. No test
   that merely matches wording was added.
4. **Deferred Minor.** A cosmetic leading `+` remains in the historical alpha
   paragraph at the end of `RELEASE_STATUS.md`. It does not change qualification
   or release admission. Deferred under the native-execution review workflow.

Seven regression methods reproduced 19 failing mutation cases before the
corresponding guards and pass afterward. The focused qualification suite passes
111 tests. The complete post-review suite passes **743 tests** (277.064 s);
all ten GitHub configuration policies and `git diff --check` pass.

All three immutable stable profiles were reassessed with the corrected verifier
against canonical archived evidence and the frozen source copy. They remain
**accepted / known-failing / releaseAdmitted=false**. No numerical result,
expected value, captured command, profile digest or compiled candidate was
changed, and no additional physical run was necessary. The
[machine-readable review record](implementation-review-result.json) identifies
the verifier code hashes and reassessment reports.

## Rulings on review boundaries

- Actual hosted execution and runner availability remain pending the separately
  approved introduction/dispatch step. Offline validation does not establish a
  hosted pass. Cost if wrong: a hosted incompatibility can require more work.
- The reviewer assessed retained runtime evidence rather than rerunning it.
  The author performed the fresh native/simulator/physical captures and both
  pre/post-fix archive checks. Cost if wrong: an execution-provenance error can
  survive a review that does not repeat execution independently.
- Minimum-OS, Intel/other-platform runtimes, sanitizer/performance and final
  release-reference admission remain separate gates. Cost if wrong: additional
  qualification must be completed before a broader support claim or release.
- Arbitrary external helpers/conformers and an unidentified consumer application
  remain outside the assessed compositions. Cost if wrong: a new consumer route
  requires its own investigation and cannot inherit these results.
- A deliberately dishonest profile author or hostile evidence producer remains
  outside the trusted-profile model. The fixed gaps enforce consistency within
  that model. Cost if wrong: this verifier cannot authenticate fabricated
  evidence or replace trusted capture/review procedures.

The earlier implementation rulings remain in the execution ledger: alpha
tooling evidence was kept separate from frozen stable evidence; native outcome
parsing was separated into its own module; corrected parsers revalidated
unchanged alpha iOS captures with preserved collector source; hosted archive
provenance uses a separate helper; and only four exact reviewed JSON data files
are exempt from the product-layout literal scanner. Their costs are, respectively,
a fresh qualification batch, module maintenance, possible recapture if the
provenance distinction were rejected, helper maintenance, and continued review
of the four data files' schema/use.
