# Unsafe-buffer workaround audit and validation

> For agentic workers: execute the authorized audit in the reviewed worktree. Use the debugging, test-driven-development and verification workflows for any resulting code change.

**Goal:** Determine which VectorCore call boundaries need a by-value unsafe-buffer workaround and validate the actual library and downstream behavior without changing compiler acceptance policy.

**Architecture:** Audit declarations and reachable calls before editing production code. Retain independent compiler controls, exercise public consumers under ordinary optimization, and add a bounded mechanical guard for the audited source rule. Prefer an evidence-backed small change over an untested public protocol migration.

**Tech stack:** Swift 6, SwiftPM native builds, Python standard-library evidence drivers and CI guards.

**Spec:** The user's November 6 authorization and supplied alias-analysis/RLE research; existing release requirements remain in `../2026-11-05-release-gates/release-admission.md`.

## Constraints

- Worktree: `/Users/goftin/dev/projects/VectorCore-fixes-2026-10-27-84798d3`, starting at `bdf990847babf567be1cbce22d2606e7102b1a47`.
- Preserve the unchanged standalone compiler probes, their numerical predicates and ordinary compiler flags.
- Preserve compiler known-issue handling, release admission, public support promises and the existing research handoff.
- Record exact compiler, SDK, source and artifact identities. Keep compiler-control results separate from library workaround results.
- No performance acceptance during this audit. No external posting, release branch, tag or publication.
- No access to the excluded inherited-environment diagnostic or raw device identifiers.

## Review focus

- Protocol witness and concrete overload selection can differ; exercise both.
- A custom downstream conformer can depend on the current public signature; assess compatibility before migration.
- Copy-on-write mutation must preserve copies and buffer lifetime, including throws and empty/tail sizes.
- Optimization can specialize public generic/inlinable code in the consumer; use separate compiled consumers.
- A successful process can return stale numbers; check outputs and omitted-write negative controls independently of exit status.

## Tasks

1. **Audit and baseline.** Inventory direct descriptors, aliases, protocol/callback routes and pointer-bearing state. Reproduce current compiler behavior and the actual exposed library route before making a production change.
2. **Choose the smallest justified remediation.** Test a proposed by-value boundary against current behavior and source compatibility. If the production numerical paths already follow the rule, retain that finding rather than claim an unnecessary rewrite fixes them.
3. **Regression coverage.** Add independently checked public consumer probes and a scoped declaration guard with adversarial fixtures; demonstrate that representative bad shapes and omitted writes are rejected. Keep the existing compiler canary intact.
4. **Validate.** Use fresh build products for affected Xcode and a passing Swift.org control. Run the normally enabled library suites and separately compiled optimized clients; retain every failure and skip. Broaden to beta after the core experiment is understood.
5. **Review and report.** Independently review the changes and evidence, publish the bounded result in this directory, and list the compiler, compatibility and release gates still open.

## Execution record

- Initial worktree is already isolated on `codex/audit-top-six-2026-10-27`; only the earlier compiler research handoff is untracked.
- Three existing Astra agents are independently auditing source reachability, public-consumer coverage and guard/policy boundaries. The parent owns native builds to avoid shared build state.
- Initial search finds one direct production unsafe-descriptor `inout` declaration: `SIMDStorage.withUnsafeMutableBufferPointer`. Its `ContiguousArray` conformance currently uses the standard-library witness. No production change has yet been selected.
- Completed source and compatibility audit: ordinary numerical interfaces already use by-value descriptors; the exposed SIMDStorage route reproduces the stale read in optimized Xcode clients. Changing the public requirement breaks conformers; a same-name adapter causes overload ambiguity. Production code and the public API remain unchanged.
- Completed six fresh native Debug/Release suites with Xcode 27.2, Xcode 27.3 beta and official Swift.org 6.4.2. Only the two Apple Release compiler canaries fail; all other normally enabled bodies pass. Every build's 388 compiled input files remain unchanged.
- Completed 288 by-value consumer observations and 12 full omitted-write controls, using both producer and client configurations. Independent review verifies source inventories, consumer/producer receipt bindings and command-stream hashes.
- Added and adversarially tested the descriptor declaration guard, numerical consumer judge, retained consumer fixtures and fixture-manifest discovery. Corrected the separately discovered output-path lint false positive with exact reviewed function exceptions; retained failing attempts and verified their controls.
- Final Python discovery: 624 tests pass. GitHub YAML/template and workflow policies pass; diff checks pass. Final policy inputs are unchanged during verification.
- Audit complete. [README.md](README.md) and [validation-summary.json](validation-summary.json) record results and limits. Compiler admission, public API migration/support decisions, platform/performance gates and exact-candidate release qualification remain open; no release branch, tag or publication was made.
