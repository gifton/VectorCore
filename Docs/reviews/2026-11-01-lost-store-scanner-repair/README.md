# Lost-store scanner repair — VC-C002

The inside-borrow scanner gap is repaired on `codex/audit-top-six-2026-10-27`, starting from `9ab1b5e`. The carried branch retains all earlier work. This closes the reported CI tooling defect within the lexical scope below; it is not a compiler fix or 0.4.0 release sign-off.

## Change

`Scripts/ci/tests/test_lost_store_reads.py` now reports an `inout:` hit when a literal-initialized local is borrowed with `withUnsafeMutableBufferPointer`, the closure passes its whole named or implicit buffer parameter as `&buffer`, and a direct read follows the call in that same closure body. The exact retained `ReturnInsideInout.swift` reproduction is tested alongside the original post-borrow reproduction and the by-value control.

The fixture matrix covers named, typed and implicit parameters, capture lists, trailing and parenthesized closures, argument labels, ordering, string interpolation and shadowed names. All 34 original spelling/control cases run with both semicolon and newline layouts. Additional regressions cover interpolation-local bindings, conditional binding lifetimes, and compiler directives versus argument labels. The allowlist still rejects unreviewed hits, stale entries and missing reasons.

The canary's comment and failure message now report the stale-value symptom without claiming that a particular upstream patch is missing. Assertions, compiler version gates and expected-issue conditions are unchanged. No library implementation, public API or compiler setting changed.

## Scope

This remains a lexical review trigger, not a dataflow analysis or proof that a callee writes the pointee. The new form requires a literal owner initializer and an identifier-style writer call. It excludes nested brace bodies and if/while/for/catch headers, aliases, rebased buffers, helper reads (including `Array(buffer)`), other pointer APIs, complex capture expressions, tuple/multiple-binding shadows, and unsupported callable/signature spellings. Accepted source is not certified immune to compiler defects.

The existing post-borrow and captured-variable checks remain active. Comparing old and new scanners on the final real tree gives **79 identical hits across 328 Swift files**, with no new `inout:` hits and no allowlist changes. No current production instance was found within this bounded scan.

## Validation

| Check | Result |
|---|---|
| Retained three-case reproduction before repair | Inside-borrow case fails; original positive and by-value negative controls pass. |
| New spelling fixtures before implementation | 16 expected failing subcases. |
| Review regressions before their fixes | 9 scope failures, then 5 header-classification failures. |
| Final scanner suite | 15 tests pass. |
| Final full Python policy suite | 565 tests pass. |
| Targeted Swift Debug and Release | Each: 9 tests in 3 suites pass, including the hard canary, captured-variable controls and adaptive reduced-result test. |
| Claim ledger | Zero problems. |
| Independent review | Approved after all three scope findings were reproduced, fixed and regression-tested. |

Swift checks used the exact previously qualified **Swift.org 6.4.2** (`swift-6.4.2-RELEASE`), macOS SDK 27.2, arm64 macOS 27.0.1, native SwiftPM, normal Debug/Release optimization and three jobs. Existing qualification build caches were reused. The initial sandboxed Debug attempt stopped at SwiftPM's `sandbox_apply`; rerunning outside the agent sandbox passed with SwiftPM's normal sandbox intact. The canary source was unchanged throughout both successful Swift runs. Python-only review changes continued during those runs; the final full policy run used stable final sources.

This repair does not repeat the full Swift suite or replace the [compiler qualification record](../2026-11-01-compiler-qualification/README.md). Xcode 27.2's bundled compiler failure remains recorded; hosted CI/toolchain adoption, other platforms, integration and performance acceptance remain separate work.

## Reproduce

From the repository root:

```sh
python3 -B Scripts/ci/tests/test_lost_store_reads.py -v
python3 -B -m unittest discover -s Scripts/ci/tests -v
python3 -B Scripts/ci/check_claim_ledger.py
```

For Swift, select the exact compiler and SDK using the [qualification instructions](../2026-11-01-compiler-qualification/README.md#reproduce-the-qualification), then run `swift test --build-system native --jobs 3 --filter LostStoreCanaryTests` with each of `-c debug` and `-c release`, explicit SDK and scratch paths. This filter executes the nine tests reported above on the qualified toolchain.

[Validation summary](validation-summary.json) records source hashes, commands and results. Raw logs, red/green patches, runner and review record are retained at `/Users/goftin/dev/projects/VectorCore-scanner-repair-2026-11-01-evidence`; [evidence-manifest.json](evidence-manifest.json) records their hashes. The original compiler investigation and qualification archives remain unchanged.
