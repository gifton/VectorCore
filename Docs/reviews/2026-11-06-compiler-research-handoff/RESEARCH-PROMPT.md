# Research request: Swift optimized stale-value miscompilation

Please investigate the compiler issue described in the attached `README.md` and evidence bundle. Start with the local observations and source files; use primary sources for external claims. We are preparing VectorCore 0.4.0, a Swift numerical-primitives library, and need a defensible explanation and release path. Correctness takes priority over rushing the release.

## Established starting point

A small program that imports no VectorCore code initializes `[Float]` to `[2]`, writes `1` through an `inout UnsafeMutableBufferPointer<Float>` helper, and then returns the old `2` under ordinary `-O` with three tested Apple compiler configurations. It returns `1` in Debug. The same unchanged probes pass with two exact Swift.org toolchains.

The five recorded matrices contain 80 compile/run cells. Every compiled program exits zero; the independent exact-value/bit-pattern predicate rejects nine positive cells and correctly rejects ten deliberately omitted-write controls. The latest tested candidate is **Xcode 27.3 beta, build 27C5020n**, frontend SHA-256 **`93a7373a7f558b211b9820797d40d7554a0fef2c917d039fa0fa894b60fd55eb`**. It still fails the original, inside-borrow/inout, and copied-buffer-descriptor/inout Release shapes.

Retained generated code shows the writer call and its `1.0` store still present, while the caller returns constant `2.0`. That stale return is already visible in optimized Swift SIL. Passing Swift.org output retains a post-call load. The first responsible transformation has not been isolated.

Two related upstream issues and fixes are leads: swiftlang/swift issues #92562 and #93141, and PRs #92580 and #93168. They involve MutableSpan and have different reported version ranges. Their exact causal relationship to our Array/unsafe-buffer example, and their inclusion in the tested proprietary Xcode frontends, are unproved.

## Questions to answer

1. **Validate source semantics first.** Are the original, inside-borrow, and copied-descriptor examples valid under Swift's API, exclusivity, lifetime, and memory-binding rules? If not, cite the precise rule and expression. Distinguish the descriptor's storage from its pointee, and a descriptor copy from a backing-array copy.
2. **Locate the incorrect transformation.** Explain the observed SIL/IR. Identify the first bad pass if possible, using actual before/after evidence. Evaluate alias analysis, escape analysis, callee effects, COW lowering, stack promotion, and specialization. Do not infer a bad pointee effect merely from `readonly` on the descriptor parameter.
3. **Trace upstream history.** Compare the relevant PRs' implementation and regression tests with our source. Identify exact commits, backports, and affected branches where evidence exists. Explain whether this is the same defect, an earlier manifestation, a related class, or presently unknown. Do not map Swift marketing versions to Xcode patch contents without evidence.
4. **Explain the toolchain discrepancy.** Xcode and Swift.org builds labeled 6.4.2 disagree. Xcode 27.2 and Swift.org 6.4.2 used the same SDK and host. Determine what additional evidence can distinguish compiler code, compiler-supplied standard-library bodies, build options, SDK effects, and runtime effects.
5. **Bound the impact.** Which source conditions matter? How should we test arbitrary element types, runtime-unknown input, allocation choices, inlining, generics, cross-module clients, optimization modes, architectures, and platforms? Do not conclude that nearly all computation is affected from these selected cases.
6. **Evaluate remedies.** Identify any concrete corrected compiler candidate or narrowly justified source accommodation, along with the evidence needed to accept it. Assess API/ABI, client-compilation, and performance implications. Moving a read inside the closure already fails with an inout writer; copying the descriptor also fails. Passing by-value and out-of-line-reader controls are not universal workaround proofs.
7. **Describe the release decision.** Give feasible exact compiler/SDK/runtime/distribution scopes and their unresolved requirements. A standalone Swift.org pass is not automatically an Xcode/App Store path. Source-package clients use their own compiler; generic and `@inlinable` bodies keep the importing compiler relevant even for binary-artifact proposals.

## Requested deliverables

- A concise conclusion with separate **observed**, **supported inference**, and **unresolved** findings.
- A causal explanation tied to actual source and generated-code evidence; identify what would falsify it.
- Primary-source links and exact commit/build identities, distinguishing implementation commits, merge commits, and backports.
- An affected / unaffected within tested scope / unknown matrix. Preserve failures and negative controls.
- A short ordered experiment plan in which each experiment separates competing explanations. Specify the source, flags, identities, observable result, and acceptance/rejection criteria. Confirm diagnostic flags against the exact compiler before prescribing them.
- Support or workaround options with their limits and required validation. If a corrected Xcode build cannot be established, say so clearly.
- A minimal upstream bug-report draft if the evidence warrants one. Drafting does not authorize posting it.

## Evidence discipline

Keep the original reproducer byte-for-byte as the acceptance baseline. Add variants as new experiments. Do not weaken assertions, hide known issues, infer success from exit zero, or treat `-Onone`, pass suppression, sanitizer silence, or a nearby passing shape as release acceptance. The numerical acceptance condition is the exact output `LOST_STORE observed=1.0 bits=3f800000 expected=1.0` with one final newline and exit zero.

The selected bundle is a partial source/evidence snapshot, not a complete Git checkout. Capture-time absolute paths in raw records are historical identities. Its manifest verifies included payloads; it does not mean every artifact named by an older summary is included. The single-file reproduction is self-contained; the historical matrix driver also records Git state and needs a real checkout.

The recent benchmark-type rename errors and Xcode child-test launcher/QoS problems are separate and have been addressed in the reviewed worktree. Their Debug successes do not close the optimized numerical gate. Historical full-library/client passes remain bound to their original revisions and toolchains.

Please do not install toolchains, change compiler/support policy, modify the release candidate, post upstream, or start performance acceptance merely to research this question. If execution is unavailable, distinguish a proposed experiment from an executed result. The value of this investigation is a better-supported explanation and a concrete path to trustworthy qualification.
