# VectorCore compiler wrong-result investigation: research handoff

Prepared **November 6, 2026**, for an independent compiler investigation ahead of VectorCore 0.4.0. Local execution evidence below was collected November 1–5; preparing this handoff did not execute a new compiler matrix. Repository context is commit **`bdf990847babf567be1cbce22d2606e7102b1a47`** on `codex/audit-top-six-2026-10-27`.

**Current conclusion:** the tested Xcode 27.3 beta compiler still produces a silent wrong result in the unchanged, small Swift reproducer. Xcode 27.2 and the separately installed Command Line Tools compiler also fail. Two exact Swift.org toolchains pass the bounded probe matrix. No compiler path has yet been qualified for the complete intended release/distribution scope.

This is a research handoff, not a claim that every VectorCore operation is broken, an adopted workaround, or release sign-off. The central task is to establish the exact cause and affected scope, and then identify a defensible compiler or source-level remedy without weakening the numerical test.

Read this document first, then [RESEARCH-PROMPT.md](RESEARCH-PROMPT.md). [matrix-summary.json](matrix-summary.json) contains the five compiler identities and all 80 cell observations. The companion ZIP supplies unchanged probe sources, raw selected command records, and optimized SIL/LLVM IR; its layout is described below.

## 1. What is established, and what is still a hypothesis

| Question | Evidence-supported answer |
| --- | --- |
| What fails? | Certain optimized programs write `Float(1)` through an unsafe mutable buffer, then observe the previous `Float(2)`. |
| Does it require VectorCore? | No. The standalone program imports no modules explicitly and uses only Swift standard-library facilities. It calls no VectorCore, Accelerate, C, or Metal kernel. |
| Is this floating-point tolerance or reduction-order error? | No. There is one exact assignment and one read. Both values are exactly representable; the observed bits are the old value's bits. |
| Does the compiler or process report failure? | No. Every program in the five matrices compiles and exits successfully. An independent output predicate detects the wrong answer. |
| Is the write instruction literally absent? | Not in the retained original-probe IR. The writer still stores `1.0`; the caller still calls it but returns the constant `2.0`. “Lost store” is the investigation's historical shorthand for the symptom. |
| Where is the first retained evidence of the wrong value? | In optimized Swift SIL, before LLVM IR generation. The first transformation responsible has not been isolated by these retained dumps. |
| Is alias analysis / redundant-load elimination implicated? | It is a strong investigative hypothesis, consistent with the source-shape controls and related upstream fixes. Exact pass attribution and patch mapping for these binaries remain unproved. |
| Does this affect nearly all VectorCore computation? | That has not been shown. The demonstrated failures are source-shape dependent. Their presence is nevertheless serious enough to block broad optimized-build qualification. |

## 2. The unchanged minimal reproducer

File: `Docs/reviews/2026-11-01-compiler-qualification/probes/LostStoreObserved.swift`. SHA-256: `d6bda153f1ed5d79a393d04ea385690fcdef616f4391d73ab47b2ed2d9221326`.

```swift
// Exact pure-Swift shape from LostStoreCanaryTests.swift. No VectorCore import.
@inline(never)
func writeOne(_ out: inout UnsafeMutableBufferPointer<Float>) {
    out[0] = 1
}

@inline(never)
func observedValue() -> Float {
    var values: [Float] = [2]
    values.withUnsafeMutableBufferPointer(writeOne)
    return values[0]
}

let observed = observedValue()
print("LOST_STORE observed=\(observed) bits=\(String(observed.bitPattern, radix: 16)) expected=1.0")
```

Required output, including one final newline:

```text
LOST_STORE observed=1.0 bits=3f800000 expected=1.0
```

Observed on the failing optimized configurations:

```text
LOST_STORE observed=2.0 bits=40000000 expected=1.0
```

The expected state transition is simply `[2] → write through element pointer → [1] → read 1`. There is no floating-point arithmetic in the kernel, no concurrency, and no data-dependent indexing. Exact bit comparison is intentional here; an epsilon comparison would obscure the property under test.

### Source-legality assessment to verify independently

Our reading is that this is valid use of the API: the array has one initialized element; index zero is in bounds; the helper uses its buffer only during the borrow; the pointer does not escape; the helper does not change the buffer's address or count; and the original array is read only after the borrow finishes. Apple's API contract permits modifying the array through the supplied buffer, limits the pointer's lifetime to the call, and directs callers to use the buffer rather than the array within the closure. [Array API contract](https://developer.apple.com/documentation/swift/array/withunsafemutablebufferpointer%28_%3A%29).

The inside-borrow variants below read through the buffer after the nested writer call completes, rather than accessing the owning array during its mutation. The copied-buffer variant copies a pointer/count descriptor, not the backing elements. Review these details against Swift's exclusivity rules before treating optimizer behavior as the only possible explanation. Successful compilation or passing Debug execution alone is not a proof of source legality. [Swift memory-safety rules](https://github.com/swiftlang/swift-book/blob/main/TSPL.docc/LanguageGuide/MemorySafety.md).

## 3. Exact tested compiler configurations

All five matrices ran natively on **arm64, macOS 27.0.1, build 26A434**. The beta follow-up's host metadata identifies model `Mac15,9`. Commands explicitly set `-swift-version 6 -target arm64-apple-macosx14.0`, an SDK, and a fresh module-cache directory per matrix. Debug means `-Onone -D DEBUG`; Release means ordinary `-O`. No optimizer pass was disabled.

The target's macOS 14 deployment floor is **not** evidence of execution on macOS 14. These matrices do not establish Intel, other operating systems, device, simulator, or minimum-runtime behavior.

| Compiler / Xcode build | Executed Swift identity | SDK | Positive cells | Deliberate negative controls | Matrix-driver exit |
| --- | --- | --- | --- | --- | --- |
| Xcode 27.2, `27B52` | `swiftlang-6.4.2.1.9`, Clang `2100.4.1.1` | macOS 27.2 | 11 pass / **3 fail** | 2 correctly rejected | 1 |
| Separate CLT installation | `swiftlang-6.2.0.19.9`, Clang `1700.3.19.1` | macOS 26.0 | 11 pass / **3 fail** | 2 correctly rejected | 1 |
| Swift.org 6.4.2 | `swift-6.4.2-RELEASE` | macOS 27.2 | 14 pass / 0 fail | 2 correctly rejected | 0 |
| Swift.org 6.0.3 | `swift-6.0.3-RELEASE` | retained macOS 15.2 | 14 pass / 0 fail | 2 correctly rejected | 0 |
| **Xcode 27.3 beta, `27C5020n`** | **`swiftlang-6.4.2.1.9`, Clang `2100.4.1.1`** | **macOS 27.3** | **11 pass / 3 fail** | **2 correctly rejected** | **1** |

Each row is eight sources × two configurations: seven positive sources and one deliberately broken source. Across the five rows there are **80 compile/run cells: 61 positive passes, nine positive failures, and ten correct negative-control rejections**. All 80 compilations and all 80 executions exit zero. These counts describe selected test cases, not a statistical estimate of how much application code is affected.

The first four matrices were recorded at clean repository revision `27a045dd648c01a4ef9c5d8d36c5afc92f36f6fb`; the beta matrix at clean revision `314c722ff50b5367c069eb09ffbc1866764cd2c3`. All eight probe files still match those captured bytes at this handoff's baseline. The historical matrix driver and process helper also retain their recorded identities.

**Version labels cannot select a safe compiler.** Xcode 27.2 and 27.3 beta print the same Swift build string but have different frontend hashes; both fail. Swift.org 6.4.2 prints the same short language version but passes these cases. Conversely, one passing older compiler does not establish that all older versions are safe. The frontend and SDKSettings hashes are listed in section 12.

### Runtime and SDK controls

The Xcode 27.2 and Swift.org 6.4.2 runs used the same SDKSettings bytes and host, which makes their comparison especially useful. The CLT and 6.0.3 rows use different SDKs and are not a one-variable compiler substitution experiment.

Separate diagnostic executions of the original Release programs for the first four rows recorded the same system Swift runtime image, `/usr/lib/swift/libswiftCore.dylib`, UUID `C9B36DA7-BB34-35AF-BC5D-91ECBCAA9ED0`. The shared-cache dylib was not independently hashed as an ordinary file. This runtime-image comparison does not extend to the beta unless separately measured, and does not equate compiler-supplied standard-library modules or serialized optimization bodies. The wrong constant in emitted SIL is stronger evidence of compile-time divergence than the runtime comparison alone.

## 4. The source-shape matrix

All positive Debug cells returned `1.0` on every listed compiler. The Release results below group the three failing compilers because they produced the same observations; this grouping is descriptive, not a claim about all Apple compilers.

| Source | Deliberate difference | Xcode 27.2 / CLT 6.2 / Xcode 27.3 beta Release | Swift.org 6.4.2 / 6.0.3 Release |
| --- | --- | --- | --- |
| `LostStoreObserved.swift` | Literal array; `inout` buffer writer; read owning array after borrow | **2.0, wrong** | 1.0 |
| `ReturnInsideInout.swift` | Read buffer inside borrow, after `writeOne(&buffer)` | **2.0, wrong** | 1.0 |
| `CopyInoutReturnInside.swift` | `var copy = buffer`; call `writeOne(&copy)`; read original buffer | **2.0, wrong** | 1.0 |
| `ByValueReturnInside.swift` | Writer receives buffer descriptor by value; read inside borrow | 1.0 | 1.0 |
| `ByValueReadAfter.swift` | By-value writer; read owning array after borrow | 1.0 | 1.0 |
| `NonLiteralReadAfter.swift` | Initialize with `[Float](repeating: 2, count: 1)` | 1.0 | 1.0 |
| `OutOfLineRead.swift` | Read array through a separate `@inline(never)` reader | 1.0 | 1.0 |
| `NoWriteNegative.swift` | Deliberately omit the store | 2.0, correctly rejected | 2.0, correctly rejected |

The inside-borrow failure is consequential: “move the read into the closure” is insufficient when the writer still receives the descriptor `inout`. Copying the descriptor also fails. Neither operation clones the underlying data.

The by-value writer still mutates the pointed-to element. Its pass suggests that parameter convention matters for these examples; it does not prove that changing every API to by-value is a correct or complete repair. The repeating initializer still uses constant inputs, so it is not a runtime-unknown-data test. The out-of-line reader is a diagnostic boundary, not a production recommendation. Inlining, specialization, ownership lowering, and surrounding code can change the optimization shape; these controls do not establish a general workaround.

The acceptance driver requires process success **and the exact complete combined stdout/stderr line**, including the bits and newline. The no-write control must produce the known stale line and fail the positive predicate in both configurations. A zero exit, a test suite's aggregate green status, or a diagnostic label cannot substitute for observing the correct result.

## 5. What the generated code shows

The original, inside-borrow, copied-inout, and by-value-inside sources have retained optimized SIL and LLVM IR for every compiler. The other four sources have execution evidence but no retained intermediate dumps in these matrices.

For `LostStoreObserved.swift`, Xcode 27.2's optimized SIL constructs a `Float` from `0x40000000`, writes it as the initial element, calls the writer, and returns that earlier SSA value. Its relevant instructions are below; intervening instructions are omitted:

```text
%12 = float_literal $Builtin.FPIEEE32, 0x40000000 // 2
%13 = struct $Float (%12)
store %13 to %11
...
%30 = apply %18(%26) : $@convention(thin) (@inout UnsafeMutableBufferPointer<Float>) -> ()
...
return %13
```

Source artifact in the companion bundle: `evidence/xcode-27.2/LostStoreObserved.sil`, function `observedValue`, lines 149–193. The unchanged instruction lines corresponding to the excerpt are 164–166, 182, and 192; comments listing SSA users have been omitted above.

Swift.org 6.4.2's corresponding SIL retains the post-call element load and returns it:

```text
%30 = apply %18(%26) : $@convention(thin) (@inout UnsafeMutableBufferPointer<Float>) -> ()
...
%39 = ref_tail_addr [immutable] %38, $Float
%40 = load %39
...
return %40
```

Source artifact: `evidence/swiftorg-6.4.2/LostStoreObserved.sil`, lines 182, 191–192, and 195. Again, surrounding instructions and user comments are omitted.

LLVM IR confirms the same divergence. Xcode 27.2's `observedValue` keeps the call at line 110 but ends at line 114 with:

```llvm
ret float 2.000000e+00
```

The writer's definition still includes, at line 201:

```llvm
store float 1.000000e+00, ptr %2, align 4
```

Swift.org 6.4.2's caller instead contains, at lines 110 and 113:

```llvm
%4 = load float, ptr %2, align 8
ret float %4
```

The original-probe caller/writer LLVM function bodies in the retained Xcode 27.3 beta output have the same relevant behavior as Xcode 27.2. Their complete emitted files are not byte-identical: paths, SDK metadata, and generated identifiers can differ.

### Mechanistic hypothesis and its limits

A plausible failure chain is: an initialized element address becomes a raw pointer stored in a buffer descriptor; the descriptor is passed indirectly to an out-of-line function; the callee loads that pointer and writes the element; an analysis fails to model that write as potentially modifying the caller's earlier value; a load is then replaced with the initialization constant.

That is an explanation to test, not an identified compiler patch. A correct optimizer could also produce constant `1.0` if it proved the actual store; retaining a load is not itself the language-level acceptance criterion. The failing return of `2.0` is the decisive observation.

The retained failing and passing helper SIL both describe descriptor reads and global writes. Their LLVM descriptor parameter is marked `readonly` in both outputs. That alone does not mean the pointed-to element cannot be written, nor does it identify a bad LLVM attribute: the descriptor and its pointee are distinct memory objects. Researchers should examine the full effects/escape model and pass sequence rather than treating an attribute spelling as sufficient proof.

The dumps are final optimized SIL/IR from separately recorded emission commands, not before/after snapshots of every pass used to produce the executable. They establish a compiler-side wrong constant in the emitted representation. They do not establish the first bad pass, a complete frontend-versus-standard-library attribution, or the precise source revision of an Apple compiler.

## 6. Related upstream work — leads, not an exact match

The following primary sources were checked on November 6, 2026. The two issues are closed and both PRs are merged. Their status does not qualify any local binary.

- **[Swift issue #92562](https://github.com/swiftlang/swift/issues/92562):** reports optimized wrong results involving `MutableSpan` over `InlineArray`. Its report describes a main-branch/6.5-development regression and says its example did not fail on the referenced 6.3.3 or 6.4.x builds. It is not the same source as our Array/unsafe-buffer case, and its version range must not be copied onto our evidence.
- **[Swift PR #92580](https://github.com/swiftlang/swift/pull/92580):** changes alias analysis to account for effects reached through a pointer stored in an indirectly passed object. The explanation connects the missed write to redundant-load elimination. Merged September 24, 2026; GitHub reports merge commit `5de3e0a9273b35839314226d69af21b43ca2a4a9`. The linked implementation commit is `3cad37591a067ebc252fd1f36ea8f5bce8d52387`. This is a relevant mechanism to compare against the local SIL, not proof of a missing fix in Xcode.
- **[Swift issue #93141](https://github.com/swiftlang/swift/issues/93141):** describes a second `MutableSpan` formulation over a local value that still failed on a revision after #92580. Its published matrix differs from ours and includes passing 6.4.0. This shows why an earlier issue's closure is insufficient evidence for nearby forms.
- **[Swift PR #93168](https://github.com/swiftlang/swift/pull/93168):** adjusts escape analysis for addresses associated with scoped lifetime dependencies. Its explanation ties the missed escape to an incorrect redundant-load optimization. Merged October 12, 2026; GitHub reports merge commit `0dd56178a64c0a8b145650800b631f8ce9847754`. The linked implementation commit is `2cb42c38e43a552e7335d2241639c7ecb24d8667`. Determine whether this mechanism can reach our older Array/inout-buffer formulation; do not assume it does.

The implementation-commit and merge-commit identities are intentionally distinguished. The consulted sources do not establish which corresponding backports, if any, are present in each proprietary Xcode frontend. In particular, our local CLT 6.2 failure cannot be explained merely by declaring that it is the same newly introduced 6.5 regression. Possible shared mechanisms, distinct defects, or earlier manifestations require investigation.

Some existing project comments name RedundantLoadElimination and #92562 more firmly than this handoff's independently supported conclusion. One kernel comment explicitly labels the suggested backport relationship unverified. Treat those comments as investigation history; the runtime observations and retained intermediate code are the evidence. This handoff does not revise production comments or assertions.

## 7. Consequences for VectorCore and its clients

VectorCore is a Swift numerical-primitives library with generic vectors, optimized fixed-size storage, unsafe-buffer access, and mixed-precision operations. The intended release covers Apple platforms. Correct optimized memory behavior is a prerequisite for that support claim.

**The evidence does not show that nearly all computation is wrong.** Several nearby shapes pass on the affected compilers; historical broad suites also have many passing tests. What remains unbounded is the population of other source shapes that could trigger similar stale-value behavior after optimization. There is no defensible percentage of affected VectorCore operations from this matrix.

The concern extends beyond a library's own implementation. Source SwiftPM consumers compile the package with their chosen compiler. A producer's passing build does not pin the consumer's toolchain, and the pure-Swift client canary can fail without linking VectorCore at all.

Binary distribution would require its own investigation. `Sources/VectorCore/Vectors/Vector.swift` declares public generic `Vector<D: StaticDimension>` and exposes `@inlinable` bodies, including storage access. Such bodies can be optimized into clients, including specialized out-of-line copies. Therefore a producer-built artifact alone would not certify every importing compiler. This is an inference from the library's current source and Swift's documented compilation model, not a tested binary workaround. [Swift library-evolution guidance](https://www.swift.org/blog/library-evolution/).

### A concrete library site and its existing regression

`Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift`, `adaptiveEuclideanDistance`, constructs a one-element result array initialized to zero. Its current implementation calls the batch kernel using a by-value buffer argument and returns the result from inside the borrow. It compares that candidate with a direct FP32 reference before accepting it.

The existing `AdaptiveReducedResultTests.acceptedReducedResultIsReturned` deliberately makes the reduced result differ in bits from the reference while remaining within the permitted error. It then checks that the function actually returns the accepted reduced result. This matters because a stale zero could cause an unnoticed reference fallback under a small threshold; an ordinary accuracy-bound assertion could still pass. For a positive reference and a stale zero, the relative-error condition accepts zero exactly when `threshold >= 1` (assuming the other guards pass). This is a consequence of the shown acceptance expression, not a claim that the current hardened function fails on the beta.

That test protects one concrete behavior. It does not turn the failing inside-borrow/inout control into a pass, or establish that the chosen library form is immune under all compiler transformations.

### Existing safeguards and gaps

- `Tests/ComprehensiveTests/LostStoreCanaryTests.swift` keeps the deliberately vulnerable pure-Swift shape. In Debug and Release compilers at least 6.4.1, its assertion is hard. Release 6.4.0 and pre-6.4 branches use intermittent known-issue handling. Consequently aggregate CI can be green while a wrong value is recorded on those older branches. The standalone predicate has no version waiver.
- The same file contains `LostStoreCanaryCapturedVarTests` for K1–K5, K7, and K8, plus the reduced-result regression. Those controls are scoped evidence; they do not prove every captured-variable body safe.
- `Scripts/ci/tests/test_lost_store_reads.py` flags selected lexical patterns for review. It documents limits involving aliases, nested bodies, helper-mediated reads, other pointer APIs, and inlining. The copied-descriptor repro illustrates why a lexical scan is not a complete compiler-safety proof. Benchmarks are outside that scanner's source-tree selection.

Do not weaken a canary, widen a known-issue exception, or change a routing gate solely to obtain green output. Any eventual configuration change needs a renewed review of the code paths it makes reachable, including optimized clients and previously waived compiler branches.

## 8. Evidence chronology and separate Xcode problems

| Evidence | What it establishes | What it does not establish |
| --- | --- | --- |
| Nov 1, Swift.org 6.4.2, source `379e610f4365e820454c8c5328c4f87dfaa2031b` | Original/inside-borrow probes pass under selected ordinary optimization modes; full native Debug/Release and a five-check downstream client pass within the documented scope | Final-source, all-platform, or App Store qualification |
| Nov 4 phase-one matrix | Eight-shape comparison exposes the copied-descriptor failure and narrows the generated-code divergence | A general workaround or first responsible pass |
| Nov 5 four-compiler matrix, source `27a045dd…` | Fresh 64-cell results, exact compiler/SDK identities, unchanged probes, runtime-image observations | A newly accepted Xcode build |
| Nov 5 installed beta, source `314c722f…` | Fresh 16-cell beta matrix reproduces the same three Release failures | Successful Release numerical qualification |
| Current source `bdf99084…` | Integrates Xcode test-host and reaper-QoS repairs; probe/kernel inputs relevant to this compiler handoff remain unchanged | A new full Release qualification of this revision |

The Nov 1 ordinary downstream client checked the pure-Swift canary, fixed-vector borrow, dynamic-vector borrow, accepted reduced-result bits, and public SoA suffix behavior. Those historical passes remain useful controls, but must not be relabeled as executions of the current release candidate.

Two recent Xcode problems are separate from the silent wrong-result issue:

1. IDE compile errors involving `ProgressReporter`, `VCProgressReporter`, and `BenchmarkSuite` came from a partial rename in the older checkout at `/Users/goftin/dev/gsuite/VSK/VectorCore`. The reviewed worktree consistently uses `BenchmarkProgressReporter` and builds with the beta. The older checkout's edits were preserved.
2. Running the reviewed tests in Xcode exposed child-test launcher compatibility and reaper-priority problems. Those were repaired. The full Xcode Debug run reported **1,940 passed / 46 skipped / zero failed**, before the final QoS adjustment, with 98 warning records. The final adjustment received focused Xcode/SwiftPM checks and scoped ASan/TSan checks. It was not followed by a claimed full-suite rerun. Debug success does not override the separately failing optimized reproducer.

Those fixes changed test/benchmark-support infrastructure, not the numerical compiler predicate. No full native Release beta suite was repeated after its standalone compiler gate failed. Benchmark timing instability and missing platform coverage remain separate release gates; no performance result from an uncontrolled test run is used here to assess the compiler.

## 9. Reproduction instructions

### Minimal single-file test

The commands below describe a rerun; they were not executed while writing this document. Use a verified installed compiler and keep its exact identity. The shown path is this machine's installed beta. On another machine, change that path and record the new identity as a separate cell. No global `xcode-select` change is necessary.

From the repository root, or the companion bundle's `repository/` directory:

```sh
VC_DEVELOPER="/Users/goftin/Downloads/Xcode-beta.app/Contents/Developer"
VC_SWIFTC="$VC_DEVELOPER/Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc"
VC_FRONTEND="$VC_DEVELOPER/Toolchains/XcodeDefault.xctoolchain/usr/bin/swift-frontend"
VC_SDK="$VC_DEVELOPER/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.3.sdk"
VC_PROBE="Docs/reviews/2026-11-01-compiler-qualification/probes/LostStoreObserved.swift"
VC_WORK="$(mktemp -d "${TMPDIR:-/tmp}/vectorcore-lost-store.XXXXXX")"
export VC_BINARY="$VC_WORK/LostStoreObserved-release"

"$VC_SWIFTC" --version
shasum -a 256 "$VC_FRONTEND" "$VC_SDK/SDKSettings.json" "$VC_PROBE"
/usr/bin/sw_vers
/usr/bin/uname -m

"$VC_SWIFTC" -swift-version 6 -target arm64-apple-macosx14.0 \
  -sdk "$VC_SDK" -module-cache-path "$VC_WORK/module-cache" -O \
  "$VC_PROBE" -o "$VC_BINARY"

python3 - <<'PY'
import os
import subprocess
import sys
r = subprocess.run([os.environ['VC_BINARY']], capture_output=True, timeout=10)
sys.stdout.buffer.write(r.stdout)
sys.stderr.buffer.write(r.stderr)
expected = b'LOST_STORE observed=1.0 bits=3f800000 expected=1.0\n'
raise SystemExit(0 if r.returncode == 0 and r.stdout + r.stderr == expected else 1)
PY
```

The expected beta result is a zero exit from the Swift program followed by **exit 1 from the Python acceptance check**. Compile failure is also a failure; do not run an older leftover executable. The example uses a fresh directory to avoid that ambiguity. Use `-Onone -D DEBUG` and a separate executable name for the Debug control. Keep `@inline(never)` and every other source byte unchanged for the baseline comparison.

### Full retained matrix driver

The historical driver is `Docs/reviews/2026-11-04-release-phase-one/compiler/run.py`, with helper `Scripts/ci/v03_process.py`. Run it from a real VectorCore Git checkout, since it records the Git revision and status. The ZIP's `repository/` is only a selected source snapshot, not a complete Git checkout; the single-file instructions above work there, but the original full driver needs the real checkout.

```sh
python3 Docs/reviews/2026-11-04-release-phase-one/compiler/run.py \
  --swiftc "$VC_SWIFTC" \
  --frontend-sha256 93a7373a7f558b211b9820797d40d7554a0fef2c917d039fa0fa894b60fd55eb \
  --sdk "$VC_SDK" \
  --output "$VC_WORK/matrix"
```

The output directory must not already exist. The driver checks the supplied frontend hash before execution, requires all eight named probes, records input hashes and exact commands, and uses bounded subprocess execution. Its default compilation deadline is 60 seconds and each program's deadline is 10 seconds. It emits SIL and IR for the four selected shapes. `test_run.py` beside the driver checks the acceptance predicate and mandatory-probe inventory.

The shown hash is for this beta only. A new candidate requires its own authenticated identity and independently recorded hash; do not pretend it is the historical binary. Preserve failing observations and diagnostic variants under new paths instead of overwriting baseline evidence. A pass obtained by disabling an optimizer, changing the reproducer, or forcing Debug is diagnostic evidence only.

## 10. Questions for the research investigation

1. **Language semantics:** Is each failing program valid under the precise Swift API, exclusivity, pointer-lifetime, and memory-binding rules? Identify any actual violation with an authoritative rule and the exact offending expression. Distinguish descriptor access from access to its pointee.
2. **First bad transformation:** Where does the correct post-write observation first become the old constant? Obtain before/after SIL around the relevant passes and inspect alias, escape, memory-effect, COW, and specialization decisions. Confirm supported diagnostic flags for the exact compiler rather than assuming flags from another version.
3. **Upstream relationship:** Are #92580 and/or #93168 causally relevant to this Array case? Compare their tests and source changes. Trace backports with exact commits and branches where public evidence allows. Do not infer inclusion in an Xcode build from a short Swift version or merge date.
4. **Toolchain comparison:** Why does Swift.org 6.4.2 pass while Apple builds with a 6.4.2 label fail? Separate compiler changes from standard-library module bodies, build configuration, SDK, and runtime. The same-SDK Xcode/Swift.org pair is the best existing starting point.
5. **Affected scope:** Which conditions are necessary or sufficient: literal initialization, stack promotion, indirect buffer argument, pointer escape, out-of-line calls, direct reads, generic specialization, cross-module optimization, or element type? Expand systematically without replacing the original case. Include integer/double/runtime-input controls only as new evidence, since they are not part of the existing matrix.
6. **Workaround quality:** Can any narrowly scoped source change preserve semantics across optimization levels and clients, with actual regression tests and a demonstrated reason it works? Evaluate API/ABI and performance costs. Explain what is and is not protected; a private library workaround cannot automatically fix unrelated client code.
7. **Release path:** What exact compiler/SDK/host/runtime/distribution combinations could be supported now? Which need new execution evidence? What remaining evidence is required before calling a compiler or a workaround accepted?

Useful falsification tests include distinguishing a memory write that never occurs from a stale caller return, showing a semantic violation if one exists, locating a bad transformation earlier than the proposed pass, and testing whether an alleged upstream repair fixes the unchanged original and copied-descriptor cases. A nearby passing variant or lack of sanitizer complaints would not falsify the observed wrong result.

Requested research output: an evidence-ranked causal assessment; primary-source links and exact commits; an affected/unaffected/unknown matrix; a minimal next experiment sequence with discriminating outcomes; and support/workaround options with explicit limits. Please report uncertainty or missing public mapping rather than filling gaps with an assumed Xcode patch history.

## 11. Acceptance requirements and distribution decisions

The immediate next gate is an exact candidate compiler passing the unchanged positive matrix and rejecting both negative controls. After that, qualify current-source full Debug and normal Release builds, the hard canary, relevant library regressions, and ordinary optimized downstream clients. Inspect skipped and known-issue results separately from aggregate exit status.

Qualification then needs the chosen platform/architecture/runtime combinations, including minimum-OS execution where claimed. Compiler acceptance alone does not settle device, Intel, tvOS, visionOS, Catalyst, sanitizers, leaks, or GPU consumer integration. Performance acceptance needs fresh products built with the accepted compiler and a controlled, exclusive measurement window. Final hosted CI and consumer acceptance must bind one exact release-candidate source revision.

Current decision options are to hold the intended broad release scope pending a suitable accepted Xcode compiler, explicitly narrow support to a qualified exact toolchain/distribution scope, or investigate a carefully bounded library accommodation or binary product change. None is adopted by this write-up.

Swift.org's installer guidance currently requires the Xcode-bundled Swift compiler for App Store submissions. Thus a passing standalone Swift.org result does not by itself establish the intended App Store path. Exact Xcode/SDK submission eligibility and supported host requirements must be checked independently for any proposed release candidate. [Swift.org installer guidance](https://www.swift.org/install/macos/package_installer/).

## 12. Identity and evidence appendix

### Frontend and SDK identities

| Compiler | `swift-frontend` SHA-256 | `SDKSettings.json` SHA-256 |
| --- | --- | --- |
| Xcode 27.2 | `448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f` | `82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6` |
| CLT 6.2 | `720112638833dcb8e8190859cde6b9657ec88c720b7b2e4fc419764f7f9ca47b` | `7a2223423d955944a74542b8a72fda83cf970fa30749ba305fa904451f3a16c1` |
| Swift.org 6.4.2 | `956438a61893525d7ce28c21b6634093f15b0c4bae7fe6eb7d98fba0c1c5e301` | `82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6` |
| Swift.org 6.0.3 | `ed347b34041039d1ec6c4f0f387d5df668222d44c9a40d344905bfa311609b41` | `2fa5c0ce1bbcd261b132b572b1a9eece3b5905b04640a44deae1a6a8812928fb` |
| Xcode 27.3 beta | `93a7373a7f558b211b9820797d40d7554a0fef2c917d039fa0fa894b60fd55eb` | `b57cfc491bdb4fd38c1c044fa558b2578fb560d630219a82f76b9b181f45789c` |

The original driver SHA-256 is `7cdb2a22236419cc4c5ab7d8be49a6e41554a1b95c80a88dc8d3a0708745e2ef`; its process helper is `a261179fa36d65bbd28d591751423266bc179519ef483c4893cc8089e6d98a94`. Probe and executable hashes, full compiler output, target, SDK capture path, revision, and observed values are in the matrix JSON files. An SDKSettings hash identifies that metadata file, not every SDK byte.

### Companion bundle layout

The companion `VectorCore-compiler-research-handoff-2026-11-06.zip` contains:

- `README.md`, `RESEARCH-PROMPT.md`, and `matrix-summary.json` — this handoff and its derived index.
- `repository/` — the eight unchanged probes in their original relative locations, the historical driver/helper and driver tests, plus selected library/canary/scanner source and historical reports for context. This is a partial source snapshot, not a buildable release checkout or a Git history.
- `evidence/<compiler>/` — each original matrix summary, selected compile/run and SIL/IR-emission command records with their stdout/stderr, and the eight intermediate files for the four emitted shapes. The original records retain capture-time paths.
- `evidence/context/` — selected structured compiler/beta summaries and identity/context records; these preserve original source and test scopes.
- `provenance.json` — origin and hash of each copied input, with generated documents identified separately.
- `manifest.json` — hashes and sizes of every payload file, excluding the manifest itself and its verification receipt to avoid self-reference.
- `verification.json` — copy, inventory, observation, and manifest verification results from assembling this handoff. It is a packaging check, not a new compiler execution result.

Compiler/SDK payloads, executable binaries, module caches, full build trees, raw device records, and inherited-environment diagnostics are not included in this research bundle. Historical summaries may name such files or hashes because those summaries are preserved unchanged; not every historically referenced artifact is packaged here. The source archives remain the authority for the broader evidence selection.

The original durable compiler evidence is under `/Users/goftin/dev/projects/VectorCore-release-gates-2026-11-05-evidence/compiler/`. The beta evidence is under `/Users/goftin/dev/projects/VectorCore-xcode-27.3-beta-2026-11-05-evidence/`, principally `compiler-probes/`. Its final archive manifest SHA-256 is `f152f207c7777268ff1fbf083ce6508b356bd333898201b9295ac25c3ba804a1`. The bundle supplies the selected files needed to investigate the compiler issue without access to those local directories.

While preparing the handoff, all 80 cell observations, all 160 compile/run command records and their 320 stdout/stderr stream hashes, and all eight source identities across the five matrices were rechecked. Native compiler execution was not repeated. Older full-library passes, the beta matrix, and the later test-harness fixes retain their own revision/configuration scope throughout this document.
