# Decision proposal: bounded VectorCore qualification

Status: proposed investigation direction; not an adopted support policy or release approval.

## Decision requested now

Authorize further qualification of the exact stable Xcode 27.2 tuple under a
proposed usage restriction, retain the strict compiler gate, or investigate a
release limited to passing toolchain/platform tuples. The recommendation is the
first option, subject to the restrictions and open evidence below.

The current release-admission rule remains unchanged until an explicit later
admission decision. Choosing the first option now authorizes the next investigation
and a draft qualification mechanism; it does not certify Xcode or admit 0.4.0.

## New evidence that makes this a real support decision

The new API still permits valid Swift code that miscompiles on affected compilers:

```swift
var values: ContiguousArray<Float> = [2]
values.withUnsafeMutableElements { buffer in
    var descriptor = buffer
    writeInout(&descriptor) // out-of-line helper writes descriptor[0] = 1
}
return values[0] // observed 2 at -O in both tested Xcode compilers
```

The complete helpers and cases are in [ResidualExposure.swift](ResidualExposure.swift).
All pointers stay inside their borrows. The descriptor is not replaced and the
owner is not reentered. Moving the read inside the borrow also fails, as does
placing the pointer in another struct passed inout. These are residual failures
of valid API compositions, not lifetime/bounds violations or caller misuse.

Both Xcode Release clients fail those three cases. All seven by-value controls
pass in all six compiler/client configurations. The other tested inout shapes
also pass, including actual optimized vector storage and a client-local custom
conformer bridge. That contrast demonstrates source-shape sensitivity; it does
not justify exceptions for the inout forms that happen to pass.

## Options

| Direction | Consequence | What remains before any release |
| --- | --- | --- |
| Continue bounded qualification of stable Xcode 27.2 (recommended investigation) | Explore support for shipped operations and documented by-value borrowed-view mutation patterns, with the additional compiler-specific restriction below. Keep the unsafe APIs available. | Validate the boundary across required routes, consumers and platforms; design a strict qualification result distinct from compiler correctness; user approval of final support wording and admission. |
| Preserve the strict compiler gate | Every original positive compiler probe must pass for an accepted tuple. Current affected Xcode tuples stay rejected. | Find and qualify a suitable passing Xcode tuple, or separately choose a narrower distribution scope. Other release preparation can continue. |
| Qualify only passing toolchain/platform tuples | Investigate a deliberately narrower release, initially using the tested Swift.org 6.4.2 native macOS arm64 tuple. | Explicit support/distribution changes and remaining platform/runtime/performance/hosted acceptance. Native Swift.org results do not establish the intended Xcode/App Store path. |

The official [Swift installer guidance](https://www.swift.org/install/macos/package_installer/)
requires App Store apps to use Xcode's bundled Swift. Source SwiftPM consumers
also compile with their own selected compiler. The existing source package cannot
pin that compiler for them. Binary distribution is a separate project and would
still need generic/inlinable importer qualification.

## Proposed restriction for the first option

On a specifically qualified affected tuple, user/conformer/helper code operating
on a borrowed pointer, buffer descriptor, span/view or pointer-bearing wrapper
must not add an inout handoff of that descriptor/view to another mutating helper.
This restriction applies transitively through the callback/helper/custom-conformer
call chain. Element-only helper functions must receive those views by value.

The specific reviewed borrow-entry adapter is a qualification starting point:
ContiguousArray's stdlib borrow supplies a descriptor-inout callback, and our
adapter forwards that supplied view to the user callback by value. The proposed
restriction concerns additional descriptor/view inout handoffs after that borrow
entry. This narrowly identified root is not a universal exemption for custom
adapters merely because they eventually forward a view by value.

This is conservative and transitive. An outer by-value function signature alone
does not establish compliance, and a passing inout example does not earn an
exception. Unqualified custom conformers, opaque imported implementations and
generic pointer-bearing state would need their own investigation. Owning-array
and owning-vector inout operations remain a separately tested category; ownership
alone is not proof against hidden borrowed-view forwarding.

This proposed exclusion is not a proved complete safe/unsafe partition. The
passing observations establish only the tested forms. Tuples that pass the
retained compiler probes would not require this additional workaround restriction
on that basis; they still need normal VectorCore qualification.

Hiding concrete optimized storage would remove one convenient route, but would
leave reconstruction inside other unsafe callbacks, raw buffer/handle access and
external conformer implementations. No additional public API change is proposed
at this decision point.

## Required qualification mechanism, if investigation proceeds

1. Record separate outcomes for compiler probes and bounded VectorCore
   qualification. An affected compiler remains recorded as failing its probes.
2. Bind the exact source candidate, compiler/frontend, SDK, runtime, architecture,
   target, language/optimization flags and producer/client products. Do not admit
   a marketing version range or infer acceptance of a newer binary.
3. Enumerate every known diagnostic case and exact observation. Unknown failures,
   missing/duplicate rows, crashes, timeouts or altered inventories block
   qualification. Keep raw native canary failures rather than turning them into
   passing test assertions.
4. Keep supported product/consumer numerical expectations strict, including
   captured-variable assertions, negative controls, generic/inlinable clients and
   independent witnesses. No waiver based solely on suite name, compiler version
   or upstream issue number.
5. Re-derive existing CI reachability: Xcode 16.2/26.3 legs and old known-issue
   branches, optional C kernels, architecture-dependent code, platform builds and
   required-job aggregation. A green current CI aggregate is insufficient.
6. A new failure in a supported route revokes that tuple's bounded qualification
   until investigated. Changes to source/toolchain/SDK/flags/platform require
   renewed scoped evidence. Downstream code outside the assessed implementation
   chains cannot be certified by this repository's source scanners.
7. Complete platform/runtime, sanitizer, controlled performance, exact-candidate
   hosted CI and downstream release-reference checks independently.

## Subsequent user decisions

- **Coverage:** Which minimum-OS, Intel and representative physical-device runtime
  checks must close before release? Existing platform support is not narrowed by
  this proposal. Historical pre-migration builds do not qualify the new candidate.
- **Infrastructure:** Whether to build an app-hosted physical iOS consumer/test
  harness; existing tool-hosted Swift package tests cannot execute there.
- **Admission:** Review the completed evidence, exact support wording and any
  qualification-rule change before accepting an affected tuple for release.

The first practical target would be the exact released Xcode 27.2 compiler, not
automatic beta acceptance. The beta remains comparative evidence. Current required
legacy CI tuples must be requalified or changed through a separate explicit
decision; they are not silently dropped.
