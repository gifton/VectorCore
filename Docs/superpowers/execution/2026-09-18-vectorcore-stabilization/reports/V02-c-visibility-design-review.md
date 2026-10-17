# V02 C-on visibility repair — bounded design review

**ACCEPT the proposed three-annotation repair, design only.** Actual corrected compilation and runtime remain unverified until the next frozen source/evidence package. No edits, builds or probes were performed.

The retained v6 Debug C-on unique-errors record contains16 diagnostics from two existing `@inlinable` helpers in QuantizedKernels.swift: Euclidean INT8 at581–585 and fused dot/magnitude accumulation at770–774. They reference an internal bridge, internal enum and internal static method that lack ABI visibility. This is an actually reached C-on compilation failure, not a failed numerical test. The reported successful v6 Debug C-off cell remains a separate result.

## Minimal sufficient source change

Add only `@usableFromInline` to:

- `Interop/PointerBridge.swift` existing internal `withSIMD4Int8Pointer<R>`;
- `Interop/CKernels.swift` existing internal `CKernels` enum;
- that enum's existing internal `dotInt8(_:_:lanes:)` static method.

All three are directly named by serialized inlinable bodies, so both the enclosing enum and method need visibility. Their signatures contain only standard-library public types and an unconstrained generic result. The bridge and dot wrapper bodies remain non-`@inlinable`; making their declarations usable from inlinable code does not serialize those bodies or recursively require every implementation detail to receive the annotation. The existing bridge `@inline(__always)` optimization hint is not itself `@inlinable`. No extra initializer, stored member, nested type or private alignment helper appears in these signatures. The C function call stays in the non-inlinable wrapper body.

This preserves source-level internal access, names/signatures and numerical bodies while making the declarations ABI-visible for the existing inlinable references. It is not a new public source API, although ABI exposure is a real consequence and should be recorded. Removing the caller's inlinable annotation or widening the wrappers to public is unnecessary for this correction. Do not annotate unrelated FP32 methods, alter gates/defaults, or change the pointer rebinding implementation.

## Actual call inventory and bounds

Source-wide searches for `withSIMD4Int8Pointer` and `CKernels.` find the bridge/dotInt8 production calls only in the two helpers above. The three public INT8 vector widths reach those helpers through their existing concrete metric wrappers. Equal symmetric scales admit the Euclidean C identity; unequal scales use its existing affine fallback. Symmetric zero-point-zero dot/cosine admit fused accumulation independently of matching scales; asymmetric/nonzero-zero-point values retain fallback behavior.

No production or benchmark caller of CKernels' FP32 dot/L2/range wrappers was found. **There is an existing test-only call**: `Tests/ComprehensiveTests/CKernelsSmokeTests.swift:testDot512Stub` invokes `CKernels.dot512` under VC_USE_C_KERNELS. Therefore “FP32 wrappers remain wholly uncalled” would be inaccurate; the precise claim is that this repair adds no FP32 production route and the proposed focused quantization filters do not select that smoke test. The other FP32 wrapper methods remain without executable source callers in this inventory.

For fixed public widths up to1536, Int8 self-dot and squared-difference totals fit Int32 (1536×16384 and1536×65025 respectively). That supports the selected route fixtures, not arbitrary custom QuantizedVectorINT8 lengths or raw C pointer/count safety. Existing comments about general hardware, cache or overflow behavior are not newly established by visibility annotations.

## Required closure evidence

Freeze the exact two-file, three-annotation delta and mechanically confirm unchanged function bodies, signatures, source access and numerical/gate files. Preserve the16-error C-on attempt as compile failure. Execute C-on Debug and Release focused `QuantizationRangeContractTests|QuantizedMetricRangeTests` with actual target compile flags and retained command/log/bundle associations. These existing tests supply all-width affine/wide/range/rounding controls for the newly compiled branches; keep child deadlines and counts qualified as in their accepted harnesses.

Complete all four Debug/Release × C-off/on public-route cells against the corrected freeze, retaining real forwarding counts and independent numerical/tail oracles. This closes the inlinable visibility class exposed by the gate and validates the actual public routes without activating dormant production FP32 wrappers. A numerical failure discovered after successful compilation requires separate root adjudication; do not treat this design acceptance as permission to repair C kernels or loosen test tolerances. No cross-platform execution, raw-length universality or performance claim follows.
