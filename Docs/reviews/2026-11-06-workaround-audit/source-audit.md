# Source audit of unsafe-buffer mutation boundaries

Source: `bdf990847babf567be1cbce22d2606e7102b1a47`. These findings concern source reachability; numerical results are recorded separately in the validation report. Three Astra reviewers examined the production call graph, external-consumer contracts and mechanical guards independently.

## Exposed public requirement

`Sources/VectorCore/Protocols/SIMDStorage.swift` declares:

```swift
mutating func withUnsafeMutableBufferPointer<R>(
    _ body: (inout UnsafeMutableBufferPointer<Element>) throws -> R
) rethrows -> R
```

The unconditional `ContiguousArray: SIMDStorage` conformance obtains its witness from the standard library. `Element` is unconstrained, so a client can use `ContiguousArray<Float>` through this protocol. `OptimizedVector.Storage: SIMDStorage` also makes the requirement reachable through that associated type. The optimized vector types expose concrete `ContiguousArray<SIMD4<Float>>` storage, which independently exposes the standard-library method.

The audit found no production call to the mutable requirement through a generic `SIMDStorage` or `OptimizedVector` constraint. This does not hide the public exposure: an ordinary importing client can call it. The fresh baseline consumer does so and demonstrates the stale value on Xcode 27.2 at `-O`.

Changing the callback to by value changes the public requirement's function type. It must be evaluated against the standard-library witness, custom external conformers, explicit function references and overload resolution. An additive differently named adapter preserves existing conformers but leaves the original route available. Removing the requirement does not remove the concrete standard-library API available through public storage.

The isolated Xcode 27.2 compatibility probes now establish these specific outcomes:

| Reduced interface change | Observed compilation result |
|---|---|
| By-value requirement, unchanged empty ContiguousArray conformance | Rejected: ContiguousArray does not conform. |
| By-value requirement plus explicitly typed same-name adapter | Compiles. |
| Same adapter plus ordinary inferred concrete closure | Rejected: ambiguous `withUnsafeMutableBufferPointer` call. |
| Same adapter plus explicitly typed legacy inout callback | Compiles; concrete standard-library overload remains accessible. |
| Existing custom conformer, original requirement | Compiles. |
| Same conformer, changed by-value requirement | Rejected: custom type does not conform. |
| Differently named additive adapter, original requirement retained | Compiles with both ContiguousArray and the custom conformer. |

The four compiling variants were also executed at `-Onone` and `-O`; all printed the expected values. The additive probe's retained legacy call happens to pass in that reduced body. It does not contradict the failing actual-library exposure probe or prove that the old requirement is repaired. Sources, exact diagnostics and runtime observations are retained in `VectorCore-workaround-audit-2026-11-06-evidence/compatibility/` beside the reviewed worktree. These are reduced interfaces, not a built production migration or complete source-compatibility proof.

## Existing by-value routes

| Surface | Current boundary | Audit implication |
|---|---|---|
| `VectorProtocol`, `VectorStorage` | Mutable-buffer callback receives descriptor by value | The ordinary vector abstraction already expresses element mutation without descriptor replacement. |
| `Vector<D>`, `DynamicVector` | Forward by-value callbacks to their storage | Public generic/inlinable code still needs consumer optimization checks. |
| `Vector{384,512,768,1536}Optimized` | Borrow concrete SIMD storage, construct Float buffer, call user callback by value | No explicit `&simd4Buffer` is forwarded to a writer. |
| `VectorMath.safeDivide` | Uses `Vector<D>.Storage == DimensionStorage` | The apparent storage call is a by-value callback, not the SIMDStorage witness. |
| Quantized vector wrappers | By-value Float callback over temporary storage, then requantization | Throw/commit semantics must be preserved if these wrappers change. No change is needed for the investigated descriptor signature. |
| `MixedPrecisionKernels.batchEuclidean512` | Output buffer is by value | The retained adaptive reduced-result test must verify the chosen result, not just an accuracy bound. |
| `MatrixDistance` helpers | Raw element pointers are by value | Owning output arrays remain `inout` for value/COW semantics. |
| Gram batch search | By-value pointer helpers inside synchronous scratch borrows | Inside-borrow observation alone fails in retained inout controls. The repeating-initializer control passes in its tested shape; neither observation establishes a general workaround guarantee. |

## Necessary mutations retained

The audit covered descriptor spellings, production typealiases, callback and protocol declarations, generic `inout T`/`Self`, `_modify` yields, and address-of uses. It found no production pointer-descriptor alias. A source scan does not prove arbitrary specialization or alias behavior.

- `TopKBuffer` owns arrays and a mutable size; merging can replace the aggregate.
- Normalization, batch-distance and autotuning routines mutate owning arrays/vectors, preserving normal COW and binding semantics.
- `ThreadSafeConfiguration.modify` intentionally permits replacement of generic state. It is not a buffer-only interface; a generic state could nevertheless contain pointers.
- Allocation calls such as `posix_memalign(&rawPtr, ...)` genuinely initialize a pointer binding. COW checks use `isKnownUniquelyReferenced(&owner)`.
- Inline storage borrows mutable bytes of its actual tuple storage. Scalar accumulators, `Hasher`, growing `Data`, counters and C output parameters mutate their values.

Removing `inout` from these unrelated operations would alter their semantics and is not a remedy for the descriptor bug.

## Bounded result and remaining concerns

The by-value recipe changes the writer boundary, not the compiler:

```swift
@inline(never)
func writeOne(_ buffer: UnsafeMutableBufferPointer<Float>) {
    buffer[0] = 1
}

values.withUnsafeMutableBufferPointer { buffer in
    writeOne(buffer)
}
```

The existing public requirement remains a known exposure during this audit. It is retained explicitly for compatibility, not treated as safe by an allowlist. New declaration checks must reject unreviewed repetitions and stale exceptions; they cannot establish that arbitrary pointer-bearing aggregates, imported declarations or generic instantiations are compiler-safe.

Some historical comments describe repeating allocation or inside-borrow observation as lost-store rules. Those are descriptions of particular source shapes, not general safety guarantees. This audit preserves the production source and records the limitation here so the native validation stays bound to one unchanged source revision.

The external Linux Swift 6.2.1 reproduction in the supplied research is useful corroboration. Its raw artifacts were not provided and are not counted as locally executed or hashed evidence. The original compiler research handoff and ZIP remain unchanged.
