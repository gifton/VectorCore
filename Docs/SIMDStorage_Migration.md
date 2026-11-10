# SIMDStorage mutable-buffer migration for 0.4.0

The forthcoming 0.4.0 API replaces the `SIMDStorage` mutable-borrow requirement
with a distinct name and a by-value buffer descriptor:

```swift
mutating func withUnsafeMutableElements<R>(
    _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
) rethrows -> R
```

This is a source and binary compatibility change. Update custom conformers and
generic callers, then rebuild VectorCore and its consumers together. There is no
default implementation that falls back to the old requirement.

## Generic callers

Replace a call to `storage.withUnsafeMutableBufferPointer` when `storage` is
constrained by `SIMDStorage` with `storage.withUnsafeMutableElements`. For example:

```swift
import VectorCore

func replaceFirst<S: SIMDStorage>(_ storage: inout S, with value: S.Element) -> Int {
    storage.withUnsafeMutableElements { buffer in
        if !buffer.isEmpty { buffer[0] = value }
        return buffer.count
    }
}
```

The owning `storage` remains `inout`: this permits mutation of the container and
its value/COW state. The callback's buffer descriptor is passed by value. A
nonthrowing callback needs no `try`; a throwing callback propagates its error.

`VectorProtocol` and `VectorStorage` keep their existing by-value
`withUnsafeMutableBufferPointer` APIs. Do not rename unrelated calls simply
because their method has the same spelling. Code generic over
`OptimizedVector` can use `vector.storage.withUnsafeMutableElements` through its
`Storage: SIMDStorage` constraint.

## Custom conformers

Rename the old mutable-borrow implementation and remove `inout` from its callback
parameter. Do not merely add a convenience method that depends on the removed
requirement. A conformer backed by ContiguousArray can forward as follows:

```swift
import VectorCore

struct MyStorage<Element>: SIMDStorage {
    var elements: ContiguousArray<Element>
    var count: Int { elements.count }

    subscript(index: Int) -> Element {
        get { elements[index] }
        set { elements[index] = newValue }
    }

    func withUnsafeBufferPointer<R>(
        _ body: (UnsafeBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        try elements.withUnsafeBufferPointer(body)
    }

    mutating func withUnsafeMutableElements<R>(
        _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        try elements.withUnsafeMutableElements(body)
    }
}
```

Conformers must call the callback exactly once, including for an empty buffer,
and return its result or propagate its error. The mutation contract does not
promise rollback: element writes made before a thrown error remain mutations.
The buffer's lifetime is limited to the callback. Do not let its pointers escape,
index outside its bounds, or access the same storage through another route while
the mutable borrow is active. The protocol cannot mechanically enforce these
obligations for an arbitrary external conformer.

An old-only implementation no longer satisfies `SIMDStorage`, and an old-method
call through a generic `S: SIMDStorage` no longer compiles. The positive and two
negative `SIMDStorage*` fixtures in `Scripts/ci/fixtures/compile-contracts/` check
that transition through the existing consumer smoke job and the retained
`consumer_buffer_value.py` driver.

## Element-mutating helper functions

A helper that only changes elements should take its buffer by value:

```swift
func writeFirst(_ buffer: UnsafeMutableBufferPointer<Float>) {
    if !buffer.isEmpty { buffer[0] = 1 }
}
```

Pass `buffer`, without `&`, to such a helper. Copying the descriptor to a local
variable and passing that copy `inout` can reintroduce the source shape under
investigation. The new API does not prohibit a caller from doing that itself.

## Compiler and compatibility limits

ContiguousArray still has the standard-library `withUnsafeMutableBufferPointer`
method. The library adds no same-name overload, so an ordinary inferred standard
borrow remains unambiguous. Public concrete vector storage can still reach that
standard-library method; this migration does not remove it or repair the compiler.

The original compiler canary and concrete inout diagnostic remain separate from
the by-value API checks. The [policy adopted on 2026-11-10](reviews/2026-11-10-compiler-acceptance/decision.md)
permits bounded qualification of exact Xcode 27.2 build 27B52 with documented
restrictions; the compiler remains known-failing. Additional descriptor/view or
pointer-bearing-wrapper inout handoffs are excluded transitively through helper
and conformer chains. Arbitrary external implementations do not inherit
qualification from this by-value signature. Passing the new API tests alone does
not qualify a candidate or admit a release. The
[migration validation report](reviews/2026-11-06-buffer-api-migration/README.md)
records exact source/toolchain evidence and remaining limits; the
[earlier audit](reviews/2026-11-06-workaround-audit/README.md) records why a
same-name overload and an additive legacy fallback were not selected.

`SIMDStorageMutationTests` exercises the public mutation contract in the native
suite. The separate-module consumer checks generic witness dispatch, concrete
and external-conformer calls, COW/tail behavior, empty buffers, throwing callbacks,
and independent omitted-write controls. These bounded checks are not a proof for
all compilers, external conformers or unsafe-pointer programs.
