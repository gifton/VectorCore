//
//  ArrayIdentity.swift
//  VectorCore
//

/// `values` as `[Target]` when `Element` is exactly `Target`, otherwise nil. Never copies the elements.
///
/// Why the reinterpretation is sound: the guard admits only `Element.self == Target.self`, so `[Element]` and
/// `[Target]` are the same type with the same layout, and `unsafeBitCast` returns the same array value (sharing its
/// buffer). Every other element type returns nil. That differs from `values as? [Target]` and `values is [Target]`,
/// which convert element by element, in these cases: `[Target?]` with no nil elements, existential or `Any` element
/// arrays and class upcasts, where the cast succeeds; and an EMPTY array of another element type, where the cast
/// succeeds vacuously (`P02ArrayIdentityTests.nilForOptionalElements` and `nilForAnotherElementType` pin nil). At the
/// converted sites the empty case is shielded, by a different guard at each kind of site (P02 final review C-M2):
/// the pairwise identity chain runs only after `pairwiseDistances`' `guard n > 0`; the `.gram` admission
/// (`Operations.gramAdmission`) runs only after `findNearestBatch`'s checks that both arrays are nonempty;
/// `SoAFP16.init(vectors:)` converts only behind `VectorType.self == T.self`, which makes the test an identity; and
/// every other site tests `query as? T` on the same generic type first.
///
/// It replaces the conditional array casts (`as? [T]`, `is [T]`, `as! [T]`), which copied the whole handle array in
/// unspecialized generic code: P02 phase 1 probe B measured one request of 16 * count + 32 bytes per cast, on success
/// and on failure, on one host with the former 16-byte vector handle (the handle stride is the storage reference's
/// since P02 phase F). Since P02 phase E every array-target type test in Sources/VectorCore uses it (`Operations`,
/// `BatchOperations`, `MixedPrecisionKernels` and the `.gram` admission `Operations.gramAdmission`), and
/// `Scripts/ci/tests/test_no_array_conditional_casts.py` fails on the cast forms it lists there (`as?`, `as!`, `is`
/// and `case`/`catch` pattern casts to `[`, `Array<` or `ContiguousArray<` targets, including parenthesized targets
/// and casts inside string interpolations; a typealias'd target is one of its stated limits). Its allowlist is empty.
/// Tests: `P02ArrayIdentityTests` (shared buffer, nil for another type, nil for Optional elements); the zero-request
/// observation of the library helper is a P02 phase B probe (`P02/phaseB`), not a test.
@inline(__always)
internal func exactArray<Element, Target>(_ values: [Element], as _: Target.Type) -> [Target]? {
    guard Element.self == Target.self else { return nil }
    return unsafeBitCast(values, to: [Target].self)
}
