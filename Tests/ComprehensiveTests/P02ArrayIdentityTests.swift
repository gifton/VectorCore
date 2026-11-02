import Foundation
import Testing
@testable import VectorCore

// P02 phase B: the type-identity array helper `exactArray(_:as:)` (Sources/VectorCore/Operations/ArrayIdentity.swift,
// root ruling RD4). The unspecialized wrappers forbid generic specialization, as P02 phase 1 probe B did, so the
// helper runs on its generic path even in Release. Allocation (zero requests) is observed by the phase B probe, not
// here.

@_semantics("optimize.sil.specialize.generic.never") @inline(never)
func p02UnspecializedExactArray<V>(_ values: [V]) -> [Vector512Optimized]? {
    exactArray(values, as: Vector512Optimized.self)
}

@_semantics("optimize.sil.specialize.generic.never") @inline(never)
func p02UnspecializedConditionalCast<V>(_ values: [V]) -> [Vector512Optimized]? {
    values as? [Vector512Optimized]
}

func p02BaseAddress<T>(_ values: [T]) -> UInt {
    values.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress) }
}

@Suite("P02 exactArray type identity")
struct P02ArrayIdentityTests {
    static let rows: [Vector512Optimized] = (0..<64).map { Vector512Optimized(repeating: Float($0)) }

    /// Same element type: the result is the same array value, sharing the source buffer, through the generic
    /// helper directly and through an unspecialized generic caller.
    @Test func sharesTheSourceBufferOnSuccess() throws {
        let source = Self.rows
        let direct = try #require(exactArray(source, as: Vector512Optimized.self))
        let viaGeneric = try #require(p02UnspecializedExactArray(source))
        for result in [direct, viaGeneric] {
            #expect(result.count == source.count)
            #expect(p02BaseAddress(result) == p02BaseAddress(source))
            #expect(result == source)
        }
        let empty = try #require(exactArray([Vector512Optimized](), as: Vector512Optimized.self))
        #expect(empty.isEmpty)
    }

    /// Another element type: nil, decided by type identity alone (an empty array of another type is nil too).
    @Test func nilForAnotherElementType() {
        #expect(exactArray((0..<4).map { _ in Vector768Optimized() }, as: Vector512Optimized.self) == nil)
        #expect(exactArray([Vector768Optimized](), as: Vector512Optimized.self) == nil)
        #expect(p02UnspecializedExactArray([DynamicVector([1, 2])]) == nil)
        #expect(p02UnspecializedExactArray([Float](repeating: 1, count: 512)) == nil)
    }

    /// Optional elements: nil. `as? [T]` differs here: it converts element by element, so `[T?]` without nil
    /// elements casts successfully (observed below), which is the semantic difference the helper documents.
    @Test func nilForOptionalElements() {
        let optional: [Vector512Optimized?] = Self.rows.map { $0 }
        #expect(exactArray(optional, as: Vector512Optimized.self) == nil)
        #expect(p02UnspecializedExactArray(optional) == nil)
        #expect(p02UnspecializedConditionalCast(optional)?.count == optional.count)
    }
}
