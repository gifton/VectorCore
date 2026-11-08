import Testing
import VectorCore

private enum SIMDMutationFailure: Error, Equatable { case expected(Int) }

@inline(never)
private func replaceFirstElement<Element>(_ buffer: UnsafeMutableBufferPointer<Element>, with value: Element) {
    if !buffer.isEmpty { buffer[0] = value }
}

@inline(never)
private func mutateSIMDStorage<S: SIMDStorage>(_ storage: inout S, replacingWith value: S.Element) -> Int {
    storage.withUnsafeMutableElements { buffer in
        replaceFirstElement(buffer, with: value)
        return 731
    }
}

// No legacy mutable-borrow method: a conformer must be able to satisfy only the
// new requirement. The separate consumer fixture checks this across modules too.
private struct NewMutationStorage<Element>: SIMDStorage {
    var values: ContiguousArray<Element>
    var count: Int { values.count }
    subscript(index: Int) -> Element {
        get { values[index] }
        set { values[index] = newValue }
    }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R {
        try values.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableElements<R>(_ body: (UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R {
        try values.withUnsafeMutableElements(body)
    }
}

@Suite struct SIMDStorageMutationTests {
    @Test func concreteAndGenericReadAfterMutation() {
        var concrete: ContiguousArray<Float> = [2]
        concrete.withUnsafeMutableElements { replaceFirstElement($0, with: 1) }
        #expect(concrete[0].bitPattern == 0x3f800000)

        var generic: ContiguousArray<Float> = [2]
        #expect(mutateSIMDStorage(&generic, replacingWith: 1) == 731)
        #expect(generic[0].bitPattern == 0x3f800000)
    }

    @Test func newOnlyConformerUsesTheRequirement() {
        var external = NewMutationStorage<Float>(values: [2, 3, 4])
        let original = external
        #expect(mutateSIMDStorage(&external, replacingWith: 1) == 731)
        #expect(external.values.map(\.bitPattern) == [0x3f800000, 0x40400000, 0x40800000])
        #expect(original.values.map(\.bitPattern) == [0x40000000, 0x40400000, 0x40800000])
    }

    @Test func callbackRunsOnceIncludingEmptyStorage() {
        for size in [0, 1, 3] {
            var values = ContiguousArray(repeating: Int(2), count: size)
            var calls = 0
            let returned = values.withUnsafeMutableElements { buffer in
                calls += 1
                #expect(buffer.count == size)
                if !buffer.isEmpty { buffer[0] = 1 }
                return "borrow-result"
            }
            #expect(calls == 1)
            #expect(returned == "borrow-result")
            #expect(values.count == size)
            #expect(Array(values) == (size == 0 ? [] : size == 1 ? [1] : [1, 2, 2]))
        }
    }

    @Test func cowCopyAndUntouchedTailArePreserved() {
        let original: ContiguousArray<Float> = [2, 3, 4]
        var changed = original
        changed.withUnsafeMutableElements { replaceFirstElement($0, with: 1) }
        #expect(changed.map(\.bitPattern) == [0x3f800000, 0x40400000, 0x40800000])
        #expect(original.map(\.bitPattern) == [0x40000000, 0x40400000, 0x40800000])
    }

    @Test func throwingBorrowKeepsMutationAndOriginalError() {
        let original: ContiguousArray<Float> = [2, 3, 4]
        var changed = original
        do {
            try changed.withUnsafeMutableElements { buffer in
                replaceFirstElement(buffer, with: 1)
                throw SIMDMutationFailure.expected(731)
            }
            Issue.record("Expected callback error")
        } catch {
            #expect(error as? SIMDMutationFailure == .expected(731))
        }
        #expect(changed.map(\.bitPattern) == [0x3f800000, 0x40400000, 0x40800000])
        #expect(original.map(\.bitPattern) == [0x40000000, 0x40400000, 0x40800000])
        // A subsequent mutation also verifies that the throwing borrow ended.
        changed.withUnsafeMutableElements { $0[1] = 5 }
        #expect(changed[1].bitPattern == 0x40a00000)
    }

    @Test func doubleAndSIMDElementsRetainTheirTypes() {
        var doubles: ContiguousArray<Double> = [2, 3]
        #expect(mutateSIMDStorage(&doubles, replacingWith: 1) == 731)
        #expect(doubles.map(\.bitPattern) == [0x3ff0000000000000, 0x4008000000000000])

        var lanes: ContiguousArray<SIMD4<Int8>> = [SIMD4(2, 4, 6, 8), SIMD4(10, 12, 14, 16)]
        #expect(mutateSIMDStorage(&lanes, replacingWith: SIMD4(1, 3, 5, 7)) == 731)
        #expect(Array(lanes) == [SIMD4(1, 3, 5, 7), SIMD4(10, 12, 14, 16)])
    }

    @Test func optimizedVectorAssociatedStorageUsesNewRequirement() {
        func mutate<V: OptimizedVector>(_ vector: inout V) {
            vector.storage.withUnsafeMutableElements { buffer in
                replaceFirstElement(buffer, with: SIMD4<Float>(1, 3, 5, 7))
            }
        }
        var vector = Vector512Optimized(repeating: 2)
        let original = vector
        mutate(&vector)
        #expect(vector.storage.count == 128)
        #expect(Array(vector.toArray().prefix(5)).map(\.bitPattern)
                == [0x3f800000, 0x40400000, 0x40a00000, 0x40e00000, 0x40000000])
        #expect(original.toArray().allSatisfy { $0.bitPattern == 0x40000000 })
    }
}
