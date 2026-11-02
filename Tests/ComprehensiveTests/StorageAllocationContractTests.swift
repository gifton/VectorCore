import Testing
import os
@testable import VectorCore

private struct AllocationDim0: StaticDimension {
    static let value = 0
}
private struct AllocationDim1: StaticDimension {
    static let value = 1
}
private struct AllocationDim9: StaticDimension {
    static let value = 9
}
private struct AllocationDim17: StaticDimension {
    static let value = 17
}
private final class AllocationBorrowError: Error {}

@Suite struct StorageAllocationContractTests {
    private func contents<S: VectorStorage>(_ value: S) -> [S.Scalar] {
        value.withUnsafeBufferPointer { Array($0) }
    }

    private func checkBorrow<S: VectorStorage>(_ initial: S) throws {
        var value = initial
        let original = contents(initial)
        var expected = original
        let retained = value
        try withExtendedLifetime(retained) {
            var calls = 0
            let result = value.withUnsafeMutableBufferPointer { buffer in
                calls += 1
                for i in buffer.indices { buffer[i] = S.Scalar(i + 11) }
                return 731
            }
            expected = original.indices.map { S.Scalar($0 + 11) }
            #expect(calls == 1 && result == 731)
            #expect(contents(value) == expected)
            #expect(contents(retained) == original)
            let error = AllocationBorrowError()
            let beforeThrow = value
            try withExtendedLifetime(beforeThrow) {
                do {
                    try value.withUnsafeMutableBufferPointer { buffer in
                        if !buffer.isEmpty { buffer[0] = 91 }
                        throw error
                    }
                    Issue.record("Throwing borrow returned normally")
                } catch let caught as AllocationBorrowError {
                    #expect(caught === error)
                }
                #expect(contents(beforeThrow) == expected)
            }
            if !expected.isEmpty { expected[0] = 91 }
            #expect(contents(value) == expected)
            #expect(contents(retained) == original)
        }
        var alternate = value
        let before = contents(value)
        withExtendedLifetime(value) {
            if alternate.count > 0 { alternate[alternate.count - 1] = 77 }
        }
        #expect(contents(value) == before)
        if !expected.isEmpty { expected[expected.count - 1] = 77 }
        #expect(contents(alternate) == expected)
        let alternateSnapshot = contents(alternate)
        withExtendedLifetime(alternate) { if value.count > 0 { value[0] = 33 } }
        #expect(contents(alternate) == alternateSnapshot)
    }

    private func hybrid<E: BinaryFloatingPoint & SIMDScalar>(_ scalar: E.Type, counts: [Int]) throws {
        for n in counts {
            var value = HybridStorage<E>(capacity: n)
            #expect(contents(value) == Array(repeating: 0, count: n))
            #expect(value.isInline == (n * MemoryLayout<E>.stride <= 64))
            let unique66 = value.isUniquelyReferenced
            #expect(unique66)
            let input = (0..<n).map(E.init)
            let array = HybridStorage<E>(from: input)
            #expect(contents(array) == input)
            #expect(contents(HybridStorage<E>(capacity: n, repeating: 7)) == Array(repeating: 7, count: n))
            if n > 0 {
                value[0] = 5
                value[n - 1] = 9
                var expected = Array(repeating: E(0), count: n)
                expected[0] = 5; expected[n - 1] = 9
                #expect(contents(value) == expected)
            }
            let retained = value
            withExtendedLifetime(retained) {
                let unique80 = value.isUniquelyReferenced
                #expect(unique80 == value.isInline)
                for fill in [E(4), E(6)] {
                    value.fill(with: fill)
                    #expect(contents(value) == Array(repeating: fill, count: n))
                    let unique84 = value.isUniquelyReferenced
                    #expect(unique84)
                }
            }
            let snapshot = value
            withExtendedLifetime(snapshot) {
                value.copyFrom(array)
                #expect(contents(value) == input)
                #expect(contents(snapshot) == Array(repeating: 6, count: n))
                #expect(contents(array) == input)
            }
            try checkBorrow(value)
        }
    }

    @Test func hybridFloatMatrix() throws {
        try hybrid(Float.self, counts: [0, 1, 15, 16, 17, 384, 512, 768, 1536])
    }
    @Test func hybridDoubleMatrix() throws {
        try hybrid(Double.self, counts: [0, 1, 7, 8, 9, 16, 17, 384])
    }

    private func dimension<D: StaticDimension, E: BinaryFloatingPoint & SIMDScalar>(
        _ dim: D.Type, _ scalar: E.Type,
        fill: (inout DimensionStorage<D, E>, E) -> Void,
        add: (inout DimensionStorage<D, E>, E) -> Void,
        multiply: (inout DimensionStorage<D, E>, E) -> Void,
        copy: (inout DimensionStorage<D, E>, DimensionStorage<D, E>) -> Void
    ) throws {
        let n = D.value, input = (0..<n).map(E.init)
        for mode: StorageMode? in [nil, .hybrid, .heap] {
            var value = DimensionStorage<D, E>(mode: mode)
            let hybrid = mode.map { if case .hybrid = $0 { true } else { false } } ?? (n <= 16)
            #expect(value.isUsingHybridStorage == hybrid)
            #expect(contents(value) == Array(repeating: 0, count: n))
            let unique118 = value.isUniquelyReferenced
            #expect(unique118)
            value.makeUnique()
            let unique120 = value.isUniquelyReferenced
            #expect(unique120)
            let repeated = DimensionStorage<D, E>(repeating: 3, mode: mode)
            #expect(repeated.isUsingHybridStorage == hybrid)
            #expect(contents(repeated) == Array(repeating: 3, count: n))
            let array = try DimensionStorage<D, E>(from: input, mode: mode)
            #expect(array.isUsingHybridStorage == hybrid && contents(array) == input)
            let generatorError = AllocationBorrowError()
            do {
                _ = try DimensionStorage<D, E>(generator: { _ in throw generatorError })
                Issue.record("Throwing generator returned normally")
            } catch let caught as AllocationBorrowError {
                #expect(caught === generatorError)
            }
            let generated = DimensionStorage<D, E>(generator: E.init)
            #expect(!generated.isUsingHybridStorage && contents(generated) == input)
            value[0] = 4; value[n - 1] = 8
            var expected = Array(repeating: E(0), count: n); expected[0] = 4; expected[n - 1] = 8
            #expect(contents(value) == expected)
            let retained = value
            withExtendedLifetime(retained) {
                let unique133 = value.isUniquelyReferenced
                #expect(unique133 == (hybrid && n * MemoryLayout<E>.stride <= 64))
                value.makeUnique()
                let unique135 = value.isUniquelyReferenced
                #expect(unique135)
                #expect(contents(retained) == expected && contents(value) == expected)
                fill(&value, 5)
                #expect(contents(value) == Array(repeating: 5, count: n))
                #expect(contents(retained) == expected)
            }
            let five = value
            withExtendedLifetime(five) {
                add(&value, 2)
                #expect(contents(value) == Array(repeating: 7, count: n))
                #expect(contents(five) == Array(repeating: 5, count: n))
            }
            let seven = value
            withExtendedLifetime(seven) {
                multiply(&value, 3)
                #expect(contents(value) == Array(repeating: 21, count: n))
                #expect(contents(seven) == Array(repeating: 7, count: n))
            }
            let twentyOne = value
            withExtendedLifetime(twentyOne) {
                copy(&value, array)
                #expect(contents(value) == input && contents(array) == input)
                #expect(contents(twentyOne) == Array(repeating: 21, count: n))
            }
            try checkBorrow(value)
        }
    }
    private func floats<D: StaticDimension>(_ d: D.Type) throws {
        try dimension(d, Float.self, fill: { $0.fill(with: $1) }, add: { $0.add(scalar: $1) },
                      multiply: { $0.multiply(by: $1) }, copy: { $0.copyFrom($1) })
    }
    private func doubles<D: StaticDimension>(_ d: D.Type) throws {
        try dimension(d, Double.self, fill: { $0.fill(with: $1) }, add: { $0.add(scalar: $1) },
                      multiply: { $0.multiply(by: $1) }, copy: { $0.copyFrom($1) })
    }
    @Test func dimensionFloatMatrix() throws {
        try floats(Dim16.self); try floats(AllocationDim17.self)
        try floats(Dim384.self); try floats(Dim512.self); try floats(Dim768.self); try floats(Dim1536.self)
    }
    @Test func dimensionDoubleMatrix() throws {
        try doubles(Dim8.self); try doubles(AllocationDim9.self)
        try doubles(Dim16.self); try doubles(AllocationDim17.self)
    }
}

private struct AllocationProvider: ArraySIMDProvider {
    struct Call: Sendable, Equatable {
        let operation: String
        let first: [UInt32]
        let second: [UInt32]
    }
    let calls = OSAllocatedUnfairLock(initialState: [Call]())
    let base = DefaultArraySIMDProvider()
    let distinctive: Bool
    func record(_ operation: String, _ first: [Float], _ second: [Float]) {
        calls.withLock { $0.append(Call(operation: operation, first: first.map(\.bitPattern), second: second.map(\.bitPattern))) }
    }
    func elementWiseMin(_ a: [Float], _ b: [Float]) -> [Float] {
        record("min", a, b)
        return distinctive ? Array(repeating: 117, count: a.count) : base.elementWiseMin(a, b)
    }
    func elementWiseMax(_ a: [Float], _ b: [Float]) -> [Float] {
        record("max", a, b)
        return distinctive ? Array(repeating: 219, count: a.count) : base.elementWiseMax(a, b)
    }
    func clip(_ a: [Float], min: Float, max: Float) -> [Float] {
        record("clip", a, [min, max])
        return distinctive ? Array(repeating: 321, count: a.count) : base.clip(a, min: min, max: max)
    }
    func add(_ a: [Float], _ b: [Float]) -> [Float] { base.add(a,b) }
    func subtract(_ a: [Float], _ b: [Float]) -> [Float] { base.subtract(a,b) }
    func multiply(_ a: [Float], by b: Float) -> [Float] { base.multiply(a,by:b) }
    func divide(_ a: [Float], by b: Float) -> [Float] { base.divide(a,by:b) }
    func dot(_ a: [Float], _ b: [Float]) -> Float { base.dot(a,b) }
    func sum(_ a: [Float]) -> Float { base.sum(a) }
    func max(_ a: [Float]) -> Float { base.max(a) }
    func min(_ a: [Float]) -> Float { base.min(a) }
    func mean(_ a: [Float]) -> Float { base.mean(a) }
    func magnitude(_ a: [Float]) -> Float { base.magnitude(a) }
    func normalize(_ a: [Float]) -> [Float] { base.normalize(a) }
    func magnitudeSquared(_ a: [Float]) -> Float { base.magnitudeSquared(a) }
    func euclideanDistanceSquared(_ a: [Float], _ b: [Float]) -> Float { base.euclideanDistanceSquared(a,b) }
    func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float { base.cosineSimilarity(a,b) }
    func elementWiseMultiply(_ a: [Float], _ b: [Float]) -> [Float] { base.elementWiseMultiply(a,b) }
    func elementWiseDivide(_ a: [Float], _ b: [Float]) -> [Float] { base.elementWiseDivide(a,b) }
    func abs(_ a: [Float]) -> [Float] { base.abs(a) }
    func sqrt(_ a: [Float]) -> [Float] { base.sqrt(a) }
    func log(_ a: [Float]) -> [Float] { base.log(a) }
    func minIndex(_ a: [Float]) -> Int { base.minIndex(a) }
    func maxIndex(_ a: [Float]) -> Int { base.maxIndex(a) }
}

extension StorageAllocationContractTests {
    @Test func dynamicResizeMatrix() {
        for n in [0, 1, 16, 17, 384, 512, 768, 1536] {
            #expect(DynamicVector(dimension: n, repeating: 13).toArray() == Array(repeating: 13, count: n))
            let input = (0..<n).map { Float($0 + 2) }
            let source = DynamicVector(input), retained = source
            withExtendedLifetime(retained) {
                for destination in Set([0, max(0, n - 1), n, n + 1, 16, 17]) {
                    var resized = source.resized(to: destination, fill: -7)
                    let expected = (0..<destination).map { $0 < n ? input[$0] : -7 }
                    #expect(resized.toArray() == expected)
                    if destination > 0 { resized[0] = 99 }
                    #expect(source.toArray() == input && retained.toArray() == input)
                }
            }
        }
    }

    private func factories<D: StaticDimension>(_ dimension: D.Type) {
        let n = D.value
        #expect(Vector<D>.linspace(from: 0, to: Float(n - 1)).toArray() == (0..<n).map(Float.init))
        #expect(Vector<D>.range(from: 2).toArray() == (0..<n).map { Float($0 + 2) })
        #expect(Vector<D>.generate { Float(2 * $0) }.toArray() == (0..<n).map { Float(2 * $0) })
        #expect(Vector<D>.geometric(initial: 3, ratio: -1).toArray() == (0..<n).map { $0 % 2 == 0 ? 3 : -3 })
        let indices: Set<Int> = [0, n - 1]
        #expect(Vector<D>.sparse(value: 5, at: indices).toArray() == (0..<n).map { indices.contains($0) ? 5 : 0 })
    }
    @Test func convenienceFactoryMatrix() {
        factories(Dim16.self); factories(AllocationDim17.self); factories(Dim384.self)
        factories(Dim512.self); factories(Dim768.self); factories(Dim1536.self)
    }

    private func assertCalls(_ provider: AllocationProvider, first: [Float], second: [Float]) {
        let expected = [AllocationProvider.Call(operation: "min", first: first.map(\.bitPattern), second: second.map(\.bitPattern)),
                        AllocationProvider.Call(operation: "max", first: first.map(\.bitPattern), second: second.map(\.bitPattern)),
                        AllocationProvider.Call(operation: "clip", first: first.map(\.bitPattern), second: [Float(-2).bitPattern, Float(3).bitPattern])]
        #expect(provider.calls.withLock { $0 } == expected)
    }

    @Test func dynamicProviderMatrix() throws {
        for n in [0, 1, 16, 17, 384, 512, 768, 1536] {
            let first = (0..<n).map { Float($0 % 9 - 4) }, second = (0..<n).map { Float(4 - $0 % 9) }
            let a = DynamicVector(first), b = DynamicVector(second)
            let provider = AllocationProvider(distinctive: true)
            try Operations.$simdProvider.withValue(provider) {
                let result291 = try a.min(b).toArray()
                #expect(result291 == Array(repeating: 117, count: n))
                let result292 = try a.max(b).toArray()
                #expect(result292 == Array(repeating: 219, count: n))
                #expect(a.clamped(to: -2...3).toArray() == Array(repeating: 321, count: n))
                assertCalls(provider, first: first, second: second)
                for operation in [0, 1] {
                    do {
                        let wrong = DynamicVector(dimension: n + 1)
                        _ = try operation == 0 ? a.min(wrong) : a.max(wrong)
                        Issue.record("Mismatched operation returned normally")
                    } catch let error as VectorError {
                        #expect(error.kind == .dimensionMismatch)
                        #expect(error.context.additionalInfo["expected_dimension"] == String(n))
                        #expect(error.context.additionalInfo["actual_dimension"] == String(n + 1))
                    }
                }
                assertCalls(provider, first: first, second: second)
            }
            let defaultProvider = DefaultArraySIMDProvider()
            try Operations.$simdProvider.withValue(defaultProvider) {
                let result310 = try a.min(b).toArray()
                #expect(result310 == zip(first, second).map { min($0, $1) })
                let result311 = try a.max(b).toArray()
                #expect(result311 == zip(first, second).map { max($0, $1) })
                #expect(a.clamped(to: -2...3).toArray() == first.map { min(3, max(-2, $0)) })
            }
            let special: [Float] = [.nan, .infinity, -.infinity, -0.0, 0.0, -1, 1]
            let x = (0..<n).map { special[$0 % special.count] }
            let y = (0..<n).map { special[(special.count - 1 - $0 % special.count)] }
            let forwarding = AllocationProvider(distinctive: false)
            try Operations.$simdProvider.withValue(forwarding) {
                let result319 = try DynamicVector(x).min(DynamicVector(y)).toArray().map(\.bitPattern)
                #expect(result319 == defaultProvider.elementWiseMin(x, y).map(\.bitPattern))
                let result320 = try DynamicVector(x).max(DynamicVector(y)).toArray().map(\.bitPattern)
                #expect(result320 == defaultProvider.elementWiseMax(x, y).map(\.bitPattern))
                #expect(DynamicVector(x).clamped(to: -2...3).toArray().map(\.bitPattern) == defaultProvider.clip(x, min: -2, max: 3).map(\.bitPattern))
                assertCalls(forwarding, first: x, second: y)
            }
        }
    }

    private func fixedProvider<D: StaticDimension>(_ dimension: D.Type) throws {
        let n = D.value
        let first = (0..<n).map { Float($0 % 9 - 4) }, second = (0..<n).map { Float(4 - $0 % 9) }
        let a = try Vector<D>(first), b = try Vector<D>(second)
        let provider = AllocationProvider(distinctive: true)
        Operations.$simdProvider.withValue(provider) {
            #expect(a.min(b).toArray() == Array(repeating: 117, count: n))
            #expect(a.max(b).toArray() == Array(repeating: 219, count: n))
            #expect(a.clamped(to: -2...3).toArray() == Array(repeating: 321, count: n))
            assertCalls(provider, first: first, second: second)
        }
        let base = DefaultArraySIMDProvider()
        Operations.$simdProvider.withValue(base) {
            #expect(a.min(b).toArray() == zip(first, second).map { min($0, $1) })
            #expect(a.max(b).toArray() == zip(first, second).map { max($0, $1) })
            #expect(a.clamped(to: -2...3).toArray() == first.map { min(3, max(-2, $0)) })
        }
        let special: [Float] = [.nan, .infinity, -.infinity, -0.0, 0.0, -1, 1]
        let x = (0..<n).map { special[$0 % special.count] }, y = (0..<n).map { special[(special.count - 1 - $0 % special.count)] }
        let lhs = try Vector<D>(x), rhs = try Vector<D>(y)
        let forwarding = AllocationProvider(distinctive: false)
        Operations.$simdProvider.withValue(forwarding) {
            #expect(lhs.min(rhs).toArray().map(\.bitPattern) == base.elementWiseMin(x, y).map(\.bitPattern))
            #expect(lhs.max(rhs).toArray().map(\.bitPattern) == base.elementWiseMax(x, y).map(\.bitPattern))
            #expect(lhs.clamped(to: -2...3).toArray().map(\.bitPattern) == base.clip(x, min: -2, max: 3).map(\.bitPattern))
            assertCalls(forwarding, first: x, second: y)
        }
    }
    @Test func fixedProviderMatrix() throws {
        try fixedProvider(AllocationDim0.self); try fixedProvider(AllocationDim1.self); try fixedProvider(Dim2.self); try fixedProvider(Dim16.self); try fixedProvider(AllocationDim17.self)
        try fixedProvider(Dim384.self); try fixedProvider(Dim512.self); try fixedProvider(Dim768.self); try fixedProvider(Dim1536.self)
    }
}
