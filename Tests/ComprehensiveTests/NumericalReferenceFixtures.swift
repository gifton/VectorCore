import Foundation
import Testing
@testable import VectorCore

// Explicit read-only subscript avoids the preexisting Swift 6.3.3 default-subscript
// MandatorySILLinker issue without replacing any numeric defaults under test.
struct N01DoubleVector: VectorFactory {
    var storage: [Double]
    var scalarCount: Int { storage.count }
    subscript(index: Int) -> Double { storage[index] }
    init() { storage = [] }
    init(_ array: [Double]) { storage = array }
    init(repeating value: Double) { storage = [value] }
    static func create(from scalars: [Double]) throws -> Self { Self(scalars) }
    func toArray() -> [Double] { storage }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Double>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Double>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { try body($0) }
    }
}

func n01AssertFloat(_ actual: [Float], _ expected: [Float]) {
    #expect(actual.count == expected.count)
    guard actual.count == expected.count else { return }
    #expect(actual.allSatisfy({ $0.isFinite }))
    guard actual.allSatisfy({ $0.isFinite }) else { return }
    var sum: Double = 0
    var compensation: Double = 0
    for (x, reference) in zip(actual, expected) {
        #expect(abs(Double(x) - Double(reference)) <= 8 * Double(max(reference.ulp, Float.leastNonzeroMagnitude)))
        let squared = Double(x) * Double(x)
        let y = squared - compensation
        let next = sum + y
        compensation = (next - sum) - y
        sum = next
    }
    #expect(abs(sum.squareRoot() - 1) <= 8 * Double(Float.ulpOfOne))
}

func n01AssertDouble(_ actual: [Double], _ expected: [Double]) {
    #expect(actual.count == expected.count)
    #expect(actual.allSatisfy({ $0.isFinite }))
    guard actual.count == expected.count && actual.allSatisfy({ $0.isFinite }) else { return }
    // Pairwise addition of bounded normalized-output squares is independent of
    // the production maximum scaling and Kahan accumulation.
    func pairwise(_ values: ArraySlice<Double>) -> Double {
        guard values.count > 1 else { return values.first ?? 0 }
        let mid = values.index(values.startIndex, offsetBy: values.count / 2)
        return pairwise(values[..<mid]) + pairwise(values[mid...])
    }
    for (x, reference) in zip(actual, expected) {
        #expect(abs(x - reference) <= 16 * max(reference.ulp, Double.leastNonzeroMagnitude))
    }
    let squared = actual.map { $0 * $0 }
    #expect(abs(pairwise(squared[...]).squareRoot() - 1) <= 32 * Double.ulpOfOne)
}

// Generated independently by N01-reference-generator.py at 2500 and 3000 decimal digits.
let n01DoubleFixtures: [([UInt64], [UInt64])] = [
    ([0x7fefffffffffffff, 0x1, 0xffdfffffffffffff, 0x0], [0x3fec9f25c5bfedd9, 0x0, 0xbfdc9f25c5bfedd9, 0x0]),
    ([0x1, 0x2, 0x3, 0x8000000000000007], [0x3fc02061446ffa9a, 0x3fd02061446ffa9a, 0x3fd83091e6a7f7e7, 0xbfec38aa37c3f68d]),
    ([0x10000000000000, 0xfffffffffffff, 0x8010000000000001], [0x3fe279a74590331c, 0x3fe279a74590331b, 0xbfe279a74590331d]),
    ([0x7fd123456789abcd, 0xffe0123456789abc, 0x5feabcdef0123456, 0x12345], [0x3fde1c751c10f865, 0xbfec3cad8e1a8956, 0x1ff77d3c7ddf338b, 0x0]),
]
let n01DoubleDivisionFixtures: [([UInt64], UInt64, [UInt64])] = [
    ([0x10000000000001, 0x800fffffffffffff, 0x1, 0x8000000000000000], 0x3, [0x4315555555555557, 0xc315555555555554, 0x3fd5555555555555, 0x8000000000000000]),
    ([0x7fefffffffffffff, 0xffefffffffffffff, 0x10000000000000, 0x1], 0x7feffffffffffffe, [0x3ff0000000000001, 0xbff0000000000001, 0x0, 0x0]),
    ([0x3ff123456789abcd, 0xbfd3456789abcdef, 0xfffffffffffff], 0x3fe23456789abcde, [0x3ffe200000000000, 0xbfe0f00000000000, 0x1c200000000000]),
    ([0x1, 0x2, 0x8000000000000003, 0x10000000000001], 0x10000000000000, [0x3cb0000000000000, 0x3cc0000000000000, 0xbcc8000000000000, 0x3ff0000000000001]),
]
// End independently generated N01 references.

struct N01LossyVector: IndexableVector {
    var storage: [Float]
    var scalarCount: Int { storage.count }
    subscript(index: Int) -> Float { storage[index] }
    init() { storage = [] }
    init(_ array: [Float]) { storage = array }
    init(repeating value: Float) { storage = [value] }
    func toArray() -> [Float] { storage }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        defer { storage = storage.map { $0.rounded(.towardZero) } }
        return try storage.withUnsafeMutableBufferPointer { try body($0) }
    }
}

struct N01RejectingFactory: VectorFactory {
    enum Failure: Error { case original }
    var storage: [Float]
    var scalarCount: Int { storage.count }
    subscript(index: Int) -> Float { storage[index] }
    init() { storage = [] }
    init(_ array: [Float]) { storage = array }
    init(repeating value: Float) { storage = [value] }
    static func create(from scalars: [Float]) throws -> Self { throw Failure.original }
    func toArray() -> [Float] { storage }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R { try storage.withUnsafeBufferPointer(body) }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R { try storage.withUnsafeMutableBufferPointer { try body($0) } }
}

// Inherits scalar division while forwarding the unrelated required witnesses.
enum N01DefaultFloatProvider: SIMDProvider {
    typealias Scalar = Float
    static func add(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) { SwiftFloatSIMDProvider.add(a,b,result:result,count:count) }
    static func subtract(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) { SwiftFloatSIMDProvider.subtract(a,b,result:result,count:count) }
    static func multiply(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) { SwiftFloatSIMDProvider.multiply(a,b,result:result,count:count) }
    static func divide(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) { SwiftFloatSIMDProvider.divide(a,b,result:result,count:count) }
    static func addScalar(_ a: UnsafePointer<Float>, scalar: Float, result: UnsafeMutablePointer<Float>, count: Int) { SwiftFloatSIMDProvider.addScalar(a,scalar:scalar,result:result,count:count) }
    static func multiplyScalar(_ a: UnsafePointer<Float>, scalar: Float, result: UnsafeMutablePointer<Float>, count: Int) { SwiftFloatSIMDProvider.multiplyScalar(a,
        scalar:scalar,result:result,count:count) }
    static func negate(_ a: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) { SwiftFloatSIMDProvider.negate(a,result:result,count:count) }
    static func sum(_ a: UnsafePointer<Float>, count: Int) -> Float { SwiftFloatSIMDProvider.sum(a,count:count) }
    static func sumOfMagnitudes(_ a: UnsafePointer<Float>, count: Int) -> Float { SwiftFloatSIMDProvider.sumOfMagnitudes(a,count:count) }
    static func sumOfSquares(_ a: UnsafePointer<Float>, count: Int) -> Float { SwiftFloatSIMDProvider.sumOfSquares(a,count:count) }
    static func maximum(_ a: UnsafePointer<Float>, count: Int) -> Float { SwiftFloatSIMDProvider.maximum(a,count:count) }
    static func minimum(_ a: UnsafePointer<Float>, count: Int) -> Float { SwiftFloatSIMDProvider.minimum(a,count:count) }
    static func maximumMagnitude(_ a: UnsafePointer<Float>, count: Int) -> Float { SwiftFloatSIMDProvider.maximumMagnitude(a,count:count) }
    static func dot(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, count: Int) -> Float { SwiftFloatSIMDProvider.dot(a,b,count:count) }
    static func distanceSquared(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, count: Int) -> Float { SwiftFloatSIMDProvider.distanceSquared(a,b,count:count) }
    static func copy(source: UnsafePointer<Float>, destination: UnsafeMutablePointer<Float>, count: Int) { SwiftFloatSIMDProvider.copy(source:source,destination:destination,count:count) }
    static func fill(value: Float, destination: UnsafeMutablePointer<Float>, count: Int) { SwiftFloatSIMDProvider.fill(value:value,destination:destination,count:count) }
    static func clip(_ a: UnsafePointer<Float>, low: Float, high: Float, result: UnsafeMutablePointer<Float>, count: Int) { SwiftFloatSIMDProvider.clip(a,low:low,high:high,result:result,count:count) }
}

#if !(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))
struct N01HalfVector: VectorFactory {
    var storage: [Float16]
    var scalarCount: Int { storage.count }
    subscript(index: Int) -> Float16 { storage[index] }
    init() { storage = [] }
    init(_ array: [Float16]) { storage = array }
    init(repeating value: Float16) { storage = [value] }
    static func create(from scalars: [Float16]) throws -> Self { Self(scalars) }
    func toArray() -> [Float16] { storage }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float16>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float16>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { try body($0) }
    }
}
#endif

struct N01SentinelArrayProvider: ArraySIMDProvider {
    let base = DefaultArraySIMDProvider()
    func add(_ a: [Float], _ b: [Float]) -> [Float] { base.add(a, b) }
    func subtract(_ a: [Float], _ b: [Float]) -> [Float] { base.subtract(a, b) }
    func multiply(_ a: [Float], by scalar: Float) -> [Float] { base.multiply(a, by: scalar) }
    func divide(_ a: [Float], by scalar: Float) -> [Float] { base.divide(a, by: scalar) }
    func dot(_ a: [Float], _ b: [Float]) -> Float { base.dot(a, b) }
    func sum(_ a: [Float]) -> Float { base.sum(a) }
    func max(_ a: [Float]) -> Float { base.max(a) }
    func min(_ a: [Float]) -> Float { base.min(a) }
    func mean(_ a: [Float]) -> Float { base.mean(a) }
    func magnitude(_ a: [Float]) -> Float { base.magnitude(a) }
    func normalize(_ a: [Float]) -> [Float] { a.map { _ in 42 } }
    func magnitudeSquared(_ a: [Float]) -> Float { base.magnitudeSquared(a) }
    func euclideanDistanceSquared(_ a: [Float], _ b: [Float]) -> Float { base.euclideanDistanceSquared(a, b) }
    func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float { base.cosineSimilarity(a, b) }
    func elementWiseMin(_ a: [Float], _ b: [Float]) -> [Float] { base.elementWiseMin(a, b) }
    func elementWiseMax(_ a: [Float], _ b: [Float]) -> [Float] { base.elementWiseMax(a, b) }
    func elementWiseMultiply(_ a: [Float], _ b: [Float]) -> [Float] { base.elementWiseMultiply(a, b) }
    func elementWiseDivide(_ a: [Float], _ b: [Float]) -> [Float] { base.elementWiseDivide(a, b) }
    func abs(_ a: [Float]) -> [Float] { base.abs(a) }
    func sqrt(_ a: [Float]) -> [Float] { base.sqrt(a) }
    func log(_ a: [Float]) -> [Float] { base.log(a) }
    func clip(_ a: [Float], min: Float, max: Float) -> [Float] { base.clip(a, min: min, max: max) }
    func minIndex(_ a: [Float]) -> Int { base.minIndex(a) }
    func maxIndex(_ a: [Float]) -> Int { base.maxIndex(a) }
}
