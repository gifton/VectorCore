import Foundation
import VectorCore

@_silgen_name("o03_start") func start(_ threshold: UInt)
@_silgen_name("o03_stop") func stop()
@_silgen_name("o03_read") func read(_ field: Int32) -> UInt64
@_silgen_name("o03_histogram") func histogram(_ index: Int32, _ field: Int32) -> UInt64
@_silgen_name("o03_provider_call") func providerCall()
@_silgen_name("o03_provider_read") func providerRead() -> UInt64
@_silgen_name("o03_calibrate_filter") func calibrateFilter()
@_silgen_name("o03_calibrate_c") func calibrateC()

struct O03D9: StaticDimension { static let value = 9; typealias Storage = DimensionStorage<Self, Float> }
struct O03D17: StaticDimension { static let value = 17; typealias Storage = DimensionStorage<Self, Float> }

@inline(never) func measure<R>(_ route: String, _ operation: String, _ n: Int, _ width: Int, emit: Bool, _ body: () -> R) -> R {
    start(UInt(n * width))
    let result = body()
    stop()
    emitMeasurement(route, operation, n, width, emit: emit)
    return result
}
func emitMeasurement(_ route: String, _ operation: String, _ n: Int, _ width: Int, emit: Bool) {
    if emit {
        var buckets: [String: UInt64] = [:]
        for i in 0..<32 { let size = histogram(Int32(i), 0); if size != 0 { buckets[String(size - 1)] = histogram(Int32(i), 1) } }
        let row: [String: Any] = ["route": route, "operation": operation, "count": n, "scalarBytes": width,
            "allocations": read(0), "requestedBytes": read(1), "atLeastPayloadEvents": read(2), "atLeastPayloadBytes": read(3),
            "providerCalls": providerRead(), "requestedSizeHistogram": buckets]
        let data = try! JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    }
}
func values<S: VectorStorage>(_ storage: S) -> [S.Scalar] { storage.withUnsafeBufferPointer { Array($0) } }

@inline(never) func hybrid<E: BinaryFloatingPoint & SIMDScalar>(_ scalar: E.Type, count n: Int, emit: Bool) {
    let width = MemoryLayout<E>.stride, route = "hybrid"
    let zero = measure(route, "zero", n, width, emit: emit) { HybridStorage<E>(capacity: n) }
    precondition(values(zero) == Array(repeating: 0, count: n))
    let repeated = measure(route, "repeating", n, width, emit: emit) { HybridStorage<E>(capacity: n, repeating: 3) }
    precondition(values(repeated) == Array(repeating: 3, count: n))
    let input = (0..<n).map(E.init)
    let constructed = measure(route, "array", n, width, emit: emit) { HybridStorage<E>(from: input) }
    precondition(values(constructed) == input)
    var unique = HybridStorage<E>(capacity: n)
    measure(route, "ten-unique-writes", n, width, emit: emit) { if n > 0 { for i in 0..<10 { unique[i % n] = E(i + 1) } } }
    var expected = Array(repeating: E(0), count: n); if n > 0 { for i in 0..<10 { expected[i % n] = E(i + 1) } }
    precondition(values(unique) == expected)
    measure(route, "fill", n, width, emit: emit) { unique.fill(with: 4) }
    precondition(values(unique) == Array(repeating: 4, count: n))
    measure(route, "ten-bulk-writes", n, width, emit: emit) { for _ in 0..<10 { unique.withUnsafeMutableBufferPointer { b in for i in b.indices { b[i] = 5 } } } }
    precondition(values(unique) == Array(repeating: 5, count: n))
    measure(route, "uniqueness", n, width, emit: emit) { _ = unique.isUniquelyReferenced }
    let retained = unique
    withExtendedLifetime(retained) {
        measure(route, "shared-first-detach", n, width, emit: emit) { unique.withUnsafeMutableBufferPointer { b in for i in b.indices { b[i] = 6 } } }
        precondition(values(retained) == Array(repeating: 5, count: n))
        measure(route, "shared-subsequent-ten-writes", n, width, emit: emit) { if n > 0 { for i in 0..<10 { unique[i % n] = 7 } } }
        precondition(values(retained) == Array(repeating: 5, count: n))
    }
    measure(route, "copyFrom", n, width, emit: emit) { unique.copyFrom(constructed) }
    precondition(values(unique) == input)
}

@inline(never) func dimension<D: StaticDimension, E: BinaryFloatingPoint & SIMDScalar>(_ dim: D.Type, _ scalar: E.Type, mode: StorageMode?, label: String, emit: Bool, fill: (inout DimensionStorage<D, E>, E) -> Void) {
    let n = D.value, width = MemoryLayout<E>.stride
    let route = "dimension-" + label
    let zero = measure(route, "zero", n, width, emit: emit) { DimensionStorage<D, E>(mode: mode) }
    precondition(values(zero) == Array(repeating: 0, count: n))
    let repeated = measure(route, "repeating", n, width, emit: emit) { DimensionStorage<D, E>(repeating: 3, mode: mode) }
    precondition(values(repeated) == Array(repeating: 3, count: n))
    let input = (0..<n).map(E.init)
    let array = measure(route, "array", n, width, emit: emit) { try! DimensionStorage<D, E>(from: input, mode: mode) }
    precondition(values(array) == input)
    var unique = DimensionStorage<D, E>(mode: mode)
    measure(route, "ten-unique-writes", n, width, emit: emit) { for i in 0..<10 { unique[i % n] = E(i + 1) } }
    var expected = Array(repeating: E(0), count: n); for i in 0..<10 { expected[i % n] = E(i + 1) }
    precondition(values(unique) == expected)
    measure(route, "fill", n, width, emit: emit) { fill(&unique, 4) }
    precondition(values(unique) == Array(repeating: 4, count: n))
    measure(route, "ten-bulk-writes", n, width, emit: emit) { for _ in 0..<10 { unique.withUnsafeMutableBufferPointer { b in for i in b.indices { b[i] = 5 } } } }
    precondition(values(unique) == Array(repeating: 5, count: n))
    measure(route, "uniqueness", n, width, emit: emit) { _ = unique.isUniquelyReferenced }
    measure(route, "makeUnique-unique", n, width, emit: emit) { unique.makeUnique() }
    let retained = unique
    withExtendedLifetime(retained) {
        measure(route, "shared-first-detach", n, width, emit: emit) { unique.makeUnique() }
        measure(route, "shared-subsequent-ten-writes", n, width, emit: emit) { for i in 0..<10 { unique[i % n] = 6 } }
        precondition(values(retained) == Array(repeating: 5, count: n))
    }
    let generated = measure(route, "generator-always-managed", n, width, emit: emit) { DimensionStorage<D, E>(generator: E.init) }
    precondition(!generated.isUsingHybridStorage && values(generated) == input)
}
func floatDimension<D: StaticDimension>(_ d: D.Type, emit: Bool) {
    for (mode, label): (StorageMode?, String) in [(nil, "default"), (.hybrid, "hybrid"), (.heap, "heap")] {
        dimension(d, Float.self, mode: mode, label: label, emit: emit, fill: { $0.fill(with: $1) })
    }
}
func doubleDimension<D: StaticDimension>(_ d: D.Type, emit: Bool) {
    for (mode, label): (StorageMode?, String) in [(nil, "default"), (.hybrid, "hybrid"), (.heap, "heap")] {
        dimension(d, Double.self, mode: mode, label: label, emit: emit, fill: { $0.fill(with: $1) })
    }
}

struct CountingProvider: ArraySIMDProvider {
    let base = DefaultArraySIMDProvider()
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
    func elementWiseMin(_ a: [Float], _ b: [Float]) -> [Float] { providerCall(); return base.elementWiseMin(a,b) }
    func elementWiseMax(_ a: [Float], _ b: [Float]) -> [Float] { providerCall(); return base.elementWiseMax(a,b) }
    func elementWiseMultiply(_ a: [Float], _ b: [Float]) -> [Float] { base.elementWiseMultiply(a,b) }
    func elementWiseDivide(_ a: [Float], _ b: [Float]) -> [Float] { base.elementWiseDivide(a,b) }
    func abs(_ a: [Float]) -> [Float] { base.abs(a) }
    func sqrt(_ a: [Float]) -> [Float] { base.sqrt(a) }
    func log(_ a: [Float]) -> [Float] { base.log(a) }
    func clip(_ a: [Float], min: Float, max: Float) -> [Float] { providerCall(); return base.clip(a,min:min,max:max) }
    func minIndex(_ a: [Float]) -> Int { base.minIndex(a) }
    func maxIndex(_ a: [Float]) -> Int { base.maxIndex(a) }
}

func dynamic(_ n: Int, emit: Bool) {
    let repeated = measure("dynamic", "repeating", n, 4, emit: emit) { DynamicVector(dimension: n, repeating: 3) }
    precondition(repeated.toArray() == Array(repeating: 3, count: n))
    let a = DynamicVector((0..<n).map { Float($0 % 7) }), b = DynamicVector((0..<n).map { Float(6 - $0 % 7) })
    for destination in Set([0, max(0,n-1), n, n+1, 16, 17]).sorted() {
        let result = measure("dynamic", "resize-to-\(destination)", n, 4, emit: emit) { a.resized(to: destination, fill: -2) }
        precondition(result.toArray() == Array(a.toArray().prefix(destination)) + Array(repeating: -2, count: max(0, destination-n)))
        precondition(a.toArray() == (0..<n).map { Float($0 % 7) })
    }
    let low = measure("dynamic", "min", n, 4, emit: emit) { try! a.min(b) }
    // Split for Swift 6.2 (CLT) type-checker time limits; semantics unchanged.
    let lowExpected: [Float] = (0..<n).map { (i: Int) -> Float in Float(min(i % 7, 6 - i % 7)) }
    precondition(providerRead() == 1 && low.toArray() == lowExpected)
    let high = measure("dynamic", "max", n, 4, emit: emit) { try! a.max(b) }
    // Split for Swift 6.2 (CLT) type-checker time limits; semantics unchanged.
    let highExpected: [Float] = (0..<n).map { (i: Int) -> Float in Float(max(i % 7, 6 - i % 7)) }
    precondition(providerRead() == 1 && high.toArray() == highExpected)
    let clipped = measure("dynamic", "clamped", n, 4, emit: emit) { a.clamped(to: 2...4) }
    precondition(providerRead() == 1 && clipped.toArray() == (0..<n).map { Float(max(2,min(4,$0 % 7))) })
}
func fixed<D: StaticDimension>(_ d: D.Type, emit: Bool) where D.Storage: VectorStorageOperations {
    let n = D.value
    let range = measure("fixed", "range", n, 4, emit: emit) { Vector<D>.range(from: 2) }
    precondition(range.toArray() == (0..<n).map { Float($0+2) })
    let line = measure("fixed", "linspace", n, 4, emit: emit) { Vector<D>.linspace(from: 0, to: Float(n-1)) }
    precondition(line.toArray() == (0..<n).map(Float.init))
    let generated = measure("fixed", "generate", n, 4, emit: emit) { Vector<D>.generate { Float($0*2) } }
    precondition(generated.toArray() == (0..<n).map { Float($0*2) })
    let geometric = measure("fixed", "geometric", n, 4, emit: emit) { Vector<D>.geometric(initial: 3, ratio: -1) }
    precondition(geometric.toArray() == (0..<n).map { $0 % 2 == 0 ? 3 : -3 })
    let indices: Set<Int> = [0,n-1]
    let sparse = measure("fixed", "sparse", n, 4, emit: emit) { Vector<D>.sparse(value: 5, at: indices) }
    precondition(sparse.toArray() == (0..<n).map { indices.contains($0) ? 5 : 0 })
    let a = Vector<D>(repeating: 2), b = Vector<D>(repeating: 4)
    let low = measure("fixed", "min", n, 4, emit: emit) { a.min(b) }; precondition(providerRead()==1 && low.toArray()==Array(repeating:2,count:n))
    let high = measure("fixed", "max", n, 4, emit: emit) { a.max(b) }; precondition(providerRead()==1 && high.toArray()==Array(repeating:4,count:n))
    let clip = measure("fixed", "clamped", n, 4, emit: emit) { a.clamped(to: 3...5) }; precondition(providerRead()==1 && clip.toArray()==Array(repeating:3,count:n))
}
@inline(never) func concreteManagedControl(emit: Bool) {
    var unwrapped = DimensionStorage<Dim512, Float>()
    start(2048)
    for i in 0..<10 { unwrapped[i] = Float(i+1) }
    stop()
    emitMeasurement("direct-measurement-dimension-default", "ten-unique-writes", 512, 4, emit: emit)
    precondition(values(unwrapped) == (0..<512).map { $0 < 10 ? Float($0+1) : 0 })
    var direct = DimensionStorage<Dim512, Float>()
    measure("concrete-dimension-default", "ten-unique-writes", 512, 4, emit: emit) { for i in 0..<10 { direct[i] = Float(i+1) } }
    precondition(values(direct) == (0..<512).map { $0 < 10 ? Float($0+1) : 0 })
    var vector = Vector<Dim512>()
    measure("concrete-fixed-default", "ten-unique-writes", 512, 4, emit: emit) { for i in 0..<10 { vector[i] = Float(i+1) } }
    precondition(vector.toArray() == (0..<512).map { $0 < 10 ? Float($0+1) : 0 })
    let generated = measure("concrete-fixed-default", "range", 512, 4, emit: emit) { Vector<Dim512>.range() }
    precondition(generated.toArray() == (0..<512).map(Float.init))
}

func run(emit: Bool) {
    concreteManagedControl(emit: emit)
    for n in [0,1,15,16,17,384,512,768,1536] { hybrid(Float.self,count:n,emit:emit) }
    for n in [0,1,7,8,9,16,17,384] { hybrid(Double.self,count:n,emit:emit) }
    floatDimension(Dim16.self,emit:emit); floatDimension(O03D17.self,emit:emit)
    floatDimension(Dim384.self,emit:emit); floatDimension(Dim512.self,emit:emit); floatDimension(Dim768.self,emit:emit); floatDimension(Dim1536.self,emit:emit)
    doubleDimension(Dim8.self,emit:emit); doubleDimension(O03D9.self,emit:emit); doubleDimension(Dim16.self,emit:emit); doubleDimension(O03D17.self,emit:emit)
    Operations.$simdProvider.withValue(CountingProvider()) {
        for n in [0,1,16,17,384,512,768,1536] { dynamic(n,emit:emit) }
        fixed(Dim16.self,emit:emit); fixed(O03D17.self,emit:emit); fixed(Dim384.self,emit:emit); fixed(Dim512.self,emit:emit); fixed(Dim768.self,emit:emit); fixed(Dim1536.self,emit:emit)
    }
}
calibrateFilter(); calibrateC()
// Prime actual code/metadata outside the credited second pass; every row has fresh setup.
run(emit:false)
let nothing = measure("control", "no-op", 0, 4, emit:true) { 42 }; precondition(nothing==42 && read(0)==0)
let array = measure("control", "array", 8192, 1, emit:true) { Array(repeating: UInt8(7),count:8192) }; precondition(array.reduce(0) { $0 + Int($1) }==57344 && read(2)>=1)
run(emit:true)
print("O03_PROBE_COMPLETE")
