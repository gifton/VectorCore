import VectorCore

// Built twice: inheriting the public default and providing a direct witness.
private struct ConsumerVector: VectorProtocol {
    var storage: [Float]
    var scalarCount: Int { storage.count }
    #if EXPLICIT_WITNESS
    subscript(index: Int) -> Float { storage[index] }
    #endif
    init() { storage = [0] }
    init(_ array: [Float]) { storage = array }
    init(repeating value: Float) { storage = [value] }
    func toArray() -> [Float] { storage }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (UnsafeMutableBufferPointer<Float>) throws -> R
    ) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { try body($0) }
    }
}

@inline(never)
private func collectionElements<C: Collection>(_ value: C) -> [C.Element] { Array(value) }

@inline(never)
private func collectionElement<C: Collection>(_ value: C, at index: Int) -> C.Element where C.Index == Int {
    value[index]
}

@inline(never)
private func vectorElement<V: VectorProtocol>(_ value: V, at index: Int) -> V.Scalar { value[index] }

@main
struct CollectionWitnessConsumer {
    static func main() {
        let vector = ConsumerVector([1.25, -2.5, 4])
        precondition(abs(vector[0] - 1.25) < 0.0001, "direct subscript")
        precondition(abs(collectionElement(vector, at: 1) + 2.5) < 0.0001, "Collection witness")
        precondition(abs(vectorElement(vector, at: 2) - 4) < 0.0001, "VectorProtocol witness")
        let elements = collectionElements(vector)
        precondition(elements.count == 3, "iteration count")
        for (actual, expected) in zip(elements, [Float(1.25), -2.5, 4]) {
            precondition(abs(actual - expected) < 0.0001, "iteration values")
        }
        precondition(vector.startIndex == 0 && vector.endIndex == 3 && vector.index(after: 1) == 2)
        precondition(collectionElements(ConsumerVector([])).isEmpty, "empty collection")
        var copy = vector
        copy.storage[0] = 8
        precondition(abs(vector[0] - 1.25) < 0.0001 && abs(copy[0] - 8) < 0.0001, "value semantics")
        print("COLLECTION_WITNESS_OK")
    }
}
