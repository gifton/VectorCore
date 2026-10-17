import VectorCore

// Public imports only. MISSING_UPDATE is the intentionally incompatible conformance.
struct ConsumerCollection: MutableVectorCollection {
    enum Rejection: Error { case injected }
    private var values: [Int: DynamicVector] = [7: DynamicVector([1, 2])]
    var count: Int { values.count }
    var dimension: Int { 2 }
    var allIDs: [Int] { Array(values.keys) }
    func vector(for id: Int) -> DynamicVector? { values[id] }
    mutating func add(id: Int, vector: DynamicVector) throws {
        guard vector.scalarCount == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: vector.scalarCount)
        }
        guard values[id] == nil else { throw VectorError(.invalidConfiguration, message: "Duplicate ID") }
        values[id] = vector
    }
    mutating func remove(id: Int) -> DynamicVector? { values.removeValue(forKey: id) }
    mutating func removeAll() { values.removeAll() }

    #if !MISSING_UPDATE
    mutating func update(id: Int, vector: DynamicVector) throws {
        guard vector.scalarCount == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: vector.scalarCount)
        }
        guard values[id] != nil else { throw VectorError.indexOutOfBounds(index: 0, dimension: dimension) }
        guard vector[0] >= 0 else { throw Rejection.injected }
        values[id] = vector
    }
    #endif
}

func replace<C: MutableVectorCollection>(_ collection: inout C, id: C.ID, vector: C.Vector) throws {
    try collection.update(id: id, vector: vector)
}

var collection = ConsumerCollection()
#if !MISSING_UPDATE
do {
    try replace(&collection, id: 7, vector: DynamicVector([-1, 2]))
    fatalError("Expected injected rejection")
} catch ConsumerCollection.Rejection.injected {
    precondition(collection.vector(for: 7)?.toArray() == [1, 2])
    precondition(collection.count == 1 && collection.allIDs == [7])
}
#endif
try replace(&collection, id: 7, vector: DynamicVector([3, 4]))
precondition(collection.vector(for: 7)?.toArray() == [3, 4])
precondition(collection.searchEuclidean(query: DynamicVector([3, 4]), k: 1).distances == [0])
print("R02_PUBLIC_TRANSACTIONAL_WITNESS_OK")
