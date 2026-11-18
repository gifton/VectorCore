import MinimalLibrary
private struct V: MinimalVector {
    typealias Element = Float
    var storage: [Float]
    func toArray() -> [Float] { storage }
    #if EXPLICIT_WITNESS
    subscript(index: Int) -> Float { storage[index] }
    #endif
}
private func first<C: Collection>(_ value: C) -> C.Element where C.Index == Int { value[0] }
public func probe() -> Float { first(V(storage: [1])) }
