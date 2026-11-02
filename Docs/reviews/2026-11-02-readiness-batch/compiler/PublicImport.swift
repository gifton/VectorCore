import VectorCore
private struct LC5AccessVector: VectorProtocol {
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
        return try storage.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { try body($0) }
    }
}



private func first<C: Collection>(_ value: C) -> C.Element where C.Index == Int { value[0] }
public func probe() -> Float { first(LC5AccessVector([1])) }
