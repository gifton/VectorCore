import VectorCore

// Compile-negative: supplying only the old mutable borrow must not satisfy the
// new protocol. Every other SIMDStorage requirement is supplied here.
struct LegacyOnlyBuffer: SIMDStorage {
    private var elements: ContiguousArray<Float> = [2]
    var count: Int { elements.count }
    subscript(index: Int) -> Float {
        get { elements[index] }
        set { elements[index] = newValue }
    }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try elements.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (inout UnsafeMutableBufferPointer<Float>) throws -> R
    ) rethrows -> R {
        try elements.withUnsafeMutableBufferPointer(body)
    }
}
