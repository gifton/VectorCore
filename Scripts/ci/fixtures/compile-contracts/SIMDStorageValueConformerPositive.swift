import VectorCore

// Compile-positive: a downstream conformer supplies only the new mutable borrow.
// Runtime values and the separate-module witness are checked by BufferValueConsumer.
struct ValueOnlyBuffer: SIMDStorage {
    private var elements: ContiguousArray<Float> = [2]
    var count: Int { elements.count }
    subscript(index: Int) -> Float {
        get { elements[index] }
        set { elements[index] = newValue }
    }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try elements.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableElements<R>(
        _ body: (UnsafeMutableBufferPointer<Float>) throws -> R
    ) rethrows -> R {
        try elements.withUnsafeMutableBufferPointer { buffer in try body(buffer) }
    }
}

func valueGenericCall<S: SIMDStorage, R>(
    _ storage: inout S, _ body: (UnsafeMutableBufferPointer<S.Element>) throws -> R
) rethrows -> R {
    try storage.withUnsafeMutableElements(body)
}

func valueOnlyConformerCall() -> Float {
    var storage = ValueOnlyBuffer()
    return valueGenericCall(&storage) { buffer in
        buffer[0] = 1
        return buffer[0]
    }
}

// No same-name overload was added: an ordinary inferred stdlib borrow remains
// unambiguous. This is a compile contract, not an optimizer-correctness claim.
func unambiguousStandardBorrow() -> Float {
    var values: ContiguousArray<Float> = [2]
    return values.withUnsafeMutableBufferPointer { buffer in
        buffer[0] = 1
        return buffer[0]
    }
}
