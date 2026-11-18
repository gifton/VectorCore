import VectorCore

// Compile as the separate BufferValueWitness module. OMIT_WRITES must be passed
// when compiling this file for the deliberately broken consumer control.
@inline(never)
public func writeFirstByValue<Element>(_ buffer: UnsafeMutableBufferPointer<Element>, _ replacement: Element) {
    #if !OMIT_WRITES
    if !buffer.isEmpty { buffer[0] = replacement }
    #endif
}

// A downstream conformer implements only the by-value SIMDStorage requirement.
// Its backing array owns the elements; no pointer escapes either borrow.
public struct ExternalBuffer<Element>: SIMDStorage {
    private var elements: ContiguousArray<Element>

    public init(_ values: [Element]) { elements = ContiguousArray(values) }
    public var count: Int { elements.count }
    public subscript(index: Int) -> Element {
        get { elements[index] }
        set { elements[index] = newValue }
    }
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R {
        try elements.withUnsafeBufferPointer(body)
    }
    public mutating func withUnsafeMutableElements<R>(
        _ body: (UnsafeMutableBufferPointer<Element>) throws -> R
    ) rethrows -> R {
        try elements.withUnsafeMutableBufferPointer { buffer in try body(buffer) }
    }
}

// Deliberately not @inlinable. The driver inspects this module's emitted SIL for
// the generic SIMDStorage witness call; source spelling alone is not that proof.
@inline(never)
public func withValueBuffer<S: SIMDStorage, R>(
    _ storage: inout S, _ body: (UnsafeMutableBufferPointer<S.Element>) throws -> R
) rethrows -> R {
    try storage.withUnsafeMutableElements(body)
}
