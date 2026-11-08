// Exploratory residual-exposure probe. It does not qualify a compiler or release.
// Probe contract: every writer receives nonempty initialized storage during a
// synchronous borrow; its descriptor and pointers must not escape.
import Foundation
import VectorCore

@inline(never)
private func writeValue(_ buffer: UnsafeMutableBufferPointer<Float>) {
    #if !OMIT_WRITES
    buffer[0] = 1
    #endif
}

@inline(never)
private func writeInout(_ buffer: inout UnsafeMutableBufferPointer<Float>) {
    #if !OMIT_WRITES
    buffer[0] = 1
    #endif
}

@inline(never)
private func writeSIMDValue(_ buffer: UnsafeMutableBufferPointer<SIMD4<Float>>) {
    #if !OMIT_WRITES
    buffer[0] = SIMD4<Float>(repeating: 1)
    #endif
}

@inline(never)
private func writeSIMDInout(_ buffer: inout UnsafeMutableBufferPointer<SIMD4<Float>>) {
    #if !OMIT_WRITES
    buffer[0] = SIMD4<Float>(repeating: 1)
    #endif
}

private struct FloatView { var buffer: UnsafeMutableBufferPointer<Float> }

@inline(never)
private func writeAggregate(_ view: inout FloatView) {
    #if !OMIT_WRITES
    view.buffer[0] = 1
    #endif
}

@inline(never)
private func concreteValue() -> Float {
    var values: ContiguousArray<Float> = [2]
    values.withUnsafeMutableElements(writeValue)
    return values[0]
}

@inline(never)
private func concreteCopiedInout() -> Float {
    var values: ContiguousArray<Float> = [2]
    values.withUnsafeMutableElements { buffer in
        var descriptor = buffer
        writeInout(&descriptor)
    }
    return values[0]
}

@inline(never)
private func insideCopiedInout() -> Float {
    var values: ContiguousArray<Float> = [2]
    return values.withUnsafeMutableElements { buffer in
        var descriptor = buffer
        writeInout(&descriptor)
        return buffer[0]
    }
}

@inline(never)
private func aggregateInout() -> Float {
    var values: ContiguousArray<Float> = [2]
    values.withUnsafeMutableElements { buffer in
        var view = FloatView(buffer: buffer)
        writeAggregate(&view)
    }
    return values[0]
}

@inline(never)
private func genericValue<S: SIMDStorage>(_ storage: inout S) where S.Element == Float {
    storage.withUnsafeMutableElements(writeValue)
}

@inline(never)
private func genericCopiedInout<S: SIMDStorage>(_ storage: inout S) where S.Element == Float {
    storage.withUnsafeMutableElements { buffer in
        var descriptor = buffer
        writeInout(&descriptor)
    }
}

@inline(never)
private func throughGenericValue() -> Float {
    var values: ContiguousArray<Float> = [2]
    genericValue(&values)
    return values[0]
}

@inline(never)
private func throughGenericCopiedInout() -> Float {
    var values: ContiguousArray<Float> = [2]
    genericCopiedInout(&values)
    return values[0]
}

@inline(never)
private func invokeByValue<R>(_ buffer: UnsafeMutableBufferPointer<Float>,
                              _ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
    try body(buffer)
}

@inline(never)
private func invokeByInout<R>(_ buffer: inout UnsafeMutableBufferPointer<Float>,
                              _ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
    try body(buffer)
}

private struct ForwardingStorage: SIMDStorage {
    var values: ContiguousArray<Float> = [2]
    let indirect: Bool
    var count: Int { values.count }
    subscript(index: Int) -> Float {
        get { values[index] }
        set { values[index] = newValue }
    }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try values.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableElements<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        let useInout = indirect
        return try values.withUnsafeMutableBufferPointer { buffer in
            if useInout { return try invokeByInout(&buffer, body) }
            return try invokeByValue(buffer, body)
        }
    }
}

@inline(never)
private func customValue() -> Float {
    var values = ForwardingStorage(indirect: false)
    values.withUnsafeMutableElements(writeValue)
    return values[0]
}

@inline(never)
private func customInout() -> Float {
    var values = ForwardingStorage(indirect: true)
    values.withUnsafeMutableElements(writeValue)
    return values[0]
}

@inline(never)
private func optimizedStdlibInout() -> Float {
    var vector = Vector512Optimized(repeating: 2)
    vector.storage.withUnsafeMutableBufferPointer(writeSIMDInout)
    return vector[0]
}

@inline(never)
private func optimizedStorageValue() -> Float {
    var vector = Vector512Optimized(repeating: 2)
    vector.storage.withUnsafeMutableElements(writeSIMDValue)
    return vector[0]
}

@inline(never)
private func optimizedStorageCopiedInout() -> Float {
    var vector = Vector512Optimized(repeating: 2)
    vector.storage.withUnsafeMutableElements { buffer in
        var descriptor = buffer
        writeSIMDInout(&descriptor)
    }
    return vector[0]
}

@inline(never)
private func optimizedFloatValue() -> Float {
    var vector = Vector512Optimized(repeating: 2)
    vector.withUnsafeMutableBufferPointer(writeValue)
    return vector[0]
}

@inline(never)
private func optimizedFloatCopiedInout() -> Float {
    var vector = Vector512Optimized(repeating: 2)
    vector.withUnsafeMutableBufferPointer { buffer in
        var descriptor = buffer
        writeInout(&descriptor)
    }
    return vector[0]
}

@inline(never)
private func fixedValue() -> Float {
    var vector = Vector<Dim3>(repeating: 2)
    vector.withUnsafeMutableBufferPointer(writeValue)
    return vector[0]
}

@inline(never)
private func fixedCopiedInout() -> Float {
    var vector = Vector<Dim3>(repeating: 2)
    vector.withUnsafeMutableBufferPointer { buffer in
        var descriptor = buffer
        writeInout(&descriptor)
    }
    return vector[0]
}

@inline(never)
private func dynamicValue() -> Float {
    var vector = DynamicVector(dimension: 17, repeating: 2)
    vector.withUnsafeMutableBufferPointer(writeValue)
    return vector[0]
}

@inline(never)
private func dynamicCopiedInout() -> Float {
    var vector = DynamicVector(dimension: 17, repeating: 2)
    vector.withUnsafeMutableBufferPointer { buffer in
        var descriptor = buffer
        writeInout(&descriptor)
    }
    return vector[0]
}

@main private struct ResidualExposure {
    static func main() {
        let rows: [(String, Float)] = [
            ("new.concrete.value", concreteValue()),
            ("new.concrete.copied-inout", concreteCopiedInout()),
            ("new.inside.copied-inout", insideCopiedInout()),
            ("new.aggregate.inout", aggregateInout()),
            ("new.generic.value", throughGenericValue()),
            ("new.generic.copied-inout", throughGenericCopiedInout()),
            ("custom.value-bridge", customValue()),
            ("custom.inout-bridge", customInout()),
            ("optimized.storage.stdlib-inout", optimizedStdlibInout()),
            ("optimized.storage.value", optimizedStorageValue()),
            ("optimized.storage.copied-inout", optimizedStorageCopiedInout()),
            ("optimized.float.value", optimizedFloatValue()),
            ("optimized.float.copied-inout", optimizedFloatCopiedInout()),
            ("fixed.value", fixedValue()),
            ("fixed.copied-inout", fixedCopiedInout()),
            ("dynamic.value", dynamicValue()),
            ("dynamic.copied-inout", dynamicCopiedInout()),
        ]
        var failures = 0
        for (name, value) in rows {
            print("BUFFER_VALUE \(name) observed=\(String(value.bitPattern, radix: 16))")
            if value.bitPattern != 0x3f800000 { failures += 1 }
        }
        print("BUFFER_VALUE_COMPLETE count=\(rows.count) failures=\(failures)")
        exit(failures == 0 ? 0 : 1)
    }
}
