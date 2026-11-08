import Darwin
import VectorCore
import BufferValueWitness

// Ordinary downstream consumer: no @testable import and no altered library API.
// All mutations under investigation use writeFirstByValue in the companion
// module. Define OMIT_WRITES only when compiling that companion for the negative control.
// Contract rows encode calls, callback result, inside-value count, after count,
// copy count, then inside/after/copy values. Float values are exact bit patterns.
private struct Reporter {
    var count = 0
    var failures = 0

    mutating func record(_ id: String, _ observed: [String], _ expected: [String]) {
        count += 1
        if observed != expected { failures += 1 }
        print("BUFFER_VALUE \(id) observed=\(observed.joined(separator: ","))")
    }
}

private func floatBits(_ value: Float) -> [String] { [String(value.bitPattern, radix: 16)] }
private func doubleBits(_ value: Double) -> [String] { [String(value.bitPattern, radix: 16)] }
private func floatLanes(_ value: SIMD4<Float>) -> [String] {
    [value.x, value.y, value.z, value.w].flatMap(floatBits)
}
private func int8Lanes(_ value: SIMD4<Int8>) -> [String] {
    [value.x, value.y, value.z, value.w].map(String.init)
}

private func expectedFrame<E>(initial: [E], replacement: E, encode: (E) -> [String]) -> [String] {
    // The oracle describes the public first-element-write contract, independently
    // of the writer's implementation and of the observed storage contents.
    let expected: [E] = initial.isEmpty ? [] : [replacement] + Array(initial.dropFirst())
    return ["1", "731", initial.isEmpty ? "0" : "1", String(initial.count), String(initial.count)]
        + (initial.isEmpty ? [] : encode(replacement)) + expected.flatMap(encode) + initial.flatMap(encode)
}

private func observedFrame<E>(calls: Int, returned: (Int, E?), after: [E], copy: [E], encode: (E) -> [String]) -> [String] {
    [String(calls), String(returned.0), returned.1 == nil ? "0" : "1", String(after.count), String(copy.count)]
        + (returned.1.map(encode) ?? []) + after.flatMap(encode) + copy.flatMap(encode)
}

@inline(never)
private func concreteSingleton() -> Float {
    var values: ContiguousArray<Float> = [2]
    values.withUnsafeMutableElements { buffer in writeFirstByValue(buffer, 1) }
    return values[0]
}

@inline(never)
private func genericSingleton<S: SIMDStorage>(_ input: S, replacement: S.Element) -> S.Element {
    var value = input
    value.withUnsafeMutableElements { buffer in writeFirstByValue(buffer, replacement) }
    return value[0]
}

private func concreteContract(_ id: String, initial: [Float], reporter: inout Reporter) {
    var value = ContiguousArray(initial)
    let copy = value
    withExtendedLifetime(copy) {
        var calls = 0
        let returned: (Int, Float?) = value.withUnsafeMutableElements { buffer in
            calls += 1
            let alias = buffer
            writeFirstByValue(buffer, 1)
            return (731, alias.first)
        }
        reporter.record(id, observedFrame(calls: calls, returned: returned, after: Array(value), copy: Array(copy), encode: floatBits),
                        expectedFrame(initial: initial, replacement: 1, encode: floatBits))
    }
}

private func simdContract<S: SIMDStorage>(
    _ id: String, initial: S, original: [S.Element], replacement: S.Element,
    externalBridge: Bool = false, encode: (S.Element) -> [String], reporter: inout Reporter
) {
    var value = initial
    let copy = value
    withExtendedLifetime(copy) {
        var calls = 0
        let body: (UnsafeMutableBufferPointer<S.Element>) -> (Int, S.Element?) = { buffer in
            calls += 1
            let alias = buffer
            writeFirstByValue(buffer, replacement)
            return (731, alias.first)
        }
        let returned: (Int, S.Element?)
        if externalBridge {
            returned = withValueBuffer(&value, body)
        } else {
            returned = value.withUnsafeMutableElements(body)
        }
        let after = value.withUnsafeBufferPointer { Array($0) }
        let retained = copy.withUnsafeBufferPointer { Array($0) }
        reporter.record(id, observedFrame(calls: calls, returned: returned, after: after, copy: retained, encode: encode),
                        expectedFrame(initial: original, replacement: replacement, encode: encode))
    }
}

private enum BorrowFailure: Error { case expected }

private func throwingContract<S: SIMDStorage>(
    _ id: String, initial: S, externalBridge: Bool, reporter: inout Reporter
) where S.Element == Float {
    var value = initial
    let copy = value
    withExtendedLifetime(copy) {
        var calls = 0
        var caught = 0
        let body: (UnsafeMutableBufferPointer<Float>) throws -> Void = { buffer in
            calls += 1
            writeFirstByValue(buffer, 1)
            throw BorrowFailure.expected
        }
        do {
            if externalBridge { try withValueBuffer(&value, body) }
            else { try value.withUnsafeMutableElements(body) }
        } catch BorrowFailure.expected { caught = 1 }
        catch { caught = -1 }
        let after = value.withUnsafeBufferPointer { Array($0) }
        let retained = copy.withUnsafeBufferPointer { Array($0) }
        reporter.record(id, [String(calls), String(caught), String(after.count), String(retained.count)]
                            + after.flatMap(floatBits) + retained.flatMap(floatBits),
                        ["1", "1", "3", "3", "3f800000", "40400000", "40800000", "40000000", "40400000", "40800000"])
    }
}

private func storageContract<S: VectorStorage>(
    _ id: String, initial: S, original: [S.Scalar], encode: (S.Scalar) -> [String], reporter: inout Reporter
) {
    var value = initial
    let copy = value
    withExtendedLifetime(copy) {
        var calls = 0
        let returned: (Int, S.Scalar?) = value.withUnsafeMutableBufferPointer { buffer in
            calls += 1
            let alias = buffer
            writeFirstByValue(buffer, S.Scalar(1))
            return (731, alias.first)
        }
        let after = value.withUnsafeBufferPointer { Array($0) }
        let retained = copy.withUnsafeBufferPointer { Array($0) }
        reporter.record(id, observedFrame(calls: calls, returned: returned, after: after, copy: retained, encode: encode),
                        expectedFrame(initial: original, replacement: S.Scalar(1), encode: encode))
    }
}

private func vectorContract<V: VectorProtocol>(
    _ id: String, initial: V, count: Int, reporter: inout Reporter
) where V.Scalar == Float {
    var value = initial
    let copy = value
    withExtendedLifetime(copy) {
        var calls = 0
        let returned: (Int, Float?) = value.withUnsafeMutableBufferPointer { buffer in
            calls += 1
            let alias = buffer
            writeFirstByValue(buffer, 1)
            return (731, alias.first)
        }
        reporter.record(id, observedFrame(calls: calls, returned: returned, after: value.toArray(), copy: copy.toArray(), encode: floatBits),
                        expectedFrame(initial: Array(repeating: Float(2), count: count), replacement: 1, encode: floatBits))
    }
}

@inline(never)
private func inlinableArithmetic<V: VectorProtocol>(_ input: V, addend: V) -> [Float] where V.Scalar == Float {
    var result = input
    result.inPlaceAdd(addend)
    result.inPlaceMultiply(2)
    return result.toArray()
}

@main
private struct BufferValueConsumer {
    static func main() {
        var reporter = Reporter()
        reporter.record("concrete.float.singleton", floatBits(concreteSingleton()), ["3f800000"])
        concreteContract("concrete.float.empty", initial: [], reporter: &reporter)
        concreteContract("concrete.float.tail", initial: [2, 3, 4], reporter: &reporter)
        reporter.record("generic.float.singleton", floatBits(genericSingleton(ContiguousArray<Float>([2]), replacement: 1)), ["3f800000"])
        simdContract("generic.float.empty", initial: ContiguousArray<Float>(), original: [], replacement: 1, encode: floatBits, reporter: &reporter)
        simdContract("generic.float.tail", initial: ContiguousArray<Float>([2, 3, 4]), original: [2, 3, 4], replacement: 1, encode: floatBits, reporter: &reporter)
        simdContract("generic.double.one", initial: ContiguousArray<Double>([2]), original: [2], replacement: 1, encode: doubleBits, reporter: &reporter)
        let oldFloat = SIMD4<Float>(2, 4, 6, 8), newFloat = SIMD4<Float>(1, 3, 5, 7)
        let oldInt8 = SIMD4<Int8>(2, 4, 6, 8), newInt8 = SIMD4<Int8>(1, 3, 5, 7)
        simdContract("generic.simd4float.tail", initial: ContiguousArray(repeating: oldFloat, count: 3), original: [oldFloat, oldFloat, oldFloat], replacement: newFloat, encode: floatLanes, reporter: &reporter)
        simdContract("generic.simd4int8.tail", initial: ContiguousArray(repeating: oldInt8, count: 3), original: [oldInt8, oldInt8, oldInt8], replacement: newInt8, encode: int8Lanes, reporter: &reporter)
        simdContract("external.float.tail", initial: ExternalBuffer<Float>([2, 3, 4]), original: [2, 3, 4], replacement: 1, externalBridge: true, encode: floatBits, reporter: &reporter)
        simdContract("external.simd4float.tail", initial: ExternalBuffer([oldFloat, oldFloat, oldFloat]), original: [oldFloat, oldFloat, oldFloat], replacement: newFloat, externalBridge: true, encode: floatLanes, reporter: &reporter)
        throwingContract("generic.float.throw", initial: ContiguousArray<Float>([2, 3, 4]), externalBridge: false, reporter: &reporter)
        throwingContract("external.float.throw", initial: ExternalBuffer<Float>([2, 3, 4]), externalBridge: true, reporter: &reporter)
        storageContract("storage.float.empty", initial: HybridStorage<Float>(capacity: 0, repeating: 2), original: [], encode: floatBits, reporter: &reporter)
        storageContract("storage.float.inline16", initial: HybridStorage<Float>(capacity: 16, repeating: 2), original: Array(repeating: 2, count: 16), encode: floatBits, reporter: &reporter)
        storageContract("storage.float.heap17", initial: HybridStorage<Float>(capacity: 17, repeating: 2), original: Array(repeating: 2, count: 17), encode: floatBits, reporter: &reporter)
        storageContract("storage.double.inline8", initial: HybridStorage<Double>(capacity: 8, repeating: 2), original: Array(repeating: 2, count: 8), encode: doubleBits, reporter: &reporter)
        storageContract("storage.double.heap9", initial: HybridStorage<Double>(capacity: 9, repeating: 2), original: Array(repeating: 2, count: 9), encode: doubleBits, reporter: &reporter)
        vectorContract("vector.dim3", initial: Vector<Dim3>(repeating: 2), count: 3, reporter: &reporter)
        vectorContract("vector.dim512", initial: Vector<Dim512>(repeating: 2), count: 512, reporter: &reporter)
        vectorContract("vector.dynamic.empty", initial: DynamicVector(), count: 0, reporter: &reporter)
        vectorContract("vector.dynamic17", initial: DynamicVector(dimension: 17, repeating: 2), count: 17, reporter: &reporter)
        vectorContract("vector.optimized512", initial: Vector512Optimized(repeating: 2), count: 512, reporter: &reporter)
        let original = Vector<Dim3>(x: 2, y: 3, z: 4)
        let result = inlinableArithmetic(original, addend: Vector<Dim3>(repeating: 1))
        reporter.record("vector.inlinable.arithmetic", result.flatMap(floatBits) + original.toArray().flatMap(floatBits),
                        ["40c00000", "41000000", "41200000", "40000000", "40400000", "40800000"])
        print("BUFFER_VALUE_COMPLETE count=\(reporter.count) failures=\(reporter.failures)")
        exit(reporter.failures == 0 ? 0 : 1)
    }
}
