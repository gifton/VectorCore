// Diagnostic of the concrete standard-library API, deliberately retaining its
// descriptor-inout writer. Generic SIMDStorage no longer exposes that member;
// SIMDStorageLegacyCallNegative.swift checks its rejection separately.
import Foundation
import VectorCore

@inline(never)
private func writeInout(_ out: inout UnsafeMutableBufferPointer<Float>) { out[0] = 1 }

@inline(never)
private func writeValue(_ out: UnsafeMutableBufferPointer<Float>) { out[0] = 1 }

@inline(never)
private func concreteInout() -> Float {
    var values: ContiguousArray<Float> = [2]
    values.withUnsafeMutableBufferPointer(writeInout)
    return values[0]
}

@inline(never)
private func concreteValue() -> Float {
    var values: ContiguousArray<Float> = [2]
    values.withUnsafeMutableBufferPointer { writeValue($0) }
    return values[0]
}

@main
private struct BufferInoutExposure {
    static func main() {
        let rows: [(String, Float)] = [
            ("concrete-inout", concreteInout()),
            ("concrete-value", concreteValue()),
        ]
        var failures = 0
        for (name, value) in rows {
            print("BUFFER_VALUE \(name) observed=\(String(value.bitPattern, radix: 16))")
            if value.bitPattern != Float(1).bitPattern { failures += 1 }
        }
        print("BUFFER_VALUE_COMPLETE count=2 failures=\(failures)")
        exit(failures == 0 ? 0 : 1)
    }
}
