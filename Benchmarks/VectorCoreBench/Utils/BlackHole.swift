import Foundation

// Anti-elision sink used by the serial benchmark measurement driver.
// Mixes the bytes of the value representation into a global sink with @_optimize(none) and
// @inline(never) so the call has a visible side effect.
//
// Limits, stated as contract rather than measured fact:
// - The sink bytes cover only the representation of `x` itself. For an array, buffer or class handle that
//   is the handle, not the elements it refers to; no annotation here proves traversal of heap elements.
//   Consume produced scores through `OutputFold.fold` (or a numeric reduction) and sink that scalar.
// - The sink is a single unsynchronized global. It is not thread-safe. Call it only from the serial
//   measurement driver (the case body on the measuring task), never from provider worker closures; workers
//   return values to the parent, which sinks them.
//   `BenchmarkSinkConfinementContractTests.testSinkIsConfinedToSerialDriver` lexically rejects `blackHole(` inside provider range closures. Concurrent consumption would need a per-instance sink
//   design, not a global lock on every measurement.
fileprivate final class _BlackHoleBox: @unchecked Sendable { var sink: UInt64 = 0 }
fileprivate let _blackHoleBox = _BlackHoleBox()

@inline(never)
@_optimize(none)
public func blackHole<T>(_ x: T) {
    withUnsafeBytes(of: x) { raw in
        var acc: UInt64 = 0x9E3779B185EBCA87
        var i = 0
        while i < raw.count {
            acc &+= UInt64(raw[i])
            acc &*= 0xBF58476D1CE4E5B9
            i += 1
        }
        _blackHoleBox.sink ^= acc
    }
}

@inline(never)
@_optimize(none)
public func blackHoleInout<T>(_ x: inout T) {
    withUnsafeMutableBytes(of: &x) { raw in
        var acc: UInt64 = 0x94D049BB133111EB
        for b in raw { acc &+= UInt64(b) }
        _blackHoleBox.sink &+= acc
    }
}
