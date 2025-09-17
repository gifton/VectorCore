import Foundation

// A stronger black-hole to prevent DCE in Release builds.
// Mixes bytes into a global sink with @_optimize(none) and @inline(never)
// to force a visible side-effect.
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
