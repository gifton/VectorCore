// Bounded source-shape control; see the adjacent README for its tested scope.
@inline(never)
func writeOne(_ out: inout UnsafeMutableBufferPointer<Float>) {
    // Intentionally omit the store: every toolchain must be rejected.
}

@inline(never)
func observedValue() -> Float {
    var values: [Float] = [2]
    return values.withUnsafeMutableBufferPointer { buffer in
        writeOne(&buffer)
        return buffer[0]
    }
}

let observed = observedValue()
print("LOST_STORE observed=\(observed) bits=\(String(observed.bitPattern, radix: 16)) expected=1.0")
