// Bounded source-shape control; see the adjacent README for its tested scope.
@inline(never)
func writeOne(_ out: inout UnsafeMutableBufferPointer<Float>) {
    out[0] = 1
}

@inline(never)
func observedValue() -> Float {
    var values: [Float] = [2]
    return values.withUnsafeMutableBufferPointer { buffer in
        var copy = buffer
        writeOne(&copy)
        return buffer[0]
    }
}

let observed = observedValue()
print("LOST_STORE observed=\(observed) bits=\(String(observed.bitPattern, radix: 16)) expected=1.0")
