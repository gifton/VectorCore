// Bounded source-shape control; see the adjacent README for its tested scope.
@inline(never)
func writeOne(_ out: UnsafeMutableBufferPointer<Float>) {
    out[0] = 1
}

@inline(never)
func observedValue() -> Float {
    var values: [Float] = [2]
    values.withUnsafeMutableBufferPointer { buffer in writeOne(buffer) }
    return values[0]
}

let observed = observedValue()
print("LOST_STORE observed=\(observed) bits=\(String(observed.bitPattern, radix: 16)) expected=1.0")
