// Bounded source-shape control; see the adjacent README for its tested scope.
@inline(never)
func writeOne(_ out: inout UnsafeMutableBufferPointer<Float>) {
    out[0] = 1
}

@inline(never)
func observedValue() -> Float {
    var values = [Float](repeating: 2, count: 1)
    values.withUnsafeMutableBufferPointer(writeOne)
    return values[0]
}

let observed = observedValue()
print("LOST_STORE observed=\(observed) bits=\(String(observed.bitPattern, radix: 16)) expected=1.0")
