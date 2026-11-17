// Exact pure-Swift shape from LostStoreCanaryTests.swift. No VectorCore import.
@inline(never)
func writeOne(_ out: UnsafeMutableBufferPointer<Float>) {
    out[0] = 1
}

@inline(never)
func observedValue() -> Float {
    var values: [Float] = [2]
    return values.withUnsafeMutableBufferPointer { buffer in
        writeOne(buffer)
        return buffer[0]
    }
}

let observed = observedValue()
print("LOST_STORE observed=\(observed) bits=\(String(observed.bitPattern, radix: 16)) expected=1.0")
