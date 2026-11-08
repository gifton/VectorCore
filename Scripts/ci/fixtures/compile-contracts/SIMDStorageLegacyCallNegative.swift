import VectorCore

// Compile-negative: the removed member must not remain available through the
// SIMDStorage constraint, even though concrete ContiguousArray still has it.
func legacyGenericCall<S: SIMDStorage>(_ storage: inout S) {
    storage.withUnsafeMutableBufferPointer { _ in }
}
