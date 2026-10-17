import VectorCore
let vector = MixedPrecisionKernels.Vector768FP16(fp16Values: Array(repeating: 0, count: 768))
let pointer: UnsafePointer<UInt16>? = vector.fp16Storage
// Typechecking only: never dereference the legacy pointer.
