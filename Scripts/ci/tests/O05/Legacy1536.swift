import VectorCore
let vector = MixedPrecisionKernels.Vector1536FP16(fp16Values: Array(repeating: 0, count: 1536))
let pointer: UnsafePointer<UInt16>? = vector.fp16Storage
// Typechecking only: never dereference the legacy pointer.
