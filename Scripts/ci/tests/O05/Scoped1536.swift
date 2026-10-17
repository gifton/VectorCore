import VectorCore
final class Sentinel: Error {}
let patterns: [UInt16] = [0, 0x8000, 1, 0x03ff, 0x0400, 0x7bff, 0x7c00, 0xfc00, 0x7e01, 0xfe55]
let bits = (0..<1536).map { patterns[$0 % patterns.count] }
var owner: MixedPrecisionKernels.Vector1536FP16? = .init(fp16Values: bits)
var calls = 0
let checksum = owner!.withUnsafeFP16BufferPointer { buffer in
    calls += 1
    precondition(buffer.count == 1536 && Array(buffer) == bits)
    return buffer.reduce(UInt64(0)) { $0 + UInt64($1) }
}
precondition(calls == 1 && checksum == bits.reduce(UInt64(0)) { $0 + UInt64($1) })
let owned: [UInt16] = owner!.withUnsafeFP16BufferPointer { Array($0) }
owner = nil
precondition(owned == bits)
let sentinel = Sentinel()
do {
    try MixedPrecisionKernels.Vector1536FP16(fp16Values: bits).withUnsafeFP16BufferPointer { buffer in
        precondition(Array(buffer) == bits)
        throw sentinel
    }
    preconditionFailure("throwing callback returned")
} catch let error as Sentinel {
    precondition(error === sentinel)
}
print("O05_SCOPED_1536_COMPLETE")
