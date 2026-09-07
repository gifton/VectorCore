import Foundation
import VectorCore

// SplitMix64: the exact seed and conversion below define the benchmark fixtures.
struct Generator {
    var state: UInt64 = 0x5EED_2026_0905_0042
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58_476D_1CE4_E5B9
        value = (value ^ (value >> 27)) &* 0x94D0_49BB_1331_11EB
        return value ^ (value >> 31)
    }
}

@inline(never)
func checksum<I: BinaryInteger>(_ indices: [I], _ scores: [Float]) -> UInt64 {
    precondition(indices.count == scores.count)
    var hash: UInt64 = 0xCBF2_9CE4_8422_2325
    for position in indices.indices {
        hash = (hash ^ UInt64(indices[position])) &* 0x100_0000_01B3
        hash = (hash ^ UInt64(scores[position].bitPattern)) &* 0x100_0000_01B3
    }
    return hash
}

let n = 100_000
let warmups = 5
let samples = 15
let iterations = 10
var consumed: UInt64 = 0
// Untimed downstream compilation/runtime smoke for public inlinable entry points.
let query = Vector512Optimized(repeating: 1)
let candidates = [Vector512Optimized(repeating: 0), Vector512Optimized(repeating: 2)]
let euclidean = TopKSelection.nearestEuclidean512(k: 1, query: query, candidates: candidates)
let dot = TopKSelection.nearestDotProduct512(k: 1, query: query, candidates: candidates)
precondition(euclidean.indices == [0] && dot.indices == [1])
print("downstream_inlinable_smoke,passed")
print("config,n=\(n),warmups=\(warmups),samples=\(samples),iterations=\(iterations),seed=5eed202609050042")
print("dataset,api,k,median_ns,checksum,sample_ns")
for dataset in ["mixed", "duplicates"] {
    var generator = Generator()
    let scores: [Float] = (0..<n).map { _ in
        let random = generator.next()
        if dataset == "duplicates" { return Float(Int(random % 17) - 8) }
        return Float(Int(random >> 40) - 8_388_608) / 128
    }
    for k in [10, 20_000] {
        for api in ["array", "pointer"] {
            var times: [UInt64] = []
            var reference: UInt64?
            // The pointer remains valid across every sample. Hashing happens after
            // each timed selection, so every returned index and score is consumed.
            scores.withUnsafeBufferPointer { buffer in
                for sample in -warmups..<samples {
                    var elapsed: UInt64 = 0
                    for _ in 0..<iterations {
                        let hash: UInt64
                        if api == "array" {
                            let start = DispatchTime.now().uptimeNanoseconds
                            let result = TopKSelection.select(k: k, from: scores)
                            elapsed += DispatchTime.now().uptimeNanoseconds - start
                            precondition(result.count == k)
                            hash = checksum(result.indices, result.distances)
                        } else {
                            let start = DispatchTime.now().uptimeNanoseconds
                            let result = TopKSelection.select(k: k, from: buffer.baseAddress!, count: n)
                            elapsed += DispatchTime.now().uptimeNanoseconds - start
                            precondition(result.indices.count == k)
                            hash = checksum(result.indices, result.distances)
                        }
                        if let expected = reference { precondition(hash == expected) }
                        else { reference = hash }
                        consumed &+= hash
                    }
                    if sample >= 0 { times.append(elapsed / UInt64(iterations)) }
                }
            }
            let median = times.sorted()[times.count / 2]
            print("\(dataset),\(api),\(k),\(median),\(reference!),\(times.map(String.init).joined(separator: ";"))")
        }
    }
}
print("consumed,\(consumed)")
