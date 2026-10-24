import Foundation
import VectorCore
import VectorCoreC

@_silgen_name("v02_reset") func resetCounter()
@_silgen_name("v02_count") func readCounter() -> UInt64

struct RouteFailure: Error { let message: String }
func require(_ condition: Bool, _ message: String) throws { if !condition { throw RouteFailure(message: message) } }
func codes(_ dim: Int, candidate: Bool) -> [Int8] {
    var result: [Int8] = []
    for index in 0..<dim {
        let value: Int = candidate ? ((index * 5 + 3) % 19 - 9) : (index % 15 - 7)
        result.append(Int8(value))
    }
    return result
}
func storage(_ bytes: [Int8]) -> ContiguousArray<SIMD4<Int8>> {
    ContiguousArray(stride(from: 0, to: bytes.count, by: 4).map { SIMD4(bytes[$0], bytes[$0+1], bytes[$0+2], bytes[$0+3]) })
}
func parameters(_ mode: String) throws -> (LinearQuantizationParams, LinearQuantizationParams) {
    let unit = LinearQuantizationParams(minValue: -127, maxValue: 127)
    switch mode {
    case "unequal": return (unit, LinearQuantizationParams(minValue: -254, maxValue: 254))
    case "asymmetric": let p = LinearQuantizationParams(minValue: -30, maxValue: 50, symmetric: false); return (p,p)
    case "nonzero-zero-point":
        let p = LinearQuantizationParams(minValue: 0, maxValue: 255, symmetric: false)
        try require(!p.isSymmetric && p.zeroPoint != 0, "asymmetric nonzero zero-point fixture")
        return (p,p)
    default: return (unit,unit)
    }
}
func oracle(_ q: [Int8], _ c: [Int8], _ qp: LinearQuantizationParams, _ cp: LinearQuantizationParams) -> [String: Double] {
    var dot: Int64 = 0, qs: Int64 = 0, cs: Int64 = 0
    var differenceSquare = 0.0
    for index in q.indices {
        let a = Int64(q[index]) - Int64(qp.zeroPoint), b = Int64(c[index]) - Int64(cp.zeroPoint)
        dot += a * b; qs += a * a; cs += b * b
        let difference = Double(a) * Double(qp.scale) - Double(b) * Double(cp.scale)
        differenceSquare += difference * difference
    }
    return ["euclidean": sqrt(differenceSquare), "dot": Double(dot) * (Double(qp.scale) * Double(cp.scale)),
            "cosine": 1 - max(-1, min(1, Double(dot) / (sqrt(Double(qs)) * sqrt(Double(cs)))))]
}
func check(dim: Int, mode: String, metric: String, expected: Double, calls: Int, operation: () -> Float) throws {
    resetCounter()
    let actual = operation()
    let observed = readCounter()
    try require(observed == UInt64(calls), "count D=\(dim) \(mode) \(metric): \(observed) != \(calls)")
    try require(actual.isFinite && abs(Double(actual) - expected) <= max(1e-5, abs(expected) * 2e-6), "oracle D=\(dim) \(mode) \(metric): \(actual) vs \(expected)")
    let row: [String: Any] = ["dimension": dim, "mode": mode, "metric": metric, "calls": observed, "expectedCalls": calls, "actual": actual, "reference": expected]
    print("V02_ROUTE " + String(decoding: try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]), as: UTF8.self))
}
func width512(enabled: Bool) throws {
    let q = codes(512, candidate: false), c = codes(512, candidate: true)
    for mode in ["equal", "unequal", "asymmetric", "nonzero-zero-point"] {
        let (qp, cp) = try parameters(mode)
        let query = Vector512INT8(storage: storage(q), params: qp)
        let candidate = Vector512INT8(storage: storage(c), params: cp)
        let expected = oracle(q, c, qp, cp)
        let fusedCount = enabled && ["equal", "unequal"].contains(mode) ? 3 : 0
        try check(dim: 512, mode: mode, metric: "euclidean", expected: expected["euclidean"]!, calls: enabled && mode == "equal" ? 3 : 0) { query.euclideanDistance(to: candidate) }
        try check(dim: 512, mode: mode, metric: "dot", expected: expected["dot"]!, calls: fusedCount) { query.dotProduct(candidate) }
        try check(dim: 512, mode: mode, metric: "cosine", expected: expected["cosine"]!, calls: fusedCount) { query.cosineDistance(to: candidate) }
    }
}
func width768(enabled: Bool) throws {
    let q = codes(768, candidate: false), c = codes(768, candidate: true)
    for mode in ["equal", "unequal", "asymmetric", "nonzero-zero-point"] {
        let (qp, cp) = try parameters(mode)
        let query = Vector768INT8(storage: storage(q), params: qp)
        let candidate = Vector768INT8(storage: storage(c), params: cp)
        let expected = oracle(q, c, qp, cp)
        let fusedCount = enabled && ["equal", "unequal"].contains(mode) ? 3 : 0
        try check(dim: 768, mode: mode, metric: "euclidean", expected: expected["euclidean"]!, calls: enabled && mode == "equal" ? 3 : 0) { query.euclideanDistance(to: candidate) }
        try check(dim: 768, mode: mode, metric: "dot", expected: expected["dot"]!, calls: fusedCount) { query.dotProduct(candidate) }
        try check(dim: 768, mode: mode, metric: "cosine", expected: expected["cosine"]!, calls: fusedCount) { query.cosineDistance(to: candidate) }
    }
}
func width1536(enabled: Bool) throws {
    let q = codes(1536, candidate: false), c = codes(1536, candidate: true)
    for mode in ["equal", "unequal", "asymmetric", "nonzero-zero-point"] {
        let (qp, cp) = try parameters(mode)
        let query = Vector1536INT8(storage: storage(q), params: qp)
        let candidate = Vector1536INT8(storage: storage(c), params: cp)
        let expected = oracle(q, c, qp, cp)
        let fusedCount = enabled && ["equal", "unequal"].contains(mode) ? 3 : 0
        try check(dim: 1536, mode: mode, metric: "euclidean", expected: expected["euclidean"]!, calls: enabled && mode == "equal" ? 3 : 0) { query.euclideanDistance(to: candidate) }
        try check(dim: 1536, mode: mode, metric: "dot", expected: expected["dot"]!, calls: fusedCount) { query.dotProduct(candidate) }
        try check(dim: 1536, mode: mode, metric: "cosine", expected: expected["cosine"]!, calls: fusedCount) { query.cosineDistance(to: candidate) }
    }
}
@main struct Main {
    static func main() throws {
        let enabled = CommandLine.arguments.dropFirst().first == "1"
        try require(VectorCoreCompiledFeatures.useCKernels == enabled, "linked feature mismatch")
        print("V02_ROUTE_ENTER")
        try width512(enabled: enabled); try width768(enabled: enabled); try width1536(enabled: enabled)
        let q = codes(36, candidate: false), c = codes(36, candidate: true)
        for n in [0,1,3,4,15,16,17,31,32,33] {
            let reference = (0..<n).reduce(Int64(0)) { $0 + Int64(q[$1]) * Int64(c[$1]) }
            resetCounter()
            let actual = q.withUnsafeBufferPointer { a in c.withUnsafeBufferPointer { b in vc_dot_int8(a.baseAddress!, b.baseAddress!, n) } }
            try require(Int64(actual) == reference && readCounter() == 1, "direct C tail \(n)")
            print("V02_C_TAIL \(n) calls=1")
        }
        print("V02_ROUTE_COMPLETE widths=3 routes=36 tails=10")
    }
}
