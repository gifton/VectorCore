import Foundation
@testable import VectorCore

func params(_ scale: Float) throws -> LinearQuantizationParams {
    let payload: [String: Any] = ["scale": scale, "zeroPoint": 0, "minValue": -1, "maxValue": 1, "isSymmetric": true]
    return try JSONDecoder().decode(LinearQuantizationParams.self, from: JSONSerialization.data(withJSONObject: payload))
}
func decoded<V: QuantizedVectorINT8>(_ v: V) -> [Double] {
    v.storage.flatMap { lane in
        (0..<4).map { Double(v.quantizationParams.scale) * (Double(lane[$0]) - Double(v.quantizationParams.zeroPoint)) }
    }
}
func emit(_ name: String, _ actual: Float, _ expected: Double) {
    let rounded = Float(expected)
    print("\(name) actual=\(actual) actualBits=\(actual.bitPattern) referenceDouble=\(expected) referenceFloat=\(rounded) referenceBits=\(rounded.bitPattern)")
}
func probe<V: QuantizedVectorINT8>(_ type: V.Type, dot: (V,V)->Float) throws {
    let n = V.laneCount * 4
    let tiny = V(repeating: 1e-23)
    let x = decoded(tiny)
    emit("dot/\(n)/underflow", dot(tiny,tiny), x.reduce(0) { $0 + $1 * $1 })
    let codes = ContiguousArray(repeating: SIMD4<Int8>(repeating: 1), count: V.laneCount)
    let a = V(storage: codes, params: try params(1))
    let b = V(storage: codes, params: try params(Float(1).nextUp))
    let da = decoded(a), db = decoded(b)
    let expected = sqrt(zip(da,db).reduce(0) { $0 + ($1.0-$1.1)*($1.0-$1.1) })
    emit("euclidean/\(n)/unequalScale", QuantizedKernels.euclidean_generic_int8(query:a,candidate:b), expected)
    let zero = V(storage: ContiguousArray(repeating: .zero,count:V.laneCount),params:try params(1))
    for magnitude: Float in [1e-30,1e20] {
        var values = Array(repeating:Float(0),count:n); values[0]=magnitude
        let finite = try V.FloatVectorType(values)
        emit("mixedEuclidean/\(n)/\(magnitude)", QuantizedKernels.euclidean_generic_mixed(query:zero,candidate:finite),Double(magnitude))
        let candidates = SoAINT8<V.FloatVectorType>(from:[try V.FloatVectorType(Array(repeating:0,count:n))],params:try params(1))
        var result:[Float]=[-123]
        result.withUnsafeMutableBufferPointer { QuantizedKernels.batchEuclidean_generic_mixed(query:finite,candidates:candidates,results:$0) }
        emit("SoAMixedEuclidean/\(n)/\(magnitude)",result[0],Double(magnitude))
    }
    let ones = try V.FloatVectorType(Array(repeating:Float(1).nextUp,count:n))
    let soa = SoAINT8<V.FloatVectorType>(from:[ones],params:try params(Float(1).nextUp))
    var result:[Float]=[-123]
    result.withUnsafeMutableBufferPointer { QuantizedKernels.batchEuclidean_generic_int8(query:a,candidates:soa,results:$0) }
    emit("SoAInt8Euclidean/\(n)/unequalScale",result[0],expected)
}
try probe(Vector512INT8.self,dot:QuantizedKernels.dotProduct512)
try probe(Vector768INT8.self,dot:QuantizedKernels.dotProduct768)
try probe(Vector1536INT8.self,dot:QuantizedKernels.dotProduct1536)
