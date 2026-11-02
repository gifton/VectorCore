import Foundation
import VectorCore

struct DirectedDistance: DistanceMetric {
    typealias Scalar = Float
    let name = "directed"
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        a[0] - b[0]
    }
}
@main struct Probe {
 static func main() async throws {
  let mode = CommandLine.arguments.dropFirst().first ?? "normal"
  if mode == "singleton" {
    print(TriangularMatrix<Float>.fromFullMatrix([[0]]) as Any)
    return
  }
  if mode == "zero-batch" {
    let result = try await BatchOperations.process([DynamicVector([1])], batchSize: 0) { $0 }
    print(result)
    return
  }
  for n in [999,1000,1024] {
    for base: Float in [10000, 1e18] {
      let vectors = (0..<n).map { DynamicVector([base + Float($0 % 2)]) }
      let s = await BatchOperations.statistics(for: vectors)
      print("stats", n,base,s.meanMagnitude,s.stdMagnitude)
    }
  }
  print("entropy small",Vector<Dim4>(repeating: 1e-9).entropy, Vector<Dim4>(repeating: .greatestFiniteMagnitude).entropy)
  print("entropy large",Vector<Dim128>(repeating: 1e-10).entropy, Vector<Dim128>(repeating: .greatestFiniteMagnitude).entropy)
  for n in [1,4,4097] {
    let soa = SoA512.build(from: Array(repeating: Vector512Optimized(repeating: 1),count:n))
    var output = Array(repeating: Float(-77), count:n+2)
    try output.withUnsafeMutableBufferPointer { ptr in
      try BatchKernels_SoA.batchCosineDistance512Safe(query: Vector512Optimized(repeating:0),soa:soa,out:ptr)
    }
    print("suffix",n,Array(output.suffix(2)))
  }
  let negative = try await BatchOperations.process([DynamicVector([1])],batchSize:-1) { $0 }
  print("negative batch",negative.count)
  for n in [2,100] {
    let rows = (0..<n).map { DynamicVector([Float($0)]) }
    let m = await BatchOperations.pairwiseDistances(rows,metric:DirectedDistance())
    print("asymmetry",n,m[0][1],m[1][0])
  }
  let dot = await BatchOperations.pairwiseDistances([DynamicVector([2])],metric:DotProductDistance())
  print("dot diagonal",dot)
 }
}
