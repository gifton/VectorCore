import Foundation
import VectorCore

@main struct Probe {
 static func main() throws {
  if CommandLine.arguments.contains("overflow") {
   _ = try PCAModel.fit([DynamicVector([Float(1)]), DynamicVector([Float(2)])], components: 1, config: PCAConfig(oversampling: Int.max))
   return
  }
  let providers: [(String, any LinearAlgebraProvider)] = [("Swift", SwiftLinearAlgebraProvider()), ("LAPACK", LAPACKLinearAlgebraProvider())]
  for (name, p) in providers {
   for scale: Float in [1, 1e-12, 1e15] {
    let s = try p.svdThin([scale, 0, scale, scale], rows: 2, columns: 2)
    let correlation = s.u[0] * s.u[2] + s.u[1] * s.u[3]
    print("SVD", name, "scale", scale, "scaled singular values", s.singularValues.map { $0 / scale }, "U column correlation", correlation)
   }
   for scale: Float in [1e-25, 1e20] {
    let q = try p.qrThin([scale, scale], rows: 2, columns: 1)
    print("QR", name, "scale", scale, "q", q.q, "r", q.r)
   }
  }
  let samples = [DynamicVector([Float(16777216)]), DynamicVector([Float(16777218)])]
  let m = try PCAModel.fit(samples, components: 1)
  print("PCA mean", m.mean, "variance", m.explainedVariance, "projected", try m.transform(samples), "expected mean 16777217, variance 2")
 }
}
