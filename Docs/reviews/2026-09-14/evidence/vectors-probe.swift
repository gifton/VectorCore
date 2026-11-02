import VectorCore
import Foundation
@main struct Probe {
 static func genericNormalize<V: VectorProtocol>(_ v: V) -> Result<V,VectorError> { v.normalized() }
 static func genericPlusEquals<V: VectorProtocol>(_ a: inout V,_ b: V) { a += b }
 static func main() throws {
  switch CommandLine.arguments.dropFirst().first ?? "safe" {
  case "batch-zero": print(try VectorTypeFactory.batch(dimension: 0, from: []).count)
  case "inplace-short":
   var a = DynamicVector([1,2,3]); a.inPlaceAdd(DynamicVector([4])); print(a.toArray())
  default:
   var a = DynamicVector([1,2]); let b = a
   a.storage = .init(from: [1,2,3])
   print("mutated dynamic shape",a.scalarCount,a.storage.count,a.toArray().count,"equal",a == b,"hashEqual",a.hashValue == b.hashValue)
   var c = DynamicVector([1,2]); c.inPlaceAdd(DynamicVector([4,5,6])); print("inplace-long",c.toArray())
   print("nanApprox", DynamicVector([Float.nan]).isApproximatelyEqual(to: DynamicVector([42])))
   var x = Vector512Optimized(); var y = x; y[128] = 1
   print("interior distinct",x != y,"hashEqual",x.hashValue == y.hashValue)
   x.storage = [SIMD4<Float>(repeating: 1)]
   x.withUnsafeBufferPointer { print("optimized claimed count",$0.count,"actual scalar count",x.storage.count * 4) }
   var g = DynamicVector([1,2]); genericPlusEquals(&g,DynamicVector([4,5,6])); print("generic += long",g.toArray())
   let optimizedTiny = Vector512Optimized(generator: { $0 == 0 ? Float.leastNonzeroMagnitude : 0 }); print("direct optimized normalize",optimizedTiny.normalized()); print("generic optimized normalize", try genericNormalize(optimizedTiny).get().toArray().prefix(2)); print("optimized division tiny",(optimizedTiny / Float.leastNonzeroMagnitude).toArray().prefix(2))
   let tiny = DynamicVector([Float.leastNonzeroMagnitude,0]); print("tiny normalization",try tiny.normalized().get().toArray())
  }
 }
}
