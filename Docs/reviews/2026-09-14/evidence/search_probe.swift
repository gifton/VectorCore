import Foundation
import VectorCore
@main struct Probe {
 static func main() async throws {
  let q = DynamicVector([0,0])
  let cs = [DynamicVector([2.5,0]), DynamicVector([2,2])]
  let m = ChebyshevDistance()
  print("chebyshev scores", cs.map { m.distance(q, $0) })
  print("TopK chebyshev", TopKSelection.nearest(k: 1, query:q, candidates:cs, metric:m).indices)
  print("Operations chebyshev", try await Operations.findNearest(to:q,in:cs,k:1,metric:m).map(\.index))
  let bulk = MinHeap(from:[1,2,3,4], capacity:2, compare:>)
  var streaming = MinHeap<Int>(capacity:2, compare:>)
  for x in [1,2,3,4] { streaming.insert(x) }
  print("bulk heap", bulk.sorted(), "stream heap", streaming.sorted())
  let data = Data(#"{"indices":[0],"distances":[]}"#.utf8)
  let result = try JSONDecoder().decode(TopKResult.self, from:data)
  print("decoded lengths",result.count,result.distances.count,"tuples",result.toTuples().count)
  var legacy = KNearestHeap(k:1)
  legacy.insert(index:0,distance:.nan)
  legacy.insert(index:1,distance:0)
  print("legacy nan",legacy.getSorted())
  let ones = try Vector512Optimized(Array(repeating:Float(1000),count:512))
  for delta:Float in [0.125,0.25,0.5,1,2] {
   var nearArray = ones.toArray(); nearArray[0] += delta
   let near = try Vector512Optimized(nearArray)
   let candidates = [near,ones] + Array(repeating: near,count:254)
   let direct = try await Operations.findNearest(to:ones,in:candidates,k:1)
   let batched = try await Operations.findNearestBatch(queries:Array(repeating:ones,count:8),in:candidates,k:1)
   print("GEMM",delta,"direct",direct,"batch",batched[0])
  }
 }
}
