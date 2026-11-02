import Foundation
import VectorCore

@main struct ExtraProbe {
 static func main() async throws {
  let mode = CommandLine.arguments.dropFirst().first ?? "normal"
  let range = (Int.max-2)..<Int.max
  switch mode {
  case "factory-zero": print(try VectorTypeFactory.batch(dimension:0,from:[]).count)
  case "pca-overflow": _ = try PCAModel.fit([DynamicVector([1]),DynamicVector([2])],components:1,config:PCAConfig(oversampling:Int.max))
  case "cpu-execute": print(try await CPUComputeProvider.parallel.parallelExecute(items:range) { $0 })
  case "cpu-each": try await CPUComputeProvider.parallel.parallelForEach(items:range) { _ in }
  case "cpu-each-hint": try await CPUComputeProvider.parallel.parallelForEach(items:range,minChunk:256) { _ in }
  case "cpu-reduce": print(try await CPUComputeProvider.parallel.parallelReduce(items:range,initial:0,{ $0.count },+))
  case "cpu-reduce-hint": print(try await CPUComputeProvider.parallel.parallelReduce(items:range,initial:0,minChunk:256,{ $0.count },+))
  case "cpu-large-hint": try await CPUComputeProvider.parallel.parallelForEach(items:1..<3,minChunk:Int.max) { _ in }
  case "negative-heap": print(MinHeap<Int>(from:[],capacity:-1,compare:<).count)
  case "normal":
   print("tupleStride",MemoryLayout<(Int,Float)>.stride)
   print("sequentialNearMax",try await CPUComputeProvider.sequential.parallelExecute(items:range) { $0 })
   let c = SimpleVectorCollection<DynamicVector>(dimension:1)
   for (id,x): (Int,Float) in [(1,0.5),(2,1),(3,2)] { try c.add(id:id,vector:DynamicVector([x])) }
   let results = c.searchDotProduct(query:DynamicVector([1]),k:3)
   print("actualSearchScores",results.results.map { ($0.id,$0.distance,$0.computedScore) })
   let v = Vector<Dim768>(repeating:1)
   print("similarity",try await [v,v].cosineSimilarityMatrix())
   var evaluated = 0
   func message() -> String { evaluated += 1; return "disabled" }
   Logger.configuration.minimumLevel = .critical
   Logger(category:"validation").debug(message())
   print("disabledLoggerEvaluations",evaluated)
   let heap = MinHeap<Int>(from:[1,2,3,4],capacity:2,compare:<)
   var stream = MinHeap<Int>(capacity:2,compare:<)
   for x in [1,2,3,4] { stream.insert(x) }
   print("oppositeHeap",heap.sorted(),stream.sorted())
  default: fatalError("unknown mode")
  }
 }
}
