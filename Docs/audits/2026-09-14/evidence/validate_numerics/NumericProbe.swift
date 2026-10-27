import Foundation
@testable import VectorCore
func genericNorm<V: VectorProtocol>(_ v: V) -> Result<V,VectorError> { v.normalized() }
func genericDivide<V: VectorProtocol>(_ v: V, _ s: V.Scalar) -> V { v / s }
@main struct NumericProbe {
 static func main() async throws {
  for s: Float in [Float.leastNonzeroMagnitude,1e-40,1e-38] {
   let v=DynamicVector([s,0]); var w=v; w /= s
   print("divide",s,"generic",genericDivide(v,s).toArray(),"inplace",w.toArray(),"norm",genericNorm(v))
   var out=[Float](repeating:0,count:2)
   [s,0].withUnsafeBufferPointer { p in out.withUnsafeMutableBufferPointer { q in AccelerateFloatProvider.divideByScalar(p.baseAddress!,scalar:s,result:q.baseAddress!,count:2) } }
   print("accelerate divide",s,out)
  }
  for n in [1,4,5,17,512] {
   for s: Float in [1e-38,Float.greatestFiniteMagnitude] {
    let raw=UnsafeMutableRawPointer.allocate(byteCount:n*4,alignment:16)
    let ptr=raw.bindMemory(to:Float.self,capacity:n); ptr.initialize(repeating:s,count:n)
    NormalizeKernels.normalizeUnchecked(ptr,dimension:n)
    let vals=Array(UnsafeBufferPointer(start:ptr,count:n)); let norm=sqrt(vals.reduce(0.0) {$0+Double($1)*Double($1)})
    print("pointer norm",n,s,norm,"first",vals[0]); ptr.deinitialize(count:n); raw.deallocate()
   }
  }
  for s: Float in [1,1e-12,1e15] {
   let a=try SwiftLinearAlgebraProvider().svdThin([s,0,s,s],rows:2,columns:2)
   print("svd",s,a.singularValues.map{$0/s},"Udot",a.u[0]*a.u[2]+a.u[1]*a.u[3])
  }
  for s: Float in [1e-25,1e20] {
   let q=try SwiftLinearAlgebraProvider().qrThin([s,s],rows:2,columns:1)
   let a=try SwiftLinearAlgebraProvider().svdThin([s,s],rows:2,columns:1)
   print("qr",s,q.q,q.r,"single svd",a.u,a.singularValues)
  }
  for offset:Float in [0,16777216] {
   let xs=[DynamicVector([offset]),DynamicVector([offset+2])]
   let m=try PCAModel.fit(xs,components:1)
   print("pca offset",offset,"mean",m.mean,"variance",m.explainedVariance,"proj",try m.transform(xs))
  }
  for n in [999,1000,1024] {
   for base:Float in [10000,1e18] {
    let xs=(0..<n).map{DynamicVector([base+Float($0%2)])}
    print("stats",n,base,await BatchOperations.statistics(for:xs))
   }
  }
  for s: Float in [1e-10,1,Float.greatestFiniteMagnitude] { print("entropy",s,Vector<Dim4>(repeating:s).entropy,Vector<Dim128>(repeating:s).entropy) }
  for v: Float in [.nan,.infinity,-Float.infinity] {
   let a=DynamicVector([v]); print("approx",v,a.isApproximatelyEqual(to:DynamicVector([42])),a.isApproximatelyEqual(to:a))
   do { print("safeLog strict",v,try Vector<Dim4>(repeating:v).safeLog(options:.strict).toArray()) } catch {print("safeLog strict",v,"threw")}
  }
  for values:[Float] in [[1,2],[-10,-1],[1.5,1.5]] {
   let p=QuantizationSchemes.computeQuantizationParams(values:values,strategy:.perVector)
   let restored=values.map {x -> Float in let q=max(-128,min(127,(x/p.scales[0]+p.offsets[0]).rounded()));return (q-p.offsets[0])*p.scales[0]}
   print("calibration",values,p.scales,p.offsets,restored)
  }
  for s: Float in [1e-10,1,1e10] {
   let q512=Vector512INT8(repeating:s), q768=Vector768INT8(repeating:s),q1536=Vector1536INT8(repeating:s)
   print("INT8 cosine",s,q512.cosineDistance(to:q512),q768.cosineDistance(to:q768),q1536.cosineDistance(to:q1536))
  }
  let zero=Vector512Optimized(repeating:0), one=Vector512Optimized(repeating:1),near=Vector512Optimized(repeating:1.0001),big=Vector512Optimized(repeating:70000)
  print("adaptive threshold 0",MixedPrecisionKernels.adaptiveEuclideanDistance(query:one,candidate:near,threshold:0),"reference",one.euclideanDistance(to:near))
  print("adaptive asymmetric",MixedPrecisionKernels.adaptiveEuclideanDistance(query:zero,candidate:big),MixedPrecisionKernels.adaptiveEuclideanDistance(query:big,candidate:zero))
  print("validator NaN",PrecisionValidator.validateVectorEquality512(Vector512Optimized(repeating:.nan),zero,tolerance:0.001))
  for value:Float in [1e-12,65505] {
   MixedPrecisionDiagnostics.shared.enable()
   let q=MixedPrecisionKernels.Vector512FP16(from:Vector512Optimized(repeating:value))
   let stats=MixedPrecisionDiagnostics.shared.getStatistics()
   print("diagnostics",value,"decoded",q.toFP32()[0],"underflow",stats.underflowToZero,"subnormal",stats.subnormalValues,"overflow",stats.overflowToInfinity,"min",stats.minValueSeen)
   MixedPrecisionDiagnostics.shared.disable()
  }
  let tuner=QuantizedAutoTuner.shared;tuner.clearCache()
  let first=tuner.selectOptimalQuantization(for:"dot",dimension:512,candidateCount:2000,accuracyRequirement:0.02)
  let cached=tuner.selectOptimalQuantization(for:"dot",dimension:512,candidateCount:2000,accuracyRequirement:0.2)
  tuner.clearCache();let fresh=tuner.selectOptimalQuantization(for:"dot",dimension:512,candidateCount:2000,accuracyRequirement:0.2)
  print("tuner",first,cached,fresh)
  for spread:Float in [1,0.001,0.00001,1e10] {
   let (a,b)=UMAPLayout.fitABParams(spread:spread,minDist:spread*0.1)
   print("umap curve",spread,a,b,1/(1+Double(a)*pow(Double(spread),2*Double(b))))
  }
  for scale:Float in [1,1e-30,1e30] {
   let g=try KNNGraph(pointCount:3,rowOffsets:[0,2,2,2],neighborIndices:[1,2],distances:[scale,2*scale])
   print("umap smooth",scale,FuzzySimplicialSet.smoothDistances(g).sigma,FuzzySimplicialSet.build(g).weights)
  }
  let graph=try KNNGraph(pointCount:3,rowOffsets:[0,2,4,6],neighborIndices:[1,2,0,2,0,1],distances:[1,2,1,1,2,1])
  print("umap infinite lr",try Operations.umap(graph:graph,config:UMAPConfig(epochs:5,learningRate:.infinity)).coordinates)
  let knn=try KNNGraph.bruteForce([10000,10002,10001].map{DynamicVector([Float($0)])},neighbors:1)
  print("knn reference",knn.neighborIndices,knn.distances)
 }
}
