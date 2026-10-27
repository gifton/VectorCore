import Foundation
@testable import VectorCore
final class Token { let id: Int; init(_ id: Int) { self.id = id } }
enum Stop: Error { case stop }
func inspect<V: VectorProtocol>(_ type: V.Type, n: Int) throws where V.Scalar == Float, V.Storage == ContiguousArray<SIMD4<Float>> {
 var zero = try V(Array(repeating: Float(0), count:n))
 var altered = zero; altered.storage[n/8][0] = 1
 var negzero = zero; negzero.storage[0][0] = -Float.zero
 print("hash", n, "interiorDistinct", zero != altered, "collision", zero.hashValue == altered.hashValue, "signedZeroEqual",zero == negzero,"signedZeroHash",zero.hashValue == negzero.hashValue)
 zero.storage = [SIMD4<Float>(repeating:1)]
 print("shape", n, zero.scalarCount,zero.storage.count*4,zero.withUnsafeBufferPointer{$0.count})
}
@main struct Probe {
 static func main() async throws {
  if CommandLine.arguments.count > 1 {
   let n=Int(CommandLine.arguments[1])!
   let raw=UnsafeMutableRawPointer.allocate(byteCount:128,alignment:64)
   defer {raw.deallocate()}
   let ptr=raw.bindMemory(to:Float.self,capacity:32).advanced(by:1)
   ptr.initialize(repeating:1,count:n)
   NormalizeKernels.normalizeUnchecked(ptr,dimension:n)
   print("misaligned normalized",n,ptr[0]); return
  }
  try inspect(Vector384Optimized.self,n:384); try inspect(Vector512Optimized.self,n:512); try inspect(Vector768Optimized.self,n:768); try inspect(Vector1536Optimized.self,n:1536)
  for n in [16,17,384,512,768,1536] {
   var v=DynamicVector(dimension:n,repeating:1)
   var changes=0
   for i in 0..<10 {
    let before=v.withUnsafeBufferPointer{UInt(bitPattern:$0.baseAddress!)}
    v[i]=Float(i)
    let after=v.withUnsafeBufferPointer{UInt(bitPattern:$0.baseAddress!)}
    if before != after {changes += 1}
   }
   print("actual DynamicVector address changes",n,changes)
  }
  for n in [3,4,16,32,33,64,65] {
   let original=TriangularMatrix<Float>(size:n,defaultValue:1)
   var changed=original
   changed[0,1]=99
   print("matrix copy",n,original[0,1],changed[0,1])
   var unique=TriangularMatrix<Float>(size:n,defaultValue:1)
   var changes=0
   for _ in 0..<10 {
    let before=unique.storage.withUnsafeBufferPointer{UInt(bitPattern:$0.baseAddress!)}
    unique[0,1]=2
    let after=unique.storage.withUnsafeBufferPointer{UInt(bitPattern:$0.baseAddress!)}
    if before != after {changes += 1}
   }
   print("actual matrix address changes",n,changes)
  }
  var failedCompact=0
  for n in 5...512 {
   let original=TieredStorage(Array(repeating:Float(1),count:n))
   var copy=original; copy[0]=7
   if original[0] != 1 {failedCompact += 1}
  }
  print("compact counts aliasing",failedCompact,"of",508)
  for n in [0,1,4,5,512,513,2048,2049] {
   var x=TieredStorage(Array(repeating:Float(1),count:n))
   if n>0 {try? x.withUnsafeMutableBufferPointer { p in p[0]=19; throw Stop.stop }}
   print("throwing bulk",n,n>0 ? x[0]:0)
  }
  for n in [3,4,32,33,65] {
   weak var weakToken: Token?
   do {let token=Token(n); weakToken=token; let matrix=TriangularMatrix<Token>(size:n,defaultValue:token); withExtendedLifetime(matrix){}}
   print("token retained after matrix dies",n,weakToken != nil)
  }
  var hint=NormalizationHint(vector:DynamicVector([1,0]))
  hint.vector[0]=2; print("direct stale hint",hint.isNormalized,hint.magnitude,hint.vector.magnitude)
  hint=NormalizationHint(vector:DynamicVector([1,0]))
  try? hint.withUnsafeMutableBufferPointer {p in p[0]=3; throw Stop.stop}
  print("throw stale hint",hint.isNormalized,hint.magnitude,hint.vector.magnitude)
  let params=LinearQuantizationParams(minValue:-1,maxValue:1)
  let int8=Vector512INT8(storage:[SIMD4<Int8>(repeating:0)],params:params)
  print("INT8 short accepted",int8.scalarCount,int8.storage.count*4)
 }
}
