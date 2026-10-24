import Foundation
import VectorCore
func check<V:VectorProtocol>(_ type:V.Type) throws {
 let v=V()
 let data=try JSONEncoder().encode(v)
 var obj=(try? JSONSerialization.jsonObject(with:data)) as? [String:Any]
 if obj != nil {obj!["storage"]=[]; let forged=try JSONSerialization.data(withJSONObject:obj!); let restored=try JSONDecoder().decode(V.self,from:forged); print("decode empty storage",type,restored.scalarCount,String(data:forged,encoding:.utf8)!.prefix(150))}
 else {do { _=try JSONDecoder().decode(V.self,from:Data("[]".utf8)); print("decoded empty array",type)} catch {print("reject empty array",type)}}
}
func genericAdd<V:VectorProtocol>(_ a:inout V,_ b:V){a += b}
func genericSubtract<V:VectorProtocol>(_ a:inout V,_ b:V){a -= b}
@main struct Probe { static func main() async throws {
 try check(Vector384Optimized.self);try check(Vector512Optimized.self);try check(Vector768Optimized.self);try check(Vector1536Optimized.self)
 try check(Vector512INT8.self);try check(Vector768INT8.self);try check(Vector1536INT8.self)
 let b=DynamicVector([4,5,6])
 var a=DynamicVector([1,2]);a.inPlaceAdd(b);print("long add",a.toArray())
 a=DynamicVector([1,2]);a.inPlaceSubtract(b);print("long subtract",a.toArray())
 a=DynamicVector([1,2]);a.inPlaceElementwiseMultiply(b);print("long hadamard",a.toArray())
 a=DynamicVector([1,2]);genericAdd(&a,b);print("long generic +=",a.toArray())
 a=DynamicVector([1,2]);genericSubtract(&a,b);print("long generic -=",a.toArray())
 let cache=SoAFP16Cache512();let zero=Vector512Optimized()
 _=await cache.getOrCreate(for:[zero])
 var falseHits=0
 for coordinate in 0..<512 {
  var different=zero;different[coordinate]=1
  if await cache.get(for:[different]) != nil {falseHits += 1}
 }
 print("cache independent-coordinate false hits",falseHits,"of",512)
}}
