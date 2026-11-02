import Foundation
import VectorCore
@main struct P { static func main() {
 let s:Float=1e-40
 var w=DynamicVector([s,0]); print("before",w.toArray()); w.inPlaceDivide(s); print("direct method",w.toArray())
 var x=DynamicVector([s,0]); x /= s; print("operator",x.toArray())
 let v=DynamicVector([s,0]); print("outofplace",(v/s).toArray(),"input",v.toArray())
 print("scalar",s/s,Float(0)/s)
} }
