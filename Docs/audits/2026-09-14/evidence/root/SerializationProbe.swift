import Foundation
import VectorCore
@main struct SerializationProbe {
 static func main() throws {
  let empty = DynamicVector()
  let encoded = empty.encodeBinary()
  print("empty",empty.scalarCount,"bytes",encoded.count)
  do { _ = try DynamicVector.decodeBinary(from: encoded); print("empty binary accepted") } catch { print("empty binary rejected",error) }
  let restored = try JSONDecoder().decode(DynamicVector.self, from: JSONEncoder().encode(empty))
  print("empty JSON dimension",restored.scalarCount)
  let bits: [UInt32] = [0,0x80000000,0x7fc12345,0x7f800000,0xff800000,1]
  let vector = DynamicVector(bits.map(Float.init(bitPattern:)))
  let bytes = vector.encodeBinary()
  print("binary hex",bytes.map { String(format:"%02x", $0) }.joined())
  let decoded = try DynamicVector.decodeBinary(from: bytes)
  print("bit patterns match", decoded.toArray().map(\.bitPattern) == bits)
  var padding = Data([99,98,97]); padding.append(bytes)
  let slice = padding[3...]
  print("slice start",slice.startIndex,"bits match",try DynamicVector.decodeBinary(from:slice).toArray().map(\.bitPattern) == bits)
  let huge = DynamicVector(Array(repeating: Float(0),count:10001)).encodeBinary()
  do { _ = try DynamicVector.decodeBinary(from:huge); print("large binary accepted") } catch { print("large binary rejected",error) }
  print("FP32 handle",MemoryLayout<Vector512Optimized>.stride,"FP16 handle",MemoryLayout<MixedPrecisionKernels.Vector512FP16>.stride)
 }
}
