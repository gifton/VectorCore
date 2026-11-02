import Foundation
import VectorCore

// Consumer-owned witnesses select DynamicVector's existing method explicitly.
struct StoredVector: BinaryCodable {
    let value: DynamicVector
    func encodeBinary() throws -> Data { value.encodeBinary() }
    static func decodeBinary(from data: Data) throws -> StoredVector {
        StoredVector(value: try DynamicVector.decodeBinary(from: data))
    }
}
func encodeGeneric<T: BinaryEncodable>(_ value: T) throws -> Data { try value.encodeBinary() }
func decodeGeneric<T: BinaryDecodable>(_ type: T.Type, _ data: Data) throws -> T {
    try type.decodeBinary(from: data)
}

@main struct B01PublicConsumer {
    static func main() throws {
        let emptyValues = [DynamicVector(), DynamicVector([]), DynamicVector([Float(7)]).dropLast()]
        for value in emptyValues + [DynamicVector([1,-2,3]), DynamicVector(dimension:10000,repeating:0.5)] {
            let wrapper = StoredVector(value:value)
            let concrete = try wrapper.encodeBinary()
            let generic = try encodeGeneric(wrapper)
            let existential: any BinaryEncodable = wrapper
            let erased = try existential.encodeBinary()
            precondition(concrete == generic && generic == erased)
            let decoded = try decodeGeneric(StoredVector.self,concrete)
            precondition(decoded.value.toArray().map(\.bitPattern) == value.toArray().map(\.bitPattern))
        }
        let oversized = StoredVector(value:DynamicVector(dimension:10001,repeating:0))
        let encoded = try oversized.encodeBinary()
        do { _ = try decodeGeneric(StoredVector.self,encoded);fatalError("Expected cap rejection") }
        catch let error as VectorError { precondition(error.kind == .invalidDimension) }
        print("B01_PUBLIC_COMPLETE concrete generic existential decode empty cap")
    }
}
