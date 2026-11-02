import Foundation
import Testing
@testable import VectorCore

@Suite("Serialization Edge Cases")
struct SerializationEdgeCaseTests {
    @Test func testDynamicVectorSerialization() throws {
        for values: [Float] in [[], [0], [1, -2.5, 4]] {
            let value = DynamicVector(values)
            let json = try JSONEncoder().encode(value)
            let object = try #require(JSONSerialization.jsonObject(with: json) as? [String: Any])
            #expect(object["dimension"] as? Int == values.count)
            #expect((object["elements"] as? [Any])?.count == values.count)
            #expect(try JSONDecoder().decode(DynamicVector.self, from: json).toArray().map(\.bitPattern) == values.map(\.bitPattern))
            #expect(try DynamicVector.decodeBinary(from: value.encodeBinary()).toArray().map(\.bitPattern) == values.map(\.bitPattern))
        }
    }

    @Test func testJSONToBinaryConversion() throws {
        let json = Data(#"{"dimension":3,"elements":[1,-2.5,0]}"#.utf8)
        let value = try JSONDecoder().decode(DynamicVector.self, from: json)
        #expect(value.encodeBinary() == Data(B01Golden.ordinaryFrame))
        let decoded = try DynamicVector.decodeBinary(from: value.encodeBinary())
        #expect(decoded.toArray().map(\.bitPattern) == B01Golden.ordinaryWords)
    }

    @Test func testExplicitMixedFormatBatchRoundTrips() throws {
        for index in 0..<12 {
            let value = DynamicVector([Float(index), Float(index) / 4])
            let decoded: DynamicVector
            if index.isMultiple(of: 2) {
                decoded = try DynamicVector.decodeBinary(from: value.encodeBinary())
            } else {
                decoded = try JSONDecoder().decode(DynamicVector.self, from: JSONEncoder().encode(value))
            }
            #expect(decoded.toArray().map(\.bitPattern) == value.toArray().map(\.bitPattern))
        }
    }

    @Test func testConcreteOptimizedJSONAndFP32BinaryDestinations() throws {
        func check<V: OptimizedVector, D: StaticDimension>(_ type: V.Type, _ dimension: D.Type) throws {
            let words = (0..<D.value).map { Float($0 % 17 - 8) / 4 }
            let value = try V(words)
            let json = try JSONEncoder().encode(value)
            #expect((try JSONSerialization.jsonObject(with: json) as? [Any])?.count == D.value)
            let decoded = try JSONDecoder().decode(V.self, from: json)
            #expect(decoded.toArray().map(\.bitPattern) == words.map(\.bitPattern))
            let binary = value.encodeBinary()
            #expect(binary.count == 16 + 4 * D.value)
            #expect(try DynamicVector.decodeBinary(from: binary).toArray().map(\.bitPattern) == words.map(\.bitPattern))
            #expect(try Vector<D>.decodeBinary(from: binary).toArray().map(\.bitPattern) == words.map(\.bitPattern))
        }
        try check(Vector384Optimized.self, Dim384.self)
        try check(Vector512Optimized.self, Dim512.self)
        try check(Vector768Optimized.self, Dim768.self)
        try check(Vector1536Optimized.self, Dim1536.self)
    }

    @Test func testQuantizedVectorSerialization() throws {
        let parameterJSON = Data(#"{"scale":0.25,"zeroPoint":17,"minValue":-7,"maxValue":9,"isSymmetric":false}"#.utf8)
        let params = try JSONDecoder().decode(LinearQuantizationParams.self, from: parameterJSON)
        func check<V: QuantizedVectorINT8, D: StaticDimension>(_ type: V.Type, _ dimension: D.Type) throws where V.Scalar == Float {
            let codes = SIMD4<Int8>(-128, -1, 0, 127)
            let storage = ContiguousArray(repeating: codes, count: V.laneCount)
            let value = V(storage: storage, params: params)
            let json = try JSONEncoder().encode(value)
            let object = try #require(JSONSerialization.jsonObject(with: json) as? [String: Any])
            #expect(object["storage"] != nil)
            #expect(object["quantizationParams"] != nil)
            let decoded = try JSONDecoder().decode(V.self, from: json)
            #expect(decoded.storage == storage)
            #expect(decoded.quantizationParams == params)
            let expected = (0..<D.value).map { Float(0.25 * (Double(codes[$0 % 4]) - 17)) }
            #expect(value.toArray().map(\.bitPattern) == expected.map(\.bitPattern))
            let binary = value.encodeBinary()
            #expect(binary.count == 16 + 4 * D.value)
            #expect(try DynamicVector.decodeBinary(from: binary).toArray().map(\.bitPattern) == expected.map(\.bitPattern))
            #expect(try Vector<D>.decodeBinary(from: binary).toArray().map(\.bitPattern) == expected.map(\.bitPattern))
            var invalid = object
            invalid["storage"] = []
            do {
                _ = try JSONDecoder().decode(V.self, from: JSONSerialization.data(withJSONObject: invalid))
                Issue.record("Accepted invalid INT8 storage")
            } catch DecodingError.dataCorrupted { }
            catch { Issue.record("Wrong INT8 decoder error: \(error)") }
        }
        try check(Vector512INT8.self, Dim512.self)
        try check(Vector768INT8.self, Dim768.self)
        try check(Vector1536INT8.self, Dim1536.self)
    }

    @Test func testMixedPrecisionExplicitFP32Serialization() throws {
        func check<V: OptimizedVector, D: StaticDimension>(_ rounded: V, _ dimension: D.Type) throws {
            let words = rounded.toArray().map(\.bitPattern)
            #expect(words == Array(repeating: Float(1).bitPattern, count: D.value))
            #expect(words[0] != Float(1.0001).bitPattern)
            let binary = rounded.encodeBinary()
            #expect(try DynamicVector.decodeBinary(from: binary).toArray().map(\.bitPattern) == words)
            #expect(try Vector<D>.decodeBinary(from: binary).toArray().map(\.bitPattern) == words)
        }
        try check(MixedPrecisionKernels.Vector512FP16(from: Vector512Optimized(repeating: 1.0001)).toFP32(), Dim512.self)
        try check(MixedPrecisionKernels.Vector768FP16(from: Vector768Optimized(repeating: 1.0001)).toFP32(), Dim768.self)
        try check(MixedPrecisionKernels.Vector1536FP16(from: Vector1536Optimized(repeating: 1.0001)).toFP32(), Dim1536.self)
    }

    @Test func testJSONDefaultRejectsNonfinite() {
        for word: UInt32 in [0x7f80_0000, 0xff80_0000, 0x7fc0_0001] {
            do {
                _ = try JSONEncoder().encode(DynamicVector([Float(bitPattern: word)]))
                Issue.record("Default JSON accepted nonfinite value")
            } catch EncodingError.invalidValue { }
            catch { Issue.record("Wrong JSON encoding error: \(error)") }
        }
    }

    @Test func testJSONExplicitNonfiniteStrategyClassification() throws {
        let encoder = JSONEncoder()
        encoder.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "Inf", negativeInfinity: "-Inf", nan: "NaN")
        let decoder = JSONDecoder()
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity: "Inf", negativeInfinity: "-Inf", nan: "NaN")
        let value = DynamicVector([Float.infinity, -Float.infinity, Float(bitPattern: 0x7fc0_0123), 1.25])
        let decoded = try decoder.decode(DynamicVector.self, from: encoder.encode(value))
        #expect(decoded[0].isInfinite && decoded[0].sign == .plus)
        #expect(decoded[1].isInfinite && decoded[1].sign == .minus)
        #expect(decoded[2].isNaN)
        #expect(decoded[3].bitPattern == Float(1.25).bitPattern)
    }

    @Test func testFixedObjectSchemaAndOldArrayRejection() throws {
        let value = try JSONDecoder().decode(Vector<Dim3>.self, from: Data(#"{"elements":[1,2,3]}"#.utf8))
        #expect(value.toArray().map(\.bitPattern) == [Float(1), 2, 3].map(\.bitPattern))
        do {
            _ = try JSONDecoder().decode(Vector<Dim3>.self, from: Data("[1,2,3]".utf8))
            Issue.record("Accepted unsupported array schema")
        } catch DecodingError.typeMismatch { }
        catch { Issue.record("Wrong schema error: \(error)") }
    }

    @Test func testWhitespaceAndNestedVectorArray() throws {
        let json = Data(" [ { \"elements\" : [ 1, 2, 3 ] }, { \"elements\" : [ 4, 5, 6 ] } ] ".utf8)
        let values = try JSONDecoder().decode([Vector<Dim3>].self, from: json)
        #expect(values.count == 2)
        #expect(values[0].toArray().map(\.bitPattern) == [Float(1), 2, 3].map(\.bitPattern))
        #expect(values[1].toArray().map(\.bitPattern) == [Float(4), 5, 6].map(\.bitPattern))
    }

    @Test func testFixtureSpecificPrettyJSONSize() throws {
        let value = DynamicVector([1, 2, 3])
        let encoder = JSONEncoder(); encoder.outputFormatting = .prettyPrinted
        let pretty = try encoder.encode(value)
        #expect(pretty.count > value.encodeBinary().count)
        #expect(try JSONDecoder().decode(DynamicVector.self, from: pretty).toArray().map(\.bitPattern) == value.toArray().map(\.bitPattern))
    }

    @Test func testConcurrentIndependentValueRoundTrips() async throws {
        try await withThrowingTaskGroup(of: Bool.self) { group in
            for index in 0..<10 {
                group.addTask {
                    let value = DynamicVector([Float(index), -Float(index)])
                    let result = try DynamicVector.decodeBinary(from: value.encodeBinary())
                    return result.toArray().map(\.bitPattern) == value.toArray().map(\.bitPattern)
                }
            }
            for try await success in group { #expect(success) }
        }
    }
}
