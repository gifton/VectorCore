import Foundation
import Testing
@testable import VectorCore

@Suite struct DecodedInvariantTests {
    private struct Envelope: Decodable { let result: TopKResult }
    @Test(arguments: [(0,1),(1,0),(2,1),(1,2)])
    func topKRejectsMismatchedCounts(_ counts: (Int,Int)) throws {
        let object: [String:Any] = ["result":["indices":Array(repeating:7,count:counts.0),"distances":Array(repeating:Float(1),count:counts.1)]]
        let data = try JSONSerialization.data(withJSONObject: object)
        do {
            let value = try JSONDecoder().decode(Envelope.self, from:data)
            Issue.record("Unexpected successful decoding: indices=\(value.result.indices.count), distances=\(value.result.distances.count)")
        } catch DecodingError.dataCorrupted(let context) {
            #expect(context.codingPath.map(\.stringValue) == ["result","distances"])
            #expect(context.debugDescription.contains(String(counts.0)))
            #expect(context.debugDescription.contains(String(counts.1)))
        } catch { Issue.record("Wrong error family: \(error)") }
    }
}

private enum S03Zero: StaticDimension {
    static let value = 0
}
private enum S03Wire {
    static func decoder() -> JSONDecoder {
        let d=JSONDecoder()
        d.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity:"Inf",negativeInfinity:"-Inf",nan:"NaN")
        return d
    }
    static func encoder() -> JSONEncoder {
        let e=JSONEncoder(); e.outputFormatting=[.sortedKeys]
        e.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity:"Inf",negativeInfinity:"-Inf",nan:"NaN")
        return e
    }
    static func data(_ object:Any) throws -> Data { try JSONSerialization.data(withJSONObject:object,options:[.fragmentsAllowed,.sortedKeys]) }
    static func decode<T:Decodable>(_ type:T.Type,_ object:Any) throws -> T { try decoder().decode(type,from:data(object)) }
    static func fixture(_ name:String) throws -> Data {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try Data(contentsOf:root.appendingPathComponent("Scripts/ci/fixtures/S03-"+name+".json"))
    }
    static func corrupted<T:Decodable>(_ type:T.Type,_ object:Any,path:[String]) throws {
        do { _ = try decode(type,object); Issue.record("Unexpected accepted invalid \(type)") }
        catch DecodingError.dataCorrupted(let context) { #expect(context.codingPath.map(\.stringValue) == path) }
        catch { Issue.record("Expected dataCorrupted, got \(error)") }
    }
    static func ordinaryError<T:Decodable>(_ type:T.Type,_ object:Any,kind:String) throws {
        do { _ = try decode(type,object); Issue.record("Unexpected accepted malformed \(type)") }
        catch DecodingError.keyNotFound { #expect(kind == "missing") }
        catch DecodingError.typeMismatch { #expect(kind == "type") }
        catch DecodingError.dataCorrupted { #expect(kind == "numeric") }
        catch DecodingError.valueNotFound { #expect(kind == "lane") }
        catch { Issue.record("Unexpected error \(error)") }
    }
    static func tuple(_ p:LinearQuantizationParams,expected:[String:Any]) throws {
        let scale=try #require(expected["scale"] as? NSNumber).floatValue
        let low=try #require(expected["minValue"] as? NSNumber).floatValue
        let high=try #require(expected["maxValue"] as? NSNumber).floatValue
        let zero=try #require(expected["zeroPoint"] as? NSNumber).int32Value
        let symmetric=try #require(expected["isSymmetric"] as? Bool)
        #expect(p.scale.isFinite && p.minValue.isFinite && p.maxValue.isFinite)
        #expect(p.scale.bitPattern == scale.bitPattern && p.minValue.bitPattern == low.bitPattern && p.maxValue.bitPattern == high.bitPattern)
        #expect(p.zeroPoint == zero && p.isSymmetric == symmetric)
    }
    static var params:[String:Any] { ["scale":Float(0.25),"zeroPoint":17,"minValue":Float(-7),"maxValue":Float(9),"isSymmetric":false] }
}

extension DecodedInvariantTests {
    @Test func topKValidLabelsValuesAndAdapters() throws {
        for original in [TopKResult(),TopKResult(indices:[Int.min,-1,-1,Int.max],distances:[-0.0,-2,3,Float.leastNonzeroMagnitude]),TopKResult(indices:[4,3,2],distances:[.infinity,-.infinity,.nan])] {
            let decoded=try S03Wire.decoder().decode(TopKResult.self,from:S03Wire.encoder().encode(original))
            #expect(decoded.indices == original.indices)
            try #require(decoded.distances.count == original.distances.count)
            let tuples=decoded.toTuples()
            let adapted=SearchResults(from:decoded,candidatesSearched:-9,searchTimeNanos:UInt64.max,isExhaustive:false)
            let mapped=adapted.mapIDs(String.init)
            #expect(tuples.count == original.count && adapted.count == original.count && mapped.count == original.count)
            #expect(mapped.candidatesSearched == -9 && mapped.searchTimeNanos == UInt64.max && !mapped.isExhaustive)
            for i in original.indices.indices {
                #expect(decoded[i].index == original.indices[i] && tuples[i].index == original.indices[i])
                #expect(adapted[i].id == original.indices[i] && mapped[i].id == String(original.indices[i]))
                for value in [decoded[i].distance,tuples[i].distance,adapted[i].distance,mapped[i].distance] {
                    if original.distances[i].isNaN { #expect(value.isNaN) }
                    else { #expect(value.bitPattern == original.distances[i].bitPattern) }
                }
            }
            #expect(Array(adapted).count == original.count)
        }
        for key in ["indices","distances"] {
            var missing:[String:Any] = ["indices":[1],"distances":[1]];missing.removeValue(forKey:key)
            try S03Wire.ordinaryError(TopKResult.self,missing,kind:"missing")
            var wrong:[String:Any] = ["indices":[1],"distances":[1]];wrong[key]="wrong"
            try S03Wire.ordinaryError(TopKResult.self,wrong,kind:"type")
        }
        try S03Wire.ordinaryError(TopKResult.self,["indices":[UInt64.max],"distances":[1]],kind:"numeric")
        // Finite JSON number outside Float range remains an ordinary numeric decode error.
        do { _ = try JSONDecoder().decode(TopKResult.self,from:Data("{\"indices\":[1],\"distances\":[1e100]}".utf8));Issue.record("Float overflow accepted") }
        catch DecodingError.dataCorrupted {} catch { Issue.record("Wrong overflow error: \(error)") }
    }

    @Test func legacyBaseFixtures() throws {
        func unchanged<T:Codable>(_ type:T.Type,_ name:String) throws {
            let data=try S03Wire.fixture(name)
            let decoded=try S03Wire.decoder().decode(type,from:data)
            #expect(try S03Wire.encoder().encode(decoded) == data)
        }
        try unchanged(TopKResult.self,"topk")
        try unchanged(SearchResults<Int>.self,"search")
        try unchanged(DynamicVector.self,"dynamic")
        try unchanged(Vector512INT8.self,"int8")
        let search=try S03Wire.decoder().decode(SearchResults<Int>.self,from:S03Wire.fixture("search"))
        #expect(search.count == 2 && search[0].score == 9 && search[1].score == nil)
        #expect(search.candidatesSearched == -3 && search.searchTimeNanos == UInt64.max && !search.isExhaustive)
        let one=SearchResult(id:"arbitrary",distance:Float(-3),score:Float(-9))
        #expect(try S03Wire.decoder().decode(SearchResult<String>.self,from:S03Wire.encoder().encode(one)) == one)
    }

    @Test func genericAndOptimizedShapes() throws {
        func generic<D:StaticDimension>(_ type:D.Type) throws {
            let good:[String:Any] = ["elements":Array(repeating:Float(2),count:D.value)]
            let vector=try S03Wire.decode(Vector<D>.self,good)
            #expect(vector.scalarCount == D.value)
            #expect(try S03Wire.encoder().encode(vector) == S03Wire.data(good))
            for count in Set([max(0,D.value-1),D.value+1]) where count != D.value {
                do { _ = try S03Wire.decode(Vector<D>.self,["elements":Array(repeating:0,count:count)]);Issue.record("Bad generic count accepted") }
                catch let e as VectorError { #expect(e.kind == .dimensionMismatch) }
                catch { Issue.record("Wrong generic error: \(error)") }
            }
        }
        try generic(S03Zero.self);try generic(Dim2.self);try generic(Dim3.self);try generic(Dim4.self)
        try generic(Dim8.self);try generic(Dim16.self);try generic(Dim32.self);try generic(Dim64.self)
        try generic(Dim128.self);try generic(Dim256.self);try generic(Dim384.self);try generic(Dim512.self)
        try generic(Dim768.self);try generic(Dim1024.self);try generic(Dim1536.self);try generic(Dim2048.self);try generic(Dim3072.self)
        func optimized<V:VectorProtocol>(_ type:V.Type,width:Int) throws where V.Scalar == Float {
            let values=[Float](repeating:2,count:width)
            let value=try S03Wire.decode(type,values)
            #expect(value.toArray() == values)
            #expect(try S03Wire.encoder().encode(value) == S03Wire.data(values))
            for count in [width-1,width+1] {
                do { _ = try S03Wire.decode(type,Array(repeating:0,count:count));Issue.record("Bad optimized count accepted") }
                catch let e as VectorError { #expect(e.kind == .dimensionMismatch) }
                catch { Issue.record("Wrong optimized error: \(error)") }
            }
        }
        try optimized(Vector384Optimized.self,width:384);try optimized(Vector512Optimized.self,width:512)
        try optimized(Vector768Optimized.self,width:768);try optimized(Vector1536Optimized.self,width:1536)
        for count in [0,1,3] {
            let value=try S03Wire.decode(DynamicVector.self,["dimension":count,"elements":Array(repeating:2,count:count)])
            #expect(value.scalarCount == count)
        }
        for dimension in [Int.min,-1,1,Int.max] {
            try S03Wire.corrupted(DynamicVector.self,["dimension":dimension,"elements":[]],path:["elements"])
        }
        try S03Wire.corrupted(DynamicVector.self,["dimension":0,"elements":[1]],path:["elements"])
    }

    @Test func quantizedShapesAndParameterMatrix() throws {
        func width<V:QuantizedVectorINT8>(_ type:V.Type,lanes:Int) throws {
            let storage=Array(repeating:[-128,-1,0,127],count:lanes)
            for count in [0,lanes-1,lanes+1] {
                try S03Wire.corrupted(type,["storage":Array(repeating:[0,0,0,0],count:count),"quantizationParams":S03Wire.params],path:["storage"])
            }
            for lane in [[0,0,0],[0,0,0,0,0],[0,0,0,128],[0,0,0,-129]] {
                var bad=storage;bad[0]=lane
                do { _ = try S03Wire.decode(type,["storage":bad,"quantizationParams":S03Wire.params]);Issue.record("Malformed SIMD accepted") }
                catch is DecodingError {} catch { Issue.record("Not an ordinary lane decoder error: \(error)") }
            }
            try parameterCases { object,kind in
                let nested:[String:Any] = ["storage":storage,"quantizationParams":object]
                if kind == "valid" {
                    let decoded=try S03Wire.decode(type,nested)
                    #expect(decoded.storage.count == lanes)
                    for lane in decoded.storage { #expect(lane == SIMD4<Int8>(-128,-1,0,127)) }
                    let encoded=try S03Wire.encoder().encode(decoded)
                    let roundtrip=try S03Wire.decoder().decode(type,from:encoded)
                    #expect(roundtrip.storage == decoded.storage)
                    try S03Wire.tuple(decoded.quantizationParams,expected:object)
                    try S03Wire.tuple(roundtrip.quantizationParams,expected:object)
                    let keys=try #require(JSONSerialization.jsonObject(with:encoded) as? [String:Any])
                    #expect(Set(keys.keys) == ["storage","quantizationParams"])
                } else if kind == "invalid" { try S03Wire.corrupted(type,nested,path:["quantizationParams"]) }
                else { try S03Wire.ordinaryError(type,nested,kind:kind) }
            }
        }
        try parameterCases { object,kind in
            if kind == "valid" {
                let decoded=try S03Wire.decode(LinearQuantizationParams.self,object)
                try S03Wire.tuple(decoded,expected:object)
                let encoded=try S03Wire.encoder().encode(decoded)
                let roundtrip=try S03Wire.decoder().decode(LinearQuantizationParams.self,from:encoded)
                try S03Wire.tuple(roundtrip,expected:object)
                let keys=try #require(JSONSerialization.jsonObject(with:encoded) as? [String:Any])
                #expect(Set(keys.keys) == Set(object.keys))
            } else if kind == "invalid" { try S03Wire.corrupted(LinearQuantizationParams.self,object,path:[]) }
            else { try S03Wire.ordinaryError(LinearQuantizationParams.self,object,kind:kind) }
        }
        try width(Vector512INT8.self,lanes:128)
        try width(Vector768INT8.self,lanes:192)
        try width(Vector1536INT8.self,lanes:384)
    }

    private func parameterCases(_ check:([String:Any],String) throws -> Void) throws {
        // Independent tuples, not the production validity predicate or calibrator.
        let base=S03Wire.params
        var valid:[ [String:Any] ] = [base]
        for zero in [Int32.min,Int32.max] { var p=base;p["zeroPoint"]=zero;valid.append(p) }
        var subnormal=base;subnormal["scale"]=Float.leastNonzeroMagnitude;subnormal["minValue"]=Float(4);subnormal["maxValue"]=Float(4);valid.append(subnormal)
        var overflowing=base;overflowing["scale"]=Float.greatestFiniteMagnitude;overflowing["zeroPoint"]=Int32.min;valid.append(overflowing)
        var symmetric=base;symmetric["isSymmetric"]=true;symmetric["zeroPoint"]=0;valid.append(symmetric)
        for p in valid { try check(p,"valid") }
        for value:Any in [0,-1,"NaN","Inf","-Inf"] { var p=base;p["scale"]=value;try check(p,"invalid") }
        for key in ["minValue","maxValue"] {
            for value in ["NaN","Inf","-Inf"] { var p=base;p[key]=value;try check(p,"invalid") }
        }
        var reversed=base;reversed["minValue"]=Float(10);try check(reversed,"invalid")
        var offset=base;offset["isSymmetric"]=true;try check(offset,"invalid")
        for key in ["scale","zeroPoint","minValue","maxValue","isSymmetric"] {
            var p=base;p.removeValue(forKey:key);try check(p,"missing")
            p=base;p[key]=[1,2];try check(p,"type")
        }
        for value in [Int64(Int32.min)-1,Int64(Int32.max)+1] { var p=base;p["zeroPoint"]=value;try check(p,"numeric") }
    }

    @Test func normalizationHintsIgnoreOnlyAdvisoryFields() throws {
        func check<V:IndexableVector>(_ type:V.Type,valid:Any,invalid:Any) throws {
            for metadata:[String:Any] in [[:],["isNormalized":true,"cachedMagnitude":1],["isNormalized":false,"cachedMagnitude":42],["isNormalized":["obsolete"],"cachedMagnitude":["wrong":"type"]]] {
                var object=metadata;object["vector"]=valid
                let hint=try S03Wire.decode(NormalizationHint<V>.self,object)
                #expect(!hint.isNormalized && hint.cachedMagnitude == nil)
                let vector=try S03Wire.decode(type,valid)
                #expect(hint.toArray().map(\.bitPattern) == vector.toArray().map(\.bitPattern))
                object["vector"]=invalid
                do { _ = try S03Wire.decode(NormalizationHint<V>.self,object);Issue.record("Nested invalid vector accepted") }
                catch let error as VectorError { #expect(error.kind == .dimensionMismatch) }
                catch DecodingError.dataCorrupted(let context) { #expect(context.codingPath.map(\.stringValue) == ["vector","elements"]) }
                catch { Issue.record("Wrong nested error: \(error)") }
            }
        }
        try check(Vector<Dim3>.self,valid:["elements":[1,2,3]],invalid:["elements":[1,2]])
        try check(Vector512Optimized.self,valid:Array(repeating:1,count:512),invalid:Array(repeating:1,count:511))
        try check(DynamicVector.self,valid:["dimension":2,"elements":[1,2]],invalid:["dimension":3,"elements":[1,2]])
    }
}
