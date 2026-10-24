import XCTest
@testable import VectorCore

func q01Params(_ scale: Float, _ zero: Int32 = 0, symmetric: Bool = true) throws -> LinearQuantizationParams {
    let data = try JSONSerialization.data(withJSONObject: ["scale": Double(scale), "zeroPoint": Int64(zero), "minValue": -1, "maxValue": 1, "isSymmetric": symmetric])
    return try JSONDecoder().decode(LinearQuantizationParams.self, from: data)
}

final class QuantizationRangeContractTests: XCTestCase {
    func testCalibrationEndpoints() {
        let tiny = Float.leastNonzeroMagnitude
        for bounds: (Float, Float) in [(tiny, tiny), (1,2), (-2,-1), (-1,2), (0,0), (-Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)] {
            for symmetric in [true, false] {
                let p = LinearQuantizationParams(minValue: bounds.0, maxValue: bounds.1, symmetric: symmetric)
                XCTAssertTrue(p.scale.isFinite && p.scale > 0)
                for x in [bounds.0, bounds.1] {
                    let y = p.dequantize(p.quantize(x))
                    let tolerance = max(Double(tiny), Double(p.scale) + 2 * Double(abs(x).ulp) + 2 * Double(abs(y).ulp))
                    XCTAssertTrue(y.isFinite && tolerance.isFinite)
                    XCTAssertLessThanOrEqual(abs(Double(x)-Double(y)), tolerance)
                }
                for code in -128...127 { XCTAssertTrue(p.dequantize(Int8(code)).isFinite) }
                if bounds.0 == tiny { XCTAssertEqual(p.scale, tiny) }
                if !symmetric {
                    let other = QuantizationSchemes.computeQuantizationParams(values: [bounds.0,bounds.1], strategy: .perVector)
                    XCTAssertEqual(other.scales[0], p.scale)
                    XCTAssertEqual(other.offsets[0], Float(p.zeroPoint))
                }
            }
        }
    }
}

import Testing

@Suite(.serialized)
struct Q01ChildContracts {
    private func route<V: QuantizedVectorINT8>(_ type: V.Type, name: String, value: Float) throws where V.Scalar == Float {
        let values = Array(repeating: value, count: V.laneCount * 4)
        let fp = try V.FloatVectorType(values)
        switch name {
        case "repeating": _ = V(repeating: value)
        case "from": _ = V(from: fp, params: nil)
        case "supplied": _ = V(from: fp, params: try q01Params(1))
        case "soa": _ = SoAINT8<V.FloatVectorType>(from: [fp])
        case "soaSupplied": _ = SoAINT8<V.FloatVectorType>(from: [fp], params: try q01Params(1))
        case "calibrate": _ = QuantizationCalibrator.calibrate(vectors: [fp])
        case "distribution": _ = QuantizationCalibrator.calibrateWithDistribution(vectors: [fp])
        case "borrow":
            var q = V(repeating: 1)
            q.withUnsafeMutableBufferPointer { $0[0] = value }
        case "legacyBorrow":
            var q = V(storage: ContiguousArray(repeating: .zero, count: V.laneCount), params: try q01Params(.greatestFiniteMagnitude, .min, symmetric: false))
            q.withUnsafeMutableBufferPointer { _ in }
        case "capacityMixed", "capacityInteger":
            let finite = try V.FloatVectorType(Array(repeating: Float(1), count: V.laneCount * 4))
            let candidates = SoAINT8<V.FloatVectorType>(from: [finite])
            var results: [Float] = []
            results.withUnsafeMutableBufferPointer {
                if name == "capacityMixed" { QuantizedKernels.batchEuclidean_generic_mixed(query: finite, candidates: candidates, results: $0) }
                else { QuantizedKernels.batchEuclidean_generic_int8(query: V(from: finite, params: candidates.quantizationParams), candidates: candidates, results: $0) }
            }
        default: throw NSError(domain: "Q01 unknown route", code: 1)
        }
    }
    @Test func childProbe() throws {
        guard let specification = ProcessInfo.processInfo.environment["Q01_CHILD"] else { return }
        let parts = specification.split(separator: ":").map(String.init)
        let name = parts[0]
        let value: Float = parts.count > 2 ? (["nan": Float.nan, "inf": Float.infinity, "negativeInf": -Float.infinity][parts[2]] ?? 1) : 1
        switch name {
        case "outlier": print("Q01_VALUE \(try q01Params(.leastNonzeroMagnitude).quantize(.greatestFiniteMagnitude))")
        case "scalar": _ = try q01Params(1).quantize(value)
        case "scheme": _ = QuantizationSchemes.computeQuantizationParams(values:[value],strategy:.perVector)
        case "bounds": _ = LinearQuantizationParams(minValue:value,maxValue:value)
        case "reversed": _ = LinearQuantizationParams(minValue:2,maxValue:1)
        case "perDimension": _ = QuantizationSchemes.computeQuantizationParams(values:[],strategy:.perDimension)
        case "perChannel": _ = QuantizationCalibrator.calibrate(vectors:[Vector512Optimized](),strategy:.perChannel)
        case "percentile":
            let percentile: Float = ["zero":0,"negative":-1,"over":101][parts[2]] ?? value
            _ = QuantizationCalibrator.calibrateWithDistribution(vectors:[Vector512Optimized](),percentile:percentile)
        default:
            switch parts[1] {
            case "512": try route(Vector512INT8.self,name:name,value:value)
            case "768": try route(Vector768INT8.self,name:name,value:value)
            default: try route(Vector1536INT8.self,name:name,value:value)
            }
        }
        print("Q01_ACCEPTED")
    }
    private func runChild(_ specification: String, diagnostic: String?) throws {
        var env = ProcessInfo.processInfo.environment
        env["Q01_CHILD"] = specification; env["SWIFT_BACKTRACE"] = "enable=no"
        let directory = env["Q01_CHILD_LOG_DIR"] ?? "/private/tmp"
        let url = URL(fileURLWithPath:"\(directory)/Q01-child-\(specification.replacingOccurrences(of: ":", with: "-"))-\(UUID().uuidString).log")
        _ = FileManager.default.createFile(atPath:url.path,contents:nil)
        let file = try FileHandle(forWritingTo:url)
        let process = try launchChildProbe(.swiftTesting(filter: "Q01ChildContracts/childProbe"), environment: env, output: file,
                                           deadline: 15, cleanup: .kill(seconds: 2))
        if process.timedOut {
            let cleanup = process.cleanup
            try file.close()
            Issue.record("Q01 child execution deadline: \(url.path); cleanup=\(cleanup)")
            return
        }
        try file.close()
        let log = try String(contentsOf:url,encoding:.utf8)
        if let diagnostic {
            #expect(process.terminationReason == .uncaughtSignal,"\(url.path): \(log)")
            #expect(log.contains(diagnostic),"\(url.path): \(log)")
            #expect(!log.contains("Q01_ACCEPTED"))
        } else {
            #expect(process.terminationStatus == 0,"\(url.path): \(log)")
            #expect(log.contains("Q01_VALUE 127"))
        }
    }
    @Test func finiteOutlierSaturates() throws { try runChild("outlier",diagnostic:nil) }
    @Test func namedRejectionMatrix() throws {
        let finite = "Quantization requires finite input coordinates"
        for bad in ["nan","inf","negativeInf"] {
            for name in ["scalar","scheme"] { try runChild("\(name):512:\(bad)",diagnostic:finite) }
            try runChild("bounds:512:\(bad)",diagnostic:"Quantization requires finite ordered bounds")
            for width in [512,768,1536] {
                for name in ["repeating","from","supplied","soa","soaSupplied","calibrate","distribution","borrow"] {
                    try runChild("\(name):\(width):\(bad)",diagnostic:finite)
                }
            }
        }
        for width in [512,768,1536] {
            try runChild("legacyBorrow:\(width):valid",diagnostic:finite)
            for name in ["capacityMixed","capacityInteger"] {
                try runChild("\(name):\(width):valid",diagnostic:"Quantized batch results capacity must cover vectorCount")
            }
        }
        for bad in ["zero","negative","over","nan","inf","negativeInf"] {
            try runChild("percentile:512:\(bad)",diagnostic:"Quantization percentile must be finite and in (0,100]")
        }
        try runChild("reversed",diagnostic:"Quantization requires finite ordered bounds")
        try runChild("perDimension",diagnostic:"Quantization perDimension calibration requires a dataset-level API")
        try runChild("perChannel",diagnostic:"Quantization perChannel calibration is unsupported")
    }

}

extension QuantizationRangeContractTests {
    func testExponentSweepAndSignedMetadata() throws {
        let magnitudes: [Float] = [.leastNonzeroMagnitude, .leastNormalMagnitude.nextDown, .leastNormalMagnitude, .leastNormalMagnitude.nextUp,
            .greatestFiniteMagnitude] + stride(from:-140,through:120,by:7).map { Float(pow(2.0,Double($0))) }
        for m in magnitudes {
            for (lo,hi) in [(m,m),(m/2,m),(-m,-m/2),(-m,m),(Float(0),m)] {
                for symmetric in [true,false] {
                    let p = LinearQuantizationParams(minValue:lo,maxValue:hi,symmetric:symmetric)
                    for code in -128...127 { XCTAssertTrue(p.dequantize(Int8(code)).isFinite) }
                    for input in [lo,hi] {
                        let decoded = p.dequantize(p.quantize(input))
                        let tolerance = max(Double(Float.leastNonzeroMagnitude), Double(p.scale)+2*Double(abs(input).ulp)+2*Double(abs(decoded).ulp))
                        XCTAssertTrue(input.isFinite && decoded.isFinite && tolerance.isFinite)
                        XCTAssertLessThanOrEqual(abs(Double(input)-Double(decoded)),tolerance)
                    }
                    if !symmetric {
                        let alternate = QuantizationSchemes.computeQuantizationParams(values:[lo,hi],strategy:.perVector)
                        XCTAssertEqual(alternate.scales[0].bitPattern,p.scale.bitPattern)
                        XCTAssertEqual(alternate.offsets[0],Float(p.zeroPoint))
                    }
                }
            }
        }
        for symmetric in [true,false] {
            let p = LinearQuantizationParams(minValue:-0.0,maxValue:0,symmetric:symmetric)
            XCTAssertEqual(p.minValue.bitPattern,Float(-0.0).bitPattern)
            XCTAssertEqual(p.maxValue.bitPattern,Float(0).bitPattern)
            XCTAssertEqual(p.scale,1); XCTAssertEqual(p.zeroPoint,0); XCTAssertEqual(p.isSymmetric,symmetric)
        }
    }

    func testParameterDecodeMatrix() throws {
        let decoder = JSONDecoder()
        decoder.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity:"Inf",negativeInfinity:"-Inf",nan:"NaN")
        let base: [String:Any] = ["scale":1,"zeroPoint":0,"minValue":-1,"maxValue":1,"isSymmetric":true]
        var invalid: [[String:Any]] = []
        for key in ["scale","minValue","maxValue"] {
            for value in ["Inf","-Inf","NaN"] { var v=base;v[key]=value;invalid.append(v) }
        }
        for scale in [0,-1] { var v=base;v["scale"]=scale;invalid.append(v) }
        var reversed=base;reversed["minValue"]=2;invalid.append(reversed)
        var symmetric=base;symmetric["zeroPoint"]=1;invalid.append(symmetric)
        for tuple in invalid {
            let data=try JSONSerialization.data(withJSONObject:tuple)
            XCTAssertThrowsError(try decoder.decode(LinearQuantizationParams.self,from:data)) { XCTAssertTrue($0 is DecodingError) }
            for n in [512,768,1536] {
                let payload:[String:Any] = ["storage":Array(repeating:[0,0,0,0],count:n/4),"quantizationParams":tuple]
                let vectorData=try JSONSerialization.data(withJSONObject:payload)
                switch n {
                case 512: XCTAssertThrowsError(try decoder.decode(Vector512INT8.self,from:vectorData))
                case 768: XCTAssertThrowsError(try decoder.decode(Vector768INT8.self,from:vectorData))
                default: XCTAssertThrowsError(try decoder.decode(Vector1536INT8.self,from:vectorData))
                }
            }
        }
        for z: Int32 in [.min,-129,0,128,.max] {
            for scale: Float in [.leastNonzeroMagnitude,1,.greatestFiniteMagnitude] {
                let p=try q01Params(scale,z,symmetric:false)
                XCTAssertEqual(p.scale.bitPattern,scale.bitPattern);XCTAssertEqual(p.zeroPoint,z)
                let data=try JSONEncoder().encode(p)
                let object=try XCTUnwrap(JSONSerialization.jsonObject(with:data) as? [String:Any])
                XCTAssertEqual(Set(object.keys),["scale","zeroPoint","minValue","maxValue","isSymmetric"])
                XCTAssertEqual(try decoder.decode(LinearQuantizationParams.self,from:data),p)
                for code in -128...127 {
                    let x=Int8(code), expected=Float(Double(scale)*(Double(code)-Double(z)))
                    XCTAssertEqual(p.dequantize(x).bitPattern,expected.bitPattern)
                    let decoded=QuantizedKernels.convertToFP32_NEON(int8Vec:SIMD4(repeating:x),scale:scale,zeroPoint:z)
                    for j in 0..<4 { XCTAssertEqual(decoded[j].bitPattern,expected.bitPattern) }
                }
            }
        }
    }

    private enum BorrowError: Error { case sentinel }
    private func checkConversionAndBorrow<V:QuantizedVectorINT8>(_ type:V.Type) throws where V.Scalar == Float {
        let n=V.laneCount*4
        for zero: Int32 in [.min, .max] {
            let params = try q01Params(.greatestFiniteMagnitude, zero, symmetric: false)
            let original = V(storage: ContiguousArray(repeating: SIMD4(-128, -1, 0, 127), count: V.laneCount), params: params)
            let encoded = try JSONEncoder().encode(original)
            let decoded = try JSONDecoder().decode(V.self, from: encoded)
            XCTAssertEqual(decoded.storage, original.storage)
            XCTAssertEqual(decoded.quantizationParams.scale.bitPattern, params.scale.bitPattern)
            XCTAssertEqual(decoded.quantizationParams.zeroPoint, zero)
            XCTAssertEqual(decoded.quantizationParams.minValue.bitPattern, params.minValue.bitPattern)
            XCTAssertEqual(decoded.quantizationParams.maxValue.bitPattern, params.maxValue.bitPattern)
            XCTAssertEqual(decoded.quantizationParams.isSymmetric, params.isSymmetric)
        }
        for p in [try q01Params(0.25),try q01Params(.leastNonzeroMagnitude),try q01Params(0.25,17,symmetric:false),try q01Params(1,.max,symmetric:false)] {
            let inputs:[Float]=[-Float.greatestFiniteMagnitude,-Float(0.125).nextUp,-0.125,-Float(0.125).nextDown,0,Float(0.125).nextDown,0.125,Float(0.125).nextUp,.greatestFiniteMagnitude]
            let values=(0..<n).map { inputs[$0 % inputs.count] }
            let fp=try V.FloatVectorType(values)
            let v=V(from:fp,params:p)
            for i in 0..<n {
                let expected=Int8(max(-128,min(127,(Double(values[i])/Double(p.scale)+Double(p.zeroPoint)).rounded(.toNearestOrAwayFromZero))))
                XCTAssertEqual(v.storage[i/4][i%4],expected)
            }
            for count in 0...5 {
                let soa=SoAINT8<V.FloatVectorType>(from:Array(repeating:fp,count:count),params:p)
                XCTAssertEqual(soa.quantizationParams,p)
                for d in 0..<soa.dimension {
                    for k in 0..<(soa.simdGroups*4) {
                        XCTAssertEqual(soa.storage[d*soa.simdGroups+k/4][k%4],k<count ? p.quantize(values[d]) : p.quantize(0))
                    }
                }
            }
        }
        for value:Float in [0,.leastNonzeroMagnitude,1,-1,.greatestFiniteMagnitude] {
            let values=Array(repeating:value,count:n)
            let array=try V(values), repeating=V(repeating:value), from=V(from:try V.FloatVectorType(values),params:nil)
            XCTAssertEqual(array.storage,repeating.storage);XCTAssertEqual(array.storage,from.storage)
        }
        for invalid:Float in [.nan,.infinity,-.infinity] {
            var values=Array(repeating:Float(0),count:n);values[1]=invalid;values[n-1]=invalid
            XCTAssertThrowsError(try V(values)) { error in
                guard let error = error as? VectorError else { return XCTFail("Expected VectorError") }
                XCTAssertEqual(error.kind,.invalidData)
                XCTAssertEqual(error.context.additionalInfo["indices"],"1,\(n-1)")
            }
            XCTAssertThrowsError(try V([invalid])) { XCTAssertTrue(String(describing:$0).contains("dimension")) }
        }
        let legacy=try q01Params(0.25,17,symmetric:false)
        let original=V(storage:ContiguousArray(repeating:SIMD4(1,2,3,4),count:V.laneCount),params:legacy)
        let encoder=JSONEncoder();encoder.outputFormatting = .sortedKeys
        let before=try encoder.encode(original)
        for invalid:Float in [1,.nan,.infinity,-.infinity] {
            var value=original
            XCTAssertThrowsError(try value.withUnsafeMutableBufferPointer { buffer in buffer[0]=invalid;throw BorrowError.sentinel }) { XCTAssertTrue($0 is BorrowError) }
            XCTAssertEqual(value.storage,original.storage);XCTAssertEqual(value.quantizationParams,original.quantizationParams)
        }
        var value=original
        value.withUnsafeMutableBufferPointer { $0[0]=12 }
        XCTAssertNotEqual(value.storage,original.storage)
        XCTAssertEqual(try encoder.encode(original),before)
        value=original;value.withUnsafeMutableBufferPointer { _ in }
        XCTAssertTrue(value.quantizationParams.isSymmetric)
        let overflow=V(storage:original.storage,params:try q01Params(.greatestFiniteMagnitude,.min,symmetric:false))
        value=overflow
        XCTAssertThrowsError(try value.withUnsafeMutableBufferPointer { _ in throw BorrowError.sentinel }) { XCTAssertTrue($0 is BorrowError) }
        XCTAssertEqual(value.storage,overflow.storage);XCTAssertEqual(value.quantizationParams,overflow.quantizationParams)
        let empty=SoAINT8<V.FloatVectorType>(from:[])
        XCTAssertEqual(empty.quantizationParams,LinearQuantizationParams(minValue:0,maxValue:0))
        XCTAssertEqual(SoAINT8<V.FloatVectorType>.calibrateParams(vectors:[]),empty.quantizationParams)
    }
    func testConversionAndTransactionalBorrowAllWidths() throws {
        try checkConversionAndBorrow(Vector512INT8.self)
        try checkConversionAndBorrow(Vector768INT8.self)
        try checkConversionAndBorrow(Vector1536INT8.self)
    }
    func testDistributionAndEmptyCalibration() throws {
        for p:Float in [.leastNonzeroMagnitude,50,99.9,100] {
            let empty=QuantizationCalibrator.calibrateWithDistribution(vectors:[Vector512Optimized](),percentile:p)
            XCTAssertEqual(empty,LinearQuantizationParams(minValue:0,maxValue:0))
            for count in [1,2,3,4,5,Int.max-1,Int.max] {
                let index=AffineQuantizationCore.percentileIndex(count:count,percentile:p)
                XCTAssertGreaterThanOrEqual(index,0);XCTAssertLessThan(index,count)
                if p == 100 { XCTAssertEqual(index,count-1) }
            }
        }
        XCTAssertEqual(AffineQuantizationCore.percentileIndex(count:3,percentile:50),1)
        XCTAssertEqual(AffineQuantizationCore.percentileIndex(count:2,percentile:50),0)
        let vector=try Vector512Optimized((0..<512).map { $0%2 == 0 ? -2 : 7 })
        XCTAssertEqual(QuantizationCalibrator.calibrateWithDistribution(vectors:[vector],percentile:50).maxValue,2)
        XCTAssertEqual(QuantizationCalibrator.calibrateWithDistribution(vectors:[vector],percentile:100).maxValue,7)
        for strategy:QuantizationStrategy in [.symmetric,.asymmetric] {
            let p=QuantizationCalibrator.calibrate(vectors:[Vector512Optimized](),strategy:strategy)
            XCTAssertEqual(p,LinearQuantizationParams(minValue:0,maxValue:0,symmetric:strategy == .symmetric))
        }
    }
}
