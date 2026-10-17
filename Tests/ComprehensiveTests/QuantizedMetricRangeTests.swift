import XCTest
@testable import VectorCore

final class QuantizedMetricRangeTests: XCTestCase {
    func check<V: QuantizedVectorINT8>(_ type: V.Type, dot: (V,V)->Float, cosine: (V,V)->Float) throws {
        let n = V.laneCount * 4
        let codes = ContiguousArray(repeating: SIMD4<Int8>(repeating: 1), count: V.laneCount)
        let a = V(storage: codes, params: try q01Params(1))
        let b = V(storage: codes, params: try q01Params(Float(1).nextUp))
        XCTAssertEqual(QuantizedKernels.euclidean_generic_int8(query:a,candidate:b), Float(sqrt(Double(n))*Double(Float.ulpOfOne)))
        let tiny = V(repeating: 1e-23)
        let p = tiny.quantizationParams
        let x = Double(p.scale)*Double(tiny.storage[0][0])
        XCTAssertEqual(dot(tiny,tiny), Float(Double(n)*x*x))
        XCTAssertEqual(cosine(tiny,tiny), 0, accuracy: 8 * Float.ulpOfOne)
        let zero = V(storage: ContiguousArray(repeating: .zero,count:V.laneCount),params:try q01Params(1))
        for magnitude: Float in [1e-30,1e20] {
            var values = Array(repeating:Float(0),count:n); values[0]=magnitude
            let finite = try V.FloatVectorType(values)
            XCTAssertEqual(QuantizedKernels.euclidean_generic_mixed(query:zero,candidate:finite),magnitude)
            let candidates = SoAINT8<V.FloatVectorType>(from:[try V.FloatVectorType(Array(repeating:0,count:n))],params:try q01Params(1))
            var result:[Float]=[-123, -456]
            result.withUnsafeMutableBufferPointer { QuantizedKernels.batchEuclidean_generic_mixed(query:finite,candidates:candidates,results:$0) }
            XCTAssertEqual(result,[magnitude,-456])
        }
        let ones = try V.FloatVectorType(Array(repeating:Float(1).nextUp,count:n))
        let soa = SoAINT8<V.FloatVectorType>(from:[ones],params:try q01Params(Float(1).nextUp))
        var result:[Float]=[-123,-456]
        result.withUnsafeMutableBufferPointer { QuantizedKernels.batchEuclidean_generic_int8(query:a,candidates:soa,results:$0) }
        XCTAssertEqual(result,[Float(sqrt(Double(n))*Double(Float.ulpOfOne)),-456])
        let zeros = ContiguousArray(repeating: SIMD4<Int8>.zero, count: V.laneCount)
        let largeA = V(storage: zeros, params: try q01Params(8388609,-2139095551,symmetric:false))
        let largeB = V(storage: zeros, params: try q01Params(8388610,-2139095296,symmetric:false))
        XCTAssertEqual(QuantizedKernels.euclidean_generic_int8(query:largeA,candidate:largeB),Float(sqrt(Double(n))))
        let mixedA = V(storage: zeros, params: try q01Params(8388609,-2139095041,symmetric:false))
        let mixedB = try V.FloatVectorType(Array(repeating:Float(17944031912787968),count:n))
        XCTAssertEqual(QuantizedKernels.euclidean_generic_mixed(query:mixedA,candidate:mixedB),Float(sqrt(Double(n))))
        var ca = zeros; ca[0] = SIMD4(-127,-126,-125,0)
        var cb = zeros; cb[0] = SIMD4(1,-2,1,0)
        let cancelA = V(storage:ca,params:try q01Params(.greatestFiniteMagnitude,.min,symmetric:false))
        let cancelB = V(storage:cb,params:try q01Params(pow(2,100)))
        XCTAssertEqual(dot(cancelA,cancelB),0)
        let unscaledB = V(storage:cb,params:try q01Params(1))
        XCTAssertEqual(dot(cancelA,unscaledB),0)
        ca[0][2] = -124
        let neighbor = V(storage:ca,params:cancelA.quantizationParams)
        XCTAssertEqual(dot(neighbor,unscaledB),Float.greatestFiniteMagnitude)
        XCTAssertEqual(dot(neighbor,cancelB),Float.infinity)
        let negative = V(storage:ContiguousArray(repeating:SIMD4(repeating:-128),count:V.laneCount),params:try q01Params(1,.max,symmetric:false))
        let positive = V(storage:ContiguousArray(repeating:SIMD4(repeating:127),count:V.laneCount),params:try q01Params(1,.min,symmetric:false))
        // Exact integer oracle: -(2^31+127)^2 * dimension.
        let expectedBits: [Int: UInt32] = [512: 3808428033, 768: 3812622337, 1536: 3821010945]
        close(dot(negative,positive),Float(bitPattern:expectedBits[n]!))
        XCTAssertEqual(cosine(negative,positive),2,accuracy:8*Float.ulpOfOne)
    }
    func testAllWidthsEX04AndExactCancellation() throws {
        try check(Vector512INT8.self,dot:QuantizedKernels.dotProduct512,cosine:QuantizedKernels.cosine512)
        try check(Vector768INT8.self,dot:QuantizedKernels.dotProduct768,cosine:QuantizedKernels.cosine768)
        try check(Vector1536INT8.self,dot:QuantizedKernels.dotProduct1536,cosine:QuantizedKernels.cosine1536)
    }
}

extension QuantizedMetricRangeTests {
    func testSymmetricFinalRounding() throws {
        let ones=ContiguousArray(repeating:SIMD4<Int8>(repeating:1),count:128)
        let zeros=ContiguousArray(repeating:SIMD4<Int8>.zero,count:128)
        for bits:UInt32 in [0x3dcccccd,0x3f800001,0x00800001,0x00000001,0x70ffffff] {
            let scale=Float(bitPattern:bits), p=try q01Params(scale)
            let a=Vector512INT8(storage:ones,params:p),b=Vector512INT8(storage:zeros,params:p)
            XCTAssertEqual(a.euclideanDistance(to:b).bitPattern,Float(Double(scale)*sqrt(512.0)).bitPattern)
        }
    }

    private func close(_ actual:Float,_ expected:Float,file:StaticString=#filePath,line:UInt=#line) {
        if expected.isInfinite { XCTAssertEqual(actual,expected,file:file,line:line);return }
        if expected.isNaN { XCTAssertTrue(actual.isNaN,file:file,line:line);return }
        if expected == 0 { XCTAssertEqual(actual,0,file:file,line:line);return }
        XCTAssertTrue(actual.isFinite,file:file,line:line)
        XCTAssertLessThanOrEqual(abs(Double(actual)-Double(expected)),8*Double(expected.ulp)+8*Double(Float.leastNonzeroMagnitude),file:file,line:line)
    }

    private func matrix<V:QuantizedVectorINT8>(_ type:V.Type,dot:(V,V)->Float,cosine:(V,V)->Float,euclidean:(V,V)->Float,mixed:(V,V.FloatVectorType)->Float,
        reverse:(V.FloatVectorType,V)->Float) throws {
        let n=V.laneCount*4
        let scales:[Float]=[.leastNonzeroMagnitude,Float(pow(2.0,-80)),1,Float(1).nextUp,Float(pow(2.0,80)),.greatestFiniteMagnitude]
        for scale in scales {
            for zero:Int32 in [0,-17,.min,.max] {
                let p=try q01Params(scale,zero,symmetric:zero == 0)
                let a=V(storage:ContiguousArray(repeating:SIMD4(1,-2,3,-4),count:V.laneCount),params:p)
                let b=V(storage:ContiguousArray(repeating:SIMD4(-4,3,-2,1),count:V.laneCount),params:p)
                let da=(0..<n).map { Double(scale)*(Double(a.storage[$0/4][$0%4])-Double(zero)) }
                let db=(0..<n).map { Double(scale)*(Double(b.storage[$0/4][$0%4])-Double(zero)) }
                let dotExpected=zip(da,db).reduce(0.0) { $0+$1.0*$1.1 }
                let distanceExpected=sqrt(zip(da,db).reduce(0.0) { $0+($1.0-$1.1)*($1.0-$1.1) })
                close(dot(a,b),Float(dotExpected));close(QuantizedKernels.dot_generic(query:a,candidate:b),Float(dotExpected))
                close(euclidean(a,b),Float(distanceExpected));close(QuantizedKernels.euclidean_generic_int8(query:a,candidate:b),Float(distanceExpected))
                XCTAssertEqual(cosine(a,a),0,accuracy:8*Float.ulpOfOne)
                let cosExpected=1-dotExpected/(sqrt(da.reduce(0) { $0+$1*$1 })*sqrt(db.reduce(0) { $0+$1*$1 }))
                XCTAssertEqual(cosine(a,b),Float(cosExpected),accuracy:8*Float.ulpOfOne)
            }
        }
        let p=try q01Params(1)
        let zero=V(storage:ContiguousArray(repeating:.zero,count:V.laneCount),params:p)
        let x=V(storage:ContiguousArray(repeating:SIMD4(1,0,-1,0),count:V.laneCount),params:p)
        let opposite=V(storage:ContiguousArray(repeating:SIMD4(-1,0,1,0),count:V.laneCount),params:p)
        let orthogonal=V(storage:ContiguousArray(repeating:SIMD4(0,1,0,-1),count:V.laneCount),params:p)
        XCTAssertEqual(cosine(x,opposite),2,accuracy:8*Float.ulpOfOne)
        XCTAssertEqual(cosine(x,orthogonal),1);XCTAssertEqual(cosine(x,zero),1)
        XCTAssertEqual(dot(x,orthogonal),0)
        for exponent in [-100,-20,0,20,60] {
            let factor=Float(pow(2.0,Double(exponent)))
            let pa=try q01Params(8388609*factor,-2139095551,symmetric:false)
            let pb=try q01Params(8388610*factor,-2139095296,symmetric:false)
            let a=V(storage:zero.storage,params:pa),b=V(storage:zero.storage,params:pb)
            let expected=Float(sqrt(Double(n))*Double(factor))
            close(euclidean(a,b),expected)
            let pm=try q01Params(8388609*factor,-2139095041,symmetric:false)
            let m=V(storage:zero.storage,params:pm)
            let finite=try V.FloatVectorType(Array(repeating:Float(17944031912787968)*factor,count:n))
            close(mixed(m,finite),expected);close(reverse(finite,m),expected)
            // The nearest Float source quantizes to code 2; raw code 0 is not reachable here.
            let soa=SoAINT8<V.FloatVectorType>(from:[try V.FloatVectorType(Array(repeating:Float(17944036190978560)*factor,count:n))],params:pb)
            XCTAssertEqual(soa.storage[0][0],2)
            var output:[Float]=[-777,-888,-999]
            // Values are read inside the closure (lost-store workaround W2; swiftlang/swift#92562).
            let integer: [Float] = output.withUnsafeMutableBufferPointer { buf in
                let suffix=UnsafeMutableBufferPointer(rebasing:buf[1...])
                QuantizedKernels.batchEuclidean_generic_int8(query:a,candidates:soa,results:suffix)
                return Array(buf)
            }
            close(integer[1],Float(sqrt(Double(n))*16777221*Double(factor)));XCTAssertEqual(integer[0],-777);XCTAssertEqual(integer[2],-999)
            let soaMixed=SoAINT8<V.FloatVectorType>(from:[finite],params:pm)
            XCTAssertEqual(soaMixed.storage[0][0],0)
            let mixedOut: [Float] = output.withUnsafeMutableBufferPointer {
                QuantizedKernels.batchEuclidean_generic_mixed(query:finite,candidates:soaMixed,results:$0); return Array($0)
            }
            close(mixedOut[0],expected)
        }
        for count in 0...5 {
            let fp=try V.FloatVectorType(Array(repeating:Float(1),count:n))
            let candidates=SoAINT8<V.FloatVectorType>(from:Array(repeating:fp,count:count),params:p)
            for kind in 0...1 {
                var output=Array(repeating:Float(-777),count:count+2)
                output.withUnsafeMutableBufferPointer { buf in
                    let results=UnsafeMutableBufferPointer(rebasing:buf[1...])
                    if kind == 0 { QuantizedKernels.batchEuclidean_generic_int8(query:zero,candidates:candidates,results:results) }
                    else { QuantizedKernels.batchEuclidean_generic_mixed(query:try! V.FloatVectorType(Array(repeating:0,count:n)),candidates:candidates,results:results) }
                }
                XCTAssertEqual(output.first,-777);XCTAssertEqual(output.last,-777)
                for k in 0..<count { close(output[k+1],Float(sqrt(Double(n)))) }
            }
        }
        for special:Float in [.nan,.infinity,-.infinity] {
            var values=Array(repeating:Float(0),count:n);values[0]=special
            let fp=try V.FloatVectorType(values)
            close(mixed(zero,fp),special.isNaN ? .nan : .infinity)
            close(reverse(fp,zero),special.isNaN ? .nan : .infinity)
            let candidates=SoAINT8<V.FloatVectorType>(from:[try V.FloatVectorType(Array(repeating:0,count:n))],params:p)
            var result:[Float]=[-1]
            // Values are read inside the closure (lost-store workaround W2; swiftlang/swift#92562).
            let observed: [Float] = result.withUnsafeMutableBufferPointer {
                QuantizedKernels.batchEuclidean_generic_mixed(query:fp,candidates:candidates,results:$0); return Array($0)
            }
            close(observed[0],special.isNaN ? .nan : .infinity)
        }
    }
    func testMetricAndBatchMatrixAllWidths() throws {
        try matrix(Vector512INT8.self,dot:{$0.dotProduct($1)},cosine:{$0.cosineDistance(to:$1)},euclidean:{$0.euclideanDistance(to:$1)},mixed:QuantizedKernels.euclidean512,
            reverse:QuantizedKernels.euclidean512)
        try matrix(Vector768INT8.self,dot:{$0.dotProduct($1)},cosine:{$0.cosineDistance(to:$1)},euclidean:{$0.euclideanDistance(to:$1)},mixed:QuantizedKernels.euclidean768,
            reverse:QuantizedKernels.euclidean768)
        try matrix(Vector1536INT8.self,dot:{$0.dotProduct($1)},cosine:{$0.cosineDistance(to:$1)},euclidean:{$0.euclideanDistance(to:$1)},mixed:QuantizedKernels.euclidean1536,
            reverse:QuantizedKernels.euclidean1536)
    }

    func testWideIntegerIndependentConstants() {
        // Constants independently generated with Python arbitrary integers, retained in Q01-exact-oracle.json.
        let cases:[([Int64],Double)] = [
            ([Int64.max,Int64.max,2],18446744073709551616),
            ([-Int64.max,-Int64.max,-2],-18446744073709551616),
            ([Int64.max,Int64.max,2,-1],18446744073709551616),
            ([-Int64.max,-Int64.max,-2,1],-18446744073709551616),
            ([Int64.max,Int64.max,2,-Int64.max,-Int64.max,-2],0),
            ([Int64.max,-Int64.max,-1],-1)
        ]
        for (terms,expected) in cases {
            var sum=AffineWideInteger();for term in terms { sum.add(term) }
            XCTAssertEqual(sum.doubleValue,expected)
        }
        var positive=AffineWideInteger(),negative=AffineWideInteger()
        for _ in 0..<1536 { positive.add(4611686563888258816);negative.add(-4611686563888258816) }
        XCTAssertEqual(positive.doubleValue,Double(bitPattern:4933693392583393316))
        XCTAssertEqual(negative.doubleValue,-Double(bitPattern:4933693392583393316))
    }
}
