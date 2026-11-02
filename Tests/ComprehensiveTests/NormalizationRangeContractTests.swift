import Foundation
import Testing
@testable import VectorCore

@Suite struct NormalizationRangeContractTests {
    @Test func finiteExtremeNormalization() throws {
        for scale in [Float.leastNonzeroMagnitude, Float.greatestFiniteMagnitude] {
            let input = DynamicVector([scale, -scale, scale, -scale])
            let output = try input.normalized().get()
            #expect(output[0] == 0.5)
            #expect(output[1] == -0.5)
            #expect(output[2] == 0.5)
            #expect(output[3] == -0.5)
        }
    }

    private func check<V: VectorProtocol>(_ input: V, _ expected: [Float]) throws where V.Scalar == Float {
        let bits = input.toArray().map(\.bitPattern)
        let checked: Result<V,VectorError> = input.normalized()
        n01AssertFloat(try checked.get().toArray(), expected)
        n01AssertFloat(input.normalizedUnchecked().toArray(),expected)
        #expect(input.toArray().map(\.bitPattern) == bits)
    }

    @Test func finiteRangeRouteMatrix() throws {
        for dimension in [1,4,5,17,31,32,33,384,512,768,1536] {
            for scale in [Float.leastNonzeroMagnitude,Float.leastNormalMagnitude,Float(1e-38),Float(1),Float.greatestFiniteMagnitude] {
                let nonzero = dimension == 1 ? 1 : 4
                var values = [Float](repeating: 0,count: dimension)
                var reference = values
                for j in 0..<nonzero { values[j] = j % 2 == 0 ? scale : -scale; reference[j] = (j % 2 == 0 ? 1 : -1) / Float(nonzero).squareRoot() }
                try check(DynamicVector(values),reference)
                for provider: any ArraySIMDProvider in [DefaultArraySIMDProvider(),SwiftSIMDProvider(),AccelerateArraySIMDProvider()] {
                    n01AssertFloat(provider.normalize(values),reference)
                }
                if dimension == 4 {
                    let fixed = try Vector<Dim4>(values)
                    try check(fixed,reference)
                    n01AssertFloat(fixed.normalized(tolerance:0).toArray(),reference)
                    n01AssertFloat(fixed.normalizedFast().toArray(),reference)
                    var mutating = fixed; mutating.normalizeFast(); n01AssertFloat(mutating.toArray(),reference)
                }
                switch dimension {
                case 384: let v = try Vector384Optimized(values); try check(v,reference); n01AssertFloat(try v.normalized().get().toArray(),
                    reference); n01AssertFloat(v.normalizedUnchecked().toArray(),reference)
                case 512: let v = try Vector512Optimized(values); try check(v,reference); n01AssertFloat(try v.normalized().get().toArray(),
                    reference); n01AssertFloat(v.normalizedUnchecked().toArray(),reference)
                case 768: let v = try Vector768Optimized(values); try check(v,reference); n01AssertFloat(try v.normalized().get().toArray(),
                    reference); n01AssertFloat(v.normalizedUnchecked().toArray(),reference)
                case 1536: let v = try Vector1536Optimized(values); try check(v,reference); n01AssertFloat(try v.normalized().get().toArray(),
                    reference); n01AssertFloat(v.normalizedUnchecked().toArray(),reference)
                default: break
                }
            }
        }
        for exponent in [-147,-126,-40,0,80,125] {
            let a = Float(sign:.plus,exponent:exponent,significand:1)
            for values in [[3*a,4*a],[-4*a,3*a]] {
                let reference: [Float] = values[0] > 0 ? [0.6,0.8] : [-0.8,0.6]
                try check(DynamicVector(values),reference)
            }
        }
    }

    @Test func nativeDoubleIndependentOracles() throws {
        for (inputBits, expectedBits) in n01DoubleFixtures {
            let values = inputBits.map(Double.init(bitPattern:))
            let expected = expectedBits.map(Double.init(bitPattern:))
            for reverse in [false,true] {
                let input = N01DoubleVector(reverse ? values.reversed().map { $0 } : values)
                let oracle = reverse ? expected.reversed().map { $0 } : expected
                n01AssertDouble(try input.normalized().get().toArray(),oracle)
                n01AssertDouble(input.normalizedUnchecked().toArray(),oracle)
            }
        }
        for count in [1,4,5,17,31,32,33,384,512,768,1536] {
            for scale in [Double.leastNonzeroMagnitude,Double.leastNormalMagnitude,Double.greatestFiniteMagnitude] {
                let nonzero = count == 1 ? 1 : 4
                var values = [Double](repeating:0,count:count)
                var expected = values
                for i in 0..<nonzero { values[i] = i%2 == 0 ? scale : -scale; expected[i] = (i%2 == 0 ? 1 : -1)/Double(nonzero).squareRoot() }
                n01AssertDouble(try N01DoubleVector(values).normalized().get().toArray(),expected)
            }
        }
    }

    @Test func pointerNaturalOffsetsAndSentinels() {
        for count in [0,1,4,5,17,31,32,33] {
            for offset in 0..<4 {
                for scale in [Float.leastNonzeroMagnitude,Float.greatestFiniteMagnitude] {
                    let pointer = UnsafeMutablePointer<Float>.allocate(capacity: count + 12)
                    pointer.initialize(repeating: 91,count:count+12)
                    defer { pointer.deinitialize(count:count+12); pointer.deallocate() }
                    let start = 4+offset
                    for i in 0..<count { pointer[start+i] = i == 0 ? -scale : 0 }
                    NormalizeKernels.normalizeUnchecked(pointer+start,dimension:count)
                    #expect((0..<start).allSatisfy { pointer[$0] == 91 })
                    #expect((start+count..<count+12).allSatisfy { pointer[$0] == 91 })
                    if count > 0 { n01AssertFloat(Array(UnsafeBufferPointer(start:pointer+start,count:count)),[-1]+Array(repeating:0,count:count-1)) }
                }
            }
        }
    }

    @Test func checkedFailuresAndPermissiveBits() throws {
        let nan = Float(bitPattern:0x7fc01234)
        for values: [Float] in [[],[0,-0.0],[nan,0],[0,.infinity],[-.infinity,-0.0],[nan,.infinity]] {
            let input = DynamicVector(values)
            let result: Result<DynamicVector,VectorError> = input.normalized()
            guard case .failure(let error) = result else { Issue.record("Expected normalize failure"); continue }
            let description = String(describing:error)
            #expect(description.contains("normalize"))
            #expect(description.contains(values.isEmpty ? "empty" : values.contains { !$0.isFinite } ? "nonfinite" : "zero"))
            #expect(input.toArray().map(\.bitPattern) == values.map(\.bitPattern))
            for provider: any ArraySIMDProvider in [DefaultArraySIMDProvider(),SwiftSIMDProvider(),AccelerateArraySIMDProvider()] {
                #expect(provider.normalize(values).map(\.bitPattern) == values.map(\.bitPattern))
            }
        }
        for invalid in [Float(0),nan,Float.infinity,-Float.infinity] {
            let values = [invalid,-0.0,0,0]
            let v = try Vector<Dim4>(values)
            #expect(v.normalizedFast().toArray().map(\.bitPattern) == values.map(\.bitPattern))
            #expect(v.normalized(tolerance:0).toArray().map(\.bitPattern) == values.map(\.bitPattern))
            var copy = v; copy.normalizeFast()
            #expect(copy.toArray().map(\.bitPattern) == values.map(\.bitPattern))
            var a = Vector384Optimized(repeating:invalid)
            var b = Vector512Optimized(repeating:invalid)
            var c = Vector768Optimized(repeating:invalid)
            var d = Vector1536Optimized(repeating:invalid)
            let beforeA = a.toArray().map(\.bitPattern), beforeB = b.toArray().map(\.bitPattern), beforeC = c.toArray().map(\.bitPattern), beforeD = d.toArray().map(\.bitPattern)
            #expect(throws: VectorError.self) { try NormalizeKernels.normalize384_inplace(&a).get() }
            #expect(throws: VectorError.self) { try NormalizeKernels.normalize512_inplace(&b).get() }
            #expect(throws: VectorError.self) { try NormalizeKernels.normalize768_inplace(&c).get() }
            #expect(throws: VectorError.self) { try NormalizeKernels.normalize1536_inplace(&d).get() }
            #expect(a.toArray().map(\.bitPattern) == beforeA); #expect(b.toArray().map(\.bitPattern) == beforeB)
            #expect(c.toArray().map(\.bitPattern) == beforeC); #expect(d.toArray().map(\.bitPattern) == beforeD)
        }
    }

    @Test func strictToleranceBoundary() throws {
        let tolerance: Float = 0.25
        for value in [Float(1.25).nextDown,1.25,Float(1.25).nextUp] {
            let input = try Vector<Dim4>([value,0,-0.0,0])
            let result = input.normalized(tolerance:tolerance)
            if abs(Double(value)-1) < Double(tolerance) {
                #expect(result.toArray().map(\.bitPattern) == input.toArray().map(\.bitPattern))
            } else { n01AssertFloat(result.toArray(),[1,0,-0.0,0]) }
        }
        let input = try Vector<Dim4>([2,0,0,0])
        #expect(input.normalized(tolerance:Float.greatestFiniteMagnitude).toArray() == input.toArray())
        for tolerance: Float in [0,-0.0] { n01AssertFloat(input.normalized(tolerance:tolerance).toArray(),[1,0,0,0]) }
        let near = try Vector<Dim4>([Float(1).nextUp,0,0,0])
        #expect(near.normalizedFast().toArray().map(\.bitPattern) == near.toArray().map(\.bitPattern))
    }

    @Test func bothBatchRoutesAtSchedulingBoundary() async throws {
        let rows: [[Float]] = [[],[Float.leastNonzeroMagnitude],[Float.greatestFiniteMagnitude,0],[0,-0.0],[Float(bitPattern:0x7fc01234),1],[.infinity,0,1]]
        for count in [0,1,999,1000,1001] {
            let inputs = (0..<count).map { DynamicVector(rows[$0%rows.count]) }
            let operations = try await Operations.normalize(inputs)
            let transform = try await inputs.normalized()
            for output in [operations,transform] {
                #expect(output.count == count)
                for i in 0..<count {
                    let row = rows[i%rows.count]
                    if i%rows.count == 1 { n01AssertFloat(output[i].toArray(),[1]) }
                    else if i%rows.count == 2 { n01AssertFloat(output[i].toArray(),[1,0]) }
                    else { #expect(output[i].toArray().map(\.bitPattern) == row.map(\.bitPattern)) }
                }
            }
        }
    }

    @Test func actualPreparedAndUnpreparedMatrixPropagation() throws {
        let finite = try Vector384Optimized([Float.greatestFiniteMagnitude,0,0,0] + Array(repeating:0,count:380))
        let tiny = try Vector384Optimized([0,Float.leastNonzeroMagnitude,0,0] + Array(repeating:0,count:380))
        let zero = Vector384Optimized()
        let invalid = try Vector384Optimized([.nan,0,0,0] + Array(repeating:0,count:380))
        let infinite = try Vector384Optimized([0,.infinity,0,0] + Array(repeating:0,count:380))
        let queries = [finite,zero,invalid,tiny]
        let candidates = [tiny,infinite,zero,finite]
        let direct = MatrixDistance.cosineDistanceMatrix(queries:queries,candidates:candidates)
        var prepared = [Float](repeating:0,count:16)
        MatrixDistance.cosineDistanceMatrix(queries:queries,prepared:MatrixDistance.prepare(candidates,normalized:true),into:&prepared)
        for result in [direct,prepared] {
            for q in 0..<4 { for c in 0..<4 {
                if q == 2 || c == 1 { #expect(result[q*4+c].isNaN) }
                else { #expect(result[q*4+c] == ((q == 0 && c == 3) || (q == 3 && c == 0) ? 0 : 1)) }
            } }
        }
    }

    private func hints<V: IndexableVector>(_ value: V) throws {
        let checked = try #require(value.normalizedWithHint())
        #expect(checked.isNormalized && checked.cachedMagnitude == 1)
        let unchecked = value.normalizedUncheckedWithHint()
        #expect(unchecked.isNormalized && unchecked.cachedMagnitude == 1)
        let wrapper = NormalizationHint(vector:value,isNormalized:false)
        #expect(try #require(wrapper.normalizedWithHint()).isNormalized)
        #expect(wrapper.normalizedUncheckedWithHint().isNormalized)
    }

    @Test func hintActualWritebackAndNormalizedCaches() throws {
        for scale in [Float.leastNonzeroMagnitude,Float.greatestFiniteMagnitude] {
            try hints(DynamicVector([scale,scale,scale,scale]))
            try hints(Vector<Dim4>(repeating:scale))
            try hints(Vector384Optimized(repeating:scale))
            try hints(Vector512Optimized(repeating:scale))
            try hints(Vector768Optimized(repeating:scale))
            try hints(Vector1536Optimized(repeating:scale))
        }
        let lossy = N01LossyVector([1,1,1,1])
        #expect(lossy.normalizedWithHint() == nil)
        let result = lossy.normalizedUncheckedWithHint()
        #expect(!result.isNormalized && result.cachedMagnitude == nil)
        let wrapper = NormalizationHint(vector:lossy,isNormalized:false)
        #expect(wrapper.normalizedWithHint() == nil)
        #expect(!wrapper.normalizedUncheckedWithHint().isNormalized)
        #expect(wrapper.normalizedUncheckedWithHint().cachedMagnitude == nil)
        let a = CosineBatchHelpers.buildNormalized_512(candidates:[Vector512Optimized(repeating:.leastNonzeroMagnitude),Vector512Optimized(),Vector512Optimized(repeating:.nan)])
        let b = CosineBatchHelpers.buildNormalized_768(candidates:[Vector768Optimized(repeating:.greatestFiniteMagnitude),Vector768Optimized(),Vector768Optimized(repeating:.infinity)])
        let c = CosineBatchHelpers.buildNormalized_1536(candidates:[Vector1536Optimized(repeating:.leastNonzeroMagnitude),Vector1536Optimized(),Vector1536Optimized(repeating:.nan)])
        n01AssertFloat(a[0].toArray(),Array(repeating:Float(1/Double(512).squareRoot()),count:512))
        n01AssertFloat(b[0].toArray(),Array(repeating:Float(1/Double(768).squareRoot()),count:768))
        n01AssertFloat(c[0].toArray(),Array(repeating:Float(1/Double(1536).squareRoot()),count:1536))
        #expect(a[1].isZero && b[1].isZero && c[1].isZero)
        #expect(a[2][0].isNaN && b[2][0].isInfinite && c[2][0].isNaN)
        for invalidQuery in [false,true] {
            var output = [Float](repeating:0,count:3)
            output.withUnsafeMutableBufferPointer { CosineBatchHelpers.batch_cosine_preNormalized_512(queryNormalized:a[invalidQuery ? 2 : 0],candidatesNormalized:a,out:$0) }
            #expect(output[2].isNaN)
            if invalidQuery { #expect(output.allSatisfy { $0.isNaN }) } else { #expect(abs(output[0]) < 1e-5 && output[1] == 1) }
            output.withUnsafeMutableBufferPointer { CosineBatchHelpers.batch_cosine_preNormalized_768(queryNormalized:b[invalidQuery ? 2 : 0],candidatesNormalized:b,out:$0) }
            #expect(output[2].isNaN)
            if invalidQuery { #expect(output.allSatisfy { $0.isNaN }) } else { #expect(abs(output[0]) < 1e-5 && output[1] == 1) }
            output.withUnsafeMutableBufferPointer { CosineBatchHelpers.batch_cosine_preNormalized_1536(queryNormalized:c[invalidQuery ? 2 : 0],candidatesNormalized:c,out:$0) }
            #expect(output[2].isNaN)
            if invalidQuery { #expect(output.allSatisfy { $0.isNaN }) } else { #expect(abs(output[0]) < 1e-5 && output[1] == 1) }
        }
    }

    private func safePolicy<D: StaticDimension>(_ type: D.Type) throws {
        let payload = Float(bitPattern:0x7fc01234)
        let cases: [[Float]] = [[0,-0.0],[payload],[.infinity],[-.infinity],[payload,.infinity,-.infinity],[payload,3,4],[.infinity,1],[-.infinity,1],[.leastNonzeroMagnitude],
            [.greatestFiniteMagnitude], [3,4]]
        for raw in 0..<32 {
            let options = NonFiniteHandling(rawValue:raw)
            for pattern in cases {
                for placement in [0,D.value/2,D.value-pattern.count] where placement+pattern.count <= D.value {
                    var input = [Float](repeating:-0.0,count:D.value)
                    for (j,value) in pattern.enumerated() { input[placement+j] = value }
                    let vector = try Vector<D>(input)
                    let nanIndices = input.indices.filter { input[$0].isNaN }
                    let posIndices = input.indices.filter { input[$0] == .infinity }
                    let negIndices = input.indices.filter { input[$0] == -.infinity }
                    let invalid = !nanIndices.isEmpty || !posIndices.isEmpty || !negIndices.isEmpty
                    if raw & 16 != 0 && invalid {
                        do { _ = try vector.safeNormalized(options:options); Issue.record("Expected strict nonfinite rejection") }
                        catch let error as NonFiniteError {
                            switch error {
                            case .nanDetected(let indices): #expect(indices == nanIndices && posIndices.isEmpty && negIndices.isEmpty)
                            case .infinityDetected(let indices): #expect(indices == posIndices && nanIndices.isEmpty && negIndices.isEmpty)
                            case .negativeInfinityDetected(let indices): #expect(indices == negIndices && nanIndices.isEmpty && posIndices.isEmpty)
                            case .mixedNonFiniteValues(let n,let p,let m): #expect(n == nanIndices && p == posIndices && m == negIndices)
                            }
                        }
                        continue
                    }
                    if raw & 8 != 0 && !nanIndices.isEmpty {
                        #expect(try vector.safeNormalized(options:options).toArray().allSatisfy { $0.isNaN })
                        continue
                    }
                    var handled = input
                    for j in handled.indices {
                        if handled[j].isNaN && raw & 1 != 0 { handled[j] = 0 }
                        else if handled[j] == .infinity && raw & 2 != 0 { handled[j] = .greatestFiniteMagnitude }
                        else if handled[j] == -.infinity && raw & 4 != 0 { handled[j] = -.greatestFiniteMagnitude }
                    }
                    if handled.contains(where:{ !$0.isFinite }) {
                        #expect(try vector.safeNormalized(options:options).toArray().map(\.bitPattern) == handled.map(\.bitPattern))
                    } else if handled.allSatisfy({ $0 == 0 }) {
                        if raw & 16 != 0 {
                            do { _ = try vector.safeNormalized(options:options); Issue.record("Expected strict zero rejection") }
                            catch let error as VectorError { #expect(String(describing:error).contains("normalization")) }
                        } else if raw & 8 != 0 { #expect(try vector.safeNormalized(options:options).toArray().allSatisfy { $0.isNaN }) }
                        else { #expect(try vector.safeNormalized(options:options).toArray().map(\.bitPattern) == handled.map(\.bitPattern)) }
                    } else {
                        // Independent Float reference: exact widened products, sorted sum;
                        // fixture has at most three nonzero terms, not the production Kahan path.
                        let norm = handled.map { Double($0)*Double($0) }.sorted().reduce(0,+).squareRoot()
                        let expected = handled.map { Float(Double($0)/norm) }
                        n01AssertFloat(try vector.safeNormalized(options:options).toArray(),expected)
                        if raw == 7 { n01AssertFloat(try vector.safeNormalized().toArray(),expected) }
                    }
                    #expect(vector.toArray().map(\.bitPattern) == input.map(\.bitPattern))
                }
            }
        }
    }

    @Test func safeNormalizedAllIndependentFlags() throws {
        try safePolicy(Dim4.self)
        try safePolicy(Dim512.self)
        let control = try Vector<Dim4>([Float.leastNonzeroMagnitude,0,-Float.leastNonzeroMagnitude,-0.0])
        #expect(try control.safeDivide(by:.leastNonzeroMagnitude).toArray() == [1,0,-1,0])
    }

    @Test func batchFactoryErrorsUnchanged() async {
        for count in [1,1000] {
            let rows = Array(repeating:N01RejectingFactory([.leastNonzeroMagnitude]),count:count)
            do { _ = try await Operations.normalize(rows); Issue.record("Expected factory rejection") }
            catch { #expect(error is N01RejectingFactory.Failure) }
            do { _ = try await rows.normalized(); Issue.record("Expected factory rejection") }
            catch { #expect(error is N01RejectingFactory.Failure) }
        }
    }

    @Test func guardChildProbe() {
        guard let specification = ProcessInfo.processInfo.environment["N01_GUARD_PROBE"] else { return }
        switch specification {
        case "empty": _ = DynamicVector([]).normalizedUnchecked()
        case "zero": _ = DynamicVector([0,-0.0]).normalizedUnchecked()
        case "nan": _ = DynamicVector([1,.nan]).normalizedUnchecked()
        case "infinity": _ = DynamicVector([1,.infinity]).normalizedUnchecked()
        case "double": _ = N01DoubleVector([0]).normalizedUnchecked()
        case "optimized384": _ = Vector384Optimized().normalizedUnchecked()
        case "optimized512": _ = Vector512Optimized(repeating:.nan).normalizedUnchecked()
        case "optimized768": _ = Vector768Optimized(repeating:.infinity).normalizedUnchecked()
        case "optimized1536": _ = Vector1536Optimized().normalizedUnchecked()
        case "pointerZero", "pointerNaN", "pointerInfinity", "negativeDimension":
            var values: [Float] = specification == "pointerNaN" ? [1,.nan] : specification == "pointerInfinity" ? [1,.infinity] : [0,-0.0]
            values.withUnsafeMutableBufferPointer { NormalizeKernels.normalizeUnchecked($0.baseAddress!,dimension:specification == "negativeDimension" ? -1 : 2) }
        case "toleranceNegative": _ = Vector<Dim4>().normalized(tolerance:-1)
        case "toleranceNaN": _ = Vector<Dim4>().normalized(tolerance:.nan)
        case "toleranceInfinity": _ = Vector<Dim4>().normalized(tolerance:.infinity)
        case "toleranceNegativeInfinity": _ = Vector<Dim4>().normalized(tolerance:-.infinity)
        case "divideZero": _ = DynamicVector([]) / Float(0)
        case "divideNegativeZero": _ = DynamicVector([]) / Float(-0.0)
        case "divideAssignZero": var vector = DynamicVector([]); vector /= Float(0)
        case "divideInPlaceZero": var vector = DynamicVector([]); vector.inPlaceDivide(-0.0)
        case "negativeCountSwift": var x: Float = 1; withUnsafeMutablePointer(to:&x) { SwiftFloatSIMDProvider.divideByScalar(UnsafePointer($0),scalar:1,result:$0,count:-1) }
        case "negativeCountAccelerate": var x: Float = 1; withUnsafeMutablePointer(to:&x) { AccelerateFloatProvider.divideByScalar(UnsafePointer($0),scalar:1,result:$0,count:-1) }
        case "negativeCountDefault": var x: Double = 1; withUnsafeMutablePointer(to:&x) { SwiftDoubleSIMDProvider.divideByScalar(UnsafePointer($0),scalar:1,result:$0,count:-1) }
        default: Issue.record("Unknown child specification")
        }
        print("N01_GUARD_ACCEPTED")
    }

    @Test func invalidInputsRejectInChildren() throws {
        if ProcessInfo.processInfo.environment["N01_SKIP_GUARDS"] == "1" { return }
        let cases = ["empty","zero","nan","infinity","double","optimized384","optimized512","optimized768","optimized1536","pointerZero","pointerNaN","pointerInfinity",
            "negativeDimension","toleranceNegative","toleranceNaN","toleranceInfinity","toleranceNegativeInfinity","divideZero","divideNegativeZero","divideAssignZero",
            "divideInPlaceZero","negativeCountSwift","negativeCountAccelerate","negativeCountDefault"]
        for specification in cases {
            var environment = ProcessInfo.processInfo.environment
            environment["N01_GUARD_PROBE"] = specification
            environment["SWIFT_BACKTRACE"] = "enable=no"
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("N01-guard-\(specification)-\(UUID().uuidString).log")
            _ = FileManager.default.createFile(atPath:url.path,contents:nil)
            let handle = try FileHandle(forWritingTo:url)
            let process = try launchChildProbe(.swiftTesting(filter: "NormalizationRangeContractTests/guardChildProbe"),
                                               environment: environment, output: handle, deadline: 15,
                                               cleanup: .terminateThenKill(grace: 1, seconds: 1))
            let timedOut = process.timedOut
            try handle.close()
            let diagnostic = try String(contentsOf:url,encoding:.utf8)
            print("N01_CHILD_CAPTURE \(specification) \(url.path)")
            if timedOut {
                throw NSError(domain: "N01.childTimeout", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "Child deadline exceeded: \(url.path)"])
            }
            let intended = specification.hasPrefix("tolerance") ? "tolerance must be finite and nonnegative" : specification.hasPrefix("divide") ? "Division by zero"
                : specification.hasPrefix("negativeCount") ? "count must be nonnegative" : specification == "negativeDimension" ? "dimension must be nonnegative" : "requires finite nonzero input"
            #expect(process.terminationReason == .uncaughtSignal,"\(specification): \(diagnostic)")
            #expect(diagnostic.contains(intended),"\(specification): \(diagnostic)")
            #expect(!diagnostic.contains("N01_GUARD_ACCEPTED"))
        }
    }

    @Test func halfWideningAndScalarMagnitude() throws {
        #if !(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))
        let count = 65536
        let result = try N01HalfVector(Array(repeating:Float16(1),count:count)).normalized().get()
        #expect(result.toArray().allSatisfy { $0 == Float16(1.0/256.0) })
        #endif
        for scale in [Float.leastNonzeroMagnitude,Float.leastNormalMagnitude] {
            var values = [Float](repeating:0,count:384)
            values[0] = scale
            let vector = try Vector384Optimized(values)
            #expect(vector.magnitude.bitPattern == scale.bitPattern)
            #expect(vector.magnitudeSquared.bitPattern == Float(Double(scale)*Double(scale)).bitPattern)
        }
        #expect(Vector512Optimized(repeating:.greatestFiniteMagnitude).magnitude.isInfinite)
        #expect(Vector768Optimized(repeating:.nan).magnitude.isNaN)
        #expect(Vector1536Optimized(repeating:.infinity).magnitudeSquared.isInfinite)
        NormalizeKernels.normalizeUnchecked(UnsafeMutablePointer<Float>(bitPattern:1)!,dimension:0)
    }

    @Test func suppliedArrayProviderWitnessIsHonored() async throws {
        let rows = [DynamicVector([Float.leastNonzeroMagnitude,0]),DynamicVector([.nan]),DynamicVector([])]
        let output = try await Operations.$simdProvider.withValue(N01SentinelArrayProvider()) {
            try await Operations.normalize(rows)
        }
        #expect(output.map { $0.toArray() } == [[42,42],[42],[]])
        for provider: any ArraySIMDProvider in [DefaultArraySIMDProvider(),SwiftSIMDProvider(),AccelerateArraySIMDProvider()] {
            let result = try await Operations.$simdProvider.withValue(provider) { try await Operations.normalize([rows[0]]) }
            n01AssertFloat(result[0].toArray(),[1,0])
        }
    }

    @Test func doublePowerOfTwoDirectionAndFloatMixedScales() throws {
        for exponent in [-1072,-1022,-500,0,500,1019] {
            let a = Double(sign:.plus,exponent:exponent,significand:1)
            n01AssertDouble(try N01DoubleVector([3*a,4*a,0,-0.0]).normalized().get().toArray(),[0.6,0.8,0,-0.0])
            n01AssertDouble(try N01DoubleVector([-4*a,0,3*a,-0.0]).normalized().get().toArray(),[-0.8,0,0.6,-0.0])
        }
        let a = Float(sign:.plus,exponent:126,significand:1)
        let input: [Float] = [a,-a/2,.leastNonzeroMagnitude,0]
        let expected: [Float] = [Float(2/Double(5).squareRoot()),Float(-1/Double(5).squareRoot()),0,0]
        try check(DynamicVector(input),expected)
        try check(Vector<Dim4>(input),expected)
        try check(DynamicVector(input.reversed()),expected.reversed())
    }
}
