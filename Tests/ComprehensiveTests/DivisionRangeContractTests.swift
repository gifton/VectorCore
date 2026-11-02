import Testing
@testable import VectorCore

@Suite struct DivisionRangeContractTests {
    @Test func leastSubnormalScalarDivision() {
        let s = Float.leastNonzeroMagnitude
        let v = DynamicVector([s, 0, -s]) / s
        #expect(v[0] == 1)
        #expect(v[1].bitPattern == Float(0).bitPattern)
        #expect(v[2] == -1)
        for provider: any ArraySIMDProvider in [DefaultArraySIMDProvider(), SwiftSIMDProvider(), AccelerateArraySIMDProvider()] {
            let result = provider.divide([s, 0, -s], by: s)
            #expect(result == [1, 0, -1])
        }
    }

    @Test func rawBackendsIEEEAndAlias() {
        for count in [0,1,3,4,5,7,8,9,15,16,17,31,32,33,384,512,768,1536] {
            for exponent in -149...127 {
                let divisor = Float(sign: .plus, exponent: exponent, significand: 1)
                let pattern: [Float] = [divisor, -divisor, 0, -0.0, .infinity, -.infinity, .nan]
                let input = (0..<count).map { pattern[$0 % pattern.count] }
                for divide in [N01DefaultFloatProvider.divideByScalar, SwiftFloatSIMDProvider.divideByScalar, AccelerateFloatProvider.divideByScalar, FloatSIMD.divideByScalar,
                    SIMDOperations.FloatProvider.divideByScalar] {
                    var source = [Float(91)] + input + [92]
                    var destination = [Float](repeating: 93, count: count + 2)
                    source.withUnsafeBufferPointer { src in
                        destination.withUnsafeMutableBufferPointer { dst in
                            divide(src.baseAddress! + 1, divisor, dst.baseAddress! + 1, count)
                        }
                    }
                    #expect(destination.first == 93 && destination.last == 93)
                    for i in 0..<count {
                        let expected = Float(Double(input[i]) / Double(divisor))
                        if expected.isNaN { #expect(destination[i+1].isNaN) }
                        else { #expect(destination[i+1].bitPattern == expected.bitPattern) }
                    }
                    source.withUnsafeMutableBufferPointer { buf in
                        divide(UnsafePointer(buf.baseAddress! + 1), divisor, buf.baseAddress! + 1, count)
                    }
                    #expect(source.first == 91 && source.last == 92)
                    for i in 0..<count {
                        if destination[i+1].isNaN { #expect(source[i+1].isNaN) }
                        else { #expect(source[i+1].bitPattern == destination[i+1].bitPattern) }
                    }
                }
            }
        }
    }

    private func protocolDivide<V: VectorProtocol>(_ value: V, _ scalar: V.Scalar) -> V { value / scalar }
    private func factoryDivide<V: VectorFactory>(_ value: V, _ scalar: V.Scalar) -> V { value / scalar }
    private func protocolAssign<V: VectorProtocol>(_ value: V, _ scalar: V.Scalar) -> V { var v = value; v /= scalar; return v }
    private func factoryAssign<V: VectorFactory>(_ value: V, _ scalar: V.Scalar) -> V { var v = value; v /= scalar; return v }
    private func vectorRoutes<V: VectorProtocol>(_ input: V, _ scale: Float) where V.Scalar == Float {
        let expected = input.toArray().map { Float(Double($0) / Double(scale)) }
        #expect(protocolDivide(input, scale).toArray().map(\.bitPattern) == expected.map(\.bitPattern))
        #expect(protocolAssign(input, scale).toArray().map(\.bitPattern) == expected.map(\.bitPattern))
        var copy = input; copy.inPlaceDivide(scale)
        #expect(copy.toArray().map(\.bitPattern) == expected.map(\.bitPattern))
    }

    @Test func genericFactoryAndConcreteOperators() throws {
        for s in [Float.leastNonzeroMagnitude, -Float.leastNonzeroMagnitude, Float.leastNormalMagnitude, Float.greatestFiniteMagnitude] {
            let dynamic = DynamicVector([s, 0, -s, -0.0])
            vectorRoutes(dynamic, s)
            #expect(factoryDivide(dynamic, s).toArray() == [1,0,-1,0])
            #expect(factoryAssign(dynamic, s).toArray() == [1,0,-1,0])
            let fixed = try Vector<Dim4>([s,0,-s,-0.0])
            vectorRoutes(fixed, s)
            #expect(factoryDivide(fixed, s).toArray() == [1,0,-1,0])
            #expect(factoryAssign(fixed, s).toArray() == [1,0,-1,0])
            let a = Vector384Optimized(repeating: s)
            let b = Vector512Optimized(repeating: s)
            let c = Vector768Optimized(repeating: s)
            let d = Vector1536Optimized(repeating: s)
            vectorRoutes(a,s); vectorRoutes(b,s); vectorRoutes(c,s); vectorRoutes(d,s)
            #expect((a/s).toArray().allSatisfy { $0 == 1 })
            #expect((b/s).toArray().allSatisfy { $0 == 1 })
            #expect((c/s).toArray().allSatisfy { $0 == 1 })
            #expect((d/s).toArray().allSatisfy { $0 == 1 })
        }
        for exponent in [-1074,-1073,-1023,-1022,-1000,-512,-1,0,1,512,1022,1023] {
            let s = Double(sign: .plus, exponent: exponent, significand: 1)
            let input = N01DoubleVector([s,0,-s,-0.0])
            for result in [protocolDivide(input,s), factoryDivide(input,s), protocolAssign(input,s),factoryAssign(input,s)] {
                #expect(result.toArray().map(\.bitPattern) == [Double(1),0,-1,-0.0].map(\.bitPattern))
            }
            var copy = input; copy.inPlaceDivide(s)
            #expect(copy.toArray().map(\.bitPattern) == [Double(1),0,-1,-0.0].map(\.bitPattern))
        }
    }

    @Test func doubleDefaultAccelerateAndForwarding() {
        for count in [0,1,3,4,5,7,8,9,15,16,17,31,32,33,384,512,768,1536] {
            for s in [Double.leastNonzeroMagnitude, -Double.leastNonzeroMagnitude, Double.leastNormalMagnitude, Double.greatestFiniteMagnitude, 0, -0.0, .infinity, -.infinity, .nan] {
                let pattern: [Double] = [s, -s, 0, -0.0, 1, -1, .infinity, -.infinity, .nan]
                let input = (0..<count).map { pattern[$0 % pattern.count] }
                for divide in [SwiftDoubleSIMDProvider.divideByScalar, AccelerateDoubleProvider.divideByScalar, DoubleSIMD.divideByScalar] {
                    for alias in [false,true] {
                        var source = [Double(91)] + input + [92]
                        var destination = [Double](repeating: 93, count: count + 2)
                        if alias {
                            source.withUnsafeMutableBufferPointer { p in divide(UnsafePointer(p.baseAddress!+1),s,p.baseAddress!+1,count) }
                            #expect(source.first == 91 && source.last == 92)
                            destination = source
                        } else {
                            source.withUnsafeBufferPointer { src in destination.withUnsafeMutableBufferPointer { dst in divide(src.baseAddress!+1,s,dst.baseAddress!+1,count) } }
                            #expect(destination.first == 93 && destination.last == 93)
                        }
                        for i in 0..<count {
                            let expected = input[i] / s
                            if expected.isNaN { #expect(destination[i+1].isNaN) }
                            else { #expect(destination[i+1].bitPattern == expected.bitPattern) }
                        }
                    }
                }
            }
        }
    }

    @Test func floatProviderSpecialDivisorsAndAdapters() {
        for divisor: Float in [0,-0.0,.infinity,-.infinity,.nan,Float.leastNonzeroMagnitude] {
            let input: [Float] = [0,-0.0,1,-1,.infinity,-.infinity,.nan]
            let expected = input.map { $0/divisor }
            for provider: any ArraySIMDProvider in [DefaultArraySIMDProvider(),SwiftSIMDProvider(),AccelerateArraySIMDProvider()] {
                let result = provider.divide(input,by: divisor)
                for i in input.indices {
                    if expected[i].isNaN { #expect(result[i].isNaN) }
                    else { #expect(result[i].bitPattern == expected[i].bitPattern) }
                }
                #expect(provider.divide([],by: divisor).isEmpty)
            }
            for divide in [N01DefaultFloatProvider.divideByScalar,SwiftFloatSIMDProvider.divideByScalar,AccelerateFloatProvider.divideByScalar,FloatSIMD.divideByScalar] {
                var result = input
                result.withUnsafeMutableBufferPointer { p in divide(UnsafePointer(p.baseAddress!),divisor,p.baseAddress!,p.count) }
                for i in input.indices {
                    if expected[i].isNaN { #expect(result[i].isNaN) }
                    else { #expect(result[i].bitPattern == expected[i].bitPattern) }
                }
            }
        }
    }

    @Test func elementwiseDivisionControls() throws {
        for scale in [Float.leastNonzeroMagnitude,Float.leastNormalMagnitude,1,Float.greatestFiniteMagnitude] {
            let lhs = DynamicVector([scale,-scale,0,-0.0])
            let rhs = DynamicVector([scale,scale,scale,scale])
            #expect((lhs ./ rhs).toArray().map(\.bitPattern) == [Float(1),-1,0,-0.0].map(\.bitPattern))
            for (label,divide) in [("Swift",SwiftFloatSIMDProvider.divide),("Accelerate",AccelerateFloatProvider.divide)] {
                var result = [Float](repeating:0,count:4)
                lhs.toArray().withUnsafeBufferPointer { a in rhs.toArray().withUnsafeBufferPointer { b in result.withUnsafeMutableBufferPointer { c in divide(a.baseAddress!,
                    b.baseAddress!,c.baseAddress!,4) } } }
                #expect(result.map(\.bitPattern) == [Float(1),-1,0,-0.0].map(\.bitPattern),"\(label) scale \(scale.bitPattern)")
            }
        }
        for scale in [Double.leastNonzeroMagnitude,Double.leastNormalMagnitude,1,Double.greatestFiniteMagnitude] {
            let lhs = [scale,-scale,0,-0.0]
            let rhs = [Double](repeating:scale,count:4)
            for (label,divide) in [("SwiftDouble",SwiftDoubleSIMDProvider.divide),("AccelerateDouble",AccelerateDoubleProvider.divide)] {
                var result = [Double](repeating:0,count:4)
                lhs.withUnsafeBufferPointer { a in rhs.withUnsafeBufferPointer { b in result.withUnsafeMutableBufferPointer { c in divide(a.baseAddress!,b.baseAddress!,c.baseAddress!,4) } } }
                #expect(result.map(\.bitPattern) == [Double(1),-1,0,-0.0].map(\.bitPattern),"\(label) scale \(scale.bitPattern)")
            }
        }
    }

    @Test func binaryAccelerateClassMatrix() {
        func verify<S: BinaryFloatingPoint>(_ pattern: [S], divide: (UnsafePointer<S>,UnsafePointer<S>,UnsafeMutablePointer<S>,Int)->Void) {
            for count in [0,1,3,4,5,7,8,9,15,16,17,31,32,33,384,512,768,1536] {
                for shift in pattern.indices {
                    let numerator = (0..<count).map { pattern[$0%pattern.count] }
                    let denominator = (0..<count).map { pattern[($0+shift)%pattern.count] }
                    for alias in 0..<3 {
                        var a = [S(91)] + numerator + [92]
                        var b = [S(93)] + denominator + [94]
                        var output = [S](repeating:95,count:count+2)
                        if alias == 1 {
                            a.withUnsafeMutableBufferPointer { ap in b.withUnsafeBufferPointer { bp in divide(UnsafePointer(ap.baseAddress!+1),bp.baseAddress!+1,ap.baseAddress!+1,count) } }
                            #expect(a.first == 91 && a.last == 92); output = a
                        } else if alias == 2 {
                            a.withUnsafeBufferPointer { ap in b.withUnsafeMutableBufferPointer { bp in divide(ap.baseAddress!+1,UnsafePointer(bp.baseAddress!+1),bp.baseAddress!+1,count) } }
                            #expect(b.first == 93 && b.last == 94); output = b
                        } else {
                            a.withUnsafeBufferPointer { ap in b.withUnsafeBufferPointer { bp in output.withUnsafeMutableBufferPointer { op in divide(ap.baseAddress!+1,
                                bp.baseAddress!+1,op.baseAddress!+1,count) } } }
                            #expect(output.first == 95 && output.last == 95)
                        }
                        for j in 0..<count {
                            let expected = numerator[j]/denominator[j]
                            if expected.isNaN { #expect(output[j+1].isNaN) }
                            else { #expect(output[j+1] == expected); #expect(output[j+1].sign == expected.sign) }
                        }
                    }
                }
            }
        }
        let floats: [Float] = [.leastNonzeroMagnitude,-.leastNonzeroMagnitude,.leastNormalMagnitude,.greatestFiniteMagnitude,0,-0.0,1,-1,.infinity,-.infinity,.nan]
        let doubles: [Double] = [.leastNonzeroMagnitude,-.leastNonzeroMagnitude,.leastNormalMagnitude,.greatestFiniteMagnitude,0,-0.0,1,-1,.infinity,-.infinity,.nan]
        verify(floats,divide:AccelerateFloatProvider.divide)
        verify(doubles,divide:AccelerateDoubleProvider.divide)
        for shift in floats.indices {
            let denominator = floats.indices.map { floats[($0+shift)%floats.count] }
            for provider: any ArraySIMDProvider in [DefaultArraySIMDProvider(),SwiftSIMDProvider(),AccelerateArraySIMDProvider()] {
                let output = provider.elementWiseDivide(floats,denominator)
                for j in floats.indices {
                    let expected = floats[j]/denominator[j]
                    if expected.isNaN { #expect(output[j].isNaN) }
                    else { #expect(output[j].bitPattern == expected.bitPattern) }
                }
            }
        }
    }

    @Test func independentDoubleQuotientFixtures() {
        for (inputBits,divisorBits,expectedBits) in n01DoubleDivisionFixtures {
            let input = inputBits.map(Double.init(bitPattern:))
            let divisor = Double(bitPattern:divisorBits)
            let vector = N01DoubleVector(input)
            for result in [protocolDivide(vector,divisor),factoryDivide(vector,divisor),protocolAssign(vector,divisor),factoryAssign(vector,divisor)] {
                #expect(result.toArray().map(\.bitPattern) == expectedBits)
            }
            for divide in [SwiftDoubleSIMDProvider.divideByScalar,AccelerateDoubleProvider.divideByScalar,DoubleSIMD.divideByScalar] {
                var output = input
                output.withUnsafeMutableBufferPointer { p in divide(UnsafePointer(p.baseAddress!),divisor,p.baseAddress!,p.count) }
                #expect(output.map(\.bitPattern) == expectedBits)
            }
        }
    }
}
