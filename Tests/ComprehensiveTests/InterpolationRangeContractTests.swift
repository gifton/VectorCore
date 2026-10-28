import Testing
import VectorCore

private enum InterpolationZero: StaticDimension { static let value = 0 }
private enum InterpolationOne: StaticDimension { static let value = 1 }
private enum InterpolationFive: StaticDimension { static let value = 5 }

@Suite struct InterpolationRangeContractTests {
    @Test func oppositeFiniteExtrema() throws {
        let maximum = Float.greatestFiniteMagnitude
        let expected: [Float] = [-maximum, 0, maximum]
        #expect(Vector<Dim3>.linspace(from: -maximum, to: maximum).toArray() == expected)
        #expect(DynamicVector.linspace(dimension: 3, from: -maximum, to: maximum).toArray() == expected)
        let a = Vector<Dim3>(repeating: -maximum), b = Vector<Dim3>(repeating: maximum)
        #expect(a.lerp(to: b, t: 0.5).toArray() == [0, 0, 0])
        #expect(a.lerpUnclamped(to: b, t: 0.5).toArray() == [0, 0, 0])
        #expect(a.smoothstep(to: b, t: 0.5).toArray() == [0, 0, 0])
        #expect(try DynamicVector([-maximum]).lerp(to: DynamicVector([maximum]), t: 0.5).toArray() == [0])
    }

    @Test func convexWeightsAndExtrapolation() throws {
        let maximum = Float.greatestFiniteMagnitude
        for (start, end): (Float, Float) in [(-maximum, maximum), (maximum, -maximum), (-7, 13), (maximum / 2, maximum)] {
            let fixed = Vector<InterpolationFive>.linspace(from: start, to: end).toArray()
            let dynamic = DynamicVector.linspace(dimension: 5, from: start, to: end).toArray()
            for i in 0...4 {
                let t = Float(i) / 4
                // Weighted endpoint sum is independent of the implementation's difference form.
                let expected = Float((1 - Double(t)) * Double(start) + Double(t) * Double(end))
                #expect(fixed[i].isFinite && dynamic[i].isFinite)
                #expect(abs(Double(fixed[i]) - Double(expected)) <= Double(expected.ulp) * 2)
                #expect(abs(Double(dynamic[i]) - Double(expected)) <= Double(expected.ulp) * 2)
                let a = Vector<Dim3>(repeating: start), b = Vector<Dim3>(repeating: end)
                #expect(abs(Double(a.lerp(to: b, t: t)[0]) - Double(expected)) <= Double(expected.ulp) * 2)
            }
        }
        let a = Vector<Dim3>(repeating: 2), b = Vector<Dim3>(repeating: 4)
        #expect(a.lerpUnclamped(to: b, t: -1).toArray() == [0, 0, 0])
        #expect(a.lerpUnclamped(to: b, t: 2).toArray() == [6, 6, 6])
        #expect(a.lerp(to: b, t: -1) == a && a.lerp(to: b, t: 2) == b)
        #expect(Vector<Dim3>(repeating: 0).lerpUnclamped(to: .init(repeating: maximum), t: 2)[0] == .infinity)
    }

    @Test func endpointBitsAndDegenerateWidths() throws {
        for (start, end): (Float, Float) in [(-0.0, 0.0), (0.0, -0.0), (-3, 7)] {
            let fixed = Vector<Dim3>.linspace(from: start, to: end)
            let dynamic = DynamicVector.linspace(dimension: 3, from: start, to: end)
            #expect(fixed[0].bitPattern == start.bitPattern && fixed[2].bitPattern == end.bitPattern)
            #expect(dynamic[0].bitPattern == start.bitPattern && dynamic[2].bitPattern == end.bitPattern)
            #expect(Vector<InterpolationZero>.linspace(from: start, to: end).toArray().isEmpty)
            #expect(Vector<InterpolationOne>.linspace(from: start, to: end)[0].bitPattern == start.bitPattern)
            #expect(DynamicVector.linspace(dimension: 0, from: start, to: end).toArray().isEmpty)
            #expect(DynamicVector.linspace(dimension: 1, from: start, to: end)[0].bitPattern == start.bitPattern)
            let a = Vector<Dim3>(repeating: start), b = Vector<Dim3>(repeating: end)
            #expect(a.lerp(to: b, t: 0)[0].bitPattern == start.bitPattern)
            #expect(a.lerpUnclamped(to: b, t: 1)[0].bitPattern == end.bitPattern)
            #expect(a.smoothstep(to: b, t: 1)[0].bitPattern == end.bitPattern)
            #expect(try DynamicVector([start]).lerp(to: DynamicVector([end]), t: 1)[0].bitPattern == end.bitPattern)
        }
        #expect(throws: VectorError.self) { try DynamicVector([1]).lerp(to: DynamicVector([1, 2]), t: 0) }
    }

    @Test func nonfiniteControlsRetainLegacyClassification() {
        for start: Float in [.infinity, -.infinity, .nan] {
            #expect(Vector<Dim3>.linspace(from: start, to: 1).toArray().allSatisfy { $0.isNaN })
            #expect(DynamicVector.linspace(dimension: 3, from: start, to: 1).toArray().allSatisfy { $0.isNaN })
            #expect(Vector<Dim3>(repeating: start).lerp(to: .init(repeating: 1), t: 0.5)[0].isNaN)
        }
        let positive = Vector<Dim3>.linspace(from: 1, to: .infinity)
        #expect(positive[0].isNaN && positive[1] == .infinity && positive[2] == .infinity)
        let dynamic = DynamicVector.linspace(dimension: 3, from: 1, to: .infinity)
        #expect(dynamic[0].isNaN && dynamic[1] == .infinity && dynamic[2] == .infinity)
    }
}
