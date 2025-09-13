import Foundation
import Testing
@testable import VectorCore

//@Suite("Dynamic Vector Tests")
struct DynamicVectorTests {

    @Test
    func testResizedIncreasingKeepsPrefixAndFills() {
        let v = DynamicVector([1,2,3].map(Float.init))
        let r = v.resized(to: 6, fill: 9)
        #expect(r.scalarCount == 6)
        #expect(r[0] == 1 && r[1] == 2 && r[2] == 3)
        #expect(r[3] == 9 && r[4] == 9 && r[5] == 9)
    }

    @Test
    func testResizedDecreasingDropsTail() {
        let v = DynamicVector([1,2,3,4,5].map(Float.init))
        let r = v.resized(to: 3)
        #expect(r.scalarCount == 3)
        #expect(r.toArray() == [1,2,3])
    }

    @Test
    func testAppendingAddsElementAndIncrementsDim() {
        let v = DynamicVector([0,1].map(Float.init))
        let r = v.appending(5)
        #expect(r.scalarCount == 3)
        #expect(r.toArray() == [0,1,5])
    }

    @Test
    func testDropLastRemovesAndDecrementsDim() {
        let v = DynamicVector([0,1,2].map(Float.init))
        let r = v.dropLast()
        #expect(r.scalarCount == 2)
        #expect(r.toArray() == [0,1])
    }

    @Test
    func testSliceValidBoundsProducesCorrectSubvector() {
        let v = DynamicVector([0,1,2,3,4].map(Float.init))
        let s = v.slice(from: 1, to: 4)
        #expect(s.scalarCount == 3)
        #expect(s.toArray() == [1,2,3])
    }

    @Test
    func testEqualityAndHashableConsistency() {
        let a = DynamicVector([1,2,3].map(Float.init))
        let b = DynamicVector([1,2,3].map(Float.init))
        let c = DynamicVector([1,2,4].map(Float.init))
        #expect(a == b)
        #expect(a != c)
        var set: Set<DynamicVector> = []
        set.insert(a)
        set.insert(b) // should not add duplicate
        set.insert(c)
        #expect(set.count == 2)
    }

    @Test
    func testDebugDescriptionContainsDimensionAndPreview() {
        let v = DynamicVector([0.1234, 2.0, 3.0, 4.0].map(Float.init))
        let desc = v.debugDescription
        #expect(desc.contains("dim: 4") || desc.contains("dim: 4"))
        #expect(desc.contains("0.1234") || desc.contains("0.123"))
    }

    @Test
    func testRandomGeneratesWithinRange() {
        let v = DynamicVector.random(dimension: 32, in: -0.25...0.25)
        #expect(v.scalarCount == 32)
        for i in 0..<v.scalarCount { #expect(v[i] >= -0.25 && v[i] <= 0.25) }
    }

    @Test
    func testRandomUnitProducesUnitOrZero() {
        let v = DynamicVector.randomUnit(dimension: 16)
        let mag = v.magnitude
        #expect((approxEqual(mag, 1.0, tol: 1e-3)) || approxEqual(mag, 0.0, tol: 1e-6))
    }

    @Test
    func testCollectionIndicesAndIteration() {
        let v = DynamicVector([1,2,3,4].map(Float.init))
        #expect(v.startIndex == 0)
        #expect(v.endIndex == 4)
        var sum: Float = 0
        for x in v { sum += x }
        #expect(sum == 10)
    }

    @Test
    func testWithUnsafeBufferPointerMatchesArray() {
        let arr: [Float] = [0,1,2,3,4]
        let v = DynamicVector(arr)
        v.withUnsafeBufferPointer { buf in
            #expect(buf.count == arr.count)
            for i in 0..<arr.count { #expect(buf[i] == arr[i]) }
        }
    }

    @Test
    func testWithUnsafeMutableBufferPointerMutation() {
        var v = DynamicVector([0,0,0].map(Float.init))
        v.withUnsafeMutableBufferPointer { buf in
            for i in 0..<buf.count { buf[i] = Float(i + 1) }
        }
        #expect(v.toArray() == [1,2,3])
    }

    @Test
    func testZeroAndOnesFactories() {
        let z = DynamicVector.zero(dimension: 5)
        let o = DynamicVector.ones(dimension: 5)
        #expect(z.scalarCount == 5 && o.scalarCount == 5)
        for i in 0..<5 { #expect(z[i] == 0 && o[i] == 1) }
    }

}
