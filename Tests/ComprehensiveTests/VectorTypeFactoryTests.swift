import Testing
@testable import VectorCore

@Suite("Vector Type Factory")
struct VectorTypeFactorySuite {
    // Creation with explicit Dimension type
    @Test
    func testCreate_WithDimensionType_Succeeds() {
        let values = Array(repeating: Float(0.5), count: Dim128.value)
        let v = try! VectorTypeFactory.create(Dim128.self, from: values)
        #expect(v.scalarCount == Dim128.value)
        v.withUnsafeBufferPointer { buf in
            #expect(buf.count == Dim128.value)
            #expect(approxEqual(buf[0], 0.5) && approxEqual(buf[127], 0.5))
        }
    }

    @Test
    func testCreate_WithDimensionType_DimensionMismatchThrows() {
        let values = Array(repeating: Float(1.0), count: 127)
        do {
            _ = try VectorTypeFactory.create(Dim128.self, from: values)
            Issue.record("Expected dimensionMismatch not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    // vector(of:from:)
    @Test
    func testVectorOf_ReturnsOptimizedTypesForSupportedDims() {
        let supported = [128, 256, 512, 768, 1536, 3072]
        for d in supported {
            let vals = Array(repeating: Float(0.25), count: d)
            let anyVec = try! VectorTypeFactory.vector(of: d, from: vals)
            #expect(anyVec.scalarCount == d)
            switch d {
            case 128: #expect((anyVec as? Vector<Dim128>) != nil)
            case 256: #expect((anyVec as? Vector<Dim256>) != nil)
            case 512: #expect((anyVec as? Vector<Dim512>) != nil)
            case 768: #expect((anyVec as? Vector<Dim768>) != nil)
            case 1536: #expect((anyVec as? Vector<Dim1536>) != nil)
            case 3072: #expect((anyVec as? Vector<Dim3072>) != nil)
            default: Issue.record("Unexpected supported dim: \(d)")
            }
        }
    }

    @Test
    func testVectorOf_UnsupportedDimensionReturnsDynamicVector() {
        let d = 384
        let vals = Array(repeating: Float(0.5), count: d)
        let anyVec = try! VectorTypeFactory.vector(of: d, from: vals)
        #expect((anyVec as? DynamicVector) != nil)
        #expect(anyVec.scalarCount == d)
    }

    @Test
    func testVectorOf_DimensionMismatchThrows() {
        let d = 256
        let vals = Array(repeating: Float(1.0), count: d - 1)
        do {
            _ = try VectorTypeFactory.vector(of: d, from: vals)
            Issue.record("Expected dimensionMismatch not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    // random / zeros / ones / pattern
    @Test
    func testRandom_InRangeAndCorrectDimension() {
        let d = 33 // unsupported to exercise DynamicVector path
        let anyVec = VectorTypeFactory.random(dimension: d, range: -0.25...0.25)
        #expect(anyVec.scalarCount == d)
        let arr = anyVec.toArray()
        for i in 0..<d { #expect(arr[i] >= -0.25 && arr[i] <= 0.25) }
    }

    @Test
    func testZeros_ReturnsZeroVectorOfDimension() {
        let d = 1536 // supported path
        let anyVec = VectorTypeFactory.zeros(dimension: d)
        #expect(anyVec.scalarCount == d)
        let arr = anyVec.toArray()
        for i in stride(from: 0, to: d, by: max(1, d/17)) { #expect(approxEqual(arr[i], 0)) }
        // Spot-check ends
        #expect(approxEqual(arr[0], 0) && approxEqual(arr[d-1], 0))
    }

    @Test
    func testOnes_ReturnsOnesVectorOfDimension() {
        let d = 512 // supported path
        let anyVec = VectorTypeFactory.ones(dimension: d)
        #expect(anyVec.scalarCount == d)
        let arr = anyVec.toArray()
        for i in stride(from: 0, to: d, by: max(1, d/17)) { #expect(approxEqual(arr[i], 1)) }
        #expect(approxEqual(arr[0], 1) && approxEqual(arr[d-1], 1))
    }

    @Test
    func testWithPattern_GeneratesExpectedValues() {
        let anyVec = VectorTypeFactory.withPattern(dimension: 6) { Float($0) }
        #expect(anyVec.toArray() == [0, 1, 2, 3, 4, 5])
    }

    // optimalDimension / isSupported
    @Test
    func testOptimalDimension_ChoosesNearestSupported() {
        #expect(VectorTypeFactory.optimalDimension(for: 500) == 512)
        #expect(VectorTypeFactory.optimalDimension(for: 700) == 768)
        #expect(VectorTypeFactory.optimalDimension(for: 1400) == 1536)
        #expect(VectorTypeFactory.optimalDimension(for: 2000) == 1536)
        #expect(VectorTypeFactory.optimalDimension(for: 2900) == 3072)
    }

    @Test
    func testIsSupported_ForSupportedAndUnsupportedDims() {
        for d in [128, 256, 512, 768, 1536, 3072] { #expect(VectorTypeFactory.isSupported(dimension: d)) }
        for d in [2, 64, 1024, 2000, 4096] { #expect(!VectorTypeFactory.isSupported(dimension: d)) }
    }

    // basis vectors
    @Test
    func testBasis_ValidIndexProducesOneHot() {
        // Unsupported dim path
        do {
            let d = 10
            let idx = 3
            let anyVec = try VectorTypeFactory.basis(dimension: d, index: idx)
            let arr = anyVec.toArray()
            #expect(arr.count == d)
            var sum: Float = 0
            for i in 0..<d { sum += arr[i] }
            #expect(approxEqual(sum, 1))
            #expect(approxEqual(arr[idx], 1))
        } catch { Issue.record("Unexpected error: \(error)") }

        // Supported dim path
        do {
            let d = 128
            let idx = 5
            let anyVec = try VectorTypeFactory.basis(dimension: d, index: idx)
            #expect((anyVec as? Vector<Dim128>) != nil)
            let arr = anyVec.toArray()
            var sum: Float = 0
            for i in 0..<d { sum += arr[i] }
            #expect(approxEqual(sum, 1))
            #expect(approxEqual(arr[idx], 1))
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testBasis_IndexOutOfBoundsThrows() {
        do {
            _ = try VectorTypeFactory.basis(dimension: 8, index: 8)
            Issue.record("Expected indexOutOfBounds not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .indexOutOfBounds)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    // batch creation
    @Test
    func testBatch_CreatesCorrectCountAndValues() {
        let d = 4
        let values: [Float] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        do {
            let batch = try VectorTypeFactory.batch(dimension: d, from: values)
            #expect(batch.count == 3)
            #expect(batch[0].toArray() == [0, 1, 2, 3])
            #expect(batch[1].toArray() == [4, 5, 6, 7])
            #expect(batch[2].toArray() == [8, 9, 10, 11])
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testBatch_NonMultipleCountThrows() {
        let d = 4
        let bad: [Float] = [0, 1, 2, 3, 4]
        do {
            _ = try VectorTypeFactory.batch(dimension: d, from: bad)
            Issue.record("Expected invalidData not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .invalidData)
        } catch { Issue.record("Unexpected error: \(error)") }
    }
}
