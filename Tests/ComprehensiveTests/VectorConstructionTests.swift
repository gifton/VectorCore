import Testing
@testable import VectorCore

@Suite("Vector Construction")
struct VectorConstructionSuite {
    @Test
    func testGenericVectorInitZeroAndRepeating() throws {
        let v0 = Vector<Dim32>()
        #expect(v0.scalarCount == 32)
        #expect(v0[0] == 0 && v0[31] == 0)
        
        let v1 = Vector<Dim32>(repeating: 1.5)
        #expect(v1.scalarCount == 32)
        #expect(v1[0] == 1.5 && v1[31] == 1.5)
    }
    
    @Test
    func testGenericVectorInitFromArray_Success() throws {
        let arr = Array(repeating: Float(2.5), count: 64)
        let v = try Vector<Dim64>(arr)
        #expect(v.scalarCount == 64)
        #expect(v[0] == 2.5 && v[63] == 2.5)
    }
    
    @Test
    func testGenericVectorInitFromArray_DimensionMismatchThrows() {
        let arr = Array(repeating: Float(1.0), count: 10)
        do {
            _ = try Vector<Dim8>(arr)
            Issue.record("Expected dimension mismatch error not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }
    
    @Test
    func testGenericVectorStaticFactoriesZerosOnes() {
        let z = Vector<Dim16>.zero
        let o = Vector<Dim16>.ones
        #expect(z.scalarCount == 16 && z[0] == 0 && z[15] == 0)
        #expect(o.scalarCount == 16 && o[0] == 1 && o[15] == 1)
    }
    
    @Test
    func testGenericVectorRandomInRange() {
        let v = Vector<Dim8>.random(in: -0.5...0.5)
        #expect(v.scalarCount == 8)
        for i in 0..<8 { #expect(v[i] >= -0.5 && v[i] <= 0.5) }
    }
    
    @Test
    func testGenericVectorRandomUnitProducesUnitOrZero() {
        let v = Vector<Dim32>.randomUnit()
        let mag = v.magnitude
        #expect((approxEqual(mag, 1.0, tol: 1e-3)) || approxEqual(mag, 0.0, tol: 1e-6))
    }
    
    @Test
    func testDynamicVectorInitDimensionRepeating() {
        let v = DynamicVector(dimension: 5, repeating: 3.0)
        #expect(v.scalarCount == 5)
        for i in 0..<5 { #expect(v[i] == 3.0) }
    }
    
    @Test
    func testDynamicVectorInitFromArrayAndSequence() throws {
        let arr: [Float] = [0, 1, 2, 3]
        let v1 = DynamicVector(arr)
        #expect(v1.scalarCount == 4 && v1[2] == 2)
        
        let seq = stride(from: 0, to: 6, by: 1).map { Float($0) }
        let v2 = DynamicVector(seq)
        #expect(v2.scalarCount == 6 && v2[5] == 5)
    }
    
    @Test
    func testDynamicVectorInitWithGenerator() {
        let v = DynamicVector(dimension: 4) { Float($0 * 2) }
        #expect(v.scalarCount == 4)
        #expect(v[0] == 0 && v[1] == 2 && v[2] == 4 && v[3] == 6)
    }
    
    @Test
    func testDynamicVectorZeroOnesRandomFactories() {
        let z = DynamicVector.zero(dimension: 3)
        let o = DynamicVector.ones(dimension: 3)
        let r = DynamicVector.random(dimension: 3, in: -1...1)
        #expect(z[0] == 0 && z[2] == 0)
        #expect(o[0] == 1 && o[2] == 1)
        for i in 0..<3 { #expect(r[i] >= -1 && r[i] <= 1) }
    }
    
    @Test
    func testVectorTypeFactoryVectorOfSupportedDimensions() throws {
        let dims = [128, 256, 512]
        for d in dims {
            let values = Array(repeating: Float(0.25), count: d)
            let anyVec = try VectorTypeFactory.vector(of: d, from: values)
            let array = anyVec.toArray()
            #expect(array.count == d)
            #expect(array.first == 0.25 && array.last == 0.25)
        }
    }
    
    @Test
    func testVectorTypeFactoryVectorOfUnsupportedDimensionReturnsDynamic() throws {
        let d = 384
        let values = Array(repeating: Float(0.5), count: d)
        let anyVec = try VectorTypeFactory.vector(of: d, from: values)
        // Ensure it’s a DynamicVector for unsupported dimension
        #expect((anyVec as? DynamicVector) != nil)
    }
    
    @Test
    func testVectorTypeFactoryVectorOfDimensionMismatchThrows() {
        let d = 256
        let values = Array(repeating: Float(1.0), count: d - 1)
        do {
            _ = try VectorTypeFactory.vector(of: d, from: values)
            Issue.record("Expected dimension mismatch error not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }
    
    @Test
    func testVectorTypeFactoryBatchSuccessAndError() {
        let d = 4
        let values: [Float] = [0,1,2,3,  4,5,6,7] // 2 vectors
        do {
            let batch = try VectorTypeFactory.batch(dimension: d, from: values)
            #expect(batch.count == 2)
            #expect(batch[0].toArray() == [0,1,2,3])
            #expect(batch[1].toArray() == [4,5,6,7])
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
        
        let bad: [Float] = [0,1,2,3,4] // not multiple of 4
        do {
            _ = try VectorTypeFactory.batch(dimension: d, from: bad)
            Issue.record("Expected error for non-multiple count not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .invalidData)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }
    
    @Test
    func testVectorTypeFactoryBasisInBoundsAndOutOfBounds() {
        do {
            let anyVec = try VectorTypeFactory.basis(dimension: 8, index: 3)
            let array = anyVec.toArray()
            #expect(array.count == 8)
            for i in 0..<8 { #expect(array[i] == (i == 3 ? 1.0 : 0.0)) }
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
        
        do {
            _ = try VectorTypeFactory.basis(dimension: 8, index: 8) // OOB
            Issue.record("Expected indexOutOfBounds not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .indexOutOfBounds)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }
    
    @Test
    func testGenericVectorInitSequence_DimensionMismatchThrows() {
        // Sequence with mismatched count for Dim8
        let seq = (0..<10).lazy.map { _ in Float(1) }
        do {
            _ = try Vector<Dim8>(seq)
            Issue.record("Expected dimension mismatch for sequence init not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }
    
    @Test
    func testGenericVectorGeneratorInitializer() throws {
        let v = Vector<Dim16>(generator: { i in Float(i * 2) })
        #expect(v.scalarCount == 16)
        for i in 0..<16 { #expect(v[i] == Float(i * 2)) }
    }
    
    @Test
    func testDim2InitAndComponents_XY() {
        var v = Vector<Dim2>(x: 1, y: 2)
        #expect(v.x == 1 && v.y == 2)
        #expect(v[0] == 1 && v[1] == 2)
        // Mutate via properties
        v.x = 3; v.y = 4
        #expect(v[0] == 3 && v[1] == 4)
    }
    
    @Test
    func testDim3InitAndComponents_XYZ() {
        var v = Vector<Dim3>(x: 1, y: 2, z: 3)
        #expect(v.x == 1 && v.y == 2 && v.z == 3)
        #expect(v[0] == 1 && v[1] == 2 && v[2] == 3)
        // Mutate via properties
        v.x = 4; v.y = 5; v.z = 6
        #expect(v[0] == 4 && v[1] == 5 && v[2] == 6)
    }
    
    @Test
    func testDim4InitAndComponents_XYZW() {
        var v = Vector<Dim4>(x: 1, y: 2, z: 3, w: 4)
        #expect(v.x == 1 && v.y == 2 && v.z == 3 && v.w == 4)
        #expect(v[0] == 1 && v[1] == 2 && v[2] == 3 && v[3] == 4)
        // Mutate via properties
        v.x = 5; v.y = 6; v.z = 7; v.w = 8
        #expect(v[0] == 5 && v[1] == 6 && v[2] == 7 && v[3] == 8)
    }
    
    @Test
    func testVectorTypeFactoryWithPattern() {
        let anyVec = VectorTypeFactory.withPattern(dimension: 6) { Float($0) }
        let arr = anyVec.toArray()
        #expect(arr == [0,1,2,3,4,5])
    }
    
    @Test
    func testDynamicVectorEmptyInitZeroDimension() {
        let v = DynamicVector()
        #expect(v.scalarCount == 0)
        #expect(v.toArray().isEmpty)
    }
    
}
