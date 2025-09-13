import Testing
@testable import VectorCore

@Suite("Convenience APIs Throwing Tests")
struct ConvenienceAPIsThrowingTests {

    @Test
    func testVectorBasisThrowing_invalidIndex_throws() {
        do {
            _ = try Vector<Dim4>.basisThrowing(axis: -1)
            Issue.record("Expected indexOutOfBounds not thrown for negative axis")
        } catch let e as VectorError {
            #expect(e.kind == .indexOutOfBounds)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testVectorRepeatingPatternThrowing_empty_throws() {
        do {
            _ = try Vector<Dim4>.repeatingPatternThrowing([])
            Issue.record("Expected invalidData not thrown for empty pattern")
        } catch let e as VectorError {
            #expect(e.kind == .invalidData)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testDynamicVectorBasisThrowing_invalidInputs_throw() {
        do {
            _ = try DynamicVector.basisThrowing(dimension: 4, axis: 4) // out of bounds
            Issue.record("Expected indexOutOfBounds not thrown for axis == dim")
        } catch let e as VectorError {
            #expect(e.kind == .indexOutOfBounds)
        } catch { Issue.record("Unexpected error: \(error)") }
        do {
            _ = try DynamicVector.basisThrowing(dimension: 0, axis: 0)
            Issue.record("Expected invalidDimension not thrown for dimension 0")
        } catch let e as VectorError {
            #expect(e.kind == .invalidDimension)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testDynamicDimensionValidating_throwsOnZero() {
        do {
            _ = try DynamicDimension.make(0)
            Issue.record("Expected invalidDimension not thrown for DynamicDimension.make(0)")
        } catch let e as VectorError {
            #expect(e.kind == .invalidDimension)
        } catch { Issue.record("Unexpected error: \(error)") }
    }
}

