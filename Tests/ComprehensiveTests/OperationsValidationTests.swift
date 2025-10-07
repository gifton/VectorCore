import Testing
@testable import VectorCore

@Suite("Operations Validation Tests")
struct OperationsValidationTests {

    @Test
    func testFindNearest_kZero_throwsInvalidDimension() async {
        let q = Vector<Dim2>(x: 0, y: 0)
        let db = [Vector<Dim2>(x: 1, y: 1)]
        do {
            _ = try await Operations.findNearest(to: q, in: db, k: 0)
            Issue.record("Expected invalidDimension not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .invalidDimension)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testFindNearest_emptyVectors_throwsInvalidDimension() async {
        let q = Vector<Dim2>(x: 0, y: 0)
        do {
            _ = try await Operations.findNearest(to: q, in: [Vector<Dim2>](), k: 1)
            Issue.record("Expected invalidDimension not thrown for empty vectors")
        } catch let e as VectorError {
            #expect(e.kind == .invalidDimension)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testFindNearest_dimensionMismatch_throws() async {
        let q = DynamicVector([0, 0, 0])
        let db = [DynamicVector([1, 2])] // mismatched dims
        do {
            _ = try await Operations.findNearest(to: q, in: db, k: 1)
            Issue.record("Expected dimensionMismatch not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testFindNearestBatch_emptyQueries_throws() async {
        let db = [Vector<Dim2>(x: 0, y: 0)]
        do {
            _ = try await Operations.findNearestBatch(queries: [Vector<Dim2>](), in: db, k: 1)
            Issue.record("Expected invalidDimension not thrown for empty queries")
        } catch let e as VectorError {
            #expect(e.kind == .invalidDimension)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testFindNearestBatch_inconsistentQueryDims_throws() async {
        let queries: [DynamicVector] = [DynamicVector([0, 0, 0]), DynamicVector([0, 0])] // inconsistent
        let db = [DynamicVector([1, 2, 3])]
        do {
            _ = try await Operations.findNearestBatch(queries: queries, in: db, k: 1)
            Issue.record("Expected dimensionMismatch for inconsistent queries")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testDistanceMatrix_mismatchedSetDims_throws() async {
        // Use DynamicVector so the generic type matches while dimensions differ at runtime
        let setA = [DynamicVector([0, 0])]
        let setB = [DynamicVector([0, 0, 0])]
        do {
            _ = try await Operations.distanceMatrix(between: setA, and: setB)
            Issue.record("Expected dimensionMismatch not thrown for mismatched sets")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test
    func testDistanceMatrix_emptyShapes() async throws {
        let nonEmpty = [Vector<Dim2>(x: 0, y: 0), Vector<Dim2>(x: 1, y: 1)]
        let empty: [Vector<Dim2>] = []
        let m1 = try await Operations.distanceMatrix(between: empty, and: nonEmpty)
        #expect(m1.count == 0)
        let m2 = try await Operations.distanceMatrix(between: nonEmpty, and: empty)
        #expect(m2.count == nonEmpty.count)
        #expect(m2.allSatisfy { $0.isEmpty })
    }

    // Note: weightedCentroid test removed as the API is now internal
    // The functionality is tested via internal tests if needed
}
