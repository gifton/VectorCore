import Foundation
import os
import Testing
import VectorCore

// D02 phase 4 (W4/W5), phase 5 fix 1: the public entries that read the three provider hooks, each checked with a spy
// installed through the hook's task-local binding at the real route, and the separate boundaries that read no hook.
// The rows are the `Operations` documentation's hook table (Provider hooks), one `expectRow` per row: `entry` is the
// row's first cell without backticks, and `calls` is its exact, ordered call list. test_d02_surface_guards.py
// (`test_hook_table_rows_equal_the_route_tests`) requires the table and these `expectRow` calls to be equal, row for row,
// and fails when a hook read appears in a function the table does not cover (`test_hook_reads_have_table_rows`).
// Separate boundaries (no hook read): SIMDOperations.FloatProvider static kernels, direct BatchKernels, and the
// BLAS-backed MatrixDistance entries.

final class D02CallLog: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock(initialState: [String]())
    func note(_ name: String) { lock.withLock { $0.append(name) } }
    var calls: [String] { lock.withLock { $0 } }
}

/// Forwards to a sequential CPU provider and records every scheduler call.
final class D02ComputeSpy: ComputeProvider, @unchecked Sendable {
    let log = D02CallLog()
    let base = CPUComputeProvider.sequential
    var device: ComputeDevice { .cpu }
    var maxConcurrency: Int { 1 }
    var deviceInfo: ComputeDeviceInfo { base.deviceInfo }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        log.note("execute"); return try await base.execute(work)
    }
    func parallelExecute<T: Sendable>(items: Range<Int>, _ work: @Sendable @escaping (Int) async throws -> T) async throws -> [T] {
        log.note("parallelExecute"); return try await base.parallelExecute(items: items, work)
    }
    func parallelForEach(items: Range<Int>, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        log.note("parallelForEach"); try await base.parallelForEach(items: items, body)
    }
    func parallelForEach(items: Range<Int>, minChunk: Int, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        log.note("parallelForEach"); try await base.parallelForEach(items: items, minChunk: minChunk, body)
    }
    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R, _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                     _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        log.note("parallelReduce"); return try await base.parallelReduce(items: items, initial: initial, rangeWork, combine)
    }
    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R, minChunk: Int,
                                     _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                     _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        log.note("parallelReduce")
        return try await base.parallelReduce(items: items, initial: initial, minChunk: minChunk, rangeWork, combine)
    }
}

/// A BatchKernelProvider whose answers cannot come from a CPU route.
final class D02BatchKernelSpy: BatchKernelProvider, @unchecked Sendable {
    let log = D02CallLog()
    let base = CPUComputeProvider.sequential
    var device: ComputeDevice { .gpu(index: 3) }
    var maxConcurrency: Int { 1 }
    var deviceInfo: ComputeDeviceInfo { base.deviceInfo }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T { try await base.execute(work) }
    func parallelExecute<T: Sendable>(items: Range<Int>, _ work: @Sendable @escaping (Int) async throws -> T) async throws -> [T] {
        try await base.parallelExecute(items: items, work)
    }
    func parallelForEach(items: Range<Int>, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        try await base.parallelForEach(items: items, body)
    }
    func parallelForEach(items: Range<Int>, minChunk: Int, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        try await base.parallelForEach(items: items, minChunk: minChunk, body)
    }
    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R, _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                     _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        try await base.parallelReduce(items: items, initial: initial, rangeWork, combine)
    }
    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R, minChunk: Int,
                                     _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                     _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        try await base.parallelReduce(items: items, initial: initial, minChunk: minChunk, rangeWork, combine)
    }
    func batchDistance<V: VectorProtocol>(query: V, candidates: [V], metric: any DistanceMetric) async throws -> [Float]
    where V.Scalar == Float { log.note("batchDistance"); return candidates.map { _ in 9 } }
    func findNearest<V: VectorProtocol>(query: V, candidates: [V], k: Int, metric: any DistanceMetric) async throws
    -> [(index: Int, distance: Float)] where V.Scalar == Float { log.note("findNearest"); return [(index: 5, distance: 0.25)] }
    func findNearestBatch<V: VectorProtocol>(queries: [V], candidates: [V], k: Int, metric: any DistanceMetric) async throws
    -> [[(index: Int, distance: Float)]] where V.Scalar == Float {
        log.note("findNearestBatch"); return queries.map { _ in [(index: 6, distance: 0.5)] }
    }
}

/// Forwards to the default array provider and records every call.
final class D02SIMDSpy: ArraySIMDProvider, @unchecked Sendable {
    let record = D02CallLog()
    let base = DefaultArraySIMDProvider()
    func add(_ a: [Float], _ b: [Float]) -> [Float] { record.note("add"); return base.add(a, b) }
    func subtract(_ a: [Float], _ b: [Float]) -> [Float] { record.note("subtract"); return base.subtract(a, b) }
    func multiply(_ a: [Float], by s: Float) -> [Float] { record.note("multiply"); return base.multiply(a, by: s) }
    func divide(_ a: [Float], by s: Float) -> [Float] { record.note("divide"); return base.divide(a, by: s) }
    func dot(_ a: [Float], _ b: [Float]) -> Float { record.note("dot"); return base.dot(a, b) }
    func sum(_ a: [Float]) -> Float { record.note("sum"); return base.sum(a) }
    func max(_ a: [Float]) -> Float { record.note("max"); return base.max(a) }
    func min(_ a: [Float]) -> Float { record.note("min"); return base.min(a) }
    func mean(_ a: [Float]) -> Float { record.note("mean"); return base.mean(a) }
    func magnitude(_ a: [Float]) -> Float { record.note("magnitude"); return base.magnitude(a) }
    func normalize(_ a: [Float]) -> [Float] { record.note("normalize"); return base.normalize(a) }
    func magnitudeSquared(_ a: [Float]) -> Float { record.note("magnitudeSquared"); return base.magnitudeSquared(a) }
    func euclideanDistanceSquared(_ a: [Float], _ b: [Float]) -> Float {
        record.note("euclideanDistanceSquared"); return base.euclideanDistanceSquared(a, b)
    }
    func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float { record.note("cosineSimilarity"); return base.cosineSimilarity(a, b) }
    func elementWiseMin(_ a: [Float], _ b: [Float]) -> [Float] { record.note("elementWiseMin"); return base.elementWiseMin(a, b) }
    func elementWiseMax(_ a: [Float], _ b: [Float]) -> [Float] { record.note("elementWiseMax"); return base.elementWiseMax(a, b) }
    func elementWiseMultiply(_ a: [Float], _ b: [Float]) -> [Float] {
        record.note("elementWiseMultiply"); return base.elementWiseMultiply(a, b)
    }
    func elementWiseDivide(_ a: [Float], _ b: [Float]) -> [Float] { record.note("elementWiseDivide"); return base.elementWiseDivide(a, b) }
    func abs(_ a: [Float]) -> [Float] { record.note("abs"); return base.abs(a) }
    func sqrt(_ a: [Float]) -> [Float] { record.note("sqrt"); return base.sqrt(a) }
    func log(_ a: [Float]) -> [Float] { record.note("log"); return base.log(a) }
    func clip(_ a: [Float], min: Float, max: Float) -> [Float] { record.note("clip"); return base.clip(a, min: min, max: max) }
    func minIndex(_ a: [Float]) -> Int { record.note("minIndex"); return base.minIndex(a) }
    func maxIndex(_ a: [Float]) -> Int { record.note("maxIndex"); return base.maxIndex(a) }
}

/// Forwards to the pure-Swift provider and records each factorization.
final class D02LinearAlgebraSpy: LinearAlgebraProvider, @unchecked Sendable {
    let log = D02CallLog()
    let base = SwiftLinearAlgebraProvider()
    func qrThin(_ a: [Float], rows: Int, columns: Int) throws -> QRFactorization {
        log.note("qrThin"); return try base.qrThin(a, rows: rows, columns: columns)
    }
    func svdThin(_ a: [Float], rows: Int, columns: Int) throws -> SingularValueDecomposition {
        log.note("svdThin"); return try base.svdThin(a, rows: rows, columns: columns)
    }
    func symmetricEigen(_ a: [Float], dimension: Int, computeEigenvectors: Bool) throws -> SymmetricEigenDecomposition {
        log.note("symmetricEigen"); return try base.symmetricEigen(a, dimension: dimension, computeEigenvectors: computeEigenvectors)
    }
}

private func d02Values(_ n: Int, _ seed: Int) -> [Float] { (0..<n).map { Float(($0 * 7 + seed * 13) % 23) / 11 - 1 } }

/// The spy an `expectRow` installs: `.compute` (a plain scheduler spy) and `.batchKernel` (a `BatchKernelProvider` spy)
/// through `computeProvider`, `.simd` through `simdProvider`, `.linearAlgebra` through `linearAlgebraProvider`.
enum D02Hook { case compute, batchKernel, simd, linearAlgebra }

/// One hook-table row: runs `body` with a fresh spy installed through `hook` and expects EXACTLY `calls`, in order.
func expectRow(_ entry: String, _ hook: D02Hook, _ calls: [String], _ body: () async throws -> Void,
               sourceLocation: SourceLocation = #_sourceLocation) async throws {
    let recorded: [String]
    switch hook {
    case .compute:
        let spy = D02ComputeSpy()
        try await Operations.$computeProvider.withValue(spy) { try await body() }
        recorded = spy.log.calls
    case .batchKernel:
        let spy = D02BatchKernelSpy()
        try await Operations.$computeProvider.withValue(spy) { try await body() }
        recorded = spy.log.calls
    case .simd:
        let spy = D02SIMDSpy()
        try await Operations.$simdProvider.withValue(spy) { try await body() }
        recorded = spy.record.calls
    case .linearAlgebra:
        let spy = D02LinearAlgebraSpy()
        try await Operations.$linearAlgebraProvider.withValue(spy) { try await body() }
        recorded = spy.log.calls
    }
    #expect(recorded == calls, "hook-table row: \(entry)", sourceLocation: sourceLocation)
}

@Suite("D02 provider hook routes")
struct D02ProviderHookRouteTests {
    @Test("computeProvider: every Operations entry that reads it calls the installed provider")
    func computeProviderRoutes() async throws {
        let candidates = (0..<64).map { DynamicVector(d02Values(16, $0)) }
        let query = DynamicVector(d02Values(16, 99))
        // An installed BatchKernelProvider answers findNearest and findNearestBatch itself.
        var single: [NearestNeighborResult] = [], batch: [[NearestNeighborResult]] = []
        try await expectRow("Operations.findNearest, an installed BatchKernelProvider", .batchKernel, ["findNearest"]) {
            single = try await Operations.findNearest(to: query, in: candidates, k: 2)
        }
        try await expectRow("Operations.findNearestBatch, an installed BatchKernelProvider", .batchKernel,
                            ["findNearestBatch"]) {
            batch = try await Operations.findNearestBatch(queries: [query, query], in: candidates, k: 2)
        }
        #expect(single.map(\.index) == [5] && single.map(\.distance) == [0.25])
        #expect(batch.map { $0.map(\.index) } == [[6], [6]])
        // A plain provider schedules the CPU routes.
        try await expectRow("Operations.findNearest, generic route", .compute, ["parallelReduce"]) {
            _ = try await Operations.findNearest(to: query, in: candidates, k: 2)
        }
        let q512 = try Vector512Optimized(d02Values(512, 1))
        let c512 = try (0..<8).map { try Vector512Optimized(d02Values(512, $0 + 2)) }
        try await expectRow("Operations.findNearest, Vector512Optimized Top-K route", .compute, ["parallelReduce"]) {
            _ = try await Operations.findNearest(to: q512, in: c512, k: 2)
        }
        try await expectRow("Operations.findNearestBatch, default .perQuery route, one query", .compute,
                            ["parallelExecute", "parallelReduce"]) {
            _ = try await Operations.findNearestBatch(queries: [query], in: candidates, k: 2)
        }
        try await expectRow("Operations.distanceMatrix", .compute, ["parallelExecute"]) {
            _ = try await Operations.distanceMatrix(between: [query], and: candidates)
        }
        try await expectRow("Operations.normalize", .compute, ["parallelExecute"]) {
            _ = try await Operations.normalize([query])
        }
    }

    @Test("simdProvider: the Operations and Vector entries that read it call the installed provider")
    func simdProviderRoutes() async throws {
        let a = try Vector<Dim64>(d02Values(64, 1)), b = try Vector<Dim64>(d02Values(64, 2))
        let small = try Vector<Dim8>(d02Values(8, 3)), smallB = try Vector<Dim8>(d02Values(8, 6))
        var nanValues = d02Values(64, 7)
        nanValues[3] = .nan
        let withNaN = try Vector<Dim64>(nanValues)
        let dynamic = DynamicVector(d02Values(16, 4)), dynamicB = DynamicVector(d02Values(16, 5))
        try await expectRow("Operations.normalize", .simd, ["normalize"]) { _ = try await Operations.normalize([dynamic]) }
        try await expectRow("Operations.centroid", .simd, ["multiply"]) { _ = try Operations.centroid(of: [dynamic, dynamicB]) }
        try await expectRow("Operations.statistics, two vectors", .simd, ["magnitude", "magnitude"]) {
            _ = try Operations.statistics(for: [dynamic, dynamicB])
        }
        try await expectRow("Vector .*", .simd, ["elementWiseMultiply"]) { _ = a .* b }
        try await expectRow("Vector ./", .simd, ["elementWiseDivide"]) { _ = a ./ b }
        try await expectRow("Vector mean", .simd, ["mean"]) { _ = a.mean }
        try await expectRow("Vector sum", .simd, ["sum"]) { _ = a.sum }
        try await expectRow("Vector softmax()", .simd, ["max", "divide"]) { _ = a.softmax() }
        try await expectRow("Vector clamped(to:)", .simd, ["clip"]) { _ = a.clamped(to: -0.5...0.5) }
        try await expectRow("Vector l1Norm, more than 32 elements", .simd, ["abs"]) { _ = a.l1Norm }
        try await expectRow("Vector l1Norm, 32 elements or fewer", .simd, []) { _ = small.l1Norm }
        try await expectRow("Vector lInfinityNorm", .simd, ["abs"]) { _ = a.lInfinityNorm }
        try await expectRow("Vector lInfinityNorm, an input holding a NaN", .simd, []) { _ = withNaN.lInfinityNorm }
        try await expectRow("Vector manhattanDistance(to:), more than 32 elements", .simd, ["subtract", "abs"]) {
            _ = a.manhattanDistance(to: b)
        }
        try await expectRow("Vector manhattanDistance(to:), 32 elements or fewer", .simd, []) {
            _ = small.manhattanDistance(to: smallB)
        }
        try await expectRow("Vector chebyshevDistance(to:)", .simd, ["abs"]) { _ = a.chebyshevDistance(to: b) }
        try await expectRow("Vector min(_:)", .simd, ["elementWiseMin"]) { _ = a.min(b) }
        try await expectRow("Vector max(_:)", .simd, ["elementWiseMax"]) { _ = a.max(b) }
        try await expectRow("Vector minElement()", .simd, ["minIndex"]) { _ = a.minElement() }
        try await expectRow("Vector maxElement()", .simd, ["maxIndex"]) { _ = a.maxElement() }
        try await expectRow("Vector clamp(to:)", .simd, ["clip"]) { var c = a; c.clamp(to: 0...1) }
        try await expectRow("Vector absoluteValue()", .simd, ["abs"]) { _ = a.absoluteValue() }
        try await expectRow("Vector squareRoot()", .simd, ["sqrt"]) { _ = a.squareRoot() }
        try await expectRow("DynamicVector min(_:)", .simd, ["elementWiseMin"]) { _ = try dynamic.min(dynamicB) }
        try await expectRow("DynamicVector max(_:)", .simd, ["elementWiseMax"]) { _ = try dynamic.max(dynamicB) }
        try await expectRow("DynamicVector clamped(to:)", .simd, ["clip"]) { _ = dynamic.clamped(to: 0...1) }
    }

    @Test("linearAlgebraProvider: PCAModel.fit, Operations.pca and Operations.umap call the installed provider")
    func linearAlgebraProviderRoute() async throws {
        let vectors = (0..<6).map { DynamicVector(d02Values(4, $0)) }
        try await expectRow("PCAModel.fit, exact path", .linearAlgebra, ["svdThin"]) {
            _ = try PCAModel.fit(vectors, components: 2)
        }
        // Sketch path: n = 40, d = 32, k = 2 gives sketch width 2 + min(8, 32 - 2) = 10 < min(n, d)
        // (PCAShape.fit), so the randomized range finder runs: 2 × powerIterations + 1 qrThin (default
        // powerIterations 2), then svdThin of the projected panel.
        var state: UInt32 = 0x0BAD_5EED
        let wide = (0..<40).map { _ in
            DynamicVector((0..<32).map { _ in
                state = state &* 1_664_525 &+ 1_013_904_223
                return Float(Int32(bitPattern: state) >> 8) / Float(1 << 23)
            })
        }
        try await expectRow("PCAModel.fit, sketch path", .linearAlgebra,
                            ["qrThin", "qrThin", "qrThin", "qrThin", "qrThin", "svdThin"]) {
            _ = try PCAModel.fit(wide, components: 2)
        }
        // Public entries that reach the hook through PCAModel.fit (phase 5 fix 1, A-M1).
        try await expectRow("Operations.pca", .linearAlgebra, ["svdThin"]) {
            _ = try Operations.pca(vectors, components: 2)
        }
        try await expectRow("Operations.umap, default .pca initialization", .linearAlgebra, ["svdThin"]) {
            _ = try Operations.umap(vectors, config: UMAPConfig(neighbors: 4, epochs: 5))
        }
    }

    @Test("W5: static Float kernels, direct BatchKernels and the BLAS MatrixDistance entries read no hook")
    func separateBoundariesReadNoHook() throws {
        let compute = D02ComputeSpy(), simd = D02SIMDSpy(), algebra = D02LinearAlgebraSpy()
        let a = d02Values(512, 1), b = d02Values(512, 2)
        let q = try Vector512Optimized(a)
        let candidates = try [Vector512Optimized(b), Vector512Optimized(a)]
        let observed: (Float, [Float], [Float]) = Operations.$computeProvider.withValue(compute) {
            Operations.$simdProvider.withValue(simd) {
                Operations.$linearAlgebraProvider.withValue(algebra) {
                    let dot = a.withUnsafeBufferPointer { ap in
                        b.withUnsafeBufferPointer { bp in SIMDOperations.FloatProvider.dot(ap.baseAddress!, bp.baseAddress!, count: 512) }
                    }
                    var direct = [Float](repeating: .nan, count: 2)   // read inside the closure (swiftlang/swift#92562)
                    let directCopy: [Float] = direct.withUnsafeMutableBufferPointer { out in
                        BatchKernels.range_euclid2_512(query: q, candidates: candidates, range: 0..<2, out: out)
                        return Array(out)
                    }
                    var matrix = [Float](repeating: .nan, count: 2)
                    MatrixDistance.euclideanSquaredMatrix(queries: [q], candidates: candidates, into: &matrix)
                    let matrixCopy: [Float] = matrix.withUnsafeBufferPointer { Array($0) }
                    return (dot, directCopy, matrixCopy)
                }
            }
        }
        #expect(compute.log.calls.isEmpty && simd.record.calls.isEmpty && algebra.log.calls.isEmpty)
        #expect(observed.0.isFinite && observed.1.allSatisfy(\.isFinite) && observed.2.allSatisfy(\.isFinite))
    }
}
