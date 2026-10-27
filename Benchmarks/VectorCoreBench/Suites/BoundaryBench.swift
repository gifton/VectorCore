import Foundation
import VectorCore
import VectorCoreBenchmarking

/// Opt-in measurement-boundary companions (`--suites boundary`). Never part of the default suites and never paired.
///
/// For squared Euclidean distance over one immutable FP32 query/candidate fixture, each route is measured twice:
/// - `*-reused`: reused-input scope. Route preparation (SoA transposition, FP16 conversion) and the output buffer are
///   built before timing; the timed body is the current kernel plus a fold of every produced score.
/// - `*-setup`: setup-inclusive scope. Every invocation performs the route's preparation, creates its output and
///   consumes it.
/// `consumer-only` entries time only the fold on a precomputed output. SoA-FP16 has no squared-Euclidean route, so its
/// entries are reported unavailable with a reason and are never invoked or substituted.
/// The kernels are the existing ones, called through stored closures (`BoundaryRoutes`); the closure indirection is
/// part of every companion's timed body and is the same for the reused and setup variants.
struct BoundaryBench: BenchmarkSuite {
    static let name = "boundary"

    enum Variant: String, CaseIterable {
        case aosFP32Reused = "aos-fp32-reused", aosFP32Setup = "aos-fp32-setup"
        case soaFP32Reused = "soa-fp32-reused", soaFP32Setup = "soa-fp32-setup"
        case aosFP16Reused = "aos-fp16-reused", aosFP16Setup = "aos-fp16-setup"
        case soaFP16Reused = "soa-fp16-reused", soaFP16Setup = "soa-fp16-setup"
        case consumerOnly = "consumer-only"
    }

    static func caseName(_ variant: Variant, dim: Int, n: Int) -> String { "batch.euclidean2.\(dim).N\(n).\(variant.rawValue)" }
    static func matrixConsumerName(dim: Int, n: Int) -> String { "matrix.euclidean2.\(dim).N\(n).consumer-only" }

    static func catalog(options: ValidatedOptions, progress: BenchmarkProgressReporter) -> [BenchmarkCase] {
        var cases: [BenchmarkCase] = []
        for dim in options.dims {
            for n in options.batchNs {
                switch dim {
                case 512: cases += entries(BoundaryKernelSets.d512, options: options, n: n)
                case 768: cases += entries(BoundaryKernelSets.d768, options: options, n: n)
                case 1536: cases += entries(BoundaryKernelSets.d1536, options: options, n: n)
                default: break
                }
            }
        }
        return cases
    }

    /// Time one shared body: the warmup and the measurement run the same closure, so they cannot diverge.
    static func runMeasured(label: String, options: ValidatedOptions, units: Int, body: () -> Void) throws -> BenchResult {
        try Harness.warmup(body)
        return try Harness.measure(name: label, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                                   unitCount: units, samples: options.samples, body)
    }

    /// Correctness against an independent Double reference computed outside timing (finite-aware comparison).
    static func validated<Vector>(_ measured: BenchResult, outputs: [Float], fixture: BatchFixtures.Fixture<Vector>,
                                  reference: (Vector, Vector) -> Double) -> BenchResult {
        let references = fixture.candidates.map { reference(fixture.query, $0) }
        return measured.withCorrectness(Correctness.vector(currents: outputs, references: references))
    }

    /// Sampled-pair correctness for the precomputed matrix (all pairs for N <= 8), as the `matrix` suite does.
    static func validatedMatrix<Vector: UnifiedVectorBuffer>(_ measured: BenchResult, output: [Float], vectors: [Vector]) -> BenchResult {
        var currents: [Float] = [], references: [Double] = []
        for (i, j) in MatrixDistanceBench.sampledPairs(count: vectors.count) {
            currents.append(output[i * vectors.count + j])
            references.append(vectors[i].withUnsafeContiguousBytes { aRaw in
                vectors[j].withUnsafeContiguousBytes { bRaw in
                    let a = aRaw.bindMemory(to: Float.self), b = bRaw.bindMemory(to: Float.self)
                    var sum = 0.0
                    for k in a.indices { let difference = Double(a[k]) - Double(b[k]); sum += difference * difference }
                    return sum
                }
            })
        }
        return measured.withCorrectness(Correctness.vector(currents: currents, references: references))
    }

    static func entries<Vector: UnifiedVectorBuffer, SoAStorage, Half>(_ r: BoundaryRoutes<Vector, SoAStorage, Half>,
                                                                       options: ValidatedOptions, n: Int) -> [BenchmarkCase] {
        let dim = r.dim, seed = options.runSeed
        var cases: [BenchmarkCase] = []
        cases.append(BenchmarkCase(caseName(.aosFP32Reused, dim: dim, n: n),
                                   boundary: Boundaries.record(caseName(.aosFP32Reused, dim: dim, n: n), options)) {
            let f = try BatchFixtures.makeFixture(dim: dim, count: n, runSeed: seed, vector: r.makeVector)
            nonisolated(unsafe) let out = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
            defer { out.deallocate() }
            out.initialize(repeating: 0)
            let body = {
                r.aos(f.query, f.candidates, 0..<n, out)
                blackHole(OutputFold.fold(UnsafeBufferPointer(out)))
            }
            let measured = try runMeasured(label: caseName(.aosFP32Reused, dim: dim, n: n), options: options, units: n, body: body)
            return [validated(measured, outputs: Array(out), fixture: f, reference: r.reference)]
        })
        cases.append(BenchmarkCase(caseName(.aosFP32Setup, dim: dim, n: n),
                                   boundary: Boundaries.record(caseName(.aosFP32Setup, dim: dim, n: n), options)) {
            let f = try BatchFixtures.makeFixture(dim: dim, count: n, runSeed: seed, vector: r.makeVector)
            let body = {
                var scores = [Float](repeating: 0, count: n)
                scores.withUnsafeMutableBufferPointer { r.aos(f.query, f.candidates, 0..<n, $0) }
                blackHole(scores.withUnsafeBufferPointer { OutputFold.fold($0) })
            }
            let measured = try runMeasured(label: caseName(.aosFP32Setup, dim: dim, n: n), options: options, units: n, body: body)
            var outputs = [Float](repeating: 0, count: n)
            outputs.withUnsafeMutableBufferPointer { r.aos(f.query, f.candidates, 0..<n, $0) }
            return [validated(measured, outputs: outputs, fixture: f, reference: r.reference)]
        })
        cases.append(BenchmarkCase(caseName(.soaFP32Reused, dim: dim, n: n),
                                   boundary: Boundaries.record(caseName(.soaFP32Reused, dim: dim, n: n), options)) {
            let f = try BatchFixtures.makeFixture(dim: dim, count: n, runSeed: seed, vector: r.makeVector)
            let soa = r.buildSoA(f.candidates)
            nonisolated(unsafe) let out = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
            defer { out.deallocate() }
            out.initialize(repeating: 0)
            let body = {
                r.soa(f.query, soa, out)
                blackHole(OutputFold.fold(UnsafeBufferPointer(out)))
            }
            let measured = try runMeasured(label: caseName(.soaFP32Reused, dim: dim, n: n), options: options, units: n, body: body)
            return [validated(measured, outputs: Array(out), fixture: f, reference: r.reference)]
        })
        cases.append(BenchmarkCase(caseName(.soaFP32Setup, dim: dim, n: n),
                                   boundary: Boundaries.record(caseName(.soaFP32Setup, dim: dim, n: n), options)) {
            let f = try BatchFixtures.makeFixture(dim: dim, count: n, runSeed: seed, vector: r.makeVector)
            let body = {
                let scores = r.soaAPI(f.query, f.candidates)
                blackHole(scores.withUnsafeBufferPointer { OutputFold.fold($0) })
            }
            let measured = try runMeasured(label: caseName(.soaFP32Setup, dim: dim, n: n), options: options, units: n, body: body)
            return [validated(measured, outputs: r.soaAPI(f.query, f.candidates), fixture: f, reference: r.reference)]
        })
        cases.append(BenchmarkCase(caseName(.aosFP16Reused, dim: dim, n: n),
                                   boundary: Boundaries.record(caseName(.aosFP16Reused, dim: dim, n: n), options)) {
            let f = try BatchFixtures.makeFixture(dim: dim, count: n, runSeed: seed, vector: r.makeVector)
            let half = r.toFP16(f.candidates)
            nonisolated(unsafe) let out = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
            defer { out.deallocate() }
            out.initialize(repeating: 0)
            let body = {
                r.fp16(f.query, half, 0..<n, out)
                blackHole(OutputFold.fold(UnsafeBufferPointer(out)))
            }
            let measured = try runMeasured(label: caseName(.aosFP16Reused, dim: dim, n: n), options: options, units: n, body: body)
            return [validated(measured, outputs: Array(out), fixture: f, reference: r.reference)]
        })
        cases.append(BenchmarkCase(caseName(.aosFP16Setup, dim: dim, n: n),
                                   boundary: Boundaries.record(caseName(.aosFP16Setup, dim: dim, n: n), options)) {
            let f = try BatchFixtures.makeFixture(dim: dim, count: n, runSeed: seed, vector: r.makeVector)
            let body = {
                let half = r.toFP16(f.candidates)
                var scores = [Float](repeating: 0, count: n)
                scores.withUnsafeMutableBufferPointer { r.fp16(f.query, half, 0..<n, $0) }
                blackHole(scores.withUnsafeBufferPointer { OutputFold.fold($0) })
            }
            let measured = try runMeasured(label: caseName(.aosFP16Setup, dim: dim, n: n), options: options, units: n, body: body)
            let half = r.toFP16(f.candidates)
            var outputs = [Float](repeating: 0, count: n)
            outputs.withUnsafeMutableBufferPointer { r.fp16(f.query, half, 0..<n, $0) }
            return [validated(measured, outputs: outputs, fixture: f, reference: r.reference)]
        })
        for variant in [Variant.soaFP16Reused, Variant.soaFP16Setup] {
            cases.append(BenchmarkCase(caseName(variant, dim: dim, n: n), boundary: Boundaries.record(caseName(variant, dim: dim, n: n),
                options),
                                       availability: .unavailable(reason: Boundaries.soaFP16UnavailableReason)) { [] })
        }
        cases.append(BenchmarkCase(caseName(.consumerOnly, dim: dim, n: n),
                                   boundary: Boundaries.record(caseName(.consumerOnly, dim: dim, n: n), options)) {
            let f = try BatchFixtures.makeFixture(dim: dim, count: n, runSeed: seed, vector: r.makeVector)
            var scores = [Float](repeating: 0, count: n)
            scores.withUnsafeMutableBufferPointer { r.aos(f.query, f.candidates, 0..<n, $0) }
            let precomputed = scores
            let body = { blackHole(precomputed.withUnsafeBufferPointer { OutputFold.fold($0) }) }
            let measured = try runMeasured(label: caseName(.consumerOnly, dim: dim, n: n), options: options, units: n, body: body)
            return [validated(measured, outputs: precomputed, fixture: f, reference: r.reference)]
        })
        cases.append(BenchmarkCase(matrixConsumerName(dim: dim, n: n),
                                   boundary: Boundaries.record(matrixConsumerName(dim: dim, n: n), options)) {
            let vectors = try (0..<n).map { try r.makeVector(MatrixDistanceBench.inputs(dim: dim, row: $0, runSeed: seed)) }
            let pairs = try BenchmarkArithmetic.product(n, n)
            var precomputed = [Float](repeating: 0, count: pairs)
            MatrixDistance.euclideanSquaredMatrix(queries: vectors, candidates: vectors, into: &precomputed)
            let body = { blackHole(precomputed.withUnsafeBufferPointer { OutputFold.fold($0) }) }
            let measured = try runMeasured(label: matrixConsumerName(dim: dim, n: n), options: options, units: pairs, body: body)
            return [validatedMatrix(measured, output: precomputed, vectors: vectors)]
        })
        return cases
    }
}

/// The existing kernels of one width, as closures. An immutable value with no mutable state, so it may cross the
/// `@Sendable` case boundary (`@unchecked` because the stored closures are not declared `@Sendable`).
struct BoundaryRoutes<Vector, SoAStorage, Half>: @unchecked Sendable {
    let dim: Int
    let makeVector: ([Float]) throws -> Vector
    let aos: (Vector, [Vector], Range<Int>, UnsafeMutableBufferPointer<Float>) -> Void
    let buildSoA: ([Vector]) -> SoAStorage
    let soa: (Vector, SoAStorage, UnsafeMutableBufferPointer<Float>) -> Void
    let soaAPI: (Vector, [Vector]) -> [Float]
    let toFP16: ([Vector]) -> [Half]
    let fp16: (Vector, [Half], Range<Int>, UnsafeMutableBufferPointer<Float>) -> Void
    let reference: (Vector, Vector) -> Double
}

/// The concrete kernel sets, one per width. The kernels are the existing public ones; nothing is re-implemented.
enum BoundaryKernelSets {
    static var d512: BoundaryRoutes<Vector512Optimized, SoA<Vector512Optimized>, MixedPrecisionKernels.Vector512FP16> {
        BoundaryRoutes(
            dim: 512, makeVector: { try Vector512Optimized($0) },
            aos: { BatchKernels.range_euclid2_512(query: $0, candidates: $1, range: $2, out: $3) },
            buildSoA: { SoA<Vector512Optimized>.build(from: $0) },
            soa: { BatchKernels_SoA.euclid2_512(query: $0, soa: $1, out: $2) },
            soaAPI: { BatchKernels_SoA.batchEuclideanSquared512(query: $0, candidates: $1) },
            toFP16: { MixedPrecisionKernels.convertToFP16_512($0) },
            fp16: { MixedPrecisionKernels.range_euclid2_mixed_512(query: $0, candidatesFP16: $1, range: $2, out: $3) },
            reference: { q, c in q.withUnsafeBufferPointer { qb in c.withUnsafeBufferPointer { cb in DoubleRef.euclid2(qb, cb) } } })
    }

    static var d768: BoundaryRoutes<Vector768Optimized, SoA<Vector768Optimized>, MixedPrecisionKernels.Vector768FP16> {
        BoundaryRoutes(
            dim: 768, makeVector: { try Vector768Optimized($0) },
            aos: { BatchKernels.range_euclid2_768(query: $0, candidates: $1, range: $2, out: $3) },
            buildSoA: { SoA<Vector768Optimized>.build(from: $0) },
            soa: { BatchKernels_SoA.euclid2_768(query: $0, soa: $1, out: $2) },
            soaAPI: { BatchKernels_SoA.batchEuclideanSquared768(query: $0, candidates: $1) },
            toFP16: { MixedPrecisionKernels.convertToFP16_768($0) },
            fp16: { MixedPrecisionKernels.range_euclid2_mixed_768(query: $0, candidatesFP16: $1, range: $2, out: $3) },
            reference: { q, c in q.withUnsafeBufferPointer { qb in c.withUnsafeBufferPointer { cb in DoubleRef.euclid2(qb, cb) } } })
    }

    static var d1536: BoundaryRoutes<Vector1536Optimized, SoA<Vector1536Optimized>, MixedPrecisionKernels.Vector1536FP16> {
        BoundaryRoutes(
            dim: 1536, makeVector: { try Vector1536Optimized($0) },
            aos: { BatchKernels.range_euclid2_1536(query: $0, candidates: $1, range: $2, out: $3) },
            buildSoA: { SoA<Vector1536Optimized>.build(from: $0) },
            soa: { BatchKernels_SoA.euclid2_1536(query: $0, soa: $1, out: $2) },
            soaAPI: { BatchKernels_SoA.batchEuclideanSquared1536(query: $0, candidates: $1) },
            toFP16: { MixedPrecisionKernels.convertToFP16_1536($0) },
            fp16: { MixedPrecisionKernels.range_euclid2_mixed_1536(query: $0, candidatesFP16: $1, range: $2, out: $3) },
            reference: { q, c in q.withUnsafeBufferPointer { qb in c.withUnsafeBufferPointer { cb in DoubleRef.euclid2(qb, cb) } } })
    }
}
