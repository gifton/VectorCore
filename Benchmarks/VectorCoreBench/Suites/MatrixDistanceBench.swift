import Foundation
import VectorCore
import VectorCoreBenchmarking

/// Real pairwise matrix route; unitCount counts N*N produced entries.
/// Each invocation is a public API-call measurement: `MatrixDistance.euclideanSquaredMatrix` packs rows and
/// norms internally on every call, so this is not a prepacked GEMM kernel timing. The caller-owned N*N output
/// is reused and every produced score is folded by `OutputFold` after each invocation.
/// Correctness checks all entries for N<=8 and a bounded deterministic sample otherwise.
struct MatrixDistanceBench: BenchmarkSuite {
    static let name = "matrix"
    static func catalog(options: ValidatedOptions, progress: BenchmarkProgressReporter) -> [BenchmarkCase] {
        var cases: [BenchmarkCase] = []
        if options.dims.contains(512) {
            for n in options.batchNs {
                let label = "matrix.euclidean2.512.N\(n).optimized"
                cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let vectors = try (0..<n).map { try Vector512Optimized(inputs(dim: 512, row: $0, runSeed: options.runSeed)) }
                    return [try measure(vectors, name: label, options: options)]
                })
            }
        }
        if options.dims.contains(768) {
            for n in options.batchNs {
                let label = "matrix.euclidean2.768.N\(n).optimized"
                cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let vectors = try (0..<n).map { try Vector768Optimized(inputs(dim: 768, row: $0, runSeed: options.runSeed)) }
                    return [try measure(vectors, name: label, options: options)]
                })
            }
        }
        if options.dims.contains(1536) {
            for n in options.batchNs {
                let label = "matrix.euclidean2.1536.N\(n).optimized"
                cases.append(BenchmarkCase(label, boundary: Boundaries.record(label, options)) {
                    let vectors = try (0..<n).map { try Vector1536Optimized(inputs(dim: 1536, row: $0, runSeed: options.runSeed)) }
                    return [try measure(vectors, name: label, options: options)]
                })
            }
        }
        return cases
    }
    static func inputs(dim: Int, row: Int, runSeed: UInt64) -> [Float] {
        InputFactory.randomArray(count: dim, seed: InputSeeds.matrix(dim: dim, row: row, runSeed: runSeed))
    }
    /// The value the timed body sinks after each invocation: the fold of every produced score.
    static func consumedValue(_ output: [Float]) -> UInt64 {
        output.withUnsafeBufferPointer { OutputFold.fold($0) }
    }
    static func sampledPairs(count: Int) -> [(Int, Int)] {
        if count <= 8 { return (0..<count).flatMap { i in (0..<count).map { (i, $0) } } }
        let rows = [0, 1, count / 2, count - 2, count - 1]
        return rows.flatMap { i in rows.map { (i, $0) } }
    }
    static func measure<V: UnifiedVectorBuffer>(_ vectors: [V], name: String, options: ValidatedOptions, onMeasuredOutput: (([Float]) -> Void)? = nil) throws -> BenchResult {
        let n = vectors.count
        let pairs = try BenchmarkArithmetic.product(n, n)
        var output = [Float](repeating: 0, count: pairs)
        func once() {
            MatrixDistance.euclideanSquaredMatrix(queries: vectors, candidates: vectors, into: &output)
            // Every produced score is consumed (u64-bitfold-sink-v1). The earlier two-diagonal-entry
            // observation could not see any off-diagonal score or the middle entry.
            blackHole(consumedValue(output))
        }
        try Harness.warmup(once)
        let measured = try Harness.measure(name: name, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: pairs, samples: options.samples, once)
        // Test-only observation after timing; normal catalogs leave this nil.
        onMeasuredOutput?(output)
        var currents: [Float] = []
        var references: [Double] = []
        for (i, j) in sampledPairs(count: n) {
            currents.append(output[i * n + j])
            let reference = vectors[i].withUnsafeContiguousBytes { aRaw in
                vectors[j].withUnsafeContiguousBytes { bRaw in
                    let a = aRaw.bindMemory(to: Float.self), b = bRaw.bindMemory(to: Float.self)
                    var sum = 0.0
                    for k in a.indices { let difference = Double(a[k]) - Double(b[k]); sum += difference * difference }
                    return sum
                }
            }
            references.append(reference)
        }
        return BenchResult(name: measured.name, iterations: measured.iterations, totalNanoseconds: measured.totalNanoseconds,
                           unitCount: measured.unitCount, samples: measured.samples, meanNsPerOp: measured.meanNsPerOp,
                           medianNsPerOp: measured.medianNsPerOp, p90NsPerOp: measured.p90NsPerOp,
                           stddevNsPerOp: measured.stddevNsPerOp, rsdPercent: measured.rsdPercent,
                           correctness: Correctness.vector(currents: currents, references: references))
    }
}
