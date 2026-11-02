import Foundation

/// Utilities for converting raw benchmark results to structured cases and writing JSON output
public enum JSONWriter {
    /// Convert raw harness results to structured benchmark cases with derived metrics
    ///
    /// - Parameter results: Array of raw BenchResult from harness
    /// - Returns: Array of BenchCase with computed metrics and parsed parameters
    /// - Throws: `BenchmarkReportError` when a result has no boundary or an invalid one. A report entry without
    ///   measurement identity is an integrity error, never an accidentally compatible default.
    public static func toCases(_ results: [BenchResult]) throws -> [BenchCase] {
        var seen = Set<String>()
        for r in results where !seen.insert(r.name).inserted { throw BenchmarkReportError("Duplicate case ID: \(r.name)") }
        return try results.map { r in
            guard let boundary = r.boundary else { throw BenchmarkReportError("Missing measurement boundary: \(r.name)") }
            if let problem = boundary.validationError() { throw BenchmarkReportError("Invalid measurement boundary (\(problem)): "
                + "\(r.name)") }
            let nsPerOp = Double(r.totalNanoseconds) / Double(max(r.iterations, 1))
            let unitCount = max(r.unitCount, 1)
            let nsPerUnit = nsPerOp / Double(unitCount)
            let throughput = ((Double(r.iterations) * Double(unitCount)) / (Double(r.totalNanoseconds) / 1_000_000_000.0))
            let params = CaseParsing.parse(name: r.name)
            let exec = deriveExecution(from: params)
            // Derive GFLOP/s for dot and dist.dot
            let gflops: Double?
            if params.kind == "dot", let dim = params.dim {
                gflops = (2.0 * Double(dim)) / nsPerOp
            } else if params.kind == "dist", params.metric == "dot", let dim = params.dim {
                gflops = (2.0 * Double(dim)) / nsPerOp
            } else {
                gflops = nil
            }
            return BenchCase(
                name: r.name,
                params: params,
                execution: exec,
                iterations: r.iterations,
                totalNS: r.totalNanoseconds,
                nsPerOp: nsPerOp,
                unitCount: unitCount,
                nsPerUnit: nsPerUnit,
                throughputPerSec: throughput,
                gflops: gflops,
                suspicious: nsPerOp < 5.0,
                samples: r.samples > 1 ? r.samples : nil,
                meanNsPerOp: r.meanNsPerOp,
                medianNsPerOp: r.medianNsPerOp,
                p90NsPerOp: r.p90NsPerOp,
                stddevNsPerOp: r.stddevNsPerOp,
                rsdPercent: r.rsdPercent,
                correctness: r.correctness.map { CorrectnessOut(from: $0) },
                boundary: boundary,
                sampleNsPerInvocation: r.sampleNsPerInvocation.isEmpty ? nil : r.sampleNsPerInvocation,
                sampleIterations: r.sampleIterations.isEmpty ? nil : r.sampleIterations,
                warmupIterations: r.warmupIterations,
                perOperation: PerOperationStatistics(sampleNsPerInvocation: r.sampleNsPerInvocation,
                    unitsPerInvocation: boundary.unitsPerInvocation)
            )
        }
    }

    /// Derive execution information from parsed case parameters. Name-derived display hints only;
    /// the authoritative identity is `BenchCase.boundary`.
    ///
    /// Extracts feature flags like isOptimized, isFused, usesSoA from variant strings
    private static func deriveExecution(from p: CaseParams) -> ExecutionInfo {
        let metric = (p.metric ?? "").lowercased()
        let isOptimized: Bool? = p.variant.map { !$0.lowercased().contains("generic") }
        let isFused: Bool? = p.variant.map { $0.lowercased().contains("fused") }
        let isPreNorm: Bool? = p.variant.map { $0.lowercased().contains("prenorm") }
        let usesSoA: Bool? = p.variant.map { $0.lowercased().contains("soa") }
        let usesFP16: Bool? = p.variant.map { $0.lowercased().contains("fp16") }
        let usesSquared: Bool? = p.kind == "batch" ? (metric == "euclidean2") : nil
        return ExecutionInfo(
            kind: p.kind,
            metric: p.metric,
            dim: p.dim,
            variant: p.variant,
            provider: p.provider,
            n: p.n,
            isOptimized: isOptimized,
            isFused: isFused,
            isPreNormalized: isPreNorm,
            usesSoA: usesSoA,
            usesMixedPrecision: usesFP16,
            usesSquaredDistance: usesSquared
        )
    }

    /// Write complete benchmark run to JSON file
    ///
    /// Automatically creates parent directories if they don't exist
    ///
    /// - Parameters:
    ///   - run: Complete benchmark run with metadata and results
    ///   - path: Output file path
    /// - Throws: File I/O errors or encoding errors
    public static func write(run: BenchRun, to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(run)
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }
}
