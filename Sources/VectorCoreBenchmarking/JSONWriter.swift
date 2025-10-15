import Foundation

/// Utilities for converting raw benchmark results to structured cases and writing JSON output
public enum JSONWriter {
    /// Convert raw harness results to structured benchmark cases with derived metrics
    ///
    /// - Parameter results: Array of raw BenchResult from harness
    /// - Returns: Array of BenchCase with computed metrics and parsed parameters
    public static func toCases(_ results: [BenchResult]) -> [BenchCase] {
        results.map { r in
            let nsPerOp = Double(r.totalNanoseconds) / Double(max(r.iterations, 1))
            let unitCount = max(r.unitCount, 1)
            let nsPerUnit = nsPerOp / Double(unitCount)
            let throughput = (Double(r.iterations * unitCount) / (Double(r.totalNanoseconds) / 1_000_000_000.0))
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
                correctness: r.correctness.map { CorrectnessOut(from: $0) }
            )
        }
    }

    /// Derive execution information from parsed case parameters
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
