import Foundation

public struct BenchCase: Codable {
    public let name: String
    public let params: CaseParams
    public let iterations: Int
    public let totalNS: UInt64
    public let nsPerOp: Double
    public let unitCount: Int
    public let nsPerUnit: Double
    public let throughputPerSec: Double
    public let gflops: Double?
    public let suspicious: Bool
    public let samples: Int?
    public let meanNsPerOp: Double?
    public let medianNsPerOp: Double?
    public let p90NsPerOp: Double?
    public let stddevNsPerOp: Double?
    public let rsdPercent: Double?
}

public struct BenchRun: Codable {
    public let metadata: BenchMetadata
    public let results: [BenchCase]
}

enum JSONWriter {
    static func toCases(_ results: [BenchResult]) -> [BenchCase] {
        results.map { r in
            let nsPerOp = Double(r.totalNanoseconds) / Double(max(r.iterations, 1))
            let unitCount = max(r.unitCount, 1)
            let nsPerUnit = nsPerOp / Double(unitCount)
            let throughput = (Double(r.iterations * unitCount) / (Double(r.totalNanoseconds) / 1_000_000_000.0))
            let params = CaseParsing.parse(name: r.name)
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
                rsdPercent: r.rsdPercent
            )
        }
    }

    static func write(run: BenchRun, to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(run)
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }
}
