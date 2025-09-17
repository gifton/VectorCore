import Foundation
import VectorCore

// Entry point for vectorcore-bench

// MARK: - CLI Parsing

struct CLIOptions {
    var suites: [String] = ["dot", "distance", "normalize", "batch"]
    var minTimeSeconds: Double = 0.5
    var repeats: Int? = nil
    var format: String = "pretty"
    var outputPath: String? = nil
    var profile: String = "short"
    var help: Bool = false
    var dims: [Int] = [512, 768, 1536]
}

func parseCLI() -> CLIOptions {
    var opts = CLIOptions()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = it.next() {
        switch arg {
        case "--help", "-h":
            opts.help = true
        case "--suites":
            if let v = it.next() { opts.suites = v.split(separator: ",").map { String($0) } }
        case "--min-time":
            if let v = it.next(), let d = Double(v) { opts.minTimeSeconds = d }
        case "--repeats":
            if let v = it.next(), let n = Int(v) { opts.repeats = n }
        case "--format":
            if let v = it.next() { opts.format = v }
        case "--out":
            if let v = it.next() { opts.outputPath = v }
        case "--profile":
            if let v = it.next() { opts.profile = v }
        case "--dims":
            if let v = it.next() {
                let parts = v.split(separator: ",").compactMap { Int($0) }
                if !parts.isEmpty { opts.dims = parts }
            }
        default:
            break
        }
    }
    return opts
}

func printUsage() {
    let usage = """
    vectorcore-bench — VectorCore benchmarks

    Usage:
      vectorcore-bench [--suites dot,distance,normalize,batch] [--min-time 0.5] [--repeats N]
                       [--format pretty|json|csv] [--out path] [--profile short|full]
                       [--dims 512,768,1536]

    Options:
      --suites      Comma-separated suite list
      --min-time    Target measurement time per case in seconds (default 0.5)
      --repeats     Fixed repeat count (overrides min-time if provided)
      --format      Output format: pretty|json|csv
      --out         Output file path for machine-readable formats
      --profile     short|full preset (placeholder)
      --dims        Comma-separated dimensions (default: 512,768,1536)
      -h, --help    Show this help message
    """
    print(usage)
}

protocol BenchmarkSuite {
    static var name: String { get }
    static func run(options: CLIOptions) async -> [BenchResult]
}

func printPretty(results: [BenchResult]) {
    print("\n=== VectorCore Bench (Phase 2) ===")
    for r in results {
        let nsPerIter = Double(r.totalNanoseconds) / Double(max(r.iterations, 1))
        let unitNs = nsPerIter / Double(max(r.unitCount, 1))
        let msTotal = Double(r.totalNanoseconds) / 1_000_000.0
        let name = r.name.padding(toLength: 40, withPad: " ", startingAt: 0)
        var line = "\(name)  iters=\(String(format: "%8d", r.iterations))  total=\(String(format: "%8.3f", msTotal)) ms  time/op=\(String(format: "%10.2f", nsPerIter)) ns"
        if r.unitCount > 1 {
            let throughput = (Double(r.iterations * r.unitCount) / (Double(r.totalNanoseconds) / 1_000_000_000.0))
            line += String(format: "  unit/op=%8.2f ns  throughput=%8.0f vec/s", unitNs, throughput)
        }
        if r.name.hasPrefix("dot.") || r.name.hasPrefix("dist.dot.") {
            let parts = r.name.split(separator: ".")
            let dimStr = r.name.hasPrefix("dot.") ? (parts.count > 1 ? parts[1] : "") : (parts.count > 2 ? parts[2] : "")
            if let n = Int(dimStr) {
                let gflops = (2.0 * Double(n)) / nsPerIter
                line += String(format: "  gflop/s=%6.2f", gflops)
            }
        }
        if nsPerIter < 5.0 {
            line += "  [WARN <5ns suspicious]"
        }
        print(line)
    }
}

@main
struct App {
    static func main() async {
        let options = parseCLI()
        if options.help {
            printUsage()
            return
        }

        var allResults: [BenchResult] = []
        let registry: [String: BenchmarkSuite.Type] = [
            NoopBench.name: NoopBench.self,
            DotProductBench.name: DotProductBench.self,
            DistanceBench.name: DistanceBench.self,
            NormalizationBench.name: NormalizationBench.self,
            BatchBench.name: BatchBench.self
        ]

        for s in options.suites {
            guard let suite = registry[s.lowercased()] else {
                fputs("Warning: unknown suite: \(s)\n", stderr)
                continue
            }
            let results = await suite.run(options: options)
            allResults.append(contentsOf: results)
        }

        // Always print pretty for local visibility
        printPretty(results: allResults)

        // Write machine-readable outputs when requested
        let format = options.format.lowercased()
        let metadata = EnvCapture.collectMetadata(suites: options.suites, dims: options.dims, buildConfig: "release")
        let cases = JSONWriter.toCases(allResults)
        let defaultDir = ".bench"
        try? FileManager.default.createDirectory(atPath: defaultDir, withIntermediateDirectories: true)

        if format == "json" {
            let path = options.outputPath ?? defaultDir + "/results.json"
            let run = BenchRun(metadata: metadata, results: cases)
            do { try JSONWriter.write(run: run, to: path) } catch {
                fputs("Failed to write JSON to \(path): \(error)\n", stderr)
            }
        } else if format == "csv" {
            let path = options.outputPath ?? defaultDir + "/results.csv"
            do { try CSVWriter.write(cases: cases, metadata: metadata, to: path) } catch {
                fputs("Failed to write CSV to \(path): \(error)\n", stderr)
            }
        }
    }
}
