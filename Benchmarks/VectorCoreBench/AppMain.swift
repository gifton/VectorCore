import Foundation
import VectorCore

// Entry point for vectorcore-bench

// MARK: - CLI Parsing

struct CLIOptions {
    var suites: [String] = ["dot", "distance", "normalize", "batch"]
    var minTimeSeconds: Double = 0.5
    var repeats: Int? = nil
    var samples: Int = 1
    var format: String = "pretty"
    var outputPath: String? = nil
    var profile: String = "short"
    var help: Bool = false
    var dims: [Int] = [512, 768, 1536]
    var batchNs: [Int] = [100, 1_000] // profile-driven; default for short
    // Track user overrides so profile presets don't clobber explicit flags
    var _userSetMinTime: Bool = false
    var _userSetDims: Bool = false
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
            if let v = it.next(), let d = Double(v) { opts.minTimeSeconds = d; opts._userSetMinTime = true }
        case "--repeats":
            if let v = it.next(), let n = Int(v) { opts.repeats = n }
        case "--samples":
            if let v = it.next(), let n = Int(v) { opts.samples = max(1, n) }
        case "--format":
            if let v = it.next() { opts.format = v }
        case "--out":
            if let v = it.next() { opts.outputPath = v }
        case "--profile":
            if let v = it.next() { opts.profile = v }
        case "--dims":
            if let v = it.next() {
                let parts = v.split(separator: ",").compactMap { Int($0) }
                if !parts.isEmpty { opts.dims = parts; opts._userSetDims = true }
            }
        default:
            break
        }
    }
    return applyProfile(opts)
}

func printUsage() {
    let usage = """
    vectorcore-bench — VectorCore benchmarks

    Usage:
      vectorcore-bench [--suites dot,distance,normalize,batch] [--min-time 0.5] [--repeats N] [--samples N]
                       [--format pretty|json|csv] [--out path] [--profile short|full]
                       [--dims 512,768,1536]

    Options:
      --suites      Comma-separated suite list
      --min-time    Target measurement time per case in seconds (default 0.5)
      --repeats     Fixed repeat count (overrides min-time if provided)
      --samples     Repeat whole measurement N times and aggregate (default 1)
      --format      Output format: pretty|json|csv
      --out         Output file path for machine-readable formats
      --profile     short|full preset (sets min-time, dims, batch sizes)
      --dims        Comma-separated dimensions (default: 512,768,1536)
      -h, --help    Show this help message
    """
    print(usage)
}

// Apply profile defaults without clobbering explicit user-provided flags
func applyProfile(_ options: CLIOptions) -> CLIOptions {
    var opts = options
    let prof = opts.profile.lowercased()
    switch prof {
    case "full":
        if !opts._userSetMinTime { opts.minTimeSeconds = 1.0 }
        // Keep dims unless overridden; default already set
        opts.batchNs = [100, 1_000, 10_000]
        if opts.samples < 1 { opts.samples = 1 }
    default: // "short" and any unknown
        if !opts._userSetMinTime { opts.minTimeSeconds = 0.2 }
        // dims default stays as-is; short batch sizes
        opts.batchNs = [100, 1_000]
        if opts.samples < 1 { opts.samples = 1 }
    }
    return opts
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
        if r.samples > 1, let med = r.medianNsPerOp, let p90 = r.p90NsPerOp, let rsd = r.rsdPercent {
            line += String(format: "  med=%8.2f ns p90=%8.2f ns RSD=%5.2f%%", med, p90, rsd)
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

    // A/B summary: compare euclidean (sqrt) vs euclidean2 (squared) for batch cases
    // Group by dim, N, variant, provider
    struct Key: Hashable { let dim: Int?; let n: Int?; let variant: String?; let provider: String? }
    var byKey: [Key: [String: BenchResult]] = [:]
    for r in results {
        let p = CaseParsing.parse(name: r.name)
        guard p.kind == "batch" else { continue }
        let key = Key(dim: p.dim, n: p.n, variant: p.variant, provider: p.provider)
        byKey[key, default: [:]][p.metric ?? ""] = r
    }

    var abLines: [String] = []
    for (key, dict) in byKey {
        guard let r2 = dict["euclidean2"], let r1 = dict["euclidean"] else { continue }
        let ns1 = Double(r1.totalNanoseconds) / Double(max(r1.iterations, 1))
        let ns2 = Double(r2.totalNanoseconds) / Double(max(r2.iterations, 1))
        let unitNs1 = ns1 / Double(max(r1.unitCount, 1))
        let unitNs2 = ns2 / Double(max(r2.unitCount, 1))
        let delta = (unitNs1 - unitNs2) / unitNs1 * 100.0
        let dimS = key.dim.map(String.init) ?? "?"
        let nS = key.n.map(String.init) ?? "?"
        let varS = key.variant ?? "?"
        let provS = key.provider ?? "?"
        let line = String(format: "AB batch d=%@ N=%@ %@ %@  euclid2 vs euclid: unit/op=%8.2f ns vs %8.2f ns  Δ=%6.2f%%",
                          dimS, nS, varS, provS, unitNs2, unitNs1, delta)
        abLines.append(line)
    }
    if !abLines.isEmpty {
        print("\n--- A/B Summary (euclidean2 vs euclidean) ---")
        for l in abLines.sorted() { print(l) }
    }

    // A/B summary: cosine fused vs preNormalized for batch cases
    var cosByKey: [Key: [String: BenchResult]] = [:]
    for r in results {
        let p = CaseParsing.parse(name: r.name)
        guard p.kind == "batch", p.metric == "cosine" else { continue }
        let key = Key(dim: p.dim, n: p.n, variant: p.variant, provider: p.provider)
        cosByKey[key, default: [:]][p.variant ?? ""] = r
    }
    var cosLines: [String] = []
    for (key, dict) in cosByKey {
        guard let rf = dict["optimized-fused"], let rp = dict["optimized-preNorm"] else { continue }
        let nsF = Double(rf.totalNanoseconds) / Double(max(rf.iterations, 1))
        let nsP = Double(rp.totalNanoseconds) / Double(max(rp.iterations, 1))
        let unitF = nsF / Double(max(rf.unitCount, 1))
        let unitP = nsP / Double(max(rp.unitCount, 1))
        let delta = (unitF - unitP) / unitF * 100.0
        let dimS = key.dim.map(String.init) ?? "?"
        let nS = key.n.map(String.init) ?? "?"
        let provS = key.provider ?? "?"
        let line = String(format: "AB batch d=%@ N=%@ cosine %@  preNorm vs fused: unit/op=%8.2f ns vs %8.2f ns  Δ=%6.2f%%",
                          dimS, nS, provS, unitP, unitF, delta)
        cosLines.append(line)
    }
    if !cosLines.isEmpty {
        print("\n--- A/B Summary (cosine preNorm vs fused) ---")
        for l in cosLines.sorted() { print(l) }
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
            BatchBench.name: BatchBench.self,
            MemoryBench.name: MemoryBench.self
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
