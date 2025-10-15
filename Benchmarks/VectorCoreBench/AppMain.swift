import Foundation
import VectorCore
import VectorCoreBenchmarking

// Entry point for vectorcore-bench

// MARK: - CLI Parsing

struct CLIOptions {
    // Default to active suites only (graph removed)
    var suites: [String] = ["dot", "distance", "normalize", "batch"]
    var minTimeSeconds: Double = 0.5
    var repeats: Int? = nil
    var samples: Int = 1
    var format: String = "pretty"
    var outputPath: String? = nil
    var profile: String = "short"
    var mode: String? = nil // quick|full|smoke (overrides profile when set)
    var runLabel: String? = nil
    // Filters
    var filters: [String] = []      // include patterns
    var excludes: [String] = []     // exclude patterns
    var filterMode: String = "glob" // glob|regex
    var help: Bool = false
    var dims: [Int] = [512, 768, 1536]
    var batchNs: [Int] = [100, 1_000] // profile-driven; default for short
    // Track user overrides so profile presets don't clobber explicit flags
    var _userSetMinTime: Bool = false
    var _userSetDims: Bool = false
    var _userSetBatchNs: Bool = false
    // Reproducible randomness: global run seed used to derive per-case seeds
    var runSeed: UInt64 = 0
    var _userSetRunSeed: Bool = false
    // Feature toggles (tri-state; nil means "not set")
    var preferSoA: Bool? = nil
    var useMixedPrecision: Bool? = nil
    var abCompare: Bool? = nil
    var useUnderscored: Bool? = nil
    var useCKernels: Bool? = nil
    var releaseSize: Bool? = nil
    var abOnly: Bool? = nil
    // Correctness thresholds and gating
    var maxRelError: Double? = nil
    var maxAbsError: Double? = nil
    var strictCorrectness: Bool = false
    // Progress reporting
    var progressFormat: String = "none" // none|text|json
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
        case "--mode":
            if let v = it.next() { opts.mode = v }
        case "--run-label":
            if let v = it.next() { opts.runLabel = v }
        case "--filter":
            if let v = it.next() { opts.filters = v.split(separator: ",").map { String($0) }.filter { !$0.isEmpty } }
        case "--exclude":
            if let v = it.next() { opts.excludes = v.split(separator: ",").map { String($0) }.filter { !$0.isEmpty } }
        case "--filter-mode":
            if let v = it.next() { opts.filterMode = v }
        case "--prefer-soa":
            opts.preferSoA = true
        case "--no-prefer-soa":
            opts.preferSoA = false
        case "--use-mixed-precision":
            opts.useMixedPrecision = true
        case "--no-use-mixed-precision":
            opts.useMixedPrecision = false
        case "--ab":
            opts.abCompare = true
        case "--no-ab":
            opts.abCompare = false
        case "--use-underscored":
            opts.useUnderscored = true
        case "--no-use-underscored":
            opts.useUnderscored = false
        case "--use-ckernels":
            opts.useCKernels = true
        case "--no-use-ckernels":
            opts.useCKernels = false
        case "--release-size":
            opts.releaseSize = true
        case "--no-release-size":
            opts.releaseSize = false
        case "--ab-only":
            opts.abOnly = true
        case "--no-ab-only":
            opts.abOnly = false
        case "--max-rel-error":
            if let v = it.next(), let d = Double(v) { opts.maxRelError = d }
        case "--max-abs-error":
            if let v = it.next(), let d = Double(v) { opts.maxAbsError = d }
        case "--strict-correctness":
            opts.strictCorrectness = true
        case "--progress-format":
            if let v = it.next() { opts.progressFormat = v }
        case "--dims":
            if let v = it.next() {
                let parts = v.split(separator: ",").compactMap { Int($0) }
                if !parts.isEmpty { opts.dims = parts; opts._userSetDims = true }
            }
        case "--batch-ns":
            if let v = it.next() {
                let parts = v.split(separator: ",").compactMap { Int($0) }
                if !parts.isEmpty { opts.batchNs = parts; opts._userSetBatchNs = true }
            }
        case "--run-seed":
            if let v = it.next() {
                if let s = UInt64(v) { opts.runSeed = s; opts._userSetRunSeed = true }
                else if v.lowercased().hasPrefix("0x"), let s = UInt64(v.dropFirst(2), radix: 16) { opts.runSeed = s; opts._userSetRunSeed = true }
            }
        default:
            break
        }
    }
    var out = applyProfile(opts)
    out = applyMode(out)
    // Fallback to env var when CLI not provided
    if !out._userSetRunSeed, let env = ProcessInfo.processInfo.environment["VC_RUN_SEED"] {
        if let s = UInt64(env) { out.runSeed = s }
        else if env.lowercased().hasPrefix("0x"), let s = UInt64(env.dropFirst(2), radix: 16) { out.runSeed = s }
    }
    if out.runLabel == nil, let envLabel = ProcessInfo.processInfo.environment["VC_RUN_LABEL"], !envLabel.isEmpty {
        out.runLabel = envLabel
    }
    // Resolve feature toggles with env fallbacks and sensible defaults
    if out.preferSoA == nil {
        if let s = ProcessInfo.processInfo.environment["VC_SOA"], s == "1" { out.preferSoA = true } else { out.preferSoA = false }
    }
    if out.useMixedPrecision == nil {
        if let s = ProcessInfo.processInfo.environment["VC_MIXED_PRECISION"], s == "1" { out.useMixedPrecision = true } else { out.useMixedPrecision = false }
    }
    if out.abCompare == nil {
        if let s = ProcessInfo.processInfo.environment["VC_BATCH_AB"] { out.abCompare = (s != "0") } else { out.abCompare = true }
    }
    if out.abOnly == nil {
        if let s = ProcessInfo.processInfo.environment["VC_AB_ONLY"], s == "1" { out.abOnly = true } else { out.abOnly = false }
    }
    // abOnly implies abCompare
    if out.abOnly == true { out.abCompare = true }
    // Threshold env fallbacks
    if out.maxRelError == nil, let s = ProcessInfo.processInfo.environment["VC_MAX_REL_ERROR"], let d = Double(s) {
        out.maxRelError = d
    }
    if out.maxAbsError == nil, let s = ProcessInfo.processInfo.environment["VC_MAX_ABS_ERROR"], let d = Double(s) {
        out.maxAbsError = d
    }
    if out.useUnderscored == nil {
        if let s = ProcessInfo.processInfo.environment["VC_USE_UNDERSCORED"], s == "1" { out.useUnderscored = true } else { out.useUnderscored = false }
    }
    if out.useCKernels == nil {
        if let s = ProcessInfo.processInfo.environment["VC_USE_CKERNELS"], s == "1" { out.useCKernels = true } else { out.useCKernels = false }
    }
    if out.releaseSize == nil {
        if let s = ProcessInfo.processInfo.environment["VC_RELEASE_SIZE"], s == "1" { out.releaseSize = true } else { out.releaseSize = false }
    }
    return out
}

func printUsage() {
    let usage = """
    vectorcore-bench — VectorCore benchmarks

    Usage:
      vectorcore-bench [--suites dot,distance,normalize,batch] [--min-time 0.5] [--repeats N] [--samples N]
                       [--format pretty|json|csv] [--out path] [--profile short|full] [--mode quick|full|smoke]
                       [--dims 512,768,1536] [--batch-ns 100,1000,10000]
                       [--run-seed <u64>] [--run-label <string>]
                       [--filter <glob1,glob2>] [--exclude <glob1,glob2>] [--filter-mode glob|regex]
                       [--max-rel-error <d>] [--max-abs-error <d>] [--strict-correctness]

    Options:
      --suites      Comma-separated suite list
      --min-time    Target measurement time per case in seconds (default 0.5)
      --repeats     Fixed repeat count (overrides min-time if provided)
      --samples     Repeat whole measurement N times and aggregate (default 1)
      --format      Output format: pretty|json|csv
      --out         Output file path for machine-readable formats
      --profile     short|full preset (sets min-time, dims, batch sizes)
      --mode        quick|full|smoke preset (overrides profile; sets min-time, dims, batch sizes)
      --dims        Comma-separated dimensions (default: 512,768,1536)
      --batch-ns    Comma-separated candidate counts for batch benches (default varies by mode/profile)
      --run-seed    Global seed (u64 or 0xHEX) to derive per-case inputs (default 0; also reads VC_RUN_SEED)
      --run-label   Optional label for this run (also reads VC_RUN_LABEL)
      --filter      Include only cases matching any pattern (glob default; use --filter-mode regex for regex)
      --exclude     Exclude cases matching any pattern
      --filter-mode glob|regex (default glob)
      --prefer-soa  Prefer Structure-of-Arrays kernels for batch when available (fallback VC_SOA=1)
      --use-mixed-precision  Enable FP16 mixed-precision kernels when available (fallback VC_MIXED_PRECISION=1)
      --ab          Enable A/B comparisons (euclidean vs euclidean2, fused vs preNorm) (fallback VC_BATCH_AB, default on)
      --use-underscored      Build/measure with underscored attributes enabled
      --use-ckernels         Enable C kernel paths (when available)
      --release-size         Use size-optimized build profile for A/B
      --ab-only      Restrict generated cases to A/B pairs (batch euclidean vs euclidean2; cosine fused vs preNorm)
      --max-rel-error  Max allowed relative error for correctness gating (env VC_MAX_REL_ERROR)
      --max-abs-error  Max allowed absolute error for correctness gating (env VC_MAX_ABS_ERROR)
      --strict-correctness  Exit non-zero when any case exceeds thresholds
      Defaults: FP32 rel<=1e-6; FP16 rel<=2e-3; cosine(FP16) rel<=3e-3
      --progress-format  Progress output format: none|text|json (default: none)
                         Emits structured progress events to stderr for UI tools
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
        if !opts._userSetBatchNs { opts.batchNs = [100, 1_000, 10_000] }
        if opts.samples < 1 { opts.samples = 1 }
    default: // "short" and any unknown
        if !opts._userSetMinTime { opts.minTimeSeconds = 0.2 }
        // dims default stays as-is; short batch sizes
        if !opts._userSetBatchNs { opts.batchNs = [100, 1_000] }
        if opts.samples < 1 { opts.samples = 1 }
    }
    return opts
}

// Apply mode presets; takes precedence over profile for overlapping parameters.
func applyMode(_ options: CLIOptions) -> CLIOptions {
    var opts = options
    guard let modeRaw = opts.mode?.lowercased() else { return opts }
    switch modeRaw {
    case "quick":
        if !opts._userSetMinTime { opts.minTimeSeconds = 0.2 }
        if !opts._userSetDims { opts.dims = [512, 768, 1536] }
        if !opts._userSetBatchNs { opts.batchNs = [100, 1_000] }
    case "full":
        if !opts._userSetMinTime { opts.minTimeSeconds = 1.0 }
        if !opts._userSetDims { opts.dims = [512, 768, 1536] }
        if !opts._userSetBatchNs { opts.batchNs = [100, 1_000, 10_000] }
    case "smoke":
        if !opts._userSetMinTime { opts.minTimeSeconds = 0.1 }
        if !opts._userSetDims { opts.dims = [512] }
        if !opts._userSetBatchNs { opts.batchNs = [100] }
    default:
        break
    }
    return opts
}

protocol BenchmarkSuite {
    static var name: String { get }
    static func run(options: CLIOptions, progress: ProgressReporter) async -> [BenchResult]
}

func printPretty(results: [BenchResult], options: CLIOptions) {
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

    // A/B summaries (gated by flag)
    guard options.abCompare ?? true else { return }
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

    // Correctness summary (show when any case reported correctness stats)
    let corrCases = results.compactMap { r in r.correctness.map { (r, $0) } }
    if !corrCases.isEmpty {
        var worstRel: (name: String, val: Double)? = nil
        var worstAbs: (name: String, val: Double)? = nil
        var failCount = 0
        var passCount = 0
        var fp16Fail = 0
        var fp16Pass = 0
        var soaFail = 0
        var soaPass = 0

        for (r, c) in corrCases {
            let policy = defaultThresholdPolicy(for: r)
            let relThr = options.maxRelError ?? policy.maxRel
            let absThr = options.maxAbsError ?? policy.maxAbs
            let isFailRel = relThr.map { c.maxRelError > $0 } ?? false
            let isFailAbs = absThr.map { c.maxAbsError > $0 } ?? false
            let failed = isFailRel || isFailAbs
            if failed { failCount += 1 } else { passCount += 1 }

            if policy.isFP16 { if failed { fp16Fail += 1 } else { fp16Pass += 1 } }
            if policy.isSoA { if failed { soaFail += 1 } else { soaPass += 1 } }

            if worstRel == nil || c.maxRelError > worstRel!.val { worstRel = (r.name, c.maxRelError) }
            if worstAbs == nil || c.maxAbsError > worstAbs!.val { worstAbs = (r.name, c.maxAbsError) }
        }

        print("\n--- Correctness Summary ---")
        print("cases=\(corrCases.count)  pass=\(passCount)  fail=\(failCount)")
        if let w = worstRel { print(String(format: "worst maxRel=%.3e  (%@)", w.val, w.name)) }
        if let w = worstAbs { print(String(format: "worst maxAbs=%.3e  (%@)", w.val, w.name)) }
        if fp16Pass + fp16Fail > 0 {
            print("FP16 pass=\(fp16Pass) fail=\(fp16Fail)  (default rel<=\(String(format: "%.1e", DefaultThresholds.fp16RelDefault)))")
        }
        if soaPass + soaFail > 0 {
            print("SoA pass=\(soaPass) fail=\(soaFail)  (uses FP32 defaults)")
        }
        if options.maxRelError == nil && options.maxAbsError == nil {
            print("defaults: FP32 rel<=\(String(format: "%.1e", DefaultThresholds.fp32RelDefault)), FP16 rel<=\(String(format: "%.1e", DefaultThresholds.fp16RelDefault)), cosine(FP16) rel<=\(String(format: "%.1e", DefaultThresholds.fp16CosineRelDefault))")
        } else {
            let relS = options.maxRelError.map { String(format: "%.3e", $0) } ?? "nil"
            let absS = options.maxAbsError.map { String(format: "%.3e", $0) } ?? "nil"
            print("overrides: maxRel=\(relS) maxAbs=\(absS)")
        }
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

        // Create progress reporter based on CLI flag
        let progressFormat = ProgressFormat(rawValue: options.progressFormat) ?? .none
        let progress = ProgressReporter(format: progressFormat)

        // AB-only: restrict to batch benchmarks since A/B pairs are defined there
        let suiteNames: [String] = (options.abOnly ?? false) ? ["batch"] : options.suites
        for s in suiteNames {
            guard let suite = registry[s.lowercased()] else {
                fputs("Warning: unknown suite: \(s)\n", stderr)
                continue
            }
            progress.suiteStarted(s)
            let suiteStartTime = Date()
            let results = await suite.run(options: options, progress: progress)
            allResults.append(contentsOf: results)
            let suiteDuration = Date().timeIntervalSince(suiteStartTime) * 1000.0
            progress.suiteCompleted(suite: s, cases: results.count, durationMs: suiteDuration)
        }

        // Always print pretty for local visibility
        let filtered: [BenchResult]
        if !options.filters.isEmpty || !options.excludes.isEmpty {
            filtered = allResults.filter { Filters.shouldRun(name: $0.name, options: options) }
        } else {
            filtered = allResults
        }
        printPretty(results: filtered, options: options)

        // Write machine-readable outputs when requested
        let format = options.format.lowercased()
        let flags = RunFlags(
            preferSoA: options.preferSoA ?? false,
            useMixedPrecision: options.useMixedPrecision ?? false,
            abCompare: options.abCompare ?? true,
            abOnly: options.abOnly ?? false,
            useUnderscored: options.useUnderscored ?? false,
            useCKernels: options.useCKernels ?? false,
            releaseSize: options.releaseSize ?? false
        )
        // Persist only explicit CLI/global thresholds; calibrated defaults are applied per-case during evaluation
        let thr = RunThresholds(maxRelError: options.maxRelError, maxAbsError: options.maxAbsError)
        let runFilters = (!options.filters.isEmpty || !options.excludes.isEmpty) ? RunFilters(mode: options.filterMode, include: options.filters.isEmpty ? nil : options.filters, exclude: options.excludes.isEmpty ? nil : options.excludes) : nil
        let metadata = EnvCapture.collectMetadata(suites: options.suites, dims: options.dims, buildConfig: "release", runSeed: options.runSeed, runLabel: options.runLabel, flags: flags, thresholds: (thr.maxRelError != nil || thr.maxAbsError != nil) ? thr : nil, filters: runFilters)
        let cases = JSONWriter.toCases(filtered)
        let ab = (flags.abCompare ? ABComparisons.buildComparisons(cases: cases) : [])

        if format == "json" {
            let path = options.outputPath ?? defaultRunPath(for: metadata, ext: "json")
            let run = BenchRun(metadata: metadata, results: cases, abComparisons: ab.isEmpty ? nil : ab)
            do {
                try JSONWriter.write(run: run, to: path)
                print("\nSaved JSON results to: \(path)")
            } catch {
                fputs("Failed to write JSON to \(path): \(error)\n", stderr)
            }
        } else if format == "csv" {
            // Default CSV alongside JSON with same layout
            let path = options.outputPath ?? defaultRunPath(for: metadata, ext: "csv")
            do {
                try CSVWriter.write(cases: cases, metadata: metadata, to: path)
                print("\nSaved CSV results to: \(path)")
            } catch {
                fputs("Failed to write CSV to \(path): \(error)\n", stderr)
            }
        }

        // Correctness gating: evaluate after writing results so artifacts are preserved
        if options.strictCorrectness || thr.maxRelError != nil || thr.maxAbsError != nil {
            let violations = evaluateCorrectnessWithPolicy(results: filtered, overrideRel: thr.maxRelError, overrideAbs: thr.maxAbsError)
            if violations.count > 0 {
                print("\nCorrectness violations: \(violations.count)")
                for v in violations.prefix(10) { print("- \(v)") }
                if options.strictCorrectness { exit(1) }
            } else {
                print("\nCorrectness violations: 0")
            }
        }
    }
}

// Build default run file path: .bench/runs/<device-tag>/<timestamp>_<git-sha>.<ext>
private func defaultRunPath(for meta: BenchMetadata, ext: String) -> String {
    let baseDir = ".bench/runs"
    let device = sanitizePathComponent(meta.deviceTag)
    // Timestamp in UTC: yyyyMMdd_HHmmss
    let df = DateFormatter()
    df.calendar = Calendar(identifier: .gregorian)
    df.locale = Locale(identifier: "en_US_POSIX")
    df.timeZone = TimeZone(secondsFromGMT: 0)
    df.dateFormat = "yyyyMMdd_HHmmss"
    let ts = df.string(from: Date())
    let sha = meta.gitSHA ?? "nogit"
    let file = "\(ts)_\(sha).\(ext)"
    return "\(baseDir)/\(device)/\(file)"
}

private func sanitizePathComponent(_ s: String) -> String {
    let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._")
    let mapped = s.unicodeScalars.map { allowed.contains($0) ? Character($0) : "-" }
    var result = String(mapped)
    // Collapse repeated dashes
    while result.contains("--") { result = result.replacingOccurrences(of: "--", with: "-") }
    return result.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
}

// Scan results and return list of violating case names with max errors
private enum DefaultThresholds {
    // Baseline FP32 default (relative)
    static let fp32RelDefault = 1e-6
    // Mixed-precision FP16 defaults (relative)
    static let fp16RelDefault = 2e-3
    // Cosine can accumulate more error in FP16
    static let fp16CosineRelDefault = 3e-3
}

private struct ThresholdPolicy {
    let maxRel: Double?
    let maxAbs: Double?
    let isFP16: Bool
    let isSoA: Bool
}

private func defaultThresholdPolicy(for r: BenchResult) -> ThresholdPolicy {
    let name = r.name.lowercased()
    let p = CaseParsing.parse(name: r.name)
    let isFP16 = name.contains("fp16")
    let isSoA = name.contains("soa")
    let metric = (p.metric ?? "")

    // Relative threshold selection
    let rel: Double
    if isFP16 {
        if metric == "cosine" { rel = DefaultThresholds.fp16CosineRelDefault }
        else { rel = DefaultThresholds.fp16RelDefault }
    } else {
        rel = DefaultThresholds.fp32RelDefault
    }
    // Absolute threshold left nil by default (relative dominates)
    let abs: Double? = nil
    return ThresholdPolicy(maxRel: rel, maxAbs: abs, isFP16: isFP16, isSoA: isSoA)
}

private func evaluateCorrectnessWithPolicy(results: [BenchResult], overrideRel: Double?, overrideAbs: Double?) -> [String] {
    var out: [String] = []
    for r in results {
        guard let c = r.correctness else { continue }
        let policy = defaultThresholdPolicy(for: r)
        let relThr = overrideRel ?? policy.maxRel
        let absThr = overrideAbs ?? policy.maxAbs
        let relBad = relThr.map { c.maxRelError > $0 } ?? false
        let absBad = absThr.map { c.maxAbsError > $0 } ?? false
        if relBad || absBad {
            let desc = String(format: "%@  maxRel=%.3e maxAbs=%.3e  thr(rel=%@ abs=%@)",
                              r.name,
                              c.maxRelError, c.maxAbsError,
                              relThr.map { String(format: "%.1e", $0) } ?? "nil",
                              absThr.map { String(format: "%.1e", $0) } ?? "nil")
            out.append(desc)
        }
    }
    return out
}
