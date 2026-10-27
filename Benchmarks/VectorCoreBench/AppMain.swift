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
    var configurationError: String? = nil
    var help: Bool = false
    var dims: [Int] = [512, 768, 1536]
    var batchNs: [Int] = [100, 1_000] // profile-driven; default for short
    // Track user overrides so profile presets don't clobber explicit flags
    var _userSetMinTime: Bool = false
    var _userSetDims: Bool = false
    var _userSetBatchNs: Bool = false
    var _userSetSuites: Bool = false
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
      --suites      Comma-separated suite list. Opt-in only (never default, never auto-paired): boundary, the setup-inclusive
                    companions, reusable-output counterparts and consumer-only records (see Docs/Benchmark_Measurement_Boundaries.md)
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
      --use-ckernels         Assert this binary was compiled with VC_USE_C_KERNELS
      --ab-only      Restrict generated cases to A/B pairs (batch euclidean vs euclidean2; cosine fused vs preNorm).
                     Runs only the batch suite: an explicit --suites must be exactly batch (fallback VC_AB_ONLY=1)
      --max-rel-error  Max allowed relative error for correctness gating (env VC_MAX_REL_ERROR)
      --max-abs-error  Max allowed absolute error for correctness gating (env VC_MAX_ABS_ERROR)
      --strict-correctness  Exit non-zero when any case exceeds thresholds
      Defaults: derived per-route bounds for the modeled Euclidean and matrix routes (see CorrectnessPolicy);
                FP32 rel<=1e-6, FP16 rel<=2e-3, cosine(FP16) rel<=3e-3 for every other validated route
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

/// Scope, work unit, consumer and per-operation summary appended to a result line.
private func boundarySuffix(_ r: BenchResult) -> String {
    guard let b = r.boundary else { return "" }
    var text = "  scope=\(b.timingScope.rawValue) unit=\(b.workUnit)x\(b.unitsPerInvocation) consumer=\(b.consumer)"
    if let stats = PerOperationStatistics(sampleNsPerInvocation: r.sampleNsPerInvocation, unitsPerInvocation: b.unitsPerInvocation) {
        text += String(format: "  per-%@ median=%.4f ns/%@ RSD=%.2f%%", b.workUnit, stats.median, b.workUnit, stats.rsdPercent)
    }
    return text
}

private func printSamples(_ r: BenchResult) {
    guard r.sampleNsPerInvocation.count > 1 else { return }
    print("    samples ns/invocation: " + r.sampleNsPerInvocation.map { String(format: "%.2f", $0) }.joined(separator: " "))
    guard let b = r.boundary,
          let stats = PerOperationStatistics(sampleNsPerInvocation: r.sampleNsPerInvocation,
              unitsPerInvocation: b.unitsPerInvocation) else { return }
    print("    samples ns/\(b.workUnit): " + stats.samples.map { String(format: "%.4f", $0) }.joined(separator: " "))
}

private func printComparisons(_ comparisons: [ABComparison]) {
    for comparison in comparisons {
        print("AB \(comparison.comparison) D=\(comparison.dim) N=\(comparison.n) \(comparison.variant ?? "") "
              + "\(comparison.provider ?? "") delta=\(comparison.deltaPercent)%")
        print("    purpose: \(comparison.purpose)")
        if let left = comparison.leftBoundary { print("    left  \(comparison.leftName): \(left.canonicalJSON())") }
        if let right = comparison.rightBoundary { print("    right \(comparison.rightName): \(right.canonicalJSON())") }
    }
}

func printPretty(results: [BenchResult], options: ValidatedOptions, comparisons: [ABComparison], unavailable: [UnavailableCase] = []) {
    print("\n=== VectorCore Bench (Phase 2) ===")
    print(BenchmarkCompiledFeatures.bannerLine())
    print("statistic: \(BenchMetadata.currentStatisticUnit) (per-<unit> fields are ns per declared work unit; time/op, med= and p90= "
        + "are ns per invocation)")
    for r in results {
        let nsPerIter = Double(r.totalNanoseconds) / Double(max(r.iterations, 1))
        let unitNs = nsPerIter / Double(max(r.unitCount, 1))
        let msTotal = Double(r.totalNanoseconds) / 1_000_000.0
        let name = r.name.padding(toLength: 40, withPad: " ", startingAt: 0)
        var line = "\(name)  iters=\(String(format: "%8d", r.iterations))  total=\(String(format: "%8.3f", msTotal)) ms  time/op=\(String(format: "%10.2f", nsPerIter)) ns"
        if r.unitCount > 1 {
            let throughput = ((Double(r.iterations) * Double(r.unitCount)) / (Double(r.totalNanoseconds) / 1_000_000_000.0))
            line += String(format: "  unit/op=%8.2f ns  throughput=%8.0f vec/s", unitNs, throughput)
        }
        if r.samples > 1, let med = r.medianNsPerOp, let p90 = r.p90NsPerOp, let rsd = r.rsdPercent {
            line += String(format: "  med=%8.2f ns/inv p90=%8.2f ns/inv RSD=%5.2f%%", med, p90, rsd)
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
        line += boundarySuffix(r)
        print(line)
        printSamples(r)
    }

    printComparisons(comparisons)
    for entry in unavailable {
        print("UNAVAILABLE \(entry.name): \(entry.reason)")
    }

    printCorrectnessSummary(results: results, options: options)
}

private func printCorrectnessSummary(results: [BenchResult], options: ValidatedOptions) {
    let corrCases = results.compactMap { r in r.correctness.map { (r, $0) } }
    let declaredNotApplicable = results.filter {
        $0.correctness == nil && CorrectnessPolicy.validation(forCaseName: $0.name) != .required
    }.count
    if !corrCases.isEmpty {
        var worstRel: (name: String, val: Double)? = nil
        var worstAbs: (name: String, val: Double)? = nil
        var failCount = 0
        var passCount = 0
        var derived = 0
        var empirical = 0

        for (r, c) in corrCases {
            // The same decision the gate records (evaluateCorrectnessWithPolicy), so the summary cannot disagree with it.
            let failed = CorrectnessPolicy.failure(for: r, overrideRel: options.maxRelError, overrideAbs: options.maxAbsError) != nil
            if failed { failCount += 1 } else { passCount += 1 }
            if case .derived = CorrectnessPolicy.rule(forCaseName: r.name).basis { derived += 1 } else { empirical += 1 }
            if c.status == .valid {
                if worstRel == nil || c.maxRelError > worstRel!.val { worstRel = (r.name, c.maxRelError) }
                if worstAbs == nil || c.maxAbsError > worstAbs!.val { worstAbs = (r.name, c.maxAbsError) }
            }
        }

        print("\n--- Correctness Summary ---")
        print("cases=\(corrCases.count)  pass=\(passCount)  fail=\(failCount)")
        if let w = worstRel { print(String(format: "worst maxRel=%.3e  (%@)", w.val, w.name)) }
        if let w = worstAbs { print(String(format: "worst maxAbs=%.3e  (%@)", w.val, w.name)) }
        if options.maxRelError == nil && options.maxAbsError == nil {
            let fixed = [CorrectnessPolicy.fp32EmpiricalRelative, CorrectnessPolicy.fp16EmpiricalRelative,
                         CorrectnessPolicy.fp16CosineEmpiricalRelative].map { String(format: "%.1e", $0) }
            print("thresholds: \(derived) case(s) derived per route from worst-case rounding bounds (CorrectnessPolicy); "
                + "\(empirical) case(s) on empirical fixed thresholds: FP32 rel<=\(fixed[0]), FP16 rel<=\(fixed[1]), "
                + "cosine(FP16) rel<=\(fixed[2])")
        } else {
            let relS = options.maxRelError.map { String(format: "%.3e", $0) } ?? "nil"
            let absS = options.maxAbsError.map { String(format: "%.3e", $0) } ?? "nil"
            print("overrides: maxRel=\(relS) maxAbs=\(absS)")
        }
    }
    if declaredNotApplicable > 0 {
        print("not validated (declared not applicable in CorrectnessPolicy): \(declaredNotApplicable) case(s)")
    }
}

/// The `fixture-injected` timing scope exists only for injected test suites; a production registry must never contain it.
func injectedScopeProblem(cases: [BenchmarkCase], injectedRegistry: Bool) -> String? {
    guard !injectedRegistry else { return nil }
    let offenders = cases.filter { $0.boundary.timingScope == .fixtureInjected }.map(\.name)
    return offenders.isEmpty ? nil : "fixture-injected scope in a production registry: \(offenders.joined(separator: ","))"
}

/// The explanation printed when every selected entry is unavailable: nil when something is executable, otherwise one header
/// line and one `UNAVAILABLE <id>: <reason>` line per entry.
func unavailableOnlyMessage(cases: [BenchmarkCase]) -> String? {
    guard !cases.isEmpty, cases.allSatisfy({ $0.availability != .available }) else { return nil }
    var text = "Invalid configuration: no executable selected cases (all selected entries are unavailable)\n"
    for item in cases {
        if case .unavailable(let reason) = item.availability { text += "UNAVAILABLE \(item.name): \(reason)\n" }
    }
    return text
}

public struct BenchmarkRunner {
    public static func runCLI() async -> Int32 {
        await run(options: parseCLI())
    }

    static func run(options input: CLIOptions, registry injectedRegistry: [String: BenchmarkSuite.Type]? = nil, onCase: (@Sendable (String) -> Void)? = nil) async -> Int32 {
        let options: ValidatedOptions
        let allowed: Set<String> = Set(injectedRegistry?.keys.map { $0 }
            ?? ["dot", "distance", "normalize", "batch", "matrix", "memory", "noop", "boundary"])
        do {
            options = try ValidatedOptions(input, allowedSuites: allowed)
        } catch {
            fputs("Invalid configuration: \(error)\n", stderr); return 1
        }
        if input.help { printUsage(); return 0 }
        var allResults: [BenchResult] = []
        var allUnavailable: [UnavailableCase] = []
        let thermalStart = EnvCapture.thermalState()
        let registry: [String: BenchmarkSuite.Type] = injectedRegistry ?? [
            NoopBench.name: NoopBench.self,
            DotProductBench.name: DotProductBench.self,
            DistanceBench.name: DistanceBench.self,
            NormalizationBench.name: NormalizationBench.self,
            BatchBench.name: BatchBench.self,
            MemoryBench.name: MemoryBench.self,
            MatrixDistanceBench.name: MatrixDistanceBench.self,
            BoundaryBench.name: BoundaryBench.self
        ]

        // Create progress reporter based on CLI flag
        let progressFormat = ProgressFormat(rawValue: options.progressFormat) ?? .none
        let progress = BenchmarkProgressReporter(format: progressFormat)

        let plans = options.suites.map { suite in
            (suite, registry[suite]!.catalog(options: options, progress: progress).filter { options.caseFilter.matches($0.name) })
        }
        let selectedCases = plans.flatMap { $0.1 }
        if let problem = injectedScopeProblem(cases: selectedCases, injectedRegistry: injectedRegistry != nil) {
            fputs("Invalid configuration: \(problem)\n", stderr); return 1
        }
        let executableCases = selectedCases.filter { $0.availability == .available }
        let selectedIDs = executableCases.map(\.name)
        let unavailableIDs = selectedCases.filter { $0.availability != .available }.map(\.name)
        guard !selectedCases.isEmpty else { fputs("Invalid configuration: empty selected case set\n", stderr); return 1 }
        if let message = unavailableOnlyMessage(cases: selectedCases) { fputs(message, stderr); return 1 }
        do {
            let missing = try ABComparisons.missingSides(boundaries: selectedCases.map { ($0.name, $0.boundary) })
            if options.abOnly == true && !missing.isEmpty { throw BenchmarkReportError(missing.joined(separator: "; ")) }
        } catch { fputs("Invalid configuration: \(error)\n", stderr); return 1 }
        for (suite, cases) in plans where !cases.isEmpty {
            progress.suiteStarted(suite)
            let start = Date()
            let executed = await CaseExecution.execute(cases, options: options, progress: progress, suite: suite, onCase: onCase)
            allResults += executed.results
            allUnavailable += executed.unavailable
            progress.suiteCompleted(suite: suite, cases: cases.count, durationMs: Date().timeIntervalSince(start) * 1000)
        }
        if allResults.isEmpty { allResults = [BenchResult(name: "empty-run", iterations: 0, totalNanoseconds: 0, executionFailure: "Selected cases returned no measurements")] }
        let filtered = allResults // Selection already happened before case setup; never post-filter results.
        let executionFailures = filtered.compactMap { r in
            reportFailure(for: r).map { BenchmarkFailure(name: r.name, reason: $0, boundary: r.boundary) }
        }
        let reportable = filtered.filter { reportFailure(for: $0) == nil }
        let violations = evaluateCorrectnessWithPolicy(results: filtered, overrideRel: options.maxRelError, overrideAbs: options.maxAbsError)
        let failures = executionFailures + violations
        for failure in failures { fputs("\(failure.name): \(failure.reason)\n", stderr) }

        // Write machine-readable outputs when requested
        let format = options.format.lowercased()
        let flags = RunFlags(
            preferSoA: options.preferSoA ?? false,
            useMixedPrecision: options.useMixedPrecision ?? false,
            abCompare: options.abCompare ?? true,
            abOnly: options.abOnly ?? false,
            useCKernels: VectorCoreCompiledFeatures.useCKernels,
            requestedCKernels: options.useCKernels
        )
        // Persist only explicit CLI/global thresholds; CorrectnessPolicy's per-case thresholds are applied during evaluation
        let thr = RunThresholds(maxRelError: options.maxRelError, maxAbsError: options.maxAbsError)
        let runFilters = (!options.filters.isEmpty || !options.excludes.isEmpty) ? RunFilters(mode: options.filterMode,
            include: options.filters.isEmpty ? nil : options.filters, exclude: options.excludes.isEmpty ? nil : options.excludes) : nil
        let metadata = EnvCapture.collectMetadata(suites: plans.filter { !$0.1.isEmpty }.map { $0.0 }, dims: options.dims,
            buildConfig: BenchmarkCompiledFeatures.configuration, runSeed: options.runSeed, runLabel: options.runLabel, flags: flags,
            thresholds: (thr.maxRelError != nil || thr.maxAbsError != nil) ? thr : nil, filters: runFilters, selectedCaseIDs: selectedIDs, samples: options.samples,
            repeats: options.repeats, minTimeSeconds: options.minTimeSeconds, batchNs: options.batchNs, profile: options.profile,
                mode: options.mode,
            unavailableCaseIDs: unavailableIDs, thermalStateStart: thermalStart,
            harnessCalibration: options.repeats.map { "not used (fixed --repeats \($0): exactly that many invocations per sample)" }
                ?? "\(HarnessConfig.calibrationVersion); clock tick \(ClockResolution.tickNanoseconds) ns (\(ClockResolution.source))")
        let cases: [BenchCase]
        let ab: [ABComparison]
        let missingSides: [String]
        do {
            cases = try JSONWriter.toCases(reportable)
            if flags.abCompare {
                let report = try ABComparisons.report(cases: cases)
                ab = report.comparisons; missingSides = report.missingSides
            } else { ab = []; missingSides = [] }
        } catch { fputs("Report integrity error: \(error)\n", stderr); return 1 }
        for missing in missingSides { fputs("\(missing)\n", stderr) }
        printPretty(results: reportable, options: options, comparisons: ab, unavailable: allUnavailable)

        if format == "json" {
            let path = options.outputPath ?? defaultRunPath(for: metadata, ext: "json")
            let run = BenchRun(metadata: metadata, results: cases, abComparisons: ab.isEmpty ? nil : ab, failures: failures,
                abMissingSides: missingSides, unavailable: allUnavailable)
            do {
                try JSONWriter.write(run: run, to: path)
                print("\nSaved JSON results to: \(path)")
            } catch {
                fputs("Failed to write JSON to \(path): \(error)\n", stderr)
                return 1
            }
        } else if format == "csv" {
            // Default CSV alongside JSON with same layout
            let path = options.outputPath ?? defaultRunPath(for: metadata, ext: "csv")
            do {
                try CSVWriter.write(cases: cases, metadata: metadata, failures: failures, unavailable: allUnavailable, to: path)
                print("\nSaved CSV results to: \(path)")
            } catch {
                fputs("Failed to write CSV to \(path): \(error)\n", stderr)
                return 1
            }
        }

        return !executionFailures.isEmpty || (options.strictCorrectness && !violations.isEmpty) ? 1 : 0
    }
}

// Build default run file path: .bench/runs/<device-tag>/<timestamp>_<git-sha>.<ext>. The SHA is `meta.gitSHA`, the
// `git rev-parse HEAD` of the benchmark process's working directory at run time, not the identity of the measured
// source ("nogit" when unavailable).
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

/// The correctness gate: one failure per result that `CorrectnessPolicy` rejects (see its derivations and declarations).
func evaluateCorrectnessWithPolicy(results: [BenchResult], overrideRel: Double?, overrideAbs: Double?) -> [BenchmarkFailure] {
    var out: [BenchmarkFailure] = []
    if results.isEmpty { out.append(BenchmarkFailure(name: "run", reason: "No selected results; missing validation")) }
    for r in results {
        if let failure = CorrectnessPolicy.failure(for: r, overrideRel: overrideRel, overrideAbs: overrideAbs) { out.append(failure) }
    }
    return out
}

func reportFailure(for result: BenchResult) -> String? {
    if let error = result.executionFailure { return error }
    guard result.iterations > 0, result.totalNanoseconds > 0, result.unitCount > 0, result.samples > 0 else { return "Invalid timing counts or duration" }
    let statistics = [result.meanNsPerOp, result.medianNsPerOp, result.p90NsPerOp, result.stddevNsPerOp, result.rsdPercent].compactMap { $0 }
    guard statistics.allSatisfy({ $0.isFinite && $0 >= 0 }) else { return "Nonfinite or negative timing statistic" }
    return nil
}
