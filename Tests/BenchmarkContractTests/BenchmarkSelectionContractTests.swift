import Foundation
import XCTest
import VectorCore
@testable import VectorCoreBenchSupport
@testable import VectorCoreBenchmarking

private final class CaseLedger: @unchecked Sendable {
    private let lock = NSLock()
    private var names: [String] = []
    func record(_ name: String) { lock.lock(); defer { lock.unlock() }; names.append(name) }
    var snapshot: [String] { lock.lock(); defer { lock.unlock() }; return names }
}
private actor ExcludedCounter { var count = 0; func record() { count += 1 } }
private enum ExcludedFailure: Error { case called }
private struct ExcludedProvider: ComputeProvider {
    let counter: ExcludedCounter
    var device: ComputeDevice { .cpu }
    var maxConcurrency: Int { 1 }
    var deviceInfo: ComputeDeviceInfo { .init(name: "excluded", maxThreads: 1, preferredChunkSize: 1) }
    func execute<T: Sendable>(_ body: @Sendable @escaping () async throws -> T) async throws -> T { try await body() }
    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R, minChunk: Int,
        _ body: @Sendable @escaping (Range<Int>) async throws -> R,
        _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        await counter.record(); throw ExcludedFailure.called
    }
}

final class BenchmarkSelectionContractTests: XCTestCase {
    private let progress = BenchmarkProgressReporter(format: .none)
    private let suites: [(String, any BenchmarkSuite.Type)] = [
        ("dot", DotProductBench.self), ("distance", DistanceBench.self), ("normalize", NormalizationBench.self),
        ("batch", BatchBench.self), ("matrix", MatrixDistanceBench.self), ("memory", MemoryBench.self), ("noop", NoopBench.self)
    ]
    private func options(_ arguments: [String]) throws -> ValidatedOptions {
        try ValidatedOptions(parseCLI(arguments: arguments, environment: [:]))
    }
    // Independent family table. This never obtains expected IDs from the catalog/parser.
    private func expected(_ suite: String, dims: [Int], counts: [Int], ab: Bool, soa: Bool, fp16: Bool) -> Set<String> {
        var names = Set<String>()
        for d in dims {
            switch suite {
            case "dot": for v in ["generic", "optimized"] { names.insert("dot.\(d).\(v)") }
            case "distance": for m in ["euclidean", "cosine", "manhattan", "dot", "chebyshev", "hamming", "minkowski"] {
                for v in ["generic", "optimized"] { names.insert("dist.\(m).\(d).\(v)") }
            }
            case "normalize": for status in ["success", "zeroFail"] { for v in ["generic", "optimized"] { names.insert("normalize.\(status).\(d).\(v)") } }
            case "memory": for prefix in ["mem.alloc.aligned", "mem.copy", "mem.fill", "mem.pool.acquire"] { names.insert("\(prefix).\(d)") }
            case "matrix": for n in counts { names.insert("matrix.euclidean2.\(d).N\(n).optimized") }
            case "batch":
                for n in counts {
                    for provider in ["sequential", "parallel", "automatic"] {
                        for metric in ab ? ["euclidean2", "euclidean"] : ["euclidean2"] {
                            for variant in ["generic", "optimized"] { names.insert("batch.\(metric).\(d).N\(n).\(variant).\(provider)") }
                        }
                        for strategy in ["fused", "preNorm"] { names.insert("batch.cosine.\(d).N\(n).optimized-\(strategy).\(provider)") }
                    }
                    if soa && n >= 100 { names.insert("batch.euclidean2.\(d).N\(n).optimized-soa") }
                    // The independent eligibility boundary is the documented 12 MiB footprint plus N>=100.
                    if fp16 && n >= 100 && n * d * 4 > 12 * 1024 * 1024 {
                        names.insert("batch.euclidean2.\(d).N\(n).optimized-fp16")
                        names.insert("batch.cosine.\(d).N\(n).optimized-fp16")
                    }
                }
            default: break
            }
        }
        if suite == "noop" { names.insert("noop.scalar-add") }
        return names
    }
    func testIndependentCatalogTables() throws {
        let presets: [(String, [Int], [Int])] = [("smoke", [512], [100]), ("quick", [512,768,1536], [100,1000]), ("full", [512,768,1536], [100,1000,10000])]
        for (mode, dims, counts) in presets {
            for ab in [false, true] { for optional in [false, true] {
                for (family, suite) in suites {
                    let value = try options(["--suites", family, "--mode", mode, ab ? "--ab" : "--no-ab", optional ? "--prefer-soa" : "--no-prefer-soa", optional
                        ? "--use-mixed-precision" : "--no-use-mixed-precision"])
                    let catalog = suite.catalog(options: value, progress: progress)
                    let actual = catalog.map(\.name)
                    XCTAssertEqual(actual.count, Set(actual).count)
                    // Every catalog case carries a structurally valid boundary supplied by the catalog itself.
                    for item in catalog {
                        XCTAssertNil(item.boundary.validationError(), item.name)
                        XCTAssertEqual(item.boundary.version, MeasurementBoundary.currentVersion, item.name)
                        XCTAssertEqual(item.boundary.timingScope, .reusedInput, item.name)
                    }
                    XCTAssertEqual(Set(actual), expected(family, dims: dims, counts: counts, ab: ab, soa: optional, fp16: optional), "\(family) \(mode) \(ab) \(optional)")
                }
            } }
        }
        let value = try options(["--suites", "batch", "--prefer-soa", "--use-mixed-precision", "--ab-only"])
        let batchCatalog = BatchBench.catalog(options: value, progress: progress)
        let names = batchCatalog.map(\.name)
        XCTAssertEqual(Set(names), expected("batch", dims: [512,768,1536], counts: [100,1000], ab: true, soa: false, fp16: false))
        let cases = try JSONWriter.toCases(batchCatalog.map { BenchResult(name: $0.name, iterations: 1, totalNanoseconds: 100,
            unitCount: $0.boundary.unitsPerInvocation, boundary: $0.boundary) })
        let report = try ABComparisons.report(cases: cases)
        // Plan-time (pre-execution) pairing agrees: nothing is missing for the complete --ab-only selection.
        XCTAssertTrue(try ABComparisons.missingSides(boundaries: batchCatalog.map { ($0.name, $0.boundary) }).isEmpty)
        XCTAssertEqual(report.comparisons.filter { $0.comparison == "euclidean2_vs_euclidean" }.count, 36)
        XCTAssertEqual(report.comparisons.filter { $0.comparison == "cosine_prenorm_vs_fused" }.count, 18)
        XCTAssertTrue(report.missingSides.isEmpty)
    }
    func testSelectionFiltersAndPrecedence() throws {
        let value = try options(["--suites", "dot,dot", "--profile", "full", "--mode", "smoke", "--dims", "1536,512,1536", "--batch-ns", "3,2,3", "--min-time", "0",
            "--samples", "2", "--repeats", "3", "--run-seed", "0x10"])
        XCTAssertEqual(value.suites, ["dot"]); XCTAssertEqual(value.dims, [1536,512]); XCTAssertEqual(value.batchNs, [3,2])
        XCTAssertEqual(value.minTimeSeconds, 0); XCTAssertEqual(value.samples, 2); XCTAssertEqual(value.repeats, 3); XCTAssertEqual(value.runSeed, 16)
        let parsed = parseCLI(arguments: ["--suites", "noop", "--run-seed", "7", "--no-ab"], environment: ["VC_RUN_SEED": "bad", "VC_BATCH_AB": "bad"])
        XCTAssertEqual(try ValidatedOptions(parsed).runSeed, 7)
        let glob = try CaseFilter(include: ["dot.*.optimized"], exclude: ["*.768.*"], mode: "glob")
        XCTAssertTrue(glob.matches("dot.512.optimized")); XCTAssertFalse(glob.matches("dot.768.optimized")); XCTAssertFalse(glob.matches("prefix.dot.512.optimized"))
        let regex = try CaseFilter(include: ["512"], exclude: [], mode: "regex")
        XCTAssertTrue(regex.matches("prefix.dot.512.optimized"))
        XCTAssertThrowsError(try CaseFilter(include: ["["], exclude: [], mode: "regex"))
        for arguments in [["--mystery"], ["--dims"], ["--dims", "512,no"], ["--dims", "512,"], ["--suites", "wat"], ["--mode", "wat"], ["--profile", "wat"], ["--format",
            "wat"], ["--filter-mode", "wat"], ["--progress-format", "wat"], ["--samples", "0"], ["--repeats", "-1"], ["--run-seed", "0x"], ["--run-seed", "-1"], ["--min-time",
            "nan"], ["--min-time", "18446744073.709553"], ["--release-size"], ["--no-use-underscored"]] {
            XCTAssertThrowsError(try options(arguments), "\(arguments)")
        }
        for key in ["VC_SOA", "VC_MIXED_PRECISION", "VC_BATCH_AB", "VC_AB_ONLY", "VC_USE_CKERNELS"] {
            XCTAssertThrowsError(try ValidatedOptions(parseCLI(arguments: [], environment: [key: "true"])))
        }
        XCTAssertNoThrow(try options(["--suites", "noop", "--dims", String(Int.max)]))
        XCTAssertThrowsError(try options(["--suites", "memory", "--dims", String(Int.max)]))
        XCTAssertThrowsError(try options(["--suites", "matrix", "--batch-ns", String(Int.max)]))
        XCTAssertNoThrow(try options(["--suites", "memory", "--dims", "1"]))
        XCTAssertThrowsError(try options(["--suites", "dot", "--dims", "1"]))
    }
    @MainActor func testActualOptimizedOnlyAndExcludedProviderAllWidths() async throws {
        let options = try self.options(["--suites", "dot", "--repeats", "1", "--min-time", "0", "--filter", "*.optimized"])
        let ledger = CaseLedger()
        let results = await CaseExecution.run(DotProductBench.catalog(options: options, progress: progress), options: options, progress: progress, suite: "dot", onCase: ledger.record)
        XCTAssertEqual(Set(ledger.snapshot), Set(["dot.512.optimized", "dot.768.optimized", "dot.1536.optimized"]))
        XCTAssertEqual(results.count, 3); XCTAssertTrue(results.allSatisfy { $0.executionFailure == nil && $0.iterations == 1 })
        for dim in [512,768,1536] {
            let options = try self.options(["--suites", "batch", "--dims", String(dim), "--batch-ns", "1", "--filter", "not-a-case"])
            let counter = ExcludedCounter()
            let results = await BatchBench.run(options: options, progress: progress, provider: ExcludedProvider(counter: counter))
            let count = await counter.count
            XCTAssertTrue(results.isEmpty); XCTAssertEqual(count, 0)
        }
    }
    func testIndependentPairingAndIntegrity() throws {
        func boundary(_ name: String) throws -> MeasurementBoundary { try XCTUnwrap(Boundaries.lookup(name, runSeed: 0), name) }
        func record(_ name: String, _ time: UInt64, units: Int = 1, boundary override: MeasurementBoundary? = nil) throws -> BenchResult {
            let b = try override ?? boundary(name)
            return .init(name: name, iterations: 1, totalNanoseconds: time, unitCount: units, boundary: b)
        }
        let left = "batch.cosine.512.N2.optimized-preNorm.sequential", right = "batch.cosine.512.N2.optimized-fused.sequential"
        for (time, delta) in [(UInt64(20), 20.0), (30, -20)] {
            let report = try ABComparisons.report(cases: JSONWriter.toCases([record(left, time, units: 2), record(right, 25, units: 2)]))
            XCTAssertEqual(report.comparisons.count, 1); XCTAssertEqual(report.comparisons[0].deltaPercent, delta)
            XCTAssertEqual(report.comparisons[0].variant, "optimized")
            XCTAssertEqual(report.comparisons[0].family, "cosine_prenorm_vs_fused")
            XCTAssertFalse(report.comparisons[0].purpose.isEmpty)
            XCTAssertNotNil(report.comparisons[0].leftBoundary); XCTAssertNotNil(report.comparisons[0].rightBoundary)
        }
        XCTAssertEqual(try ABComparisons.missingSides(boundaries: [(left, boundary(left))]).count, 1)
        XCTAssertThrowsError(try ABComparisons.report(cases: JSONWriter.toCases([record(left, 20, units: 2), record(left, 25, units: 2)])))
        XCTAssertThrowsError(try ABComparisons.report(cases: JSONWriter.toCases([record(left, 20, units: 2), record(right, 25, units: 3)])))
        // Different dimension never pairs: the two sides sit in different declared groups.
        let wide = right.replacingOccurrences(of: "512", with: "768")
        let unpaired = try ABComparisons.report(cases: JSONWriter.toCases([record(left, 20, units: 2), record(wide, 25, units: 2)]))
        XCTAssertTrue(unpaired.comparisons.isEmpty); XCTAssertEqual(unpaired.missingSides.count, 2)
        let encoded = try JSONEncoder().encode(ABComparisons.report(cases: JSONWriter.toCases([record(left, 20, units: 2), record(right,
            25, units: 2)])).comparisons)
        let text = String(bytes: encoded, encoding: .utf8) ?? ""
        // Preparation strings are derived from the declared boundaries, not from the family name.
        XCTAssertTrue(text.contains("l2-normalized-outside-timing"))
        XCTAssertTrue(text.contains("(outside timing)"))
        XCTAssertTrue(text.contains("no separate input preparation"))
    }
    func testFamilyAllowedDifferenceMutationTable() throws {
        // For each declared family: every boundary field outside the allowed set, mutated on one side, must be
        // rejected; every allowed field, mutated, must still pair. Mutations are derived from the flattened
        // boundary, so a newly added field is covered automatically.
        let pairs: [(family: String, left: String, right: String, allowed: Set<String>)] = [
            ("euclidean2_vs_euclidean", "batch.euclidean2.512.N2.optimized.sequential", "batch.euclidean.512.N2.optimized.sequential",
                ["operation", "route"]),
            ("euclidean2_vs_euclidean", "batch.euclidean2.768.N2.generic.parallel", "batch.euclidean.768.N2.generic.parallel",
                ["operation", "route"]),
            ("cosine_prenorm_vs_fused", "batch.cosine.1536.N2.optimized-preNorm.automatic",
                "batch.cosine.1536.N2.optimized-fused.automatic",
             ["operation", "route", "fixture.preparation", "inclusions.inputPreparation"])
        ]
        func mutated(_ boundary: MeasurementBoundary, path: String) throws -> MeasurementBoundary {
            var object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(boundary)) as? [String: Any])
            func set(_ dictionary: inout [String: Any], _ parts: ArraySlice<String>) {
                let key = parts.first!
                if parts.count > 1 { var nested = dictionary[key] as! [String: Any]; set(&nested,
                    parts.dropFirst()); dictionary[key] = nested; return }
                switch dictionary[key] {
                case let text as String:
                    let flips: [String: String] = ["included": "excluded", "excluded": "included", "notApplicable": "included",
                        "unverified": "included", "reused-input": "setup-inclusive"]
                    dictionary[key] = flips[text] ?? text + "~mutated"
                case let number as Int: dictionary[key] = number + 1
                case let flag as Bool: dictionary[key] = !flag
                case let list as [Any]: dictionary[key] = list + ["mutated-note"]
                default: dictionary[key] = "mutated"
                }
            }
            set(&object, path.split(separator: ".").map(String.init)[...])
            return try JSONDecoder().decode(MeasurementBoundary.self, from: JSONSerialization.data(withJSONObject: object))
        }
        for (family, leftName, rightName, allowed) in pairs {
            let left = try XCTUnwrap(Boundaries.lookup(leftName, runSeed: 0)), right = try XCTUnwrap(Boundaries.lookup(rightName,
                runSeed: 0))
            func report(_ l: MeasurementBoundary, _ r: MeasurementBoundary) throws -> ABReport {
                try ABComparisons.report(cases: JSONWriter.toCases([
                    BenchResult(name: leftName, iterations: 1, totalNanoseconds: 100, unitCount: l.unitsPerInvocation, boundary: l),
                    BenchResult(name: rightName, iterations: 1, totalNanoseconds: 100, unitCount: r.unitsPerInvocation, boundary: r)]))
            }
            XCTAssertEqual(try report(left, right).comparisons.count, 1, family)
            XCTAssertEqual(ComparisonFamilies.allowedDifferences(family: family), allowed)
            // `comparison.family`/`group` define the pair; `comparison.side` differs by construction.
            let paths = Set(right.flattened().keys).subtracting(["comparison.side"])
            var rejected = 0, accepted = 0
            for path in paths.sorted() {
                guard right.flattened()[path] != nil else { continue }
                let mutant = try mutated(right, path: path)
                if allowed.contains(path) {
                    // An allowed difference is permitted; it never blocks the pair.
                    XCTAssertEqual(try report(left, mutant).comparisons.count, 1, "\(family) allowed \(path)"); accepted += 1
                } else if path == "comparison.family" || path == "comparison.group" {
                    // These move the member out of its declared group, so no pair forms (and both sides are reported missing).
                    let outcome = try? report(left, mutant)
                    XCTAssertTrue(outcome == nil || outcome!.comparisons.isEmpty, "\(family) \(path)"); rejected += 1
                } else {
                    XCTAssertThrowsError(try report(left, mutant), "\(family) disallowed \(path)"); rejected += 1
                }
            }
            XCTAssertGreaterThan(rejected, 15, family); XCTAssertEqual(accepted, allowed.intersection(paths).count, family)
        }
    }
    func testCrossScopeNeverPairsAndUndeclaredNamesDoNotPair() throws {
        let leftName = "batch.euclidean2.512.N2.optimized.sequential", rightName = "batch.euclidean.512.N2.optimized.sequential"
        let left = try XCTUnwrap(Boundaries.lookup(leftName, runSeed: 0))
        var right = try XCTUnwrap(Boundaries.lookup(rightName, runSeed: 0))
        right.timingScope = .setupInclusive
        func cases(_ l: MeasurementBoundary, _ r: MeasurementBoundary) throws -> [BenchCase] {
            try JSONWriter.toCases([BenchResult(name: leftName, iterations: 1, totalNanoseconds: 100, unitCount: 2, boundary: l),
                                    BenchResult(name: rightName, iterations: 1, totalNanoseconds: 100, unitCount: 2, boundary: r)])
        }
        XCTAssertThrowsError(try ABComparisons.report(cases: cases(left, right))) { XCTAssertTrue("\($0)".contains("Cross-scope")) }
        // Same names, boundaries that declare no comparison: the old name pattern must not pair them.
        var l = left, r = try XCTUnwrap(Boundaries.lookup(rightName, runSeed: 0))
        l.comparison = nil; r.comparison = nil
        let report = try ABComparisons.report(cases: cases(l, r))
        XCTAssertTrue(report.comparisons.isEmpty); XCTAssertTrue(report.missingSides.isEmpty)
        // An unknown declared family is an integrity error, not a silent skip.
        var unknown = left; unknown.comparison = ComparisonMembership(family: "made_up", side: "left", group: "g")
        XCTAssertThrowsError(try ABComparisons.report(cases: cases(unknown, r)))
    }
    func testReportRequiresBoundaryIdentity() throws {
        XCTAssertThrowsError(try JSONWriter.toCases([BenchResult(name: "dot.512.optimized", iterations: 1,
            totalNanoseconds: 10)])) { XCTAssertTrue("\($0)".contains("Missing measurement boundary")) }
        var invalid = try XCTUnwrap(Boundaries.lookup("dot.512.optimized", runSeed: 0)); invalid.unitsPerInvocation = 0
        XCTAssertThrowsError(try JSONWriter.toCases([BenchResult(name: "dot.512.optimized", iterations: 1, totalNanoseconds: 10,
            boundary: invalid)]))
        // Boundary identity changes when the measured work changes.
        let a = try XCTUnwrap(Boundaries.lookup("matrix.euclidean2.512.N17.optimized", runSeed: 0))
        XCTAssertNotEqual(a.canonicalJSON(), try XCTUnwrap(Boundaries.lookup("matrix.euclidean2.512.N65.optimized",
            runSeed: 0)).canonicalJSON())
        XCTAssertNotEqual(a.canonicalJSON(), try XCTUnwrap(Boundaries.lookup("matrix.euclidean2.512.N17.optimized",
            runSeed: 1)).canonicalJSON())
        XCTAssertEqual(a.consumer, "u64-bitfold-sink-v1"); XCTAssertEqual(a.workUnit,
            "produced-entry"); XCTAssertEqual(a.unitsPerInvocation, 289)
        XCTAssertEqual(a.inclusions.routeInternalPacking, .included)
        XCTAssertTrue(a.notes.contains { $0.contains("not a prepacked GEMM") })
    }
    func testQ6ConsumerLabelsAreAccurate() throws {
        func consumer(_ name: String) throws -> String { try XCTUnwrap(Boundaries.lookup(name, runSeed: 0), name).consumer }
        for dim in [512, 768, 1536] {
            for variant in ["generic", "optimized"] {
                XCTAssertEqual(try consumer("normalize.success.\(dim).\(variant)"), "u64-bitfold-sink-v1")
                XCTAssertEqual(try consumer("normalize.zeroFail.\(dim).\(variant)"), "error-outcome-sink-v1")
                XCTAssertEqual(try consumer("dot.\(dim).\(variant)"), "scalar-sink-v1")
                XCTAssertEqual(try consumer("dist.euclidean.\(dim).\(variant)"), "scalar-sink-v1")
            }
            XCTAssertEqual(try consumer("batch.euclidean2.\(dim).N4.optimized.sequential"), "provider-float-sum-observe-sink-v1")
            XCTAssertEqual(try consumer("batch.euclidean2.\(dim).N4.optimized-soa"), "float-sum-all-outputs-sink-v1")
            XCTAssertEqual(try consumer("mem.copy.\(dim)"), "escaped-pointer-retention-v1")
        }
        XCTAssertEqual(try consumer("noop.scalar-add"), "scalar-sink-v1")
        // No catalog entry may use a label that claims partial consumption of a multi-element numeric output.
        let partial = ["first-coordinate-sink-v1", "branch-constant-sink-v1", "single-element-sink-v1", "diagonal-pair-sink-v0"]
        let value = try options(["--suites", "dot,distance,normalize,batch,matrix,memory,noop,boundary", "--mode", "quick", "--prefer-soa",
            "--use-mixed-precision"])
        for (_, suite) in suites + [("boundary", BoundaryBench.self as any BenchmarkSuite.Type)] {
            for item in suite.catalog(options: value, progress: progress) { XCTAssertFalse(partial.contains(item.boundary.consumer),
                item.name) }
        }
    }

    func testCheckedArithmeticAndClock() throws {
        XCTAssertEqual(try BenchmarkArithmetic.nanoseconds(0), 0)
        XCTAssertThrowsError(try BenchmarkArithmetic.nanoseconds(Double(UInt64.max) / 1e9))
        XCTAssertThrowsError(try BenchmarkArithmetic.product(Int.max, 2))
        XCTAssertThrowsError(try BenchmarkArithmetic.sum(UInt64.max, 1))
        XCTAssertThrowsError(try BenchmarkArithmetic.count(UInt64.max))
        XCTAssertThrowsError(try BenchmarkArithmetic.duration(2,1))
        XCTAssertThrowsError(try BenchmarkArithmetic.duration(1,1))
        var calls = 0
        XCTAssertThrowsError(try Harness.measure(name: "overflow", minTimeSeconds: 0, repeats: Int.max, samples: 2) { calls += 1 })
        XCTAssertEqual(calls, 0)
        var clock: [UInt64] = [2,1]
        XCTAssertThrowsError(try Harness.measure(name: "clock", minTimeSeconds: 0, repeats: 1, now: { clock.removeFirst() }) { calls += 1 })
        XCTAssertEqual(calls, 1)
        var samplesClock: [UInt64] = [0,UInt64.max,0,1]
        XCTAssertThrowsError(try Harness.measure(name: "sum", minTimeSeconds: 0, repeats: 1, samples: 2, now: { samplesClock.removeFirst() }) {})
    }
    func testMetadataBoundedCommands() {
        XCTAssertEqual(EnvCapture.observe(command: ["/usr/bin/printf", "comma,quote\"\nline"]).value, "comma,quote\"\nline")
        XCTAssertNotNil(EnvCapture.observe(command: ["/usr/bin/false"]).unavailableReason)
        let timed = EnvCapture.observe(command: ["/bin/sleep", "1"], timeout: 0.02)
        XCTAssertNil(timed.value); XCTAssertTrue(timed.unavailableReason?.contains("timeout") == true)
        XCTAssertNotNil(EnvCapture.observe(command: ["/not/a/program"]).unavailableReason)
    }
    @MainActor func testMatrixInputsAndActualControls() async throws {
        for dim in [512,768,1536] {
            let row = MatrixDistanceBench.inputs(dim: dim, row: 0, runSeed: 3)
            XCTAssertEqual(row.map(\.bitPattern), MatrixDistanceBench.inputs(dim: dim, row: 0, runSeed: 3).map(\.bitPattern))
            XCTAssertNotEqual(row, MatrixDistanceBench.inputs(dim: dim, row: 1, runSeed: 3))
            XCTAssertNotEqual(row, MatrixDistanceBench.inputs(dim: dim, row: 0, runSeed: 4))
            XCTAssertNotEqual(Array(row.prefix(17)), Array(row.dropFirst(17).prefix(17)))
            XCTAssertTrue(row.allSatisfy(\.isFinite)); XCTAssertTrue(row.contains { $0 < 0 }); XCTAssertTrue(row.contains { $0 > 0 })
            let options = try self.options(["--suites", "matrix", "--dims", String(dim), "--batch-ns", "3", "--repeats", "2", "--samples", "2", "--min-time", "0"])
            let results = await MatrixDistanceBench.run(options: options, progress: progress)
            XCTAssertEqual(results.count, 1); let result = try XCTUnwrap(results.first)
            XCTAssertNil(result.executionFailure); XCTAssertEqual(result.name, "matrix.euclidean2.\(dim).N3.optimized")
            XCTAssertEqual(result.iterations, 4); XCTAssertEqual(result.samples, 2); XCTAssertEqual(result.unitCount, 9)
            XCTAssertEqual(result.correctness?.samples, 9); XCTAssertEqual(result.correctness?.status, .valid)
            // Valid finite observations can exceed strict tolerance, including diagonal Gram residuals.
            XCTAssertTrue(try XCTUnwrap(result.correctness?.maxRelError).isFinite)
        }
    }
}

extension BenchmarkSelectionContractTests {
    /// `binarySizes.vectorCoreLibraryBytes` matches a loaded image by file name only. The previous predicate
    /// (`path.contains("VectorCore")`) matched any image under a VectorCore checkout, so reports recorded the executable's
    /// size as the library's.
    func testVectorCoreLibraryImageMatchesFileNameOnly() throws {
        let libraries = ["/usr/local/lib/libVectorCore.dylib",
                         "/Applications/App.app/Contents/Frameworks/VectorCore.framework/VectorCore",
                         "/Library/Frameworks/VectorCore.framework/Versions/A/VectorCore"]
        let others = ["/Users/me/dev/VectorCore/out/Products/Release/vectorcore-bench",
                      "/Users/me/VectorCore/out/Products/Debug/VectorCorePackageTests.xctest/Contents/MacOS/VectorCorePackageTests",
                      "/usr/local/lib/libVectorCoreBenchmarking.dylib", "/usr/local/lib/libVectorCoreC.dylib",
                      "/tmp/VectorCore", "/tmp/VectorCore.framework.old/VectorCore", "/usr/lib/swift/libswiftCore.dylib", ""]
        for path in libraries { XCTAssertTrue(EnvCapture.isVectorCoreLibraryImage(path), path) }
        for path in others { XCTAssertFalse(EnvCapture.isVectorCoreLibraryImage(path), path) }
        // Control: the old predicate accepts the checkout executable that the new matcher rejects.
        XCTAssertTrue(others[0].contains("VectorCore"))
        #if canImport(Darwin)
        // Live: whether VectorCore is a separate image is read from dladdr on a VectorCore type's metadata, independently of
        // the matcher. A library image is detected iff VectorCore is separate from the image holding this test (a SwiftPM
        // build links it statically: none is detected although images whose paths contain "VectorCore" are loaded).
        func image(of type: Any.Type) -> String? {
            var info = Dl_info()
            guard dladdr(unsafeBitCast(type, to: UnsafeRawPointer.self), &info) != 0, let name = info.dli_fname else { return nil }
            return String(cString: name)
        }
        let coreImage = try XCTUnwrap(image(of: Vector512Optimized.self))
        let testImage = try XCTUnwrap(image(of: BenchmarkSelectionContractTests.self))
        let detected = EnvCapture.loadedImagePaths().filter(EnvCapture.isVectorCoreLibraryImage)
        XCTAssertEqual(!detected.isEmpty, coreImage != testImage, "\(coreImage) vs \(testImage): \(detected)")
        if coreImage == testImage { XCTAssertTrue(EnvCapture.loadedImagePaths().contains { $0.contains("VectorCore") }) }
        #endif
    }
}

private final class AllocationLedger: @unchecked Sendable {
    private let lock = NSLock()
    private var attempts = 0
    private var live = Set<UInt>()
    private var releases: [Int] = []
    private var duplicateRelease = false
    let failAt: Int?
    init(failAt: Int? = nil) { self.failAt = failAt }
    func allocate(_ count: Int, _ alignment: Int) throws -> UnsafeMutablePointer<Float> {
        lock.lock(); attempts += 1; let fail = attempts == failAt; lock.unlock()
        if fail { throw ExcludedFailure.called }
        let pointer: UnsafeMutablePointer<Float> = try AlignedMemory.allocateAligned(type: Float.self, count: count, alignment: alignment)
        lock.lock(); live.insert(UInt(bitPattern: pointer)); lock.unlock()
        return pointer
    }
    func release(_ pointer: UnsafeMutablePointer<Float>, _ initialized: Int) {
        lock.lock(); let owned = live.remove(UInt(bitPattern: pointer)) != nil
        if !owned { duplicateRelease = true }
        releases.append(initialized); lock.unlock()
        if owned { pointer.deinitialize(count: initialized); AlignedMemory.deallocate(pointer) }
    }
    var snapshot: (Int, Int, [Int], Bool) { lock.lock(); defer { lock.unlock() }; return (attempts, live.count, releases, duplicateRelease) }
    var allocator: BenchmarkAllocator { BenchmarkAllocator(allocate: allocate, release: release) }
}

extension BenchmarkSelectionContractTests {
    @MainActor func testAllocatorFailureCleanupAndLazySetup() async throws {
        for (name, failAt, releaseCounts) in [("mem.copy.1", 1, []), ("mem.copy.1", 2, [1024]), ("mem.fill.1", 1, []), ("mem.alloc.aligned.1", 2, [8])] {
            let value = try options(["--suites", "memory", "--dims", "1", "--filter", name, "--repeats", "1", "--min-time", "0"])
            let ledger = AllocationLedger(failAt: failAt)
            let catalog = MemoryBench.catalog(options: value, allocator: ledger.allocator)
            XCTAssertEqual(ledger.snapshot.0, 0)
            let results = await CaseExecution.run(catalog, options: value, progress: progress, suite: "memory")
            XCTAssertEqual(results.count, 1); XCTAssertNotNil(results[0].executionFailure)
            let snapshot = ledger.snapshot
            XCTAssertEqual(snapshot.0, failAt); XCTAssertEqual(snapshot.1, 0); XCTAssertEqual(snapshot.2, releaseCounts); XCTAssertFalse(snapshot.3)
        }
        for name in ["mem.copy.1", "mem.fill.1"] {
            let value = try options(["--suites", "memory", "--dims", "1", "--filter", name, "--repeats", "2", "--min-time", "0"])
            let ledger = AllocationLedger()
            let results = await CaseExecution.run(MemoryBench.catalog(options: value, allocator: ledger.allocator), options: value, progress: progress, suite: "memory")
            XCTAssertNil(results[0].executionFailure); XCTAssertEqual(results[0].unitCount, 1024)
            let snapshot = ledger.snapshot
            XCTAssertEqual(snapshot.0, name.contains("copy") ? 2 : 1); XCTAssertEqual(snapshot.1, 0); XCTAssertEqual(snapshot.2.count, snapshot.0); XCTAssertFalse(snapshot.3)
        }
        let value = try options(["--suites", "memory", "--dims", "1", "--filter", "mem.pool.acquire.1", "--repeats", "1"])
        let ledger = AllocationLedger()
        let results = await CaseExecution.run(MemoryBench.catalog(options: value, allocator: ledger.allocator, poolAcquire: { _,_ in nil }), options: value, progress: progress, suite: "memory")
        XCTAssertTrue(results[0].executionFailure?.contains("pool acquisition") == true); XCTAssertEqual(ledger.snapshot.0, 0)
    }
}

extension BenchmarkSelectionContractTests {
    func testCompleteSmallMatrixIndependentBinaryOracle() throws {
        func verify<V: UnifiedVectorBuffer>(_ rows: [V], integerRows: [[Int]]) {
            let output = MatrixDistance.euclideanSquaredMatrix(queries: rows, candidates: rows)
            for i in rows.indices { for j in rows.indices {
                var sum: Int64 = 0
                for k in integerRows[i].indices {
                    let difference = Int64(integerRows[i][k] - integerRows[j][k])
                    sum += difference * difference
                }
                // Integer coordinates divided by 8 make every tested multiply/sum exact in Float.
                XCTAssertEqual(output[i * rows.count + j], Float(Double(sum) / 64))
            } }
        }
        for dimension in [512,768,1536] {
            let integers = (0..<4).map { row in (0..<dimension).map { ($0 * (row + 1) + row * 3) % 13 - 6 } }
            let inputs = integers.map { $0.map { Float($0) / 8 } }
            switch dimension {
            case 512: verify(try inputs.map(Vector512Optimized.init), integerRows: integers)
            case 768: verify(try inputs.map(Vector768Optimized.init), integerRows: integers)
            default: verify(try inputs.map(Vector1536Optimized.init), integerRows: integers)
            }
        }
    }
}

extension BenchmarkSelectionContractTests {
    @MainActor func testRepresentativeCaseBoundaryAllFamilies() async throws {
        for dim in [512,768,1536] {
            for (family, suite, id) in [
                ("distance", DistanceBench.self as any BenchmarkSuite.Type, "dist.euclidean.\(dim).optimized"),
                ("normalize", NormalizationBench.self as any BenchmarkSuite.Type, "normalize.success.\(dim).optimized"),
                ("batch", BatchBench.self as any BenchmarkSuite.Type, "batch.euclidean2.\(dim).N2.generic.sequential")
            ] {
                let value = try options(["--suites", family, "--dims", String(dim), "--batch-ns", "2", "--filter", id, "--repeats", "1", "--min-time", "0"])
                let ledger = CaseLedger()
                let results = await CaseExecution.run(suite.catalog(options: value, progress: progress), options: value, progress: progress, suite: family, onCase: ledger.record)
                XCTAssertEqual(ledger.snapshot, [id]); XCTAssertEqual(results.map(\.name), [id]); XCTAssertNil(results[0].executionFailure)
                // The catalog's own record is attached, and the harness-reported unit count agrees with it.
                XCTAssertEqual(results[0].boundary, Boundaries.lookup(id, runSeed: 0), id)
                XCTAssertEqual(results[0].unitCount, results[0].boundary?.unitsPerInvocation, id)
                XCTAssertEqual(results[0].sampleNsPerInvocation.count, 1, id); XCTAssertEqual(results[0].sampleIterations, [1], id)
                XCTAssertNotNil(results[0].warmupIterations, id)
            }
        }
        for (family, suite, ids) in [
            ("memory", MemoryBench.self as any BenchmarkSuite.Type, ["mem.alloc.aligned.1", "mem.copy.1", "mem.fill.1", "mem.pool.acquire.1"]),
            ("noop", NoopBench.self as any BenchmarkSuite.Type, ["noop.scalar-add"])
        ] {
            let value = try options(["--suites", family, "--dims", "1", "--repeats", "1", "--min-time", "0"])
            let ledger = CaseLedger()
            let results = await CaseExecution.run(suite.catalog(options: value, progress: progress), options: value, progress: progress, suite: family, onCase: ledger.record)
            XCTAssertEqual(ledger.snapshot, ids); XCTAssertEqual(results.map(\.name), ids)
            XCTAssertTrue(results.allSatisfy { $0.executionFailure == nil })
            for result in results {
                XCTAssertEqual(result.unitCount, result.name.contains("copy") || result.name.contains("fill") ? 1024 : 1)
                XCTAssertEqual(result.unitCount, result.boundary?.unitsPerInvocation, result.name)
            }
        }
    }
    func testCSVFixtureForIndependentParser() throws {
        let text = "comma,quote\"\nline\rreturn"
        let metadata = BenchMetadata(package: "VectorCore", packageVersion: "fixture", gitSHA: nil, date: "fixture",
            os: "fixture", arch: "arm64", cpuCores: 1, deviceModel: text, swiftVersion: text,
            buildConfiguration: VectorCoreCompiledFeatures.configuration, deviceTag: text, binarySizes: .init(benchExecutableBytes: 1, vectorCoreLibraryBytes: nil),
            suites: ["fixture"], dims: [512], runSeed: 0, runLabel: text,
            flags: .init(preferSoA: false, useMixedPrecision: false, abCompare: false, abOnly: false, useCKernels: VectorCoreCompiledFeatures.useCKernels),
            thresholds: nil, filters: nil, coreConfiguration: VectorCoreCompiledFeatures.configuration,
            benchmarkConfiguration: BenchmarkCompiledFeatures.configuration, selectedCaseIDs: [text],
            runtimeToolchain: .init(value: text))
        var boundary = Boundaries.fixtureInjected(name: text)
        boundary.route = text; boundary.operation = "op," + text; boundary.unitsPerInvocation = 5
        let cases = try JSONWriter.toCases([BenchResult(name: text, iterations: 2, totalNanoseconds: 50, unitCount: 5, samples: 2,
            meanNsPerOp: 25, medianNsPerOp: 25, p90NsPerOp: 30,
            boundary: boundary, sampleNsPerInvocation: [24, 26], sampleIterations: [1, 1], warmupIterations: 3)])
        let unavailable = [UnavailableCase(name: text, boundary: boundary, reason: text)]
        let csvOutputPath = ProcessInfo.processInfo.environment["V02_CSV_OUTPUT"]
            ?? FileManager.default.temporaryDirectory.appendingPathComponent("V02-csv-" + UUID().uuidString + ".csv").path
        try CSVWriter.write(cases: cases, metadata: metadata, failures: [.init(name: text, reason: text), .init(name: text, reason: text,
            boundary: boundary)], unavailable: unavailable,
            to: csvOutputPath)
        XCTAssertTrue(FileManager.default.fileExists(atPath: csvOutputPath))
        print("V02_CSV_FIXTURE_COMPLETE \(csvOutputPath)")
    }
}

extension BenchmarkSelectionContractTests {
    func testMeasuredMatrixStatisticsFromExactObservedOutput() throws {
        func verify<V: UnifiedVectorBuffer>(_ vectors: [V], inputs: [[Float]], dim: Int, options: ValidatedOptions) throws {
            var observations = 0
            var output: [Float] = []
            let measured = try MatrixDistanceBench.measure(vectors, name: "matrix.euclidean2.\(dim).N3.optimized", options: options, onMeasuredOutput: {
                observations += 1; output = $0
            })
            XCTAssertEqual(observations, 1); XCTAssertEqual(output.count, 9)
            var absoluteErrors: [Double] = [], relativeErrors: [Double] = []
            for row in 0..<3 { for column in 0..<3 {
                var reference = 0.0
                for coordinate in 0..<dim {
                    let difference = Double(inputs[row][coordinate]) - Double(inputs[column][coordinate])
                    reference += difference * difference
                }
                XCTAssertTrue(reference.isFinite); XCTAssertTrue(output[row * 3 + column].isFinite)
                let error = abs(Double(output[row * 3 + column]) - reference)
                absoluteErrors.append(error); relativeErrors.append(error / max(abs(reference), Correctness.referenceFloor))
            } }
            let stats = try XCTUnwrap(measured.correctness)
            XCTAssertEqual(stats.status, .valid); XCTAssertEqual(stats.samples, 9)
            XCTAssertEqual(stats.maxAbsError, absoluteErrors.max()!)
            XCTAssertEqual(stats.meanAbsError, absoluteErrors.reduce(0,+) / 9)
            XCTAssertEqual(stats.maxRelError, relativeErrors.max()!)
            XCTAssertEqual(stats.meanRelError, relativeErrors.reduce(0,+) / 9)
            // The diagonal references are zero: the matrix route is gated on absolute error (T11, CorrectnessPolicy), never on a
            // relative error against a zero reference.
            let violations = evaluateCorrectnessWithPolicy(results: [measured], overrideRel: nil, overrideAbs: nil)
            let limit = try XCTUnwrap(CorrectnessPolicy.rule(forCaseName: "matrix.euclidean2.\(dim).N3.optimized").maxAbs)
            XCTAssertEqual(!violations.isEmpty, absoluteErrors.max()! > limit)
            XCTAssertTrue(violations.isEmpty, "\(violations)")
        }
        for dim in [512,768,1536] {
            let options = try self.options(["--suites", "matrix", "--dims", String(dim), "--batch-ns", "3", "--repeats", "1", "--samples", "2", "--min-time", "0"])
            let inputs = (0..<3).map { InputFactory.randomArray(count: dim, seed: InputSeeds.matrix(dim: dim, row: $0, runSeed: 0)) }
            switch dim {
            case 512: try verify(inputs.map(Vector512Optimized.init), inputs: inputs, dim: dim, options: options)
            case 768: try verify(inputs.map(Vector768Optimized.init), inputs: inputs, dim: dim, options: options)
            default: try verify(inputs.map(Vector1536Optimized.init), inputs: inputs, dim: dim, options: options)
            }
        }
    }
}

// MARK: - V04 output consumption

final class BenchmarkOutputFoldContractTests: XCTestCase {
    /// The observation the matrix body made before V04: only output[0] and output[N*N-1].
    private func baselineObservation(_ output: [Float]) -> [UInt32] { [output[0].bitPattern, output[output.count - 1].bitPattern] }

    private func flipped(_ output: [Float], at index: Int) -> [Float] {
        var copy = output
        copy[index] = Float(bitPattern: copy[index].bitPattern ^ 1)
        return copy
    }

    /// Generic over the row type so the real fixture generator and the real API produce the output.
    private func realOutput<V: UnifiedVectorBuffer>(_ vectors: [V]) -> [Float] {
        var output = [Float](repeating: 0, count: vectors.count * vectors.count)
        MatrixDistance.euclideanSquaredMatrix(queries: vectors, candidates: vectors, into: &output)
        return output
    }

    private func checkPerturbation(_ output: [Float], n: Int, label: String) {
        let first = 0, last = output.count - 1, middle = output.count / 2
        let base = MatrixDistanceBench.consumedValue(output)
        XCTAssertEqual(base, MatrixDistanceBench.consumedValue(output), label)
        // Each position is changed individually; the consumer used by the timed body must see every change.
        for (position, index) in [("first", first), ("middle", middle), ("last", last), ("off-diagonal (0,1)", 1), ("off-diagonal (n-1,0)",
            (n - 1) * n)] {
            let changed = flipped(output, at: index)
            XCTAssertNotEqual(MatrixDistanceBench.consumedValue(changed), base, "\(label) \(position) index \(index)")
            // Restoring the value restores the consumer value.
            XCTAssertEqual(MatrixDistanceBench.consumedValue(flipped(changed, at: index)), base, "\(label) restore \(position)")
        }
        // Mutation control: the pre-V04 observation is blind to the middle and off-diagonal positions, so
        // these assertions fail for it. This proves the assertions above can detect weak consumption.
        let weak = baselineObservation(output)
        XCTAssertNotEqual(baselineObservation(flipped(output, at: first)), weak)
        XCTAssertNotEqual(baselineObservation(flipped(output, at: last)), weak)
        XCTAssertEqual(baselineObservation(flipped(output, at: middle)), weak, "\(label): control must be blind to the middle entry")
        XCTAssertEqual(baselineObservation(flipped(output, at: 1)), weak, "\(label): control must be blind to off-diagonal entries")
    }

    func testMatrixConsumerSeesFirstMiddleLastAndOffDiagonalAtRepresentativeSizes() throws {
        for n in [17, 65] {
            let vectors = try (0..<n).map { try Vector512Optimized(MatrixDistanceBench.inputs(dim: 512, row: $0, runSeed: 0)) }
            let output = realOutput(vectors)
            XCTAssertEqual(output.count, n * n)
            // For odd N the middle index floor(N*N/2) is a diagonal entry (k*N + k with k = (N-1)/2).
            let middle = output.count / 2
            XCTAssertEqual(middle / n, middle % n, "odd N middle entry is on the diagonal")
            checkPerturbation(output, n: n, label: "512 N\(n)")
        }
    }

    func testMatrixConsumerSeesEveryPositionAtAllWidthsSmallN() throws {
        for dim in [512, 768, 1536] {
            let n = 3
            let output: [Float]
            switch dim {
            case 512: output = realOutput(try (0..<n).map { try Vector512Optimized(MatrixDistanceBench.inputs(dim: dim, row: $0,
                runSeed: 2)) })
            case 768: output = realOutput(try (0..<n).map { try Vector768Optimized(MatrixDistanceBench.inputs(dim: dim, row: $0,
                runSeed: 2)) })
            default: output = realOutput(try (0..<n).map { try Vector1536Optimized(MatrixDistanceBench.inputs(dim: dim, row: $0,
                runSeed: 2)) })
            }
            checkPerturbation(output, n: n, label: "\(dim) N3")
            // Exhaustive for the small matrix: every single position is observed.
            let base = MatrixDistanceBench.consumedValue(output)
            for index in output.indices { XCTAssertNotEqual(MatrixDistanceBench.consumedValue(flipped(output, at: index)), base,
                "\(dim) index \(index)") }
        }
    }

    func testFoldSensitivityToSignedZeroNaNPayloadAndOrder() {
        func fold(_ values: [Float]) -> UInt64 { values.withUnsafeBufferPointer { OutputFold.fold($0) } }
        XCTAssertEqual(fold([]), OutputFold.offsetBasis)
        XCTAssertNotEqual(fold([0.0]), fold([-0.0]))
        XCTAssertNotEqual(fold([Float(bitPattern: 0x7FC00000)]), fold([Float(bitPattern: 0x7FC00001)]))
        XCTAssertNotEqual(fold([1, 2]), fold([2, 1]))
        XCTAssertEqual(fold([1, 2, 3]), fold([1, 2, 3]))
    }

    private func checkNormalizeFold(_ coordinates: [Float], label: String) {
        func fold(_ values: [Float]) -> UInt64 { values.withUnsafeBufferPointer { OutputFold.fold($0) } }
        let base = fold(coordinates)
        // The pre-V04 observation read only unit[0]; it is blind to every other coordinate.
        let oldObservation = coordinates[0].bitPattern
        for index in [0, 1, coordinates.count / 2, coordinates.count - 1] {
            var changed = coordinates; changed[index] = Float(bitPattern: changed[index].bitPattern ^ 1)
            XCTAssertNotEqual(fold(changed), base, "\(label) coordinate \(index)")
            if index != 0 { XCTAssertEqual(changed[0].bitPattern, oldObservation,
                "\(label): old observation is blind to coordinate \(index)") }
        }
    }

    func testNormalizeSuccessFoldCoversEveryCoordinateAllWidthsAndVariants() throws {
        let optimized512 = try Vector512Optimized(repeating: 2.0).normalized().get().withUnsafeBufferPointer { Array($0) }
        let optimized768 = try Vector768Optimized(repeating: 2.0).normalized().get().withUnsafeBufferPointer { Array($0) }
        let optimized1536 = try Vector1536Optimized(repeating: 2.0).normalized().get().withUnsafeBufferPointer { Array($0) }
        let generic512 = try Vector<Dim512>(repeating: 2.0).normalized().get().withUnsafeBufferPointer { Array($0) }
        let generic768 = try Vector<Dim768>(repeating: 2.0).normalized().get().withUnsafeBufferPointer { Array($0) }
        let generic1536 = try Vector<Dim1536>(repeating: 2.0).normalized().get().withUnsafeBufferPointer { Array($0) }
        for (label, coordinates, count) in [("optimized512", optimized512, 512), ("optimized768", optimized768, 768), ("optimized1536",
            optimized1536, 1536),
                                            ("generic512", generic512, 512), ("generic768", generic768, 768), ("generic1536", generic1536,
                                                1536)] {
            XCTAssertEqual(coordinates.count, count, label)
            checkNormalizeFold(coordinates, label: label)
        }
    }
}

// MARK: - V04 sink confinement and measurement seams

final class BenchmarkMeasurementSeamContractTests: XCTestCase {
    private let progress = BenchmarkProgressReporter(format: .none)

    private func options(_ arguments: [String]) throws -> ValidatedOptions {
        try ValidatedOptions(parseCLI(arguments: arguments, environment: [:]))
    }

    private final class PhaseLog: MeasurementPhaseObserver, @unchecked Sendable {
        private let lock = NSLock()
        private var events: [String] = []
        func begin(_ phase: MeasurementPhase, caseName: String) { lock.withLock { events.append("begin:\(phase.rawValue):\(caseName)") } }
        func end(_ phase: MeasurementPhase, caseName: String) { lock.withLock { events.append("end:\(phase.rawValue):\(caseName)") } }
        var snapshot: [String] { lock.withLock { events } }
    }

    @MainActor func testPhaseObserverOrderAndRecorderFromActualCase() async throws {
        let value = try options(["--suites", "dot", "--dims", "512", "--filter", "dot.512.optimized", "--repeats", "2", "--samples", "3",
            "--min-time", "0"])
        let log = PhaseLog()
        let results = await MeasurementPhases.$observer.withValue(log) {
            await CaseExecution.run(DotProductBench.catalog(options: value, progress: progress), options: value, progress: progress,
                suite: "dot")
        }
        let result = try XCTUnwrap(results.first); XCTAssertNil(result.executionFailure)
        XCTAssertEqual(result.sampleNsPerInvocation.count, 3); XCTAssertEqual(result.sampleIterations, [2, 2, 2])
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(result.warmupIterations), 1)
        // Per-sample averages reproduce the reported total: sum(ns/invocation * iterations) == totalNanoseconds (within float rounding).
        let reconstructed = zip(result.sampleNsPerInvocation, result.sampleIterations).reduce(0.0) { $0 + $1.0 * Double($1.1) }
        XCTAssertEqual(reconstructed, Double(result.totalNanoseconds), accuracy: Double(result.totalNanoseconds) * 1e-9 + 1)
        let events = log.snapshot
        let name = "dot.512.optimized"
        XCTAssertEqual(events.first, "begin:setup:\(name)"); XCTAssertEqual(events.last, "end:validation:\(name)")
        XCTAssertEqual(events.filter { $0 == "begin:timed:\(name)" }.count, 3)
        XCTAssertEqual(events.filter { $0 == "end:timed:\(name)" }.count, 3)
        XCTAssertEqual(events.filter { $0 == "begin:warmup:\(name)" }.count, 1)
        XCTAssertLessThan(try XCTUnwrap(events.firstIndex(of: "begin:warmup:\(name)")),
            try XCTUnwrap(events.firstIndex(of: "begin:timed:\(name)")))
        XCTAssertEqual(events.filter { $0 == "begin:validation:\(name)" }.count, 1)
        // Balanced: every begin has exactly one matching end, in strict nesting order (never two phases open at once).
        var open: String?
        for event in events {
            let parts = event.split(separator: ":").map(String.init)
            if parts[0] == "begin" {
                XCTAssertNil(open, "begin \(parts[1]) while \(open ?? "") is open"); open = parts[1]
            } else {
                XCTAssertEqual(open, parts[1], "end \(parts[1]) without matching begin"); open = nil
            }
        }
        XCTAssertNil(open, "no phase left open")
        XCTAssertEqual(events.filter { $0.hasPrefix("begin:") }.count, events.filter { $0.hasPrefix("end:") }.count)
        // Without an observer the same case still reports its raw samples.
        let plain = await CaseExecution.run(DotProductBench.catalog(options: value, progress: progress), options: value,
            progress: progress, suite: "dot")
        XCTAssertEqual(plain.first?.sampleNsPerInvocation.count, 3)
    }

    @MainActor func testResultNameAndUnitContractsFailExplicitly() async throws {
        let value = try options(["--suites", "noop", "--repeats", "1", "--min-time", "0"])
        let boundary = try XCTUnwrap(Boundaries.lookup("noop.scalar-add", runSeed: 0))
        let renamed = BenchmarkCase("noop.scalar-add", boundary: boundary) { [BenchResult(name: "other", iterations: 1,
            totalNanoseconds: 5)] }
        let wrongUnits = BenchmarkCase("noop.scalar-add", boundary: boundary) { [BenchResult(name: "noop.scalar-add", iterations: 1,
            totalNanoseconds: 5, unitCount: 7)] }
        for item in [renamed, wrongUnits] {
            let results = await CaseExecution.run([item], options: value, progress: progress, suite: "noop")
            XCTAssertEqual(results.count, 1)
            XCTAssertNotNil(results[0].executionFailure)
            XCTAssertEqual(results[0].name, "noop.scalar-add"); XCTAssertEqual(results[0].boundary, boundary)
        }
    }

    @MainActor func testUnavailableEntriesAreRecordedNeverInvoked() async throws {
        let value = try options(["--suites", "noop", "--repeats", "1", "--min-time", "0"])
        let boundary = try XCTUnwrap(Boundaries.lookup("noop.scalar-add", runSeed: 0))
        let counter = ExcludedCounter()
        let item = BenchmarkCase("noop.scalar-add", boundary: boundary, availability: .unavailable(reason: "route does not exist")) {
            await counter.record(); return []
        }
        let output = await CaseExecution.execute([item], options: value, progress: progress, suite: "noop")
        XCTAssertTrue(output.results.isEmpty)
        XCTAssertEqual(output.unavailable, [UnavailableCase(name: "noop.scalar-add", boundary: boundary, reason: "route does not exist")])
        let invoked = await counter.count
        XCTAssertEqual(invoked, 0)
    }

    private enum UnavailableOnlySuite: BenchmarkSuite {
        static var name: String { "unavailable-only" }
        static func catalog(options: ValidatedOptions, progress: BenchmarkProgressReporter) -> [BenchmarkCase] {
            let boundary = Boundaries.lookup("noop.scalar-add", runSeed: 0)!
            return [BenchmarkCase("noop.scalar-add", boundary: boundary, availability: .unavailable(reason: "fixture")) { [] }]
        }
        static func run(options: ValidatedOptions, progress: BenchmarkProgressReporter) async -> [BenchResult] { [] }
    }

    private enum MixedAvailabilitySuite: BenchmarkSuite {
        static var name: String { "mixed-availability" }
        static func catalog(options: ValidatedOptions, progress: BenchmarkProgressReporter) -> [BenchmarkCase] {
            let boundary = Boundaries.lookup("noop.scalar-add", runSeed: 0)!
            var missing = boundary; missing.operation = "control.unavailable-route"
            return [
                BenchmarkCase("noop.scalar-add", boundary: boundary) {
                    [BenchResult(name: "noop.scalar-add", iterations: 1, totalNanoseconds: 100, correctness: Correctness.scalar(current: 1,
                        reference: 1))]
                },
                BenchmarkCase("noop.missing-route", boundary: missing, availability: .unavailable(reason: "route missing")) { [] }
            ]
        }
        static func run(options: ValidatedOptions, progress: BenchmarkProgressReporter) async -> [BenchResult] { [] }
    }

    @MainActor func testRunnerExitContractForUnavailableEntries() async throws {
        var only = CLIOptions(); only.suites = ["unavailable-only"]; only.repeats = 1
        let code = await BenchmarkRunner.run(options: only, registry: ["unavailable-only": UnavailableOnlySuite.self])
        XCTAssertEqual(code, 1)
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("V04-mixed-" + UUID().uuidString + ".json").path
        var mixed = CLIOptions(); mixed.suites = ["mixed-availability"]; mixed.repeats = 1; mixed.format = "json"
        mixed.outputPath = path; mixed.abCompare = false
        let mixedCode = await BenchmarkRunner.run(options: mixed, registry: ["mixed-availability": MixedAvailabilitySuite.self])
        XCTAssertEqual(mixedCode, 0)
        let run = try JSONDecoder().decode(BenchRun.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        XCTAssertEqual(run.results.map(\.name), ["noop.scalar-add"])
        XCTAssertEqual(run.unavailable.map(\.name), ["noop.missing-route"]); XCTAssertEqual(run.unavailable.first?.reason, "route missing")
        XCTAssertEqual(run.metadata.unavailableCaseIDs, ["noop.missing-route"])
        XCTAssertEqual(run.metadata.selectedCaseIDs, ["noop.scalar-add"])
        XCTAssertEqual(run.metadata.schemaVersion, 3); XCTAssertEqual(run.metadata.measurementBoundaryVersion, "V04-boundaries-1")
        XCTAssertTrue(run.metadata.harnessCalibration.hasPrefix("not used (fixed --repeats 1"), run.metadata.harnessCalibration)
        XCTAssertEqual(run.metadata.statisticUnit, "per-sample-average-ns-per-operation")
        XCTAssertEqual(run.metadata.invocationStatisticUnit, "per-sample-average-ns-per-invocation")
        XCTAssertEqual(run.results.first?.boundary?.consumer, "scalar-sink-v1")
    }
}

extension BenchmarkMeasurementSeamContractTests {
    private struct MultiResultSuite {
        static func cases(count: Int) -> [BenchmarkCase] {
            let boundary = Boundaries.lookup("noop.scalar-add", runSeed: 0)!
            return [BenchmarkCase("noop.scalar-add", boundary: boundary) {
                (0..<count).map { _ in BenchResult(name: "noop.scalar-add", iterations: 1, totalNanoseconds: 5) }
            }]
        }
    }

    @MainActor func testProductionCaseMustReturnExactlyOneResult() async throws {
        let value = try options(["--suites", "noop", "--repeats", "1", "--min-time", "0"])
        for count in [0, 2] {
            let results = await CaseExecution.run(MultiResultSuite.cases(count: count), options: value, progress: progress, suite: "noop")
            XCTAssertEqual(results.count, 1, "count \(count)")
            XCTAssertTrue(results[0].executionFailure?.contains("exactly one") == true, "count \(count)")
            XCTAssertEqual(results[0].boundary, Boundaries.lookup("noop.scalar-add", runSeed: 0))
        }
        // Duplicate result names are rejected by the report even when A/B comparison is off.
        let boundary = try XCTUnwrap(Boundaries.lookup("noop.scalar-add", runSeed: 0))
        let duplicate = BenchResult(name: "noop.scalar-add", iterations: 1, totalNanoseconds: 5, boundary: boundary)
        XCTAssertThrowsError(try JSONWriter.toCases([duplicate, duplicate])) { XCTAssertTrue("\($0)".contains("Duplicate case ID")) }
    }

    func testFixtureInjectedScopeIsRejectedInProductionRegistry() throws {
        let injected = BenchmarkCase("fixture", boundary: Boundaries.fixtureInjected(name: "fixture"), acceptsAnyResultName: true) { [] }
        let production = BenchmarkCase("noop.scalar-add", boundary: try XCTUnwrap(Boundaries.lookup("noop.scalar-add", runSeed: 0))) { [] }
        XCTAssertNotNil(injectedScopeProblem(cases: [injected, production], injectedRegistry: false))
        XCTAssertNil(injectedScopeProblem(cases: [injected, production], injectedRegistry: true))
        XCTAssertNil(injectedScopeProblem(cases: [production], injectedRegistry: false))
    }

    func testPerOperationStatisticsFollowDeclaredWorkUnit() throws {
        let stats = try XCTUnwrap(PerOperationStatistics(sampleNsPerInvocation: [24, 26], unitsPerInvocation: 5))
        XCTAssertEqual(stats.samples, [4.8, 5.2]); XCTAssertEqual(stats.median, 5.0, accuracy: 1e-12); XCTAssertEqual(stats.mean, 5.0,
            accuracy: 1e-12)
        XCTAssertEqual(stats.p90, 5.2, accuracy: 1e-12); XCTAssertEqual(stats.stddev, 0.2 * 2.0.squareRoot(), accuracy: 1e-12)
        XCTAssertEqual(stats.rsdPercent, stats.stddev / 5.0 * 100, accuracy: 1e-12)
        XCTAssertNil(PerOperationStatistics(sampleNsPerInvocation: [],
            unitsPerInvocation: 5)); XCTAssertNil(PerOperationStatistics(sampleNsPerInvocation: [1], unitsPerInvocation: 0))
        // Through the report: matrix units are produced entries (N*N), not invocations.
        let boundary = try XCTUnwrap(Boundaries.lookup("matrix.euclidean2.512.N17.optimized", runSeed: 0))
        let result = BenchResult(name: "matrix.euclidean2.512.N17.optimized", iterations: 4, totalNanoseconds: 4_000, unitCount: 289,
            samples: 2,
                                 boundary: boundary, sampleNsPerInvocation: [1_000, 2_890], sampleIterations: [2, 2])
        let reported = try XCTUnwrap(JSONWriter.toCases([result]).first)
        XCTAssertEqual(try XCTUnwrap(reported.sampleNsPerOperation)[1], 10.0, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(reported.medianNsPerOperation), (1_000.0 / 289 + 10) / 2, accuracy: 1e-12)
        XCTAssertNotNil(reported.stddevNsPerOperation); XCTAssertNotNil(reported.rsdPercentPerOperation)
        // A real measured case reports the new fields end to end.
    }
}

// MARK: - V04 phase C: the opt-in boundary suite

final class BenchmarkBoundarySuiteContractTests: XCTestCase {
    private let progress = BenchmarkProgressReporter(format: .none)

    private func options(_ arguments: [String]) throws -> ValidatedOptions {
        try ValidatedOptions(parseCLI(arguments: arguments, environment: [:]))
    }

    /// Independent expected IDs: nothing here is read from the catalog or the parser.
    private func expectedIDs(dims: [Int], counts: [Int]) -> (executable: Set<String>, unavailable: Set<String>) {
        var executable = Set<String>(), unavailable = Set<String>()
        for d in dims {
            for n in counts {
                for variant in ["aos-fp32-reused", "aos-fp32-setup", "soa-fp32-reused", "soa-fp32-setup", "aos-fp16-reused",
                    "aos-fp16-setup", "consumer-only"] {
                    executable.insert("batch.euclidean2.\(d).N\(n).\(variant)")
                }
                executable.insert("matrix.euclidean2.\(d).N\(n).consumer-only")
                for variant in ["soa-fp16-reused", "soa-fp16-setup"] { unavailable.insert("batch.euclidean2.\(d).N\(n).\(variant)") }
            }
        }
        return (executable, unavailable)
    }

    func testIndependentSelectionTableAllWidthsAndCounts() throws {
        for (dims, counts) in [([512, 768, 1536], [2, 3]), ([512], [257, 4096]), ([768, 1536], [100])] {
            let value = try options(["--suites", "boundary", "--dims", dims.map(String.init).joined(separator: ","),
                                     "--batch-ns", counts.map(String.init).joined(separator: ",")])
            let catalog = BoundaryBench.catalog(options: value, progress: progress)
            let expected = expectedIDs(dims: dims, counts: counts)
            XCTAssertEqual(catalog.count, Set(catalog.map(\.name)).count)
            XCTAssertEqual(Set(catalog.filter { $0.availability == .available }.map(\.name)), expected.executable)
            XCTAssertEqual(Set(catalog.filter { $0.availability != .available }.map(\.name)), expected.unavailable)
            for item in catalog {
                XCTAssertNil(item.boundary.validationError(), item.name)
                if case .unavailable(let reason) = item.availability {
                    // Every width carries the reason, which names why no SoA-FP16 squared-Euclidean route exists.
                    XCTAssertTrue(reason.contains("sqrt") && reason.contains("SoA-FP16"), item.name)
                    XCTAssertTrue(item.boundary.route.hasPrefix("unavailable: "), item.name)
                    // Nothing is constructed or prepared for an entry that is never invoked.
                    XCTAssertEqual(item.boundary.inclusions.inputConstruction, .notApplicable, item.name)
                    XCTAssertEqual(item.boundary.inclusions.inputPreparation, .notApplicable, item.name)
                    XCTAssertTrue(item.boundary.notes.contains { $0.contains("unavailable; never invoked") }, item.name)
                }
            }
        }
    }

    func testBoundarySuiteIsNeverDefaultOrAutoSelected() throws {
        XCTAssertFalse(CLIOptions().suites.contains("boundary"))
        // Opt-in flags and --ab-only do not select it, and do not add its entries to the default batch catalog.
        let value = try options(["--suites", "batch", "--prefer-soa", "--use-mixed-precision", "--ab-only", "--dims", "512", "--batch-ns",
            "100"])
        XCTAssertEqual(value.suites, ["batch"])
        for item in BatchBench.catalog(options: value, progress: progress) {
            XCTAssertFalse(item.name.contains("aos-fp32") || item.name.contains("-reused") || item.name.contains("-setup")
                || item.name.contains("consumer-only"), item.name)
        }
        XCTAssertThrowsError(try options(["--suites", "boundary", "--dims", "100"]))
        XCTAssertThrowsError(try options(["--suites", "boundary", "--batch-ns", String(Int.max)]))
    }

    private final class Ledger: @unchecked Sendable {
        private let lock = NSLock(); private var names: [String] = []
        func record(_ name: String) { lock.withLock { names.append(name) } }
        var snapshot: [String] { lock.withLock { names } }
    }

    @MainActor func testExcludedAndUnavailableRoutesAreNeverInvoked() async throws {
        let value = try options(["--suites", "boundary", "--dims", "512", "--batch-ns", "3", "--filter", "*.aos-fp32-reused,*soa-fp16*",
                                 "--repeats", "1", "--min-time", "0"])
        let ledger = Ledger()
        let output = await CaseExecution.execute(BoundaryBench.catalog(options: value, progress: progress), options: value,
            progress: progress,
                                                 suite: "boundary", onCase: ledger.record)
        // Only the selected executable case ran; the unavailable entries are recorded without running.
        XCTAssertEqual(ledger.snapshot, ["batch.euclidean2.512.N3.aos-fp32-reused"])
        XCTAssertEqual(output.results.map(\.name), ["batch.euclidean2.512.N3.aos-fp32-reused"])
        XCTAssertEqual(Set(output.unavailable.map(\.name)), Set(["batch.euclidean2.512.N3.soa-fp16-reused",
            "batch.euclidean2.512.N3.soa-fp16-setup"]))
        XCTAssertTrue(output.unavailable.allSatisfy { $0.reason == Boundaries.soaFP16UnavailableReason })
    }

    @MainActor func testOnlyUnavailableSelectionExitsOne() async throws {
        var input = CLIOptions()
        input.suites = ["boundary"]; input.dims = [512]; input.batchNs = [2]; input.filters = ["*soa-fp16*"]; input.repeats = 1
        let code = await BenchmarkRunner.run(options: input)
        XCTAssertEqual(code, 1)
        // The exit path explains itself: a header and one line per entry with its reason.
        let value = try options(["--suites", "boundary", "--dims", "512", "--batch-ns", "2", "--filter", "*soa-fp16*"])
        let selected = BoundaryBench.catalog(options: value, progress: progress).filter { value.caseFilter.matches($0.name) }
        XCTAssertEqual(selected.count, 2)
        let message = try XCTUnwrap(unavailableOnlyMessage(cases: selected))
        XCTAssertTrue(message.hasPrefix("Invalid configuration: no executable selected cases"))
        for item in selected {
            guard case .unavailable(let reason) = item.availability else { return XCTFail(item.name) }
            XCTAssertTrue(message.contains("UNAVAILABLE \(item.name): \(reason)\n"), item.name)
        }
        // Not printed when something is executable, or when nothing is selected at all.
        let mixed = BoundaryBench.catalog(options: value, progress: progress)
        XCTAssertNil(unavailableOnlyMessage(cases: mixed)); XCTAssertNil(unavailableOnlyMessage(cases: []))
    }

    @MainActor func testExecutableCompanionsReportCorrectnessAndNoABRows() async throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("V04-boundary-" + UUID().uuidString + ".json").path
        var input = CLIOptions()
        input.suites = ["boundary"]; input.dims = [512, 768, 1536]; input.batchNs = [3]; input.repeats = 256; input.samples = 2
        // A consumer-only sample of a few floats is below the clock resolution at one repeat ("Unusable benchmark clock interval").
        input.minTimeSeconds = 0; input.format = "json"; input.outputPath = path
        let code = await BenchmarkRunner.run(options: input)
        XCTAssertEqual(code, 0)
        let run = try JSONDecoder().decode(BenchRun.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        XCTAssertNil(run.abComparisons); XCTAssertTrue(run.abMissingSides.isEmpty)
        XCTAssertEqual(Set(run.results.map(\.name)), expectedIDs(dims: [512, 768, 1536], counts: [3]).executable)
        XCTAssertEqual(Set(run.unavailable.map(\.name)), expectedIDs(dims: [512, 768, 1536], counts: [3]).unavailable)
        XCTAssertEqual(Set(run.metadata.unavailableCaseIDs), Set(run.unavailable.map(\.name)))
        XCTAssertEqual(Set(run.metadata.selectedCaseIDs), Set(run.results.map(\.name)))
        for result in run.results {
            let correctness = try XCTUnwrap(result.correctness, result.name)
            XCTAssertEqual(correctness.status, .valid, result.name); XCTAssertGreaterThan(correctness.samples, 0, result.name)
            XCTAssertNotNil(correctness.maxRelError, result.name); XCTAssertNotNil(correctness.meanAbsError, result.name)
            let boundary = try XCTUnwrap(result.boundary, result.name)
            XCTAssertEqual(boundary.unitsPerInvocation, result.unitCount, result.name)
            XCTAssertEqual(boundary.consumer, "u64-bitfold-sink-v1"); XCTAssertNil(boundary.comparison, result.name)
            XCTAssertEqual(result.name.hasSuffix("-setup"), boundary.timingScope == .setupInclusive, result.name)
            XCTAssertEqual(result.name.hasSuffix("consumer-only"), boundary.timingScope == .consumerOnly, result.name)
            if result.name.contains("fp16") {
                // The reduced-precision error is measured on this fixture and is nonzero for generic inputs.
                XCTAssertGreaterThan(try XCTUnwrap(correctness.maxAbsError), 0, result.name)
            }
        }
    }

    func testBoundaryRecordsAreHonestAboutScopeAndWork() throws {
        func boundary(_ name: String) throws -> MeasurementBoundary { try XCTUnwrap(Boundaries.lookup(name, runSeed: 0), name) }
        for d in [512, 768, 1536] {
            let reused = try boundary("batch.euclidean2.\(d).N257.soa-fp32-reused"),
                setup = try boundary("batch.euclidean2.\(d).N257.soa-fp32-setup")
            XCTAssertEqual(reused.timingScope, .reusedInput); XCTAssertEqual(setup.timingScope, .setupInclusive)
            XCTAssertEqual(reused.inclusions.inputPreparation, .excluded); XCTAssertEqual(setup.inclusions.routeInternalPacking, .included)
            XCTAssertEqual(setup.inclusions.routeInternalAllocation, .included)
            let half = try boundary("batch.euclidean2.\(d).N257.aos-fp16-setup")
            XCTAssertEqual(half.inclusions.inputPreparation, .included); XCTAssertEqual(half.inclusions.callerOutputAllocation, .included)
            XCTAssertEqual(try boundary("batch.euclidean2.\(d).N257.aos-fp16-reused").inclusions.inputPreparation, .excluded)
            XCTAssertEqual(try boundary("batch.euclidean2.\(d).N257.aos-fp32-setup").inclusions.callerOutputAllocation, .included)
            XCTAssertEqual(try boundary("batch.euclidean2.\(d).N257.aos-fp32-reused").inclusions.callerOutputAllocation, .excluded)
            // Direct kernels: no provider scheduling or observation.
            XCTAssertEqual(try boundary("batch.euclidean2.\(d).N257.aos-fp32-reused").inclusions.providerScheduling, .notApplicable)
            let consumer = try boundary("batch.euclidean2.\(d).N257.consumer-only")
            XCTAssertEqual(consumer.timingScope, .consumerOnly); XCTAssertEqual(consumer.inclusions.kernelWork, .excluded)
            XCTAssertEqual(try boundary("matrix.euclidean2.\(d).N17.consumer-only").unitsPerInvocation, 289)
            // Different scopes have different identity even for the same underlying fixture.
            XCTAssertNotEqual(reused.canonicalJSON(), setup.canonicalJSON())
            XCTAssertEqual(reused.fixture.seeds, setup.fixture.seeds)
        }
        // defaultEligible records the existing predicates (Q1) and is only present on boundary-suite SoA/FP16 entries.
        XCTAssertEqual(try boundary("batch.euclidean2.512.N100.soa-fp32-reused").defaultEligible, true)
        XCTAssertEqual(try boundary("batch.euclidean2.512.N99.soa-fp32-reused").defaultEligible, false)
        XCTAssertEqual(try boundary("batch.euclidean2.512.N4096.aos-fp16-reused").defaultEligible, false)
        XCTAssertEqual(try boundary("batch.euclidean2.1536.N4096.aos-fp16-setup").defaultEligible, true)
        XCTAssertEqual(try boundary("batch.euclidean2.1536.N257.aos-fp16-reused").defaultEligible, false)
        XCTAssertNil(try boundary("batch.euclidean2.512.N257.aos-fp32-reused").defaultEligible)
        XCTAssertNil(try boundary("batch.euclidean2.512.N257.consumer-only").defaultEligible)
    }

    private func flipped(_ values: [Float], at index: Int) -> [Float] {
        var copy = values
        copy[index] = Float(bitPattern: copy[index].bitPattern ^ 1)
        return copy
    }

    private func checkRoutes<V: UnifiedVectorBuffer, S, H>(_ routes: BoundaryRoutes<V, S, H>, counts: [Int]) throws {
        func fold(_ values: [Float]) -> UInt64 { values.withUnsafeBufferPointer { OutputFold.fold($0) } }
        for n in counts {
            let f = try BatchFixtures.makeFixture(dim: routes.dim, count: n, runSeed: 0, vector: routes.makeVector)
            // The default batch suite builds its fixture through the same `BatchFixtures.makeFixture`; that is pinned lexically in
            // `BenchmarkBoundaryDriftContractTests.testBatchBenchBuildsItsFixtureThroughTheSharedBuilder`.
            var aos = [Float](repeating: 0, count: n)
            aos.withUnsafeMutableBufferPointer { routes.aos(f.query, f.candidates, 0..<n, $0) }
            var soaOut = [Float](repeating: 0, count: n)
            let soaStorage = routes.buildSoA(f.candidates)
            soaOut.withUnsafeMutableBufferPointer { routes.soa(f.query, soaStorage, $0) }
            let soaAPI = routes.soaAPI(f.query, f.candidates)
            var half = [Float](repeating: 0, count: n)
            let converted = routes.toFP16(f.candidates)
            half.withUnsafeMutableBufferPointer { routes.fp16(f.query, converted, 0..<n, $0) }
            for (label, output) in [("aos", aos), ("soa", soaOut), ("soaAPI", soaAPI), ("fp16", half)] {
                XCTAssertEqual(output.count, n, label)
                XCTAssertTrue(output.allSatisfy(\.isFinite), label)
                let base = fold(output)
                // First, middle and last scores are each perturbed individually; the consumer sees every change.
                for index in Set([0, n / 2, n - 1]) {
                    XCTAssertNotEqual(fold(flipped(output, at: index)), base, "\(routes.dim) N\(n) \(label) index \(index)")
                }
                // Mutation control: a first-element-only observation sees the first score but is blind to the middle and last
                // (n >= 3), so the assertions above can detect weak consumption.
                XCTAssertNotEqual(flipped(output, at: 0)[0].bitPattern, output[0].bitPattern)
                for index in Set([n / 2, n - 1]) where index != 0 {
                    XCTAssertEqual(flipped(output, at: index)[0].bitPattern, output[0].bitPattern, "\(label) control index \(index)")
                }
            }
            // Same fixture, same answer: the three FP32 routes agree with the independent Double reference.
            for (index, candidate) in f.candidates.enumerated() {
                let reference = routes.reference(f.query, candidate)
                for output in [aos, soaOut, soaAPI] { XCTAssertEqual(Double(output[index]), reference, accuracy: max(1e-5 * reference,
                    1e-5)) }
                XCTAssertEqual(Double(half[index]), reference, accuracy: max(1e-2 * reference, 1e-2))
            }
        }
    }

    func testCompanionFoldPerturbationAndFixtureAgreementAllWidths() throws {
        try checkRoutes(BoundaryKernelSets.d512, counts: [3, 257, 4096])
        try checkRoutes(BoundaryKernelSets.d768, counts: [3, 257])
        try checkRoutes(BoundaryKernelSets.d1536, counts: [3, 257])
    }
}

// MARK: - V04 phase C fix: geometric calibration

/// A deterministic virtual clock: each body invocation advances time by `bodyNanoseconds`, and readings are quantized to
/// whole ticks the way CLOCK_UPTIME_RAW is quantized to mach ticks.
private final class VirtualClock: @unchecked Sendable {
    private let lock = NSLock()
    private var elapsed: UInt64 = 0
    private(set) var calls = 0
    /// Every clock read as (body invocations so far, reading); consecutive pairs are the harness's timed segments.
    private(set) var reads: [(calls: Int, value: UInt64)] = []
    let bodyNanoseconds: UInt64, tick: UInt64
    init(bodyNanoseconds: UInt64, tick: UInt64) { self.bodyNanoseconds = bodyNanoseconds; self.tick = tick }
    func body() { lock.withLock { calls += 1; elapsed += bodyNanoseconds } }
    func now() -> UInt64 {
        lock.withLock {
            let value = tick == 0 ? 0 : elapsed / tick * tick
            reads.append((calls, value))
            return value
        }
    }

    /// The final timed segment: invocations run and its measured interval.
    var lastSegment: (invocations: Int, interval: UInt64)? {
        lock.withLock {
            guard reads.count >= 2 else { return nil }
            let a = reads[reads.count - 2], b = reads[reads.count - 1]
            return (b.calls - a.calls, b.value - a.value)
        }
    }
}

final class BenchmarkCalibrationContractTests: XCTestCase {
    func testSubTickBodyAutoScalesToAnAccurateEstimate() throws {
        // A 5 ns body under a 42 ns tick: one invocation reads as 0 ns, which used to fail the case.
        let clock = VirtualClock(bodyNanoseconds: 5, tick: 42)
        let result = try Harness.measure(name: "sub-tick", minTimeSeconds: 0, repeats: nil, samples: 3, calibrationTickNanoseconds: 42,
                                         now: clock.now) { clock.body() }
        XCTAssertEqual(result.sampleNsPerInvocation.count, 3)
        for (sample, iterations) in zip(result.sampleNsPerInvocation, result.sampleIterations) {
            XCTAssertGreaterThan(iterations, 1000)
            XCTAssertEqual(sample, 5.0, accuracy: 0.1)
        }
    }

    func testSlowBodyCalibratesOnOneInvocationExactlyAsBefore() throws {
        // 840 ns = 20 ticks >= 10 ticks: the first calibration interval is accepted, so counts are those of the previous rule.
        let clock = VirtualClock(bodyNanoseconds: 840, tick: 42)
        let result = try Harness.measure(name: "slow", minTimeSeconds: 0, repeats: nil, samples: 1, calibrationTickNanoseconds: 42,
                                         now: clock.now) { clock.body() }
        XCTAssertEqual(result.iterations, 95, "target 100 us * 8/10 / 840 ns")
        XCTAssertEqual(clock.calls, 96, "one calibration invocation plus the measured loop")
    }

    func testRecalibrationPerSampleUsesTheSameRule() throws {
        let clock = VirtualClock(bodyNanoseconds: 5, tick: 42)
        let result = try Harness.measure(name: "resample", minTimeSeconds: 0, repeats: nil, samples: 5, calibrationTickNanoseconds: 42,
                                         now: clock.now) { clock.body() }
        XCTAssertEqual(result.sampleIterations.count, 5)
        XCTAssertTrue(result.sampleNsPerInvocation.allSatisfy { abs($0 - 5.0) < 0.1 })
    }

    func testClockThatNeverAdvancesHitsTheCapAndFailsExplicitly() {
        let clock = VirtualClock(bodyNanoseconds: 5, tick: 0)
        XCTAssertThrowsError(try Harness.measure(name: "frozen", minTimeSeconds: 0, repeats: nil, calibrationTickNanoseconds: 42,
                                                 now: clock.now) { clock.body() }) { error in
            let text = "\(error)"
            XCTAssertTrue(text.contains("Unusable benchmark clock interval"), text)
            XCTAssertTrue(text.contains("\(HarnessConfig.maxCalibrationInvocations) invocations"), text)
            XCTAssertTrue(text.contains("--repeats"), text)
        }
        // 1 + 2 + ... + 2^24 invocations ran before giving up: bounded, never silent.
        XCTAssertEqual(clock.calls, (HarnessConfig.maxCalibrationInvocations << 1) - 1)
    }

    func testExplicitRepeatsOneIsExactlyOneInvocationWithAnActionableMessage() {
        let clock = VirtualClock(bodyNanoseconds: 5, tick: 42)
        XCTAssertThrowsError(try Harness.measure(name: "fixed", minTimeSeconds: 0, repeats: 1, calibrationTickNanoseconds: 42,
                                                 now: clock.now) { clock.body() }) { error in
            let text = "\(error)"
            XCTAssertTrue(text.contains("Unusable benchmark clock interval") && text.contains("--repeats") && text.contains("auto-scale"),
                text)
        }
        XCTAssertEqual(clock.calls, 1, "--repeats 1 runs exactly one invocation and does not recalibrate")
        // A fixed count that spans enough time succeeds and reports exactly that many iterations.
        let spanning = VirtualClock(bodyNanoseconds: 5, tick: 42)
        let result = try? Harness.measure(name: "fixed-ok", minTimeSeconds: 0, repeats: 64, calibrationTickNanoseconds: 42,
                                          now: spanning.now) { spanning.body() }
        XCTAssertEqual(result?.iterations, 64); XCTAssertEqual(spanning.calls, 64)
    }

    @MainActor func testAsyncHarnessUsesTheSameCalibration() async throws {
        let clock = VirtualClock(bodyNanoseconds: 5, tick: 42)
        let result = try await Harness.measureAsync(name: "async-sub-tick", minTimeSeconds: 0, repeats: nil, samples: 2,
                                                    calibrationTickNanoseconds: 42, now: clock.now) { clock.body() }
        XCTAssertTrue(result.sampleNsPerInvocation.allSatisfy { abs($0 - 5.0) < 0.1 })
        let frozen = VirtualClock(bodyNanoseconds: 5, tick: 0)
        do {
            _ = try await Harness.measureAsync(name: "async-frozen", minTimeSeconds: 0, repeats: nil, calibrationTickNanoseconds: 42,
                                               now: frozen.now) { frozen.body() }
            XCTFail("expected the cap error")
        } catch { XCTAssertTrue("\(error)".contains("Unusable benchmark clock interval")) }
    }

    func testTickFormulaWithInjectedTimebaseValues() {
        // Independent expected values, computed by hand: ceil(numer / denom).
        XCTAssertEqual(ClockResolution.ticks(numer: 125, denom: 3), 42)   // 41.67 -> 42 (Apple Silicon)
        XCTAssertEqual(ClockResolution.ticks(numer: 1, denom: 1), 1)
        XCTAssertEqual(ClockResolution.ticks(numer: 3, denom: 2), 2)      // 1.5 -> 2
        XCTAssertEqual(ClockResolution.ticks(numer: 4, denom: 2), 2)      // exact
        XCTAssertEqual(ClockResolution.ticks(numer: 1_000_000_000, denom: 24_000_000), 42)
        XCTAssertNil(ClockResolution.ticks(numer: 0, denom: 1))
        XCTAssertNil(ClockResolution.ticks(numer: 1, denom: 0))
        XCTAssertEqual(ClockResolution.fallbackNanoseconds, 1000)
        // The running host's tick is that formula applied to its real timebase, and the source string names it.
        var info = mach_timebase_info_data_t()
        XCTAssertEqual(mach_timebase_info(&info), KERN_SUCCESS)
        XCTAssertEqual(ClockResolution.tickNanoseconds, ClockResolution.ticks(numer: info.numer, denom: info.denom))
        XCTAssertTrue(ClockResolution.source.contains("mach_timebase_info \(info.numer)/\(info.denom)"), ClockResolution.source)
    }

    /// A clock that replays scripted readings, then repeats the last one.
    private final class ScriptedClock: @unchecked Sendable {
        private let lock = NSLock(); private var readings: [UInt64]; private(set) var calls = 0
        init(_ readings: [UInt64]) { self.readings = readings }
        func body() { lock.withLock { calls += 1 } }
        func now() -> UInt64 { lock.withLock { readings.count > 1 ? readings.removeFirst() : readings[0] } }
    }

    func testImplausiblyLargeIntervalThrowsInsteadOfTrapping() {
        // First interval 0 (grow to 2 invocations), then a reading near UInt64.max: `interval + count - 1` used to overflow here.
        for script in [[UInt64(0), 0, 0, UInt64.max], [0, UInt64.max - 1], [0, UInt64.max / 16 + 1]] {
            let clock = ScriptedClock(script)
            XCTAssertThrowsError(try Harness.measure(name: "jump", minTimeSeconds: 0, repeats: nil, calibrationTickNanoseconds: 42,
                                                     now: clock.now) { clock.body() }, "\(script)") { error in
                XCTAssertTrue("\(error)".contains("not a plausible duration"), "\(error)")
            }
        }
        // The pure estimate accepts the largest plausible interval and rounds up exactly.
        XCTAssertEqual(try Calibration.estimate(interval: 1001, invocations: 10, tick: 42), 101)
        XCTAssertEqual(try Calibration.estimate(interval: UInt64.max / 16, invocations: 4, tick: 42), 288_230_376_151_711_744)
        XCTAssertNil(try Calibration.estimate(interval: 419, invocations: 1, tick: 42))
    }

    /// Sweep min-time just above the 100 us floor's 80 us target so the proportional top-up is only a few 5 ns invocations. Whatever the
    /// harness measured last (the top-up when it ran, otherwise the main loop) must span at least 10 ticks, and the top-up must have
    /// actually run for a meaningful share of the sweep.
    /// Main-actor isolated, like every async test method in this target, so the assertions run on the main actor after each await.
    /// Observed on this toolchain, mechanism not established: some failures recorded from a nonisolated async XCTest method after
    /// awaited work were not reported, so such a test could pass with failing assertions.
    @MainActor private func assertSweepProperty(_ run: (Int, VirtualClock, Double) async throws -> BenchResult) async throws {
        var topUps = 0
        for step in 0..<80 {
            let clock = VirtualClock(bodyNanoseconds: 5, tick: 42)
            let result = try await run(step, clock, Double(80_000 + 7 * step) / 1e9)
            XCTAssertEqual(result.sampleNsPerInvocation[0], 5.0, accuracy: 1.0, "step \(step)")
            let last = try XCTUnwrap(clock.lastSegment, "step \(step)")
            XCTAssertGreaterThanOrEqual(last.interval, 10 * 42, "step \(step): last segment of \(last.invocations) invocations")
            if last.invocations != result.iterations { topUps += 1 }
        }
        XCTAssertGreaterThan(topUps, 20, "the sweep must exercise the top-up loop")
    }

    @MainActor func testExtensionLoopNeverMeasuresASubTickInterval() async throws {
        XCTAssertEqual(Calibration.extensionFloor(iterations: 16_000, elapsed: 80_000, tick: 42), 85)
        XCTAssertEqual(Calibration.extensionFloor(iterations: 5, elapsed: 0, tick: 42), 5)
        try await assertSweepProperty { step, clock, minTime in
            try Harness.measure(name: "extension-\(step)", minTimeSeconds: minTime, repeats: nil, calibrationTickNanoseconds: 42,
                                now: clock.now) { clock.body() }
        }
    }

    @MainActor func testAsyncExtensionLoopNeverMeasuresASubTickInterval() async throws {
        try await assertSweepProperty { step, clock, minTime in
            try await Harness.measureAsync(name: "async-extension-\(step)", minTimeSeconds: minTime, repeats: nil,
                                           calibrationTickNanoseconds: 42, now: clock.now) { clock.body() }
        }
    }

    func testAbsoluteTimeTickFromTheDoubleULP() {
        // Hand-computed. 8.1e8 s lies in [2^29, 2^30): the seconds grid is 2^-23 s = 119.2 ns, but the product 8.1e17 lies in
        // [2^59, 2^60), whose ULP is 2^7 = 128 ns, the coarser grid -> 128.
        XCTAssertEqual(ClockResolution.absoluteTimeTick(at: 8.1e8, timebaseTick: 42), 128)
        XCTAssertEqual(ClockResolution.absoluteTimeTick(at: 1.0, timebaseTick: 42),
            42)          // both grids are far below the timebase tick
        // 3.0e9 s lies in [2^31, 2^32): seconds grid 2^-21 s = 476.84 -> 477 ns; the product 3e18 lies in [2^61, 2^62): ULP 2^9 = 512.
        XCTAssertEqual(ClockResolution.absoluteTimeTick(at: 3.0e9, timebaseTick: 42), 512)
        XCTAssertGreaterThan(ClockResolution.absoluteTimeTick(at: 3.0e9, timebaseTick: 42), ClockResolution.absoluteTimeTick(at: 8.1e8,
            timebaseTick: 42))
        XCTAssertEqual(ClockResolution.absoluteTimeTick(at: 8.1e8, timebaseTick: 500), 500)      // never below the timebase tick
        // Edge epochs: zero and negative values stay on the timebase tick; unusable values use the conservative fallback.
        XCTAssertEqual(ClockResolution.absoluteTimeTick(at: 0, timebaseTick: 42), 42)
        XCTAssertEqual(ClockResolution.absoluteTimeTick(at: -8.1e8, timebaseTick: 42), 128)
        for unusable in [Double.nan, .infinity, -.infinity, 1e300] {
            XCTAssertEqual(ClockResolution.absoluteTimeTick(at: unusable, timebaseTick: 42), 1000, "\(unusable)")
        }
        // The platform that reads mach time is unchanged: its tick is still the plain timebase ceiling.
        #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
        XCTAssertTrue(ClockResolution.source.hasPrefix("mach_timebase_info"), ClockResolution.source)
        #endif
    }

}
