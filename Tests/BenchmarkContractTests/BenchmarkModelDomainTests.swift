import Foundation
import XCTest
import VectorCore
@testable import VectorCoreBenchSupport
import VectorCoreBenchmarking

private enum ModelDomainResultSuite: BenchmarkSuite {
    static let name = "model-domain-result"
    static func catalog(options: ValidatedOptions, progress: BenchmarkProgressReporter) -> [BenchmarkCase] {
        if ProcessInfo.processInfo.environment["F012_TEST_RESULT_NAME"] == "unavailable" {
            let rejected = "batch.euclidean2.512.N16776696.generic.sequential"
            return [
                BenchmarkCase(rejected, boundary: Boundaries.fixtureInjected(name: rejected),
                              availability: .unavailable(reason: "test unavailable route")) { fatalError("unavailable case executed") },
                BenchmarkCase("noop.scalar-add", boundary: Boundaries.lookup("noop.scalar-add", runSeed: 0)!) {
                    [BenchResult(name: "noop.scalar-add", iterations: 1, totalNanoseconds: 100)]
                }
            ]
        }
        return [BenchmarkCase(name, boundary: Boundaries.fixtureInjected(name: name), acceptsAnyResultName: true) {
            await run(options: options, progress: progress)
        }]
    }
    static func run(options: ValidatedOptions, progress: BenchmarkProgressReporter) async -> [BenchResult] {
        let resultName = ProcessInfo.processInfo.environment["F012_TEST_RESULT_NAME"]!
        return [BenchResult(name: resultName, iterations: 1, totalNanoseconds: 100,
                            correctness: Correctness.scalar(current: 1, reference: 1))]
    }
}

final class BenchmarkModelDomainTests: XCTestCase {
    private static let overrides: [[String]] = [[], ["--max-rel-error", "1"], ["--max-abs-error", "1"],
                                                ["--max-rel-error", "1", "--max-abs-error", "1"]]
    private struct Boundary {
        let dim: Int
        let metric: String
        let variant: String
        let firstRejected: Int
    }
    // Hand-derived from the four route counts at k32 = 2^24; independent of policy helpers.
    private static let boundaries: [Boundary] = [
        .init(dim: 512, metric: "euclidean2", variant: "generic", firstRejected: 16_776_696),
        .init(dim: 768, metric: "euclidean2", variant: "generic", firstRejected: 16_776_440),
        .init(dim: 1536, metric: "euclidean2", variant: "generic", firstRejected: 16_775_672),
        .init(dim: 512, metric: "euclidean", variant: "generic", firstRejected: 16_776_702),
        .init(dim: 768, metric: "euclidean", variant: "generic", firstRejected: 16_776_446),
        .init(dim: 1536, metric: "euclidean", variant: "generic", firstRejected: 16_775_678),
        .init(dim: 512, metric: "euclidean2", variant: "optimized", firstRejected: 16_777_178),
        .init(dim: 768, metric: "euclidean2", variant: "optimized", firstRejected: 16_777_162),
        .init(dim: 1536, metric: "euclidean2", variant: "optimized", firstRejected: 16_777_114),
        .init(dim: 512, metric: "euclidean", variant: "optimized", firstRejected: 16_777_177),
        .init(dim: 768, metric: "euclidean", variant: "optimized", firstRejected: 16_777_161),
        .init(dim: 1536, metric: "euclidean", variant: "optimized", firstRejected: 16_777_113)
    ]

    // The callback terminates before any case setup. The process tests therefore exercise
    // the production registry and runner without allocating the enormous batch fixture.
    @MainActor static func runProcessFixture(encoded: String) async throws -> Never {
        let environment = ProcessInfo.processInfo.environment
        let arguments = try JSONDecoder().decode([String].self, from: Data(encoded.utf8))
        let options = parseCLI(arguments: arguments, environment: [:])
        if environment["F012_TEST_RESULT_NAME"] != nil {
            exit(await BenchmarkRunner.run(options: options, registry: [ModelDomainResultSuite.name: ModelDomainResultSuite.self]))
        }
        exit(await BenchmarkRunner.run(options: options, onCase: { _ in
            fputs("F012_CASE_EXECUTION_REACHED\n", stderr)
            exit(97)
        }))
    }

    func testUnsupportedModelsRejectedBeforeExecution() throws {
        for strict in [false, true] {
            for overrides in Self.overrides {
                for format in ["pretty", "json", "csv"] {
                    var arguments = batchArguments(count: 16_776_696)
                    arguments += overrides
                    if strict { arguments.append("--strict-correctness") }
                    let result = try launch(arguments: arguments, format: format)
                    XCTAssertEqual(result.status, 1, result.log)
                    XCTAssertTrue(result.log.contains("Invalid configuration:"), result.log)
                    XCTAssertTrue(result.log.contains("correctness model"), result.log)
                    XCTAssertFalse(result.log.contains("F012_CASE_EXECUTION_REACHED"), result.log)
                    XCTAssertFalse(result.reportExists, result.log)
                }
            }
        }
    }

    func testBoundaryBelowReachesExecutionWithoutAllocatingFixture() throws {
        let result = try launch(arguments: batchArguments(count: 16_776_695), format: "json")
        XCTAssertEqual(result.status, 97, result.log)
        XCTAssertTrue(result.log.contains("F012_CASE_EXECUTION_REACHED"), result.log)
        XCTAssertFalse(result.reportExists)
    }

    func testProviderModelDomainBoundaries() throws {
        for row in Self.boundaries {
            for provider in ["sequential", "parallel", "automatic", "injected"] {
                for n in [1, 100, 1_000, row.firstRejected - 1, row.firstRejected, row.firstRejected + 1] {
                    let name = "batch.\(row.metric).\(row.dim).N\(n).\(row.variant).\(provider)"
                    for overrides in Self.overrides {
                        let options = try ValidatedOptions(parseCLI(arguments: ["--suites", "batch", "--dims", String(row.dim),
                            "--batch-ns", String(n), "--filter", name, "--ab"] + overrides, environment: [:]))
                        let catalog = BatchBench.catalog(options: options, provider: provider == "injected" ? CPUComputeProvider.sequential : nil)
                        let selected = catalog.filter { options.caseFilter.matches($0.name) }
                        XCTAssertEqual(selected.map(\.name), [name])
                        let model = try XCTUnwrap(CorrectnessPolicy.errorModel(forCaseName: name))
                        XCTAssertEqual(model.k32, 16_777_216 + n - row.firstRejected, name)
                        let problem = CorrectnessPolicy.domainProblem(forCaseName: name)
                        if n < row.firstRejected {
                            XCTAssertNil(problem, name)
                            let threshold = try XCTUnwrap(CorrectnessPolicy.rule(forCaseName: name).maxRel)
                            XCTAssertTrue(threshold.isFinite && threshold > 0, name)
                        } else {
                            XCTAssertTrue(problem?.contains(name) == true, name)
                            XCTAssertTrue(problem?.contains("k*u < 1") == true, name)
                        }
                    }
                }
            }
        }
    }

    func testDomainCheckRejectsUnrepresentableProviderCounts() {
        for row in Self.boundaries {
            for provider in ["sequential", "parallel", "automatic", "injected"] {
                for n in [Int.max - 1_536, Int.max - 1, Int.max] {
                    let name = "batch.\(row.metric).\(row.dim).N\(n).\(row.variant).\(provider)"
                    XCTAssertTrue(CorrectnessPolicy.hasDerivedModel(forCaseName: name), name)
                    XCTAssertNotNil(CorrectnessPolicy.domainProblem(forCaseName: name), name)
                }
            }
        }
        for name in ["batch.cosine.512.N\(Int.max).optimized-fused.injected", "batch.euclidean2.512.N\(Int.max).optimized-fp16",
                     "batch.euclidean2.512.N0.generic.sequential", "batch.euclidean2.512.N-1.generic.sequential", "fixture"] {
            XCTAssertFalse(CorrectnessPolicy.hasDerivedModel(forCaseName: name), name)
            XCTAssertNil(CorrectnessPolicy.domainProblem(forCaseName: name), name)
        }
    }

    func testNonObservedModelsDoNotAcquireProviderCountLimit() throws {
        for dim in [512, 768, 1536] {
            for variant in ["aos-fp32-reused", "aos-fp32-setup", "consumer-only", "optimized-soa", "soa-fp32-reused", "soa-fp32-setup"] {
                let name = "batch.euclidean2.\(dim).N\(Int.max).\(variant)"
                XCTAssertNil(CorrectnessPolicy.domainProblem(forCaseName: name), name)
                XCTAssertTrue(try XCTUnwrap(CorrectnessPolicy.rule(forCaseName: name).maxRel).isFinite, name)
            }
            for variant in ["optimized", "consumer-only"] {
                let name = "matrix.euclidean2.\(dim).N\(Int.max).\(variant)"
                XCTAssertNil(CorrectnessPolicy.domainProblem(forCaseName: name), name)
                XCTAssertTrue(try XCTUnwrap(CorrectnessPolicy.rule(forCaseName: name).maxAbs).isFinite, name)
            }
        }
    }

    func testModelMatrixCoversDerivedCatalogFamilies() throws {
        let options = try ValidatedOptions(parseCLI(arguments: ["--suites", "batch,boundary,matrix", "--dims", "512,768,1536",
            "--batch-ns", "10000", "--ab", "--prefer-soa", "--use-mixed-precision"], environment: [:]))
        let progress = BenchmarkProgressReporter(format: .none)
        let catalog = BatchBench.catalog(options: options, progress: progress)
            + BoundaryBench.catalog(options: options, progress: progress)
            + MatrixDistanceBench.catalog(options: options, progress: progress)
        var routes = Set<String>()
        for item in catalog where item.availability == .available {
            XCTAssertNil(CorrectnessPolicy.domainProblem(forCaseName: item.name), item.name)
            if let model = CorrectnessPolicy.errorModel(forCaseName: item.name) { routes.insert(model.route) }
        }
        XCTAssertEqual(routes, ["generic-euclid2", "vdsp-euclid", "optimized-euclid2", "optimized-euclid", "soa-euclid2", "matrix-norm-identity"])
    }

    func testSelectedValidRoutesReachExecutionDespiteExcludedInvalidRoutes() throws {
        let base = ["--suites", "batch", "--dims", "512", "--batch-ns", "16776696", "--repeats", "1", "--samples", "1"]
        let selections = [
            ["--filter", "*.optimized.sequential"],
            ["--exclude", "*.generic.*"],
            ["--filter-mode", "regex", "--filter", "^batch[.]euclidean2[.]512[.]N16776696[.]optimized[.]sequential$"],
            ["--filter", "batch.cosine.*"],
            ["--prefer-soa", "--filter", "*.optimized-soa"],
            ["--use-mixed-precision", "--filter", "*.optimized-fp16"]
        ]
        for selection in selections {
            let result = try launch(arguments: base + selection, format: "json")
            XCTAssertEqual(result.status, 97, result.log)
            XCTAssertTrue(result.log.contains("F012_CASE_EXECUTION_REACHED"), result.log)
            XCTAssertFalse(result.reportExists)
        }
        let rejected = try launch(arguments: base + ["--ab-only", "--filter", "batch.euclidean*.512.N16776696.generic.sequential"], format: "json")
        XCTAssertEqual(rejected.status, 1, rejected.log)
        XCTAssertTrue(rejected.log.contains("correctness model"), rejected.log)
        XCTAssertFalse(rejected.log.contains("F012_CASE_EXECUTION_REACHED"), rejected.log)
    }

    func testUnsupportedInjectedResultsAreReportedWithoutTrapping() throws {
        for n in [16_776_696, Int.max] {
            let name = "batch.euclidean2.512.N\(n).generic.injected"
            for strict in [false, true] {
                for overrides in Self.overrides {
                    for format in ["pretty", "json", "csv"] {
                        var arguments = ["--suites", ModelDomainResultSuite.name, "--no-ab", "--repeats", "1"] + overrides
                        if strict { arguments.append("--strict-correctness") }
                        let result = try launch(arguments: arguments, format: format, resultName: name)
                        XCTAssertEqual(result.status, strict ? 1 : 0, result.log)
                        XCTAssertTrue(result.log.contains("Unsupported correctness model"), result.log)
                        XCTAssertTrue(result.log.contains("unsupported correctness model: 1 case(s)"), result.log)
                        XCTAssertEqual(result.reportExists, format != "pretty", result.log)
                        if format == "json" {
                            let report = try JSONDecoder().decode(BenchRun.self, from: Data(result.report.utf8))
                            XCTAssertEqual(report.failures.count, 1)
                            XCTAssertEqual(report.failures.first?.name, name)
                            XCTAssertTrue(report.failures.first?.reason.contains("Unsupported correctness model") == true)
                        } else if format == "csv" {
                            XCTAssertTrue(result.report.contains("Unsupported correctness model"))
                        }
                    }
                }
            }
        }
    }

    func testUnavailableUnsupportedModelDoesNotRejectExecutableSelection() throws {
        let result = try launch(arguments: ["--suites", ModelDomainResultSuite.name, "--no-ab", "--strict-correctness", "--repeats", "1"],
                                format: "json", resultName: "unavailable")
        XCTAssertEqual(result.status, 0, result.log)
        let report = try JSONDecoder().decode(BenchRun.self, from: Data(result.report.utf8))
        XCTAssertTrue(report.failures.isEmpty)
        XCTAssertEqual(report.results.map(\.name), ["noop.scalar-add"])
        XCTAssertEqual(report.unavailable.count, 1)
    }

    private func batchArguments(count: Int) -> [String] {
        ["--suites", "batch", "--dims", "512", "--batch-ns", String(count), "--no-ab",
         "--filter", "batch.euclidean2.512.N\(count).generic.sequential", "--repeats", "1", "--samples", "1"]
    }

    private func launch(arguments: [String], format: String, resultName: String? = nil) throws
        -> (status: Int32, log: String, reportExists: Bool, report: String) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("F012-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let report = directory.appendingPathComponent("report.\(format)")
        let log = directory.appendingPathComponent("process.log")
        var environment: [String: String] = [:]
        environment["F012_TEST_ARGUMENTS"] = String(decoding: try JSONEncoder().encode(
            arguments + ["--format", format, "--out", report.path]), as: UTF8.self)
        environment["F012_TEST_RESULT_NAME"] = resultName
        let status = try BenchmarkRunnerContractTests.launchFixture(environment: environment, output: log)
        return (status, try String(contentsOf: log, encoding: .utf8),
                FileManager.default.fileExists(atPath: report.path), (try? String(contentsOf: report, encoding: .utf8)) ?? "")
    }
}
