import XCTest
import Foundation
@testable import VectorCoreBenchSupport
import VectorCoreBenchmarking
import VectorCore

private enum InjectedFailure: Error { case provider }
private actor CallCounter {
    var count = 0
    func step(failAt: Int) throws {
        count += 1
        if count == failAt { throw InjectedFailure.provider }
    }
}

private enum FixtureSuite: BenchmarkSuite {
    static var name: String { "fixture" }
    static func run(options: ValidatedOptions, progress: BenchmarkProgressReporter) async -> [BenchResult] {
        let scenario = ProcessInfo.processInfo.environment["BENCHMARK_TEST_SCENARIO"] ?? "match"
        if scenario.hasPrefix("observed-") {
            let parts = scenario.split(separator: "-")
            let phase = String(parts[1])
            let value: Float = parts[2] == "wrong" ? 0 : (parts[2] == "nan" ? .nan : (parts[2] == "positive" ? .infinity : -.infinity))
            let values = ObservationCounter()
            let observations = ProviderObservations(reference: 1)
            let body: @Sendable () async -> Void = {
                observations.observe(await values.next(badAt: phase == "later" ? 2 : 1, value: value))
            }
            return await BatchExecution.run(name: "observed.provider.fixture") {
                if phase == "warmup" { try await Harness.warmupAsync(body, minWarmupNs: 0) }
                let result = try await Harness.measureAsync(name: "observed.provider.fixture", minTimeSeconds: 0, repeats: phase == "calibration" ? nil : 1, samples: 3, body)
                return [result.withCorrectness(observations.combining(Correctness.scalar(current: 1, reference: 1)))]
            }
        }
        if scenario.hasPrefix("batch-return-") {
            var batchOptions = CLIOptions()
            batchOptions.suites = ["batch"]
            batchOptions.dims = [512, 768, 1536]
            batchOptions.batchNs = [2]
            batchOptions.repeats = 1
            batchOptions.abCompare = true
            return await BatchBench.run(options: try! ValidatedOptions(batchOptions), progress: progress, provider: ReturnedValueProvider(scenario: scenario))
        }
        if scenario == "empty" { return [] }
        if scenario.hasPrefix("provider") {
            let counter = CallCounter()
            return await BatchExecution.run(name: "batch.euclidean.512.fixture") {
                let failAt = scenario == "provider-later-sample" ? 2 : 1
                if scenario == "provider-warmup" {
                    try await Harness.warmupAsync({ try await counter.step(failAt: failAt) }, minWarmupNs: 0)
                }
                let measured = try await Harness.measureAsync(name: "batch.euclidean.512.fixture", minTimeSeconds: 0.001, repeats: scenario == "provider-calibration" ? nil
                    : 1, samples: 3, warmupNs: scenario == "provider-builtin-warmup" ? 1 : 0) {
                    try await counter.step(failAt: failAt)
                }
                return [measured]
            }
        }
        var correctness: CorrectnessStats? = Correctness.vector(currents: [1,2], references: [1,2])
        switch scenario {
        case "mismatch": correctness = Correctness.vector(currents: [1,3], references: [1,2])
        case "nan-current": correctness = Correctness.vector(currents: [.nan], references: [1])
        case "nan-reference": correctness = Correctness.vector(currents: [1], references: [.nan])
        case "positive-infinity": correctness = Correctness.vector(currents: [.infinity], references: [.infinity])
        case "negative-infinity": correctness = Correctness.vector(currents: [-.infinity], references: [1])
        case "short": correctness = Correctness.vector(currents: [1], references: [1,2])
        case "long": correctness = Correctness.vector(currents: [1,2], references: [1])
        case "empty-comparison": correctness = Correctness.vector(currents: [], references: [])
        case "missing": correctness = nil
        default: break
        }
        return [BenchResult(name: "batch.euclidean.512.fixture", iterations: 1, totalNanoseconds: 100, meanNsPerOp: scenario == "nonfinite-timing" ? .nan : nil, correctness: correctness)]
    }
}

final class BenchmarkRunnerContractTests: XCTestCase {
    // This subprocess entry exists only in the test bundle. It invokes the production runner.
    @MainActor func testProcessFixture() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["BENCHMARK_TEST_SCENARIO"] != nil else { return }
        var options = CLIOptions()
        options.suites = ["fixture"]
        options.format = env["BENCHMARK_TEST_FORMAT"] ?? "json"
        options.outputPath = env["BENCHMARK_TEST_OUTPUT"]!
        options.strictCorrectness = env["BENCHMARK_TEST_STRICT"] != "0"
        options.abCompare = false
        exit(await BenchmarkRunner.run(options: options, registry: ["fixture": FixtureSuite.self]))
    }

    func testProcessTable() throws {
        let scenarios = ["match", "mismatch", "nan-current", "nan-reference", "positive-infinity", "negative-infinity", "short", "long", "empty-comparison", "missing",
            "empty", "provider-warmup", "provider-builtin-warmup", "provider-calibration", "provider-fixed", "provider-later-sample", "nonfinite-timing"]
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("benchmark-contract-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for scenario in scenarios {
            let artifact = root.appendingPathComponent(scenario + ".json")
            let status = try launch(scenario: scenario, artifact: artifact, strict: !scenario.hasPrefix("provider") && scenario != "nonfinite-timing")
            XCTAssertEqual(status, scenario == "match" ? 0 : 1, scenario)
            let run = try JSONDecoder().decode(BenchRun.self, from: Data(contentsOf: artifact))
            XCTAssertEqual(run.failures.isEmpty, scenario == "match", scenario)
            if scenario.hasPrefix("provider") || scenario == "nonfinite-timing" { XCTAssertTrue(run.results.isEmpty, scenario) }
            if scenario.hasPrefix("provider") {
                XCTAssertTrue(run.failures.contains { $0.name == "batch.euclidean.512.fixture" && $0.reason.contains("provider") }, scenario)
            }
            if ["nan-current", "nan-reference", "positive-infinity", "negative-infinity", "short", "long", "empty-comparison"].contains(scenario) {
                let c = try XCTUnwrap(run.results.first?.correctness)
                XCTAssertEqual(c.status, .invalid)
                XCTAssertNotNil(c.reason)
                XCTAssertEqual(c.samples, 0)
                XCTAssertNil(c.maxAbsError); XCTAssertNil(c.meanAbsError)
                XCTAssertNil(c.maxRelError); XCTAssertNil(c.meanRelError)
            }
        }
        for format in ["json", "csv"] {
            XCTAssertEqual(try launch(scenario: "match", artifact: root, strict: false, format: format), 1)
        }
        let csvFailure = root.appendingPathComponent("failure.csv")
        XCTAssertEqual(try launch(scenario: "provider-fixed", artifact: csvFailure, strict: false, format: "csv"), 1)
        XCTAssertTrue(try String(contentsOf: csvFailure, encoding: .utf8).contains("failure_name,failure_reason"))
        print("Process artifacts: \(root.path)")
    }

    private func launch(scenario: String, artifact: URL, strict: Bool, format: String = "json") throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
        process.arguments = ["-XCTest", "BenchmarkContractTests.BenchmarkRunnerContractTests/testProcessFixture", Bundle(for: Self.self).bundlePath]
        var env = ProcessInfo.processInfo.environment.filter { ["PATH", "HOME", "TMPDIR", "DYLD_FRAMEWORK_PATH", "DYLD_LIBRARY_PATH", "SDKROOT"].contains($0.key) }
        env["BENCHMARK_TEST_SCENARIO"] = scenario
        env["BENCHMARK_TEST_OUTPUT"] = artifact.path
        env["BENCHMARK_TEST_STRICT"] = strict ? "1" : "0"
        env["BENCHMARK_TEST_FORMAT"] = format
        process.environment = env
        let output = artifact.deletingLastPathComponent().appendingPathComponent("\(scenario)-\(format).log")
        _ = FileManager.default.createFile(atPath: output.path, contents: nil)
        let handle = try FileHandle(forWritingTo: output)
        defer { try? handle.close() }
        process.standardOutput = handle
        process.standardError = handle
        let exited = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in exited.signal() }
        try process.run()
        if exited.wait(timeout: .now() + 15) == .timedOut {
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
            let cleanup = exited.wait(timeout: .now() + 2)
            throw NSError(domain: "BenchmarkRunnerContractTests.childTimeout", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "\(scenario): child deadline; cleanup=\(cleanup); log=\(output.path)"])
        }
        return process.terminationStatus
    }

    func testActualBatchProviderReturnFamily() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("benchmark-provider-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let expectedNames = Set([512, 768, 1536].flatMap { dim in
            [("euclidean2", "generic"), ("euclidean2", "optimized"), ("euclidean", "generic"), ("euclidean", "optimized"), ("cosine", "optimized-fused"), ("cosine", "optimized-preNorm")].map {
                "batch.\($0.0).\(dim).N2.\($0.1).injected"
            }
        })
        for value in ["wrong", "nan", "positive-infinity", "negative-infinity", "throws"] {
            let scenario = "batch-return-" + value
            let artifact = root.appendingPathComponent(scenario + ".json")
            XCTAssertEqual(try launch(scenario: scenario, artifact: artifact, strict: true), 1)
            let run = try JSONDecoder().decode(BenchRun.self, from: Data(contentsOf: artifact))
            let failedNames = Set(run.failures.map(\.name))
            XCTAssertEqual(failedNames, expectedNames, value)
            if value == "throws" {
                XCTAssertTrue(run.results.isEmpty)
                for name in expectedNames { XCTAssertTrue(run.failures.contains { $0.name == name && $0.reason.contains("provider") }, name) }
                continue
            }
            XCTAssertEqual(Set(run.results.map(\.name)), expectedNames)
            XCTAssertTrue(run.failures.allSatisfy { !$0.reason.isEmpty })
            for result in run.results {
                guard let correctness = result.correctness else { XCTFail("Missing correctness: \(result.name)"); continue }
                if value == "wrong" {
                    XCTAssertEqual(correctness.status, .valid, result.name)
                    XCTAssertGreaterThan(try XCTUnwrap(correctness.maxRelError), 1e-6, result.name)
                } else {
                    XCTAssertEqual(correctness.status, .invalid, result.name)
                    XCTAssertTrue(correctness.reason?.contains("Nonfinite") == true, result.name)
                }
            }
        }
        print("Provider artifacts: \(root.path)")
    }

    func testBadObservationSurvivesLaterGoodResultsAndReference() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("benchmark-observed-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for phase in ["warmup", "calibration", "fixed", "later"] {
            for value in ["wrong", "nan", "positive", "negative"] {
                let scenario = "observed-\(phase)-\(value)"
                let artifact = root.appendingPathComponent(scenario + ".json")
                XCTAssertEqual(try launch(scenario: scenario, artifact: artifact, strict: true), 1, scenario)
                let run = try JSONDecoder().decode(BenchRun.self, from: Data(contentsOf: artifact))
                XCTAssertEqual(run.failures.map(\.name), ["observed.provider.fixture"], scenario)
                XCTAssertFalse(try XCTUnwrap(run.failures.first).reason.isEmpty)
                let c = try XCTUnwrap(run.results.first?.correctness)
                XCTAssertEqual(c.status, value == "wrong" ? .valid : .invalid, scenario)
                if value == "wrong" { XCTAssertEqual(try XCTUnwrap(c.maxRelError), 1, accuracy: 1e-12) }
            }
        }
        print("Observation artifacts: \(root.path)")
    }

    func testProviderObservationWeightedMerge() {
        let observations = ProviderObservations(reference: 1)
        observations.observe(2)
        observations.observe(1)
        let combined = observations.combining(Correctness.scalar(current: 3, reference: 2))
        XCTAssertEqual(combined.samples, 3)
        XCTAssertEqual(combined.maxAbsError, 1, accuracy: 1e-12)
        XCTAssertEqual(combined.meanAbsError, 2.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(combined.maxRelError, 1, accuracy: 1e-12)
        XCTAssertEqual(combined.meanRelError, 0.5, accuracy: 1e-12)
    }

    func testAllErrorStatisticsMustBeFinite() throws {
        for index in 0..<4 {
            for bad in [Double.nan, .infinity, -.infinity] {
                var values = [0.0, 0.0, 0.0, 0.0]
                values[index] = bad
                let c = CorrectnessStats(samples: 1, maxAbsError: values[0], meanAbsError: values[1], maxRelError: values[2], meanRelError: values[3])
                XCTAssertEqual(c.status, .invalid)
                XCTAssertNoThrow(try JSONEncoder().encode(CorrectnessOut(from: c)))
            }
        }
    }

    func testScalarFiniteDomainAndMeasuredErrors() {
        for pair: (Float, Double) in [(.nan, 1), (1, .nan), (.infinity, .infinity), (-.infinity, 1)] {
            XCTAssertEqual(Correctness.scalar(current: pair.0, reference: pair.1).status, .invalid)
        }
        let scalar = Correctness.scalar(current: 3, reference: 2)
        XCTAssertEqual(scalar.status, .valid)
        XCTAssertEqual(scalar.maxAbsError, 1, accuracy: 1e-12)
        XCTAssertEqual(scalar.meanAbsError, 1, accuracy: 1e-12)
        XCTAssertEqual(scalar.maxRelError, 0.5, accuracy: 1e-12)
        XCTAssertEqual(scalar.meanRelError, 0.5, accuracy: 1e-12)
    }
}

private struct ReturnedValueProvider: ComputeProvider {
    let scenario: String
    var device: ComputeDevice { .cpu }
    var maxConcurrency: Int { 1 }
    var deviceInfo: ComputeDeviceInfo { ComputeDeviceInfo(name: "Injected provider", maxThreads: 1, preferredChunkSize: 1) }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T { try await work() }
    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R, minChunk: Int,
                                    _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                    _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        if scenario == "batch-return-throws" { throw InjectedFailure.provider }
        let value: Float
        switch scenario {
        case "batch-return-nan": value = .nan
        case "batch-return-positive-infinity": value = .infinity
        case "batch-return-negative-infinity": value = -.infinity
        default: value = 0
        }
        return value as! R
    }
}

private actor ObservationCounter {
    var count = 0
    func next(badAt: Int, value: Float) -> Float {
        count += 1
        return count == badAt ? value : 1
    }
}
