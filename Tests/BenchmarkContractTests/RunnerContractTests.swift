import XCTest
import Foundation
import VectorCoreTestSupport
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
        if let control = env["LAUNCHER_CONTROL"], control != "environment" {
            FileHandle.standardOutput.write(Data("LAUNCHER_CONTROL_READY \(control)\n".utf8))
            if control == "hang" { while true { sleep(60) } }
            exit(control == "failure" ? 7 : 0)
        }
        if env["LAUNCHER_CONTROL"] == "environment" {
            var captured = env
            captured["fixturePID"] = String(getpid())
            captured["fixturePGID"] = String(getpgrp())
            try JSONEncoder().encode(captured).write(to: URL(fileURLWithPath: env["LAUNCHER_CONTROL_OUTPUT"]!))
            return
        }
        if let encoded = env["F012_TEST_ARGUMENTS"] {
            try await BenchmarkModelDomainTests.runProcessFixture(encoded: encoded)
        }
        guard env["BENCHMARK_TEST_SCENARIO"] != nil else { return }
        var options = CLIOptions()
        options.suites = ["fixture"]
        options.format = env["BENCHMARK_TEST_FORMAT"] ?? "json"
        options.outputPath = env["BENCHMARK_TEST_OUTPUT"]!
        options.strictCorrectness = env["BENCHMARK_TEST_STRICT"] != "0"
        options.abCompare = false
        exit(await BenchmarkRunner.run(options: options, registry: ["fixture": FixtureSuite.self]))
    }

    func testFixturePreservesEnvironmentAndMarksChild() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("launcher-environment-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let capture = root.appendingPathComponent("environment.json")
        let inherited = ProcessInfo.processInfo.environment
        let sentinels = ["ASAN_OPTIONS": "detect_leaks=0", "TSAN_OPTIONS": "halt_on_error=1",
                         "SWIFT_BACKTRACE": "enable=no",
                         "DYLD_LIBRARY_PATH": (inherited["DYLD_LIBRARY_PATH"] ?? "") + ":/tmp/vectorcore-library-sentinel",
                         "DYLD_FRAMEWORK_PATH": (inherited["DYLD_FRAMEWORK_PATH"] ?? "") + ":/tmp/vectorcore-framework-sentinel"]
        var environment = sentinels
        environment["LAUNCHER_CONTROL"] = "environment"
        environment["LAUNCHER_CONTROL_OUTPUT"] = capture.path
        XCTAssertEqual(try Self.launchFixture(environment: environment, output: root.appendingPathComponent("child.log")), 0)
        let observed = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: capture))
        for (key, value) in sentinels { XCTAssertEqual(observed[key], value, key) }
        for key in ["VECTORCORE_TEST_EXTENDED", "VECTORCORE_STRICT_PERF"] {
            XCTAssertEqual(observed[key], inherited[key], key)
        }
        XCTAssertEqual(observed["VECTORCORE_CHILD_PROBE"], "BenchmarkContractTests.BenchmarkRunnerContractTests/testProcessFixture")
        XCTAssertEqual(observed["fixturePID"], observed["fixturePGID"])
        // These are environment-string checks; sanitizer runtime loading is not established here.
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
        let environment = ["BENCHMARK_TEST_SCENARIO": scenario, "BENCHMARK_TEST_OUTPUT": artifact.path,
                           "BENCHMARK_TEST_STRICT": strict ? "1" : "0", "BENCHMARK_TEST_FORMAT": format]
        let output = artifact.deletingLastPathComponent().appendingPathComponent("\(scenario)-\(format).log")
        return try Self.launchFixture(environment: environment, output: output)
    }

    // Both runner and model-domain fixtures use the shared group launcher and ordinary budget.
    static func launchFixture(environment: [String: String], output: URL) throws -> Int32 {
        let run = try runFixture(environment: environment, output: output)
        if run.timedOut {
            throw NSError(domain: "BenchmarkRunnerContractTests.childTimeout", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                            "child deadline; pid=\(run.processIdentifier) pgid=\(run.processGroup) cleaned=\(run.cleaned) "
                            + "status=\(run.statusDescription) reason=\(run.reasonDescription) log=\(output.path)"])
        }
        return try run.reaped().status
    }

    private static func runFixture(environment: [String: String], output: URL,
                                   deadline: Double = ChildProbeBudget.standard.deadline,
                                   cleanup: ChildProbeCleanup = ChildProbeBudget.standard.cleanup) throws -> ChildProbeRun {
        var inherited = ProcessInfo.processInfo.environment
        inherited.merge(environment) { _, supplied in supplied }
        _ = FileManager.default.createFile(atPath: output.path, contents: nil)
        let handle = try FileHandle(forWritingTo: output)
        defer { try? handle.close() }
        return try launchChildProbe(.xctest(selector: "BenchmarkContractTests.BenchmarkRunnerContractTests/testProcessFixture",
                                           bundlePath: Bundle(for: Self.self).bundlePath),
                                    environment: inherited, output: handle, deadline: deadline, cleanup: cleanup)
    }

    func testFixtureOrdinaryExitAndFailure() throws {
        let output = FileManager.default.temporaryDirectory.appendingPathComponent("launcher-exits-\(UUID().uuidString).log")
        defer { try? FileManager.default.removeItem(at: output) }
        for (control, expected): (String, Int32) in [("success", 0), ("failure", 7)] {
            let run = try Self.runFixture(environment: ["LAUNCHER_CONTROL": control], output: output)
            XCTAssertFalse(run.timedOut)
            XCTAssertTrue(run.cleaned)
            XCTAssertEqual(try run.reaped(), ChildProbeTermination(status: expected, reason: .exit))
            XCTAssertEqual(run.processIdentifier, run.processGroup)
            XCTAssertTrue(try String(contentsOf: output, encoding: .utf8).contains("LAUNCHER_CONTROL_READY \(control)"))
        }
    }

    func testFixtureHangHasBoundedGroupCleanup() throws {
        let output = FileManager.default.temporaryDirectory.appendingPathComponent("launcher-hang-\(UUID().uuidString).log")
        defer { try? FileManager.default.removeItem(at: output) }
        let start = ContinuousClock.now
        // Deliberate deadline control: keep independent of the ordinary probe budget.
        let run = try Self.runFixture(environment: ["LAUNCHER_CONTROL": "hang"], output: output,
                                      deadline: 3, cleanup: .kill(seconds: 2))
        XCTAssertLessThan(start.duration(to: .now), .seconds(6))
        XCTAssertTrue(run.timedOut)
        XCTAssertTrue(run.cleaned)
        XCTAssertEqual(try run.reaped(), ChildProbeTermination(status: SIGKILL, reason: .uncaughtSignal))
        XCTAssertTrue(try String(contentsOf: output, encoding: .utf8).contains("LAUNCHER_CONTROL_READY hang"))
        XCTAssertEqual(kill(run.processIdentifier, 0), -1)
        XCTAssertEqual(errno, ESRCH)
    }

    func testEnvCaptureDescendantEnvironmentAndGroupsAreObserved() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("launcher-metadata-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for name in ["git", "swiftc"] {
            let script = root.appendingPathComponent(name)
            let body = """
            #!/bin/sh
            group=$(/bin/ps -o pgid= -p $$ | /usr/bin/tr -d ' ')
            printf '%s\\n' "name=\(name) pid=$$ ppid=$PPID pgid=$group marker=$VECTORCORE_CHILD_PROBE asan=$ASAN_OPTIONS tsan=$TSAN_OPTIONS"
            """
            try body.write(to: script, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        }
        let artifact = root.appendingPathComponent("metadata.json")
        let run = try Self.runFixture(environment: ["BENCHMARK_TEST_SCENARIO": "match", "BENCHMARK_TEST_OUTPUT": artifact.path,
                                                    "PATH": root.path + ":/usr/bin:/bin", "ASAN_OPTIONS": "detect_leaks=0",
                                                    "TSAN_OPTIONS": "halt_on_error=1"], output: root.appendingPathComponent("child.log"))
        if run.timedOut { throw NSError(domain: "metadata fixture deadline", code: 1) }
        XCTAssertEqual(try run.reaped(), ChildProbeTermination(status: 0, reason: .exit))
        let metadata = try JSONDecoder().decode(BenchRun.self, from: Data(contentsOf: artifact)).metadata
        for (name, observation) in [("git", metadata.runtimeCheckout), ("swiftc", metadata.runtimeToolchain)] {
            let value = try XCTUnwrap(observation.value)
            let fields = Dictionary(uniqueKeysWithValues: value.split(separator: " ").compactMap { field -> (String, String)? in
                let parts = field.split(separator: "=", maxSplits: 1)
                return parts.count == 2 ? (String(parts[0]), String(parts[1])) : nil
            })
            XCTAssertEqual(fields["name"], name)
            XCTAssertEqual(fields["ppid"], String(run.processIdentifier))
            XCTAssertGreaterThan(try XCTUnwrap(fields["pid"].flatMap(Int32.init)), 1)
            XCTAssertGreaterThan(try XCTUnwrap(fields["pgid"].flatMap(Int32.init)), 1)
            XCTAssertEqual(fields["marker"], "BenchmarkContractTests.BenchmarkRunnerContractTests/testProcessFixture")
            XCTAssertEqual(fields["asan"], "detect_leaks=0")
            XCTAssertEqual(fields["tsan"], "halt_on_error=1")
            // Group equality is an observation of Foundation.Process on the executing toolchain.
            // No assertion here claims that a descendant which starts another group is contained.
            print("ENV_CAPTURE_DESCENDANT leader=\(run.processIdentifier) group=\(run.processGroup) \(value)")
        }
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
