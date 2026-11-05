import XCTest
import Foundation
import VectorCoreTestSupport
@testable import VectorCoreBenchSupport
@testable import VectorCoreBenchmarking
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
        if let command = env["METADATA_DIRECT_COMMAND"] {
            let argv = try JSONDecoder().decode([String].self, from: Data(command.utf8))
            var targetEnvironment = env
            if let value = env["METADATA_TEST_LOADER_VALUE"], let key = env["METADATA_TEST_LOADER_KEY"] {
                targetEnvironment[key] = value
            }
            if let encoded = env["METADATA_DIRECT_ENVIRONMENT"] {
                targetEnvironment.merge(try JSONDecoder().decode([String: String].self, from: Data(encoded.utf8))) { _, supplied in supplied }
            }
            do {
                try replaceMarkedChildProbe(command: argv, environment: targetEnvironment)
            } catch {
                guard let result = env["METADATA_DIRECT_ERROR_RESULT"] else { throw error }
                try (error as NSError).domain.write(toFile: result, atomically: true, encoding: .utf8)
                return
            }
        }
        if let command = env["METADATA_TEST_COMMAND"], let result = env["METADATA_TEST_RESULT"] {
            let argv = try JSONDecoder().decode([String].self, from: Data(command.utf8))
                .map { $0 == "@OWNER_PID@" ? String(getpid()) : $0 }
            if env["METADATA_TEST_IGNORED_SIGNALS"] == "1" {
                var mask = sigset_t()
                sigemptyset(&mask)
                for number in [SIGINT, SIGQUIT, SIGTERM] {
                    signal(number, SIG_IGN)
                    sigaddset(&mask, number)
                }
                pthread_sigmask(SIG_BLOCK, &mask, nil)
            }
            if let value = env["METADATA_TEST_LOADER_VALUE"] {
                setenv(env["METADATA_TEST_LOADER_KEY"] ?? "DYLD_LIBRARY_PATH", value, 1)
            }
            let observation = EnvCapture.observe(command: argv, timeout: Double(env["METADATA_TEST_TIMEOUT"] ?? "2")!)
            try JSONEncoder().encode(observation).write(to: URL(fileURLWithPath: result))
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
            // The metadata command is now a child of the dedicated guardian.
            XCTAssertNotEqual(fields["ppid"], String(run.processIdentifier))
            XCTAssertEqual(fields["ppid"], fields["pgid"])
            XCTAssertGreaterThan(try XCTUnwrap(fields["pid"].flatMap(Int32.init)), 1)
            XCTAssertGreaterThan(try XCTUnwrap(fields["pgid"].flatMap(Int32.init)), 1)
            XCTAssertEqual(fields["marker"], "BenchmarkContractTests.BenchmarkRunnerContractTests/testProcessFixture")
            XCTAssertEqual(fields["asan"], "detect_leaks=0")
            XCTAssertEqual(fields["tsan"], "halt_on_error=1")
            // A command that deliberately starts another group is outside metadata ownership.
            print("ENV_CAPTURE_DESCENDANT leader=\(run.processIdentifier) group=\(run.processGroup) \(value)")
        }
    }

    // Removing the group sweep must fail both the normal-exit and deadline rows.
    func testMetadataLifetimeNormalExitAndTimeout() throws {
        for scenario in ["exit", "failure", "timeout"] {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent("metadata-lifetime-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let pidFile = root.appendingPathComponent("pid")
            let control = root.appendingPathComponent("continue")
            try Data().write(to: control)
            let script = """
            /bin/sh -c 'i=0; while [ -f "$1" ] && [ "$i" -lt 1500 ]; do /bin/sleep 0.02; i=$((i+1)); done' leaf "$3" &
            child=$!
            printf '%s' "$child" > "$1"
            /bin/ps -o pid=,ppid=,pgid= -p $$ -p "$child"
            if [ "$2" = exit ]; then printf 'complete'; exit 0; fi
            if [ "$2" = failure ]; then exit 7; fi
            wait
            """
            let result = root.appendingPathComponent("result.json")
            let argv = ["/bin/sh", "-c", script, "fixture", pidFile.path, scenario, control.path]
            let run = try Self.runFixture(environment: [
                "METADATA_TEST_COMMAND": try XCTUnwrap(String(data: JSONEncoder().encode(argv), encoding: .utf8)),
                "METADATA_TEST_RESULT": result.path, "METADATA_TEST_TIMEOUT": "1"
            ], output: root.appendingPathComponent("child.log"))
            XCTAssertFalse(run.timedOut)
            XCTAssertEqual(try run.reaped().status, 0)
            let pid = try XCTUnwrap(Int32(String(contentsOf: pidFile, encoding: .utf8)))
            defer { Self.finishMetadataFixture(control: control, pid: pid) }
            let end = ContinuousClock.now.advanced(by: .seconds(2))
            while kill(pid, 0) == 0 && ContinuousClock.now < end { usleep(10_000) }
            XCTAssertEqual(kill(pid, 0), -1, "left descendant alive: \(scenario), pid=\(pid)")
            let observation = try JSONDecoder().decode(RuntimeObservation.self, from: Data(contentsOf: result))
            if scenario == "exit" {
                XCTAssertTrue(observation.value?.hasSuffix("complete") == true)
                print("METADATA_LIFETIME \(scenario) \(observation.value ?? "missing")")
            } else {
                XCTAssertTrue(observation.unavailableReason?.contains(scenario == "timeout" ? "timeout" : "failed: 7") == true)
            }
        }
    }

    func testMetadataTimerSurvivesObserverAndGuardianDeath() throws {
        for victim in ["observer", "guardian"] {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent("metadata-death-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let pidFile = root.appendingPathComponent("pid")
            let control = root.appendingPathComponent("continue")
            try Data().write(to: control)
            let script = """
            printf '%s' "$$" > "$1"
            /bin/ps -axo pid=,ppid=,pgid=,comm= | /usr/bin/awk -v group="$PPID" '$3 == group' > "$1.groups"
            if [ "$3" = observer ]; then kill -KILL "$2"; else kill -KILL "$PPID"; fi
            i=0; while [ -f "$4" ] && [ "$i" -lt 1500 ]; do /bin/sleep 0.02; i=$((i+1)); done
            """
            let argv = ["/bin/sh", "-c", script, "fixture", pidFile.path, "@OWNER_PID@", victim, control.path]
            let run = try Self.runFixture(environment: [
                "METADATA_TEST_COMMAND": try XCTUnwrap(String(data: JSONEncoder().encode(argv), encoding: .utf8)),
                "METADATA_TEST_RESULT": root.appendingPathComponent("result.json").path,
                "METADATA_TEST_TIMEOUT": "1", "METADATA_TEST_IGNORED_SIGNALS": "1"
            ], output: root.appendingPathComponent("child.log"))
            XCTAssertFalse(run.timedOut)
            let pid = try XCTUnwrap(Int32(String(contentsOf: pidFile, encoding: .utf8)))
            defer { Self.finishMetadataFixture(control: control, pid: pid) }
            let end = ContinuousClock.now.advanced(by: .seconds(3))
            while kill(pid, 0) == 0 && ContinuousClock.now < end { usleep(10_000) }
            XCTAssertEqual(kill(pid, 0), -1, "timer lost after \(victim) death, pid=\(pid)")
            let groups = try String(contentsOfFile: pidFile.path + ".groups", encoding: .utf8)
            print("METADATA_DEATH \(victim) outer=\(run.processIdentifier) \(groups)")
            if victim == "observer" {
                XCTAssertEqual(try run.reaped(), ChildProbeTermination(status: SIGKILL, reason: .uncaughtSignal))
            } else {
                let data = try Data(contentsOf: root.appendingPathComponent("result.json"))
                let observation = try JSONDecoder().decode(RuntimeObservation.self, from: data)
                XCTAssertNil(observation.value)
                XCTAssertNotNil(observation.unavailableReason)
            }
        }
    }

    // Fixture cleanup uses its private control file, never a possibly recycled numeric PID.
    private static func finishMetadataFixture(control: URL, pid: pid_t) {
        try? FileManager.default.removeItem(at: control)
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while kill(pid, 0) == 0 && ContinuousClock.now < deadline { usleep(10_000) }
        XCTAssertEqual(kill(pid, 0), -1, "fixture did not finish after independent cleanup")
    }

    func testMetadataEscapedOutputHandleDoesNotBlockObservation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("metadata-escape-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let control = root.appendingPathComponent("continue")
        try Data().write(to: control)
        let pidFile = root.appendingPathComponent("pid")
        let script = """
        set -m
        /bin/sh -c 'i=0; while [ -f "$1" ] && [ "$i" -lt 1500 ]; do /bin/sleep 0.02; i=$((i+1)); done' escaped "$1" &
        printf '%s' "$!" > "$2"
        printf escaped-output
        """
        let argv = ["/bin/sh", "-c", script, "fixture", control.path, pidFile.path]
        let result = root.appendingPathComponent("result.json")
        let run = try Self.runFixture(environment: [
            "METADATA_TEST_COMMAND": try XCTUnwrap(String(data: JSONEncoder().encode(argv), encoding: .utf8)),
            "METADATA_TEST_RESULT": result.path, "METADATA_TEST_TIMEOUT": "0.5"
        ], output: root.appendingPathComponent("child.log"))
        XCTAssertFalse(run.timedOut)
        XCTAssertEqual(try run.reaped().status, 0)
        let pid = try XCTUnwrap(Int32(String(contentsOf: pidFile, encoding: .utf8)))
        defer { Self.finishMetadataFixture(control: control, pid: pid) }
        let observation = try JSONDecoder().decode(RuntimeObservation.self, from: Data(contentsOf: result))
        XCTAssertEqual(observation.value, "escaped-output")
        // Deliberate group escape is not contained; its inherited output descriptor cannot block us.
        XCTAssertEqual(kill(pid, 0), 0)
    }

    private static let metadataEnvironmentReaderSource = #"""
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <unistd.h>
    int main(int argc, char **argv) {
        if (argc < 2 || argc > 3) return 64;
        FILE *output = stdout;
        const char *capture = getenv("VECTORCORE_METADATA_BASELINE_OUTPUT");
        if (capture != NULL) {
            output = fopen(capture, "wb");
            if (output == NULL) return 74;
            fprintf(stderr, "METADATA_BASELINE_TARGET pid=%d pgid=%d\n", getpid(), getpgrp());
        }
        if (argc == 3) {
            if (strcmp(argv[2], "strip") == 0) {
                if (unsetenv(argv[1]) != 0) return 74;
            } else if (strcmp(argv[2], "error") == 0) {
                fputs("partial output is not a value", output);
                return 7;
            } else return 64;
        }
        const char *value = getenv(argv[1]);
        if (value == NULL) return 1;
        return fprintf(output, "%s\n", value) < 0 ? 74 : 0;
    }
    """#

    private static func makeMetadataEnvironmentReader(in root: URL, source: String) throws -> URL {
        let architecture: String
        #if arch(arm64)
        architecture = "arm64"
        #elseif arch(x86_64)
        architecture = "x86_64"
        #else
        throw NSError(domain: "MetadataEnvironmentReader.architecture", code: 1)
        #endif
        let reader = root.appendingPathComponent("environment-reader")
        let sourceFile = root.appendingPathComponent("environment-reader.c")
        let compilerLog = root.appendingPathComponent("compiler.log")
        try source.write(to: sourceFile, atomically: true, encoding: .utf8)

        func observeSetup(_ command: [String], label: String, timeout: String = "2",
                          environment: [String: String] = [:]) throws -> RuntimeObservation {
            let result = root.appendingPathComponent(label + ".json")
            var requested = environment
            requested["METADATA_TEST_COMMAND"] = try XCTUnwrap(String(data: JSONEncoder().encode(command), encoding: .utf8))
            requested["METADATA_TEST_RESULT"] = result.path
            requested["METADATA_TEST_TIMEOUT"] = timeout
            let run = try Self.runFixture(environment: requested, output: root.appendingPathComponent(label + ".log"))
            guard !run.timedOut, try run.reaped().status == 0 else {
                throw NSError(domain: "MetadataEnvironmentReader.fixture", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: label + " outer fixture failed"])
            }
            return try JSONDecoder().decode(RuntimeObservation.self, from: Data(contentsOf: result))
        }

        // Build a test-owned reader for the explicit ordinary host architecture.
        // The adjacent architecture and execution preflights enforce that fixture contract.
        let script = #"""
        set -e
        /usr/bin/xcrun --sdk macosx --find clang > "$4"
        /usr/bin/xcrun --sdk macosx clang -v -arch "$3" "$1" -o "$2" >> "$4" 2>&1
        printf METADATA_READER_COMPILED
        """#
        let compile = ["/bin/sh", "-c", script, "metadata-reader-build", sourceFile.path,
                       reader.path, architecture, compilerLog.path]
        let built = try observeSetup(compile, label: "compile", timeout: "10")
        let diagnostics = (try? String(contentsOf: compilerLog, encoding: .utf8)) ?? "compiler log unavailable"
        print("METADATA_READER_BUILD command=\(compile) architecture=\(architecture)\n\(diagnostics)")
        guard built.value == "METADATA_READER_COMPILED" else {
            throw NSError(domain: "MetadataEnvironmentReader.compile", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "\(built); \(diagnostics)"])
        }
        let inspected = try observeSetup(["/usr/bin/lipo", "-archs", reader.path], label: "architecture")
        guard inspected.value == architecture else {
            throw NSError(domain: "MetadataEnvironmentReader.architecture", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Expected \(architecture): \(inspected)"])
        }
        let executed = try observeSetup([reader.path, "VECTORCORE_METADATA_READER_READY"], label: "execute",
                                       environment: ["VECTORCORE_METADATA_READER_READY": "native-reader-ready"])
        guard executed.value == "native-reader-ready" else {
            throw NSError(domain: "MetadataEnvironmentReader.execution", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "\(executed)"])
        }
        return reader
    }

    func testMetadataEnvironmentReaderRejectsFailedBuild() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("metadata-invalid-reader-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertThrowsError(try Self.makeMetadataEnvironmentReader(in: root, source: "#error intentional fixture build failure\n")) { error in
            XCTAssertEqual((error as NSError).domain, "MetadataEnvironmentReader.compile")
            print("METADATA_READER_EXPECTED_BUILD_FAILURE \(error)")
        }
    }

    private static func metadataEnvironmentWasStripped(termination: ChildProbeTermination?, timedOut: Bool,
                                                       cleaned: Bool, output: Data, expected: String) throws -> Bool {
        guard !timedOut, cleaned, let termination, termination.reason == .exit else {
            throw NSError(domain: "MetadataEnvironmentBaseline.execution", code: 1)
        }
        if termination.status == 0 && output == Data((expected + "\n").utf8) { return false }
        if termination.status == 1 && output.isEmpty { return true }
        throw NSError(domain: "MetadataEnvironmentBaseline.output", code: Int(termination.status))
    }

    private static func metadataExpectedStripping(key: String, protected: Bool, baseline: Bool) -> Bool {
        protected && key.hasPrefix("DYLD_") && baseline
    }

    func testMetadataEnvironmentBaselineRejectsAmbiguousResults() throws {
        for key in ["DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH"] {
            for baseline in [false, true] {
                XCTAssertEqual(Self.metadataExpectedStripping(key: key, protected: true, baseline: baseline), baseline)
                XCTAssertFalse(Self.metadataExpectedStripping(key: key, protected: false, baseline: baseline))
            }
        }
        let literal = "spaces; $(touch never) ' \" `touch never`"
        let bytes = Data((literal + "\n").utf8)
        XCTAssertFalse(try Self.metadataEnvironmentWasStripped(termination: .init(status: 0, reason: .exit),
                           timedOut: false, cleaned: true, output: bytes, expected: literal))
        XCTAssertTrue(try Self.metadataEnvironmentWasStripped(termination: .init(status: 1, reason: .exit),
                          timedOut: false, cleaned: true, output: Data(), expected: literal))
        let invalid: [(ChildProbeTermination?, Bool, Bool, Data)] = [
            (.init(status: 0, reason: .exit), false, true, Data()),
            (.init(status: 0, reason: .exit), false, true, Data("wrong value\n".utf8)),
            (.init(status: 1, reason: .exit), false, true, bytes),
            (.init(status: 7, reason: .exit), false, true, Data()),
            (.init(status: 1, reason: .uncaughtSignal), false, true, Data()),
            (.init(status: 0, reason: .exit), true, true, bytes),
            (.init(status: 0, reason: .exit), false, false, bytes),
            (nil, false, true, Data())
        ]
        for (termination, timedOut, cleaned, output) in invalid {
            XCTAssertThrowsError(try Self.metadataEnvironmentWasStripped(termination: termination,
                                 timedOut: timedOut, cleaned: cleaned, output: output, expected: literal))
        }
    }

    private static func metadataDirectBaseline(command: [String], key: String, value: String,
                                               environment: [String: String], root: URL) throws -> Bool {
        let capture = root.appendingPathComponent("baseline-" + UUID().uuidString)
        let log = capture.appendingPathExtension("log")
        var requested = environment
        requested["METADATA_DIRECT_COMMAND"] = try XCTUnwrap(String(data: JSONEncoder().encode(command), encoding: .utf8))
        requested["METADATA_TEST_LOADER_KEY"] = key
        requested["METADATA_TEST_LOADER_VALUE"] = value
        requested["VECTORCORE_METADATA_BASELINE_OUTPUT"] = capture.path
        let run = try Self.runFixture(environment: requested, output: log)
        let diagnostics = try String(contentsOf: log, encoding: .utf8)
        guard diagnostics.contains("METADATA_BASELINE_TARGET pid=\(run.processIdentifier) pgid=\(run.processGroup)") else {
            throw NSError(domain: "MetadataEnvironmentBaseline.identity", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: diagnostics])
        }
        let bytes = try Data(contentsOf: capture)
        print("METADATA_DIRECT_RESULT command=\(command) key=\(key) status=\(run.statusDescription) reason=\(run.reasonDescription) timedOut=\(run.timedOut) cleaned=\(run.cleaned) pid=\(run.processIdentifier) pgid=\(run.processGroup) bytes=\(bytes.count)")
        let stripped = try Self.metadataEnvironmentWasStripped(termination: run.termination, timedOut: run.timedOut,
                                                              cleaned: run.cleaned, output: bytes, expected: value)
        print("METADATA_DIRECT_BASELINE command=\(command) key=\(key) stripped=\(stripped) status=\(run.statusDescription) pid=\(run.processIdentifier) pgid=\(run.processGroup) bytes=\(bytes.count)")
        return stripped
    }

    func testMetadataDirectExecRejectsInvalidRequests() throws {
        let absent = FileManager.default.temporaryDirectory.appendingPathComponent("unmarked-exec-" + UUID().uuidString)
        XCTAssertThrowsError(try replaceMarkedChildProbe(command: [absent.path], environment: [:])) { error in
            XCTAssertEqual((error as NSError).domain, "ChildProbeReplacement.unmarked")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("metadata-exec-errors-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let cases: [([String], [String: String], String)] = [
            ([], [:], "command"), (["relative"], [:], "command"),
            (["/usr/bin/true", "nul\0argument"], [:], "command"),
            (["/usr/bin/true"], ["bad=key": "value"], "environment"),
            (["/usr/bin/true"], ["bad\0key": "value"], "environment"),
            (["/usr/bin/true"], ["": "value"], "environment"),
            (["/usr/bin/true"], ["key": "bad\0value"], "environment"),
            (["/usr/bin/true"], [childProbeMarkerKey: "different"], "environment"),
            ([root.appendingPathComponent("missing-executable").path], [:], "exec")
        ]
        for (index, entry) in cases.enumerated() {
            let result = root.appendingPathComponent("error-\(index)")
            let run = try Self.runFixture(environment: [
                "METADATA_DIRECT_COMMAND": try XCTUnwrap(String(data: JSONEncoder().encode(entry.0), encoding: .utf8)),
                "METADATA_DIRECT_ENVIRONMENT": try XCTUnwrap(String(data: JSONEncoder().encode(entry.1), encoding: .utf8)),
                "METADATA_DIRECT_ERROR_RESULT": result.path
            ], output: root.appendingPathComponent("error-\(index).log"))
            XCTAssertFalse(run.timedOut)
            XCTAssertEqual(try run.reaped(), ChildProbeTermination(status: 0, reason: .exit))
            XCTAssertEqual(try String(contentsOf: result, encoding: .utf8), "ChildProbeReplacement." + entry.2)
        }
    }

    func testMetadataEnvironmentDeterministicPolicyControls() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("metadata-policy-controls-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let reader = try Self.makeMetadataEnvironmentReader(in: root, source: Self.metadataEnvironmentReaderSource)
        let marker = root.appendingPathComponent("injected")
        let value = "literal $(touch \(marker.path)); ' \" `touch \(marker.path)`"
        for key in ["DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH"] {
            for policy in ["preserve", "strip", "error"] {
                let command = [reader.path, key] + (policy == "preserve" ? [] : [policy])
                if policy == "error" {
                    XCTAssertThrowsError(try Self.metadataDirectBaseline(command: command, key: key, value: value,
                                                                         environment: [:], root: root)) { error in
                        XCTAssertEqual((error as NSError).domain, "MetadataEnvironmentBaseline.output")
                        XCTAssertEqual((error as NSError).code, 7)
                    }
                } else {
                    XCTAssertEqual(try Self.metadataDirectBaseline(command: command, key: key, value: value,
                                                                  environment: [:], root: root), policy == "strip")
                }
                let result = root.appendingPathComponent("observation.json")
                let run = try Self.runFixture(environment: [
                    "METADATA_TEST_COMMAND": try XCTUnwrap(String(data: JSONEncoder().encode(command), encoding: .utf8)),
                    "METADATA_TEST_RESULT": result.path, "METADATA_TEST_LOADER_KEY": key, "METADATA_TEST_LOADER_VALUE": value
                ], output: root.appendingPathComponent("\(key)-\(policy).log"))
                XCTAssertFalse(run.timedOut)
                XCTAssertEqual(try run.reaped(), ChildProbeTermination(status: 0, reason: .exit))
                let observation = try JSONDecoder().decode(RuntimeObservation.self, from: Data(contentsOf: result))
                XCTAssertEqual(observation.value, policy == "preserve" ? value : nil)
                XCTAssertEqual(observation.unavailableReason, policy == "preserve" ? nil
                               : "Observation command failed: " + (policy == "strip" ? "1" : "7"))
                XCTAssertFalse(FileManager.default.fileExists(atPath: marker.path))
            }
        }
    }

    func testMetadataPreservesLoaderEnvironmentWithoutShellEvaluation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("metadata-environment-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let reader = try Self.makeMetadataEnvironmentReader(in: root, source: Self.metadataEnvironmentReaderSource)
        let result = root.appendingPathComponent("result.json")
        let marker = root.appendingPathComponent("injected")
        let startup = root.appendingPathComponent("startup")
        try "/usr/bin/touch \"\(marker.path)\"\n".write(to: startup, atomically: true, encoding: .utf8)
        let value = "/tmp/$(touch \(marker.path)); spaces ' \" `touch \(marker.path)`"
        for key in ["DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH", "LD_LIBRARY_PATH", "IFS", "armed", "PWD", "SHLVL"] {
            for protected in [false, true] {
                let argv = (protected ? ["/usr/bin/env"] : []) + [reader.path, key]
                let requested = [
                    "METADATA_TEST_COMMAND": try XCTUnwrap(String(data: JSONEncoder().encode(argv), encoding: .utf8)),
                    "METADATA_TEST_RESULT": result.path, "METADATA_TEST_LOADER_VALUE": value, "METADATA_TEST_LOADER_KEY": key,
                    "ENV": startup.path, "BASH_ENV": startup.path, "IFS": ":",
                    "BASH_FUNC_printf%%": "() { /usr/bin/touch \"\(marker.path)\"; builtin printf \"$@\"; }"
                ]
                let baseline = try Self.metadataDirectBaseline(command: argv, key: key, value: value,
                                                               environment: requested, root: root)
                let stripped = Self.metadataExpectedStripping(key: key, protected: protected, baseline: baseline)
                if !(protected && key.hasPrefix("DYLD_")) { XCTAssertFalse(baseline, "key=\(key) protected=\(protected)") }
                let run = try Self.runFixture(environment: requested,
                                              output: root.appendingPathComponent("environment-\(key)-\(protected).log"))
                XCTAssertFalse(run.timedOut)
                XCTAssertEqual(try run.reaped().status, 0)
                let observation = try JSONDecoder().decode(RuntimeObservation.self, from: Data(contentsOf: result))
                XCTAssertEqual(observation.value, stripped ? nil : value, "key=\(key) protected=\(protected): \(observation)")
                XCTAssertEqual(observation.unavailableReason, stripped ? "Observation command failed: 1" : nil)
                XCTAssertFalse(FileManager.default.fileExists(atPath: marker.path))
            }
        }
    }

    func testMetadataOutputAndArgumentBoundaries() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("metadata-output-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let literal = "spaces; $(touch " + root.appendingPathComponent("injected").path + ") ' \" $HOME"
        let cases: [([String], String?)] = [
            (["/usr/bin/printf", "%s", literal], literal),
            (["/usr/bin/false"], nil),
            (["printf", "must-not-resolve-a-shell-builtin"], nil),
            (["/bin/sh", "-c", "kill -INT $$; printf SIGINT-was-ignored"], nil),
            (["/bin/sh", "-c", "kill -QUIT $$; printf SIGQUIT-was-ignored"], nil),
            (["/bin/sh", "-c", "kill -TERM $$; printf SIGTERM-was-ignored"], nil),
            (["/bin/sleep", "30"], nil),
            (["/bin/sh", "-c", "printf partial; exit 7"], nil),
            (["/bin/sh", "-c", "/usr/bin/head -c 100000 /dev/zero"], nil),
            (["/bin/sh", "-c", "printf '\\377'"], nil),
            (["/usr/bin/printf", "truncated\0argument"], nil)
        ]
        for (index, fixture) in cases.enumerated() {
            let result = root.appendingPathComponent("result.json")
            let start = ContinuousClock.now
            let run = try Self.runFixture(environment: [
                "METADATA_TEST_COMMAND": try XCTUnwrap(String(data: JSONEncoder().encode(fixture.0), encoding: .utf8)),
                "METADATA_TEST_RESULT": result.path, "METADATA_TEST_TIMEOUT": "0.5", "METADATA_TEST_IGNORED_SIGNALS": "1"
            ], output: root.appendingPathComponent("child-\(index).log"))
            XCTAssertFalse(run.timedOut)
            XCTAssertEqual(try run.reaped().status, 0)
            XCTAssertLessThan(start.duration(to: .now), .seconds(5))
            let observation = try JSONDecoder().decode(RuntimeObservation.self, from: Data(contentsOf: result))
            XCTAssertEqual(observation.value, fixture.1, "case \(index)")
            if fixture.1 == nil { XCTAssertNotNil(observation.unavailableReason, "case \(index)") }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("injected").path))
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
