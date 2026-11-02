import Testing
import Foundation
import Darwin
import VectorCore

// D03 public-consumer child. This file imports VectorCore without @testable, so
// the child exercises only the public Logger/LogConfiguration surface against the
// real global configuration and the real OSLog/Debug-console backend.
//
// The child is a fresh process (the test binary re-invoked with D03_CHILD set), so
// its global minimumLevel changes cannot affect the parent or any other suite.
// Parent suite serialization is not what provides that isolation.

/// Result of one bounded child run. Termination properties are only read after the
/// child is known to have exited within the deadline.
struct D03ChildResult {
    let status: Int32
    let reason: Int
    let output: String
    let path: String
}

private let d03Levels: [LogLevel] = [.debug, .info, .warning, .error, .critical]

/// Runs the test binary again, filtered to `filter`, with `D03_CHILD=scenario`.
/// stdout and stderr go to a file under `FileManager.default.temporaryDirectory`.
/// The child is launched by `launchChildProbe` (ChildProbeLauncher.swift), which drops the
/// parent's selection flags and report sinks (`--xunit-output <path>` among them), so a child is
/// no longer given the parent's report path (the argument filter is pinned by
/// ChildProbeLauncherTests.selectionIsReplacedNotEdited). Every child is still awaited inside its
/// parent test. Deadline: 15 s execution, then SIGTERM + 1 s and SIGKILL + 1 s to the child's
/// process group. Throws on timeout before any termination property is read.
func d03RunChild(scenario: String, filter: String) throws -> D03ChildResult {
    var env = ProcessInfo.processInfo.environment
    env["D03_CHILD"] = scenario
    env["SWIFT_BACKTRACE"] = "enable=no"
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("D03-\(scenario)-\(UUID().uuidString).log")
    _ = FileManager.default.createFile(atPath: url.path, contents: nil)
    let handle = try FileHandle(forWritingTo: url)
    defer { try? handle.close() }
    let process = try launchChildProbe(.swiftTesting(filter: filter), environment: env, output: handle, deadline: 15,
                                       cleanup: .terminateThenKill(grace: 1, seconds: 1))
    if process.timedOut {
        let cleanup = process.cleanup
        try handle.close()
        print("D03_CHILD_TIMEOUT \(scenario) cleanupReaped=\(cleanup == .success) path=\(url.path)")
        // No termination property is read after a deadline, even if cleanup timed out.
        throw NSError(domain: "D03.childTimeout", code: cleanup == .success ? 1 : 2,
                      userInfo: [NSLocalizedDescriptionKey: "\(scenario): child deadline; capture=\(url.path)"])
    }
    try handle.close()
    let output = try String(contentsOf: url, encoding: .utf8)
    let termination = try process.reaped()
    let result = D03ChildResult(status: termination.status,
                                reason: termination.reason.rawValue,
                                output: output, path: url.path)
    print("D03_CHILD_RESULT \(scenario) status=\(result.status) reason=\(result.reason) path=\(url.path)")
    return result
}

private final class D03Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    func next() -> Int { lock.lock(); defer { lock.unlock() }; value += 1; return value }
    var count: Int { lock.lock(); defer { lock.unlock() }; return value }
}

@Suite struct LoggingPublicConsumerTests {
    @Test func publicConsumerChild() throws {
        let result = try d03RunChild(scenario: "public",
                                     filter: "LoggingPublicConsumerTests/childProbe")
        #expect(result.status == 0, "public child failed: \(result.output.prefix(4000))")
        #expect(result.output.contains("D03_PUBLIC_COMPLETE"))
    }

    /// Runs only inside the child process. Uses the real public initializer and the
    /// real global configuration; changes only minimumLevel and restores it.
    @Test func childProbe() {
        guard ProcessInfo.processInfo.environment["D03_CHILD"] == "public" else { return }
        let logger = Logger(subsystem: "com.vectorcore.d03.consumer", category: "D03Public")
        let original = Logger.configuration.minimumLevel
        var mismatches: [String] = []
        for minimum in d03Levels {
            Logger.configuration.minimumLevel = minimum
            for level in d03Levels {
                let counter = D03Counter()
                switch level {
                case .debug: logger.debug("d03 \(counter.next())")
                case .info: logger.info("d03 \(counter.next())")
                case .warning: logger.warning("d03 \(counter.next())")
                case .error: logger.error("d03 \(counter.next())")
                case .critical: logger.critical("d03 \(counter.next())")
                }
                let expected = level >= minimum ? 1 : 0
                print("D03_PUBLIC_CASE level=\(level) minimum=\(minimum) evaluated=\(counter.count) expected=\(expected)")
                if counter.count != expected {
                    mismatches.append("\(level)@\(minimum): evaluated=\(counter.count) expected=\(expected)")
                }
            }
        }
        Logger.configuration.minimumLevel = original
        #expect(Logger.configuration.minimumLevel == original)
        #expect(mismatches.isEmpty, "message evaluation mismatches: \(mismatches)")
        // Public timer smoke: constructs and logs through the public surface only.
        let timer = PerformanceTimer(operation: "d03", logger: logger)
        timer.log(threshold: 3600)
        timer.log(threshold: .nan)
        print("D03_PUBLIC_COMPLETE mismatches=\(mismatches.count)")
    }
}
