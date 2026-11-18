import VectorCoreTestSupport
import Foundation
import Testing
@testable import VectorCore

@Suite struct TuningInputBoundaryTests {
    @Test func invalidTuningChild() {
        guard let scenario = ProcessInfo.processInfo.environment["F008_INVALID_TUNING"] else { return }
        let threshold = scenario == "threshold" ? -1 : scenario == "threshold-min" ? Int.min : 0
        let chunk = scenario == "chunk-zero" ? 0 : scenario == "chunk-negative" ? -1 : scenario == "chunk-min" ? Int.min : 1
        _ = Tuning(breakEvenN: threshold, minChunk: chunk, modeBias: .neutral)
        print("F008_INVALID_TUNING_ACCEPTED")
    }

    @Test(.enabled(if: childProbeProcessesSupported, "Requires macOS child-process isolation"), arguments: ["threshold", "threshold-min", "chunk-zero", "chunk-negative", "chunk-min"])
    func invalidTuningRejectsAtConstruction(_ scenario: String) throws {
        let environment = ProcessInfo.processInfo.environment.merging([
            "F008_INVALID_TUNING": scenario, "SWIFT_BACKTRACE": "enable=no"
        ]) { _, new in new }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("F008-\(scenario)-\(UUID().uuidString).log")
        _ = FileManager.default.createFile(atPath: url.path, contents: nil)
        let output = try FileHandle(forWritingTo: url)
        defer { try? output.close() }
        let result = try launchChildProbe(.swiftTesting(filter: "TuningInputBoundaryTests/invalidTuningChild"),
                                          environment: environment, output: output, deadline: ChildProbeBudget.standard.deadline, cleanup: ChildProbeBudget.standard.cleanup)
        let diagnostic = try String(contentsOf: url, encoding: .utf8)
        try #require(!result.timedOut, "Child timeout: \(url.path)")
        #expect(result.terminationReason == .uncaughtSignal)
        #expect(!diagnostic.contains("F008_INVALID_TUNING_ACCEPTED"))
        #if DEBUG
        #expect(diagnostic.contains(scenario.hasPrefix("threshold") ? "breakEvenN must be nonnegative" : "minChunk must be positive"))
        #endif
        print("F008 child \(scenario): \(url.path)")
    }
}
