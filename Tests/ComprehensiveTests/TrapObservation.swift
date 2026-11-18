import Foundation
import VectorCoreTestSupport
import Darwin
import Testing

/// Written directly to the descriptor so a later signal cannot discard buffered stage evidence.
func trapStage(_ text: String) {
    FileHandle.standardOutput.write(Data((text + "\n").utf8))
}

/// A width query is an observable boundary in the adversarial dimension fixtures.
/// Guard-removal evidence for these fixtures is recorded by Scripts/ci/trap_guard_mutations.py.
func trapDimensionRead(_ width: Int) -> Int {
    trapStage("TRAP_DIMENSION_READ")
    return width
}

/// Returns concrete reasons an observed child cannot establish the requested rejection.
/// `dimensionReads` counts only queries after the flushed pre-call marker, excluding fixture setup.
func trapObservationFailures(_ run: ChildProbeRun, output: String, id: String,
                             diagnostic: String?, dimensionReads: Int? = nil,
                             forbidden: [String] = []) -> [String] {
    var failures: [String] = []
    if run.timedOut || !run.cleaned { failures.append("child did not finish and reap within budget") }
    if run.terminationReason != .uncaughtSignal { failures.append("expected trap signal") }
    if ![SIGTRAP, SIGILL].contains(run.terminationStatus ?? 0) { failures.append("unexpected signal class") }
    let before = "TRAP_BEFORE \(id)\n"
    let returned = "TRAP_RETURNED \(id)"
    guard let boundary = output.range(of: before) else {
        return failures + ["missing flushed pre-call marker"]
    }
    let after = String(output[boundary.upperBound...])
    if after.contains(returned) { failures.append("invalid call returned") }
    for marker in forbidden where after.contains(marker) { failures.append("downstream stage reached: \(marker)") }
    if let expected = dimensionReads {
        let actual = after.components(separatedBy: "TRAP_DIMENSION_READ\n").count - 1
        if actual != expected { failures.append("dimension reads: expected \(expected), observed \(actual)") }
    }
    #if DEBUG
    if let diagnostic, !after.contains(diagnostic) { failures.append("missing named diagnostic: \(diagnostic)") }
    #endif
    return failures
}

@Suite struct TrapObservationTests {
    @Test func controlledSignalChild() {
        guard let mode = ProcessInfo.processInfo.environment["TRAP_OBSERVATION_CONTROL"] else { return }
        if mode != "before" { trapStage("TRAP_BEFORE control") }
        if mode == "downstream" { trapStage("CONTROL_DOWNSTREAM") }
        raise(mode == "unrelated" ? SIGABRT : SIGTRAP)
    }

    @Test(.enabled(if: childProbeProcessesSupported, "Requires macOS child-process isolation"), arguments: ["before", "unrelated", "downstream"])
    func unrelatedSignalsCannotSatisfyObservation(_ mode: String) throws {
        var env = ProcessInfo.processInfo.environment
        env["TRAP_OBSERVATION_CONTROL"] = mode
        env["SWIFT_BACKTRACE"] = "enable=no"
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("trap-control-\(UUID().uuidString).log")
        defer { try? FileManager.default.removeItem(at: path) }
        _ = FileManager.default.createFile(atPath: path.path, contents: nil)
        let handle = try FileHandle(forWritingTo: path)
        defer { try? handle.close() }
        let run = try launchChildProbe(.swiftTesting(filter: "TrapObservationTests/controlledSignalChild"),
                                       environment: env, output: handle, deadline: ChildProbeBudget.standard.deadline,
                                       cleanup: ChildProbeBudget.standard.cleanup)
        try handle.close()
        let output = try String(contentsOf: path, encoding: .utf8)
        let failures = trapObservationFailures(run, output: output, id: "control", diagnostic: nil,
                                               forbidden: ["CONTROL_DOWNSTREAM"])
        let expected = mode == "before" ? "missing flushed pre-call marker" :
            mode == "unrelated" ? "unexpected signal class" : "downstream stage reached: CONTROL_DOWNSTREAM"
        #expect(failures.contains(expected), "\(mode): \(failures): \(output)")
    }
}
