import Foundation
import Testing
@testable import VectorCore

// P01 phase A shared support for the P01 contract suites.
//
// Gate: each still-RED P01 suite ran only when P01_RED=1. On the unchanged base its RED assertions failed by design;
// the phase that landed a behaviour (C: TopKBuffer, D: VectorMath, E: Operations) lifted the gate of the suite it
// turned green. Phases B (hashing), C (TopKBuffer), D (VectorMath) and E (Operations) have landed, so every P01
// suite runs by default and P01_RED no longer selects anything.
enum P01RedGate {
    static var isEnabled: Bool { ProcessInfo.processInfo.environment["P01_RED"] == "1" }
    /// One gate per suite, so the phase that turned a suite green lifted exactly its own gate.
    /// Phase C lifted `topKBuffer`, phase D lifted `vectorMath` and phase E lifted `operations`. The hash suite's
    /// gate (phase B) is gone.
    /// Lifted by phase C: the TopKBuffer zero-capacity guard landed, so `P01TopKBufferRedTests` runs by default.
    static var topKBuffer: Bool { true }
    /// Lifted by phase D: `VectorMath.nearestNeighbors` uses the common selector, so
    /// `VectorMathNearestContractTests` runs by default.
    static var vectorMath: Bool { true }
    /// Lifted by phase E: Operations generic small-k streams through `parallelReduce` without calibration and
    /// the batch fallback validates candidates once, so `P01OperationsRedTests` runs by default.
    static var operations: Bool { true }
}

/// Unbuffered marker line (written with FileHandle so it reaches a child's capture before any later trap).
func p01Marker(_ text: String) {
    FileHandle.standardOutput.write(Data((text + "\n").utf8))
}

/// Lock-protected append-only log of Int observations, shared with callbacks.
final class P01Log: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Int] = []
    func append(_ value: Int) { lock.lock(); defer { lock.unlock() }; values.append(value) }
    func reset() { lock.lock(); defer { lock.unlock() }; values.removeAll() }
    func snapshot() -> [Int] { lock.lock(); defer { lock.unlock() }; return values }
    var count: Int { snapshot().count }
}

/// The captured result of one bounded child.
struct P01ChildResult {
    let scenario: String
    let run: ChildProbeRun
    let output: String
    let capturePath: String

    /// The child's output tail for an assertion message (the trap or issue text is near the end).
    var tail: String { String(output.suffix(4000)) }
}

/// Runs one P01 child scenario through the T10 launcher: 15 s execution deadline, SIGTERM then SIGKILL of the
/// child's own process group with at most 2 s of cleanup, file capture, and `timedOut` read before any
/// termination property. `removing` drops environment keys with those prefixes (for example VC_AUTOTUNE).
func p01RunChild(_ scenario: String, filter: String, removing prefixes: [String] = []) throws -> P01ChildResult {
    var environment = ProcessInfo.processInfo.environment
    for key in environment.keys where prefixes.contains(where: { key.hasPrefix($0) }) {
        environment.removeValue(forKey: key)
    }
    environment["P01_SCENARIO"] = scenario
    // P01_CAPTURE_DIRECTORY keeps the capture file as evidence. Otherwise the capture lives in a per-run
    // temporary subdirectory that is removed on every exit path.
    let retained = environment["P01_CAPTURE_DIRECTORY"].map { URL(fileURLWithPath: $0) }
    let directory = retained
        ?? FileManager.default.temporaryDirectory.appendingPathComponent("P01-\(scenario)-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer {
        if retained == nil { try? FileManager.default.removeItem(at: directory) }
    }
    let path = directory.appendingPathComponent("P01-\(scenario)-\(UUID().uuidString).log")
    guard FileManager.default.createFile(atPath: path.path, contents: nil) else {
        throw NSError(domain: "P01.capture", code: 2, userInfo: [NSLocalizedDescriptionKey: "cannot create \(path.path)"])
    }
    let file = try FileHandle(forWritingTo: path)
    var closed = false
    defer { if !closed { try? file.close() } }
    let run = try launchChildProbe(.swiftTesting(filter: filter), environment: environment, output: file,
                                   deadline: 15, cleanup: .terminateThenKill(grace: 1, seconds: 1))
    let timedOut = run.timedOut
    closed = true
    try file.close()
    if retained != nil { print("P01_CHILD_CAPTURE \(scenario) \(path.path)") }
    if timedOut {
        print("P01_CHILD_RESULT \(scenario) timedOut=true cleaned=\(run.cleaned)")
        throw NSError(domain: "P01.childTimeout", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(scenario) timed out"])
    }
    let output = try String(contentsOf: path, encoding: .utf8)
    print("P01_CHILD_RESULT \(scenario) status=\(run.statusDescription) reason=\(run.reasonDescription)")
    return P01ChildResult(scenario: scenario, run: run, output: output, capturePath: retained == nil ? "" : path.path)
}
