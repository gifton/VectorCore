import Foundation
import Darwin
import os
import CryptoKit
import Testing
@testable import VectorCore

private func e05Marker(_ text: String) {
    FileHandle.standardOutput.write(Data((text + "\n").utf8))
}
private struct E05ReentrantMetric: DistanceMetric {
    var name: String { "E05-reentrant-metric" }
    func distance<V: VectorProtocol>(_ lhs: V, _ rhs: V) -> Float where V.Scalar == Float {
        e05Marker("E05_CALLBACK_ENTRY operations")
        _ = AutoTuning.get(dim: lhs.scalarCount, kind: .euclid)
        var sum: Float = 0
        for i in 0..<lhs.scalarCount { sum += abs(lhs[i] - rhs[i]) }
        return 3 + 2 * sum
    }
}

@Suite struct AutoTuningConcurrencyTests {
    private static let executableHash: String = {
        guard let bytes = try? Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[0])) else { return "UNREADABLE_EXECUTABLE" }
        return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }()
    private func runChild(_ scenario: String) throws {
        var environment = ProcessInfo.processInfo.environment
        environment["E05_SCENARIO"] = scenario
        environment["VC_AUTOTUNE"] = scenario == "disabled" ? "0" : "1"
        environment.removeValue(forKey: "VC_AUTOTUNE_BIAS")
        environment.removeValue(forKey: "VC_AUTOTUNE_PRINT")
        if scenario == "bias-seq" { environment["VC_AUTOTUNE_BIAS"] = "seq" }
        if scenario == "bias-parallel" { environment["VC_AUTOTUNE_BIAS"] = "parallel" }
        if scenario == "printing" { environment["VC_AUTOTUNE_PRINT"] = "1" }
        let directory = environment["E05_CHILD_LOG_DIR"] ?? "/private/tmp"
        let stem = "\(directory)/E05-child-\(scenario)-\(UUID().uuidString)"
        let output = URL(fileURLWithPath: stem + ".log")
        _ = FileManager.default.createFile(atPath: output.path, contents: nil)
        let file = try FileHandle(forWritingTo: output)
        let process = try launchChildProbe(.swiftTesting(filter: "AutoTuningConcurrencyTests/childProbe"),
                                           environment: environment, output: file, deadline: 15, cleanup: .kill(seconds: 2))
        let arguments = process.arguments, timedOut = process.timedOut, cleaned = process.cleaned
        try file.close()
        let log = try String(contentsOf: output, encoding: .utf8)
        let metadata: [String: Any] = ["executable": CommandLine.arguments[0], "executableSHA256": Self.executableHash, "arguments": arguments,
            "scenario": scenario, "environment": environment.filter { $0.key.hasPrefix("VC_AUTOTUNE") || $0.key.hasPrefix("E05_") }, "pid": process.processIdentifier,
                "timeout": timedOut, "cleaned": cleaned,
            "exit": process.terminationStatus.map(Int.init) ?? -999, "log": output.path,
            "executionDeadlineSeconds": 15, "cleanupDeadlineSeconds": 2]
        try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: stem + ".json"))
        print("E05 child scenario=\(scenario) timedOut=\(timedOut) cleaned=\(cleaned) capture=\(output.path)")
        #expect(log.contains("E05_BODY_BEGIN \(scenario)"))
        if timedOut { Issue.record("E05 child execution deadline: \(output.path); callbackEntered=\(log.contains("E05_CALLBACK_ENTRY")); cleaned=\(cleaned)"); return }
        #expect(process.terminationStatus == 0, "\(log)")
        #expect(log.contains("E05_RESULT \(scenario)"), "\(log)")
        #expect(log.contains("E05_BODY_END \(scenario)"), "\(log)")
    }
    @Test func directReentrantCallback() throws { try runChild("direct") }
    @Test func actualOperationsReentrantMetric() throws { try runChild("operations") }

    @Test func isolatedContractMatrix() throws {
        for scenario in ["cached", "disabled", "cycle", "recursive-set", "paused", "manual-success", "manual-invalid", "manual-throw", "unrelated", "throw-retry",
            "time-overlap", "time-failure", "time-nested", "time-slow", "parameters", "probes", "bias-seq", "bias-parallel", "printing"] {
            try runChild(scenario)
        }
    }

    @Test func childProbe() async throws {
        guard let scenario = ProcessInfo.processInfo.environment["E05_SCENARIO"] else { return }
        e05Marker("E05_BODY_BEGIN \(scenario)")
        switch scenario {
        case "direct":
            var callbacks = 0
            let tuning = AutoTuning.calibrateIfNeeded(dim: 17, kind: .dot, providerCores: 2) {
                callbacks += 1
                e05Marker("E05_CALLBACK_ENTRY direct")
                #expect(AutoTuning.get(dim: 17, kind: .dot) == nil)
                let provisional = AutoTuning.calibrateIfNeeded(dim: 17, kind: .dot, providerCores: 2) {
                    callbacks += 1
                    return CalibrationProbes(nsPerCandidateSeq: 100, parallelOverheadNs: 1000, effectiveParallelFactor: 2)
                }
                #expect(provisional.breakEvenN == 1000 && provisional.minChunk == 256)
                return CalibrationProbes(nsPerCandidateSeq: 100, parallelOverheadNs: 1000, effectiveParallelFactor: 2)
            }
            #expect(callbacks == 1 && tuning.breakEvenN == 200 && tuning.minChunk == 64)
            #expect(AutoTuning.get(dim: 17, kind: .dot)?.breakEvenN == 200)
        case "operations":
            #expect(!(Operations.computeProvider is any BatchKernelProvider))
            #expect(AutoTuning.get(dim: 17, kind: .euclid) == nil)
            let query = DynamicVector(Array(repeating: Float(0), count: 17))
            let candidates = [Float(1), 3, 2].map { value in
                var coordinates = Array(repeating: Float(0), count: 17); coordinates[0] = value
                return DynamicVector(coordinates)
            }
            let result = try await Operations.findNearest(to: query, in: candidates, k: candidates.count, metric: E05ReentrantMetric())
            #expect(result.count == 3)
            #expect(result.map(\.index) == [0, 2, 1])
            #expect(result.map(\.distance) == [5, 7, 9])
            #expect(AutoTuning.get(dim: 17, kind: .euclid) != nil)
        default: try runContractScenario(scenario)
        }
        e05Marker("E05_RESULT \(scenario)")
        e05Marker("E05_BODY_END \(scenario)")
    }
}

extension AutoTuningConcurrencyTests {
    private final class Sentinel: Error, Sendable {}
    private func probes() -> CalibrationProbes {
        CalibrationProbes(nsPerCandidateSeq: 100, parallelOverheadNs: 1000, effectiveParallelFactor: 2)
    }
    private func same(_ actual: Tuning?, _ expected: Tuning) {
        #expect(actual?.breakEvenN == expected.breakEvenN)
        #expect(actual?.minChunk == expected.minChunk)
        #expect(actual?.modeBias == expected.modeBias)
    }
    private func wait(_ gate: DispatchSemaphore, _ name: String) -> Bool {
        let completed = gate.wait(timeout: .now() + 3) == .success
        #expect(completed, "E05 fixture gate deadline: \(name)")
        return completed
    }
    private func runContractScenario(_ scenario: String) throws {
        switch scenario {
        case "cached", "disabled", "printing":
            let manual = Tuning(breakEvenN: 321, minChunk: 192, modeBias: .favorParallel)
            AutoTuning.set(dim: 42, kind: .dot, tuning: manual)
            var calls = 0
            let result = AutoTuning.calibrateIfNeeded(dim: 42, kind: .dot, providerCores: 2) { calls += 1; return probes() }
            #expect(calls == 0)
            if scenario == "disabled" {
                same(result, Tuning(breakEvenN: 1000, minChunk: 256, modeBias: .neutral))
                #expect(AutoTuning.get(dim: 42, kind: .dot) == nil)
            } else { same(result, manual) }
            if scenario == "printing" {
                let measured = AutoTuning.calibrateIfNeeded(dim: 43, kind: .dot, providerCores: 2) { probes() }
                #expect(measured.breakEvenN == 200)
            }
        case "recursive-set":
            let manual = Tuning(breakEvenN: 321, minChunk: 192, modeBias: .favorParallel)
            let result = AutoTuning.calibrateIfNeeded(dim: 53, kind: .dot, providerCores: 2) {
                AutoTuning.set(dim: 53, kind: .dot, tuning: manual)
                return probes()
            }
            same(result, manual); same(AutoTuning.get(dim: 53, kind: .dot), manual)
        case "cycle":
            var calls = 0
            let result = AutoTuning.calibrateIfNeeded(dim: 51, kind: .dot, providerCores: 2) {
                calls += 1
                _ = AutoTuning.calibrateIfNeeded(dim: 52, kind: .dot, providerCores: 2) {
                    calls += 1
                    let fallback = AutoTuning.calibrateIfNeeded(dim: 51, kind: .dot, providerCores: 2) { calls += 100; return probes() }
                    #expect(fallback.breakEvenN == 1000)
                    return probes()
                }
                return probes()
            }
            #expect(calls == 2 && result.breakEvenN == 200)
        case "paused", "manual-success", "manual-invalid", "manual-throw":
            pausedCallback(scenario)
        case "unrelated":
            unrelatedCallbacks()
        case "throw-retry":
            let sentinel = Sentinel()
            do {
                _ = try AutoTuning.calibrateIfNeeded(dim: 61, kind: .dot, providerCores: 2) { throw sentinel }
                Issue.record("Expected sentinel")
            } catch { #expect((error as? Sentinel) === sentinel) }
            #expect(AutoTuning.get(dim: 61, kind: .dot) == nil)
            var retried = false
            let result = AutoTuning.calibrateIfNeeded(dim: 61, kind: .dot, providerCores: 2) { retried = true; return probes() }
            #expect(retried && result.breakEvenN == 200)
        case "time-overlap", "time-failure", "time-nested", "time-slow":
            try timeAccounting(scenario)
        case "parameters": parameterTable()
        case "probes": probeTable()
        case "bias-seq", "bias-parallel":
            let tuning = AutoTuning.calibrateIfNeeded(dim: 1536, kind: .dot, providerCores: 2) {
                CalibrationProbes(nsPerCandidateSeq: 100, parallelOverheadNs: 50000, effectiveParallelFactor: 2)
            }
            if scenario == "bias-seq" { same(tuning, Tuning(breakEvenN: 1500, minChunk: 512, modeBias: .favorSequential)) }
            else { same(tuning, Tuning(breakEvenN: 660, minChunk: 256, modeBias: .favorParallel)) }
        default: Issue.record("Unknown E05 scenario: \(scenario)")
        }
    }

    private func pausedCallback(_ mode: String) {
        let entered = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0), done = DispatchSemaphore(value: 0)
        let result = OSAllocatedUnfairLock<Tuning?>(initialState: nil)
        let count = OSAllocatedUnfairLock(initialState: 0)
        let flags = OSAllocatedUnfairLock(initialState: (released: false, identical: false))
        let sentinel = Sentinel()
        DispatchQueue.global().async {
            defer { done.signal() }
            do {
                let tuning = try AutoTuning.calibrateIfNeeded(dim: 70, kind: .dot, providerCores: 2) {
                    count.withLock { $0 += 1 }
                    entered.signal()
                    let released = release.wait(timeout: .now() + 3) == .success
                    flags.withLock { $0.released = released }
                    if mode == "manual-throw" { throw sentinel }
                    return CalibrationProbes(nsPerCandidateSeq: mode == "manual-invalid" ? .nan : 100, parallelOverheadNs: 1000, effectiveParallelFactor: 2)
                }
                result.withLock { $0 = tuning }
            } catch { flags.withLock { $0.identical = (error as? Sentinel) === sentinel } }
        }
        guard wait(entered, "paused entered") else { release.signal(); return }
        #expect(AutoTuning.get(dim: 70, kind: .dot) == nil)
        var duplicate = false
        let provisional = AutoTuning.calibrateIfNeeded(dim: 70, kind: .dot, providerCores: 2) { duplicate = true; return probes() }
        #expect(!duplicate && provisional.breakEvenN == 1000 && provisional.minChunk == 256)
        #expect(AutoTuning.get(dim: 70, kind: .dot) == nil)
        let manual = Tuning(breakEvenN: 321, minChunk: 192, modeBias: .favorParallel)
        if mode != "paused" { AutoTuning.set(dim: 70, kind: .dot, tuning: manual); same(AutoTuning.get(dim: 70, kind: .dot), manual) }
        release.signal()
        guard wait(done, "paused completion") else { return }
        #expect(count.withLock { $0 } == 1 && flags.withLock { $0.released })
        if mode == "manual-throw" { #expect(flags.withLock { $0.identical }); #expect(result.withLock { $0 } == nil) }
        else { same(result.withLock { $0 }, mode == "paused" ? Tuning(breakEvenN: 200, minChunk: 64, modeBias: .neutral) : manual) }
        if mode != "paused" {
            same(AutoTuning.calibrateIfNeeded(dim: 70, kind: .dot, providerCores: 2) { duplicate = true; return probes() }, manual)
            #expect(!duplicate)
        }
    }
    private func unrelatedCallbacks() {
        let entered = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0), done = DispatchSemaphore(value: 0)
        let calls = OSAllocatedUnfairLock(initialState: 0), releases = OSAllocatedUnfairLock(initialState: 0)
        for dim in [80, 81] {
            DispatchQueue.global().async {
                _ = AutoTuning.calibrateIfNeeded(dim: dim, kind: .dot, providerCores: 2) {
                    calls.withLock { $0 += 1 }; entered.signal()
                    if release.wait(timeout: .now() + 3) == .success { releases.withLock { $0 += 1 } }
                    return CalibrationProbes(nsPerCandidateSeq: 100, parallelOverheadNs: 1000, effectiveParallelFactor: 2)
                }
                done.signal()
            }
        }
        _ = wait(entered, "first unrelated entry"); _ = wait(entered, "second unrelated entry")
        #expect(calls.withLock { $0 } == 2)
        release.signal(); release.signal()
        _ = wait(done, "first unrelated completion"); _ = wait(done, "second unrelated completion")
        #expect(releases.withLock { $0 } == 2)
        #expect(AutoTuning.get(dim: 80, kind: .dot)?.breakEvenN == 200)
        #expect(AutoTuning.get(dim: 81, kind: .dot)?.breakEvenN == 200)
    }
}

extension AutoTuningConcurrencyTests {
    private func timeAccounting(_ mode: String) throws {
        let clock = OSAllocatedUnfairLock(initialState: Double(0))
        let state = AutoTuning.State(nowMilliseconds: { clock.withLock { $0 } })
        if mode == "time-overlap" {
            let entered = DispatchSemaphore(value: 0)
            let releaseA = DispatchSemaphore(value: 0), releaseB = DispatchSemaphore(value: 0)
            let doneA = DispatchSemaphore(value: 0), doneB = DispatchSemaphore(value: 0)
            let completed = OSAllocatedUnfairLock(initialState: 0)
            for (dim, release, done) in [(1, releaseA, doneA), (2, releaseB, doneB)] {
                DispatchQueue.global().async {
                    _ = state.calibrateIfNeeded(dim: dim, kind: .dot, providerCores: 2) {
                        entered.signal()
                        if release.wait(timeout: .now() + 3) == .success { completed.withLock { $0 += 1 } }
                        return CalibrationProbes(nsPerCandidateSeq: 100, parallelOverheadNs: 1000, effectiveParallelFactor: 2)
                    }
                    done.signal()
                }
            }
            _ = wait(entered, "clock A entry"); _ = wait(entered, "clock B entry")
            clock.withLock { $0 = 11 }; releaseB.signal(); _ = wait(doneB, "clock B done")
            var admitted = false
            _ = state.calibrateIfNeeded(dim: 3, kind: .dot, providerCores: 2) { admitted = true; return probes() }
            #expect(admitted) // Only the completed 11 ms callback is charged yet.
            releaseA.signal(); _ = wait(doneA, "clock A done")
            #expect(completed.withLock { $0 } == 2)
        } else if mode == "time-failure" {
            let sentinel = Sentinel()
            do {
                _ = try state.calibrateIfNeeded(dim: 1, kind: .dot, providerCores: 2) { clock.withLock { $0 = 21 }; throw sentinel }
                Issue.record("Expected timed failure")
            } catch { #expect((error as? Sentinel) === sentinel) }
            #expect(state.get(dim: 1, kind: .dot) == nil)
        } else if mode == "time-slow" {
            _ = state.calibrateIfNeeded(dim: 1, kind: .dot, providerCores: 2) { clock.withLock { $0 = 21 }; return probes() }
        } else {
            _ = state.calibrateIfNeeded(dim: 1, kind: .dot, providerCores: 2) {
                clock.withLock { $0 = 5 }
                _ = state.calibrateIfNeeded(dim: 2, kind: .dot, providerCores: 2) { clock.withLock { $0 = 12 }; return probes() }
                clock.withLock { $0 = 13 }
                return probes()
            }
        }
        var called = false
        let fallback = state.calibrateIfNeeded(dim: 99, kind: .dot, providerCores: 2) { called = true; return probes() }
        #expect(!called)
        same(fallback, Tuning(breakEvenN: 1000, minChunk: 256, modeBias: .neutral))
        same(state.get(dim: 99, kind: .dot), fallback)
    }

    private func parameterTable() {
        for (dim, cores) in [(-1, 2), (Int.min, 2), (1, 0), (1, -1), (1, Int.min), (Int.max, 0)] {
            let state = AutoTuning.State(nowMilliseconds: { 0 })
            let manual = Tuning(breakEvenN: 321, minChunk: 192, modeBias: .favorParallel)
            state.set(dim: dim, kind: .dot, tuning: manual)
            var called = false
            let result = state.calibrateIfNeeded(dim: dim, kind: .dot, providerCores: cores) { called = true; return probes() }
            #expect(!called)
            same(result, Tuning(breakEvenN: 1000, minChunk: dim >= 1024 ? 512 : 256, modeBias: .neutral))
            same(state.get(dim: dim, kind: .dot), manual)
            #expect(state.get(dim: dim, kind: .euclid) == nil)
            _ = state.calibrateIfNeeded(dim: dim, kind: .euclid, providerCores: cores) { called = true; return probes() }
            #expect(!called && state.get(dim: dim, kind: .euclid) == nil)
        }
        for (dim, cores, chunk) in [(0, 1, 128), (0, Int.max, 64), (Int.max, 1, 192), (Int.max, Int.max, 64)] {
            var called = false
            let result = AutoTuning.calibrateIfNeeded(dim: dim, kind: cores == 1 ? .euclid : .dot, providerCores: cores) { called = true; return probes() }
            #expect(called && result.breakEvenN == 200 && result.minChunk == chunk)
        }
    }
    private func probeTable() {
        let state = AutoTuning.State(nowMilliseconds: { 0 })
        var cases: [(Double, Double, Double, Int, Int)] = []
        for bad: Double in [.nan, .infinity, -.infinity, -1, 0] { cases.append((bad, 1000, 2, 1000, 256)); cases.append((100, 1000, bad, 1000, 256)) }
        for bad: Double in [.nan, .infinity, -.infinity, -1] { cases.append((100, bad, 2, 1000, 256)) }
        cases += [
            (100, 0, 2, 200, 64),
            (100, 1000, 1, Int.max / 2, 704),
            (1e-12, 1000, 2, Int.max / 2, 704),
            (1e-9, Double.greatestFiniteMagnitude / 4, 2, 1000, 256),
            (100, Double.greatestFiniteMagnitude, 2, 1000, 256),
            (Double.greatestFiniteMagnitude, 1000, Double.greatestFiniteMagnitude, 1000, 256),
            (1, 1e100, 2, Int.max / 2, 704),
            (100, 1000, 2, 200, 64)
        ]
        for (index, entry) in cases.enumerated() {
            let (a, overhead, factor, threshold, chunk) = entry
            var calls = 0
            let value = state.calibrateIfNeeded(dim: index, kind: .dot, providerCores: 2) {
                calls += 1
                return CalibrationProbes(nsPerCandidateSeq: a, parallelOverheadNs: overhead, effectiveParallelFactor: factor)
            }
            #expect(calls == 1)
            same(value, Tuning(breakEvenN: threshold, minChunk: chunk, modeBias: .neutral))
            same(state.get(dim: index, kind: .dot), value)
            _ = state.calibrateIfNeeded(dim: index, kind: .dot, providerCores: 2) { calls += 1; return probes() }
            #expect(calls == 1)
        }
        var entered = false
        let next = state.calibrateIfNeeded(dim: 999, kind: .dot, providerCores: 2) { entered = true; return probes() }
        #expect(entered && next.breakEvenN == 200)
    }
}
