import Foundation
import Darwin
import MachO
import CryptoKit
import os
import Testing
@testable import VectorCore

private func q03Marker(_ message: String) {
    FileHandle.standardOutput.write(Data((message + "\n").utf8))
}

@Suite struct FP16DiagnosticsBoundaryTests {
    private func runChild(_ scenario: String) throws {
        var environment = ProcessInfo.processInfo.environment
        environment["Q03_SCENARIO"] = scenario
        if environment["Q03_TSAN"] == "1" {
            let runtimes = (0..<_dyld_image_count()).compactMap { index -> String? in
                guard let name = _dyld_get_image_name(index) else { return nil }
                let path = String(cString: name)
                return path.contains("libclang_rt.tsan") ? path : nil
            }
            guard runtimes.count == 1 else {
                throw NSError(domain: "Q03 sanitizer setup", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "Expected one loaded TSAN runtime, found \(runtimes)"])
            }
            var libraries = (environment["DYLD_INSERT_LIBRARIES"] ?? "").split(separator: ":").map(String.init)
            if !libraries.contains(runtimes[0]) { libraries.append(runtimes[0]) }
            environment["DYLD_INSERT_LIBRARIES"] = libraries.joined(separator: ":")
        }
        let directory = environment["Q03_CHILD_LOG_DIR"] ?? "/private/tmp"
        let stem = "\(directory)/Q03-child-\(scenario)-\(UUID().uuidString)"
        let output = URL(fileURLWithPath: stem + ".log")
        _ = FileManager.default.createFile(atPath: output.path, contents: nil)
        let file = try FileHandle(forWritingTo: output)
        let deadline = environment["Q03_TSAN"] == "1" ? 45 : 15
        let child = try launchChildProbe(.swiftTesting(filter: "FP16DiagnosticsBoundaryTests/childProbe"),
                                         environment: environment, output: file,
                                         deadline: Double(deadline), cleanup: .kill(seconds: 2))
        let arguments = child.arguments, timedOut = child.timedOut, cleaned = child.cleaned
        try file.close()
        let log = try String(contentsOf: output, encoding: .utf8)
        func hash(_ path: String) -> String {
            guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return "unreadable" }
            return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }
        var binary = CommandLine.arguments[0]
        if let index = arguments.firstIndex(of: "--test-bundle-path") {
            let path = arguments[index + 1]
            binary = path.hasSuffix(".xctest") ? path + "/Contents/MacOS/" + URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent : path
        }
        var metadata: [String: Any] = ["host": CommandLine.arguments[0], "hostSHA256": hash(CommandLine.arguments[0]),
            "loadedBundle": binary, "loadedBundleSHA256": hash(binary), "arguments": arguments, "scenario": scenario,
            "pid": child.processIdentifier, "timeout": timedOut, "cleaned": cleaned,
            "executionDeadlineSeconds": deadline, "cleanupDeadlineSeconds": 2, "log": output.path, "logSHA256": hash(output.path),
            "environment": environment.filter { $0.key.hasPrefix("Q03_") || $0.key.hasPrefix("TSAN_") || $0.key.hasPrefix("DYLD_") }]
        if timedOut {
            try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: stem + ".json"))
            throw NSError(domain: "Q03 child execution deadline", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Q03 child timed out; cleanup completed: \(cleaned); capture: \(output.path)"])
        }
        let termination = try child.reaped()
        let status = Int(termination.status)
        metadata["exit"] = status
        metadata["terminationReason"] = termination.reason.rawValue
        try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: stem + ".json"))
        print("Q03 child scenario=\(scenario) exit=\(status) timeout=\(timedOut) capture=\(output.path)")
        #expect(!timedOut && cleaned, "Q03 deadline failure: \(output.path)")
        #expect(log.contains("Q03_BODY_BEGIN \(scenario)"), "Q03 child body missing: \(log)")
        if environment["Q03_UNSAFE_BASELINE"] == "1" {
            print("Q03 baseline TSAN reportedRace=\(log.contains("ThreadSanitizer: data race")) status=\(status)")
            #expect(log.contains("Q03_BODY_END \(scenario)"), "Q03 baseline did not complete: \(log)")
        } else {
            #expect(status == 0, "\(log)")
            #expect(log.contains("Q03_RESULT \(scenario)"), "\(log)")
            #expect(log.contains("Q03_BODY_END \(scenario)"), "\(log)")
        }
    }

    @Test func isolatedBoundaryRecords() throws { try runChild("records") }
    @Test func isolatedConstructors() throws { try runChild("constructors") }
    @Test func isolatedConcurrency() throws { try runChild("concurrency") }
    @Test func childProbe() throws {
        guard let scenario = ProcessInfo.processInfo.environment["Q03_SCENARIO"] else { return }
        q03Marker("Q03_BODY_BEGIN \(scenario)")
        switch scenario {
        case "records": boundaryRecords(MixedPrecisionDiagnostics.shared)
        case "constructors": try constructorMatrix()
        case "concurrency":
            try concurrentSingleton()
            stableConcurrentTotals()
            atomicBatchSnapshots()
        default: throw NSError(domain: "Q03 unknown child scenario", code: 1)
        }
        q03Marker("Q03_RESULT \(scenario)")
        q03Marker("Q03_BODY_END \(scenario)")
    }

    private func check(_ stats: MixedPrecisionDiagnostics.OverflowStatistics, rows: [Q03OracleRow], repetitions: Int = 1) {
        #expect(stats.totalConversions == rows.count * repetitions)
        #expect(stats.overflowToInfinity == rows.filter { $0.category == 1 }.count * repetitions)
        #expect(stats.underflowToZero == rows.filter { $0.category == 2 }.count * repetitions)
        #expect(stats.subnormalValues == rows.filter { $0.category == 3 }.count * repetitions)
        #expect(stats.nanValues == rows.filter { $0.category == 4 }.count * repetitions)
        let finite = rows.map { Float(bitPattern: $0.input) }.filter(\.isFinite)
        var lower: Float = 0, upper: Float = 0
        if let first = finite.first {
            lower = first; upper = first
            for value in finite.dropFirst() {
                if value < lower || (value == 0 && lower == 0 && value.sign == .minus) { lower = value }
                if value > upper || (value == 0 && upper == 0 && value.sign == .plus) { upper = value }
            }
        }
        #expect(stats.minValueSeen.bitPattern == lower.bitPattern)
        #expect(stats.maxValueSeen.bitPattern == upper.bitPattern)
    }

    private func boundaryRecords(_ recorder: MixedPrecisionDiagnostics) {
        defer { recorder.disable() }
        for row in q03Oracle {
            recorder.enable()
            recorder.recordConversion(Float(bitPattern: row.input))
            check(recorder.getStatistics(), rows: [row])
            recorder.reset()
            recorder.recordBatchConversion([Float(bitPattern: row.input)])
            check(recorder.getStatistics(), rows: [row])
        }
        for rows in [q03Oracle, q03Oracle.filter { Float(bitPattern: $0.input).isFinite && Float(bitPattern: $0.input) > 0 },
                     q03Oracle.filter { Float(bitPattern: $0.input).isFinite && Float(bitPattern: $0.input) < 0 },
                     q03Oracle.filter { !Float(bitPattern: $0.input).isFinite }, []] {
            recorder.enable()
            recorder.recordBatchConversion(rows.map { Float(bitPattern: $0.input) })
            check(recorder.getStatistics(), rows: rows)
            recorder.reset()
            for row in rows { recorder.recordConversion(Float(bitPattern: row.input)) }
            check(recorder.getStatistics(), rows: rows)
        }
    }

    private func concurrentSingleton() throws {
        let recorder = MixedPrecisionDiagnostics.shared
        let a = Vector512Optimized(repeating: 1), b = Vector768Optimized(repeating: 1), c = Vector1536Optimized(repeating: 1)
        recorder.enable()
        defer { recorder.disable() }
        let start = DispatchSemaphore(value: 0), group = DispatchGroup()
        let completedLoops = OSAllocatedUnfairLock(initialState: Array(repeating: 0, count: 6))
        for worker in 0..<6 {
            group.enter()
            DispatchQueue.global().async {
                defer { group.leave() }
                guard start.wait(timeout: .now() + 5) == .success else {
                    Issue.record("Q03 worker \(worker) start gate timed out")
                    return
                }
                for index in 0..<2000 {
                    switch worker {
                    case 0:
                        if index % 3 == 0 { recorder.enable() }
                        else if index % 3 == 1 { recorder.disable() }
                        else { recorder.reset() }
                    case 1:
                        _ = recorder.isRecording
                        let s = recorder.getStatistics()
                        #expect(s.totalConversions >= 0 && s.overflowToInfinity >= 0 && s.underflowToZero >= 0 && s.subnormalValues >= 0 && s.nanValues >= 0)
                    case 2: recorder.recordConversion(1)
                    case 3: recorder.recordBatchConversion(Array(repeating: 1, count: 8))
                    case 4: _ = MixedPrecisionKernels.Vector512FP16(from: a)
                    default:
                        if index % 2 == 0 { _ = MixedPrecisionKernels.Vector768FP16(from: b) }
                        else { _ = MixedPrecisionKernels.Vector1536FP16(from: c) }
                    }
                }
                completedLoops.withLock { $0[worker] = 2000 }
            }
        }
        for _ in 0..<6 { start.signal() }
        guard group.wait(timeout: .now() + 35) == .success else {
            q03Marker("Q03_WORKER_DEADLINE")
            _exit(72)
        }
        let observed = completedLoops.withLock { $0 }
        #expect(observed == Array(repeating: 2000, count: 6))
        q03Marker("Q03_WORKERS_JOINED completedLoops=\(observed)")
    }
}

extension FP16DiagnosticsBoundaryTests {
    private var one: Q03OracleRow { .init(name: "one", input: 0x3f800000, output: 0x3c00, category: 0) }

    @Test func localRecordsAndStateTransitions() {
        let recorder = MixedPrecisionDiagnostics()
        boundaryRecords(recorder)
        recorder.recordConversion(1)
        recorder.recordBatchConversion([1])
        check(recorder.getStatistics(), rows: [])
        recorder.enable()
        recorder.recordConversion(1)
        recorder.recordBatchConversion([])
        check(recorder.getStatistics(), rows: [one])
        recorder.disable()
        recorder.recordConversion(Float.greatestFiniteMagnitude)
        check(recorder.getStatistics(), rows: [one])
        recorder.reset()
        #expect(!recorder.isRecording)
        check(recorder.getStatistics(), rows: [])
        recorder.enable()
        recorder.recordConversion(1)
        recorder.enable()
        #expect(recorder.isRecording)
        check(recorder.getStatistics(), rows: [])
        recorder.reset()
        #expect(recorder.isRecording)
        let plus = Q03OracleRow(name: "+0", input: 0, output: 0, category: 0)
        let minus = Q03OracleRow(name: "-0", input: 0x80000000, output: 0x8000, category: 0)
        let nan = q03Oracle.first { $0.category == 4 }!
        for rows in [[plus], [minus], [plus, minus], [minus, plus], [nan, one], [nan, minus]] {
            recorder.reset()
            for row in rows { recorder.recordConversion(Float(bitPattern: row.input)) }
            check(recorder.getStatistics(), rows: rows)
            recorder.reset()
            recorder.recordBatchConversion(rows.map { Float(bitPattern: $0.input) })
            check(recorder.getStatistics(), rows: rows)
            recorder.reset()
            for row in rows { recorder.recordBatchConversion([Float(bitPattern: row.input)]) }
            check(recorder.getStatistics(), rows: rows)
        }
    }

    @Test func counterSaturationUsesRealCommit() {
        for category in 1...4 {
            let recorder = MixedPrecisionDiagnostics()
            recorder.enable()
            var seed = MixedPrecisionDiagnostics.OverflowStatistics()
            seed.totalConversions = Int.max - 2
            switch category {
            case 1: seed.overflowToInfinity = Int.max - 2
            case 2: seed.underflowToZero = Int.max - 2
            case 3: seed.subnormalValues = Int.max - 2
            default: seed.nanValues = Int.max - 2
            }
            recorder.seedCountersForTesting(seed)
            let row = q03Oracle.first { $0.category == category }!
            let value = Float(bitPattern: row.input)
            recorder.recordBatchConversion([value, value, value])
            recorder.recordConversion(value)
            let stats = recorder.getStatistics()
            #expect(stats.totalConversions == Int.max)
            #expect([stats.overflowToInfinity, stats.underflowToZero, stats.subnormalValues, stats.nanValues][category - 1] == Int.max)
            #expect(stats.overflowRate.isFinite && stats.underflowRate.isFinite)
            #expect(stats.overflowRate == (category == 1 ? 1 : 0))
            #expect(stats.underflowRate == (category == 2 ? 1 : 0))
            #expect(!stats.summary.isEmpty)
            recorder.disable()
            recorder.recordBatchConversion([value])
            #expect(recorder.getStatistics().totalConversions == Int.max)
            recorder.reset()
            check(recorder.getStatistics(), rows: [])
        }
    }

    private func convert(_ values: [Float], width: Int) throws -> [UInt16] {
        switch width {
        case 512: return MixedPrecisionKernels.Vector512FP16(from: try Vector512Optimized(values)).withUnsafeFP16BufferPointer { Array($0) }
        case 768: return MixedPrecisionKernels.Vector768FP16(from: try Vector768Optimized(values)).withUnsafeFP16BufferPointer { Array($0) }
        default: return MixedPrecisionKernels.Vector1536FP16(from: try Vector1536Optimized(values)).withUnsafeFP16BufferPointer { Array($0) }
        }
    }

    private func checkConstructor(_ rows: [Q03OracleRow], width: Int, label: String) throws {
        let recorder = MixedPrecisionDiagnostics.shared
        let values = rows.map { Float(bitPattern: $0.input) }
        let original = values.map(\.bitPattern)
        recorder.disable()
        let disabled = try convert(values, width: width)
        recorder.enable()
        let actual = try convert(values, width: width)
        check(recorder.getStatistics(), rows: rows)
        #expect(actual == disabled)
        #expect(values.map(\.bitPattern) == original)
        for index in rows.indices {
            if rows[index].category == 4 {
                #expect(actual[index] & 0x7c00 == 0x7c00 && actual[index] & 0x03ff != 0)
            } else {
                #expect(actual[index] == rows[index].output, "width \(width), \(label), index \(index)")
            }
        }
    }

    @Test func suppliedRoundedBitsDriveClassification() {
        let recorder = MixedPrecisionDiagnostics()
        recorder.enable()
        // Deliberately distinct supplied outputs establish this internal recorder's
        // source of classification; this is not a claim that the converter emits them.
        recorder.recordBatchConversion(ContiguousArray([SIMD4<Float>(repeating: 1)]),
                                       rounded: ContiguousArray([0, 0x7c00, 1, 0x3c00]))
        let s = recorder.getStatistics()
        #expect(s.totalConversions == 4 && s.overflowToInfinity == 1 && s.underflowToZero == 1 && s.subnormalValues == 1 && s.nanValues == 0)
        #expect(s.minValueSeen == 1 && s.maxValueSeen == 1)
    }

    private func constructorMatrix() throws {
        let recorder = MixedPrecisionDiagnostics.shared
        defer { recorder.disable() }
        for width in [512, 768, 1536] {
            let positions = Array(0..<8) + Array((width / 2)..<(width / 2 + 8)) + Array((width - 8)..<width)
            for row in q03Oracle {
                var rows = Array(repeating: one, count: width)
                for position in positions { rows[position] = row }
                try checkConstructor(rows, width: width, label: row.name)
            }
            let groups = [q03Oracle,
                          q03Oracle.filter { Float(bitPattern: $0.input).isFinite && Float(bitPattern: $0.input) > 0 },
                          q03Oracle.filter { Float(bitPattern: $0.input).isFinite && Float(bitPattern: $0.input) < 0 },
                          q03Oracle.filter { !Float(bitPattern: $0.input).isFinite },
                          q03Oracle.filter { Float(bitPattern: $0.input) == 0 }]
            for (index, group) in groups.enumerated() {
                let rows = (0..<width).map { group[$0 % group.count] }
                try checkConstructor(rows, width: width, label: "batch class \(index)")
            }
            recorder.enable()
            let raw = (0..<width).map { q03Oracle[$0 % q03Oracle.count].output }
            let reverse: [Float]
            switch width {
            case 512:
                let vector = MixedPrecisionKernels.Vector512FP16(fp16Values: raw)
                #expect(vector.withUnsafeFP16BufferPointer { Array($0) } == raw)
                reverse = vector.toFP32().toArray()
            case 768:
                let vector = MixedPrecisionKernels.Vector768FP16(fp16Values: raw)
                #expect(vector.withUnsafeFP16BufferPointer { Array($0) } == raw)
                reverse = vector.toFP32().toArray()
            default:
                let vector = MixedPrecisionKernels.Vector1536FP16(fp16Values: raw)
                #expect(vector.withUnsafeFP16BufferPointer { Array($0) } == raw)
                reverse = vector.toFP32().toArray()
            }
            // Float16 is unavailable on Intel macOS / Mac Catalyst: the toFP32 reverse-decode oracle is arm64/iOS-family only.
            #if !(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))
            for index in raw.indices {
                let expected = Float(Float16(bitPattern: raw[index]))
                if expected.isNaN { #expect(reverse[index].isNaN) }
                else { #expect(reverse[index].bitPattern == expected.bitPattern) }
            }
            #endif
            check(recorder.getStatistics(), rows: [])
            let lowLevel = MixedPrecisionKernels.convertToFP16(ContiguousArray(repeating: SIMD4<Float>(repeating: 1), count: width / 4))
            #expect(Array(lowLevel) == Array(repeating: UInt16(0x3c00), count: width))
            check(recorder.getStatistics(), rows: [])
            let storage: ContiguousArray<UInt16>
            switch width {
            case 512: storage = try MixedPrecisionKernels.SoA512FP16(vectors: Array(repeating: Vector512Optimized(repeating: 1), count: 5)).storage
            case 768: storage = try MixedPrecisionKernels.SoA768FP16(vectors: Array(repeating: Vector768Optimized(repeating: 1), count: 5)).storage
            default: storage = try MixedPrecisionKernels.SoA1536FP16(vectors: Array(repeating: Vector1536Optimized(repeating: 1), count: 5)).storage
            }
            #expect(storage.count == width * 8)
            check(recorder.getStatistics(), rows: [one], repetitions: width * 5)
        }
    }

    private func joined(_ workers: Int, _ body: @escaping @Sendable (Int) -> Void) {
        let group = DispatchGroup()
        for worker in 0..<workers {
            group.enter()
            DispatchQueue.global().async { body(worker); group.leave() }
        }
        guard group.wait(timeout: .now() + 30) == .success else {
            q03Marker("Q03_WORKER_DEADLINE")
            _exit(72)
        }
    }

    private func stableConcurrentTotals() {
        let local = MixedPrecisionDiagnostics()
        local.enable()
        let values = q03Oracle.map { Float(bitPattern: $0.input) }
        joined(4) { _ in for _ in 0..<100 { local.recordBatchConversion(values) } }
        check(local.getStatistics(), rows: q03Oracle, repetitions: 400)
        let shared = MixedPrecisionDiagnostics.shared
        let a = Vector512Optimized(repeating: 1), b = Vector768Optimized(repeating: 1), c = Vector1536Optimized(repeating: 1)
        shared.enable()
        joined(4) { _ in
            for _ in 0..<100 {
                shared.recordConversion(1)
                shared.recordBatchConversion(Array(repeating: 1, count: 8))
                _ = MixedPrecisionKernels.Vector512FP16(from: a)
                _ = MixedPrecisionKernels.Vector768FP16(from: b)
                _ = MixedPrecisionKernels.Vector1536FP16(from: c)
            }
        }
        check(shared.getStatistics(), rows: [one], repetitions: 400 * (9 + 512 + 768 + 1536))
        shared.disable()
        q03Marker("Q03_STABLE_TOTALS_JOINED")
    }

    private func atomicBatchSnapshots() {
        let local = MixedPrecisionDiagnostics()
        let values: [Float] = [.greatestFiniteMagnitude, 1e-12, Float(sign: .plus, exponent: -24, significand: 1), .nan, 1, 2, 3, 4]
        local.enable()
        joined(4) { worker in
            for index in 0..<500 {
                if worker == 0 {
                    if index % 3 == 0 { local.enable() }
                    else if index % 3 == 1 { local.disable() }
                    else { local.reset() }
                } else if worker == 1 {
                    let s = local.getStatistics()
                    #expect(s.overflowToInfinity == s.underflowToZero && s.underflowToZero == s.subnormalValues && s.subnormalValues == s.nanValues)
                    #expect(Double(s.totalConversions) == 8 * Double(s.nanValues))
                } else { local.recordBatchConversion(values) }
            }
        }
        let shared = MixedPrecisionDiagnostics.shared
        let a = Vector512Optimized(repeating: 1), b = Vector768Optimized(repeating: 1), c = Vector1536Optimized(repeating: 1)
        shared.enable()
        joined(5) { worker in
            for index in 0..<300 {
                switch worker {
                case 0:
                    if index % 3 == 0 { shared.enable() }
                    else if index % 3 == 1 { shared.disable() }
                    else { shared.reset() }
                case 1:
                    let s = shared.getStatistics()
                    #expect(s.totalConversions >= 0 && s.totalConversions % 256 == 0)
                    #expect(s.overflowToInfinity == 0 && s.underflowToZero == 0 && s.subnormalValues == 0 && s.nanValues == 0)
                case 2: _ = MixedPrecisionKernels.Vector512FP16(from: a)
                case 3: _ = MixedPrecisionKernels.Vector768FP16(from: b)
                default: _ = MixedPrecisionKernels.Vector1536FP16(from: c)
                }
            }
        }
        shared.disable()
        q03Marker("Q03_ATOMIC_BATCHES_JOINED")
    }
}
